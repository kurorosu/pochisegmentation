"""セグメンテーション推論クラス.

学習済みモデルを使用して画像のセグメンテーション推論を行う.
"""

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

from pochisegmentation.interfaces.model import ISegmentationModel


class PochiSegmentationPredictor:
    """セグメンテーション推論クラス.

    学習済みモデルを使用して画像のセグメンテーション推論を行う.

    Attributes:
        _model: 推論に使用するモデル.
        _transform: 推論時に適用する前処理.
        _device: 使用デバイス.
    """

    def __init__(
        self,
        model: ISegmentationModel,
        transform: v2.Compose,
        device: str = "cuda",
    ) -> None:
        """PochiSegmentationPredictorを初期化.

        Args:
            model: セグメンテーションモデル.
            transform: 推論時に適用するtransform (v2.Compose).
            device: 使用デバイス ("cuda" or "cpu").
        """
        self._model: nn.Module = model.to(device)
        self._model.eval()
        self._transform = transform
        self._device = device

    def predict(self, image_path: str | Path) -> NDArray[np.uint8]:
        """単一画像の推論を実行.

        Args:
            image_path: 入力画像のパス.

        Returns:
            予測マスク (H, W) のnumpy配列, クラスインデックス.

        Raises:
            FileNotFoundError: 画像ファイルが存在しない場合.
            ValueError: 画像の読み込みに失敗した場合.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        # 画像読み込み (BGR -> RGB)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
        image_rgb = cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return self.predict_image(image_rgb)

    def predict_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """numpy配列の画像から推論を実行.

        Args:
            image: 入力画像 (H, W, C) のnumpy配列, RGB形式.

        Returns:
            予測マスク (H, W) のnumpy配列, クラスインデックス.
        """
        # 前処理
        input_tensor = self._preprocess(image)

        # 推論
        with torch.no_grad():
            output: torch.Tensor = self._model(input_tensor.to(self._device))
            pred = output.argmax(dim=1).squeeze().cpu().numpy()

        return cast(NDArray[np.uint8], pred.astype(np.uint8))

    def predict_batch(
        self, loader: DataLoader[tuple[torch.Tensor, Any]]
    ) -> list[NDArray[np.uint8]]:
        """バッチ推論を実行.

        Args:
            loader: 画像データローダー.

        Returns:
            予測マスクのリスト.
        """
        results: list[NDArray[np.uint8]] = []

        with torch.no_grad():
            for batch in loader:
                # DataLoaderは (images, masks) または (images,) を返す可能性がある
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                outputs = self._model(images.to(self._device))
                preds = outputs.argmax(dim=1).cpu().numpy()

                for pred in preds:
                    results.append(pred.astype(np.uint8))

        return results

    def _preprocess(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """画像の前処理を実行.

        Args:
            image: 入力画像 (H, W, C) のnumpy配列, RGB形式.

        Returns:
            前処理済みテンソル (1, C, H, W).
        """
        # numpy配列をtv_tensors.Imageに変換
        # (H, W, C) -> (C, H, W)
        image_tensor = tv_tensors.Image(image.transpose(2, 0, 1))

        # transform適用
        transformed: torch.Tensor = self._transform(image_tensor)

        # バッチ次元を追加
        return transformed.unsqueeze(0)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: ISegmentationModel,
        transform: v2.Compose,
        device: str = "cuda",
    ) -> "PochiSegmentationPredictor":
        """チェックポイントからPredictorを作成.

        Args:
            checkpoint_path: チェックポイントファイルのパス.
            model: モデルインスタンス (重みはチェックポイントから読み込まれる).
            transform: 推論時に適用するtransform.
            device: 使用デバイス.

        Returns:
            初期化されたPredictor.

        Raises:
            FileNotFoundError: チェックポイントファイルが存在しない場合.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"チェックポイントファイルが見つかりません: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model=model, transform=transform, device=device)

    @property
    def model(self) -> nn.Module:
        """モデルを取得.

        Returns:
            推論モデル.
        """
        return self._model

    @property
    def device(self) -> str:
        """デバイスを取得.

        Returns:
            使用デバイス.
        """
        return self._device
