"""PochiSegmentationPredictorのテスト."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2

from pochisegmentation.inference import PochiSegmentationPredictor
from pochisegmentation.interfaces.model import ISegmentationModel


class MockModel(ISegmentationModel):
    """テスト用モックモデル."""

    def __init__(self, num_classes: int = 4) -> None:
        """MockModelを初期化."""
        super().__init__()
        self.num_classes = num_classes
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播."""
        return self.conv(x)

    def get_encoder_params(self) -> list:
        """エンコーダーパラメータを取得."""
        return []

    def get_decoder_params(self) -> list:
        """デコーダーパラメータを取得."""
        return list(self.conv.parameters())


def get_test_transform() -> v2.Compose:
    """テスト用のtransformを取得."""
    return v2.Compose(
        [
            v2.Resize((32, 32)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


class TestPochiSegmentationPredictor:
    """PochiSegmentationPredictorのテストクラス."""

    def test_init(self) -> None:
        """初期化テスト."""
        model = MockModel()
        transform = get_test_transform()

        predictor = PochiSegmentationPredictor(
            model=model,
            transform=transform,
            device="cpu",
        )

        assert predictor.model is not None
        assert predictor.device == "cpu"

    def test_predict_image(self) -> None:
        """numpy配列からの推論テスト."""
        model = MockModel()
        transform = get_test_transform()

        predictor = PochiSegmentationPredictor(
            model=model,
            transform=transform,
            device="cpu",
        )

        # ダミー画像
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = predictor.predict_image(image)

        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_predict_from_file(self) -> None:
        """ファイルからの推論テスト."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # テスト画像を保存
            image_path = Path(tmpdir) / "test.jpg"
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            model = MockModel()
            transform = get_test_transform()

            predictor = PochiSegmentationPredictor(
                model=model,
                transform=transform,
                device="cpu",
            )

            result = predictor.predict(image_path)

            assert isinstance(result, np.ndarray)
            assert result.shape == (32, 32)
            assert result.dtype == np.uint8

    def test_predict_file_not_found(self) -> None:
        """存在しないファイルの推論テスト."""
        model = MockModel()
        transform = get_test_transform()

        predictor = PochiSegmentationPredictor(
            model=model,
            transform=transform,
            device="cpu",
        )

        with pytest.raises(FileNotFoundError, match="画像ファイルが見つかりません"):
            predictor.predict("nonexistent.jpg")

    def test_predict_batch(self) -> None:
        """バッチ推論テスト."""
        model = MockModel()
        transform = get_test_transform()

        predictor = PochiSegmentationPredictor(
            model=model,
            transform=transform,
            device="cpu",
        )

        # ダミーデータローダー
        images = torch.randn(8, 3, 32, 32)
        masks = torch.randint(0, 4, (8, 32, 32))
        dataset = TensorDataset(images, masks)
        loader = DataLoader(dataset, batch_size=2)

        results = predictor.predict_batch(loader)

        assert len(results) == 8
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (32, 32)
            assert result.dtype == np.uint8

    def test_from_checkpoint(self) -> None:
        """チェックポイントからの初期化テスト."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # チェックポイント保存
            model = MockModel()
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

            # チェックポイントから読み込み
            model2 = MockModel()
            transform = get_test_transform()

            predictor = PochiSegmentationPredictor.from_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model2,
                transform=transform,
                device="cpu",
            )

            assert predictor.model is not None

    def test_from_checkpoint_not_found(self) -> None:
        """存在しないチェックポイントのテスト."""
        model = MockModel()
        transform = get_test_transform()

        with pytest.raises(
            FileNotFoundError, match="チェックポイントファイルが見つかりません"
        ):
            PochiSegmentationPredictor.from_checkpoint(
                checkpoint_path="nonexistent.pth",
                model=model,
                transform=transform,
                device="cpu",
            )

    def test_model_eval_mode(self) -> None:
        """モデルがevalモードになっていることをテスト."""
        model = MockModel()
        transform = get_test_transform()

        predictor = PochiSegmentationPredictor(
            model=model,
            transform=transform,
            device="cpu",
        )

        assert not predictor.model.training

    def test_property_model(self) -> None:
        """modelプロパティのテスト."""
        model = MockModel()
        transform = get_test_transform()

        predictor = PochiSegmentationPredictor(
            model=model,
            transform=transform,
            device="cpu",
        )

        # モデルはto()で返されるのでis比較ではなく型チェック
        assert isinstance(predictor.model, torch.nn.Module)

    def test_property_device(self) -> None:
        """deviceプロパティのテスト."""
        model = MockModel()
        transform = get_test_transform()

        predictor = PochiSegmentationPredictor(
            model=model,
            transform=transform,
            device="cpu",
        )

        assert predictor.device == "cpu"
