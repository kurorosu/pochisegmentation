"""セグメンテーション用Transformの実装.

torchvision.transforms.v2 と tv_tensors を使用して,
画像とマスクを同期的に変換する.
"""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision import tv_tensors
from torchvision.transforms import v2


class SegmentationTransform:
    """セグメンテーション用Transform.

    numpy配列の画像とマスクをtv_tensorsでラップし,
    torchvision.transforms.v2で同期変換する.

    Attributes:
        _transform: 適用するtransform (v2.Compose).
    """

    def __init__(self, transform: v2.Compose) -> None:
        """SegmentationTransformを初期化.

        Args:
            transform: 適用するv2.Compose transform.
        """
        self._transform = transform

    def __call__(
        self, image: NDArray[np.uint8], mask: NDArray[np.uint8]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """画像とマスクを同期変換.

        Args:
            image: 入力画像 (H, W, C) のnumpy配列, RGB形式.
            mask: 入力マスク (H, W) のnumpy配列, クラスインデックス.

        Returns:
            変換後の (画像テンソル, マスクテンソル) のタプル.
        """
        # numpy配列をtv_tensorsでラップ
        # 画像: (H, W, C) -> (C, H, W) に変換してからImage化
        image_tensor = tv_tensors.Image(image.transpose(2, 0, 1))
        mask_tensor = tv_tensors.Mask(mask)

        # 同期変換を適用
        transformed_image, transformed_mask = self._transform(image_tensor, mask_tensor)

        # マスクをlong型に変換 (CrossEntropyLoss用)
        if isinstance(transformed_mask, torch.Tensor):
            transformed_mask = transformed_mask.long()

        return transformed_image, transformed_mask

    @property
    def transform(self) -> v2.Compose:
        """内部のtransformを取得.

        Returns:
            v2.Compose transform.
        """
        return self._transform


def get_basic_train_transform(image_size: int = 256) -> SegmentationTransform:
    """基本的な訓練用Transformを取得.

    Args:
        image_size: リサイズ後の画像サイズ.

    Returns:
        訓練用SegmentationTransform.

    Note:
        これはデフォルトの参考実装です.
        実際の使用時は設定ファイルで独自のtransformを定義することを推奨します.
    """
    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return SegmentationTransform(transform)


def get_basic_val_transform(image_size: int = 256) -> SegmentationTransform:
    """基本的な検証用Transformを取得.

    Args:
        image_size: リサイズ後の画像サイズ.

    Returns:
        検証用SegmentationTransform.

    Note:
        これはデフォルトの参考実装です.
        実際の使用時は設定ファイルで独自のtransformを定義することを推奨します.
    """
    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return SegmentationTransform(transform)
