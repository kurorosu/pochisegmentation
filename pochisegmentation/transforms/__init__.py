"""セグメンテーション用Transform.

torchvision.transforms.v2 と tv_tensors を使用した画像・マスク同期変換.

設定ファイルでの使用例:
    from torchvision.transforms import v2
    from pochisegmentation.transforms import SegmentationTransform

    # 訓練用Transform (設定ファイルで定義)
    train_transform = SegmentationTransform(
        v2.Compose([
            v2.Resize((256, 256)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    # 検証用Transform (設定ファイルで定義)
    val_transform = SegmentationTransform(
        v2.Compose([
            v2.Resize((256, 256)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
"""

from pochisegmentation.transforms.seg_transforms import (
    SegmentationTransform,
    get_basic_train_transform,
    get_basic_val_transform,
)

__all__ = [
    "SegmentationTransform",
    "get_basic_train_transform",
    "get_basic_val_transform",
]
