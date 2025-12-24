"""SegmentationTransformのテスト."""

import numpy as np
import pytest
import torch
from torchvision.transforms import v2

from pochisegmentation.transforms.seg_transforms import (
    SegmentationTransform,
    get_basic_train_transform,
    get_basic_val_transform,
)


class TestSegmentationTransform:
    """SegmentationTransformのテストクラス."""

    def test_init(self) -> None:
        """初期化テスト."""
        transform = v2.Compose([v2.Resize((128, 128))])
        seg_transform = SegmentationTransform(transform)
        assert seg_transform is not None

    def test_call_basic(self) -> None:
        """基本的な呼び出しテスト."""
        transform = v2.Compose(
            [
                v2.Resize((128, 128)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        seg_transform = SegmentationTransform(transform)

        # ダミー画像とマスク
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 4, (100, 100), dtype=np.uint8)

        result_image, result_mask = seg_transform(image, mask)

        assert isinstance(result_image, torch.Tensor)
        assert isinstance(result_mask, torch.Tensor)
        assert result_image.shape == (3, 128, 128)
        assert result_mask.shape == (128, 128)

    def test_mask_dtype_long(self) -> None:
        """マスクがlong型に変換されることをテスト."""
        transform = v2.Compose([v2.Resize((64, 64))])
        seg_transform = SegmentationTransform(transform)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 4, (100, 100), dtype=np.uint8)

        _, result_mask = seg_transform(image, mask)

        assert result_mask.dtype == torch.int64

    def test_sync_transform(self) -> None:
        """画像とマスクが同期して変換されることをテスト."""
        # 左半分が0, 右半分が1のマスク
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, 50:] = 1

        # 水平反転のみのtransform
        transform = v2.Compose([v2.RandomHorizontalFlip(p=1.0)])
        seg_transform = SegmentationTransform(transform)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, result_mask = seg_transform(image, mask)

        # 反転後は左半分が1, 右半分が0
        result_np = result_mask.numpy()
        assert result_np[50, 0] == 1  # 左側 (元は右側)
        assert result_np[50, 99] == 0  # 右側 (元は左側)

    def test_normalize(self) -> None:
        """正規化が適用されることをテスト."""
        transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        seg_transform = SegmentationTransform(transform)

        # 全て128の画像 (正規化前は約0.5)
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        result_image, _ = seg_transform(image, mask)

        # 正規化後は0付近ではない値になる
        assert result_image.dtype == torch.float32
        # 値の範囲が変わっていることを確認
        assert result_image.min() < 0.5 or result_image.max() > 0.5

    def test_property_transform(self) -> None:
        """transformプロパティのテスト."""
        transform = v2.Compose([v2.Resize((128, 128))])
        seg_transform = SegmentationTransform(transform)

        assert seg_transform.transform is transform


class TestGetBasicTrainTransform:
    """get_basic_train_transformのテストクラス."""

    def test_default_size(self) -> None:
        """デフォルトサイズのテスト."""
        seg_transform = get_basic_train_transform()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 4, (100, 100), dtype=np.uint8)

        result_image, result_mask = seg_transform(image, mask)

        assert result_image.shape == (3, 256, 256)
        assert result_mask.shape == (256, 256)

    def test_custom_size(self) -> None:
        """カスタムサイズのテスト."""
        seg_transform = get_basic_train_transform(image_size=128)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 4, (100, 100), dtype=np.uint8)

        result_image, result_mask = seg_transform(image, mask)

        assert result_image.shape == (3, 128, 128)
        assert result_mask.shape == (128, 128)

    def test_output_normalized(self) -> None:
        """出力が正規化されていることをテスト."""
        seg_transform = get_basic_train_transform()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        result_image, _ = seg_transform(image, mask)

        assert result_image.dtype == torch.float32


class TestGetBasicValTransform:
    """get_basic_val_transformのテストクラス."""

    def test_default_size(self) -> None:
        """デフォルトサイズのテスト."""
        seg_transform = get_basic_val_transform()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 4, (100, 100), dtype=np.uint8)

        result_image, result_mask = seg_transform(image, mask)

        assert result_image.shape == (3, 256, 256)
        assert result_mask.shape == (256, 256)

    def test_deterministic(self) -> None:
        """検証用transformが決定的であることをテスト."""
        seg_transform = get_basic_val_transform()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 4, (100, 100), dtype=np.uint8)

        result1_image, result1_mask = seg_transform(image.copy(), mask.copy())
        result2_image, result2_mask = seg_transform(image.copy(), mask.copy())

        assert torch.allclose(result1_image, result2_image)
        assert torch.equal(result1_mask, result2_mask)


class TestDatasetIntegration:
    """データセットとの統合テスト."""

    def test_with_dataloader(self) -> None:
        """DataLoaderとの統合テスト."""
        from torch.utils.data import DataLoader, Dataset

        class DummyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
            def __init__(self, transform: SegmentationTransform) -> None:
                self.transform = transform

            def __len__(self) -> int:
                return 10

            def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                mask = np.random.randint(0, 4, (100, 100), dtype=np.uint8)
                return self.transform(image, mask)

        seg_transform = get_basic_train_transform(image_size=64)
        dataset = DummyDataset(seg_transform)
        loader = DataLoader(dataset, batch_size=4)

        batch_image, batch_mask = next(iter(loader))

        assert batch_image.shape == (4, 3, 64, 64)
        assert batch_mask.shape == (4, 64, 64)
        assert batch_mask.dtype == torch.int64
