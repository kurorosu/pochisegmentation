"""VOCSegmentationDatasetのテスト."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np
import pytest

from pochisegmentation.datasets.voc_dataset import VOCSegmentationDataset


@pytest.fixture
def voc_dataset_dir() -> Generator[Path, None, None]:
    """テスト用VOCデータセットディレクトリを作成するフィクスチャ."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # ディレクトリ構造を作成
        (root / "JPEGImages").mkdir()
        (root / "SegmentationClass").mkdir()
        (root / "ImageSets" / "Segmentation").mkdir(parents=True)

        # ダミー画像とマスクを作成
        for i in range(5):
            image_id = f"image_{i:03d}"

            # ダミー画像 (100x100 RGB)
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "JPEGImages" / f"{image_id}.jpg"), image)

            # ダミーマスク (100x100 グレースケール, 値は0-2)
            mask = np.random.randint(0, 3, (100, 100), dtype=np.uint8)
            cv2.imwrite(str(root / "SegmentationClass" / f"{image_id}.png"), mask)

        # train.txt を作成 (3枚)
        with open(root / "ImageSets" / "Segmentation" / "train.txt", "w") as f:
            f.write("image_000\nimage_001\nimage_002\n")

        # val.txt を作成 (2枚)
        with open(root / "ImageSets" / "Segmentation" / "val.txt", "w") as f:
            f.write("image_003\nimage_004\n")

        # class_names.txt を作成
        with open(root / "class_names.txt", "w") as f:
            f.write("_background_\ncat\ndog\n")

        yield root


class TestVOCSegmentationDataset:
    """VOCSegmentationDatasetのテストクラス."""

    def test_init_train(self, voc_dataset_dir: Path) -> None:
        """訓練データセットの初期化テスト."""
        dataset = VOCSegmentationDataset(voc_dataset_dir, split="train")
        assert len(dataset) == 3

    def test_init_val(self, voc_dataset_dir: Path) -> None:
        """検証データセットの初期化テスト."""
        dataset = VOCSegmentationDataset(voc_dataset_dir, split="val")
        assert len(dataset) == 2

    def test_getitem(self, voc_dataset_dir: Path) -> None:
        """__getitem__ のテスト."""
        dataset = VOCSegmentationDataset(voc_dataset_dir, split="train")
        image, mask = dataset[0]

        assert isinstance(image, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert image.shape == (100, 100, 3)
        assert mask.shape == (100, 100)

    def test_getitem_rgb(self, voc_dataset_dir: Path) -> None:
        """画像がRGB形式で読み込まれることをテスト."""
        dataset = VOCSegmentationDataset(voc_dataset_dir, split="train")
        image, _ = dataset[0]

        # OpenCVのBGRではなくRGBで読み込まれていることを確認
        assert image.ndim == 3
        assert image.shape[2] == 3

    def test_class_names(self, voc_dataset_dir: Path) -> None:
        """クラス名の読み込みテスト."""
        dataset = VOCSegmentationDataset(voc_dataset_dir, split="train")

        assert dataset.class_names is not None
        assert dataset.class_names == ["_background_", "cat", "dog"]

    def test_num_classes(self, voc_dataset_dir: Path) -> None:
        """クラス数のテスト."""
        dataset = VOCSegmentationDataset(voc_dataset_dir, split="train")

        assert dataset.num_classes == 3

    def test_image_ids(self, voc_dataset_dir: Path) -> None:
        """画像IDリストのテスト."""
        dataset = VOCSegmentationDataset(voc_dataset_dir, split="train")

        assert dataset.image_ids == ["image_000", "image_001", "image_002"]

    def test_missing_images_dir_raises(self, voc_dataset_dir: Path) -> None:
        """画像ディレクトリが存在しない場合のエラーテスト."""
        import shutil

        shutil.rmtree(voc_dataset_dir / "JPEGImages")

        with pytest.raises(FileNotFoundError, match="画像ディレクトリが見つかりません"):
            VOCSegmentationDataset(voc_dataset_dir, split="train")

    def test_missing_masks_dir_raises(self, voc_dataset_dir: Path) -> None:
        """マスクディレクトリが存在しない場合のエラーテスト."""
        import shutil

        shutil.rmtree(voc_dataset_dir / "SegmentationClass")

        with pytest.raises(
            FileNotFoundError, match="マスクディレクトリが見つかりません"
        ):
            VOCSegmentationDataset(voc_dataset_dir, split="train")

    def test_missing_split_file_raises(self, voc_dataset_dir: Path) -> None:
        """分割ファイルが存在しない場合のエラーテスト."""
        with pytest.raises(FileNotFoundError, match="分割ファイルが見つかりません"):
            VOCSegmentationDataset(voc_dataset_dir, split="nonexistent")

    def test_transform(self, voc_dataset_dir: Path) -> None:
        """transformが適用されることをテスト."""

        def simple_transform(
            image: np.ndarray, mask: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            # 画像を半分のサイズにリサイズ
            resized_image = cv2.resize(image, (50, 50))
            resized_mask = cv2.resize(mask, (50, 50), interpolation=cv2.INTER_NEAREST)
            return resized_image, resized_mask

        dataset = VOCSegmentationDataset(
            voc_dataset_dir, split="train", transform=simple_transform
        )
        image, mask = dataset[0]

        assert image.shape == (50, 50, 3)
        assert mask.shape == (50, 50)

    def test_no_class_names_file(self, voc_dataset_dir: Path) -> None:
        """class_names.txt がない場合のテスト."""
        import os

        os.remove(voc_dataset_dir / "class_names.txt")

        dataset = VOCSegmentationDataset(voc_dataset_dir, split="train")

        assert dataset.class_names is None
        assert dataset.num_classes is None

    def test_iteration(self, voc_dataset_dir: Path) -> None:
        """データセットのイテレーションテスト."""
        dataset = VOCSegmentationDataset(voc_dataset_dir, split="train")

        count = 0
        for image, mask in dataset:
            assert image is not None
            assert mask is not None
            count += 1

        assert count == 3

    def test_different_image_extensions(self, voc_dataset_dir: Path) -> None:
        """異なる画像拡張子のテスト."""
        # .png 拡張子の画像を追加
        image_id = "image_png"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(voc_dataset_dir / "JPEGImages" / f"{image_id}.png"), image)

        mask = np.random.randint(0, 3, (100, 100), dtype=np.uint8)
        cv2.imwrite(
            str(voc_dataset_dir / "SegmentationClass" / f"{image_id}.png"), mask
        )

        # train.txt に追加
        with open(
            voc_dataset_dir / "ImageSets" / "Segmentation" / "train.txt", "a"
        ) as f:
            f.write(f"{image_id}\n")

        dataset = VOCSegmentationDataset(voc_dataset_dir, split="train")
        assert len(dataset) == 4

        # 最後の画像 (png) を取得
        image, mask = dataset[3]
        assert image.shape == (100, 100, 3)
