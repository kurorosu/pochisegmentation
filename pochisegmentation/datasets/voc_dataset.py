"""VOC形式セグメンテーションデータセットの実装."""

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from pochisegmentation.interfaces.dataset import ISegmentationDataset


class VOCSegmentationDataset(Dataset[tuple[Any, Any]], ISegmentationDataset):
    """VOC形式セグメンテーションデータセット.

    VOC形式のディレクトリ構造:
        root/
        ├── JPEGImages/          # 元画像 (.jpg)
        ├── SegmentationClass/   # クラスマスク (.png)
        ├── ImageSets/
        │   └── Segmentation/
        │       ├── train.txt    # 訓練用ファイル名リスト
        │       └── val.txt      # 検証用ファイル名リスト
        └── class_names.txt      # クラス名リスト (オプション)

    Attributes:
        _root: データセットのルートディレクトリ.
        _split: データ分割 ("train", "val", "trainval").
        _image_ids: 画像IDのリスト.
        _images_dir: 画像ディレクトリのパス.
        _masks_dir: マスクディレクトリのパス.
        _transform: 画像とマスクに適用するtransform.
        _class_names: クラス名のリスト.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable[..., Any] | None = None,
    ) -> None:
        """VOCSegmentationDatasetを初期化.

        Args:
            root: データセットのルートディレクトリパス.
            split: データ分割 ("train", "val", "trainval").
            transform: 画像とマスクに適用するtransform.

        Raises:
            FileNotFoundError: 必要なディレクトリやファイルが見つからない場合.
            ValueError: 無効なsplitが指定された場合.
        """
        self._root = Path(root)
        self._split = split
        self._transform = transform

        # ディレクトリパスの設定
        self._images_dir = self._root / "JPEGImages"
        self._masks_dir = self._root / "SegmentationClass"
        split_file = self._root / "ImageSets" / "Segmentation" / f"{split}.txt"

        # 存在確認
        if not self._images_dir.exists():
            raise FileNotFoundError(
                f"画像ディレクトリが見つかりません: {self._images_dir}"
            )
        if not self._masks_dir.exists():
            raise FileNotFoundError(
                f"マスクディレクトリが見つかりません: {self._masks_dir}"
            )
        if not split_file.exists():
            raise FileNotFoundError(f"分割ファイルが見つかりません: {split_file}")

        # 画像IDを読み込み
        self._image_ids = self._load_image_ids(split_file)

        # クラス名を読み込み (オプション)
        self._class_names = self._load_class_names()

    def _load_image_ids(self, split_file: Path) -> list[str]:
        """分割ファイルから画像IDを読み込む.

        Args:
            split_file: 分割ファイルのパス.

        Returns:
            画像IDのリスト.
        """
        with open(split_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_class_names(self) -> list[str] | None:
        """クラス名ファイルを読み込む.

        Returns:
            クラス名のリスト. ファイルが存在しない場合はNone.
        """
        class_names_file = self._root / "class_names.txt"
        if class_names_file.exists():
            with open(class_names_file, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return None

    def _find_image_file(self, image_id: str) -> Path:
        """画像ファイルのパスを検索.

        Args:
            image_id: 画像ID (拡張子なし).

        Returns:
            画像ファイルのパス.

        Raises:
            FileNotFoundError: 画像ファイルが見つからない場合.
        """
        # 一般的な画像拡張子を試行
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_path = self._images_dir / f"{image_id}{ext}"
            if image_path.exists():
                return image_path

        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_id}")

    def __len__(self) -> int:
        """データセット内のサンプル数を返す.

        Returns:
            サンプル数.
        """
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """インデックスでサンプルを取得.

        Args:
            idx: サンプルのインデックス.

        Returns:
            (画像, マスク) のタプル.
            transformがない場合: (numpy配列, numpy配列).
            transformがある場合: transformの出力による.
        """
        image_id = self._image_ids[idx]

        # 画像を読み込み (BGR -> RGB)
        image_path = self._find_image_file(image_id)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # マスクを読み込み (グレースケール)
        mask_path = self._masks_dir / f"{image_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"マスクファイルが見つかりません: {mask_path}")

        # transformを適用
        if self._transform is not None:
            image, mask = self._transform(image, mask)

        return image, mask

    @property
    def class_names(self) -> list[str] | None:
        """クラス名のリストを取得.

        Returns:
            クラス名のリスト. 未定義の場合はNone.
        """
        return self._class_names

    @property
    def num_classes(self) -> int | None:
        """クラス数を取得.

        Returns:
            クラス数. クラス名が未定義の場合はNone.
        """
        if self._class_names is not None:
            return len(self._class_names)
        return None

    @property
    def image_ids(self) -> list[str]:
        """画像IDのリストを取得.

        Returns:
            画像IDのリスト.
        """
        return self._image_ids.copy()
