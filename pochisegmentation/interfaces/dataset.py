"""セグメンテーションデータセットのインターフェース."""

from abc import ABC, abstractmethod
from typing import Any


class ISegmentationDataset(ABC):
    """セグメンテーションデータセットのインターフェース.

    すべてのセグメンテーションデータセットはこのインターフェースを実装する.
    """

    @abstractmethod
    def __len__(self) -> int:
        """データセット内のサンプル数を返す.

        Returns:
            サンプル数.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """インデックスでサンプルを取得.

        Args:
            idx: サンプルのインデックス.

        Returns:
            (画像, マスク) のタプル.
        """
        pass
