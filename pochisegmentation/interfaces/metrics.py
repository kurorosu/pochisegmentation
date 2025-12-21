"""セグメンテーション評価指標のインターフェース."""

from abc import ABC, abstractmethod

import torch


class ISegmentationMetrics(ABC):
    """セグメンテーション評価指標のインターフェース.

    バッチ間で評価指標を計算するための統一インターフェースを提供.
    """

    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """バッチ結果を蓄積.

        Args:
            preds: 予測クラスインデックス, 形状は (B, H, W).
            targets: ターゲットクラスインデックス, 形状は (B, H, W).
        """
        pass

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """蓄積した結果から指標を計算.

        Returns:
            指標名と値の辞書.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """蓄積した状態をリセット."""
        pass
