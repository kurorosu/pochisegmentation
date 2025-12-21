"""セグメンテーション損失関数のインターフェース."""

from abc import ABC, abstractmethod

import torch


class ISegmentationLoss(ABC):
    """セグメンテーション損失関数のインターフェース.

    すべての損失関数はこのインターフェースを実装し,
    トレーナーとの互換性を確保する.
    """

    @abstractmethod
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失を計算.

        Args:
            pred: 予測テンソル, 形状は (B, num_classes, H, W).
            target: ターゲットテンソル, 形状は (B, H, W).

        Returns:
            スカラー損失テンソル.
        """
        pass
