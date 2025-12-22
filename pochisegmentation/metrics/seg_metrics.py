"""セグメンテーション評価指標の実装."""

import torch
from torchmetrics import Accuracy, F1Score
from torchmetrics.segmentation import DiceScore, MeanIoU

from pochisegmentation.interfaces.metrics import ISegmentationMetrics


class SegmentationMetrics(ISegmentationMetrics):
    """torchmetricsを使った評価指標.

    mIoU, Dice, PixelAccuracy, F1スコアを計算する.

    Attributes:
        _num_classes: クラス数.
        _device: 計算デバイス.
        _iou: MeanIoUメトリクス.
        _dice: DiceScoreメトリクス.
        _pixel_accuracy: PixelAccuracyメトリクス.
        _f1: F1スコアメトリクス.
    """

    def __init__(self, num_classes: int, device: str = "cuda") -> None:
        """評価指標を初期化.

        Args:
            num_classes: クラス数.
            device: 計算デバイス ("cuda" or "cpu").
        """
        self._num_classes = num_classes
        self._device = device

        self._iou = MeanIoU(num_classes=num_classes).to(device)
        self._dice = DiceScore(num_classes=num_classes, average="macro").to(device)
        self._pixel_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(
            device
        )
        self._f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """バッチ結果を蓄積.

        Args:
            preds: 予測クラスインデックス, 形状は (B, H, W).
            targets: ターゲットクラスインデックス, 形状は (B, H, W).
        """
        self._iou.update(preds, targets)
        self._dice.update(preds, targets)
        self._pixel_accuracy.update(preds.flatten(), targets.flatten())
        self._f1.update(preds.flatten(), targets.flatten())

    def compute(self) -> dict[str, float]:
        """蓄積した結果から指標を計算.

        Returns:
            指標名と値の辞書.
        """
        return {
            "mIoU": self._iou.compute().item(),
            "Dice": self._dice.compute().item(),
            "PixelAccuracy": self._pixel_accuracy.compute().item(),
            "F1": self._f1.compute().item(),
        }

    def reset(self) -> None:
        """蓄積した状態をリセット."""
        self._iou.reset()
        self._dice.reset()
        self._pixel_accuracy.reset()
        self._f1.reset()
