"""クラス別精度の計算."""

from typing import Any

import numpy as np
import torch
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassJaccardIndex,
)


class ClassMetrics:
    """クラス別精度を計算・保持.

    訓練・推論の両方で使用可能.

    Attributes:
        _num_classes: クラス数.
        _class_names: クラス名リスト.
        _device: 計算デバイス.
        _iou_per_class: クラス別 IoU メトリクス.
        _confusion_matrix: Confusion Matrix メトリクス.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
        device: str = "cuda",
    ) -> None:
        """初期化.

        Args:
            num_classes: クラス数.
            class_names: クラス名リスト (None の場合は "class_0", "class_1", ...).
            device: 計算デバイス.
        """
        self._num_classes = num_classes
        self._device = device

        # クラス名の設定
        if class_names is not None:
            self._class_names = class_names
        else:
            self._class_names = [f"class_{i}" for i in range(num_classes)]

        # クラス別 IoU (average=None でクラス別に取得)
        self._iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes, average=None
        ).to(device)

        # Confusion Matrix
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(
            device
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """バッチ結果を蓄積.

        Args:
            preds: 予測クラスインデックス, 形状は (B, H, W).
            targets: ターゲットクラスインデックス, 形状は (B, H, W).
        """
        # flatten して更新
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()

        self._iou_per_class.update(preds_flat, targets_flat)
        self._confusion_matrix.update(preds_flat, targets_flat)

    def compute(self) -> dict[str, Any]:
        """クラス別精度を計算.

        Returns:
            クラス別精度の辞書:
            - per_class_iou: クラス別 IoU のリスト.
            - confusion_matrix: Confusion Matrix (numpy 配列).
            - class_names: クラス名リスト.
            - num_classes: クラス数.
        """
        iou_tensor = self._iou_per_class.compute()
        cm_tensor = self._confusion_matrix.compute()

        return {
            "per_class_iou": iou_tensor.cpu().numpy().tolist(),
            "confusion_matrix": cm_tensor.cpu().numpy(),
            "class_names": self._class_names,
            "num_classes": self._num_classes,
        }

    def reset(self) -> None:
        """蓄積した状態をリセット."""
        self._iou_per_class.reset()
        self._confusion_matrix.reset()

    @property
    def class_names(self) -> list[str]:
        """クラス名リストを取得."""
        return self._class_names

    @property
    def num_classes(self) -> int:
        """クラス数を取得."""
        return self._num_classes
