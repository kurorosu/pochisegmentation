"""チェックポイントの保存・読み込みとベスト指標の管理.

ベストモデル / 最終モデルの保存・復元に加え, ベスト指標 (best_value /
best_epoch) の保持を一手に担う. ベスト判定は Early Stopping と同じ改善方向
(is_higher_better) を共有し, 既定の mIoU 経路は従来挙動と一致する.
"""

import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pochisegmentation.interfaces.model import ISegmentationModel
from pochisegmentation.training.early_stopping import is_higher_better
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager

__all__ = ["CheckpointStore"]


class CheckpointStore:
    """チェックポイントの保存・読み込みとベスト指標の管理を担うクラス.

    Args:
        model: 訓練対象のモデル.
        optimizer: オプティマイザ.
        scheduler: 学習率スケジューラ (オプション).
        device: 使用デバイス.
        logger: ロガーインスタンス.
        workspace_manager: ワークスペースマネージャ (None の場合は保存しない).
        monitor: ベスト判定に用いる監視メトリクス名.
    """

    def __init__(
        self,
        model: ISegmentationModel,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: str,
        logger: logging.Logger,
        workspace_manager: PochiWorkspaceManager | None = None,
        monitor: str = "mIoU",
    ) -> None:
        """CheckpointStoreを初期化."""
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._logger = logger
        self._workspace_manager = workspace_manager
        self._monitor = monitor
        self._higher_is_better = is_higher_better(monitor)

        # ベスト指標. mIoU / Dice は 0.0 起点 (従来挙動), val_loss は +inf 起点.
        self._best_value = 0.0 if self._higher_is_better else math.inf
        self._best_epoch = 0

    @property
    def monitor(self) -> str:
        """監視メトリクス名を取得.

        Returns:
            ベスト判定に用いる監視メトリクス名.
        """
        return self._monitor

    @property
    def best_value(self) -> float:
        """ベスト指標値を取得.

        Returns:
            これまでのベスト監視メトリクス値.
        """
        return self._best_value

    @property
    def best_epoch(self) -> int:
        """ベストエポックを取得.

        Returns:
            ベスト指標を記録したエポック番号.
        """
        return self._best_epoch

    def _is_improvement(self, value: float) -> bool:
        """指定値が現在のベストを改善しているか判定.

        Args:
            value: 現在のエポックの監視メトリクス値.

        Returns:
            改善している場合 True.
        """
        if self._higher_is_better:
            return value > self._best_value
        return value < self._best_value

    def update_best(self, metrics: dict[str, float], epoch: int) -> bool:
        """検証メトリクスを評価し, 改善時はベストモデルを保存する.

        Args:
            metrics: 評価指標の辞書.
            epoch: 現在のエポック.

        Returns:
            改善した場合 True.
        """
        value = metrics.get(self._monitor, 0.0)
        if not self._is_improvement(value):
            return False

        self._best_value = value
        self._best_epoch = epoch

        if self._workspace_manager is not None:
            models_dir = self._workspace_manager.get_models_dir()
            model_path = models_dir / "best.pth"
            self._save_checkpoint(model_path, epoch, metrics)
            self._logger.info(f"ベストモデルを保存: {model_path}")
        return True

    def save_last(self) -> Path | None:
        """最終モデルを保存する.

        Returns:
            保存先パス. ワークスペースマネージャがない場合は None.
        """
        if self._workspace_manager is None:
            return None

        models_dir = self._workspace_manager.get_models_dir()
        model_path = models_dir / "last.pth"
        self._save_checkpoint(
            model_path, self._best_epoch, {self._monitor: self._best_value}
        )
        self._logger.info(f"最終モデルを保存: {model_path}")
        return model_path

    def _save_checkpoint(
        self, path: Path, epoch: int, metrics: dict[str, float]
    ) -> None:
        """チェックポイントを保存する.

        Args:
            path: 保存先パス.
            epoch: エポック番号.
            metrics: 評価指標.
        """
        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "metrics": metrics,
            "best_miou": self._best_value,
        }

        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: Path) -> dict[str, Any]:
        """チェックポイントを読み込み, 各状態を復元する.

        Args:
            path: チェックポイントファイルパス.

        Returns:
            読み込んだチェックポイント情報.
        """
        checkpoint: dict[str, Any] = torch.load(path, map_location=self._device)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self._scheduler is not None and "scheduler_state_dict" in checkpoint:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._best_value = checkpoint.get("best_miou", self._best_value)
        self._best_epoch = checkpoint.get("epoch", 0)

        self._logger.info(
            f"チェックポイントを読み込み: {path} (Epoch {self._best_epoch + 1})"
        )

        return checkpoint
