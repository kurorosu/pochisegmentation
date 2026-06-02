"""エポックサイクルのオーケストレーション.

訓練・検証・ベスト/最終保存・Early Stopping・履歴記録を一元管理し,
PochiSegmentationTrainer から訓練ループの詳細を分離する.

不変条件:
    - 停止フラグはエポック開始前と完了後の双方でチェックする
      (num_workers=0 と併せて Ctrl+C 安全停止を成立させる).
    - 学習率はスケジューラ更新の前に記録する (そのエポックで実際に使った値).
    - ReduceLROnPlateau は mIoU を監視値として step する.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.training.checkpoint_store import CheckpointStore
from pochisegmentation.training.early_stopping import EarlyStopping
from pochisegmentation.training.epoch_runner import EpochRunner
from pochisegmentation.training.evaluator import Evaluator
from pochisegmentation.training.metrics_tracker import MetricsTracker

__all__ = ["TrainingContext", "TrainingLoop"]


@dataclass
class TrainingContext:
    """訓練ループに必要な協調オブジェクトをまとめるデータクラス.

    Attributes:
        optimizer: オプティマイザ.
        scheduler: 学習率スケジューラ.
        criterion: 損失関数 (ログ表示用).
        epoch_runner: 1 エポック訓練の実行担当.
        evaluator: 検証の実行担当.
        checkpoint_store: チェックポイント保存とベスト指標管理.
        metrics_tracker: 履歴 / クラス別メトリクスの出力担当.
        early_stopping: Early Stopping. 無効の場合は None.
        enable_amp: AMP が有効か (ログ表示用).
    """

    optimizer: Optimizer
    scheduler: LRScheduler | None
    criterion: ISegmentationLoss
    epoch_runner: EpochRunner
    evaluator: Evaluator
    checkpoint_store: CheckpointStore
    metrics_tracker: MetricsTracker
    early_stopping: EarlyStopping | None
    enable_amp: bool


class TrainingLoop:
    """訓練ループの実行を管理するクラス.

    Args:
        logger: ロガーインスタンス.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """TrainingLoopを初期化."""
        self._logger = logger

    def run(
        self,
        ctx: TrainingContext,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
        epochs: int,
        stop_flag_callback: Callable[[], bool] | None = None,
    ) -> dict[str, list[float]]:
        """訓練ループを実行する.

        Args:
            ctx: 訓練コンテキスト.
            train_loader: 訓練データローダー.
            val_loader: 検証データローダー (オプション).
            epochs: エポック数.
            stop_flag_callback: 停止フラグをチェックするコールバック関数.

        Returns:
            訓練履歴 (損失と評価指標).
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_miou": [],
            "val_dice": [],
            "val_loss": [],
            "learning_rate": [],
        }

        self._log_start(ctx, epochs)

        for epoch in range(epochs):
            # 停止フラグのチェック (エポック開始前)
            if stop_flag_callback and stop_flag_callback():
                self._logger.warning(
                    f"安全停止が要求されました。エポック {epoch} で訓練を終了します。"
                )
                break

            # 学習率を記録 (スケジューラ更新前 = そのエポックで実際に使った値)
            current_lr = ctx.optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(current_lr)

            # 訓練フェーズ
            train_loss = ctx.epoch_runner.run(train_loader)
            history["train_loss"].append(train_loss)

            lr_str = self._format_learning_rates(ctx.optimizer)
            loss_name = ctx.criterion.__class__.__name__
            self._logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"{lr_str}, Train Loss ({loss_name}): {train_loss:.4f}"
            )

            # 検証フェーズ
            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = ctx.evaluator.validate(val_loader)
                history["val_miou"].append(val_metrics.get("mIoU", 0.0))
                history["val_dice"].append(val_metrics.get("Dice", 0.0))
                history["val_loss"].append(val_metrics.get("val_loss", 0.0))

                self._logger.info(
                    f"  Val mIoU: {val_metrics.get('mIoU', 0.0):.4f}, "
                    f"Dice: {val_metrics.get('Dice', 0.0):.4f}"
                )

                # ベストモデルの保存と Early Stopping 判定
                ctx.checkpoint_store.update_best(val_metrics, epoch)
                if self._should_early_stop(ctx.early_stopping, val_metrics, epoch):
                    break

            # スケジューラ更新
            if ctx.scheduler is not None:
                if isinstance(ctx.scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau は監視する指標を渡す必要がある
                    val_miou = val_metrics.get("mIoU", 0.0) if val_loader else 0.0
                    ctx.scheduler.step(val_miou)
                else:
                    ctx.scheduler.step()

            # ラストモデルの保存 (毎エポック上書き)
            ctx.checkpoint_store.save_last()

            # 停止フラグのチェック (エポック完了後)
            if stop_flag_callback and stop_flag_callback():
                self._logger.warning(
                    f"安全停止が要求されました。エポック {epoch + 1} で訓練を終了します。"
                )
                break

        store = ctx.checkpoint_store
        self._logger.info(
            f"訓練完了. Best {store.monitor}: {store.best_value:.4f} "
            f"(Epoch {store.best_epoch + 1})"
        )

        ctx.metrics_tracker.finalize(history)

        return history

    def _should_early_stop(
        self,
        early_stopping: EarlyStopping | None,
        val_metrics: dict[str, float],
        epoch: int,
    ) -> bool:
        """Early Stopping 判定を行う.

        Args:
            early_stopping: Early Stopping インスタンス (無効なら None).
            val_metrics: 検証メトリクス.
            epoch: 現在のエポック.

        Returns:
            訓練を停止すべき場合 True.
        """
        if early_stopping is None:
            return False

        monitor_value = val_metrics.get(early_stopping.monitor)
        if monitor_value is None:
            return False

        return early_stopping.step(monitor_value, epoch)

    def _log_start(self, ctx: TrainingContext, epochs: int) -> None:
        """訓練開始時のログを出力する.

        Args:
            ctx: 訓練コンテキスト.
            epochs: エポック数.
        """
        self._logger.info(f"訓練開始: {epochs} エポック")
        if ctx.enable_amp:
            self._logger.info("AMP (混合精度訓練) 有効")
        if ctx.early_stopping is not None:
            self._logger.info(
                f"Early Stopping: {ctx.early_stopping.patience} "
                f"エポック改善なしで停止 (監視: {ctx.early_stopping.monitor})"
            )

    @staticmethod
    def _format_learning_rates(optimizer: Optimizer) -> str:
        """学習率を表示用にフォーマットする.

        層別学習率が有効な場合は encoder/decoder の両方を表示する.

        Args:
            optimizer: オプティマイザ.

        Returns:
            フォーマットされた学習率文字列.
        """
        param_groups = optimizer.param_groups
        if len(param_groups) >= 2:
            enc_lr = param_groups[0]["lr"]
            dec_lr = param_groups[1]["lr"]
            return f"LR: enc={enc_lr:.6f}, dec={dec_lr:.6f}"
        lr = param_groups[0]["lr"]
        return f"LR: {lr:.6f}"
