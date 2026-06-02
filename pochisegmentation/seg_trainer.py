"""セグメンテーション訓練クラス (ファサード).

DIP (依存性逆転原則) に基づき, 具象クラスではなくインターフェースに依存.
DI (依存性注入) により, コンストラクタで依存性を注入.

訓練ループの詳細は pochisegmentation.training 配下の部品 (EpochRunner /
Evaluator / CheckpointStore / MetricsTracker / EarlyStopping / TrainingLoop)
に委譲し, 本クラスはそれらを束ねる薄いファサードとして機能する.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from pochisegmentation.config import PochiSegConfig
from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.interfaces.model import ISegmentationModel
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.training.checkpoint_store import CheckpointStore
from pochisegmentation.training.early_stopping import EarlyStopping
from pochisegmentation.training.epoch_runner import EpochRunner
from pochisegmentation.training.evaluator import Evaluator
from pochisegmentation.training.metrics_tracker import MetricsTracker
from pochisegmentation.training.training_loop import TrainingContext, TrainingLoop
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager

__all__ = ["PochiSegmentationTrainer"]


class PochiSegmentationTrainer:
    """セグメンテーション訓練のファサード.

    DIP: 具象クラスではなくインターフェースに依存.
    DI: コンストラクタで依存性を注入.

    訓練ループ・チェックポイント・履歴出力・Early Stopping を各部品に委譲し,
    setup_training (コンストラクタ) → train() の API を提供する.
    """

    def __init__(
        self,
        model: ISegmentationModel,
        criterion: ISegmentationLoss,
        metrics: ISegmentationMetrics,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        device: str = "cuda",
        config: PochiSegConfig | None = None,
        workspace_manager: PochiWorkspaceManager | None = None,
        early_stopping_patience: int | None = None,
        enable_amp: bool = False,
        early_stopping_monitor: str = "mIoU",
    ) -> None:
        """PochiSegmentationTrainerを初期化.

        Args:
            model: セグメンテーションモデル (ISegmentationModel).
            criterion: 損失関数 (ISegmentationLoss).
            metrics: 評価指標 (ISegmentationMetrics).
            optimizer: オプティマイザ.
            scheduler: 学習率スケジューラ (オプション).
            device: 使用デバイス ("cuda" or "cpu").
            config: 訓練設定 (PochiSegConfig, オプション).
            workspace_manager: ワークスペースマネージャ (オプション).
            early_stopping_patience: Early Stopping の patience (None または 0 で無効).
            enable_amp: AMP (混合精度訓練) を有効化 (CUDA 専用).
            early_stopping_monitor: ベスト判定 / Early Stopping の監視メトリクス
                ("mIoU" / "Dice" / "val_loss").
        """
        logger_manager = LoggerManager()
        self._logger = logger_manager.get_logger("pochiseg")

        self._model = model.to(device)
        self._criterion = criterion
        self._metrics = metrics
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._config = config

        # AMP 設定 (CPU では無効化し警告)
        if enable_amp and device == "cpu":
            self._logger.warning("AMP は CUDA でのみ有効です. 無効化します.")
            enable_amp = False
        self._enable_amp = enable_amp
        scaler: GradScaler | None = GradScaler() if enable_amp else None

        # 責務別の部品を組み立てる
        self._checkpoint_store = CheckpointStore(
            model=self._model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=self._logger,
            workspace_manager=workspace_manager,
            monitor=early_stopping_monitor,
        )
        self._epoch_runner = EpochRunner(
            model=self._model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            enable_amp=enable_amp,
            scaler=scaler,
        )
        self._evaluator = Evaluator(
            model=self._model,
            criterion=criterion,
            metrics=metrics,
            device=device,
            enable_amp=enable_amp,
        )
        self._metrics_tracker = MetricsTracker(
            metrics=metrics,
            logger=self._logger,
            workspace_manager=workspace_manager,
        )
        patience = early_stopping_patience or 0
        self._early_stopping: EarlyStopping | None = (
            EarlyStopping(
                patience=patience,
                monitor=early_stopping_monitor,
                logger=self._logger,
            )
            if patience > 0
            else None
        )
        self._training_loop = TrainingLoop(self._logger)

    def train(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
        epochs: int = 50,
        stop_flag_callback: Callable[[], bool] | None = None,
    ) -> dict[str, list[float]]:
        """訓練ループを実行する.

        Args:
            train_loader: 訓練データローダー.
            val_loader: 検証データローダー (オプション).
            epochs: エポック数.
            stop_flag_callback: 停止フラグをチェックするコールバック関数.

        Returns:
            訓練履歴 (損失と評価指標).
        """
        ctx = TrainingContext(
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            criterion=self._criterion,
            epoch_runner=self._epoch_runner,
            evaluator=self._evaluator,
            checkpoint_store=self._checkpoint_store,
            metrics_tracker=self._metrics_tracker,
            early_stopping=self._early_stopping,
            enable_amp=self._enable_amp,
        )
        return self._training_loop.run(
            ctx,
            train_loader,
            val_loader,
            epochs=epochs,
            stop_flag_callback=stop_flag_callback,
        )

    def save_last_model(self) -> Path | None:
        """最終モデルを保存する.

        Returns:
            保存先パス, ワークスペースマネージャがない場合は None.
        """
        return self._checkpoint_store.save_last()

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """チェックポイントを読み込む.

        Args:
            path: チェックポイントファイルパス.

        Returns:
            チェックポイント情報.
        """
        return self._checkpoint_store.load(path)

    @property
    def model(self) -> nn.Module:
        """モデルを取得.

        Returns:
            セグメンテーションモデル.
        """
        return self._model

    @property
    def best_miou(self) -> float:
        """ベスト指標値を取得.

        Returns:
            ベスト監視メトリクス値 (既定では mIoU).
        """
        return self._checkpoint_store.best_value

    @property
    def best_epoch(self) -> int:
        """ベストエポックを取得.

        Returns:
            ベストエポック番号.
        """
        return self._checkpoint_store.best_epoch
