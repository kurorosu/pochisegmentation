"""依存性注入用コンポーネントファクトリー."""

from typing import Callable

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from pochisegmentation.config import PochiSegConfig
from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.interfaces.model import ISegmentationModel
from pochisegmentation.utils.layer_wise_lr import create_layer_wise_param_groups

# ファクトリー関数の型エイリアス
ModelFactory = Callable[..., ISegmentationModel]
LossFactory = Callable[..., ISegmentationLoss]
MetricsFactory = Callable[..., ISegmentationMetrics]


class ComponentFactory:
    """依存性注入用ファクトリー.

    DIP: 高レベルモジュール (Trainer) は抽象に依存.
    このファクトリーで具象コンポーネントを生成し, Trainerに注入する.

    Attributes:
        _model_registry: 登録されたモデルファクトリーの辞書.
        _loss_registry: 登録された損失関数ファクトリーの辞書.
        _metrics_registry: 登録された評価指標ファクトリーの辞書.
    """

    _model_registry: dict[str, ModelFactory] = {}
    _loss_registry: dict[str, LossFactory] = {}
    _metrics_registry: dict[str, MetricsFactory] = {}

    @classmethod
    def register_model(cls, name: str, model_class: ModelFactory) -> None:
        """モデルを登録.

        Args:
            name: モデル名 (設定ファイルで使用).
            model_class: ISegmentationModelを返すファクトリー (通常はクラス).
        """
        cls._model_registry[name] = model_class

    @classmethod
    def register_loss(cls, name: str, loss_class: LossFactory) -> None:
        """損失関数を登録.

        Args:
            name: 損失関数名 (設定ファイルで使用).
            loss_class: ISegmentationLossを返すファクトリー (通常はクラス).
        """
        cls._loss_registry[name] = loss_class

    @classmethod
    def register_metrics(cls, name: str, metrics_class: MetricsFactory) -> None:
        """評価指標を登録.

        Args:
            name: 評価指標名 (設定ファイルで使用).
            metrics_class: ISegmentationMetricsを返すファクトリー (通常はクラス).
        """
        cls._metrics_registry[name] = metrics_class

    @classmethod
    def create_model(cls, config: PochiSegConfig) -> ISegmentationModel:
        """設定からモデルを生成.

        Args:
            config: 訓練設定 (PochiSegConfig).

        Returns:
            生成されたモデルインスタンス.

        Raises:
            ValueError: 未登録のモデル名が指定された場合.
        """
        if config.architecture not in cls._model_registry:
            available = list(cls._model_registry.keys())
            raise ValueError(
                f"未登録のモデル: {config.architecture}. 利用可能: {available}"
            )

        return cls._model_registry[config.architecture](
            encoder_name=config.encoder_name,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            in_channels=config.in_channels,
        )

    @classmethod
    def create_loss(cls, config: PochiSegConfig) -> ISegmentationLoss:
        """設定から損失関数を生成.

        Args:
            config: 訓練設定 (PochiSegConfig).

        Returns:
            生成された損失関数インスタンス.

        Raises:
            ValueError: 未登録の損失関数名が指定された場合.
        """
        if config.loss not in cls._loss_registry:
            available = list(cls._loss_registry.keys())
            raise ValueError(f"未登録の損失関数: {config.loss}. 利用可能: {available}")

        return cls._loss_registry[config.loss](**config.loss_params)

    @classmethod
    def create_metrics(
        cls, config: PochiSegConfig, class_names: list[str] | None = None
    ) -> ISegmentationMetrics:
        """設定から評価指標を生成.

        Args:
            config: 訓練設定 (PochiSegConfig).
            class_names: クラス名リスト (オプション).

        Returns:
            生成された評価指標インスタンス.

        Raises:
            ValueError: 未登録の評価指標名が指定された場合.
        """
        if config.metrics not in cls._metrics_registry:
            available = list(cls._metrics_registry.keys())
            raise ValueError(
                f"未登録の評価指標: {config.metrics}. 利用可能: {available}"
            )

        return cls._metrics_registry[config.metrics](
            num_classes=config.num_classes,
            class_names=class_names,
            device=config.device,
        )

    @classmethod
    def get_available_models(cls) -> list[str]:
        """登録済みモデル名のリストを取得.

        Returns:
            登録済みモデル名のリスト.
        """
        return list(cls._model_registry.keys())

    @classmethod
    def get_available_losses(cls) -> list[str]:
        """登録済み損失関数名のリストを取得.

        Returns:
            登録済み損失関数名のリスト.
        """
        return list(cls._loss_registry.keys())

    @classmethod
    def get_available_metrics(cls) -> list[str]:
        """登録済み評価指標名のリストを取得.

        Returns:
            登録済み評価指標名のリスト.
        """
        return list(cls._metrics_registry.keys())

    @classmethod
    def reset(cls) -> None:
        """すべてのレジストリをクリア (テスト用)."""
        cls._model_registry.clear()
        cls._loss_registry.clear()
        cls._metrics_registry.clear()

    @classmethod
    def create_optimizer(
        cls, model: torch.nn.Module, config: PochiSegConfig
    ) -> torch.optim.Optimizer:
        """オプティマイザを作成.

        Args:
            model: モデル.
            config: 訓練設定 (PochiSegConfig).

        Returns:
            オプティマイザ.

        Raises:
            ValueError: 未対応のオプティマイザ名が指定された場合.
        """
        # 層別学習率
        if config.enable_layer_wise_lr:
            param_groups = create_layer_wise_param_groups(
                model,  # type: ignore
                encoder_lr=config.encoder_lr,
                decoder_lr=config.decoder_lr,
            )
            # 層別学習率の場合, lrはparam_groupsで設定済み
            lr = config.decoder_lr
        else:
            param_groups = model.parameters()  # type: ignore
            lr = config.learning_rate

        if config.optimizer == "Adam":
            return torch.optim.Adam(param_groups, lr=lr)
        elif config.optimizer == "AdamW":
            return torch.optim.AdamW(param_groups, lr=lr)
        elif config.optimizer == "SGD":
            return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

    @classmethod
    def create_scheduler(
        cls, optimizer: torch.optim.Optimizer, config: PochiSegConfig
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """スケジューラを作成.

        Args:
            optimizer: オプティマイザ.
            config: 訓練設定 (PochiSegConfig).

        Returns:
            スケジューラ, 設定がない場合はNone.

        Raises:
            ValueError: 未対応のスケジューラ名が指定された場合.
        """
        if config.scheduler is None:
            return None

        if config.scheduler == "CosineAnnealingLR":
            return CosineAnnealingLR(optimizer, **config.scheduler_params)
        elif config.scheduler == "StepLR":
            return StepLR(optimizer, **config.scheduler_params)
        elif config.scheduler == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optimizer, **config.scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")
