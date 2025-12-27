"""依存性注入用コンポーネントファクトリー."""

from typing import Any, Callable

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

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
    def create_model(cls, config: dict[str, Any]) -> ISegmentationModel:
        """設定からモデルを生成.

        Args:
            config: モデル設定を含む辞書.
                - architecture: モデル名 (デフォルト: "Unet").
                - encoder_name: エンコーダー名 (デフォルト: "resnet34").
                - num_classes: クラス数 (必須).
                - pretrained: 事前学習済み重みを使用するか (デフォルト: True).
                - in_channels: 入力チャンネル数 (デフォルト: 3).

        Returns:
            生成されたモデルインスタンス.

        Raises:
            ValueError: 未登録のモデル名が指定された場合.
            KeyError: 必須パラメータが設定に存在しない場合.
        """
        model_name = config.get("architecture", "Unet")
        if model_name not in cls._model_registry:
            available = list(cls._model_registry.keys())
            raise ValueError(f"未登録のモデル: {model_name}. 利用可能: {available}")

        return cls._model_registry[model_name](
            encoder_name=config.get("encoder_name", "resnet34"),
            num_classes=config["num_classes"],
            pretrained=config.get("pretrained", True),
            in_channels=config.get("in_channels", 3),
        )

    @classmethod
    def create_loss(cls, config: dict[str, Any]) -> ISegmentationLoss:
        """設定から損失関数を生成.

        Args:
            config: 損失関数設定を含む辞書.
                - loss: 損失関数名 (デフォルト: "DiceLoss").
                - loss_params: 損失関数に渡す追加パラメータ (デフォルト: {}).

        Returns:
            生成された損失関数インスタンス.

        Raises:
            ValueError: 未登録の損失関数名が指定された場合.
        """
        loss_name = config.get("loss", "DiceLoss")
        if loss_name not in cls._loss_registry:
            available = list(cls._loss_registry.keys())
            raise ValueError(f"未登録の損失関数: {loss_name}. 利用可能: {available}")

        loss_params = config.get("loss_params", {})
        return cls._loss_registry[loss_name](**loss_params)

    @classmethod
    def create_metrics(
        cls, config: dict[str, Any], class_names: list[str] | None = None
    ) -> ISegmentationMetrics:
        """設定から評価指標を生成.

        Args:
            config: 評価指標設定を含む辞書.
                - metrics: 評価指標名 (デフォルト: "SegmentationMetrics").
                - num_classes: クラス数 (必須).
                - device: 計算デバイス (デフォルト: "cuda").
            class_names: クラス名リスト (オプション).

        Returns:
            生成された評価指標インスタンス.

        Raises:
            ValueError: 未登録の評価指標名が指定された場合.
            KeyError: 必須パラメータが設定に存在しない場合.
        """
        metrics_name = config.get("metrics", "SegmentationMetrics")
        if metrics_name not in cls._metrics_registry:
            available = list(cls._metrics_registry.keys())
            raise ValueError(f"未登録の評価指標: {metrics_name}. 利用可能: {available}")

        return cls._metrics_registry[metrics_name](
            num_classes=config["num_classes"],
            class_names=class_names,
            device=config.get("device", "cuda"),
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
        cls, model: torch.nn.Module, config: dict[str, Any]
    ) -> torch.optim.Optimizer:
        """オプティマイザを作成.

        Args:
            model: モデル.
            config: 設定辞書.

        Returns:
            オプティマイザ.
        """
        # 層別学習率
        if config.get("enable_layer_wise_lr", False):
            param_groups = create_layer_wise_param_groups(
                model,  # type: ignore
                encoder_lr=config.get("encoder_lr", 1e-4),
                decoder_lr=config.get("decoder_lr", 1e-3),
            )
            # 層別学習率の場合, lrはparam_groupsで設定済み
            lr = config.get("decoder_lr", 1e-3)
        else:
            param_groups = model.parameters()  # type: ignore
            lr = config.get("learning_rate", 1e-3)

        optimizer_name = config.get("optimizer", "AdamW")

        if optimizer_name == "Adam":
            return torch.optim.Adam(param_groups, lr=lr)
        elif optimizer_name == "AdamW":
            return torch.optim.AdamW(param_groups, lr=lr)
        elif optimizer_name == "SGD":
            return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    @classmethod
    def create_scheduler(
        cls, optimizer: torch.optim.Optimizer, config: dict[str, Any]
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """スケジューラを作成.

        Args:
            optimizer: オプティマイザ.
            config: 設定辞書.

        Returns:
            スケジューラ, 設定がない場合はNone.
        """
        scheduler_name = config.get("scheduler")
        if scheduler_name is None:
            return None

        scheduler_params = config.get("scheduler_params", {})

        if scheduler_name == "CosineAnnealingLR":
            return CosineAnnealingLR(optimizer, **scheduler_params)
        elif scheduler_name == "StepLR":
            return StepLR(optimizer, **scheduler_params)
        elif scheduler_name == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
