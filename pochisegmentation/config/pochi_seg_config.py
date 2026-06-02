"""pochisegmentation.config.pochi_seg_config: 型付き設定の Pydantic モデル.

dict ベースの設定を置き換える, セグメンテーション訓練/推論の型付き設定.
フィールドは設定ファイル (configs/pochi_seg_config.py) と対話ウィザードが生成する
フラットなキー構造に揃えてある.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["PochiSegConfig"]

# スケジューラごとに必須となる scheduler_params のキー.
_SCHEDULER_REQUIRED_PARAMS: dict[str, list[str]] = {
    "CosineAnnealingLR": ["T_max"],
    "StepLR": ["step_size"],
}


class PochiSegConfig(BaseModel):
    """セグメンテーション訓練/推論の型付き設定.

    Pydantic v2 によりフィールド型と値域を宣言的に検証する.
    dict.get(...) アクセスを属性アクセスに置き換えるための単一の設定モデル.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # =========================================================================
    # 必須フィールド
    # =========================================================================
    data_root: str
    num_classes: int = Field(gt=0)

    # =========================================================================
    # モデル設定
    # =========================================================================
    architecture: Literal["Unet", "DeepLabV3Plus"] = "Unet"
    encoder_name: str = "resnet34"
    pretrained: bool = True
    in_channels: int = Field(default=3, gt=0)

    # =========================================================================
    # データ設定
    # =========================================================================
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = Field(default=256, gt=0)
    batch_size: int = Field(default=16, gt=0)
    num_workers: int = Field(default=4, ge=0)

    # =========================================================================
    # 訓練設定
    # =========================================================================
    epochs: int = Field(default=100, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    optimizer: Literal["Adam", "AdamW", "SGD"] = "AdamW"

    # =========================================================================
    # 損失関数
    # =========================================================================
    loss: Literal["DiceLoss", "FocalLoss", "JaccardLoss", "CombinedLoss"] = "DiceLoss"
    loss_params: dict[str, Any] = Field(default_factory=dict)

    # =========================================================================
    # 評価指標
    # =========================================================================
    metrics: str = "SegmentationMetrics"

    # =========================================================================
    # スケジューラ
    # =========================================================================
    scheduler: Literal["CosineAnnealingLR", "StepLR", "ReduceLROnPlateau"] | None = None
    scheduler_params: dict[str, Any] = Field(default_factory=dict)

    # =========================================================================
    # 層別学習率
    # =========================================================================
    enable_layer_wise_lr: bool = False
    encoder_lr: float = Field(default=1e-4, gt=0.0)
    decoder_lr: float = Field(default=1e-3, gt=0.0)

    # =========================================================================
    # Early Stopping
    # =========================================================================
    early_stopping_patience: int | None = Field(default=None, ge=0)
    # ベストモデル選択と Early Stopping が共有する監視メトリクス.
    early_stopping_monitor: Literal["mIoU", "Dice", "val_loss"] = "mIoU"

    # =========================================================================
    # AMP (Automatic Mixed Precision)
    # =========================================================================
    enable_amp: bool = False

    # =========================================================================
    # ワークスペース設定
    # =========================================================================
    work_dir: str = "work_dirs"
    device: Literal["cuda", "cpu"] = "cuda"

    # =========================================================================
    # 可視化・ログ設定
    # =========================================================================
    enable_metrics_export: bool = True

    @model_validator(mode="after")
    def _validate_scheduler_params(self) -> "PochiSegConfig":
        """Scheduler と scheduler_params の整合性を検証する.

        Returns:
            検証済みの自身.

        Raises:
            ValueError: スケジューラに必須のパラメータが欠けている場合.
        """
        required = _SCHEDULER_REQUIRED_PARAMS.get(self.scheduler or "", [])
        missing = [key for key in required if key not in self.scheduler_params]
        if missing:
            raise ValueError(
                f"scheduler '{self.scheduler}' には " f"パラメータ {missing} が必須です"
            )
        return self

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PochiSegConfig":
        """Dict から PochiSegConfig を生成する.

        Pydantic のバリデーションが自動実行される. 未知のキーは無視される.

        Args:
            config: 設定 dict.

        Returns:
            検証済みの PochiSegConfig.
        """
        return cls.model_validate(config)
