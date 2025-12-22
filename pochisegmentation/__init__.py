"""Pochisegmentation - Segmentation framework based on pochitrain design philosophy."""

from pochisegmentation.factories.component_factory import ComponentFactory
from pochisegmentation.losses.seg_losses import (
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    JaccardLoss,
)
from pochisegmentation.metrics.seg_metrics import SegmentationMetrics
from pochisegmentation.models.deeplabv3plus import DeepLabV3PlusModel
from pochisegmentation.models.unet import UnetModel

# モデル登録
ComponentFactory.register_model("Unet", UnetModel)
ComponentFactory.register_model("DeepLabV3Plus", DeepLabV3PlusModel)

# 損失関数登録
ComponentFactory.register_loss("DiceLoss", DiceLoss)
ComponentFactory.register_loss("FocalLoss", FocalLoss)
ComponentFactory.register_loss("JaccardLoss", JaccardLoss)

# 評価指標登録
ComponentFactory.register_metrics("SegmentationMetrics", SegmentationMetrics)

__all__ = [
    "ComponentFactory",
    "UnetModel",
    "DeepLabV3PlusModel",
    "DiceLoss",
    "FocalLoss",
    "JaccardLoss",
    "CombinedLoss",
    "SegmentationMetrics",
]
