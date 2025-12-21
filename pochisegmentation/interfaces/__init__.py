"""セグメンテーションコンポーネントのインターフェース (DIP - 依存性逆転の原則)."""

from pochisegmentation.interfaces.dataset import ISegmentationDataset
from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.interfaces.model import ISegmentationModel

__all__ = [
    "ISegmentationModel",
    "ISegmentationLoss",
    "ISegmentationMetrics",
    "ISegmentationDataset",
]
