"""セグメンテーションモデル."""

from pochisegmentation.models.deeplabv3plus import DeepLabV3PlusModel
from pochisegmentation.models.unet import UnetModel

__all__ = ["UnetModel", "DeepLabV3PlusModel"]
