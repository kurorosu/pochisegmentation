"""セグメンテーションモデル."""

from pochisegmentation.interfaces.model import ISegmentationModel
from pochisegmentation.models.deeplabv3plus import DeepLabV3PlusModel
from pochisegmentation.models.unet import UnetModel

__all__ = ["UnetModel", "DeepLabV3PlusModel", "create_model"]

# 利用可能なアーキテクチャのレジストリ
_MODEL_REGISTRY: dict[str, type[ISegmentationModel]] = {
    "Unet": UnetModel,
    "DeepLabV3Plus": DeepLabV3PlusModel,
}


def create_model(
    name: str,
    num_classes: int,
    encoder_name: str = "resnet34",
    pretrained: bool = True,
    in_channels: int = 3,
) -> ISegmentationModel:
    """アーキテクチャ名からモデルを生成する.

    ComponentFactory.create_model の薄い別表記. 設定 dict / PochiSegConfig を
    介さずに個別引数でモデルを生成したい場合に使う.

    Args:
        name: アーキテクチャ名 ("Unet" または "DeepLabV3Plus").
        num_classes: 出力クラス数.
        encoder_name: エンコーダー名.
        pretrained: ImageNet 事前学習済み重みを使用するか.
        in_channels: 入力チャンネル数.

    Returns:
        生成されたモデルインスタンス.

    Raises:
        ValueError: 未登録のアーキテクチャ名が指定された場合.
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"未登録のモデル: {name}. 利用可能: {available}")

    return _MODEL_REGISTRY[name](
        encoder_name=encoder_name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
    )
