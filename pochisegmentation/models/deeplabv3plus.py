"""DeepLabV3+モデルの実装."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from pochisegmentation.interfaces.model import ISegmentationModel


class DeepLabV3PlusModel(ISegmentationModel):
    """DeepLabV3+モデル (smpラッパー).

    segmentation_models_pytorchのDeepLabV3Plusをラップし,
    ISegmentationModelインターフェースを実装する.

    Attributes:
        _model: smp.DeepLabV3Plusインスタンス.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        num_classes: int = 1,
        pretrained: bool = True,
        in_channels: int = 3,
    ) -> None:
        """DeepLabV3+モデルを初期化.

        Args:
            encoder_name: エンコーダーのアーキテクチャ名.
            num_classes: 出力クラス数.
            pretrained: ImageNet事前学習済み重みを使用するかどうか.
            in_channels: 入力チャンネル数.
        """
        super().__init__()
        self._model: nn.Module = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
            in_channels=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播.

        Args:
            x: 入力テンソル, 形状は (B, C, H, W).

        Returns:
            出力テンソル, 形状は (B, num_classes, H, W).
        """
        return self._model(x)

    def get_encoder_params(self) -> list:
        """層別学習率用のエンコーダーパラメータを取得.

        Returns:
            エンコーダーパラメータのリスト.
        """
        return list(self._model.encoder.parameters())

    def get_decoder_params(self) -> list:
        """層別学習率用のデコーダーパラメータを取得.

        Returns:
            デコーダーおよびセグメンテーションヘッドのパラメータリスト.
        """
        return list(self._model.decoder.parameters()) + list(
            self._model.segmentation_head.parameters()
        )
