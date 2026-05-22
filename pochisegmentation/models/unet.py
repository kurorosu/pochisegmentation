"""Unetモデルの実装."""

from typing import cast

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from pochisegmentation.interfaces.model import ISegmentationModel


class UnetModel(ISegmentationModel):
    """Unetモデル (smpラッパー).

    segmentation_models_pytorchのUnetをラップし,
    ISegmentationModelインターフェースを実装する.

    Attributes:
        _model: smp.Unetインスタンス.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        num_classes: int = 1,
        pretrained: bool = True,
        in_channels: int = 3,
    ) -> None:
        """Unetモデルを初期化.

        Args:
            encoder_name: エンコーダーのアーキテクチャ名.
            num_classes: 出力クラス数.
            pretrained: ImageNet事前学習済み重みを使用するかどうか.
            in_channels: 入力チャンネル数.
        """
        super().__init__()
        self._model: nn.Module = smp.Unet(
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
        output: torch.Tensor = self._model(x)
        return output

    def get_encoder_params(self) -> list:
        """層別学習率用のエンコーダーパラメータを取得.

        Returns:
            エンコーダーパラメータのリスト.
        """
        encoder = cast(nn.Module, self._model.encoder)
        return list(encoder.parameters())

    def get_decoder_params(self) -> list:
        """層別学習率用のデコーダーパラメータを取得.

        Returns:
            デコーダーおよびセグメンテーションヘッドのパラメータリスト.
        """
        decoder = cast(nn.Module, self._model.decoder)
        segmentation_head = cast(nn.Module, self._model.segmentation_head)
        return list(decoder.parameters()) + list(segmentation_head.parameters())
