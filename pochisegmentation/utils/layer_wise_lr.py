"""層別学習率ヘルパー.

エンコーダーとデコーダーで異なる学習率を適用するためのヘルパー関数.
"""

from pochisegmentation.interfaces.model import ISegmentationModel


def create_layer_wise_param_groups(
    model: ISegmentationModel,
    encoder_lr: float,
    decoder_lr: float,
) -> list[dict]:
    """エンコーダー/デコーダー用の層別学習率パラメータグループを作成.

    ISegmentationModelインターフェースに依存するため,
    具象モデル (Unet, DeepLabV3+) を知らなくてよい.

    Args:
        model: セグメンテーションモデル (ISegmentationModel).
        encoder_lr: エンコーダーの学習率.
        decoder_lr: デコーダーの学習率.

    Returns:
        オプティマイザに渡すパラメータグループのリスト.

    Examples:
        >>> model = UnetModel(encoder_name="resnet34", num_classes=4)
        >>> param_groups = create_layer_wise_param_groups(
        ...     model, encoder_lr=1e-4, decoder_lr=1e-3
        ... )
        >>> optimizer = torch.optim.AdamW(param_groups)
    """
    return [
        {"params": model.get_encoder_params(), "lr": encoder_lr},
        {"params": model.get_decoder_params(), "lr": decoder_lr},
    ]
