"""DeepLabV3PlusModelのテスト."""

import pytest
import torch

from pochisegmentation.models.deeplabv3plus import DeepLabV3PlusModel


class TestDeepLabV3PlusModel:
    """DeepLabV3PlusModelのテストクラス."""

    def test_init_default(self) -> None:
        """デフォルトパラメータでの初期化テスト."""
        model = DeepLabV3PlusModel(num_classes=4)
        assert model is not None

    def test_init_custom_encoder(self) -> None:
        """カスタムエンコーダーでの初期化テスト."""
        model = DeepLabV3PlusModel(encoder_name="resnet50", num_classes=4)
        assert model is not None

    def test_init_without_pretrained(self) -> None:
        """事前学習なしでの初期化テスト."""
        model = DeepLabV3PlusModel(num_classes=4, pretrained=False)
        assert model is not None

    def test_forward_shape(self) -> None:
        """順伝播の出力形状テスト."""
        model = DeepLabV3PlusModel(num_classes=4, pretrained=False)
        model.eval()

        batch_size = 2
        height, width = 256, 256
        x = torch.randn(batch_size, 3, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 4, height, width)

    def test_forward_different_sizes(self) -> None:
        """異なる入力サイズでの順伝播テスト."""
        model = DeepLabV3PlusModel(num_classes=2, pretrained=False)
        model.eval()

        for size in [128, 256, 512]:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 2, size, size)

    def test_get_encoder_params(self) -> None:
        """エンコーダーパラメータ取得テスト."""
        model = DeepLabV3PlusModel(num_classes=4, pretrained=False)
        encoder_params = model.get_encoder_params()

        assert isinstance(encoder_params, list)
        assert len(encoder_params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in encoder_params)

    def test_get_decoder_params(self) -> None:
        """デコーダーパラメータ取得テスト."""
        model = DeepLabV3PlusModel(num_classes=4, pretrained=False)
        decoder_params = model.get_decoder_params()

        assert isinstance(decoder_params, list)
        assert len(decoder_params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in decoder_params)

    def test_encoder_decoder_params_disjoint(self) -> None:
        """エンコーダーとデコーダーのパラメータが重複しないことをテスト."""
        model = DeepLabV3PlusModel(num_classes=4, pretrained=False)
        encoder_params = set(id(p) for p in model.get_encoder_params())
        decoder_params = set(id(p) for p in model.get_decoder_params())

        # 重複がないことを確認
        assert encoder_params.isdisjoint(decoder_params)

    def test_custom_in_channels(self) -> None:
        """カスタム入力チャンネル数テスト."""
        model = DeepLabV3PlusModel(num_classes=4, in_channels=1, pretrained=False)
        model.eval()

        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 4, 256, 256)
