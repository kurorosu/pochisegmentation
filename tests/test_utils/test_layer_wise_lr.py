"""層別学習率ヘルパーのテスト."""

import torch

from pochisegmentation.interfaces.model import ISegmentationModel
from pochisegmentation.utils.layer_wise_lr import create_layer_wise_param_groups


class MockModel(ISegmentationModel):
    """テスト用モックモデル."""

    def __init__(self) -> None:
        """MockModelを初期化."""
        super().__init__()
        self.encoder_conv = torch.nn.Conv2d(3, 64, kernel_size=3)
        self.decoder_conv = torch.nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播."""
        x = self.encoder_conv(x)
        return self.decoder_conv(x)

    def get_encoder_params(self) -> list:
        """エンコーダーパラメータを取得."""
        return list(self.encoder_conv.parameters())

    def get_decoder_params(self) -> list:
        """デコーダーパラメータを取得."""
        return list(self.decoder_conv.parameters())


class TestCreateLayerWiseParamGroups:
    """create_layer_wise_param_groupsのテストクラス."""

    def test_returns_two_groups(self) -> None:
        """2つのパラメータグループを返すことをテスト."""
        model = MockModel()
        param_groups = create_layer_wise_param_groups(
            model, encoder_lr=1e-4, decoder_lr=1e-3
        )

        assert len(param_groups) == 2

    def test_encoder_lr(self) -> None:
        """エンコーダー学習率のテスト."""
        model = MockModel()
        encoder_lr = 1e-4
        param_groups = create_layer_wise_param_groups(
            model, encoder_lr=encoder_lr, decoder_lr=1e-3
        )

        assert param_groups[0]["lr"] == encoder_lr

    def test_decoder_lr(self) -> None:
        """デコーダー学習率のテスト."""
        model = MockModel()
        decoder_lr = 1e-3
        param_groups = create_layer_wise_param_groups(
            model, encoder_lr=1e-4, decoder_lr=decoder_lr
        )

        assert param_groups[1]["lr"] == decoder_lr

    def test_encoder_params_present(self) -> None:
        """エンコーダーパラメータが含まれていることをテスト."""
        model = MockModel()
        param_groups = create_layer_wise_param_groups(
            model, encoder_lr=1e-4, decoder_lr=1e-3
        )

        encoder_params = list(model.get_encoder_params())
        assert len(param_groups[0]["params"]) == len(encoder_params)

    def test_decoder_params_present(self) -> None:
        """デコーダーパラメータが含まれていることをテスト."""
        model = MockModel()
        param_groups = create_layer_wise_param_groups(
            model, encoder_lr=1e-4, decoder_lr=1e-3
        )

        decoder_params = list(model.get_decoder_params())
        assert len(param_groups[1]["params"]) == len(decoder_params)

    def test_can_pass_to_optimizer(self) -> None:
        """オプティマイザに渡せることをテスト."""
        model = MockModel()
        param_groups = create_layer_wise_param_groups(
            model, encoder_lr=1e-4, decoder_lr=1e-3
        )

        # オプティマイザに渡してエラーが出ないことを確認
        optimizer = torch.optim.AdamW(param_groups)
        assert optimizer is not None
        assert len(optimizer.param_groups) == 2

    def test_optimizer_uses_correct_lr(self) -> None:
        """オプティマイザが正しい学習率を使用することをテスト."""
        model = MockModel()
        encoder_lr = 1e-4
        decoder_lr = 1e-3
        param_groups = create_layer_wise_param_groups(
            model, encoder_lr=encoder_lr, decoder_lr=decoder_lr
        )

        optimizer = torch.optim.AdamW(param_groups)

        assert optimizer.param_groups[0]["lr"] == encoder_lr
        assert optimizer.param_groups[1]["lr"] == decoder_lr
