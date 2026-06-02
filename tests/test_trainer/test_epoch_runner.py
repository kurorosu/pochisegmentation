"""EpochRunner のユニットテスト."""

import torch

from pochisegmentation.training.epoch_runner import EpochRunner

from ._helpers import MockLoss, MockModel, create_dummy_dataloader


class TestEpochRunner:
    """EpochRunner のテスト."""

    def test_run_returns_positive_loss(self) -> None:
        """1エポック実行で正の平均損失を返す."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        runner = EpochRunner(
            model=model,
            criterion=MockLoss(),
            optimizer=optimizer,
            device="cpu",
        )

        loss = runner.run(create_dummy_dataloader())
        assert loss > 0

    def test_run_sets_train_mode(self) -> None:
        """実行後はモデルが train モードになっている."""
        model = MockModel()
        model.eval()
        optimizer = torch.optim.Adam(model.parameters())
        runner = EpochRunner(
            model=model,
            criterion=MockLoss(),
            optimizer=optimizer,
            device="cpu",
        )

        runner.run(create_dummy_dataloader())
        assert model.training is True

    def test_run_updates_parameters(self) -> None:
        """訓練でパラメータが更新される."""
        model = MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        runner = EpochRunner(
            model=model,
            criterion=MockLoss(),
            optimizer=optimizer,
            device="cpu",
        )

        before = model.conv.weight.detach().clone()
        runner.run(create_dummy_dataloader())
        after = model.conv.weight.detach()
        assert not torch.equal(before, after)
