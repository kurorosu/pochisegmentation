"""Evaluator のユニットテスト."""

from pochisegmentation.training.evaluator import Evaluator

from ._helpers import MockLoss, MockMetrics, MockModel, create_dummy_dataloader


class TestEvaluator:
    """Evaluator のテスト."""

    def test_validate_returns_metrics_with_val_loss(self) -> None:
        """検証で mIoU / Dice に加え val_loss を返す."""
        model = MockModel()
        evaluator = Evaluator(
            model=model,
            criterion=MockLoss(),
            metrics=MockMetrics(),
            device="cpu",
        )

        result = evaluator.validate(create_dummy_dataloader())
        assert result["mIoU"] == 0.5
        assert result["Dice"] == 0.6
        assert "val_loss" in result
        assert result["val_loss"] > 0

    def test_validate_sets_eval_mode(self) -> None:
        """検証後はモデルが eval モードになっている."""
        model = MockModel()
        model.train()
        evaluator = Evaluator(
            model=model,
            criterion=MockLoss(),
            metrics=MockMetrics(),
            device="cpu",
        )

        evaluator.validate(create_dummy_dataloader())
        assert model.training is False
