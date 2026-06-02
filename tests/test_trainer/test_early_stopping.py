"""EarlyStopping のユニットテスト."""

from pochisegmentation.training.early_stopping import EarlyStopping, is_higher_better


class TestIsHigherBetter:
    """is_higher_better のテスト."""

    def test_miou_and_dice_are_higher_better(self) -> None:
        """mIoU / Dice は高い方が良い."""
        assert is_higher_better("mIoU") is True
        assert is_higher_better("Dice") is True

    def test_val_loss_is_lower_better(self) -> None:
        """val_loss は低い方が良い."""
        assert is_higher_better("val_loss") is False


class TestEarlyStoppingHigherBetter:
    """高い方が良い指標 (mIoU) の Early Stopping テスト."""

    def test_first_step_records_best_without_stop(self) -> None:
        """初回は best を記録し停止しない."""
        es = EarlyStopping(patience=2, monitor="mIoU")
        assert es.step(0.5, 0) is False
        assert es.best_value == 0.5
        assert es.best_epoch == 0
        assert es.counter == 0

    def test_improvement_resets_counter(self) -> None:
        """改善すると counter がリセットされる."""
        es = EarlyStopping(patience=2, monitor="mIoU")
        es.step(0.5, 0)
        es.step(0.4, 1)  # 改善なし -> counter=1
        assert es.counter == 1
        es.step(0.6, 2)  # 改善 -> counter=0
        assert es.counter == 0
        assert es.best_value == 0.6
        assert es.best_epoch == 2

    def test_stops_after_patience(self) -> None:
        """patience を超える改善なしで停止する."""
        es = EarlyStopping(patience=2, monitor="mIoU")
        assert es.step(0.5, 0) is False
        assert es.step(0.5, 1) is False  # counter=1
        assert es.step(0.5, 2) is True  # counter=2 >= patience
        assert es.should_stop is True


class TestEarlyStoppingLowerBetter:
    """低い方が良い指標 (val_loss) の Early Stopping テスト."""

    def test_lower_value_is_improvement(self) -> None:
        """より小さい値が改善とみなされる."""
        es = EarlyStopping(patience=2, monitor="val_loss")
        es.step(1.0, 0)
        es.step(0.5, 1)  # 改善
        assert es.best_value == 0.5
        assert es.counter == 0

    def test_higher_value_is_not_improvement(self) -> None:
        """より大きい値は改善ではない."""
        es = EarlyStopping(patience=2, monitor="val_loss")
        es.step(1.0, 0)
        es.step(1.5, 1)  # 悪化 -> counter=1
        assert es.best_value == 1.0
        assert es.counter == 1
