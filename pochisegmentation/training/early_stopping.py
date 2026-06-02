"""Early Stopping.

検証メトリクスの改善が一定エポック停滞した場合に訓練停止を通知する.
monitor で監視指標を切り替え, mIoU / Dice は高い方を, val_loss は低い方を
改善とみなす.
"""

import logging

__all__ = ["EarlyStopping", "is_higher_better"]

# 値が大きいほど良い指標. これ以外 (val_loss など) は小さいほど良いとみなす.
_HIGHER_IS_BETTER_MONITORS = frozenset({"mIoU", "Dice"})


def is_higher_better(monitor: str) -> bool:
    """指定 monitor が「値が大きいほど良い」指標か判定する.

    ベストモデル選択と Early Stopping が同一の改善方向を共有するための単一窓口.

    Args:
        monitor: 監視するメトリクス名.

    Returns:
        値が大きいほど良い場合 True (mIoU / Dice), 小さいほど良い場合 False.
    """
    return monitor in _HIGHER_IS_BETTER_MONITORS


class EarlyStopping:
    """Early Stopping の停止判定を担うクラス.

    TrainingLoop から各エポック後に呼び出され, 停止判定を委譲される.

    Args:
        patience: 改善なしの許容エポック数.
        min_delta: 改善とみなす最小変化量.
        monitor: 監視するメトリクス名 ("mIoU" / "Dice" / "val_loss").
        logger: ロガーインスタンス.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        monitor: str = "mIoU",
        logger: logging.Logger | None = None,
    ) -> None:
        """EarlyStoppingを初期化."""
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.logger = logger

        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

        self._higher_is_better = is_higher_better(monitor)

    def _is_improvement(self, current: float, best: float) -> bool:
        """現在値が best より min_delta を超えて改善しているか判定.

        Args:
            current: 現在のメトリクス値.
            best: これまでのベスト値.

        Returns:
            改善している場合 True.
        """
        if self._higher_is_better:
            return current > best + self.min_delta
        return current < best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """エポック終了後にメトリクスを評価し停止判定を行う.

        Args:
            value: 現在のエポックのメトリクス値.
            epoch: 現在のエポック番号.

        Returns:
            訓練を停止すべき場合 True.
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False

        if self._is_improvement(value, self.best_value):
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            return False

        self.counter += 1
        if self.logger:
            self.logger.info(
                f"EarlyStopping: {self.counter}/{self.patience} "
                f"(ベスト {self.monitor}: {self.best_value:.4f}, "
                f"エポック {self.best_epoch + 1})"
            )

        if self.counter >= self.patience:
            self.should_stop = True
            if self.logger:
                self.logger.info(
                    f"Early Stopping: {self.patience} エポック改善なし, "
                    f"訓練を終了します"
                )
            return True

        return False
