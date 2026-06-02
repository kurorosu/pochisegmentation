"""MetricsTracker のユニットテスト."""

import logging
import tempfile

from pochisegmentation.training.metrics_tracker import MetricsTracker
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager

from ._helpers import MockMetrics

_HISTORY: dict[str, list[float]] = {
    "train_loss": [1.0, 0.8],
    "val_miou": [0.4, 0.5],
    "val_dice": [0.5, 0.6],
    "val_loss": [1.1, 0.9],
    "learning_rate": [1e-3, 1e-3],
}


class TestMetricsTracker:
    """MetricsTracker のテスト."""

    def test_finalize_without_workspace_is_noop(self) -> None:
        """ワークスペースなしでは何も出力せずに完了する."""
        tracker = MetricsTracker(
            metrics=MockMetrics(),
            logger=logging.getLogger("test"),
            workspace_manager=None,
        )
        # 例外なく完了すれば良い
        tracker.finalize(_HISTORY)

    def test_finalize_exports_history_csv(self) -> None:
        """ワークスペースありで訓練履歴 CSV が出力される."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wm = PochiWorkspaceManager(base_dir=tmpdir)
            wm.create_workspace()
            tracker = MetricsTracker(
                metrics=MockMetrics(),
                logger=logging.getLogger("test"),
                workspace_manager=wm,
            )

            tracker.finalize(_HISTORY)

            csv_path = wm.get_visualization_dir() / "training_history.csv"
            assert csv_path.exists()
