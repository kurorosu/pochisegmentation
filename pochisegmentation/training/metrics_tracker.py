"""訓練履歴とクラス別メトリクスの出力.

訓練履歴 (CSV / グラフ) とクラス別精度 (IoU チャート / Confusion Matrix) の
エクスポートを一手に扱う. 既存の SegmentationMetricsExporter /
ClassMetricsVisualizer に処理を委譲する.
"""

import logging

from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager
from pochisegmentation.visualization.class_metrics_visualizer import (
    ClassMetricsVisualizer,
)
from pochisegmentation.visualization.metrics_exporter import SegmentationMetricsExporter

__all__ = ["MetricsTracker"]


class MetricsTracker:
    """訓練履歴とクラス別メトリクスのエクスポートを担うクラス.

    Args:
        metrics: 評価指標 (クラス別精度の算出元).
        logger: ロガーインスタンス.
        workspace_manager: ワークスペースマネージャ (None なら何もしない).
    """

    def __init__(
        self,
        metrics: ISegmentationMetrics,
        logger: logging.Logger,
        workspace_manager: PochiWorkspaceManager | None = None,
    ) -> None:
        """MetricsTrackerを初期化."""
        self._metrics = metrics
        self._logger = logger
        self._workspace_manager = workspace_manager

    def finalize(self, history: dict[str, list[float]]) -> None:
        """訓練履歴とクラス別メトリクスを出力する.

        Args:
            history: 訓練履歴の辞書.
        """
        self._export_history(history)
        self._export_class_metrics()

    def _export_history(self, history: dict[str, list[float]]) -> None:
        """訓練履歴を CSV とグラフで保存する.

        Args:
            history: 訓練履歴の辞書.
        """
        if self._workspace_manager is None:
            return

        vis_dir = self._workspace_manager.get_visualization_dir()
        exporter = SegmentationMetricsExporter(
            output_dir=vis_dir,
            logger=self._logger,
        )
        exporter.export_all(history)

    def _export_class_metrics(self) -> None:
        """クラス別精度を可視化・保存する."""
        if self._workspace_manager is None:
            return

        # compute_class_metrics メソッドが存在するか確認
        if not hasattr(self._metrics, "compute_class_metrics"):
            return

        vis_dir = self._workspace_manager.get_visualization_dir()
        class_metrics = self._metrics.compute_class_metrics()

        visualizer = ClassMetricsVisualizer(vis_dir)
        paths = visualizer.export_all(class_metrics)

        self._logger.info(f"クラス別精度を保存: {paths['class_iou_chart']}")
        self._logger.info(f"Confusion Matrix を保存: {paths['confusion_matrix']}")
