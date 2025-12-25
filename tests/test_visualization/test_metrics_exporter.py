"""TrainingMetricsExporterのテスト."""

import tempfile
from pathlib import Path

import matplotlib

# テスト環境でGUIバックエンドを使わないように設定
matplotlib.use("Agg")

from pochisegmentation.visualization import (
    SegmentationMetricsExporter,
    TrainingMetricsExporter,
)


class TestTrainingMetricsExporter:
    """TrainingMetricsExporterクラスのテスト."""

    def test_init(self) -> None:
        """初期化のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(
                output_dir=Path(temp_dir), enable_visualization=True
            )

            assert exporter.output_dir == Path(temp_dir)
            assert exporter.enable_visualization is True
            assert len(exporter.metrics_history) == 0
            assert exporter.base_headers == [
                "epoch",
                "learning_rate",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ]

    def test_record_epoch(self) -> None:
        """エポック記録のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            exporter.record_epoch(
                epoch=1,
                learning_rate=0.001,
                train_loss=0.5,
                train_accuracy=85.0,
                val_loss=0.6,
                val_accuracy=83.0,
            )

            assert len(exporter.metrics_history) == 1
            assert exporter.metrics_history[0]["epoch"] == 1
            assert exporter.metrics_history[0]["learning_rate"] == 0.001
            assert exporter.metrics_history[0]["train_loss"] == 0.5
            assert exporter.metrics_history[0]["train_accuracy"] == 85.0
            assert exporter.metrics_history[0]["val_loss"] == 0.6
            assert exporter.metrics_history[0]["val_accuracy"] == 83.0

    def test_record_multiple_epochs(self) -> None:
        """複数エポック記録のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            for epoch in range(1, 6):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001 / epoch,
                    train_loss=1.0 / epoch,
                    train_accuracy=80.0 + epoch,
                )

            assert len(exporter.metrics_history) == 5
            assert exporter.metrics_history[0]["epoch"] == 1
            assert exporter.metrics_history[4]["epoch"] == 5
            assert exporter.metrics_history[4]["train_accuracy"] == 85.0

    def test_export_to_csv(self) -> None:
        """CSV出力のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            # メトリクスを記録
            for epoch in range(1, 4):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001,
                    train_loss=0.5,
                    train_accuracy=85.0,
                    val_loss=0.6,
                    val_accuracy=83.0,
                )

            # CSVに出力
            csv_path = exporter.export_to_csv("test_metrics.csv")

            assert csv_path is not None
            assert csv_path.exists()
            assert csv_path.name == "test_metrics.csv"

            # CSVファイルの内容確認
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 4  # ヘッダー + 3エポック
                assert "epoch,learning_rate" in lines[0]

    def test_export_to_csv_empty(self) -> None:
        """空の履歴でのCSV出力テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            csv_path = exporter.export_to_csv()

            assert csv_path is None

    def test_generate_graphs(self) -> None:
        """グラフ生成のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(
                output_dir=Path(temp_dir), enable_visualization=True
            )

            # メトリクスを記録
            for epoch in range(1, 6):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001,
                    train_loss=1.0 / epoch,
                    train_accuracy=80.0 + epoch,
                    val_loss=1.2 / epoch,
                    val_accuracy=78.0 + epoch,
                )

            # グラフを生成
            graph_paths = exporter.generate_graphs("test_graph")

            assert graph_paths is not None
            assert len(graph_paths) == 2  # 損失、精度（学習率統合）の2つ
            assert all(p.exists() for p in graph_paths)
            assert any("loss" in str(p) for p in graph_paths)
            assert any("accuracy" in str(p) for p in graph_paths)

    def test_generate_graphs_disabled(self) -> None:
        """グラフ生成が無効化された場合のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(
                output_dir=Path(temp_dir), enable_visualization=False
            )

            exporter.record_epoch(
                epoch=1, learning_rate=0.001, train_loss=0.5, train_accuracy=85.0
            )

            graph_paths = exporter.generate_graphs()

            assert graph_paths is None

    def test_export_all(self) -> None:
        """CSVとグラフの両方をエクスポートするテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(
                output_dir=Path(temp_dir), enable_visualization=True
            )

            # メトリクスを記録
            for epoch in range(1, 4):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001,
                    train_loss=0.5,
                    train_accuracy=85.0,
                )

            csv_path, graph_paths = exporter.export_all()

            assert csv_path is not None
            assert csv_path.exists()
            assert graph_paths is not None
            assert len(graph_paths) == 2  # 損失、精度（学習率統合）の2つ
            assert all(p.exists() for p in graph_paths)

    def test_get_best_epoch(self) -> None:
        """最良エポック取得のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            # メトリクスを記録（エポック3が最高精度）
            exporter.record_epoch(
                epoch=1,
                learning_rate=0.001,
                train_loss=0.5,
                train_accuracy=85.0,
                val_accuracy=80.0,
            )
            exporter.record_epoch(
                epoch=2,
                learning_rate=0.001,
                train_loss=0.4,
                train_accuracy=87.0,
                val_accuracy=85.0,
            )
            exporter.record_epoch(
                epoch=3,
                learning_rate=0.001,
                train_loss=0.3,
                train_accuracy=90.0,
                val_accuracy=88.0,
            )
            exporter.record_epoch(
                epoch=4,
                learning_rate=0.001,
                train_loss=0.35,
                train_accuracy=89.0,
                val_accuracy=86.0,
            )

            best = exporter.get_best_epoch("val_accuracy")

            assert best is not None
            assert best["epoch"] == 3
            assert best["val_accuracy"] == 88.0

    def test_get_summary(self) -> None:
        """サマリー取得のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            # メトリクスを記録
            for epoch in range(1, 6):
                exporter.record_epoch(
                    epoch=epoch,
                    learning_rate=0.001,
                    train_loss=1.0 / epoch,
                    train_accuracy=80.0 + epoch,
                    val_loss=1.2 / epoch,
                    val_accuracy=78.0 + epoch,
                )

            summary = exporter.get_summary()

            assert summary["total_epochs"] == 5
            assert summary["final_train_loss"] == 1.0 / 5
            assert summary["final_train_accuracy"] == 85.0
            assert summary["final_val_loss"] == 1.2 / 5
            assert summary["final_val_accuracy"] == 83.0
            assert summary["best_val_accuracy"] == 83.0
            assert summary["best_val_accuracy_epoch"] == 5

    def test_add_extended_headers(self) -> None:
        """拡張ヘッダー追加のテスト（Issue 9用）."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            exporter.add_extended_headers(["param_1", "param_2"])

            assert "param_1" in exporter.extended_headers
            assert "param_2" in exporter.extended_headers

            # 拡張メトリクス付きで記録
            exporter.record_epoch(
                epoch=1,
                learning_rate=0.001,
                train_loss=0.5,
                train_accuracy=85.0,
                param_1=0.123,
                param_2=0.456,
            )

            assert exporter.metrics_history[0]["param_1"] == 0.123
            assert exporter.metrics_history[0]["param_2"] == 0.456

    def test_record_without_validation_data(self) -> None:
        """検証データなしでの記録テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = TrainingMetricsExporter(output_dir=Path(temp_dir))

            exporter.record_epoch(
                epoch=1, learning_rate=0.001, train_loss=0.5, train_accuracy=85.0
            )

            assert exporter.metrics_history[0]["val_loss"] == ""
            assert exporter.metrics_history[0]["val_accuracy"] == ""

            # CSVに出力して確認
            csv_path = exporter.export_to_csv()
            assert csv_path is not None
            assert csv_path.exists()


class TestSegmentationMetricsExporter:
    """SegmentationMetricsExporterクラスのテスト."""

    def test_init(self) -> None:
        """初期化のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = SegmentationMetricsExporter(output_dir=Path(temp_dir))

            assert exporter.output_dir == Path(temp_dir)
            assert exporter.output_dir.exists()

    def test_export_history(self) -> None:
        """訓練履歴のCSV出力テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = SegmentationMetricsExporter(output_dir=Path(temp_dir))

            history = {
                "train_loss": [0.5, 0.4, 0.3],
                "val_miou": [0.6, 0.7, 0.8],
                "val_dice": [0.65, 0.75, 0.85],
                "learning_rate": [0.001, 0.0008, 0.0005],
            }

            csv_path = exporter.export_history(history)

            assert csv_path.exists()
            assert csv_path.name == "training_history.csv"

            # CSVファイルの内容確認
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 4  # ヘッダー + 3エポック
                assert "epoch,learning_rate,train_loss,val_miou,val_dice" in lines[0]

    def test_export_history_empty(self) -> None:
        """空の履歴でのCSV出力テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = SegmentationMetricsExporter(output_dir=Path(temp_dir))

            history: dict[str, list[float]] = {
                "train_loss": [],
                "val_miou": [],
                "val_dice": [],
                "learning_rate": [],
            }

            csv_path = exporter.export_history(history)

            assert csv_path.exists()
            # ヘッダーのみ
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 1

    def test_generate_graphs(self) -> None:
        """グラフ生成のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = SegmentationMetricsExporter(output_dir=Path(temp_dir))

            history = {
                "train_loss": [0.5, 0.4, 0.3],
                "val_miou": [0.6, 0.7, 0.8],
                "val_dice": [0.65, 0.75, 0.85],
                "learning_rate": [0.001, 0.0008, 0.0005],
            }

            graph_paths = exporter.generate_graphs(history)

            assert len(graph_paths) == 3  # loss, metrics, learning_rate
            assert all(p.exists() for p in graph_paths)

            # ファイル名を確認
            names = [p.name for p in graph_paths]
            assert "loss.png" in names
            assert "metrics.png" in names
            assert "learning_rate.png" in names

    def test_generate_graphs_without_validation(self) -> None:
        """検証データなしのグラフ生成テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = SegmentationMetricsExporter(output_dir=Path(temp_dir))

            history = {
                "train_loss": [0.5, 0.4, 0.3],
                "val_miou": [],
                "val_dice": [],
                "learning_rate": [0.001, 0.0008, 0.0005],
            }

            graph_paths = exporter.generate_graphs(history)

            # loss + learning_rate のみ
            assert len(graph_paths) == 2
            names = [p.name for p in graph_paths]
            assert "loss.png" in names
            assert "learning_rate.png" in names
            assert "metrics.png" not in names

    def test_generate_graphs_empty(self) -> None:
        """空の履歴でのグラフ生成テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = SegmentationMetricsExporter(output_dir=Path(temp_dir))

            history: dict[str, list[float]] = {
                "train_loss": [],
                "val_miou": [],
                "val_dice": [],
                "learning_rate": [],
            }

            graph_paths = exporter.generate_graphs(history)

            assert len(graph_paths) == 0

    def test_export_all(self) -> None:
        """CSVとグラフの両方をエクスポートするテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = SegmentationMetricsExporter(output_dir=Path(temp_dir))

            history = {
                "train_loss": [0.5, 0.4, 0.3],
                "val_miou": [0.6, 0.7, 0.8],
                "val_dice": [0.65, 0.75, 0.85],
                "learning_rate": [0.001, 0.0008, 0.0005],
            }

            csv_path, graph_paths = exporter.export_all(history)

            assert csv_path.exists()
            assert len(graph_paths) == 3
            assert all(p.exists() for p in graph_paths)

    def test_output_dir_creation(self) -> None:
        """出力ディレクトリの自動作成テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_subdir"

            exporter = SegmentationMetricsExporter(output_dir=new_dir)

            assert new_dir.exists()
            assert exporter.output_dir == new_dir
