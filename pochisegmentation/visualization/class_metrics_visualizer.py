"""クラス別精度の可視化."""

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class ClassMetricsVisualizer:
    """クラス別精度の可視化.

    棒グラフ・Confusion Matrix・CSV 出力を担当.

    Attributes:
        _output_dir: 出力ディレクトリ.
    """

    def __init__(self, output_dir: Path) -> None:
        """初期化.

        Args:
            output_dir: 出力ディレクトリ.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_class_iou_chart(
        self,
        iou_per_class: list[float],
        class_names: list[str],
        filename: str = "class_iou.png",
    ) -> Path:
        """クラス別 IoU 棒グラフを保存.

        Args:
            iou_per_class: クラス別 IoU のリスト.
            class_names: クラス名リスト.
            filename: 出力ファイル名.

        Returns:
            保存先パス.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(class_names))
        bars = ax.bar(x, iou_per_class, color="steelblue", edgecolor="black")

        # 値をバーの上に表示
        for bar, iou in zip(bars, iou_per_class):
            height = bar.get_height()
            ax.annotate(
                f"{iou:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("IoU", fontsize=12)
        ax.set_title("Per-Class IoU", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        output_path = self._output_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        return output_path

    def save_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: list[str],
        filename: str = "confusion_matrix.png",
        normalize: bool = True,
    ) -> Path:
        """Confusion Matrix を保存.

        Args:
            confusion_matrix: Confusion Matrix (num_classes x num_classes).
            class_names: クラス名リスト.
            filename: 出力ファイル名.
            normalize: 正規化するかどうか (行方向で正規化).

        Returns:
            保存先パス.
        """
        cm = confusion_matrix.astype(float)

        if normalize:
            # 行方向で正規化 (各クラスの実際のサンプル数で割る)
            row_sums = cm.sum(axis=1, keepdims=True)
            # ゼロ除算を防ぐ
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm = cm / row_sums

        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # タイトル (正規化の説明を含む)
        if normalize:
            title = "Confusion Matrix\n(Normalized by row: % of true label predicted as each class)"
        else:
            title = "Confusion Matrix\n(Raw pixel counts)"

        # ラベル設定
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        # X軸ラベルを回転
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # セル内に値を表示
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                value = cm[i, j]
                text = f"{value:{fmt}}" if normalize else f"{int(value)}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > thresh else "black",
                    fontsize=10,
                )

        plt.tight_layout()

        output_path = self._output_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        return output_path

    def save_class_metrics_csv(
        self,
        metrics: dict[str, Any],
        filename: str = "class_metrics.csv",
    ) -> Path:
        """クラス別精度を CSV 出力.

        Args:
            metrics: クラス別精度の辞書.
            filename: 出力ファイル名.

        Returns:
            保存先パス.
        """
        output_path = self._output_dir / filename

        class_names = metrics["class_names"]
        per_class_iou = metrics["per_class_iou"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "iou"])

            for i, (name, iou) in enumerate(zip(class_names, per_class_iou)):
                writer.writerow([i, name, f"{iou:.6f}"])

        return output_path

    def export_all(self, metrics: dict[str, Any]) -> dict[str, Path]:
        """全ての可視化を出力.

        Args:
            metrics: ClassMetrics.compute() の戻り値.

        Returns:
            出力ファイルパスの辞書.
        """
        class_names = metrics["class_names"]
        per_class_iou = metrics["per_class_iou"]
        confusion_matrix = metrics["confusion_matrix"]

        paths = {}

        # クラス別 IoU 棒グラフ
        paths["class_iou_chart"] = self.save_class_iou_chart(per_class_iou, class_names)

        # Confusion Matrix
        paths["confusion_matrix"] = self.save_confusion_matrix(
            confusion_matrix, class_names
        )

        # CSV
        paths["class_metrics_csv"] = self.save_class_metrics_csv(metrics)

        return paths
