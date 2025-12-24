"""セグメンテーション訓練クラス.

DIP (依存性逆転原則) に基づき, 具象クラスではなくインターフェースに依存.
DI (依存性注入) により, コンストラクタで依存性を注入.
"""

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.interfaces.model import ISegmentationModel
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager


class PochiSegmentationTrainer:
    """セグメンテーション訓練クラス.

    DIP: 具象クラスではなくインターフェースに依存.
    DI: コンストラクタで依存性を注入.

    Attributes:
        _model: 訓練対象のモデル.
        _criterion: 損失関数.
        _metrics: 評価指標.
        _optimizer: オプティマイザ.
        _scheduler: 学習率スケジューラ.
        _device: 使用デバイス.
        _config: 設定辞書.
    """

    def __init__(
        self,
        model: ISegmentationModel,
        criterion: ISegmentationLoss,
        metrics: ISegmentationMetrics,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        device: str = "cuda",
        config: dict[str, Any] | None = None,
        workspace_manager: PochiWorkspaceManager | None = None,
    ) -> None:
        """PochiSegmentationTrainerを初期化.

        Args:
            model: セグメンテーションモデル (ISegmentationModel).
            criterion: 損失関数 (ISegmentationLoss).
            metrics: 評価指標 (ISegmentationMetrics).
            optimizer: オプティマイザ.
            scheduler: 学習率スケジューラ (オプション).
            device: 使用デバイス ("cuda" or "cpu").
            config: 設定辞書 (オプション).
            workspace_manager: ワークスペースマネージャ (オプション).
        """
        self._model = model.to(device)
        self._criterion = criterion
        self._metrics = metrics
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._config = config or {}
        self._workspace_manager = workspace_manager

        # ベストスコア管理
        self._best_miou = 0.0
        self._best_epoch = 0

        # ロガー
        logger_manager = LoggerManager()
        self._logger = logger_manager.get_logger("PochiSegmentationTrainer")

    def train(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
        epochs: int = 50,
    ) -> dict[str, list[float]]:
        """訓練ループを実行.

        Args:
            train_loader: 訓練データローダー.
            val_loader: 検証データローダー (オプション).
            epochs: エポック数.

        Returns:
            訓練履歴 (損失と評価指標).
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_miou": [],
            "val_dice": [],
            "learning_rate": [],
        }

        self._logger.info(f"訓練開始: {epochs} エポック")

        for epoch in range(epochs):
            # 現在の学習率を記録
            current_lr = self._optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(current_lr)

            # 訓練フェーズ
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            self._logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}"
            )

            # 検証フェーズ
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                history["val_miou"].append(val_metrics.get("mIoU", 0.0))
                history["val_dice"].append(val_metrics.get("Dice", 0.0))

                self._logger.info(
                    f"  Val mIoU: {val_metrics.get('mIoU', 0.0):.4f}, "
                    f"Dice: {val_metrics.get('Dice', 0.0):.4f}"
                )

                # ベストモデルの保存
                self._save_best_model(val_metrics, epoch)

            # スケジューラ更新
            if self._scheduler is not None:
                self._scheduler.step()

        self._logger.info(
            f"訓練完了. Best mIoU: {self._best_miou:.4f} (Epoch {self._best_epoch + 1})"
        )

        # 訓練履歴を可視化
        self._save_training_history(history)

        return history

    def _train_epoch(
        self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """1エポックの訓練を実行.

        Args:
            loader: 訓練データローダー.

        Returns:
            平均訓練損失.
        """
        self._model.train()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(self._device)
            masks = masks.to(self._device)

            # 順伝播
            outputs = self._model(images)
            loss = self._criterion(outputs, masks)

            # 逆伝播
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate(
        self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, float]:
        """検証を実行.

        Args:
            loader: 検証データローダー.

        Returns:
            評価指標の辞書.
        """
        self._model.eval()
        self._metrics.reset()

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self._device)
                masks = masks.to(self._device)

                outputs = self._model(images)
                preds = outputs.argmax(dim=1)

                self._metrics.update(preds, masks)

        return self._metrics.compute()

    def _save_best_model(self, metrics: dict[str, float], epoch: int) -> None:
        """ベストモデルを保存.

        Args:
            metrics: 評価指標の辞書.
            epoch: 現在のエポック.
        """
        miou = metrics.get("mIoU", 0.0)
        if miou > self._best_miou:
            self._best_miou = miou
            self._best_epoch = epoch

            if self._workspace_manager is not None:
                models_dir = self._workspace_manager.get_models_dir()
                model_path = models_dir / "best.pth"
                self._save_checkpoint(model_path, epoch, metrics)
                self._logger.info(f"ベストモデルを保存: {model_path}")

    def _save_checkpoint(
        self, path: Path, epoch: int, metrics: dict[str, float]
    ) -> None:
        """チェックポイントを保存.

        Args:
            path: 保存先パス.
            epoch: エポック番号.
            metrics: 評価指標.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "metrics": metrics,
            "best_miou": self._best_miou,
        }

        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()

        torch.save(checkpoint, path)

    def save_last_model(self) -> Path | None:
        """最終モデルを保存.

        Returns:
            保存先パス, ワークスペースマネージャがない場合はNone.
        """
        if self._workspace_manager is None:
            return None

        models_dir = self._workspace_manager.get_models_dir()
        model_path = models_dir / "last.pth"
        self._save_checkpoint(model_path, self._best_epoch, {"mIoU": self._best_miou})
        self._logger.info(f"最終モデルを保存: {model_path}")
        return model_path

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """チェックポイントを読み込み.

        Args:
            path: チェックポイントファイルパス.

        Returns:
            チェックポイント情報.
        """
        checkpoint: dict[str, Any] = torch.load(path, map_location=self._device)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self._scheduler is not None and "scheduler_state_dict" in checkpoint:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._best_miou = checkpoint.get("best_miou", 0.0)
        self._best_epoch = checkpoint.get("epoch", 0)

        self._logger.info(
            f"チェックポイントを読み込み: {path} (Epoch {self._best_epoch + 1})"
        )

        return checkpoint

    def _save_training_history(self, history: dict[str, list[float]]) -> None:
        """訓練履歴をCSVとグラフで保存.

        Args:
            history: 訓練履歴の辞書.
        """
        if self._workspace_manager is None:
            return

        vis_dir = self._workspace_manager.get_visualization_dir()

        # CSVに保存
        csv_path = vis_dir / "training_history.csv"
        epochs = len(history["train_loss"])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "learning_rate", "train_loss", "val_miou", "val_dice"]
            )
            for i in range(epochs):
                writer.writerow(
                    [
                        i + 1,
                        history["learning_rate"][i],
                        history["train_loss"][i],
                        history["val_miou"][i] if history["val_miou"] else "",
                        history["val_dice"][i] if history["val_dice"] else "",
                    ]
                )
        self._logger.info(f"訓練履歴をCSVに保存: {csv_path}")

        # グラフを生成
        self._generate_training_graphs(history, vis_dir)

    def _generate_training_graphs(
        self, history: dict[str, list[float]], output_dir: Path
    ) -> None:
        """訓練グラフを生成.

        Args:
            history: 訓練履歴の辞書.
            output_dir: 出力ディレクトリ.
        """
        epochs = range(1, len(history["train_loss"]) + 1)

        # 1. 損失グラフ
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Train Loss")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        plt.tight_layout()
        loss_path = output_dir / "loss.png"
        plt.savefig(loss_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._logger.info(f"損失グラフを保存: {loss_path}")

        # 2. mIoU/Diceグラフ (検証データがある場合)
        if history["val_miou"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                epochs,
                history["val_miou"],
                "g-",
                linewidth=2,
                label="Val mIoU",
            )
            ax.plot(
                epochs,
                history["val_dice"],
                "r-",
                linewidth=2,
                label="Val Dice",
            )
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title("Validation Metrics", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right")
            plt.tight_layout()
            metrics_path = output_dir / "metrics.png"
            plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            self._logger.info(f"メトリクスグラフを保存: {metrics_path}")

        # 3. 学習率グラフ
        if history["learning_rate"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                epochs,
                history["learning_rate"],
                "m-",
                linewidth=2,
                label="Learning Rate",
            )
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Learning Rate", fontsize=12)
            ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            plt.tight_layout()
            lr_path = output_dir / "learning_rate.png"
            plt.savefig(lr_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            self._logger.info(f"学習率グラフを保存: {lr_path}")

    @property
    def model(self) -> nn.Module:
        """モデルを取得.

        Returns:
            セグメンテーションモデル.
        """
        return self._model

    @property
    def best_miou(self) -> float:
        """ベストmIoUを取得.

        Returns:
            ベストmIoU値.
        """
        return self._best_miou

    @property
    def best_epoch(self) -> int:
        """ベストエポックを取得.

        Returns:
            ベストエポック番号.
        """
        return self._best_epoch
