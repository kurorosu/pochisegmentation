"""PochiSegmentationTrainerのテスト."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.interfaces.model import ISegmentationModel
from pochisegmentation.seg_trainer import PochiSegmentationTrainer
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager


class MockModel(ISegmentationModel):
    """テスト用モックモデル."""

    def __init__(self, num_classes: int = 4) -> None:
        """MockModelを初期化."""
        super().__init__()
        self.num_classes = num_classes
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播."""
        return self.conv(x)

    def get_encoder_params(self) -> list:
        """エンコーダーパラメータを取得."""
        return []

    def get_decoder_params(self) -> list:
        """デコーダーパラメータを取得."""
        return list(self.conv.parameters())


class MockLoss(ISegmentationLoss):
    """テスト用モック損失関数."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失を計算."""
        return torch.nn.functional.cross_entropy(pred, target)


class MockMetrics(ISegmentationMetrics):
    """テスト用モック評価指標."""

    def __init__(self) -> None:
        """MockMetricsを初期化."""
        self._count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """バッチ結果を蓄積."""
        self._count += 1

    def compute(self) -> dict[str, float]:
        """指標を計算."""
        return {"mIoU": 0.5, "Dice": 0.6}

    def reset(self) -> None:
        """リセット."""
        self._count = 0


def create_dummy_dataloader(
    batch_size: int = 2, num_samples: int = 8
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """ダミーデータローダーを作成."""
    images = torch.randn(num_samples, 3, 32, 32)
    masks = torch.randint(0, 4, (num_samples, 32, 32))
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=batch_size)


class TestPochiSegmentationTrainer:
    """PochiSegmentationTrainerのテストクラス."""

    def test_init(self) -> None:
        """初期化テスト."""
        model = MockModel()
        criterion = MockLoss()
        metrics = MockMetrics()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = PochiSegmentationTrainer(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            device="cpu",
        )

        assert trainer.model is not None
        assert trainer.best_miou == 0.0
        assert trainer.best_epoch == 0

    def test_train_single_epoch(self) -> None:
        """1エポックの訓練テスト."""
        model = MockModel()
        criterion = MockLoss()
        metrics = MockMetrics()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = PochiSegmentationTrainer(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            device="cpu",
        )

        train_loader = create_dummy_dataloader()
        history = trainer.train(train_loader, epochs=1)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0

    def test_train_with_validation(self) -> None:
        """検証付き訓練テスト."""
        model = MockModel()
        criterion = MockLoss()
        metrics = MockMetrics()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = PochiSegmentationTrainer(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            device="cpu",
        )

        train_loader = create_dummy_dataloader()
        val_loader = create_dummy_dataloader()
        history = trainer.train(train_loader, val_loader=val_loader, epochs=2)

        assert "train_loss" in history
        assert "val_miou" in history
        assert "val_dice" in history
        assert len(history["val_miou"]) == 2

    def test_train_with_scheduler(self) -> None:
        """スケジューラ付き訓練テスト."""
        model = MockModel()
        criterion = MockLoss()
        metrics = MockMetrics()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        trainer = PochiSegmentationTrainer(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
        )

        train_loader = create_dummy_dataloader()
        history = trainer.train(train_loader, epochs=2)

        assert len(history["train_loss"]) == 2

    def test_best_model_tracking(self) -> None:
        """ベストモデル追跡テスト."""
        model = MockModel()
        criterion = MockLoss()
        metrics = MockMetrics()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = PochiSegmentationTrainer(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            device="cpu",
        )

        train_loader = create_dummy_dataloader()
        val_loader = create_dummy_dataloader()
        trainer.train(train_loader, val_loader=val_loader, epochs=1)

        # MockMetricsは常にmIoU=0.5を返すので, best_miouは0.5になる
        assert trainer.best_miou == 0.5
        assert trainer.best_epoch == 0

    def test_save_and_load_checkpoint(self) -> None:
        """チェックポイントの保存と読み込みテスト."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            criterion = MockLoss()
            metrics = MockMetrics()
            optimizer = torch.optim.Adam(model.parameters())

            trainer = PochiSegmentationTrainer(
                model=model,
                criterion=criterion,
                metrics=metrics,
                optimizer=optimizer,
                device="cpu",
            )

            # 訓練してチェックポイント保存
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            trainer._best_miou = 0.75
            trainer._best_epoch = 5
            trainer._save_checkpoint(checkpoint_path, 5, {"mIoU": 0.75})

            # 新しいトレーナーでチェックポイント読み込み
            model2 = MockModel()
            optimizer2 = torch.optim.Adam(model2.parameters())

            trainer2 = PochiSegmentationTrainer(
                model=model2,
                criterion=criterion,
                metrics=metrics,
                optimizer=optimizer2,
                device="cpu",
            )

            loaded = trainer2.load_checkpoint(checkpoint_path)

            assert loaded["epoch"] == 5
            assert loaded["best_miou"] == 0.75
            assert trainer2.best_miou == 0.75
            assert trainer2.best_epoch == 5

    def test_with_workspace_manager(self) -> None:
        """ワークスペースマネージャとの統合テスト."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_manager = PochiWorkspaceManager(base_dir=tmpdir)
            workspace_manager.create_workspace()

            model = MockModel()
            criterion = MockLoss()
            # mIoUを高くしてベストモデル保存をトリガー
            metrics = MockMetrics()
            optimizer = torch.optim.Adam(model.parameters())

            trainer = PochiSegmentationTrainer(
                model=model,
                criterion=criterion,
                metrics=metrics,
                optimizer=optimizer,
                device="cpu",
                workspace_manager=workspace_manager,
            )

            train_loader = create_dummy_dataloader()
            val_loader = create_dummy_dataloader()
            trainer.train(train_loader, val_loader=val_loader, epochs=1)

            # ベストモデルが保存されているか確認
            models_dir = workspace_manager.get_models_dir()
            assert (models_dir / "best.pth").exists()

    def test_save_last_model(self) -> None:
        """最終モデル保存テスト."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_manager = PochiWorkspaceManager(base_dir=tmpdir)
            workspace_manager.create_workspace()

            model = MockModel()
            criterion = MockLoss()
            metrics = MockMetrics()
            optimizer = torch.optim.Adam(model.parameters())

            trainer = PochiSegmentationTrainer(
                model=model,
                criterion=criterion,
                metrics=metrics,
                optimizer=optimizer,
                device="cpu",
                workspace_manager=workspace_manager,
            )

            path = trainer.save_last_model()
            assert path is not None
            assert path.exists()
            assert path.name == "last.pth"

    def test_save_last_model_without_workspace(self) -> None:
        """ワークスペースなしでの最終モデル保存テスト."""
        model = MockModel()
        criterion = MockLoss()
        metrics = MockMetrics()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = PochiSegmentationTrainer(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            device="cpu",
        )

        path = trainer.save_last_model()
        assert path is None

    def test_property_model(self) -> None:
        """modelプロパティのテスト."""
        model = MockModel()
        criterion = MockLoss()
        metrics = MockMetrics()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = PochiSegmentationTrainer(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            device="cpu",
        )

        assert trainer.model is model
