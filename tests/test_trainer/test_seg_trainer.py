"""PochiSegmentationTrainer (ファサード) のテスト."""

import tempfile
from pathlib import Path

import torch

from pochisegmentation.seg_trainer import PochiSegmentationTrainer
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager

from ._helpers import MockLoss, MockMetrics, MockModel, create_dummy_dataloader


class TestPochiSegmentationTrainer:
    """PochiSegmentationTrainer のテストクラス."""

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
        assert "val_loss" in history
        assert len(history["val_miou"]) == 2
        assert len(history["val_loss"]) == 2

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

        # MockMetrics は常に mIoU=0.5 を返すので, best_miou は 0.5 になる
        assert trainer.best_miou == 0.5
        assert trainer.best_epoch == 0

    def test_early_stopping_monitor_val_loss(self) -> None:
        """val_loss を監視指標に指定すると best_miou が val_loss を保持する."""
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
            early_stopping_monitor="val_loss",
        )

        train_loader = create_dummy_dataloader()
        val_loader = create_dummy_dataloader()
        history = trainer.train(train_loader, val_loader=val_loader, epochs=1)

        # val_loss 監視では最小の val_loss がベスト値として保持される
        assert trainer.best_miou == history["val_loss"][0]

    def test_save_and_load_checkpoint(self) -> None:
        """チェックポイントの保存と読み込みテスト."""
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

            # 検証付きで訓練し best.pth を保存 (best_miou=0.5)
            train_loader = create_dummy_dataloader()
            val_loader = create_dummy_dataloader()
            trainer.train(train_loader, val_loader=val_loader, epochs=1)

            best_path = workspace_manager.get_models_dir() / "best.pth"
            assert best_path.exists()

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

            loaded = trainer2.load_checkpoint(best_path)

            assert loaded["epoch"] == 0
            assert loaded["best_miou"] == 0.5
            assert trainer2.best_miou == 0.5
            assert trainer2.best_epoch == 0

    def test_with_workspace_manager(self) -> None:
        """ワークスペースマネージャとの統合テスト."""
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

            train_loader = create_dummy_dataloader()
            val_loader = create_dummy_dataloader()
            trainer.train(train_loader, val_loader=val_loader, epochs=1)

            # ベストモデルが保存されているか確認
            models_dir = workspace_manager.get_models_dir()
            assert (models_dir / "best.pth").exists()
            assert (models_dir / "last.pth").exists()

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
        """model プロパティのテスト."""
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

    def test_amp_disabled_on_cpu(self) -> None:
        """CPU では AMP が自動無効化される."""
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
            enable_amp=True,
        )

        # CPU では AMP は無効化され, 訓練が正常に完了する
        assert trainer._enable_amp is False
        train_loader = create_dummy_dataloader()
        history = trainer.train(train_loader, epochs=1)
        assert len(history["train_loss"]) == 1


class TestStopFlag:
    """停止フラグによる安全停止のテスト."""

    def test_stop_before_first_epoch(self) -> None:
        """開始前に停止フラグが立つと訓練が即終了する."""
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
        history = trainer.train(train_loader, epochs=5, stop_flag_callback=lambda: True)

        # 1 エポックも実行されない
        assert len(history["train_loss"]) == 0

    def test_stop_after_first_epoch(self) -> None:
        """1エポック完了後に停止フラグが立つと終了する."""
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

        calls = {"count": 0}

        def stop_after_one() -> bool:
            # エポック開始前チェックは False, 完了後チェックで True を返す
            calls["count"] += 1
            return calls["count"] > 1

        train_loader = create_dummy_dataloader()
        history = trainer.train(
            train_loader, epochs=5, stop_flag_callback=stop_after_one
        )

        assert len(history["train_loss"]) == 1


class TestEarlyStoppingIntegration:
    """Early Stopping のファサード統合テスト."""

    def test_early_stopping_triggers(self) -> None:
        """改善しないと patience 経過で訓練が打ち切られる."""
        model = MockModel()
        criterion = MockLoss()
        # 常に同じ mIoU=0.5 を返すので, 2 エポック目以降は改善なし
        metrics = MockMetrics()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = PochiSegmentationTrainer(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            device="cpu",
            early_stopping_patience=1,
        )

        train_loader = create_dummy_dataloader()
        val_loader = create_dummy_dataloader()
        history = trainer.train(train_loader, val_loader=val_loader, epochs=10)

        # epoch0: best 更新, epoch1: 改善なし counter=1>=1 で停止
        assert len(history["val_miou"]) == 2
