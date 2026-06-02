"""CheckpointStore のユニットテスト."""

import logging
import tempfile

import torch

from pochisegmentation.training.checkpoint_store import CheckpointStore
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager

from ._helpers import MockModel


def _make_store(
    workspace_manager: PochiWorkspaceManager | None = None,
    monitor: str = "mIoU",
) -> CheckpointStore:
    """テスト用 CheckpointStore を生成."""
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters())
    return CheckpointStore(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        logger=logging.getLogger("test"),
        workspace_manager=workspace_manager,
        monitor=monitor,
    )


class TestUpdateBest:
    """update_best のテスト."""

    def test_first_update_is_improvement(self) -> None:
        """初回更新は改善とみなされる (mIoU > 0.0)."""
        store = _make_store()
        assert store.update_best({"mIoU": 0.5}, 0) is True
        assert store.best_value == 0.5
        assert store.best_epoch == 0

    def test_no_improvement_returns_false(self) -> None:
        """同値以下は改善ではない (strict >)."""
        store = _make_store()
        store.update_best({"mIoU": 0.5}, 0)
        assert store.update_best({"mIoU": 0.5}, 1) is False
        assert store.best_epoch == 0

    def test_val_loss_monitor_lower_is_better(self) -> None:
        """val_loss 監視では小さい方が改善."""
        store = _make_store(monitor="val_loss")
        assert store.update_best({"val_loss": 1.0}, 0) is True
        assert store.update_best({"val_loss": 0.5}, 1) is True
        assert store.best_value == 0.5
        assert store.update_best({"val_loss": 0.8}, 2) is False

    def test_best_pth_saved_with_workspace(self) -> None:
        """ワークスペースありで改善時に best.pth が保存される."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wm = PochiWorkspaceManager(base_dir=tmpdir)
            wm.create_workspace()
            store = _make_store(workspace_manager=wm)

            store.update_best({"mIoU": 0.5}, 0)
            assert (wm.get_models_dir() / "best.pth").exists()


class TestSaveLastAndLoad:
    """save_last と load のテスト."""

    def test_save_last_without_workspace_returns_none(self) -> None:
        """ワークスペースなしでは None を返す."""
        store = _make_store()
        assert store.save_last() is None

    def test_save_and_load_roundtrip(self) -> None:
        """保存したチェックポイントを読み込みベスト値を復元する."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wm = PochiWorkspaceManager(base_dir=tmpdir)
            wm.create_workspace()
            store = _make_store(workspace_manager=wm)

            store.update_best({"mIoU": 0.75}, 3)
            path = store.save_last()
            assert path is not None

            loaded = store.load(path)
            assert loaded["best_miou"] == 0.75
            assert store.best_value == 0.75
            assert store.best_epoch == 3
