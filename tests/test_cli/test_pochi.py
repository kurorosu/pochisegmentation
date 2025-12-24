"""pochi.py CLIのテスト."""

import tempfile
from pathlib import Path

import pytest

from pochi import create_optimizer, create_scheduler, load_config


class TestLoadConfig:
    """load_configのテストクラス."""

    def test_load_config_success(self) -> None:
        """設定ファイルの読み込み成功テスト."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.py"
            config_path.write_text(
                """
architecture = "Unet"
encoder_name = "resnet34"
num_classes = 4
learning_rate = 0.001
"""
            )

            config = load_config(str(config_path))

            assert config["architecture"] == "Unet"
            assert config["encoder_name"] == "resnet34"
            assert config["num_classes"] == 4
            assert config["learning_rate"] == 0.001

    def test_load_config_not_found(self) -> None:
        """存在しない設定ファイルのテスト."""
        with pytest.raises(FileNotFoundError, match="設定ファイルが見つかりません"):
            load_config("nonexistent.py")


class TestCreateOptimizer:
    """create_optimizerのテストクラス."""

    def test_create_adam(self) -> None:
        """Adamオプティマイザの作成テスト."""
        import torch

        model = torch.nn.Linear(10, 10)
        config = {"optimizer": "Adam", "learning_rate": 0.001}

        optimizer = create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_adamw(self) -> None:
        """AdamWオプティマイザの作成テスト."""
        import torch

        model = torch.nn.Linear(10, 10)
        config = {"optimizer": "AdamW", "learning_rate": 0.001}

        optimizer = create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_sgd(self) -> None:
        """SGDオプティマイザの作成テスト."""
        import torch

        model = torch.nn.Linear(10, 10)
        config = {"optimizer": "SGD", "learning_rate": 0.001}

        optimizer = create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.SGD)

    def test_unknown_optimizer_raises(self) -> None:
        """未知のオプティマイザのテスト."""
        import torch

        model = torch.nn.Linear(10, 10)
        config = {"optimizer": "Unknown"}

        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(model, config)


class TestCreateScheduler:
    """create_schedulerのテストクラス."""

    def test_create_cosine_annealing(self) -> None:
        """CosineAnnealingLRスケジューラの作成テスト."""
        import torch

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        config = {"scheduler": "CosineAnnealingLR", "scheduler_params": {"T_max": 100}}

        scheduler = create_scheduler(optimizer, config)

        assert scheduler is not None

    def test_create_step_lr(self) -> None:
        """StepLRスケジューラの作成テスト."""
        import torch

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        config = {"scheduler": "StepLR", "scheduler_params": {"step_size": 10}}

        scheduler = create_scheduler(optimizer, config)

        assert scheduler is not None

    def test_no_scheduler(self) -> None:
        """スケジューラなしのテスト."""
        import torch

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        config: dict[str, str] = {}

        scheduler = create_scheduler(optimizer, config)

        assert scheduler is None

    def test_unknown_scheduler_raises(self) -> None:
        """未知のスケジューラのテスト."""
        import torch

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        config = {"scheduler": "Unknown"}

        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(optimizer, config)
