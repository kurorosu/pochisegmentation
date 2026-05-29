"""cli/pochi.py CLIのテスト."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from pochisegmentation.config import PochiSegConfig
from pochisegmentation.exceptions import ConfigFileNotFoundError
from pochisegmentation.factories import ComponentFactory
from pochisegmentation.utils.config_loader import ConfigLoader


def _make_config(**overrides: Any) -> PochiSegConfig:
    """テスト用 PochiSegConfig を生成する."""
    params: dict[str, Any] = {"data_root": "data", "num_classes": 4}
    params.update(overrides)
    return PochiSegConfig(**params)


class TestLoadConfig:
    """ConfigLoaderのテストクラス (CLI 用)."""

    def test_load_config_success(self) -> None:
        """設定ファイルの読み込み成功テスト."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
architecture = "Unet"
encoder_name = "resnet34"
num_classes = 4
learning_rate = 0.001
"""
            )

            config = ConfigLoader.load(str(config_path))

            assert config.architecture == "Unet"
            assert config.encoder_name == "resnet34"
            assert config.num_classes == 4
            assert config.learning_rate == 0.001

    def test_load_config_not_found(self) -> None:
        """存在しない設定ファイルのテスト."""
        with pytest.raises(ConfigFileNotFoundError):
            ConfigLoader.load("nonexistent.py")


class TestCreateOptimizer:
    """create_optimizerのテストクラス."""

    def test_create_adam(self) -> None:
        """Adamオプティマイザの作成テスト."""
        model = torch.nn.Linear(10, 10)
        config = _make_config(optimizer="Adam", learning_rate=0.001)

        optimizer = ComponentFactory.create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_adamw(self) -> None:
        """AdamWオプティマイザの作成テスト."""
        model = torch.nn.Linear(10, 10)
        config = _make_config(optimizer="AdamW", learning_rate=0.001)

        optimizer = ComponentFactory.create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_sgd(self) -> None:
        """SGDオプティマイザの作成テスト."""
        model = torch.nn.Linear(10, 10)
        config = _make_config(optimizer="SGD", learning_rate=0.001)

        optimizer = ComponentFactory.create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.SGD)

    def test_unknown_optimizer_rejected_at_config(self) -> None:
        """未知のオプティマイザはconfig構築時に弾かれる."""
        with pytest.raises(ValidationError):
            _make_config(optimizer="Unknown")


class TestCreateScheduler:
    """create_schedulerのテストクラス."""

    def test_create_cosine_annealing(self) -> None:
        """CosineAnnealingLRスケジューラの作成テスト."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        config = _make_config(
            scheduler="CosineAnnealingLR", scheduler_params={"T_max": 100}
        )

        scheduler = ComponentFactory.create_scheduler(optimizer, config)

        assert scheduler is not None

    def test_create_step_lr(self) -> None:
        """StepLRスケジューラの作成テスト."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        config = _make_config(scheduler="StepLR", scheduler_params={"step_size": 10})

        scheduler = ComponentFactory.create_scheduler(optimizer, config)

        assert scheduler is not None

    def test_no_scheduler(self) -> None:
        """スケジューラなしのテスト."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        config = _make_config(scheduler=None)

        scheduler = ComponentFactory.create_scheduler(optimizer, config)

        assert scheduler is None

    def test_unknown_scheduler_rejected_at_config(self) -> None:
        """未知のスケジューラはconfig構築時に弾かれる."""
        with pytest.raises(ValidationError):
            _make_config(scheduler="Unknown")
