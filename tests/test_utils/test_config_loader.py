"""ConfigLoaderのテスト."""

from pathlib import Path

import pytest

from pochisegmentation.config import PochiSegConfig
from pochisegmentation.exceptions import (
    ConfigFileNotFoundError,
    ConfigValidationError,
)
from pochisegmentation.utils.config_loader import ConfigLoader


class TestConfigLoader:
    """ConfigLoaderクラスのテスト."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """正常な設定ファイルの読み込みテスト."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
architecture = "Unet"
batch_size = 16
"""
        )

        config = ConfigLoader.load(str(config_path))

        assert isinstance(config, PochiSegConfig)
        assert config.data_root == "data/train"
        assert config.num_classes == 4
        assert config.architecture == "Unet"
        assert config.batch_size == 16

    def test_load_applies_defaults(self, tmp_path: Path) -> None:
        """デフォルト値が適用されるテスト."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
"""
        )

        config = ConfigLoader.load(str(config_path))

        # デフォルト値が適用される
        assert config.architecture == "Unet"
        assert config.encoder_name == "resnet34"
        assert config.batch_size == 16
        assert config.epochs == 100
        assert config.learning_rate == 1e-3

    def test_load_file_not_found(self) -> None:
        """存在しないファイルを読み込むとConfigFileNotFoundErrorが発生."""
        with pytest.raises(ConfigFileNotFoundError) as exc_info:
            ConfigLoader.load("nonexistent_config.py")

        assert "nonexistent_config.py" in str(exc_info.value)

    def test_validate_required_missing_keys(self, tmp_path: Path) -> None:
        """必須キーがないとConfigValidationErrorが発生."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
# data_root と num_classes がない
architecture = "Unet"
"""
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(str(config_path))

        message = str(exc_info.value)
        assert "data_root" in message
        assert "num_classes" in message

    def test_validate_required_partial_missing(self, tmp_path: Path) -> None:
        """必須キーの一部がないとエラー."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
# num_classes がない
"""
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(str(config_path))

        assert "num_classes" in str(exc_info.value)

    def test_validate_values_invalid_architecture(self, tmp_path: Path) -> None:
        """不正なarchitecture値でConfigValidationErrorが発生."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
architecture = "unet"  # 小文字は不正
"""
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(str(config_path))

        assert "architecture" in str(exc_info.value)

    def test_validate_values_invalid_optimizer(self, tmp_path: Path) -> None:
        """不正なoptimizer値でConfigValidationErrorが発生."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
optimizer = "RMSprop"  # 未対応
"""
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(str(config_path))

        assert "optimizer" in str(exc_info.value)

    def test_validate_values_invalid_loss(self, tmp_path: Path) -> None:
        """不正なloss値でConfigValidationErrorが発生."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
loss = "CrossEntropyLoss"  # 未対応
"""
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(str(config_path))

        assert "loss" in str(exc_info.value)

    def test_validate_values_invalid_device(self, tmp_path: Path) -> None:
        """不正なdevice値でConfigValidationErrorが発生."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
device = "tpu"  # 未対応
"""
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(str(config_path))

        assert "device" in str(exc_info.value)

    def test_validate_types_invalid_num_classes(self, tmp_path: Path) -> None:
        """num_classesが文字列だとConfigValidationErrorが発生."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = "abc"  # 数値化できない文字列は不正
"""
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(str(config_path))

        assert "num_classes" in str(exc_info.value)

    def test_validate_num_classes_must_be_positive(self, tmp_path: Path) -> None:
        """num_classesが0以下だとConfigValidationErrorが発生."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 0  # gt=0 制約に違反
"""
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(str(config_path))

        assert "num_classes" in str(exc_info.value)

    def test_validate_types_learning_rate_int_allowed(self, tmp_path: Path) -> None:
        """learning_rateはintでもfloatに変換され許可される."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
learning_rate = 1  # intでもOK (floatに変換)
"""
        )

        config = ConfigLoader.load(str(config_path))
        assert config.learning_rate == 1.0

    def test_validate_types_learning_rate_float(self, tmp_path: Path) -> None:
        """learning_rateはfloatで正常."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
learning_rate = 0.001
"""
        )

        config = ConfigLoader.load(str(config_path))
        assert config.learning_rate == 0.001

    def test_scheduler_none_allowed(self, tmp_path: Path) -> None:
        """schedulerはNoneでも許可される."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
scheduler = None
"""
        )

        config = ConfigLoader.load(str(config_path))
        assert config.scheduler is None

    def test_scheduler_requires_params(self, tmp_path: Path) -> None:
        """schedulerに必須paramsがないとConfigValidationErrorが発生."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
scheduler = "CosineAnnealingLR"
scheduler_params = {}  # T_max がない
"""
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(str(config_path))

        assert "T_max" in str(exc_info.value)

    def test_deeplabv3plus_architecture(self, tmp_path: Path) -> None:
        """DeepLabV3Plusアーキテクチャが正常に読み込める."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
architecture = "DeepLabV3Plus"
"""
        )

        config = ConfigLoader.load(str(config_path))
        assert config.architecture == "DeepLabV3Plus"

    def test_user_config_overrides_defaults(self, tmp_path: Path) -> None:
        """ユーザー設定がデフォルト値を上書きする."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 10
batch_size = 32
epochs = 200
learning_rate = 0.0001
"""
        )

        config = ConfigLoader.load(str(config_path))

        assert config.num_classes == 10
        assert config.batch_size == 32
        assert config.epochs == 200
        assert config.learning_rate == 0.0001
        # デフォルト値は維持
        assert config.architecture == "Unet"

    def test_layer_wise_lr_config(self, tmp_path: Path) -> None:
        """層別学習率設定が正常に読み込める."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
enable_layer_wise_lr = True
encoder_lr = 0.0001
decoder_lr = 0.001
"""
        )

        config = ConfigLoader.load(str(config_path))

        assert config.enable_layer_wise_lr is True
        assert config.encoder_lr == 0.0001
        assert config.decoder_lr == 0.001

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        """未知のキーは無視される."""
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
unknown_field = "ignored"
"""
        )

        config = ConfigLoader.load(str(config_path))
        assert not hasattr(config, "unknown_field")
