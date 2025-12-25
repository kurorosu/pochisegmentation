"""ConfigLoaderのテスト."""

import tempfile
from pathlib import Path

import pytest

from pochisegmentation.exceptions import (
    ConfigFileNotFoundError,
    ConfigKeyMissingError,
    ConfigTypeError,
    ConfigValueError,
)
from pochisegmentation.utils.config_loader import ConfigLoader


class TestConfigLoader:
    """ConfigLoaderクラスのテスト."""

    def test_load_valid_config(self) -> None:
        """正常な設定ファイルの読み込みテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
architecture = "Unet"
batch_size = 16
"""
            )

            config = ConfigLoader.load(str(config_path))

            assert config["data_root"] == "data/train"
            assert config["num_classes"] == 4
            assert config["architecture"] == "Unet"
            assert config["batch_size"] == 16

    def test_load_applies_defaults(self) -> None:
        """デフォルト値が適用されるテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
"""
            )

            config = ConfigLoader.load(str(config_path))

            # デフォルト値が適用される
            assert config["architecture"] == "Unet"
            assert config["encoder_name"] == "resnet34"
            assert config["batch_size"] == 16
            assert config["epochs"] == 100
            assert config["learning_rate"] == 1e-3

    def test_load_file_not_found(self) -> None:
        """存在しないファイルを読み込むとConfigFileNotFoundErrorが発生."""
        with pytest.raises(ConfigFileNotFoundError) as exc_info:
            ConfigLoader.load("nonexistent_config.py")

        assert "nonexistent_config.py" in str(exc_info.value)

    def test_validate_required_missing_keys(self) -> None:
        """必須キーがないとConfigKeyMissingErrorが発生."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
# data_root と num_classes がない
architecture = "Unet"
"""
            )

            with pytest.raises(ConfigKeyMissingError) as exc_info:
                ConfigLoader.load(str(config_path))

            assert "data_root" in exc_info.value.missing_keys
            assert "num_classes" in exc_info.value.missing_keys

    def test_validate_required_partial_missing(self) -> None:
        """必須キーの一部がないとエラー."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
# num_classes がない
"""
            )

            with pytest.raises(ConfigKeyMissingError) as exc_info:
                ConfigLoader.load(str(config_path))

            assert "num_classes" in exc_info.value.missing_keys
            assert "data_root" not in exc_info.value.missing_keys

    def test_validate_values_invalid_architecture(self) -> None:
        """不正なarchitecture値でConfigValueErrorが発生."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
architecture = "unet"  # 小文字は不正
"""
            )

            with pytest.raises(ConfigValueError) as exc_info:
                ConfigLoader.load(str(config_path))

            assert exc_info.value.key == "architecture"
            assert exc_info.value.value == "unet"
            assert "Unet" in exc_info.value.allowed_values
            assert "DeepLabV3Plus" in exc_info.value.allowed_values

    def test_validate_values_invalid_optimizer(self) -> None:
        """不正なoptimizer値でConfigValueErrorが発生."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
optimizer = "RMSprop"  # 未対応
"""
            )

            with pytest.raises(ConfigValueError) as exc_info:
                ConfigLoader.load(str(config_path))

            assert exc_info.value.key == "optimizer"
            assert exc_info.value.value == "RMSprop"

    def test_validate_values_invalid_loss(self) -> None:
        """不正なloss値でConfigValueErrorが発生."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
loss = "CrossEntropyLoss"  # 未対応
"""
            )

            with pytest.raises(ConfigValueError) as exc_info:
                ConfigLoader.load(str(config_path))

            assert exc_info.value.key == "loss"

    def test_validate_values_invalid_device(self) -> None:
        """不正なdevice値でConfigValueErrorが発生."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
device = "tpu"  # 未対応
"""
            )

            with pytest.raises(ConfigValueError) as exc_info:
                ConfigLoader.load(str(config_path))

            assert exc_info.value.key == "device"

    def test_validate_types_invalid_num_classes(self) -> None:
        """num_classesが文字列だとConfigTypeErrorが発生."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = "4"  # 文字列は不正
"""
            )

            with pytest.raises(ConfigTypeError) as exc_info:
                ConfigLoader.load(str(config_path))

            assert exc_info.value.key == "num_classes"
            assert exc_info.value.expected_type == int
            assert exc_info.value.actual_type == str

    def test_validate_types_invalid_batch_size(self) -> None:
        """batch_sizeが文字列だとConfigTypeErrorが発生."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
batch_size = "16"  # 文字列は不正
"""
            )

            with pytest.raises(ConfigTypeError) as exc_info:
                ConfigLoader.load(str(config_path))

            assert exc_info.value.key == "batch_size"

    def test_validate_types_invalid_pretrained(self) -> None:
        """pretrainedが文字列だとConfigTypeErrorが発生."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
pretrained = "True"  # 文字列は不正
"""
            )

            with pytest.raises(ConfigTypeError) as exc_info:
                ConfigLoader.load(str(config_path))

            assert exc_info.value.key == "pretrained"
            assert exc_info.value.expected_type == bool

    def test_validate_types_learning_rate_int_allowed(self) -> None:
        """learning_rateはintでもfloatでも許可される."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
learning_rate = 1  # intでもOK
"""
            )

            config = ConfigLoader.load(str(config_path))
            assert config["learning_rate"] == 1

    def test_validate_types_learning_rate_float(self) -> None:
        """learning_rateはfloatで正常."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
learning_rate = 0.001
"""
            )

            config = ConfigLoader.load(str(config_path))
            assert config["learning_rate"] == 0.001

    def test_scheduler_none_allowed(self) -> None:
        """schedulerはNoneでも許可される."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
scheduler = None
"""
            )

            config = ConfigLoader.load(str(config_path))
            assert config["scheduler"] is None

    def test_deeplabv3plus_architecture(self) -> None:
        """DeepLabV3Plusアーキテクチャが正常に読み込める."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text(
                """
data_root = "data/train"
num_classes = 4
architecture = "DeepLabV3Plus"
"""
            )

            config = ConfigLoader.load(str(config_path))
            assert config["architecture"] == "DeepLabV3Plus"

    def test_user_config_overrides_defaults(self) -> None:
        """ユーザー設定がデフォルト値を上書きする."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
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

            assert config["num_classes"] == 10
            assert config["batch_size"] == 32
            assert config["epochs"] == 200
            assert config["learning_rate"] == 0.0001
            # デフォルト値は維持
            assert config["architecture"] == "Unet"

    def test_layer_wise_lr_config(self) -> None:
        """層別学習率設定が正常に読み込める."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.py"
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

            assert config["enable_layer_wise_lr"] is True
            assert config["encoder_lr"] == 0.0001
            assert config["decoder_lr"] == 0.001
