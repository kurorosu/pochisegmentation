"""設定ファイルの読み込みとバリデーション.

サイレントフォールバックを排除し, 不正な設定は即座にエラーとする.
"""

import importlib.util
from pathlib import Path
from typing import Any

from pochisegmentation.exceptions import (
    ConfigFileNotFoundError,
    ConfigKeyMissingError,
    ConfigTypeError,
    ConfigValueError,
)


class ConfigLoader:
    """設定ファイルの読み込みとバリデーション.

    サイレントフォールバックを排除し, 不正な設定は即座にエラーとする.

    Raises:
        ConfigFileNotFoundError: 設定ファイルが見つからない場合.
        ConfigKeyMissingError: 必須キーがない場合.
        ConfigValueError: 設定値が許可リストにない場合.
        ConfigTypeError: 設定値の型が不正な場合.
    """

    REQUIRED_KEYS: list[str] = ["data_root", "num_classes"]

    ALLOWED_VALUES: dict[str, list[Any]] = {
        "architecture": ["Unet", "DeepLabV3Plus"],
        "optimizer": ["Adam", "AdamW", "SGD"],
        "loss": ["DiceLoss", "FocalLoss", "JaccardLoss"],
        "scheduler": ["CosineAnnealingLR", "StepLR", None],
        "device": ["cuda", "cpu"],
    }

    TYPE_CHECKS: dict[str, type] = {
        "num_classes": int,
        "batch_size": int,
        "epochs": int,
        "num_workers": int,
        "learning_rate": (int, float),  # type: ignore[dict-item]
        "image_size": int,
        "pretrained": bool,
        "enable_layer_wise_lr": bool,
        "encoder_lr": (int, float),  # type: ignore[dict-item]
        "decoder_lr": (int, float),  # type: ignore[dict-item]
    }

    DEFAULTS: dict[str, Any] = {
        "architecture": "Unet",
        "encoder_name": "resnet34",
        "pretrained": True,
        "image_size": 256,
        "batch_size": 16,
        "num_workers": 4,
        "epochs": 100,
        "learning_rate": 1e-3,
        "optimizer": "AdamW",
        "loss": "DiceLoss",
        "device": "cuda",
        "work_dir": "work_dirs",
        "enable_layer_wise_lr": False,
    }

    @classmethod
    def load(cls, config_path: str | Path) -> dict[str, Any]:
        """設定を読み込み, バリデーションとデフォルト値適用.

        Args:
            config_path (str | Path): 設定ファイルのパス.

        Returns:
            dict[str, Any]: バリデーション済みの設定辞書.

        Raises:
            ConfigFileNotFoundError: 設定ファイルが見つからない場合.
            ConfigKeyMissingError: 必須キーがない場合.
            ConfigValueError: 設定値が許可リストにない場合.
            ConfigTypeError: 設定値の型が不正な場合.
        """
        config = cls._load_file(config_path)
        cls._validate_required(config)
        cls._validate_values(config)
        cls._validate_types(config)
        return cls._apply_defaults(config)

    @classmethod
    def _load_file(cls, config_path: str | Path) -> dict[str, Any]:
        """Python設定ファイルを読み込み辞書として返す.

        Args:
            config_path (str | Path): 設定ファイルのパス.

        Returns:
            dict[str, Any]: 設定辞書.

        Raises:
            ConfigFileNotFoundError: ファイルが見つからない場合.
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigFileNotFoundError(str(config_path))

        spec = importlib.util.spec_from_file_location("config", path)
        if spec is None or spec.loader is None:
            raise ConfigFileNotFoundError(str(config_path))

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # モジュールの公開属性を辞書として取得
        config: dict[str, Any] = {}
        for key in dir(module):
            if not key.startswith("_"):
                config[key] = getattr(module, key)

        return config

    @classmethod
    def _validate_required(cls, config: dict[str, Any]) -> None:
        """必須項目チェック.

        Args:
            config (dict[str, Any]): 設定辞書.

        Raises:
            ConfigKeyMissingError: 必須キーがない場合.
        """
        missing = [key for key in cls.REQUIRED_KEYS if key not in config]
        if missing:
            raise ConfigKeyMissingError(missing)

    @classmethod
    def _validate_values(cls, config: dict[str, Any]) -> None:
        """許可値チェック (サイレントフォールバック排除).

        Args:
            config (dict[str, Any]): 設定辞書.

        Raises:
            ConfigValueError: 設定値が許可リストにない場合.
        """
        for key, allowed in cls.ALLOWED_VALUES.items():
            if key in config and config[key] not in allowed:
                raise ConfigValueError(key, config[key], allowed)

    @classmethod
    def _validate_types(cls, config: dict[str, Any]) -> None:
        """型チェック.

        Args:
            config (dict[str, Any]): 設定辞書.

        Raises:
            ConfigTypeError: 設定値の型が不正な場合.
        """
        for key, expected_type in cls.TYPE_CHECKS.items():
            if key in config:
                value = config[key]
                # タプルの場合は複数の型を許可 (例: int, float)
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        raise ConfigTypeError(key, expected_type[0], type(value))
                else:
                    if not isinstance(value, expected_type):
                        raise ConfigTypeError(key, expected_type, type(value))

    @classmethod
    def _apply_defaults(cls, config: dict[str, Any]) -> dict[str, Any]:
        """デフォルト値を適用 (未指定項目のみ).

        Args:
            config (dict[str, Any]): 設定辞書.

        Returns:
            dict[str, Any]: デフォルト値が適用された設定辞書.
        """
        result = cls.DEFAULTS.copy()
        result.update(config)
        return result
