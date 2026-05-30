"""設定ファイルの読み込みとバリデーション.

Python 設定ファイルを読み込み, Pydantic ベースの PochiSegConfig として
型付け・バリデーションする. 不正な設定は即座にエラーとする.
"""

import importlib.util
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pochisegmentation.config import PochiSegConfig
from pochisegmentation.exceptions import (
    ConfigFileNotFoundError,
    ConfigValidationError,
)

__all__ = ["ConfigLoader"]


class ConfigLoader:
    """設定ファイルの読み込みとバリデーション.

    Python 設定ファイルを読み込み, PochiSegConfig に変換する.
    Pydantic によるバリデーションでサイレントフォールバックを排除する.

    Raises:
        ConfigFileNotFoundError: 設定ファイルが見つからない場合.
        ConfigValidationError: 設定値のバリデーションに失敗した場合.
    """

    @classmethod
    def load(cls, config_path: str | Path) -> PochiSegConfig:
        """設定を読み込み, バリデーション済みの PochiSegConfig を返す.

        Args:
            config_path: 設定ファイルのパス.

        Returns:
            バリデーション済みの PochiSegConfig.

        Raises:
            ConfigFileNotFoundError: 設定ファイルが見つからない場合.
            ConfigValidationError: 設定値のバリデーションに失敗した場合.
        """
        raw_config = cls._load_file(config_path)
        try:
            return PochiSegConfig.from_dict(raw_config)
        except ValidationError as e:
            raise ConfigValidationError(str(config_path), e) from e

    @classmethod
    def _load_file(cls, config_path: str | Path) -> dict[str, Any]:
        """Python 設定ファイルを読み込み dict として返す.

        Args:
            config_path: 設定ファイルのパス.

        Returns:
            設定 dict (モジュールの公開属性).

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

        # モジュールの公開属性を dict として取得
        config: dict[str, Any] = {}
        for key in dir(module):
            if not key.startswith("_"):
                config[key] = getattr(module, key)

        return config
