"""pochisegmentation のカスタム例外.

サイレントフォールバックを防止し, 不正な設定を即座にエラーとする.
"""

from typing import Any


class PochiConfigError(Exception):
    """設定ファイル関連のエラー基底クラス."""

    pass


class ConfigFileNotFoundError(PochiConfigError):
    """設定ファイルが見つからない場合のエラー."""

    def __init__(self, config_path: str) -> None:
        """初期化.

        Args:
            config_path (str): 見つからなかった設定ファイルのパス.
        """
        self.config_path = config_path
        super().__init__(f"設定ファイルが見つかりません: {config_path}")


class ConfigKeyMissingError(PochiConfigError):
    """必須キーが設定ファイルにない場合のエラー."""

    def __init__(self, missing_keys: list[str]) -> None:
        """初期化.

        Args:
            missing_keys (list[str]): 不足している必須キーのリスト.
        """
        self.missing_keys = missing_keys
        super().__init__(f"設定ファイルに必須項目がありません: {missing_keys}")


class ConfigValueError(PochiConfigError):
    """設定値が不正な場合のエラー."""

    def __init__(
        self, key: str, value: Any, allowed_values: list[Any] | None = None
    ) -> None:
        """初期化.

        Args:
            key (str): 設定キー.
            value (Any): 不正な設定値.
            allowed_values (list[Any] | None): 許可される値のリスト. Defaults to None.
        """
        self.key = key
        self.value = value
        self.allowed_values = allowed_values
        if allowed_values:
            msg = (
                f"設定 '{key}' の値 '{value}' は無効です. "
                f"許可される値: {allowed_values}"
            )
        else:
            msg = f"設定 '{key}' の値 '{value}' は無効です."
        super().__init__(msg)


class ConfigTypeError(PochiConfigError):
    """設定値の型が不正な場合のエラー."""

    def __init__(self, key: str, expected_type: type, actual_type: type):
        """初期化.

        Args:
            key (str): 設定キー.
            expected_type (type): 期待される型.
            actual_type (type): 実際の型.
        """
        self.key = key
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(
            f"設定 '{key}' の型が不正です. "
            f"期待: {expected_type.__name__}, 実際: {actual_type.__name__}"
        )
