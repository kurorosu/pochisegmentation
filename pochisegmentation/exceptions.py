"""pochisegmentation のカスタム例外.

サイレントフォールバックを防止し, 不正な設定を即座にエラーとする.
"""

from pydantic import ValidationError

__all__ = [
    "PochiConfigError",
    "ConfigFileNotFoundError",
    "ConfigValidationError",
]


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


class ConfigValidationError(PochiConfigError):
    """設定値のバリデーションに失敗した場合のエラー.

    Pydantic の ValidationError をラップし, どの設定ファイルが原因かを示す.
    """

    def __init__(self, config_path: str, validation_error: ValidationError) -> None:
        """初期化.

        Args:
            config_path (str): バリデーションに失敗した設定ファイルのパス.
            validation_error (ValidationError): 元の Pydantic バリデーションエラー.
        """
        self.config_path = config_path
        self.validation_error = validation_error
        super().__init__(
            f"設定ファイルのバリデーションに失敗しました: {config_path}\n"
            f"{validation_error}"
        )
