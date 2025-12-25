"""CLIコマンドパッケージ."""

from pochisegmentation.cli.commands import (
    infer_command,
    interactive_main,
    train_command,
)

__all__ = ["train_command", "infer_command", "interactive_main"]
