"""CLI コマンドパッケージ.

サブコマンドごとのコマンド関数 (train / infer / interactive) を提供する.
"""

from pochisegmentation.cli.commands.infer import infer_command
from pochisegmentation.cli.commands.interactive import interactive_main
from pochisegmentation.cli.commands.train import train_command

__all__ = ["train_command", "infer_command", "interactive_main"]
