"""CLIコマンドパッケージ."""

from pochisegmentation.cli.infer import seg_infer
from pochisegmentation.cli.interactive_train import interactive_train
from pochisegmentation.cli.train import seg_train

__all__ = ["seg_train", "seg_infer", "interactive_train"]
