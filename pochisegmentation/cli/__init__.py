"""CLIコマンドパッケージ."""

from pochisegmentation.cli.infer import seg_infer
from pochisegmentation.cli.train import seg_train

__all__ = ["seg_train", "seg_infer"]
