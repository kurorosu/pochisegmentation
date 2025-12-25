"""対話型CLI."""

from pochisegmentation.cli.interactive.infer_wizard import InferConfig, InferWizard
from pochisegmentation.cli.interactive.mode_selector import select_mode
from pochisegmentation.cli.interactive.train_wizard import TrainConfig, TrainWizard

__all__ = [
    "TrainWizard",
    "TrainConfig",
    "InferWizard",
    "InferConfig",
    "select_mode",
]
