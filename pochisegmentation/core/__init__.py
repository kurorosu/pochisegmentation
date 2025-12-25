"""アプリケーションコア.

訓練・推論のオーケストレーションロジックを提供.
CLIやAPIから呼び出される.
"""

from pochisegmentation.core.inference import run_inference
from pochisegmentation.core.training import run_training

__all__ = ["run_training", "run_inference"]
