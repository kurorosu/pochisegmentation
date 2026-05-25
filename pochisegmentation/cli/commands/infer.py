"""推論コマンド.

pochi.py から呼び出される薄いアダプター層.
設定取得方法の分岐を行い, core 層を呼び出す.
"""

import argparse
import sys
from pathlib import Path

from pochisegmentation.core import run_inference
from pochisegmentation.logging.logger_manager import LoggerManager


def infer_command(args: argparse.Namespace) -> None:
    """推論コマンド.

    --model-path と --data 必須. 対話モードは python pochi.py から.

    Args:
        args: コマンドライン引数.
    """
    logger = LoggerManager().get_logger("pochiseg")

    if not args.model_path or not args.data:
        logger.error("--model-path と --data オプションが必要です")
        logger.error("対話モードを使用する場合は python pochi.py を実行してください")
        sys.exit(1)

    run_inference(
        model_path=Path(args.model_path),
        data_path=Path(args.data),
        output_dir=Path(args.output) if args.output else None,
        device=args.device,
    )
