"""訓練コマンド.

pochi.py から呼び出される薄いアダプター層.
設定取得方法の分岐を行い, core 層を呼び出す.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from pochisegmentation.core import run_training
from pochisegmentation.exceptions import PochiConfigError
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.utils.config_loader import ConfigLoader


def train_command(
    args: argparse.Namespace,
    stop_flag_callback: Callable[[], bool] | None = None,
) -> None:
    """訓練コマンド.

    --config 必須. 対話モードは python pochi.py から.

    Args:
        args: コマンドライン引数.
        stop_flag_callback: 停止フラグをチェックするコールバック関数.
    """
    logger = LoggerManager().get_logger("pochiseg")

    if not args.config:
        logger.error("--config オプションが必要です")
        logger.error("対話モードを使用する場合は python pochi.py を実行してください")
        sys.exit(1)

    logger.info(f"設定ファイルを読み込み: {args.config}")
    try:
        config = ConfigLoader.load(args.config)
    except PochiConfigError as e:
        logger.error(f"設定エラー: {e}")
        sys.exit(1)

    run_training(
        config,
        config_path=Path(args.config),
        stop_flag_callback=stop_flag_callback,
    )
