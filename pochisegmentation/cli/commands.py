"""CLIコマンド定義.

pochi.py から呼び出される薄いアダプター層.
設定取得方法の分岐を行い、core 層を呼び出す.
"""

import argparse
import sys
from pathlib import Path

from pochisegmentation.core import run_inference, run_training
from pochisegmentation.exceptions import PochiConfigError
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.utils.config_loader import ConfigLoader


def train_command(args: argparse.Namespace) -> None:
    """訓練コマンド.

    --config 必須. 対話モードは python pochi.py から.

    Args:
        args: コマンドライン引数.
    """
    logger = LoggerManager().get_logger("pochi")

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

    run_training(config, config_path=Path(args.config))


def infer_command(args: argparse.Namespace) -> None:
    """推論コマンド.

    --model-path と --data 必須. 対話モードは python pochi.py から.

    Args:
        args: コマンドライン引数.
    """
    logger = LoggerManager().get_logger("pochi")

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


def interactive_main() -> None:
    """対話モードのエントリーポイント.

    サブコマンドなしで起動された場合に呼び出される.
    訓練/推論の選択を行い、対応する対話フローを実行.
    """
    from pochisegmentation.cli.interactive.infer_wizard import InferWizard
    from pochisegmentation.cli.interactive.mode_selector import select_mode
    from pochisegmentation.cli.interactive.train_wizard import TrainWizard

    logger = LoggerManager().get_logger("pochi")

    mode = select_mode()

    if mode == "train":
        wizard = TrainWizard()
        result = wizard.run()

        if result is None:
            logger.info("訓練をキャンセルしました")
            sys.exit(0)

        config = result.to_dict()
        config_content = result.to_python_config()
        run_training(config, config_content=config_content)

    elif mode == "infer":
        infer_wizard = InferWizard()
        infer_result = infer_wizard.run()

        if infer_result is None:
            logger.info("推論をキャンセルしました")
            sys.exit(0)

        run_inference(
            model_path=Path(infer_result.model_path),
            data_path=Path(infer_result.data_path),
            output_dir=(
                Path(infer_result.output_dir) if infer_result.output_dir else None
            ),
            device=infer_result.device,
        )

    else:
        # キャンセル
        sys.exit(0)
