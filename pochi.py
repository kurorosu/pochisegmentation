r"""pochisegmentation CLIエントリーポイント.

セグメンテーションモデルの訓練と推論を行うCLIツール.

Usage:
    # 対話型モード (訓練/推論を選択)
    python pochi.py

    # 設定ファイル指定訓練
    python pochi.py train --config configs/pochi_seg_config.py

    # 引数指定推論
    python pochi.py infer --model-path work_dirs/xxx/models/best.pth \
        --data data/test/images --output results/
"""

import signal

from pochisegmentation.cli.commands import (
    infer_command,
    interactive_main,
    train_command,
)
from pochisegmentation.cli.parser import parse_args
from pochisegmentation.logging.logger_manager import LoggerManager

# グローバル変数で訓練停止フラグを管理
training_interrupted = False


def signal_handler(signum: int, frame: object) -> None:
    """Ctrl+Cのシグナルハンドラー.

    Args:
        signum: シグナル番号.
        frame: スタックフレーム.
    """
    global training_interrupted
    training_interrupted = True

    logger = LoggerManager().get_logger("pochiseg")
    logger.warning("訓練を安全に停止しています... (Ctrl+Cが検出されました)")
    logger.warning("現在のエポックが完了次第、訓練を終了します。")


def main() -> None:
    """メインエントリーポイント."""
    args = parse_args()

    # Ctrl+Cの安全な処理を設定
    signal.signal(signal.SIGINT, signal_handler)

    # 停止フラグコールバック
    def get_stop_flag() -> bool:
        return training_interrupted

    if args.command == "train":
        train_command(args, stop_flag_callback=get_stop_flag)
    elif args.command == "infer":
        infer_command(args)
    else:
        # サブコマンドなし: 対話でモード選択
        interactive_main(stop_flag_callback=get_stop_flag)


if __name__ == "__main__":
    main()
