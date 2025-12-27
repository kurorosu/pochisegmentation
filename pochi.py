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

import argparse
import signal

from pochisegmentation.cli.commands import (
    infer_command,
    interactive_main,
    train_command,
)
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

    logger = LoggerManager().get_logger("pochi")
    logger.warning("訓練を安全に停止しています... (Ctrl+Cが検出されました)")
    logger.warning("現在のエポックが完了次第、訓練を終了します。")


def main() -> None:
    """メインエントリーポイント."""
    parser = argparse.ArgumentParser(
        description="pochisegmentation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")

    # train サブコマンド (対話型モードをデフォルトに)
    train_parser = subparsers.add_parser("train", help="セグメンテーション訓練")
    train_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="設定ファイルのパス (省略時は対話モード)",
    )

    # infer サブコマンド (対話型モードをデフォルトに)
    infer_parser = subparsers.add_parser("infer", help="セグメンテーション推論")
    infer_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="モデルファイルのパス (省略時は対話モード)",
    )
    infer_parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="入力画像, ディレクトリ, またはパスリスト(.txt)のパス",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        default="",
        help="出力ディレクトリのパス (省略時: work_dir/predictions/)",
    )
    infer_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="使用デバイス (default: cuda)",
    )

    args = parser.parse_args()

    # Ctrl+Cの安全な処理を設定
    signal.signal(signal.SIGINT, signal_handler)

    # 停止フラグコールバック
    def get_stop_flag() -> bool:
        return training_interrupted

    if args.command == "train":
        train_command(args, stop_flag_callback=get_stop_flag)
    elif args.command == "infer":
        infer_command(args)
    elif args.command is None:
        # サブコマンドなし: 対話でモード選択
        interactive_main(stop_flag_callback=get_stop_flag)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
