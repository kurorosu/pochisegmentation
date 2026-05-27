r"""pochisegmentation CLI エントリーポイント.

セグメンテーションモデルの訓練と推論を行う CLI ツール.
[project.scripts] 経由で `uv run pochi` として呼び出される.

Usage:
    # 対話型モード (訓練/推論を選択)
    uv run pochi

    # 設定ファイル指定訓練
    uv run pochi train --config configs/pochi_seg_config.py

    # 引数指定推論
    uv run pochi infer --model-path work_dirs/xxx/models/best.pth \
        --data data/test/images --output results/
"""

import signal

from pochisegmentation.cli.parser import parse_args
from pochisegmentation.logging.logger_manager import LoggerManager

# グローバル変数で訓練停止フラグを管理
training_interrupted = False


def signal_handler(signum: int, frame: object) -> None:
    """Ctrl+C のシグナルハンドラー.

    Args:
        signum: シグナル番号.
        frame: スタックフレーム.
    """
    global training_interrupted
    training_interrupted = True

    logger = LoggerManager().get_logger("pochiseg")
    logger.warning("訓練を安全に停止しています... (Ctrl+C が検出されました)")
    logger.warning("現在のエポックが完了次第, 訓練を終了します.")


def main() -> None:
    """メインエントリーポイント.

    サブコマンドごとにコマンド関数を関数内で遅延 import する.
    将来 serve / mcp など optional 依存を持つコマンドを追加した際に,
    無関係なコマンドの import 失敗で起動不能にならないようにするため.
    """
    args = parse_args()

    # Ctrl+C の安全な処理を設定
    signal.signal(signal.SIGINT, signal_handler)

    # 停止フラグコールバック
    def get_stop_flag() -> bool:
        return training_interrupted

    if args.command == "train":
        from pochisegmentation.cli.commands.train import train_command

        train_command(args, stop_flag_callback=get_stop_flag)
    elif args.command == "infer":
        from pochisegmentation.cli.commands.infer import infer_command

        infer_command(args)
    else:
        # サブコマンドなし: 対話でモード選択
        from pochisegmentation.cli.commands.interactive import interactive_main

        interactive_main(stop_flag_callback=get_stop_flag)


if __name__ == "__main__":
    main()
