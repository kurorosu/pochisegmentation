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

from pochisegmentation.cli.commands import (
    infer_command,
    interactive_main,
    train_command,
)


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

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command is None:
        # サブコマンドなし: 対話でモード選択
        interactive_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
