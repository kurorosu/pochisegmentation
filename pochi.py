r"""pochisegmentation CLIエントリーポイント.

セグメンテーションモデルの訓練と推論を行うCLIツール.

Usage:
    # 対話型モード (訓練/推論を選択)
    python pochi.py

    # 対話型訓練
    python pochi.py train

    # 設定ファイル指定訓練
    python pochi.py train --config configs/pochi_seg_config.py

    # 対話型推論
    python pochi.py infer

    # 引数指定推論
    python pochi.py infer --model-path work_dirs/xxx/models/best.pth \
        --data data/test/images --output results/
"""

import argparse
import sys

import questionary
from rich.console import Console
from rich.panel import Panel

from pochisegmentation.cli import (
    interactive_infer,
    interactive_train,
    seg_infer,
    seg_train,
)


def interactive_mode_select() -> str | None:
    """対話モードで訓練/推論を選択.

    Returns:
        "train" または "infer". キャンセル時は None.
    """
    console = Console()
    console.print()
    console.print(
        Panel(
            "セグメンテーションツールへようこそ",
            title="PochiSegmentation",
            border_style="blue",
        )
    )
    console.print()

    options = [
        questionary.Choice(
            title="訓練 (Train) - モデルを訓練する",
            value="train",
        ),
        questionary.Choice(
            title="推論 (Infer) - 学習済みモデルで推論する",
            value="infer",
        ),
    ]

    result: str | None = questionary.select(
        "実行モードを選択",
        choices=options,
    ).ask()

    return result


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
        if args.config:
            # 設定ファイル指定モード
            seg_train(args)
        else:
            # 対話モード
            interactive_train()
    elif args.command == "infer":
        if args.model_path and args.data:
            # 引数指定モード
            seg_infer(args)
        else:
            # 対話モード
            interactive_infer()
    elif args.command is None:
        # サブコマンドなし: 対話でモード選択
        mode = interactive_mode_select()
        if mode == "train":
            interactive_train()
        elif mode == "infer":
            interactive_infer()
        else:
            sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
