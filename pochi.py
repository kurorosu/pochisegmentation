r"""pochisegmentation CLIエントリーポイント.

セグメンテーションモデルの訓練と推論を行うCLIツール.

Usage:
    # 対話型訓練 (推奨)
    python pochi.py train

    # 設定ファイル指定訓練
    python pochi.py train --config configs/pochi_seg_config.py

    # 推論
    python pochi.py infer --model-path work_dirs/xxx/models/best.pth \
        --data data/test/images --output results/
"""

import argparse
import sys

from pochisegmentation.cli import interactive_train, seg_infer, seg_train


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

    # infer サブコマンド
    infer_parser = subparsers.add_parser("infer", help="セグメンテーション推論")
    infer_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="モデルファイルのパス",
    )
    infer_parser.add_argument(
        "--data",
        type=str,
        required=True,
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
        seg_infer(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
