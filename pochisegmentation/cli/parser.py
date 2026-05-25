"""CLI 引数パーサー定義.

_create_parser() でサブコマンドを構築し, parse_args() で引数をパースする.
"""

import argparse


def _create_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを構築する.

    Returns:
        構築した ArgumentParser.
    """
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

    return parser


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース.

    Returns:
        パースされた引数.
    """
    return _create_parser().parse_args()
