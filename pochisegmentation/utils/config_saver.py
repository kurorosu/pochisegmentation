"""設定ファイル保存ユーティリティ.

対話モードで生成した設定を再利用可能な形式で保存する.
"""

from pathlib import Path

from .timestamp_utils import (
    find_next_index,
    format_workspace_name,
    get_current_date_str,
)

CONFIG_SAVE_BASE_DIR = Path("configs/work_dirs")


def save_config_for_reuse(config_content: str) -> Path:
    """設定ファイルを再利用用ディレクトリに保存.

    configs/work_dirs/yyyymmdd_{idx}/saved_config.py の形式で保存する.

    Args:
        config_content: 設定ファイルの内容 (Python形式の文字列).

    Returns:
        保存されたファイルのパス.
    """
    # ベースディレクトリが存在しない場合は作成
    CONFIG_SAVE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 今日の日付を取得
    date_str = get_current_date_str()

    # 次のインデックスを取得
    next_index = find_next_index(CONFIG_SAVE_BASE_DIR, date_str)

    # ワークスペース名を生成
    workspace_name = format_workspace_name(date_str, next_index)
    config_dir = CONFIG_SAVE_BASE_DIR / workspace_name
    config_dir.mkdir(parents=True, exist_ok=True)

    # 設定ファイルを保存
    config_path = config_dir / "saved_config.py"
    config_path.write_text(config_content, encoding="utf-8")

    return config_path
