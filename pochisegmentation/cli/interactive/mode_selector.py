"""モード選択.

サブコマンドなしで起動された場合に訓練/推論を選択する.
"""

import questionary
from rich.console import Console
from rich.panel import Panel


def select_mode() -> str | None:
    """訓練/推論モードを選択.

    Returns:
        "train", "infer", または None (キャンセル時).
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
        questionary.Choice(
            title="終了",
            value="exit",
        ),
    ]

    result: str | None = questionary.select(
        "実行モードを選択",
        choices=options,
    ).ask()

    # 終了選択時は None を返す (キャンセルと同じ扱い)
    if result == "exit":
        return None

    return result
