"""対話型推論の設定収集."""

from dataclasses import dataclass
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def find_model_files(base_dir: str = "work_dirs") -> list[Path]:
    """work_dirs からモデルファイルを検索.

    Args:
        base_dir: 検索ベースディレクトリ.

    Returns:
        見つかったモデルファイルのリスト.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # best.pth と last.pth を検索
    model_files: list[Path] = []
    for pth_file in base_path.glob("*/models/*.pth"):
        model_files.append(pth_file)

    # 更新日時でソート (新しい順)
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return model_files


def count_images(data_path: Path) -> int:
    """画像ファイル数をカウント.

    Args:
        data_path: 画像ファイル, ディレクトリ, またはパスリスト.

    Returns:
        画像ファイル数.
    """
    if data_path.is_file():
        if data_path.suffix == ".txt":
            # パスリストファイル
            with open(data_path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        else:
            # 単一画像ファイル
            return 1
    elif data_path.is_dir():
        # ディレクトリ内の画像をカウント
        count = 0
        for ext in ["*.jpg", "*.png", "*.bmp", "*.jpeg"]:
            count += len(list(data_path.glob(ext)))
        return count
    return 0


@dataclass
class InferenceConfig:
    """対話で収集した推論設定."""

    model_path: str
    data_path: str
    output_dir: str
    device: str

    def get_image_count(self) -> int:
        """入力画像数を取得."""
        return count_images(Path(self.data_path))


class InferenceRunner:
    """対話型推論設定収集."""

    def __init__(self) -> None:
        """初期化."""
        self.console = Console()

    def run(self) -> InferenceConfig | None:
        """対話を実行して推論設定を収集.

        Returns:
            収集した設定. キャンセル時は None.
        """
        self._show_header()

        # Step 1: モデルファイル選択
        model_path = self._ask_model_path()
        if model_path is None:
            return None

        # Step 2: 入力データ選択
        data_path = self._ask_data_path()
        if data_path is None:
            return None

        # Step 3: 出力ディレクトリ
        output_dir = self._ask_output_dir()
        if output_dir is None:
            return None

        # Step 4: デバイス選択
        device = self._ask_device()
        if device is None:
            return None

        config = InferenceConfig(
            model_path=model_path,
            data_path=data_path,
            output_dir=output_dir,
            device=device,
        )

        # 確認
        if self._confirm(config):
            return config
        return None

    def _show_header(self) -> None:
        """ヘッダー表示."""
        self.console.print()
        self.console.print(
            Panel(
                "対話型モードで推論を設定します",
                title="PochiSegmentation Inference",
                border_style="green",
            )
        )
        self.console.print()

    def _ask_model_path(self) -> str | None:
        """モデルファイルを質問.

        Returns:
            モデルファイルパス. キャンセル時は None.
        """
        # work_dirs からモデルファイルを検索
        model_files = find_model_files()

        if model_files:
            # 選択肢を作成
            options = [
                questionary.Choice(
                    title=str(p),
                    value=str(p),
                )
                for p in model_files[:10]  # 最大10件表示
            ]
            options.append(questionary.Choice(title="[手動入力]", value="__manual__"))

            result: str | None = questionary.select(
                "モデルファイル",
                choices=options,
            ).ask()

            if result is None:
                return None

            if result == "__manual__":
                return self._ask_manual_path("モデルファイルのパス")

            return result
        else:
            # モデルファイルが見つからない場合は手動入力
            self.console.print(
                "[yellow]work_dirs にモデルファイルが見つかりません[/yellow]"
            )
            return self._ask_manual_path("モデルファイルのパス")

    def _ask_data_path(self) -> str | None:
        """入力データを質問.

        Returns:
            入力データパス. キャンセル時は None.
        """
        options = [
            questionary.Choice(
                title="ディレクトリを指定",
                value="__dir__",
            ),
            questionary.Choice(
                title="単一ファイルを指定",
                value="__file__",
            ),
            questionary.Choice(
                title="パスリスト (.txt) を指定",
                value="__txt__",
            ),
        ]

        result: str | None = questionary.select(
            "入力データの種類",
            choices=options,
        ).ask()

        if result is None:
            return None

        if result == "__dir__":
            return self._ask_manual_path("画像ディレクトリのパス", default="data/test")
        elif result == "__file__":
            return self._ask_manual_path("画像ファイルのパス")
        else:  # __txt__
            return self._ask_manual_path("パスリストファイル (.txt) のパス")

    def _ask_output_dir(self) -> str | None:
        """出力ディレクトリを質問.

        Returns:
            出力ディレクトリパス (空文字列でデフォルト). キャンセル時は None.
        """
        result: str | None = questionary.text(
            "出力ディレクトリ (空欄でデフォルト)",
            default="",
        ).ask()

        return result

    def _ask_device(self) -> str | None:
        """デバイスを質問.

        Returns:
            デバイス名. キャンセル時は None.
        """
        options = [
            questionary.Choice(title="cuda - GPU を使用", value="cuda"),
            questionary.Choice(title="cpu - CPU を使用", value="cpu"),
        ]

        result: str | None = questionary.select(
            "デバイス",
            choices=options,
            default="cuda",
        ).ask()

        return result

    def _ask_manual_path(self, message: str, default: str = "") -> str | None:
        """パスを手動入力.

        Args:
            message: 質問メッセージ.
            default: デフォルト値.

        Returns:
            入力されたパス. キャンセル時は None.
        """
        result: str | None = questionary.text(
            message,
            default=default,
        ).ask()

        return result

    def _confirm(self, config: InferenceConfig) -> bool:
        """設定確認画面を表示.

        Args:
            config: 確認する設定.

        Returns:
            確認された場合 True.
        """
        # 設定サマリーを表示
        self.console.print()
        table = Table(title="設定確認", show_header=False, border_style="green")
        table.add_column("項目", style="cyan")
        table.add_column("値", style="green")

        table.add_row("モデル", config.model_path)
        table.add_row("入力", config.data_path)
        table.add_row("画像数", f"{config.get_image_count()} 枚")
        table.add_row("出力", config.output_dir or "(デフォルト)")
        table.add_row("デバイス", config.device)

        self.console.print(table)
        self.console.print()

        # 確認
        options = [
            questionary.Choice(title="推論を開始", value="start"),
            questionary.Choice(title="キャンセル", value="cancel"),
        ]

        result: str | None = questionary.select(
            "この設定で続行しますか?",
            choices=options,
        ).ask()

        return result == "start"
