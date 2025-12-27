"""推論設定の対話型ウィザード."""

from dataclasses import dataclass
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class ModelInfo:
    """モデルファイルと関連情報."""

    model_path: Path
    workspace_path: Path
    val_paths_file: Path | None  # paths/val.txt
    train_paths_file: Path | None  # paths/train.txt


def find_model_files(base_dir: str = "work_dirs") -> list[ModelInfo]:
    """work_dirs からモデルファイルと関連データパスを検索.

    Args:
        base_dir: 検索ベースディレクトリ.

    Returns:
        見つかったモデル情報のリスト.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # best.pth と last.pth を検索
    model_infos: list[ModelInfo] = []
    for pth_file in base_path.glob("*/models/*.pth"):
        # ワークスペースパス (work_dirs/EXP_NAME)
        workspace_path = pth_file.parent.parent

        # 関連データパスの検索
        val_txt = workspace_path / "paths" / "val.txt"
        train_txt = workspace_path / "paths" / "train.txt"

        model_infos.append(
            ModelInfo(
                model_path=pth_file,
                workspace_path=workspace_path,
                val_paths_file=val_txt if val_txt.exists() else None,
                train_paths_file=train_txt if train_txt.exists() else None,
            )
        )

    # 更新日時でソート (新しい順)
    model_infos.sort(key=lambda m: m.model_path.stat().st_mtime, reverse=True)
    return model_infos


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
class InferConfig:
    """対話で収集した推論設定."""

    model_path: str
    data_path: str
    output_dir: str
    device: str

    def get_image_count(self) -> int:
        """入力画像数を取得."""
        return count_images(Path(self.data_path))


class InferWizard:
    """推論設定の対話型ウィザード."""

    def __init__(self) -> None:
        """初期化."""
        self.console = Console()

    def run(self) -> InferConfig | None:
        """対話を実行して推論設定を収集.

        Returns:
            収集した設定. キャンセル時は None.
        """
        self._show_header()

        # Step 1: モデルファイル選択
        model_info = self._ask_model_path()
        if model_info is None:
            return None

        # Step 2: 入力データ選択
        data_path = self._ask_data_path(model_info)
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

        config = InferConfig(
            model_path=str(model_info.model_path),
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
                "対話型モードで推論を設定します\n[dim]Ctrl+C でいつでもキャンセルできます[/dim]",
                title="PochiSegmentation Inference",
                border_style="green",
            )
        )
        self.console.print()

    def _ask_model_path(self) -> ModelInfo | None:
        """モデルファイルを質問.

        Returns:
            モデル情報. キャンセル時は None.
        """
        # work_dirs からモデルファイルを検索
        model_infos = find_model_files()

        if model_infos:
            # 選択肢を作成
            options = [
                questionary.Choice(
                    title=str(info.model_path),
                    value=info,
                )
                for info in model_infos[:10]  # 最大10件表示
            ]
            options.append(questionary.Choice(title="[手動入力]", value="__manual__"))

            result: ModelInfo | str | None = questionary.select(
                "モデルファイル",
                choices=options,
            ).ask()

            if result is None:
                return None

            if result == "__manual__":
                path_str = self._ask_manual_path("モデルファイルのパス")
                if path_str is None:
                    return None
                return ModelInfo(
                    model_path=Path(path_str),
                    workspace_path=Path(path_str).parent.parent,
                    val_paths_file=None,
                    train_paths_file=None,
                )

            # result は ModelInfo (選択肢から選ばれた場合)
            if isinstance(result, ModelInfo):
                return result
            return None
        else:
            # モデルファイルが見つからない場合は手動入力
            self.console.print(
                "[yellow]work_dirs にモデルファイルが見つかりません[/yellow]"
            )
            path_str = self._ask_manual_path("モデルファイルのパス")
            if path_str is None:
                return None
            return ModelInfo(
                model_path=Path(path_str),
                workspace_path=Path(path_str).parent.parent,
                val_paths_file=None,
                train_paths_file=None,
            )

    def _ask_data_path(self, model_info: ModelInfo) -> str | None:
        """入力データを質問.

        モデルに紐づくデータがあれば優先表示.

        Args:
            model_info: 選択されたモデル情報.

        Returns:
            入力データパス. キャンセル時は None.
        """
        options: list[questionary.Choice | questionary.Separator] = []

        # モデルに紐づくデータを先に表示
        if model_info.val_paths_file:
            count = count_images(model_info.val_paths_file)
            options.append(
                questionary.Choice(
                    title=f"検証データ (val.txt - {count}枚)",
                    value=str(model_info.val_paths_file),
                )
            )

        if model_info.train_paths_file:
            count = count_images(model_info.train_paths_file)
            options.append(
                questionary.Choice(
                    title=f"訓練データ (train.txt - {count}枚)",
                    value=str(model_info.train_paths_file),
                )
            )

        # セパレーター（紐づくデータがある場合のみ）
        if options:
            options.append(questionary.Separator())

        # 手動入力オプション
        options.extend(
            [
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
        )

        # デフォルト選択（val.txt がデフォルト）
        default_val = None
        if model_info.val_paths_file:
            default_val = str(model_info.val_paths_file)

        result: str | None = questionary.select(
            "入力データ",
            choices=options,
            default=default_val,
        ).ask()

        if result is None:
            return None

        if result == "__dir__":
            return self._ask_manual_path("画像ディレクトリのパス", default="data/test")
        elif result == "__file__":
            return self._ask_manual_path("画像ファイルのパス")
        elif result == "__txt__":
            return self._ask_manual_path("パスリストファイル (.txt) のパス")
        else:
            # val.txt/train.txt が選択された場合
            return result

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

    def _confirm(self, config: InferConfig) -> bool:
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
