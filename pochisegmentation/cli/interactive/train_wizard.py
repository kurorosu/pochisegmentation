"""訓練設定の対話型ウィザード."""

from dataclasses import dataclass
from typing import Any, cast

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from configs.presets import (
    ARCHITECTURES,
    DATA_ROOTS,
    ENCODERS,
    LOSSES,
    OPTIMIZERS,
    SCHEDULERS,
    TRAINING_PRESETS,
    Choice,
    TrainingPreset,
)
from pochisegmentation.utils.config_saver import save_config_for_reuse


def get_scheduler_params(scheduler: str | None, epochs: int) -> dict[str, Any]:
    """スケジューラごとに適切なデフォルトパラメータを取得.

    Args:
        scheduler: スケジューラ名.
        epochs: エポック数.

    Returns:
        スケジューラのパラメータ辞書.
    """
    if scheduler == "CosineAnnealingLR":
        return {"T_max": epochs}
    elif scheduler == "StepLR":
        return {"step_size": max(1, epochs // 3)}
    elif scheduler == "ReduceLROnPlateau":
        return {"mode": "max", "patience": 5}
    else:
        return {}


@dataclass
class TrainConfig:
    """対話で収集した訓練設定."""

    data_root: str
    num_classes: int
    architecture: str
    encoder_name: str
    loss: str
    optimizer: str
    scheduler: str | None
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int
    # デフォルト値を持つフィールド
    pretrained: bool = True
    train_split: str = "train"
    val_split: str = "val"
    num_workers: int = 4
    device: str = "cuda"
    work_dir: str = "work_dirs"
    enable_layer_wise_lr: bool = True
    early_stopping_patience: int | None = None
    enable_amp: bool = False

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換."""
        return {
            "data_root": self.data_root,
            "num_classes": self.num_classes,
            "architecture": self.architecture,
            "encoder_name": self.encoder_name,
            "loss": self.loss,
            "loss_params": {"mode": "multiclass"},
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scheduler_params": get_scheduler_params(self.scheduler, self.epochs),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "image_size": self.image_size,
            "pretrained": self.pretrained,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "num_workers": self.num_workers,
            "device": self.device,
            "work_dir": self.work_dir,
            "enable_layer_wise_lr": self.enable_layer_wise_lr,
            "encoder_lr": self.learning_rate * 0.1,
            "decoder_lr": self.learning_rate,
            "early_stopping_patience": self.early_stopping_patience,
            "enable_amp": self.enable_amp,
        }

    def to_python_config(self) -> str:
        """Python 設定ファイル形式の文字列を生成."""
        lines = [
            '"""自動生成されたセグメンテーション訓練設定."""',
            "",
            "# モデル設定",
            f'architecture = "{self.architecture}"',
            f'encoder_name = "{self.encoder_name}"',
            f"pretrained = {self.pretrained}",
            f"num_classes = {self.num_classes}",
            "",
            "# データ設定",
            f'data_root = "{self.data_root}"',
            f'train_split = "{self.train_split}"',
            f'val_split = "{self.val_split}"',
            f"image_size = {self.image_size}",
            f"batch_size = {self.batch_size}",
            f"num_workers = {self.num_workers}",
            "",
            "# 訓練設定",
            f"epochs = {self.epochs}",
            f"learning_rate = {self.learning_rate}",
            f'optimizer = "{self.optimizer}"',
            "",
            "# 損失関数",
            f'loss = "{self.loss}"',
            'loss_params = {"mode": "multiclass"}',
            "",
            "# スケジューラ",
        ]

        if self.scheduler:
            lines.append(f'scheduler = "{self.scheduler}"')
            scheduler_params = get_scheduler_params(self.scheduler, self.epochs)
            lines.append(f"scheduler_params = {scheduler_params}")
        else:
            lines.append("scheduler = None")
            lines.append("scheduler_params = {}")

        lines.extend(
            [
                "",
                "# 層別学習率",
                f"enable_layer_wise_lr = {self.enable_layer_wise_lr}",
                f"encoder_lr = {self.learning_rate * 0.1}",
                f"decoder_lr = {self.learning_rate}",
                "",
                "# Early Stopping",
                f"early_stopping_patience = {self.early_stopping_patience}",
                "",
                "# AMP (Automatic Mixed Precision)",
                f"enable_amp = {self.enable_amp}",
                "",
                "# ワークスペース設定",
                f'work_dir = "{self.work_dir}"',
                f'device = "{self.device}"',
                "",
            ]
        )

        return "\n".join(lines)


# カスタム選択を表すセンチネル値
_CUSTOM_PRESET = object()


class TrainWizard:
    """訓練設定の対話型ウィザード."""

    def __init__(self) -> None:
        """初期化."""
        self.console = Console()

    def run(self) -> TrainConfig | None:
        """対話を実行して設定を収集.

        Returns:
            収集した設定. キャンセル時は None.
        """
        self._show_header()

        # Step 1: データセット
        data_root = self._ask_data_root()
        if data_root is None:
            return None

        num_classes = self._ask_num_classes()
        if num_classes is None:
            return None

        # Step 2: モデル
        architecture = self._ask_choice("アーキテクチャ", ARCHITECTURES)
        if architecture is None:
            return None

        encoder = self._ask_choice("エンコーダー", ENCODERS, default="resnet34")
        if encoder is None:
            return None

        loss = self._ask_choice("損失関数", LOSSES, default="DiceLoss")
        if loss is None:
            return None

        # Step 3: 訓練設定
        preset_result = self._ask_training_preset()
        if preset_result is None:
            # キャンセル
            return None

        if preset_result is _CUSTOM_PRESET:
            # カスタム設定
            epochs = self._ask_int("エポック数", default=50)
            if epochs is None:
                return None
            batch_size = self._ask_int("バッチサイズ", default=16)
            if batch_size is None:
                return None
            learning_rate = self._ask_float("学習率", default=1e-3)
            if learning_rate is None:
                return None
            image_size = self._ask_int("画像サイズ", default=256)
            if image_size is None:
                return None
        else:
            preset = cast(TrainingPreset, preset_result)
            epochs = preset.epochs
            batch_size = preset.batch_size
            learning_rate = preset.learning_rate
            image_size = preset.image_size

        # Step 4: 高度な設定
        optimizer = self._ask_choice("オプティマイザ", OPTIMIZERS, default="AdamW")
        if optimizer is None:
            return None

        scheduler = self._ask_scheduler()

        # Early Stopping (ReduceLROnPlateau 以外で推奨)
        early_stopping_patience = self._ask_early_stopping(scheduler)

        # AMP (混合精度訓練)
        enable_amp = self._ask_amp()

        config = TrainConfig(
            data_root=data_root,
            num_classes=num_classes,
            architecture=architecture,
            encoder_name=encoder,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            image_size=image_size,
            early_stopping_patience=early_stopping_patience,
            enable_amp=enable_amp,
        )

        # 確認
        action = self._confirm(config)
        if action == "start":
            return config
        elif action == "save":
            self._save_config(config)
            return None
        else:
            return None

    def _show_header(self) -> None:
        """ヘッダー表示."""
        self.console.print()
        self.console.print(
            Panel(
                "対話型モードで訓練を設定します\n[dim]Ctrl+C でいつでもキャンセルできます[/dim]",
                title="PochiSegmentation Training",
                border_style="blue",
            )
        )
        self.console.print()

    def _ask_choice(
        self,
        message: str,
        choices: list[Choice],
        default: str | None = None,
    ) -> str | None:
        """選択肢から選ぶ.

        Args:
            message: 質問メッセージ.
            choices: 選択肢リスト.
            default: デフォルト値.

        Returns:
            選択された値. キャンセル時は None.
        """
        options = [
            questionary.Choice(
                title=f"{c.label} - {c.description}" if c.description else c.label,
                value=c.value,
            )
            for c in choices
        ]
        result: str | None = questionary.select(
            message,
            choices=options,
            default=default,
        ).ask()
        return result

    def _ask_data_root(self) -> str | None:
        """データセットパスを質問.

        Returns:
            データセットパス. キャンセル時は None.
        """
        # 選択肢 + 手動入力オプション
        options = [
            questionary.Choice(
                title=f"{c.label} - {c.description}" if c.description else c.label,
                value=c.value,
            )
            for c in DATA_ROOTS
        ]
        options.append(questionary.Choice(title="[手動入力]", value="__manual__"))

        result: str | None = questionary.select(
            "データセット",
            choices=options,
        ).ask()

        if result is None:
            return None

        if result == "__manual__":
            manual_result: str | None = questionary.text(
                "データセットのパスを入力",
                default="data/",
            ).ask()
            return manual_result

        return result

    def _ask_num_classes(self) -> int | None:
        """クラス数を質問.

        Returns:
            クラス数. キャンセル時は None.
        """
        result: str | None = questionary.text(
            "クラス数 (背景含む)",
            default="2",
            validate=lambda x: x.isdigit() and int(x) >= 2,
        ).ask()

        if result is None:
            return None

        return int(result)

    def _ask_int(self, message: str, default: int) -> int | None:
        """整数入力.

        Args:
            message: 質問メッセージ.
            default: デフォルト値.

        Returns:
            入力値. キャンセル時は None.
        """
        result: str | None = questionary.text(
            message,
            default=str(default),
            validate=lambda x: x.isdigit() and int(x) > 0,
        ).ask()

        if result is None:
            return None

        return int(result)

    def _ask_float(self, message: str, default: float) -> float | None:
        """浮動小数点入力.

        Args:
            message: 質問メッセージ.
            default: デフォルト値.

        Returns:
            入力値. キャンセル時は None.
        """
        result: str | None = questionary.text(
            message,
            default=str(default),
            validate=lambda x: self._is_valid_float(x),
        ).ask()

        if result is None:
            return None

        return float(result)

    def _is_valid_float(self, value: str) -> bool:
        """浮動小数点としてパース可能か確認."""
        try:
            f = float(value)
            return f > 0
        except ValueError:
            return False

    def _ask_training_preset(self) -> TrainingPreset | object | None:
        """訓練プリセットを質問.

        Returns:
            選択されたプリセット, カスタム時は _CUSTOM_PRESET, キャンセル時は None.
        """
        options = [
            questionary.Choice(
                title=f"{p.label} - {p.description}",
                value=p,
            )
            for p in TRAINING_PRESETS
        ]
        options.append(
            questionary.Choice(title="Custom - 詳細設定", value=_CUSTOM_PRESET)
        )

        result: TrainingPreset | object | None = questionary.select(
            "訓練プリセット",
            choices=options,
            default=options[1],  # balanced
        ).ask()

        return result

    def _ask_scheduler(self) -> str | None:
        """スケジューラを質問.

        Returns:
            スケジューラ名, または None (スケジューラなし/キャンセル).
        """
        options = [
            questionary.Choice(
                title=f"{c.label} - {c.description}",
                value=c.value,
            )
            for c in SCHEDULERS
        ]
        options.append(questionary.Choice(title="None - スケジューラなし", value=""))

        result: str | None = questionary.select(
            "スケジューラ",
            choices=options,
            default="CosineAnnealingLR",
        ).ask()

        # 空文字列は None に変換
        if result == "":
            return None

        return result

    def _ask_early_stopping(self, scheduler: str | None) -> int | None:
        """Early Stopping の patience を質問.

        ReduceLROnPlateau の場合は推奨せず, それ以外は推奨.

        Args:
            scheduler: 選択されたスケジューラ.

        Returns:
            patience 値, または None (無効).
        """
        # ReduceLROnPlateau は停滞時にLRを減衰させるので, Early Stopping は不要な場合が多い
        if scheduler == "ReduceLROnPlateau":
            description = "停滞時にLR減衰があるため, 通常は不要"
            default_choice = "none"
        else:
            description = "改善がなければ訓練を終了 - 推奨"
            default_choice = "10"

        options = [
            questionary.Choice(
                title=(
                    f"10 エポック - {description}"
                    if default_choice == "10"
                    else "10 エポック"
                ),
                value="10",
            ),
            questionary.Choice(
                title="5 エポック",
                value="5",
            ),
            questionary.Choice(
                title="20 エポック",
                value="20",
            ),
            questionary.Choice(
                title=(
                    f"None - 無効 ({description})"
                    if default_choice == "none"
                    else "None - 無効"
                ),
                value="none",
            ),
        ]

        result: str | None = questionary.select(
            "Early Stopping (patience)",
            choices=options,
            default=default_choice,
        ).ask()

        if result is None or result == "none":
            return None

        return int(result)

    def _ask_amp(self) -> bool:
        """AMP (混合精度訓練) を質問.

        Returns:
            AMP を有効化するかどうか.
        """
        options = [
            questionary.Choice(
                title="無効 (通常精度)",
                value=False,
            ),
            questionary.Choice(
                title="有効 (メモリ削減・高速化, CUDA専用)",
                value=True,
            ),
        ]

        result: bool | None = questionary.select(
            "AMP (混合精度訓練)",
            choices=options,
            default=options[0],  # 無効
        ).ask()

        return result if result is not None else False

    def _confirm(self, config: TrainConfig) -> str | None:
        """設定確認画面を表示.

        Args:
            config: 確認する設定.

        Returns:
            "start" (訓練開始), "save" (保存), None (キャンセル).
        """
        # 設定サマリーを表示
        self.console.print()
        table = Table(title="設定確認", show_header=False, border_style="blue")
        table.add_column("項目", style="cyan")
        table.add_column("値", style="green")

        table.add_row("データセット", config.data_root)
        table.add_row("クラス数", str(config.num_classes))
        table.add_row(
            "アーキテクチャ", f"{config.architecture} + {config.encoder_name}"
        )
        table.add_row("損失関数", config.loss)
        table.add_row("オプティマイザ", config.optimizer)
        table.add_row("スケジューラ", config.scheduler or "None")
        table.add_row("エポック数", str(config.epochs))
        table.add_row("バッチサイズ", str(config.batch_size))
        table.add_row("学習率", str(config.learning_rate))
        table.add_row("画像サイズ", str(config.image_size))
        early_stopping_str = (
            f"{config.early_stopping_patience} エポック"
            if config.early_stopping_patience
            else "無効"
        )
        table.add_row("Early Stopping", early_stopping_str)
        table.add_row("AMP", "有効" if config.enable_amp else "無効")

        self.console.print(table)
        self.console.print()

        # 確認
        options = [
            questionary.Choice(title="訓練を開始", value="start"),
            questionary.Choice(title="設定を保存して終了", value="save"),
            questionary.Choice(title="キャンセル", value="cancel"),
        ]

        result: str | None = questionary.select(
            "この設定で続行しますか?",
            choices=options,
        ).ask()

        if result == "cancel" or result is None:
            return None

        return result

    def _save_config(self, config: TrainConfig) -> None:
        """設定を Python ファイルに保存.

        Args:
            config: 保存する設定.
        """
        content = config.to_python_config()
        saved_path = save_config_for_reuse(content)

        self.console.print(f"[green]設定を保存しました: {saved_path}[/green]")
        self.console.print(
            f"[cyan]再利用: python pochi.py train --config {saved_path}[/cyan]"
        )
