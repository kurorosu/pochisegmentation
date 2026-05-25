"""対話モードのコマンド.

サブコマンドなしで起動された場合の対話フローを提供する.
訓練 / 推論の選択と, 設定方法の選択ウィザードを担当する.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import questionary
from rich.console import Console

from pochisegmentation.core import run_inference, run_training
from pochisegmentation.exceptions import PochiConfigError
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.utils.config_loader import ConfigLoader
from pochisegmentation.utils.config_saver import get_saved_configs, get_training_configs

# センチネル値
BACK_SENTINEL = object()
EXIT_SENTINEL = object()


def _select_train_mode() -> str | object | None:
    """訓練モードを選択.

    Returns:
        "new", "saved", "training", BACK_SENTINEL (戻る), EXIT_SENTINEL (終了), None (キャンセル).
    """
    saved_configs = get_saved_configs()
    training_configs = get_training_configs()

    if not saved_configs and not training_configs:
        # どちらもない場合は直接新規設定へ
        console = Console()
        console.print(
            "[yellow]保存済みの設定がありません。新規設定を開始します。[/yellow]"
        )
        return "new"

    options: list[questionary.Choice] = [
        questionary.Choice(title="← 戻る", value=BACK_SENTINEL),
        questionary.Choice(title="新規設定", value="new"),
    ]

    if saved_configs:
        options.append(
            questionary.Choice(title="保存済み設定 (configs/work_dirs/)", value="saved")
        )

    if training_configs:
        options.append(
            questionary.Choice(title="過去の訓練設定 (work_dirs/)", value="training")
        )

    options.append(questionary.Choice(title="終了", value=EXIT_SENTINEL))

    result: str | object | None = questionary.select(
        "設定方法を選択",
        choices=options,
    ).ask()

    return result


def _select_config_file(configs: list[Path], title: str) -> Path | object | None:
    """設定ファイルを選択.

    Args:
        configs: 設定ファイルのリスト.
        title: 選択プロンプトのタイトル.

    Returns:
        選択された設定ファイルのパス, BACK_SENTINEL (戻る), EXIT_SENTINEL (終了), None (キャンセル).
    """
    if not configs:
        return None

    options: list[questionary.Choice] = [
        questionary.Choice(title="← 戻る", value=BACK_SENTINEL),
    ]
    options.extend(
        questionary.Choice(
            title=str(config_path.parent.name),  # ディレクトリ名 (例: 20251226_001)
            value=config_path,
        )
        for config_path in configs
    )
    options.append(questionary.Choice(title="終了", value=EXIT_SENTINEL))

    result: Path | object | None = questionary.select(
        title,
        choices=options,
    ).ask()

    return result


def interactive_main(
    stop_flag_callback: Callable[[], bool] | None = None,
) -> None:
    """対話モードのエントリーポイント.

    サブコマンドなしで起動された場合に呼び出される.
    訓練/推論の選択を行い、対応する対話フローを実行.

    Args:
        stop_flag_callback: 停止フラグをチェックするコールバック関数.
    """
    from pochisegmentation.cli.interactive.infer_wizard import InferWizard
    from pochisegmentation.cli.interactive.mode_selector import select_mode
    from pochisegmentation.cli.interactive.train_wizard import TrainWizard

    logger = LoggerManager().get_logger("pochiseg")

    while True:
        mode = select_mode()

        if mode == "train":
            while True:
                train_mode = _select_train_mode()

                if train_mode is None or train_mode is EXIT_SENTINEL:
                    logger.info("終了します")
                    sys.exit(0)

                if train_mode is BACK_SENTINEL:
                    break  # モード選択に戻る

                if train_mode == "saved":
                    # 保存済み設定を使用
                    config_path = _select_config_file(
                        get_saved_configs(), "保存済み設定を選択"
                    )
                    if config_path is BACK_SENTINEL:
                        continue  # 訓練モード選択に戻る
                    if config_path is None or config_path is EXIT_SENTINEL:
                        logger.info("終了します")
                        sys.exit(0)

                    config_path = cast(Path, config_path)
                    logger.info(f"設定ファイルを読み込み: {config_path}")
                    try:
                        config = ConfigLoader.load(str(config_path))
                    except PochiConfigError as e:
                        logger.error(f"設定エラー: {e}")
                        sys.exit(1)

                    run_training(
                        config,
                        config_path=config_path,
                        stop_flag_callback=stop_flag_callback,
                    )
                    return

                elif train_mode == "training":
                    # 過去の訓練設定を使用
                    config_path = _select_config_file(
                        get_training_configs(), "過去の訓練設定を選択"
                    )
                    if config_path is BACK_SENTINEL:
                        continue  # 訓練モード選択に戻る
                    if config_path is None or config_path is EXIT_SENTINEL:
                        logger.info("終了します")
                        sys.exit(0)

                    config_path = cast(Path, config_path)
                    logger.info(f"設定ファイルを読み込み: {config_path}")
                    try:
                        config = ConfigLoader.load(str(config_path))
                    except PochiConfigError as e:
                        logger.error(f"設定エラー: {e}")
                        sys.exit(1)

                    run_training(
                        config,
                        config_path=config_path,
                        stop_flag_callback=stop_flag_callback,
                    )
                    return

                else:
                    # 新規設定
                    wizard = TrainWizard()
                    result = wizard.run()

                    if result is None:
                        # キャンセル: 訓練モード選択に戻る
                        continue

                    config = result.to_dict()
                    config_content = result.to_python_config()
                    run_training(
                        config,
                        config_content=config_content,
                        stop_flag_callback=stop_flag_callback,
                    )
                    return

        elif mode == "infer":
            infer_wizard = InferWizard()
            infer_result = infer_wizard.run()

            if infer_result is None:
                # キャンセル: モード選択に戻る
                continue

            run_inference(
                model_path=Path(infer_result.model_path),
                data_path=Path(infer_result.data_path),
                output_dir=(
                    Path(infer_result.output_dir) if infer_result.output_dir else None
                ),
                device=infer_result.device,
            )
            return

        else:
            # キャンセル
            sys.exit(0)
