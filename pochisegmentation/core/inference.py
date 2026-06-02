"""推論のオーケストレーション.

モデル読み込みと推論の実行を行う.
CLIの入力方法（引数/対話）には依存しない.
"""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torchvision.transforms import v2

from pochisegmentation import ComponentFactory, PochiSegmentationPredictor
from pochisegmentation.exceptions import PochiConfigError
from pochisegmentation.inference.postprocess import save_prediction
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.utils.config_loader import ConfigLoader
from pochisegmentation.utils.timestamp_utils import (
    find_next_index,
    format_workspace_name,
    get_current_date_str,
)


def resolve_device(device: str) -> str:
    """デバイスを解決.

    Args:
        device: 指定されたデバイス.

    Returns:
        利用可能なデバイス.
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger = LoggerManager().get_logger("pochiseg")
        logger.warning("CUDAが利用できません. CPUを使用します.")
        return "cpu"
    return device


def get_image_paths(data_path: Path) -> list[Path]:
    """入力パスから画像パスのリストを取得.

    Args:
        data_path: 画像ファイル, ディレクトリ, またはパスリスト (.txt).

    Returns:
        画像パスのリスト.
    """
    if data_path.is_file() and data_path.suffix == ".txt":
        # テキストファイル: パスリストとして読み込み
        with open(data_path, "r", encoding="utf-8") as f:
            return [Path(line.strip()) for line in f if line.strip()]
    elif data_path.is_file():
        # 単一画像ファイル
        return [data_path]
    else:
        # ディレクトリ
        return (
            list(data_path.glob("*.jpg"))
            + list(data_path.glob("*.png"))
            + list(data_path.glob("*.bmp"))
        )


def run_inference(
    model_path: Path,
    data_path: Path,
    output_dir: Path | None = None,
    device: str = "cuda",
) -> None:
    """推論を実行.

    Args:
        model_path: モデルファイルのパス.
        data_path: 入力画像, ディレクトリ, またはパスリスト (.txt).
        output_dir: 出力ディレクトリ (Noneでデフォルト).
        device: 使用デバイス.
    """
    logger = LoggerManager().get_logger("pochiseg")

    logger.info(f"モデル読み込み: {model_path}")

    # 設定ファイル読み込み (モデルパスと同じディレクトリにあると仮定)
    config_path = model_path.parent.parent / "config.py"

    if config_path.exists():
        try:
            config = ConfigLoader.load(str(config_path))
        except PochiConfigError as e:
            logger.error(f"設定エラー: {e}")
            raise
    else:
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        logger.error("推論には訓練時の設定ファイルが必要です.")
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    # デバイス解決
    device = resolve_device(device)

    # モデル作成
    model = ComponentFactory.create_model(config)

    # Transform作成
    image_size = config.image_size
    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Predictor作成
    predictor = PochiSegmentationPredictor.from_checkpoint(
        checkpoint_path=str(model_path),
        model=model,
        transform=transform,
        device=device,
    )

    # 出力ディレクトリ作成
    work_dir = model_path.parent.parent
    if output_dir:
        resolved_output_dir = output_dir
    else:
        # work_dir/predictions/yyyymmdd_xxx/ を使用
        predictions_base = work_dir / "predictions"
        predictions_base.mkdir(parents=True, exist_ok=True)
        date_str = get_current_date_str()
        next_index = find_next_index(predictions_base, date_str)
        resolved_output_dir = predictions_base / format_workspace_name(
            date_str, next_index
        )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"出力ディレクトリ: {resolved_output_dir}")

    # 画像パス取得
    image_paths = get_image_paths(data_path)
    logger.info(f"推論対象: {len(image_paths)} 枚")

    # クラス数を取得
    num_classes = config.num_classes

    for image_path in image_paths:
        logger.info(f"推論中: {image_path}")

        # 元画像を読み込み (RGB)
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 推論
        mask = predictor.predict(image_path)

        # カラーマスク / オーバーレイの保存
        save_prediction(
            mask=mask,
            original_image=cast(NDArray[np.uint8], original_image),
            output_dir=resolved_output_dir,
            stem=image_path.stem,
            num_classes=num_classes,
            logger=logger,
        )

    logger.info("推論完了!")
