"""対話型推論コマンドの実装."""

import sys
from pathlib import Path

import cv2
import torch
from torchvision.transforms import v2

from pochisegmentation import (
    ComponentFactory,
    PochiSegmentationPredictor,
)
from pochisegmentation.cli.interactive.infer_runner import InferenceRunner
from pochisegmentation.exceptions import PochiConfigError
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.utils.config_loader import ConfigLoader
from pochisegmentation.utils.timestamp_utils import (
    find_next_index,
    format_workspace_name,
    get_current_date_str,
)
from pochisegmentation.visualization.mask_visualizer import (
    colorize_mask,
    overlay_mask_on_image,
)


def interactive_infer() -> None:
    """対話型モードで推論を実行."""
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger("pochi")

    # 対話フローを実行
    runner = InferenceRunner()
    infer_config = runner.run()

    if infer_config is None:
        logger.info("推論をキャンセルしました")
        sys.exit(0)

    model_path = Path(infer_config.model_path)
    logger.info(f"モデル読み込み: {model_path}")

    # 設定ファイル読み込み (モデルパスと同じディレクトリにあると仮定)
    config_path = model_path.parent.parent / "config.py"

    if config_path.exists():
        try:
            config = ConfigLoader.load(str(config_path))
        except PochiConfigError as e:
            logger.error(f"設定エラー: {e}")
            sys.exit(1)
    else:
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        logger.error("推論には訓練時の設定ファイルが必要です.")
        sys.exit(1)

    # デバイス
    device = infer_config.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDAが利用できません. CPUを使用します.")
        device = "cpu"

    # モデル作成
    model = ComponentFactory.create_model(config)

    # Transform作成
    image_size = config.get("image_size", 256)
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
    if infer_config.output_dir:
        output_dir = Path(infer_config.output_dir)
    else:
        # work_dir/predictions/yyyymmdd_xxx/ を使用
        predictions_base = work_dir / "predictions"
        predictions_base.mkdir(parents=True, exist_ok=True)
        date_str = get_current_date_str()
        next_index = find_next_index(predictions_base, date_str)
        output_dir = predictions_base / format_workspace_name(date_str, next_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"出力ディレクトリ: {output_dir}")

    # 推論実行
    data_path = Path(infer_config.data_path)

    if data_path.is_file() and data_path.suffix == ".txt":
        # テキストファイル: パスリストとして読み込み
        logger.info(f"パスリストファイルを読み込み: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            image_paths = [Path(line.strip()) for line in f if line.strip()]
    elif data_path.is_file():
        # 単一画像ファイル
        image_paths = [data_path]
    else:
        # ディレクトリ
        image_paths = (
            list(data_path.glob("*.jpg"))
            + list(data_path.glob("*.png"))
            + list(data_path.glob("*.bmp"))
        )

    logger.info(f"推論対象: {len(image_paths)} 枚")

    # クラス数を取得
    num_classes = config.get("num_classes", 2)

    for image_path in image_paths:
        logger.info(f"推論中: {image_path}")

        # 元画像を読み込み (RGB)
        original_image = cv2.imread(str(image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 推論
        mask = predictor.predict(image_path)

        # 1. カラーマスク単体を保存
        color_mask = colorize_mask(mask, num_classes=num_classes)
        mask_output_path = output_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_output_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        logger.info(f"マスク保存: {mask_output_path}")

        # 2. 元画像にオーバーレイした画像を保存
        overlay = overlay_mask_on_image(
            original_image, mask, alpha=0.5, num_classes=num_classes
        )
        vis_output_path = output_dir / f"{image_path.stem}_vis.png"
        cv2.imwrite(str(vis_output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logger.info(f"オーバーレイ保存: {vis_output_path}")

    logger.info("推論完了!")
