"""推論結果の後処理と保存.

予測マスクをカラーマスク / オーバーレイ画像として保存する.
着色・重ね合わせ自体は visualization.mask_visualizer に委譲する.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from pochisegmentation.visualization.mask_visualizer import (
    colorize_mask,
    overlay_mask_on_image,
)

__all__ = ["save_prediction"]


def save_prediction(
    mask: NDArray[np.uint8],
    original_image: NDArray[np.uint8],
    output_dir: Path,
    stem: str,
    num_classes: int,
    logger: logging.Logger,
    alpha: float = 0.5,
) -> tuple[Path, Path]:
    """予測マスクをカラーマスク / オーバーレイ画像として保存する.

    Args:
        mask: 予測マスク (H, W), クラスインデックス.
        original_image: 元画像 (H, W, C), RGB 形式.
        output_dir: 出力ディレクトリ.
        stem: 出力ファイル名のステム (拡張子なし).
        num_classes: クラス数.
        logger: ロガーインスタンス.
        alpha: オーバーレイの不透明度.

    Returns:
        (マスク画像パス, オーバーレイ画像パス) のタプル.
    """
    # 1. カラーマスク単体を保存
    color_mask = colorize_mask(mask, num_classes=num_classes)
    mask_output_path = output_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_output_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    logger.info(f"マスク保存: {mask_output_path}")

    # 2. 元画像にオーバーレイした画像を保存
    overlay = overlay_mask_on_image(
        original_image,
        mask,
        alpha=alpha,
        num_classes=num_classes,
    )
    vis_output_path = output_dir / f"{stem}_vis.png"
    cv2.imwrite(str(vis_output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.info(f"オーバーレイ保存: {vis_output_path}")

    return mask_output_path, vis_output_path
