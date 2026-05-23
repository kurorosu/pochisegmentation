"""セグメンテーションマスク可視化ユーティリティ.

推論結果のマスクをカラーマップで可視化する機能を提供.
"""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

# VOCスタイルのカラーパレット (最大256クラス対応)
# 各クラスに異なる色を割り当て
VOC_COLORMAP: list[tuple[int, int, int]] = [
    (0, 0, 0),  # 0: 背景 (黒)
    (128, 0, 0),  # 1: 赤系
    (0, 128, 0),  # 2: 緑系
    (128, 128, 0),  # 3: 黄系
    (0, 0, 128),  # 4: 青系
    (128, 0, 128),  # 5: 紫系
    (0, 128, 128),  # 6: シアン系
    (128, 128, 128),  # 7: グレー
    (64, 0, 0),  # 8: 暗赤
    (192, 0, 0),  # 9: 明赤
    (64, 128, 0),  # 10: オリーブ
    (192, 128, 0),  # 11: オレンジ
    (64, 0, 128),  # 12: インディゴ
    (192, 0, 128),  # 13: マゼンタ
    (64, 128, 128),  # 14: ティール
    (192, 128, 128),  # 15: ピンク
    (0, 64, 0),  # 16: 暗緑
    (128, 64, 0),  # 17: 茶
    (0, 192, 0),  # 18: 明緑
    (128, 192, 0),  # 19: ライム
    (0, 64, 128),  # 20: 暗シアン
]


def create_color_palette(num_classes: int) -> NDArray[np.uint8]:
    """カラーパレットを作成.

    Args:
        num_classes: クラス数.

    Returns:
        カラーパレット (num_classes, 3) のnumpy配列, RGB形式.
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    for i in range(num_classes):
        if i < len(VOC_COLORMAP):
            palette[i] = VOC_COLORMAP[i]
        else:
            # VOCカラーマップを超えた場合はランダム風の色を生成
            # 再現性のため、インデックスベースで決定論的に生成
            palette[i] = [
                (i * 37) % 256,
                (i * 89) % 256,
                (i * 157) % 256,
            ]

    return palette


def colorize_mask(
    mask: NDArray[np.uint8],
    num_classes: int | None = None,
    palette: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """マスクをカラー画像に変換.

    Args:
        mask: クラスインデックスマスク (H, W).
        num_classes: クラス数. Noneの場合はマスク内の最大値+1.
        palette: カスタムカラーパレット. Noneの場合はVOCスタイル.

    Returns:
        カラーマスク (H, W, 3) のnumpy配列, RGB形式.
    """
    if num_classes is None:
        num_classes = int(mask.max()) + 1

    if palette is None:
        palette = create_color_palette(num_classes)

    # カラーマスクを作成
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(num_classes):
        color_mask[mask == class_id] = palette[class_id]

    return color_mask


def colorize_mask_grayscale(
    mask: NDArray[np.uint8],
    num_classes: int | None = None,
) -> NDArray[np.uint8]:
    """マスクをグレースケール強調画像に変換.

    Args:
        mask: クラスインデックスマスク (H, W).
        num_classes: クラス数. Noneの場合はマスク内の最大値+1.

    Returns:
        グレースケールマスク (H, W) のnumpy配列.
    """
    if num_classes is None:
        num_classes = int(mask.max()) + 1

    if num_classes <= 1:
        return (mask * 255).astype(np.uint8)

    # クラスインデックスを均等に分散
    scale_factor = 255 // (num_classes - 1)
    return (mask * scale_factor).astype(np.uint8)


def overlay_mask_on_image(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    alpha: float = 0.5,
    num_classes: int | None = None,
    palette: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """マスクを元画像にオーバーレイ.

    Args:
        image: 元画像 (H, W, 3), RGB形式.
        mask: クラスインデックスマスク (H, W).
        alpha: マスクの透明度 (0.0-1.0).
        num_classes: クラス数.
        palette: カスタムカラーパレット.

    Returns:
        オーバーレイ画像 (H, W, 3) のnumpy配列, RGB形式.
    """
    # 画像サイズが異なる場合はマスクをリサイズ
    if image.shape[:2] != mask.shape:
        mask = cast(
            NDArray[np.uint8],
            cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ),
        )

    color_mask = colorize_mask(mask, num_classes, palette)

    # アルファブレンディング (背景クラス=0は透明に)
    mask_expanded = np.expand_dims(mask, axis=-1)

    # 背景以外の領域のみブレンド
    foreground_mask = mask_expanded > 0
    overlay = np.where(
        foreground_mask,
        (image * (1 - alpha) + color_mask * alpha).astype(np.uint8),
        image,
    )

    return overlay


def save_mask_visualization(
    mask: NDArray[np.uint8],
    output_path: str | Path,
    num_classes: int | None = None,
    palette: NDArray[np.uint8] | None = None,
) -> None:
    """マスクをカラー画像として保存.

    Args:
        mask: クラスインデックスマスク (H, W).
        output_path: 出力パス.
        num_classes: クラス数.
        palette: カスタムカラーパレット.
    """
    color_mask = colorize_mask(mask, num_classes, palette)
    # RGB -> BGR for OpenCV
    cv2.imwrite(str(output_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
