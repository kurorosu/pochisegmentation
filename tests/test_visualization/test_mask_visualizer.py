"""mask_visualizer.py のテスト."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from pochisegmentation.visualization.mask_visualizer import (
    VOC_COLORMAP,
    colorize_mask,
    colorize_mask_grayscale,
    create_color_palette,
    overlay_mask_on_image,
    save_mask_visualization,
)


class TestCreateColorPalette:
    """create_color_palette関数のテスト."""

    def test_basic_palette(self) -> None:
        """基本的なカラーパレット作成."""
        palette = create_color_palette(5)

        assert palette.shape == (5, 3)
        assert palette.dtype == np.uint8

        # VOCカラーマップの値と一致
        for i in range(5):
            assert tuple(palette[i]) == VOC_COLORMAP[i]

    def test_palette_with_voc_colormap_size(self) -> None:
        """VOCカラーマップサイズのパレット."""
        num_classes = len(VOC_COLORMAP)
        palette = create_color_palette(num_classes)

        assert palette.shape == (num_classes, 3)
        for i in range(num_classes):
            assert tuple(palette[i]) == VOC_COLORMAP[i]

    def test_palette_exceeding_voc_colormap(self) -> None:
        """VOCカラーマップを超えたクラス数のパレット."""
        num_classes = len(VOC_COLORMAP) + 10
        palette = create_color_palette(num_classes)

        assert palette.shape == (num_classes, 3)

        # VOCカラーマップ内の色は一致
        for i in range(len(VOC_COLORMAP)):
            assert tuple(palette[i]) == VOC_COLORMAP[i]

        # VOCカラーマップ外の色は自動生成 (決定論的)
        for i in range(len(VOC_COLORMAP), num_classes):
            expected = [
                (i * 37) % 256,
                (i * 89) % 256,
                (i * 157) % 256,
            ]
            assert list(palette[i]) == expected

    def test_palette_single_class(self) -> None:
        """1クラスのパレット."""
        palette = create_color_palette(1)

        assert palette.shape == (1, 3)
        assert tuple(palette[0]) == VOC_COLORMAP[0]  # 背景色


class TestColorizeMask:
    """colorize_mask関数のテスト."""

    def test_basic_colorization(self) -> None:
        """基本的なマスクカラー化."""
        mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)

        color_mask = colorize_mask(mask, num_classes=3)

        assert color_mask.shape == (2, 2, 3)
        assert color_mask.dtype == np.uint8

        # 各ピクセルの色を確認
        assert tuple(color_mask[0, 0]) == VOC_COLORMAP[0]
        assert tuple(color_mask[0, 1]) == VOC_COLORMAP[1]
        assert tuple(color_mask[1, 0]) == VOC_COLORMAP[2]
        assert tuple(color_mask[1, 1]) == VOC_COLORMAP[0]

    def test_colorization_without_num_classes(self) -> None:
        """num_classes未指定時のカラー化."""
        mask = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint8)

        color_mask = colorize_mask(mask)

        # num_classesはmask.max()+1=4として推測される
        assert color_mask.shape == (2, 3, 3)

    def test_colorization_with_custom_palette(self) -> None:
        """カスタムパレットでのカラー化."""
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        custom_palette = np.array(
            [[255, 0, 0], [0, 255, 0]], dtype=np.uint8  # 赤  # 緑
        )

        color_mask = colorize_mask(mask, palette=custom_palette)

        assert tuple(color_mask[0, 0]) == (255, 0, 0)
        assert tuple(color_mask[0, 1]) == (0, 255, 0)

    def test_colorization_large_mask(self) -> None:
        """大きなマスクのカラー化."""
        mask = np.random.randint(0, 5, size=(256, 256), dtype=np.uint8)

        color_mask = colorize_mask(mask, num_classes=5)

        assert color_mask.shape == (256, 256, 3)


class TestColorizeMaskGrayscale:
    """colorize_mask_grayscale関数のテスト."""

    def test_basic_grayscale(self) -> None:
        """基本的なグレースケール変換."""
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)

        gray_mask = colorize_mask_grayscale(mask, num_classes=4)

        assert gray_mask.shape == (2, 2)
        assert gray_mask.dtype == np.uint8

        # クラス数4の場合、スケール係数は 255 // 3 = 85
        scale = 255 // 3
        assert gray_mask[0, 0] == 0 * scale
        assert gray_mask[0, 1] == 1 * scale
        assert gray_mask[1, 0] == 2 * scale
        assert gray_mask[1, 1] == 3 * scale

    def test_grayscale_without_num_classes(self) -> None:
        """num_classes未指定時のグレースケール変換."""
        mask = np.array([[0, 1, 2]], dtype=np.uint8)

        gray_mask = colorize_mask_grayscale(mask)

        # num_classesはmask.max()+1=3として推測される
        scale = 255 // 2
        assert gray_mask[0, 0] == 0
        assert gray_mask[0, 1] == scale
        assert gray_mask[0, 2] == 2 * scale

    def test_grayscale_single_class(self) -> None:
        """単一クラスのグレースケール変換."""
        mask = np.zeros((3, 3), dtype=np.uint8)

        gray_mask = colorize_mask_grayscale(mask, num_classes=1)

        # 1クラスのみの場合、mask * 255
        assert np.all(gray_mask == 0)

    def test_grayscale_binary(self) -> None:
        """2クラス (バイナリ) のグレースケール変換."""
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        gray_mask = colorize_mask_grayscale(mask, num_classes=2)

        # 2クラスの場合、スケール係数は 255
        assert gray_mask[0, 0] == 0
        assert gray_mask[0, 1] == 255
        assert gray_mask[1, 0] == 255
        assert gray_mask[1, 1] == 0


class TestOverlayMaskOnImage:
    """overlay_mask_on_image関数のテスト."""

    def test_basic_overlay(self) -> None:
        """基本的なオーバーレイ."""
        image = np.full((4, 4, 3), 200, dtype=np.uint8)  # グレー画像
        mask = np.array(
            [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]], dtype=np.uint8
        )

        overlay = overlay_mask_on_image(image, mask, alpha=0.5, num_classes=3)

        assert overlay.shape == (4, 4, 3)
        assert overlay.dtype == np.uint8

        # 背景 (クラス0) は元画像のまま
        assert np.array_equal(overlay[0, 0], image[0, 0])
        assert np.array_equal(overlay[2, 2], image[2, 2])

        # 前景 (クラス1, 2) はブレンドされる
        assert not np.array_equal(overlay[0, 2], image[0, 2])

    def test_overlay_with_different_sizes(self) -> None:
        """異なるサイズの画像とマスクのオーバーレイ."""
        image = np.full((256, 256, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[32:, 32:] = 1  # 右下を前景

        overlay = overlay_mask_on_image(image, mask, alpha=0.5)

        # マスクがリサイズされて適用される
        assert overlay.shape == (256, 256, 3)

    def test_overlay_alpha_zero(self) -> None:
        """alpha=0 (完全透明) のオーバーレイ."""
        image = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.uint8)

        overlay = overlay_mask_on_image(image, mask, alpha=0.0)

        # alpha=0でも前景領域では image * 1.0 になる (完全に元画像)
        # ただし実装上は (1-0)*image + 0*color = image
        assert overlay.shape == image.shape

    def test_overlay_alpha_one(self) -> None:
        """alpha=1 (完全不透明) のオーバーレイ."""
        image = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.uint8)

        overlay = overlay_mask_on_image(image, mask, alpha=1.0, num_classes=2)

        # alpha=1では前景領域がカラーマスクの色になる
        expected_color = VOC_COLORMAP[1]
        assert tuple(overlay[0, 0]) == expected_color

    def test_overlay_with_custom_palette(self) -> None:
        """カスタムパレットでのオーバーレイ."""
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        palette = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)

        overlay = overlay_mask_on_image(
            image, mask, alpha=1.0, num_classes=2, palette=palette
        )

        # 前景は赤
        assert tuple(overlay[0, 1]) == (255, 0, 0)


class TestSaveMaskVisualization:
    """save_mask_visualization関数のテスト."""

    def test_save_mask_png(self) -> None:
        """マスクをPNG形式で保存."""
        mask = np.array(
            [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_mask.png"
            save_mask_visualization(mask, output_path, num_classes=3)

            assert output_path.exists()

            # 保存された画像を読み込んで確認
            saved_image = cv2.imread(str(output_path))
            assert saved_image is not None
            assert saved_image.shape == (3, 3, 3)

    def test_save_mask_str_path(self) -> None:
        """文字列パスでの保存."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f"{temp_dir}/mask.png"
            save_mask_visualization(mask, output_path)

            assert Path(output_path).exists()

    def test_save_mask_with_custom_palette(self) -> None:
        """カスタムパレットでの保存."""
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        palette = np.array([[0, 0, 0], [0, 255, 0]], dtype=np.uint8)  # 黒と緑

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "custom_mask.png"
            save_mask_visualization(mask, output_path, palette=palette)

            assert output_path.exists()

            # BGR形式で読み込み、緑色を確認
            saved_image = cv2.imread(str(output_path))
            # 緑 (RGB: 0,255,0 -> BGR: 0,255,0)
            assert saved_image[0, 1, 1] == 255  # G channel


class TestVOCColormap:
    """VOC_COLORMAP定数のテスト."""

    def test_colormap_length(self) -> None:
        """カラーマップの長さ."""
        assert len(VOC_COLORMAP) == 21

    def test_colormap_background(self) -> None:
        """背景色 (インデックス0) は黒."""
        assert VOC_COLORMAP[0] == (0, 0, 0)

    def test_colormap_format(self) -> None:
        """カラーマップの各エントリはRGBタプル."""
        for color in VOC_COLORMAP:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
