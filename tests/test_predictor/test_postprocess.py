"""inference.postprocess のユニットテスト."""

import logging
from pathlib import Path

import numpy as np

from pochisegmentation.inference.postprocess import save_prediction


class TestSavePrediction:
    """save_prediction のテスト."""

    def test_writes_mask_and_overlay(self, tmp_path: Path) -> None:
        """マスク画像とオーバーレイ画像が保存される."""
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[8:, 8:] = 1
        original = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)

        mask_path, vis_path = save_prediction(
            mask=mask,
            original_image=original,
            output_dir=tmp_path,
            stem="sample",
            num_classes=4,
            logger=logging.getLogger("test"),
        )

        assert mask_path == tmp_path / "sample_mask.png"
        assert vis_path == tmp_path / "sample_vis.png"
        assert mask_path.exists()
        assert vis_path.exists()
