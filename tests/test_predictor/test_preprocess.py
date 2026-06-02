"""inference.preprocess のユニットテスト."""

import numpy as np
import torch
from torchvision.transforms import v2

from pochisegmentation.inference.preprocess import preprocess_image


def _transform() -> v2.Compose:
    """テスト用 transform."""
    return v2.Compose(
        [
            v2.Resize((32, 32)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


class TestPreprocessImage:
    """preprocess_image のテスト."""

    def test_returns_batched_chw_tensor(self) -> None:
        """(H, W, C) 入力が (1, C, H, W) のテンソルに変換される."""
        image = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)

        result = preprocess_image(image, _transform())

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 32, 32)

    def test_scales_to_float(self) -> None:
        """ToDtype(scale=True) で float32 [0, 1] に変換される."""
        image = np.full((10, 10, 3), 255, dtype=np.uint8)

        result = preprocess_image(image, _transform())

        assert result.dtype == torch.float32
        assert result.max().item() <= 1.0
