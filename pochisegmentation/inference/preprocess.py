"""推論用の前処理.

numpy 画像を tv_tensors に変換し, transform を適用してモデル入力テンソルにする.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision import tv_tensors
from torchvision.transforms import v2

__all__ = ["preprocess_image"]


def preprocess_image(image: NDArray[np.uint8], transform: v2.Compose) -> torch.Tensor:
    """画像を前処理しモデル入力テンソルに変換する.

    Args:
        image: 入力画像 (H, W, C) の numpy 配列, RGB 形式.
        transform: 適用する transform (v2.Compose).

    Returns:
        前処理済みテンソル (1, C, H, W).
    """
    # numpy 配列を tv_tensors.Image に変換 (H, W, C) -> (C, H, W)
    image_tensor = tv_tensors.Image(image.transpose(2, 0, 1))

    # transform 適用
    transformed: torch.Tensor = transform(image_tensor)

    # バッチ次元を追加
    return transformed.unsqueeze(0)
