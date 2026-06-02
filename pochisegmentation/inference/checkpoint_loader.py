"""推論用のチェックポイント読み込み.

チェックポイントファイルからモデルへ重みを復元する.
"""

from pathlib import Path

import torch

from pochisegmentation.interfaces.model import ISegmentationModel

__all__ = ["load_model_weights"]


def load_model_weights(
    checkpoint_path: str | Path,
    model: ISegmentationModel,
    device: str = "cuda",
) -> None:
    """チェックポイントからモデルへ重みを読み込む.

    Args:
        checkpoint_path: チェックポイントファイルのパス.
        model: 重みを読み込む対象のモデル.
        device: 読み込み先デバイス (map_location).

    Raises:
        FileNotFoundError: チェックポイントファイルが存在しない場合.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"チェックポイントファイルが見つかりません: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
