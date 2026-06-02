"""inference.checkpoint_loader のユニットテスト."""

from pathlib import Path

import pytest
import torch

from pochisegmentation.inference.checkpoint_loader import load_model_weights
from pochisegmentation.interfaces.model import ISegmentationModel


class _MockModel(ISegmentationModel):
    """テスト用モデル."""

    def __init__(self) -> None:
        """_MockModel を初期化."""
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播."""
        return self.conv(x)

    def get_encoder_params(self) -> list:
        """エンコーダーパラメータを取得."""
        return []

    def get_decoder_params(self) -> list:
        """デコーダーパラメータを取得."""
        return list(self.conv.parameters())


class TestLoadModelWeights:
    """load_model_weights のテスト."""

    def test_loads_weights_into_model(self, tmp_path: Path) -> None:
        """チェックポイントの重みがモデルに復元される."""
        source = _MockModel()
        checkpoint_path = tmp_path / "ckpt.pth"
        torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)

        target = _MockModel()
        load_model_weights(checkpoint_path, target, device="cpu")

        assert torch.equal(target.conv.weight, source.conv.weight)

    def test_missing_file_raises(self) -> None:
        """存在しないチェックポイントで FileNotFoundError が送出される."""
        with pytest.raises(
            FileNotFoundError, match="チェックポイントファイルが見つかりません"
        ):
            load_model_weights("nonexistent.pth", _MockModel(), device="cpu")
