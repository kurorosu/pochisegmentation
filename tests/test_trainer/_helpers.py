"""test_trainer 共通のテスト用ダブルとヘルパ.

classical テスト方針に沿い, unittest.mock ではなく軽量な実装クラスを用いる.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.interfaces.model import ISegmentationModel


class MockModel(ISegmentationModel):
    """テスト用モデル (1x1 Conv)."""

    def __init__(self, num_classes: int = 4) -> None:
        """MockModelを初期化."""
        super().__init__()
        self.num_classes = num_classes
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播."""
        return self.conv(x)

    def get_encoder_params(self) -> list:
        """エンコーダーパラメータを取得."""
        return []

    def get_decoder_params(self) -> list:
        """デコーダーパラメータを取得."""
        return list(self.conv.parameters())


class MockLoss(ISegmentationLoss):
    """テスト用損失関数 (cross entropy)."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失を計算."""
        return torch.nn.functional.cross_entropy(pred, target)


class MockMetrics(ISegmentationMetrics):
    """テスト用評価指標 (常に mIoU=0.5, Dice=0.6)."""

    def __init__(self) -> None:
        """MockMetricsを初期化."""
        self._count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """バッチ結果を蓄積."""
        self._count += 1

    def compute(self) -> dict[str, float]:
        """指標を計算."""
        return {"mIoU": 0.5, "Dice": 0.6}

    def reset(self) -> None:
        """リセット."""
        self._count = 0


def create_dummy_dataloader(
    batch_size: int = 2, num_samples: int = 8
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """ダミーデータローダーを作成."""
    images = torch.randn(num_samples, 3, 32, 32)
    masks = torch.randint(0, 4, (num_samples, 32, 32))
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=batch_size)
