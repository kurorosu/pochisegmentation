"""1 エポック分の訓練実行.

AMP (混合精度訓練) 対応込みで 1 エポックの訓練ループを切り出す.
GradScaler / autocast の有無による分岐は seg_trainer の従来挙動を保つ.
"""

import torch
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.model import ISegmentationModel

__all__ = ["EpochRunner"]


class EpochRunner:
    """1 エポック分の訓練処理を実行するクラス.

    Args:
        model: 訓練対象のモデル.
        criterion: 損失関数.
        optimizer: オプティマイザ.
        device: 使用デバイス.
        enable_amp: AMP を有効化するか.
        scaler: GradScaler (AMP 有効時のみ).
    """

    def __init__(
        self,
        model: ISegmentationModel,
        criterion: ISegmentationLoss,
        optimizer: Optimizer,
        device: str,
        enable_amp: bool = False,
        scaler: GradScaler | None = None,
    ) -> None:
        """EpochRunnerを初期化."""
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = device
        self._enable_amp = enable_amp
        self._scaler = scaler

    def run(self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """1 エポックの訓練を実行する.

        Args:
            loader: 訓練データローダー.

        Returns:
            平均訓練損失.
        """
        self._model.train()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(self._device)
            masks = masks.to(self._device)

            self._optimizer.zero_grad()

            # AMP 対応
            if self._enable_amp and self._scaler is not None:
                with autocast(device_type="cuda"):
                    outputs = self._model(images)
                    loss = self._criterion(outputs, masks)
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                outputs = self._model(images)
                loss = self._criterion(outputs, masks)
                loss.backward()
                self._optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
