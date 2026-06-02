"""検証の実行.

検証ループを切り出し, 評価指標 (mIoU / Dice など) に加え val_loss も算出する.
AMP 有効時は autocast のみ適用する (GradScaler は不要).
"""

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.interfaces.model import ISegmentationModel

__all__ = ["Evaluator"]


class Evaluator:
    """検証処理を実行するクラス.

    Args:
        model: 検証対象のモデル.
        criterion: 損失関数 (val_loss 算出用).
        metrics: 評価指標.
        device: 使用デバイス.
        enable_amp: AMP を有効化するか.
    """

    def __init__(
        self,
        model: ISegmentationModel,
        criterion: ISegmentationLoss,
        metrics: ISegmentationMetrics,
        device: str,
        enable_amp: bool = False,
    ) -> None:
        """Evaluatorを初期化."""
        self._model = model
        self._criterion = criterion
        self._metrics = metrics
        self._device = device
        self._enable_amp = enable_amp

    def validate(
        self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, float]:
        """検証を実行する.

        Args:
            loader: 検証データローダー.

        Returns:
            評価指標の辞書 (mIoU / Dice などに val_loss を加えたもの).
        """
        self._model.eval()
        self._metrics.reset()
        total_loss = 0.0

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self._device)
                masks = masks.to(self._device)

                # AMP 対応 (autocast のみ, GradScaler 不要)
                if self._enable_amp:
                    with autocast(device_type="cuda"):
                        outputs = self._model(images)
                        loss = self._criterion(outputs, masks)
                else:
                    outputs = self._model(images)
                    loss = self._criterion(outputs, masks)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                self._metrics.update(preds, masks)

        result = self._metrics.compute()
        result["val_loss"] = total_loss / len(loader)
        return result
