"""セグメンテーション損失関数の実装."""

from typing import Any

import segmentation_models_pytorch as smp
import torch

from pochisegmentation.interfaces.loss import ISegmentationLoss


class DiceLoss(ISegmentationLoss):
    """Dice損失 (smpラッパー).

    Diceスコアに基づく損失関数で, クラス不均衡に対して頑健.

    Attributes:
        _loss: smp.losses.DiceLossインスタンス.
    """

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        """Dice損失を初期化.

        Args:
            mode: 損失計算モード ("binary", "multiclass", "multilabel").
            **kwargs: smp.losses.DiceLossに渡す追加引数.
        """
        self._loss = smp.losses.DiceLoss(mode=mode, **kwargs)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失を計算.

        Args:
            pred: 予測テンソル, 形状は (B, num_classes, H, W).
            target: ターゲットテンソル, 形状は (B, H, W).

        Returns:
            スカラー損失テンソル.
        """
        loss: torch.Tensor = self._loss(pred, target)
        return loss


class FocalLoss(ISegmentationLoss):
    """Focal損失 (smpラッパー).

    難しいサンプルに重みを置く損失関数で, クラス不均衡に効果的.

    Attributes:
        _loss: smp.losses.FocalLossインスタンス.
    """

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        """Focal損失を初期化.

        Args:
            mode: 損失計算モード ("binary", "multiclass", "multilabel").
            **kwargs: smp.losses.FocalLossに渡す追加引数.
        """
        self._loss = smp.losses.FocalLoss(mode=mode, **kwargs)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失を計算.

        Args:
            pred: 予測テンソル, 形状は (B, num_classes, H, W).
            target: ターゲットテンソル, 形状は (B, H, W).

        Returns:
            スカラー損失テンソル.
        """
        loss: torch.Tensor = self._loss(pred, target)
        return loss


class JaccardLoss(ISegmentationLoss):
    """Jaccard (IoU) 損失 (smpラッパー).

    Intersection over Unionに基づく損失関数.

    Attributes:
        _loss: smp.losses.JaccardLossインスタンス.
    """

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        """Jaccard損失を初期化.

        Args:
            mode: 損失計算モード ("binary", "multiclass", "multilabel").
            **kwargs: smp.losses.JaccardLossに渡す追加引数.
        """
        self._loss = smp.losses.JaccardLoss(mode=mode, **kwargs)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失を計算.

        Args:
            pred: 予測テンソル, 形状は (B, num_classes, H, W).
            target: ターゲットテンソル, 形状は (B, H, W).

        Returns:
            スカラー損失テンソル.
        """
        loss: torch.Tensor = self._loss(pred, target)
        return loss


class CombinedLoss(ISegmentationLoss):
    """複合損失関数.

    複数の損失関数を重み付きで組み合わせる.

    Attributes:
        _losses: 損失関数のリスト.
        _weights: 各損失関数の重み.
    """

    def __init__(
        self,
        losses: list[ISegmentationLoss],
        weights: list[float] | None = None,
    ) -> None:
        """複合損失関数を初期化.

        Args:
            losses: 組み合わせる損失関数のリスト.
            weights: 各損失関数の重み. Noneの場合は等しい重み.
        """
        self._losses = losses
        self._weights = weights or [1.0] * len(losses)

        if len(self._losses) != len(self._weights):
            raise ValueError(
                f"損失関数の数 ({len(self._losses)}) と "
                f"重みの数 ({len(self._weights)}) が一致しません."
            )

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失を計算.

        Args:
            pred: 予測テンソル, 形状は (B, num_classes, H, W).
            target: ターゲットテンソル, 形状は (B, H, W).

        Returns:
            重み付き損失の合計.
        """
        total: torch.Tensor = torch.tensor(0.0, device=pred.device)
        for loss, weight in zip(self._losses, self._weights):
            total = total + weight * loss(pred, target)
        return total
