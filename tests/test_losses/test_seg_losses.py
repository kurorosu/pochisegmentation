"""セグメンテーション損失関数のテスト."""

import pytest
import torch

from pochisegmentation.losses.seg_losses import (
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    JaccardLoss,
)


class TestDiceLoss:
    """DiceLossのテストクラス."""

    def test_init_default(self) -> None:
        """デフォルトパラメータでの初期化テスト."""
        loss = DiceLoss()
        assert loss is not None

    def test_init_binary_mode(self) -> None:
        """バイナリモードでの初期化テスト."""
        loss = DiceLoss(mode="binary")
        assert loss is not None

    def test_call_multiclass(self) -> None:
        """マルチクラス損失計算テスト."""
        loss = DiceLoss(mode="multiclass")

        batch_size, num_classes, height, width = 2, 4, 64, 64
        pred = torch.randn(batch_size, num_classes, height, width)
        target = torch.randint(0, num_classes, (batch_size, height, width))

        result = loss(pred, target)

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # スカラー
        assert result.item() >= 0

    def test_call_binary(self) -> None:
        """バイナリ損失計算テスト."""
        loss = DiceLoss(mode="binary")

        batch_size, height, width = 2, 64, 64
        pred = torch.randn(batch_size, 1, height, width)
        target = torch.randint(0, 2, (batch_size, height, width)).float()

        result = loss(pred, target)

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0


class TestFocalLoss:
    """FocalLossのテストクラス."""

    def test_init_default(self) -> None:
        """デフォルトパラメータでの初期化テスト."""
        loss = FocalLoss()
        assert loss is not None

    def test_call_multiclass(self) -> None:
        """マルチクラス損失計算テスト."""
        loss = FocalLoss(mode="multiclass")

        batch_size, num_classes, height, width = 2, 4, 64, 64
        pred = torch.randn(batch_size, num_classes, height, width)
        target = torch.randint(0, num_classes, (batch_size, height, width))

        result = loss(pred, target)

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.item() >= 0


class TestJaccardLoss:
    """JaccardLossのテストクラス."""

    def test_init_default(self) -> None:
        """デフォルトパラメータでの初期化テスト."""
        loss = JaccardLoss()
        assert loss is not None

    def test_call_multiclass(self) -> None:
        """マルチクラス損失計算テスト."""
        loss = JaccardLoss(mode="multiclass")

        batch_size, num_classes, height, width = 2, 4, 64, 64
        pred = torch.randn(batch_size, num_classes, height, width)
        target = torch.randint(0, num_classes, (batch_size, height, width))

        result = loss(pred, target)

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.item() >= 0


class TestCombinedLoss:
    """CombinedLossのテストクラス."""

    def test_init_default_weights(self) -> None:
        """デフォルト重みでの初期化テスト."""
        losses = [DiceLoss(), FocalLoss()]
        combined = CombinedLoss(losses)
        assert combined is not None

    def test_init_custom_weights(self) -> None:
        """カスタム重みでの初期化テスト."""
        losses = [DiceLoss(), FocalLoss()]
        weights = [0.5, 0.5]
        combined = CombinedLoss(losses, weights)
        assert combined is not None

    def test_init_mismatched_weights_raises(self) -> None:
        """重みの数が一致しない場合のエラーテスト."""
        losses = [DiceLoss(), FocalLoss()]
        weights = [0.5]  # 長さが一致しない

        with pytest.raises(ValueError, match="一致しません"):
            CombinedLoss(losses, weights)

    def test_call(self) -> None:
        """複合損失計算テスト."""
        losses = [DiceLoss(), JaccardLoss()]
        weights = [0.7, 0.3]
        combined = CombinedLoss(losses, weights)

        batch_size, num_classes, height, width = 2, 4, 64, 64
        pred = torch.randn(batch_size, num_classes, height, width)
        target = torch.randint(0, num_classes, (batch_size, height, width))

        result = combined(pred, target)

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.item() >= 0

    def test_call_weighted_sum(self) -> None:
        """重み付き合計が正しく計算されることをテスト."""
        dice = DiceLoss()
        jaccard = JaccardLoss()
        combined = CombinedLoss([dice, jaccard], [0.6, 0.4])

        batch_size, num_classes, height, width = 2, 4, 64, 64
        pred = torch.randn(batch_size, num_classes, height, width)
        target = torch.randint(0, num_classes, (batch_size, height, width))

        dice_val = dice(pred, target)
        jaccard_val = jaccard(pred, target)
        expected = 0.6 * dice_val + 0.4 * jaccard_val

        result = combined(pred, target)

        assert torch.isclose(result, expected, rtol=1e-5)
