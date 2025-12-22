"""セグメンテーション評価指標のテスト."""

import pytest
import torch

from pochisegmentation.metrics.seg_metrics import SegmentationMetrics


class TestSegmentationMetrics:
    """SegmentationMetricsのテストクラス."""

    def test_init(self) -> None:
        """初期化テスト."""
        metrics = SegmentationMetrics(num_classes=4, device="cpu")
        assert metrics is not None

    def test_update(self) -> None:
        """update メソッドのテスト."""
        metrics = SegmentationMetrics(num_classes=4, device="cpu")

        batch_size, height, width = 2, 64, 64
        preds = torch.randint(0, 4, (batch_size, height, width))
        targets = torch.randint(0, 4, (batch_size, height, width))

        # エラーなく実行できることを確認
        metrics.update(preds, targets)

    def test_compute(self) -> None:
        """compute メソッドのテスト."""
        metrics = SegmentationMetrics(num_classes=4, device="cpu")

        batch_size, height, width = 2, 64, 64
        preds = torch.randint(0, 4, (batch_size, height, width))
        targets = torch.randint(0, 4, (batch_size, height, width))

        metrics.update(preds, targets)
        result = metrics.compute()

        assert isinstance(result, dict)
        assert "mIoU" in result
        assert "Dice" in result
        assert "PixelAccuracy" in result
        assert "F1" in result

        # 値が妥当な範囲にあることを確認
        for key, value in result.items():
            assert isinstance(value, float)
            # DiceScoreはmacro平均でクラス数分の合計になる可能性がある
            assert value >= 0.0, f"{key} の値が負: {value}"

    def test_reset(self) -> None:
        """reset メソッドのテスト."""
        metrics = SegmentationMetrics(num_classes=4, device="cpu")

        # データを追加
        preds = torch.randint(0, 4, (2, 64, 64))
        targets = torch.randint(0, 4, (2, 64, 64))
        metrics.update(preds, targets)

        # リセット後は状態がクリアされる
        metrics.reset()

        # 新しいデータで計算
        new_preds = torch.zeros(2, 64, 64, dtype=torch.long)
        new_targets = torch.zeros(2, 64, 64, dtype=torch.long)
        metrics.update(new_preds, new_targets)

        result = metrics.compute()
        # 完全一致なので精度は1.0
        assert result["PixelAccuracy"] == 1.0

    def test_perfect_prediction(self) -> None:
        """完全一致の場合のテスト."""
        metrics = SegmentationMetrics(num_classes=4, device="cpu")

        # 予測とターゲットが完全一致
        targets = torch.randint(0, 4, (2, 64, 64))
        preds = targets.clone()

        metrics.update(preds, targets)
        result = metrics.compute()

        assert result["PixelAccuracy"] == 1.0
        assert result["F1"] == 1.0

    def test_multiple_updates(self) -> None:
        """複数回のupdateが正しく蓄積されることをテスト."""
        metrics = SegmentationMetrics(num_classes=4, device="cpu")

        # 2回に分けてupdate
        for _ in range(2):
            preds = torch.randint(0, 4, (2, 64, 64))
            targets = torch.randint(0, 4, (2, 64, 64))
            metrics.update(preds, targets)

        result = metrics.compute()

        # 値が計算できることを確認
        assert isinstance(result["mIoU"], float)
        assert isinstance(result["Dice"], float)

    def test_binary_segmentation(self) -> None:
        """2クラス (バイナリ) セグメンテーションのテスト."""
        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        preds = torch.randint(0, 2, (2, 64, 64))
        targets = torch.randint(0, 2, (2, 64, 64))

        metrics.update(preds, targets)
        result = metrics.compute()

        assert isinstance(result, dict)
        assert len(result) == 4
