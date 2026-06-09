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

        # 全指標が 0.0-1.0 の範囲に収まることを確認
        for key, value in result.items():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0, f"{key} の値が範囲外: {value}"

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

        # 完全一致なので全指標が 1.0
        assert result["mIoU"] == pytest.approx(1.0)
        assert result["Dice"] == pytest.approx(1.0)
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

    def test_known_values(self) -> None:
        """既知入力に対する mIoU / Dice の期待値を検証 (回帰防止)."""
        metrics = SegmentationMetrics(num_classes=2, device="cpu")

        # 2x2, 2 クラス. 1 画素のみ誤分類.
        #   target = [[0, 0], [1, 1]], pred = [[0, 0], [1, 0]]
        #   class0 IoU = 2/3, class1 IoU = 1/2 -> mIoU = 7/12 = 0.5833...
        #   class0 Dice = 0.8, class1 Dice = 2/3 -> macro Dice = 0.7333...
        target = torch.tensor([[[0, 0], [1, 1]]])
        pred = torch.tensor([[[0, 0], [1, 0]]])

        metrics.update(pred, target)
        result = metrics.compute()

        assert result["mIoU"] == pytest.approx(7 / 12)
        assert result["Dice"] == pytest.approx((0.8 + 2 / 3) / 2)
        assert result["PixelAccuracy"] == pytest.approx(0.75)
