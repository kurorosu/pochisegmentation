"""ComponentFactoryのテスト."""

from typing import Any

import pytest
import torch
from pydantic import ValidationError

from pochisegmentation.config import PochiSegConfig
from pochisegmentation.factories.component_factory import ComponentFactory
from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.interfaces.model import ISegmentationModel
from pochisegmentation.losses.seg_losses import DiceLoss, FocalLoss, JaccardLoss
from pochisegmentation.metrics.seg_metrics import SegmentationMetrics
from pochisegmentation.models.deeplabv3plus import DeepLabV3PlusModel
from pochisegmentation.models.unet import UnetModel


def _make_config(**overrides: Any) -> PochiSegConfig:
    """テスト用 PochiSegConfig を生成する."""
    params: dict[str, Any] = {"data_root": "data", "num_classes": 4}
    params.update(overrides)
    return PochiSegConfig(**params)


class TestComponentFactory:
    """ComponentFactoryのテストクラス."""

    def test_model_registration(self) -> None:
        """モデル登録のテスト."""
        # 初期状態で登録済みを確認
        available = ComponentFactory.get_available_models()
        assert "Unet" in available
        assert "DeepLabV3Plus" in available

    def test_loss_registration(self) -> None:
        """損失関数登録のテスト."""
        available = ComponentFactory.get_available_losses()
        assert "DiceLoss" in available
        assert "FocalLoss" in available
        assert "JaccardLoss" in available

    def test_metrics_registration(self) -> None:
        """評価指標登録のテスト."""
        available = ComponentFactory.get_available_metrics()
        assert "SegmentationMetrics" in available

    def test_create_model_unet(self) -> None:
        """Unetモデル生成テスト."""
        config = _make_config(
            architecture="Unet", encoder_name="resnet34", pretrained=False
        )
        model = ComponentFactory.create_model(config)

        assert isinstance(model, ISegmentationModel)
        assert isinstance(model, UnetModel)

    def test_create_model_deeplabv3plus(self) -> None:
        """DeepLabV3+モデル生成テスト."""
        config = _make_config(
            architecture="DeepLabV3Plus", encoder_name="resnet34", pretrained=False
        )
        model = ComponentFactory.create_model(config)

        assert isinstance(model, ISegmentationModel)
        assert isinstance(model, DeepLabV3PlusModel)

    def test_create_model_default_architecture(self) -> None:
        """デフォルトアーキテクチャ (Unet) のテスト."""
        config = _make_config(pretrained=False)
        model = ComponentFactory.create_model(config)

        assert isinstance(model, UnetModel)

    def test_invalid_architecture_rejected_at_config(self) -> None:
        """不正なarchitectureはconfig構築時に弾かれる."""
        with pytest.raises(ValidationError):
            _make_config(architecture="UnknownModel")

    def test_create_loss_dice(self) -> None:
        """DiceLoss生成テスト."""
        config = _make_config(loss="DiceLoss", loss_params={"mode": "multiclass"})
        loss = ComponentFactory.create_loss(config)

        assert isinstance(loss, ISegmentationLoss)
        assert isinstance(loss, DiceLoss)

    def test_create_loss_focal(self) -> None:
        """FocalLoss生成テスト."""
        config = _make_config(loss="FocalLoss")
        loss = ComponentFactory.create_loss(config)

        assert isinstance(loss, FocalLoss)

    def test_create_loss_jaccard(self) -> None:
        """JaccardLoss生成テスト."""
        config = _make_config(loss="JaccardLoss")
        loss = ComponentFactory.create_loss(config)

        assert isinstance(loss, JaccardLoss)

    def test_create_loss_default(self) -> None:
        """デフォルト損失関数 (DiceLoss) のテスト."""
        config = _make_config()
        loss = ComponentFactory.create_loss(config)

        assert isinstance(loss, DiceLoss)

    def test_invalid_loss_rejected_at_config(self) -> None:
        """不正なlossはconfig構築時に弾かれる."""
        with pytest.raises(ValidationError):
            _make_config(loss="UnknownLoss")

    def test_create_metrics(self) -> None:
        """評価指標生成テスト."""
        config = _make_config(device="cpu")
        metrics = ComponentFactory.create_metrics(config)

        assert isinstance(metrics, ISegmentationMetrics)
        assert isinstance(metrics, SegmentationMetrics)

    def test_create_metrics_default(self) -> None:
        """デフォルト評価指標のテスト."""
        config = _make_config(device="cpu")
        metrics = ComponentFactory.create_metrics(config)

        assert isinstance(metrics, SegmentationMetrics)

    def test_invalid_num_classes_rejected_at_config(self) -> None:
        """num_classes がない場合はconfig構築時に弾かれる."""
        with pytest.raises(ValidationError):
            PochiSegConfig(data_root="data")  # type: ignore[call-arg]

    def test_model_forward_works(self) -> None:
        """生成したモデルで順伝播が動作することをテスト."""
        config = _make_config(pretrained=False)
        model = ComponentFactory.create_model(config)
        model.eval()

        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 4, 256, 256)

    def test_loss_call_works(self) -> None:
        """生成した損失関数で計算が動作することをテスト."""
        config = _make_config(loss="DiceLoss", loss_params={"mode": "multiclass"})
        loss = ComponentFactory.create_loss(config)

        pred = torch.randn(2, 4, 64, 64)
        target = torch.randint(0, 4, (2, 64, 64))

        result = loss(pred, target)
        assert isinstance(result, torch.Tensor)

    def test_metrics_workflow(self) -> None:
        """生成した評価指標でワークフローが動作することをテスト."""
        config = _make_config(device="cpu")
        metrics = ComponentFactory.create_metrics(config)

        preds = torch.randint(0, 4, (2, 64, 64))
        targets = torch.randint(0, 4, (2, 64, 64))

        metrics.update(preds, targets)
        result = metrics.compute()

        assert "mIoU" in result
        assert "Dice" in result

    def test_create_optimizer_adamw(self) -> None:
        """AdamWオプティマイザ生成テスト."""
        config = _make_config(optimizer="AdamW", pretrained=False)
        model = ComponentFactory.create_model(config)
        optimizer = ComponentFactory.create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_scheduler_none(self) -> None:
        """schedulerがNoneのときNoneを返す."""
        config = _make_config(scheduler=None, pretrained=False)
        model = ComponentFactory.create_model(config)
        optimizer = ComponentFactory.create_optimizer(model, config)
        scheduler = ComponentFactory.create_scheduler(optimizer, config)

        assert scheduler is None

    def test_create_scheduler_cosine(self) -> None:
        """CosineAnnealingLRスケジューラ生成テスト."""
        config = _make_config(
            scheduler="CosineAnnealingLR",
            scheduler_params={"T_max": 100},
            pretrained=False,
        )
        model = ComponentFactory.create_model(config)
        optimizer = ComponentFactory.create_optimizer(model, config)
        scheduler = ComponentFactory.create_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
