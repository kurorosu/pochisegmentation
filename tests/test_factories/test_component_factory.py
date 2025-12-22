"""ComponentFactoryのテスト."""

import pytest
import torch

from pochisegmentation.factories.component_factory import ComponentFactory
from pochisegmentation.interfaces.loss import ISegmentationLoss
from pochisegmentation.interfaces.metrics import ISegmentationMetrics
from pochisegmentation.interfaces.model import ISegmentationModel
from pochisegmentation.losses.seg_losses import DiceLoss, FocalLoss, JaccardLoss
from pochisegmentation.metrics.seg_metrics import SegmentationMetrics
from pochisegmentation.models.deeplabv3plus import DeepLabV3PlusModel
from pochisegmentation.models.unet import UnetModel


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
        config = {
            "architecture": "Unet",
            "encoder_name": "resnet34",
            "num_classes": 4,
            "pretrained": False,
        }
        model = ComponentFactory.create_model(config)

        assert isinstance(model, ISegmentationModel)
        assert isinstance(model, UnetModel)

    def test_create_model_deeplabv3plus(self) -> None:
        """DeepLabV3+モデル生成テスト."""
        config = {
            "architecture": "DeepLabV3Plus",
            "encoder_name": "resnet34",
            "num_classes": 4,
            "pretrained": False,
        }
        model = ComponentFactory.create_model(config)

        assert isinstance(model, ISegmentationModel)
        assert isinstance(model, DeepLabV3PlusModel)

    def test_create_model_default_architecture(self) -> None:
        """デフォルトアーキテクチャ (Unet) のテスト."""
        config = {
            "num_classes": 4,
            "pretrained": False,
        }
        model = ComponentFactory.create_model(config)

        assert isinstance(model, UnetModel)

    def test_create_model_unknown_raises(self) -> None:
        """未登録モデルでエラーが発生することをテスト."""
        config = {
            "architecture": "UnknownModel",
            "num_classes": 4,
        }
        with pytest.raises(ValueError, match="未登録のモデル"):
            ComponentFactory.create_model(config)

    def test_create_loss_dice(self) -> None:
        """DiceLoss生成テスト."""
        config = {
            "loss": "DiceLoss",
            "loss_params": {"mode": "multiclass"},
        }
        loss = ComponentFactory.create_loss(config)

        assert isinstance(loss, ISegmentationLoss)
        assert isinstance(loss, DiceLoss)

    def test_create_loss_focal(self) -> None:
        """FocalLoss生成テスト."""
        config = {
            "loss": "FocalLoss",
        }
        loss = ComponentFactory.create_loss(config)

        assert isinstance(loss, FocalLoss)

    def test_create_loss_jaccard(self) -> None:
        """JaccardLoss生成テスト."""
        config = {
            "loss": "JaccardLoss",
        }
        loss = ComponentFactory.create_loss(config)

        assert isinstance(loss, JaccardLoss)

    def test_create_loss_default(self) -> None:
        """デフォルト損失関数 (DiceLoss) のテスト."""
        config: dict[str, str] = {}
        loss = ComponentFactory.create_loss(config)

        assert isinstance(loss, DiceLoss)

    def test_create_loss_unknown_raises(self) -> None:
        """未登録損失関数でエラーが発生することをテスト."""
        config = {
            "loss": "UnknownLoss",
        }
        with pytest.raises(ValueError, match="未登録の損失関数"):
            ComponentFactory.create_loss(config)

    def test_create_metrics(self) -> None:
        """評価指標生成テスト."""
        config = {
            "metrics": "SegmentationMetrics",
            "num_classes": 4,
            "device": "cpu",
        }
        metrics = ComponentFactory.create_metrics(config)

        assert isinstance(metrics, ISegmentationMetrics)
        assert isinstance(metrics, SegmentationMetrics)

    def test_create_metrics_default(self) -> None:
        """デフォルト評価指標のテスト."""
        config = {
            "num_classes": 4,
            "device": "cpu",
        }
        metrics = ComponentFactory.create_metrics(config)

        assert isinstance(metrics, SegmentationMetrics)

    def test_create_metrics_unknown_raises(self) -> None:
        """未登録評価指標でエラーが発生することをテスト."""
        config = {
            "metrics": "UnknownMetrics",
            "num_classes": 4,
        }
        with pytest.raises(ValueError, match="未登録の評価指標"):
            ComponentFactory.create_metrics(config)

    def test_create_model_missing_num_classes_raises(self) -> None:
        """num_classes がない場合にエラーが発生することをテスト."""
        config = {
            "architecture": "Unet",
        }
        with pytest.raises(KeyError):
            ComponentFactory.create_model(config)

    def test_model_forward_works(self) -> None:
        """生成したモデルで順伝播が動作することをテスト."""
        config = {
            "architecture": "Unet",
            "num_classes": 4,
            "pretrained": False,
        }
        model = ComponentFactory.create_model(config)
        model.eval()

        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 4, 256, 256)

    def test_loss_call_works(self) -> None:
        """生成した損失関数で計算が動作することをテスト."""
        config = {
            "loss": "DiceLoss",
            "loss_params": {"mode": "multiclass"},
        }
        loss = ComponentFactory.create_loss(config)

        pred = torch.randn(2, 4, 64, 64)
        target = torch.randint(0, 4, (2, 64, 64))

        result = loss(pred, target)
        assert isinstance(result, torch.Tensor)

    def test_metrics_workflow(self) -> None:
        """生成した評価指標でワークフローが動作することをテスト."""
        config = {
            "num_classes": 4,
            "device": "cpu",
        }
        metrics = ComponentFactory.create_metrics(config)

        preds = torch.randint(0, 4, (2, 64, 64))
        targets = torch.randint(0, 4, (2, 64, 64))

        metrics.update(preds, targets)
        result = metrics.compute()

        assert "mIoU" in result
        assert "Dice" in result
