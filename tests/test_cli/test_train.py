"""cli/train.py のテスト."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from pochisegmentation.cli.train import seg_train
from pochisegmentation.exceptions import PochiConfigError


class TestSegTrain:
    """seg_train関数のテスト."""

    @pytest.fixture
    def mock_config_content(self) -> str:
        """テスト用設定ファイルの内容."""
        return """
data_root = "data/train"
architecture = "Unet"
encoder_name = "resnet34"
num_classes = 4
learning_rate = 0.001
batch_size = 2
epochs = 1
device = "cpu"
"""

    @pytest.fixture
    def mock_args(self, tmp_path: Path, mock_config_content: str) -> argparse.Namespace:
        """テスト用引数."""
        config_path = tmp_path / "config.py"
        config_path.write_text(mock_config_content)
        return argparse.Namespace(config=str(config_path))

    def test_config_error_handling(self, tmp_path: Path) -> None:
        """設定エラー時のハンドリング."""
        # 存在しない設定ファイル
        args = argparse.Namespace(config=str(tmp_path / "nonexistent.py"))

        with pytest.raises(SystemExit) as exc_info:
            seg_train(args)

        assert exc_info.value.code == 1

    def test_config_invalid_content(self, tmp_path: Path) -> None:
        """不正な設定内容のハンドリング."""
        config_path = tmp_path / "invalid_config.py"
        # 必須キーが不足
        config_path.write_text("architecture = 'Unet'\n")

        args = argparse.Namespace(config=str(config_path))

        with pytest.raises(SystemExit) as exc_info:
            seg_train(args)

        assert exc_info.value.code == 1

    @patch("pochisegmentation.cli.train.PochiSegmentationTrainer")
    @patch("pochisegmentation.cli.train.DataLoader")
    @patch("pochisegmentation.cli.train.VOCSegmentationDataset")
    @patch("pochisegmentation.cli.train.ComponentFactory")
    @patch("pochisegmentation.cli.train.PochiWorkspaceManager")
    def test_successful_training_flow(
        self,
        mock_workspace_manager: MagicMock,
        mock_component_factory: MagicMock,
        mock_voc_dataset: MagicMock,
        mock_dataloader: MagicMock,
        mock_trainer: MagicMock,
        tmp_path: Path,
    ) -> None:
        """正常な訓練フローのテスト."""
        # 設定ファイル作成
        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
architecture = "Unet"
encoder_name = "resnet34"
learning_rate = 0.001
device = "cpu"
"""
        )

        # モックの設定
        mock_workspace_instance = MagicMock()
        mock_workspace_instance.create_workspace.return_value = tmp_path / "work_dirs"
        mock_workspace_manager.return_value = mock_workspace_instance

        mock_model = MagicMock()
        mock_component_factory.create_model.return_value = mock_model
        mock_component_factory.create_loss.return_value = MagicMock()
        mock_component_factory.create_metrics.return_value = MagicMock()
        mock_component_factory.create_optimizer.return_value = MagicMock()
        mock_component_factory.create_scheduler.return_value = None

        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__ = MagicMock(return_value=10)
        mock_train_dataset.get_image_paths.return_value = [Path("/img1.jpg")]

        mock_val_dataset = MagicMock()
        mock_val_dataset.__len__ = MagicMock(return_value=5)
        mock_val_dataset.get_image_paths.return_value = [Path("/img2.jpg")]

        mock_voc_dataset.side_effect = [mock_train_dataset, mock_val_dataset]

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        args = argparse.Namespace(config=str(config_path))
        seg_train(args)

        # 各コンポーネントが呼び出されたことを確認
        mock_workspace_manager.assert_called_once()
        mock_component_factory.create_model.assert_called_once()
        mock_component_factory.create_loss.assert_called_once()
        mock_component_factory.create_metrics.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_last_model.assert_called_once()

    @patch("pochisegmentation.cli.train.torch.cuda.is_available")
    @patch("pochisegmentation.cli.train.PochiSegmentationTrainer")
    @patch("pochisegmentation.cli.train.DataLoader")
    @patch("pochisegmentation.cli.train.VOCSegmentationDataset")
    @patch("pochisegmentation.cli.train.ComponentFactory")
    @patch("pochisegmentation.cli.train.PochiWorkspaceManager")
    def test_cuda_fallback_to_cpu(
        self,
        mock_workspace_manager: MagicMock,
        mock_component_factory: MagicMock,
        mock_voc_dataset: MagicMock,
        mock_dataloader: MagicMock,
        mock_trainer: MagicMock,
        mock_cuda_available: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CUDA非対応時のCPUフォールバックテスト."""
        mock_cuda_available.return_value = False

        config_path = tmp_path / "config.py"
        config_path.write_text(
            """
data_root = "data/train"
num_classes = 4
device = "cuda"
"""
        )

        # モックの設定
        mock_workspace_instance = MagicMock()
        mock_workspace_instance.create_workspace.return_value = tmp_path / "work_dirs"
        mock_workspace_manager.return_value = mock_workspace_instance

        mock_component_factory.create_model.return_value = MagicMock()
        mock_component_factory.create_loss.return_value = MagicMock()
        mock_component_factory.create_metrics.return_value = MagicMock()
        mock_component_factory.create_optimizer.return_value = MagicMock()
        mock_component_factory.create_scheduler.return_value = None

        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__ = MagicMock(return_value=10)
        mock_train_dataset.get_image_paths.return_value = []

        mock_val_dataset = MagicMock()
        mock_val_dataset.__len__ = MagicMock(return_value=5)
        mock_val_dataset.get_image_paths.return_value = []

        mock_voc_dataset.side_effect = [mock_train_dataset, mock_val_dataset]

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        args = argparse.Namespace(config=str(config_path))
        seg_train(args)

        # Trainerにdevice='cpu'が渡されることを確認
        call_kwargs = mock_trainer.call_args[1]
        assert call_kwargs["device"] == "cpu"
