"""cli/commands/infer.py の推論コマンドテスト."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pochisegmentation.cli.commands.infer import infer_command


class TestInferCommand:
    """infer_command関数のテスト."""

    @pytest.fixture
    def mock_config_content(self) -> str:
        """テスト用設定ファイルの内容."""
        return """
data_root = "data/train"
architecture = "Unet"
encoder_name = "resnet34"
num_classes = 4
learning_rate = 0.001
device = "cpu"
image_size = 256
"""

    @pytest.fixture
    def setup_test_environment(
        self, tmp_path: Path, mock_config_content: str
    ) -> tuple[Path, Path, Path]:
        """テスト環境のセットアップ."""
        # ディレクトリ構造を作成
        work_dir = tmp_path / "work_dirs" / "20241225_001"
        models_dir = work_dir / "models"
        models_dir.mkdir(parents=True)

        # 設定ファイル
        config_path = work_dir / "config.py"
        config_path.write_text(mock_config_content)

        # ダミーモデルファイル
        model_path = models_dir / "best.pth"
        torch.save({"model_state_dict": {}}, model_path)

        # テスト画像
        test_image_dir = tmp_path / "test_images"
        test_image_dir.mkdir()

        return model_path, test_image_dir, work_dir

    def test_config_not_found(self, tmp_path: Path) -> None:
        """設定ファイルが見つからない場合のエラー."""
        # モデルファイルのみ存在する (設定ファイルなし)
        model_path = tmp_path / "work_dirs" / "test" / "models" / "best.pth"
        model_path.parent.mkdir(parents=True)
        torch.save({}, model_path)

        args = argparse.Namespace(
            model_path=str(model_path),
            data=str(tmp_path),
            output="",
            device="cpu",
        )

        with pytest.raises(FileNotFoundError):
            infer_command(args)

    @patch("pochisegmentation.core.inference.PochiSegmentationPredictor")
    @patch("pochisegmentation.core.inference.ComponentFactory")
    @patch("pochisegmentation.core.inference.cv2")
    def test_successful_inference_single_image(
        self,
        mock_cv2: MagicMock,
        mock_component_factory: MagicMock,
        mock_predictor_class: MagicMock,
        tmp_path: Path,
        mock_config_content: str,
    ) -> None:
        """単一画像の正常な推論テスト."""
        # テスト環境のセットアップ
        work_dir = tmp_path / "work_dirs" / "20241225_001"
        models_dir = work_dir / "models"
        models_dir.mkdir(parents=True)

        config_path = work_dir / "config.py"
        config_path.write_text(mock_config_content)

        model_path = models_dir / "best.pth"
        torch.save({"model_state_dict": {}}, model_path)

        test_image_path = tmp_path / "test.jpg"
        test_image_path.touch()

        # モックの設定
        mock_component_factory.create_model.return_value = MagicMock()

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.zeros((256, 256), dtype=np.uint8)
        mock_predictor_class.from_checkpoint.return_value = mock_predictor

        mock_cv2.imread.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((256, 256, 3), dtype=np.uint8)

        args = argparse.Namespace(
            model_path=str(model_path),
            data=str(test_image_path),
            output="",
            device="cpu",
        )

        infer_command(args)

        # 推論が呼び出されたことを確認
        mock_predictor.predict.assert_called_once()

    @patch("pochisegmentation.core.inference.PochiSegmentationPredictor")
    @patch("pochisegmentation.core.inference.ComponentFactory")
    @patch("pochisegmentation.core.inference.cv2")
    def test_successful_inference_directory(
        self,
        mock_cv2: MagicMock,
        mock_component_factory: MagicMock,
        mock_predictor_class: MagicMock,
        tmp_path: Path,
        mock_config_content: str,
    ) -> None:
        """ディレクトリ内画像の推論テスト."""
        # テスト環境のセットアップ
        work_dir = tmp_path / "work_dirs" / "20241225_001"
        models_dir = work_dir / "models"
        models_dir.mkdir(parents=True)

        config_path = work_dir / "config.py"
        config_path.write_text(mock_config_content)

        model_path = models_dir / "best.pth"
        torch.save({"model_state_dict": {}}, model_path)

        # 複数のテスト画像
        test_image_dir = tmp_path / "test_images"
        test_image_dir.mkdir()
        (test_image_dir / "img1.jpg").touch()
        (test_image_dir / "img2.png").touch()
        (test_image_dir / "img3.bmp").touch()

        # モックの設定
        mock_component_factory.create_model.return_value = MagicMock()

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.zeros((256, 256), dtype=np.uint8)
        mock_predictor_class.from_checkpoint.return_value = mock_predictor

        mock_cv2.imread.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((256, 256, 3), dtype=np.uint8)

        args = argparse.Namespace(
            model_path=str(model_path),
            data=str(test_image_dir),
            output="",
            device="cpu",
        )

        infer_command(args)

        # 3枚分の推論が呼び出されたことを確認
        assert mock_predictor.predict.call_count == 3

    @patch("pochisegmentation.core.inference.PochiSegmentationPredictor")
    @patch("pochisegmentation.core.inference.ComponentFactory")
    @patch("pochisegmentation.core.inference.cv2")
    def test_successful_inference_path_list(
        self,
        mock_cv2: MagicMock,
        mock_component_factory: MagicMock,
        mock_predictor_class: MagicMock,
        tmp_path: Path,
        mock_config_content: str,
    ) -> None:
        """パスリストファイルからの推論テスト."""
        # テスト環境のセットアップ
        work_dir = tmp_path / "work_dirs" / "20241225_001"
        models_dir = work_dir / "models"
        models_dir.mkdir(parents=True)

        config_path = work_dir / "config.py"
        config_path.write_text(mock_config_content)

        model_path = models_dir / "best.pth"
        torch.save({"model_state_dict": {}}, model_path)

        # パスリストファイル
        test_images_dir = tmp_path / "images"
        test_images_dir.mkdir()
        (test_images_dir / "img1.jpg").touch()
        (test_images_dir / "img2.jpg").touch()

        path_list = tmp_path / "paths.txt"
        path_list.write_text(
            f"{test_images_dir / 'img1.jpg'}\n{test_images_dir / 'img2.jpg'}\n"
        )

        # モックの設定
        mock_component_factory.create_model.return_value = MagicMock()

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.zeros((256, 256), dtype=np.uint8)
        mock_predictor_class.from_checkpoint.return_value = mock_predictor

        mock_cv2.imread.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((256, 256, 3), dtype=np.uint8)

        args = argparse.Namespace(
            model_path=str(model_path),
            data=str(path_list),
            output="",
            device="cpu",
        )

        infer_command(args)

        # 2枚分の推論が呼び出されたことを確認
        assert mock_predictor.predict.call_count == 2

    @patch("pochisegmentation.core.inference.PochiSegmentationPredictor")
    @patch("pochisegmentation.core.inference.ComponentFactory")
    @patch("pochisegmentation.core.inference.cv2")
    def test_custom_output_directory(
        self,
        mock_cv2: MagicMock,
        mock_component_factory: MagicMock,
        mock_predictor_class: MagicMock,
        tmp_path: Path,
        mock_config_content: str,
    ) -> None:
        """カスタム出力ディレクトリの指定テスト."""
        # テスト環境のセットアップ
        work_dir = tmp_path / "work_dirs" / "20241225_001"
        models_dir = work_dir / "models"
        models_dir.mkdir(parents=True)

        config_path = work_dir / "config.py"
        config_path.write_text(mock_config_content)

        model_path = models_dir / "best.pth"
        torch.save({"model_state_dict": {}}, model_path)

        test_image_path = tmp_path / "test.jpg"
        test_image_path.touch()

        custom_output = tmp_path / "custom_output"

        # モックの設定
        mock_component_factory.create_model.return_value = MagicMock()

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.zeros((256, 256), dtype=np.uint8)
        mock_predictor_class.from_checkpoint.return_value = mock_predictor

        mock_cv2.imread.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((256, 256, 3), dtype=np.uint8)

        args = argparse.Namespace(
            model_path=str(model_path),
            data=str(test_image_path),
            output=str(custom_output),
            device="cpu",
        )

        infer_command(args)

        # カスタム出力ディレクトリが作成されたことを確認
        assert custom_output.exists()

    @patch("pochisegmentation.core.inference.torch.cuda.is_available")
    @patch("pochisegmentation.core.inference.PochiSegmentationPredictor")
    @patch("pochisegmentation.core.inference.ComponentFactory")
    @patch("pochisegmentation.core.inference.cv2")
    def test_cuda_fallback_to_cpu(
        self,
        mock_cv2: MagicMock,
        mock_component_factory: MagicMock,
        mock_predictor_class: MagicMock,
        mock_cuda_available: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CUDA非対応時のCPUフォールバックテスト."""
        mock_cuda_available.return_value = False

        # テスト環境のセットアップ
        work_dir = tmp_path / "work_dirs" / "20241225_001"
        models_dir = work_dir / "models"
        models_dir.mkdir(parents=True)

        config_path = work_dir / "config.py"
        config_path.write_text(
            """
data_root = "data"
num_classes = 4
device = "cuda"
"""
        )

        model_path = models_dir / "best.pth"
        torch.save({"model_state_dict": {}}, model_path)

        test_image_path = tmp_path / "test.jpg"
        test_image_path.touch()

        # モックの設定
        mock_component_factory.create_model.return_value = MagicMock()

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.zeros((256, 256), dtype=np.uint8)
        mock_predictor_class.from_checkpoint.return_value = mock_predictor

        mock_cv2.imread.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((256, 256, 3), dtype=np.uint8)

        args = argparse.Namespace(
            model_path=str(model_path),
            data=str(test_image_path),
            output="",
            device="cuda",
        )

        infer_command(args)

        # from_checkpointにdevice='cpu'が渡されることを確認
        call_kwargs = mock_predictor_class.from_checkpoint.call_args[1]
        assert call_kwargs["device"] == "cpu"

    def test_config_error_handling(self, tmp_path: Path) -> None:
        """設定ファイルの読み込みエラーハンドリング."""
        # テスト環境のセットアップ
        work_dir = tmp_path / "work_dirs" / "20241225_001"
        models_dir = work_dir / "models"
        models_dir.mkdir(parents=True)

        # 不正な設定ファイル
        config_path = work_dir / "config.py"
        config_path.write_text("invalid = 'missing required keys'")

        model_path = models_dir / "best.pth"
        torch.save({"model_state_dict": {}}, model_path)

        args = argparse.Namespace(
            model_path=str(model_path),
            data=str(tmp_path),
            output="",
            device="cpu",
        )

        with pytest.raises(Exception):
            infer_command(args)
