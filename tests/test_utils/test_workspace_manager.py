"""PochiWorkspaceManagerのテスト."""

import tempfile
from pathlib import Path

import pytest

from pochisegmentation.utils.directory_manager import PochiWorkspaceManager


class TestPochiWorkspaceManager:
    """PochiWorkspaceManagerクラスのテスト."""

    def test_workspace_creation(self) -> None:
        """ワークスペース作成の基本テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # ワークスペースが作成されることを確認
            assert workspace.exists()
            assert workspace.is_dir()

            # ワークスペース名の形式確認（yyyymmdd_xxx）
            workspace_name = workspace.name
            assert len(workspace_name) == 12  # yyyymmdd_xxx
            assert "_" in workspace_name

    def test_subdirectories_creation(self) -> None:
        """サブディレクトリ（models, paths）の作成テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # modelsディレクトリの確認
            models_dir = workspace / "models"
            assert models_dir.exists()
            assert models_dir.is_dir()

            # pathsディレクトリの確認
            paths_dir = workspace / "paths"
            assert paths_dir.exists()
            assert paths_dir.is_dir()

    def test_get_methods(self) -> None:
        """各ディレクトリ取得メソッドのテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # get_current_workspace
            current_workspace = manager.get_current_workspace()
            assert current_workspace == workspace

            # get_models_dir
            models_dir = manager.get_models_dir()
            assert models_dir == workspace / "models"
            assert models_dir.exists()

            # get_paths_dir
            paths_dir = manager.get_paths_dir()
            assert paths_dir == workspace / "paths"
            assert paths_dir.exists()

    def test_workspace_info(self) -> None:
        """ワークスペース情報取得のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)

            # ワークスペース作成前
            info_before = manager.get_workspace_info()
            assert info_before["workspace_path"] is None
            assert info_before["exists"] is False

            # ワークスペース作成後
            workspace = manager.create_workspace()
            info_after = manager.get_workspace_info()

            assert info_after["workspace_path"] == str(workspace)
            assert info_after["models_dir"] == str(workspace / "models")
            assert info_after["paths_dir"] == str(workspace / "paths")
            assert info_after["exists"] is True
            assert info_after["date"] is not None
            assert info_after["index"] is not None

    def test_save_config(self) -> None:
        """設定ファイル保存機能のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # テスト用設定ファイルを作成
            test_config_path = Path(temp_dir) / "test_config.py"
            test_config_content = "# Test configuration\ntest_param = 'test_value'\n"
            test_config_path.write_text(test_config_content, encoding="utf-8")

            # 設定ファイルを保存
            saved_path = manager.save_config(test_config_path, "saved_config.py")

            # 保存先の確認
            expected_path = workspace / "saved_config.py"
            assert saved_path == expected_path
            assert saved_path.exists()

            # 内容の確認
            saved_content = saved_path.read_text(encoding="utf-8")
            assert saved_content == test_config_content

    def test_save_dataset_paths(self) -> None:
        """データセットパス保存機能のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            # テスト用パスリスト
            train_paths = [
                "/data/train/class1/img1.jpg",
                "/data/train/class1/img2.jpg",
                "/data/train/class2/img3.jpg",
            ]
            val_paths = ["/data/val/class1/img4.jpg", "/data/val/class2/img5.jpg"]

            # パスを保存
            train_file, val_file = manager.save_dataset_paths(train_paths, val_paths)

            # ファイルパスの確認
            expected_train_file = workspace / "paths" / "train.txt"
            expected_val_file = workspace / "paths" / "val.txt"

            assert train_file == expected_train_file
            assert val_file == expected_val_file
            assert train_file.exists()
            assert val_file.exists()

            # ファイル内容の確認
            train_content = train_file.read_text(encoding="utf-8").strip().split("\n")
            val_content = val_file.read_text(encoding="utf-8").strip().split("\n")

            assert train_content == train_paths
            assert val_content == val_paths

    def test_save_dataset_paths_train_only(self) -> None:
        """訓練データのみのパス保存テスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            manager.create_workspace()

            train_paths = ["/data/train/img1.jpg", "/data/train/img2.jpg"]

            # 検証データなしで保存
            train_file, val_file = manager.save_dataset_paths(train_paths, None)

            # 訓練ファイルのみ作成されることを確認
            assert train_file.exists()
            assert val_file is None

    def test_multiple_workspace_creation(self) -> None:
        """複数ワークスペース作成のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager1 = PochiWorkspaceManager(temp_dir)
            manager2 = PochiWorkspaceManager(temp_dir)

            workspace1 = manager1.create_workspace()
            workspace2 = manager2.create_workspace()

            # 異なるワークスペースが作成されることを確認
            assert workspace1 != workspace2
            assert workspace1.exists()
            assert workspace2.exists()

            # インデックスが増加することを確認（同じ日の場合）
            if workspace1.name[:8] == workspace2.name[:8]:  # 同じ日付
                index1 = int(workspace1.name.split("_")[1])
                index2 = int(workspace2.name.split("_")[1])
                assert index2 > index1

    def test_get_available_workspaces(self) -> None:
        """利用可能ワークスペース一覧取得のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)

            # 初期状態では空
            workspaces = manager.get_available_workspaces()
            assert len(workspaces) == 0

            # ワークスペースを作成
            workspace1 = manager.create_workspace()
            workspace2 = manager.create_workspace()

            # 作成されたワークスペースが一覧に含まれる
            workspaces = manager.get_available_workspaces()
            assert len(workspaces) == 2

            workspace_names = [ws["name"] for ws in workspaces]
            assert workspace1.name in workspace_names
            assert workspace2.name in workspace_names

    def test_error_handling_no_workspace(self) -> None:
        """ワークスペース未作成時のエラーハンドリング."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)

            # ワークスペース作成前にメソッドを呼び出すとエラー
            with pytest.raises(
                RuntimeError, match="ワークスペースが作成されていません"
            ):
                manager.get_models_dir()

            with pytest.raises(
                RuntimeError, match="ワークスペースが作成されていません"
            ):
                manager.get_paths_dir()

    def test_error_handling_missing_config(self) -> None:
        """存在しない設定ファイルのエラーハンドリング."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            manager.create_workspace()

            # 存在しないファイルを指定
            non_existent_path = Path(temp_dir) / "non_existent.py"

            with pytest.raises(FileNotFoundError, match="設定ファイルが見つかりません"):
                manager.save_config(non_existent_path)

    def test_get_visualization_dir(self) -> None:
        """可視化ディレクトリ取得のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            vis_dir = manager.get_visualization_dir()
            assert vis_dir == workspace / "visualization"
            assert vis_dir.exists()

    def test_get_visualization_dir_no_workspace(self) -> None:
        """ワークスペース未作成時の可視化ディレクトリ取得エラー."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)

            with pytest.raises(
                RuntimeError, match="ワークスペースが作成されていません"
            ):
                manager.get_visualization_dir()

    def test_save_image_list(self) -> None:
        """画像リスト保存のテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            image_paths = [
                "/data/images/img1.jpg",
                "/data/images/img2.jpg",
                "/data/images/img3.png",
            ]

            saved_path = manager.save_image_list(image_paths)

            assert saved_path == workspace / "images_list.txt"
            assert saved_path.exists()

            content = saved_path.read_text(encoding="utf-8").strip().split("\n")
            assert content == image_paths

    def test_save_image_list_custom_filename(self) -> None:
        """カスタムファイル名での画像リスト保存."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            image_paths = ["/img1.jpg", "/img2.jpg"]
            saved_path = manager.save_image_list(image_paths, "custom_list.txt")

            assert saved_path == workspace / "custom_list.txt"
            assert saved_path.exists()

    def test_save_image_list_no_workspace(self) -> None:
        """ワークスペース未作成時の画像リスト保存エラー."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)

            with pytest.raises(
                RuntimeError, match="ワークスペースが作成されていません"
            ):
                manager.save_image_list(["/img.jpg"])

    def test_save_config_no_workspace(self) -> None:
        """ワークスペース未作成時の設定ファイル保存エラー."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)
            config_path = Path(temp_dir) / "config.py"
            config_path.write_text("test = 1")

            with pytest.raises(
                RuntimeError, match="ワークスペースが作成されていません"
            ):
                manager.save_config(config_path)

    def test_save_dataset_paths_no_workspace(self) -> None:
        """ワークスペース未作成時のデータセットパス保存エラー."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)

            with pytest.raises(
                RuntimeError, match="ワークスペースが作成されていません"
            ):
                manager.save_dataset_paths(["/train.jpg"])

    def test_workspace_info_with_invalid_name(self) -> None:
        """不正な形式のワークスペース名でのinfo取得."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)

            # 不正な形式のディレクトリを手動設定
            invalid_workspace = Path(temp_dir) / "invalid_workspace"
            invalid_workspace.mkdir()
            manager.current_workspace = invalid_workspace

            info = manager.get_workspace_info()

            # パースに失敗してもエラーにならず、date/indexはNone
            assert info["workspace_path"] == str(invalid_workspace)
            assert info["date"] is None
            assert info["index"] is None

    def test_get_available_workspaces_with_invalid_dirs(self) -> None:
        """不正な形式のディレクトリを含む場合の一覧取得."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PochiWorkspaceManager(temp_dir)

            # 有効なワークスペースを作成
            valid_workspace = manager.create_workspace()

            # 不正な形式のディレクトリを作成
            invalid_dir = Path(temp_dir) / "invalid_format"
            invalid_dir.mkdir()

            workspaces = manager.get_available_workspaces()

            # 有効なワークスペースのみ取得される
            assert len(workspaces) == 1
            assert workspaces[0]["name"] == valid_workspace.name


class TestInferenceWorkspaceManager:
    """InferenceWorkspaceManagerクラスのテスト."""

    def test_init(self) -> None:
        """初期化テスト."""
        from pochisegmentation.utils.directory_manager import InferenceWorkspaceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = InferenceWorkspaceManager(temp_dir)
            assert manager.base_dir == Path(temp_dir)
            assert manager.current_workspace is None

    def test_create_workspace(self) -> None:
        """推論ワークスペース作成テスト."""
        from pochisegmentation.utils.directory_manager import InferenceWorkspaceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = InferenceWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            assert workspace.exists()
            assert workspace.is_dir()

            # modelsディレクトリは作成されない
            assert not (workspace / "models").exists()

    def test_save_model_info(self) -> None:
        """モデル情報保存テスト."""
        from pochisegmentation.utils.directory_manager import InferenceWorkspaceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = InferenceWorkspaceManager(temp_dir)
            manager.create_workspace()

            model_info = {
                "model_path": "/path/to/model.pth",
                "architecture": "Unet",
                "num_classes": 4,
            }

            saved_path = manager.save_model_info(model_info)

            assert saved_path.exists()
            assert saved_path.name == "model_info.json"

            import json

            with open(saved_path, "r", encoding="utf-8") as f:
                loaded_info = json.load(f)

            assert loaded_info == model_info

    def test_save_model_info_custom_filename(self) -> None:
        """カスタムファイル名でのモデル情報保存."""
        from pochisegmentation.utils.directory_manager import InferenceWorkspaceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = InferenceWorkspaceManager(temp_dir)
            manager.create_workspace()

            model_info = {"test": "value"}
            saved_path = manager.save_model_info(model_info, "custom_info.json")

            assert saved_path.name == "custom_info.json"
            assert saved_path.exists()

    def test_save_model_info_no_workspace(self) -> None:
        """ワークスペース未作成時のモデル情報保存エラー."""
        from pochisegmentation.utils.directory_manager import InferenceWorkspaceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = InferenceWorkspaceManager(temp_dir)

            with pytest.raises(ValueError, match="ワークスペースが作成されていません"):
                manager.save_model_info({"test": "value"})

    def test_get_csv_output_path(self) -> None:
        """CSV出力パス取得テスト."""
        from pochisegmentation.utils.directory_manager import InferenceWorkspaceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = InferenceWorkspaceManager(temp_dir)
            workspace = manager.create_workspace()

            csv_path = manager.get_csv_output_path("results.csv")

            assert csv_path == workspace / "results.csv"

    def test_get_csv_output_path_no_workspace(self) -> None:
        """ワークスペース未作成時のCSVパス取得エラー."""
        from pochisegmentation.utils.directory_manager import InferenceWorkspaceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = InferenceWorkspaceManager(temp_dir)

            with pytest.raises(ValueError, match="ワークスペースが作成されていません"):
                manager.get_csv_output_path("results.csv")

    def test_get_workspace_info(self) -> None:
        """推論ワークスペース情報取得テスト."""
        from pochisegmentation.utils.directory_manager import InferenceWorkspaceManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = InferenceWorkspaceManager(temp_dir)

            # ワークスペース作成前
            info_before = manager.get_workspace_info()
            assert info_before["workspace"] is None
            assert info_before["workspace_name"] is None
            assert info_before["exists"] is False

            # ワークスペース作成後
            workspace = manager.create_workspace()
            info_after = manager.get_workspace_info()

            assert info_after["workspace"] == str(workspace)
            assert info_after["workspace_name"] == workspace.name
            assert info_after["base_dir"] == temp_dir
            assert info_after["exists"] is True
