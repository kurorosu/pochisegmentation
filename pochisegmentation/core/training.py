"""訓練のオーケストレーション.

コンポーネントの組み立てと訓練の実行を行う.
CLIの入力方法（ファイル/対話）には依存しない.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pochisegmentation import ComponentFactory, PochiSegmentationTrainer
from pochisegmentation.datasets.voc_dataset import VOCSegmentationDataset
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.transforms.seg_transforms import (
    get_basic_train_transform,
    get_basic_val_transform,
)
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager


def resolve_device(device: str) -> str:
    """デバイスを解決.

    Args:
        device: 指定されたデバイス.

    Returns:
        利用可能なデバイス.
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger = LoggerManager().get_logger("pochiseg")
        logger.warning("CUDAが利用できません. CPUを使用します.")
        return "cpu"
    return device


def create_dataloaders(
    config: dict[str, Any],
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    list[str] | None,
]:
    """訓練/検証用 DataLoader を作成.

    Args:
        config: 設定辞書.

    Returns:
        (train_loader, val_loader, class_names) のタプル.
    """
    logger = LoggerManager().get_logger("pochiseg")

    # Transform作成
    image_size = config.get("image_size", 256)
    train_transform = get_basic_train_transform(image_size)
    val_transform = get_basic_val_transform(image_size)

    # データセット作成
    data_root = config.get("data_root", "data")
    train_split = config.get("train_split", "train")
    val_split = config.get("val_split", "val")

    logger.info(f"データセットを読み込み中: {data_root}")
    train_dataset = VOCSegmentationDataset(
        root=data_root, split=train_split, transform=train_transform
    )
    val_dataset = VOCSegmentationDataset(
        root=data_root, split=val_split, transform=val_transform
    )

    logger.info(f"訓練データ: {len(train_dataset)} 枚")
    logger.info(f"検証データ: {len(val_dataset)} 枚")

    # DataLoader作成
    batch_size = config.get("batch_size", 16)
    # num_workers=0 は必須: Ctrl+C による安全停止を有効にするため.
    # マルチプロセスワーカー使用時、シグナルがワーカーに伝播しクラッシュする.
    num_workers = 0

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # クラス名取得 (データセットから)
    class_names: list[str] | None = None
    if hasattr(train_dataset, "class_names"):
        class_names = train_dataset.class_names

    return train_loader, val_loader, class_names


def run_training(
    config: dict[str, Any],
    config_path: Path | None = None,
    config_content: str | None = None,
    stop_flag_callback: Callable[[], bool] | None = None,
) -> None:
    """訓練を実行.

    Args:
        config: 設定辞書.
        config_path: 元の設定ファイルパス (ファイル指定時).
        config_content: 設定ファイルの内容 (対話時, Python形式の文字列).
        stop_flag_callback: 停止フラグをチェックするコールバック関数.
    """
    logger = LoggerManager().get_logger("pochiseg")

    # ワークスペース作成
    workspace_manager = PochiWorkspaceManager(
        base_dir=config.get("work_dir", "work_dirs")
    )
    workspace_path = workspace_manager.create_workspace()
    logger.info(f"ワークスペース作成: {workspace_path}")

    # 設定ファイル保存
    if config_path:
        workspace_manager.save_config(config_path)
    elif config_content:
        # 対話時は Python 形式で保存
        saved_config_path = Path(workspace_path) / "config.py"
        saved_config_path.write_text(config_content, encoding="utf-8")
        logger.info(f"設定ファイルを保存: {saved_config_path}")

    # デバイス解決
    device = resolve_device(config.get("device", "cuda"))

    # モデル作成
    logger.info("モデルを作成中...")
    model = ComponentFactory.create_model(config)

    # 損失関数作成
    criterion = ComponentFactory.create_loss(config)

    # DataLoader作成 (class_names も取得)
    train_loader, val_loader, class_names = create_dataloaders(config)

    # 評価指標作成 (class_names を渡す)
    metrics = ComponentFactory.create_metrics(config, class_names)

    # オプティマイザ作成
    optimizer = ComponentFactory.create_optimizer(model, config)

    # スケジューラ作成
    scheduler = ComponentFactory.create_scheduler(optimizer, config)

    # 画像パスをpathsディレクトリに保存
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    if hasattr(train_dataset, "get_image_paths") and hasattr(
        val_dataset, "get_image_paths"
    ):
        train_paths = [str(p) for p in train_dataset.get_image_paths()]  # type: ignore
        val_paths = [str(p) for p in val_dataset.get_image_paths()]  # type: ignore
        workspace_manager.save_dataset_paths(train_paths, val_paths)
        logger.info("画像パスを保存しました")

    # トレーナー作成
    trainer = PochiSegmentationTrainer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        workspace_manager=workspace_manager,
        early_stopping_patience=config.get("early_stopping_patience"),
        enable_amp=config.get("enable_amp", False),
    )

    # 訓練実行
    epochs = config.get("epochs", 100)
    logger.info(f"訓練開始: {epochs} エポック")
    trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        stop_flag_callback=stop_flag_callback,
    )

    logger.info("訓練完了!")
    logger.info(f"結果: {workspace_path}")
