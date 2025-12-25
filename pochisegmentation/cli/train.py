"""セグメンテーション訓練コマンドの実装."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pochisegmentation import (
    ComponentFactory,
    PochiSegmentationTrainer,
)
from pochisegmentation.datasets.voc_dataset import VOCSegmentationDataset
from pochisegmentation.exceptions import PochiConfigError
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.transforms.seg_transforms import (
    get_basic_train_transform,
    get_basic_val_transform,
)
from pochisegmentation.utils.config_loader import ConfigLoader
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager


def seg_train(args: argparse.Namespace) -> None:
    """セグメンテーション訓練を実行.

    Args:
        args: コマンドライン引数.
    """
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger("pochi")

    logger.info(f"設定ファイルを読み込み: {args.config}")
    try:
        config = ConfigLoader.load(args.config)
    except PochiConfigError as e:
        logger.error(f"設定エラー: {e}")
        sys.exit(1)

    # ワークスペース作成
    workspace_manager = PochiWorkspaceManager(
        base_dir=config.get("work_dir", "work_dirs")
    )
    workspace_path = workspace_manager.create_workspace()
    logger.info(f"ワークスペース作成: {workspace_path}")

    # 設定ファイルをコピー
    workspace_manager.save_config(Path(args.config))

    # デバイス
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDAが利用できません. CPUを使用します.")
        device = "cpu"

    # モデル作成
    logger.info("モデルを作成中...")
    model = ComponentFactory.create_model(config)

    # 損失関数作成
    criterion = ComponentFactory.create_loss(config)

    # 評価指標作成
    metrics = ComponentFactory.create_metrics(config)

    # オプティマイザ作成
    optimizer = ComponentFactory.create_optimizer(model, config)

    # スケジューラ作成
    scheduler = ComponentFactory.create_scheduler(optimizer, config)

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

    # 画像パスをpathsディレクトリに保存
    train_paths = [str(p) for p in train_dataset.get_image_paths()]
    val_paths = [str(p) for p in val_dataset.get_image_paths()]
    workspace_manager.save_dataset_paths(train_paths, val_paths)
    logger.info("画像パスを保存しました")

    # DataLoader作成
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

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
    )

    # 訓練実行
    epochs = config.get("epochs", 100)
    logger.info(f"訓練開始: {epochs} エポック")
    trainer.train(train_loader, val_loader, epochs=epochs)

    # 最終モデル保存
    trainer.save_last_model()

    logger.info("訓練完了!")
    logger.info(f"結果: {workspace_path}")
