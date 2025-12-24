r"""pochisegmentation CLIエントリーポイント.

セグメンテーションモデルの訓練と推論を行うCLIツール.

Usage:
    # 訓練
    python pochi.py seg-train --config configs/pochi_seg_config.py

    # 推論
    python pochi.py seg-infer --model-path work_dirs/xxx/models/best.pth \
        --data data/test/images --output results/
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from pochisegmentation import (
    ComponentFactory,
    PochiSegmentationPredictor,
    PochiSegmentationTrainer,
)
from pochisegmentation.datasets.voc_dataset import VOCSegmentationDataset
from pochisegmentation.logging.logger_manager import LoggerManager
from pochisegmentation.transforms.seg_transforms import (
    SegmentationTransform,
    get_basic_train_transform,
    get_basic_val_transform,
)
from pochisegmentation.utils.directory_manager import PochiWorkspaceManager
from pochisegmentation.utils.layer_wise_lr import create_layer_wise_param_groups
from pochisegmentation.utils.timestamp_utils import (
    find_next_index,
    format_workspace_name,
    get_current_date_str,
)
from pochisegmentation.visualization.mask_visualizer import (
    colorize_mask,
    overlay_mask_on_image,
)


def load_config(config_path: str) -> dict[str, Any]:
    """設定ファイルを読み込む.

    Args:
        config_path: 設定ファイルのパス.

    Returns:
        設定辞書.

    Raises:
        FileNotFoundError: 設定ファイルが存在しない場合.
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path_obj)
    if spec is None or spec.loader is None:
        raise ValueError(f"設定ファイルの読み込みに失敗しました: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # モジュールの属性を辞書に変換
    config: dict[str, Any] = {}
    for key in dir(config_module):
        if not key.startswith("_"):
            config[key] = getattr(config_module, key)

    return config


def create_optimizer(
    model: torch.nn.Module, config: dict[str, Any]
) -> torch.optim.Optimizer:
    """オプティマイザを作成.

    Args:
        model: モデル.
        config: 設定辞書.

    Returns:
        オプティマイザ.
    """
    # 層別学習率
    if config.get("enable_layer_wise_lr", False):
        param_groups = create_layer_wise_param_groups(
            model,  # type: ignore
            encoder_lr=config.get("encoder_lr", 1e-4),
            decoder_lr=config.get("decoder_lr", 1e-3),
        )
        # 層別学習率の場合, lrはparam_groupsで設定済み
        lr = config.get("decoder_lr", 1e-3)
    else:
        param_groups = model.parameters()  # type: ignore
        lr = config.get("learning_rate", 1e-3)

    optimizer_name = config.get("optimizer", "AdamW")

    if optimizer_name == "Adam":
        return torch.optim.Adam(param_groups, lr=lr)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(param_groups, lr=lr)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """スケジューラを作成.

    Args:
        optimizer: オプティマイザ.
        config: 設定辞書.

    Returns:
        スケジューラ, 設定がない場合はNone.
    """
    scheduler_name = config.get("scheduler")
    if scheduler_name is None:
        return None

    scheduler_params = config.get("scheduler_params", {})

    if scheduler_name == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_name == "StepLR":
        return StepLR(optimizer, **scheduler_params)
    elif scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def seg_train(args: argparse.Namespace) -> None:
    """セグメンテーション訓練を実行.

    Args:
        args: コマンドライン引数.
    """
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger("pochi")

    logger.info(f"設定ファイルを読み込み: {args.config}")
    config = load_config(args.config)

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
    optimizer = create_optimizer(model, config)

    # スケジューラ作成
    scheduler = create_scheduler(optimizer, config)

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


def seg_infer(args: argparse.Namespace) -> None:
    """セグメンテーション推論を実行.

    Args:
        args: コマンドライン引数.
    """
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger("pochi")

    logger.info(f"モデル読み込み: {args.model_path}")

    # 設定ファイル読み込み (モデルパスと同じディレクトリにあると仮定)
    model_path = Path(args.model_path)
    config_path = model_path.parent.parent / "config.py"

    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        logger.warning("デフォルト設定を使用します.")
        config = {
            "architecture": "Unet",
            "encoder_name": "resnet34",
            "num_classes": 4,
            "image_size": 256,
        }

    # デバイス
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDAが利用できません. CPUを使用します.")
        device = "cpu"

    # モデル作成
    model = ComponentFactory.create_model(config)

    # Transform作成
    image_size = config.get("image_size", 256)
    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Predictor作成
    predictor = PochiSegmentationPredictor.from_checkpoint(
        checkpoint_path=args.model_path,
        model=model,
        transform=transform,
        device=device,
    )

    # 出力ディレクトリ作成 (work_dir方式)
    # モデルパスから work_dir を特定: work_dirs/xxx/models/best.pth -> work_dirs/xxx/
    work_dir = model_path.parent.parent
    if args.output:
        # --output が指定されている場合はそれを使用
        output_dir = Path(args.output)
    else:
        # work_dir/predictions/yyyymmdd_xxx/ を使用
        predictions_base = work_dir / "predictions"
        predictions_base.mkdir(parents=True, exist_ok=True)
        date_str = get_current_date_str()
        next_index = find_next_index(predictions_base, date_str)
        output_dir = predictions_base / format_workspace_name(date_str, next_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"出力ディレクトリ: {output_dir}")

    # 推論実行
    import cv2

    data_path = Path(args.data)

    if data_path.is_file() and data_path.suffix == ".txt":
        # テキストファイル: パスリストとして読み込み
        logger.info(f"パスリストファイルを読み込み: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            image_paths = [Path(line.strip()) for line in f if line.strip()]
    elif data_path.is_file():
        # 単一画像ファイル
        image_paths = [data_path]
    else:
        # ディレクトリ
        image_paths = (
            list(data_path.glob("*.jpg"))
            + list(data_path.glob("*.png"))
            + list(data_path.glob("*.bmp"))
        )

    logger.info(f"推論対象: {len(image_paths)} 枚")

    # クラス数を取得
    num_classes = config.get("num_classes", 2)

    for image_path in image_paths:
        logger.info(f"推論中: {image_path}")

        # 元画像を読み込み (RGB)
        original_image = cv2.imread(str(image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 推論
        mask = predictor.predict(image_path)

        # 1. カラーマスク単体を保存
        color_mask = colorize_mask(mask, num_classes=num_classes)
        mask_output_path = output_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_output_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        logger.info(f"マスク保存: {mask_output_path}")

        # 2. 元画像にオーバーレイした画像を保存
        overlay = overlay_mask_on_image(
            original_image, mask, alpha=0.5, num_classes=num_classes
        )
        vis_output_path = output_dir / f"{image_path.stem}_vis.png"
        cv2.imwrite(str(vis_output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logger.info(f"オーバーレイ保存: {vis_output_path}")

    logger.info("推論完了!")


def main() -> None:
    """メインエントリーポイント."""
    parser = argparse.ArgumentParser(
        description="pochisegmentation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")

    # seg-train サブコマンド
    train_parser = subparsers.add_parser("seg-train", help="セグメンテーション訓練")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="設定ファイルのパス",
    )

    # seg-infer サブコマンド
    infer_parser = subparsers.add_parser("seg-infer", help="セグメンテーション推論")
    infer_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="モデルファイルのパス",
    )
    infer_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="入力画像, ディレクトリ, またはパスリスト(.txt)のパス",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        default="",
        help="出力ディレクトリのパス (省略時: work_dir/predictions/)",
    )
    infer_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="使用デバイス (default: cuda)",
    )

    args = parser.parse_args()

    if args.command == "seg-train":
        seg_train(args)
    elif args.command == "seg-infer":
        seg_infer(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
