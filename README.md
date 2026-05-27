# pochisegmentation

![version](https://img.shields.io/badge/version-1.0.0-blue)
![python](https://img.shields.io/badge/python-3.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A tiny but clever semantic segmentation pipeline for images — as friendly as Pochi!

## 特徴

- **対話モード** - 設定ファイル不要で訓練・推論を開始可能
- **2種類のモデル** - Unet / DeepLabV3+ (軽量・高速 vs 高精度)
- **Early Stopping** - 過学習を自動検知して訓練を停止
- **AMP (混合精度訓練)** - GPU メモリ削減・高速化
- **豊富な可視化** - 訓練履歴グラフ, Confusion Matrix, クラス別 IoU

## 動作環境

- Python 3.13
- CUDA 12.x (GPU 使用時)

## インストール

```bash
git clone https://github.com/kurorosu/pochisegmentation.git
cd pochisegmentation

# uv で依存関係インストール
uv sync
```

## クイックスタート

### 対話モードで訓練 (推奨)

設定ファイル不要で、質問に答えるだけで訓練を開始できます。

```bash
uv run pochi train
```

質問例:
```
? データセットのパス: data/my_dataset
? クラス数 (背景含む): 4
? アーキテクチャ: Unet
? エンコーダー: resnet34
? 損失関数: DiceLoss
? エポック数: 100
...
```

### 対話モードで推論

```bash
uv run pochi infer
```

### 設定ファイルを使用する場合

```bash
# 訓練
uv run pochi train --config configs/pochi_seg_config.py

# 推論
uv run pochi infer --model work_dirs/20251225_001/models/best.pth --data path/to/images/
```

## コマンド一覧

| コマンド | 説明 |
|---------|------|
| `uv run pochi train` | 対話モードで訓練 |
| `uv run pochi train --config <path>` | 設定ファイルで訓練 |
| `uv run pochi infer` | 対話モードで推論 |
| `uv run pochi infer --model <path> --data <path>` | 引数指定で推論 |

## 出力ファイル

訓練結果は `work_dirs/YYYYMMDD_XXX/` に保存されます:

```
work_dirs/20251225_001/
├── config.py                    # 設定ファイル
├── models/
│   ├── best.pth                 # 最良モデル (mIoU 基準)
│   └── latest.pth               # 最新モデル
├── logs/
│   └── training.log             # 訓練ログ
├── visualization/
│   ├── training_history.csv     # 訓練履歴
│   ├── loss.png                 # 損失グラフ
│   ├── metrics.png              # mIoU/Dice グラフ
│   ├── learning_rate.png        # 学習率グラフ
│   ├── class_iou.png            # クラス別 IoU 棒グラフ
│   ├── confusion_matrix.png     # Confusion Matrix
│   └── class_metrics.csv        # クラス別精度
└── paths/
    ├── train.txt                # 訓練画像パス
    └── val.txt                  # 検証画像パス
```

## データセット形式

VOC 形式のディレクトリ構造:

```
dataset/
├── JPEGImages/              # 入力画像 (.jpg, .png, .bmp)
├── SegmentationClass/       # マスク画像 (.png, グレースケール)
├── ImageSets/
│   └── Segmentation/
│       ├── train.txt        # 訓練用画像 ID
│       └── val.txt          # 検証用画像 ID
└── class_names.txt          # クラス名リスト (オプション)
```

### マスク画像

- 形式: グレースケール PNG
- ピクセル値: クラスインデックス (0 = 背景, 1 = クラス1, ...)

## 設定項目

### モデル

| 項目 | 説明 | 選択肢 |
|-----|------|--------|
| `architecture` | モデル | `Unet`, `DeepLabV3Plus` |
| `encoder_name` | エンコーダー | `resnet18`, `resnet34`, `resnet50`, `efficientnet-b0` など |
| `pretrained` | 事前学習済み重み | `True`, `False` |

### 訓練

| 項目 | 説明 | デフォルト |
|-----|------|-----------|
| `epochs` | エポック数 | 100 |
| `batch_size` | バッチサイズ | 16 |
| `learning_rate` | 学習率 | 1e-3 |
| `optimizer` | オプティマイザ | `AdamW` |
| `scheduler` | スケジューラ | `CosineAnnealingLR` |

### 損失関数

| 項目 | 説明 |
|-----|------|
| `DiceLoss` | クラス不均衡に強い (推奨) |
| `FocalLoss` | 難しいサンプルを重視 |
| `JaccardLoss` | IoU 直接最適化 |

### 高度な設定

| 項目 | 説明 |
|-----|------|
| `early_stopping_patience` | N エポック改善なしで停止 (None で無効) |
| `enable_amp` | 混合精度訓練 (GPU メモリ削減) |
| `enable_layer_wise_lr` | 層別学習率 (エンコーダーを低学習率に) |

## モデル比較

| モデル | 特徴 | 用途 |
|--------|------|------|
| **Unet** | 軽量・高速, スキップ接続で細部保持 | 汎用, 小規模データ |
| **DeepLabV3+** | 高精度, マルチスケール対応 | 高精度が必要な場面 |

## ドキュメント

- [ユーザーガイド](docs/user-guide.md) - 詳細な使用方法

## ライセンス

MIT License
