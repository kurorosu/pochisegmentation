# pochisegmentation

A tiny but clever semantic segmentation pipeline for images — as friendly as Pochi!

pochitrain の設計思想を継承したセグメンテーション版フレームワーク.
SOLID原則 (特にDIP: 依存性逆転) に基づくインターフェース駆動設計.

## 特徴

- **Simplicity First** - 最小限の設定で訓練開始可能
- **SOLID原則** - DI (依存性注入) と DIP (依存性逆転) を重視
- **2種類のモデル** - Unet / DeepLabV3+ (軽量・高速 vs 高精度)
- **豊富な可視化** - 訓練履歴グラフ, 推論マスク, オーバーレイ画像

## 技術スタック

| カテゴリ | ライブラリ |
|---------|-----------|
| モデル | [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) |
| 損失関数 | smp.losses (Dice, Focal, Jaccard) |
| 評価指標 | [torchmetrics](https://torchmetrics.readthedocs.io/) (mIoU, Dice) |
| Transform | torchvision.transforms.v2 (tv_tensors) |
| 画像処理 | opencv-python |

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/kurorosu/pochisegmentation.git
cd pochisegmentation

# uv で仮想環境作成と依存関係インストール
uv sync

# 開発用依存関係も含める
uv sync --group dev

# pre-commit フックのインストール
uv run pre-commit install
```

## クイックスタート

### 1. 設定ファイルの準備

`configs/pochi_seg_config.py` を編集してデータパスを設定:

```python
# データ設定
data_root = "path/to/your/dataset"  # VOC形式のデータセット
num_classes = 4                      # クラス数 (背景含む)

# モデル設定
architecture = "Unet"                # "Unet" or "DeepLabV3Plus"
encoder_name = "resnet34"            # エンコーダー

# 訓練設定
epochs = 100
batch_size = 16
learning_rate = 1e-3
```

### 2. 訓練

```bash
python pochi.py seg-train --config configs/pochi_seg_config.py
```

訓練結果は `work_dirs/YYYYMMDD_XXX/` に保存されます:

```
work_dirs/20251225_001/
├── config.py              # 使用した設定ファイルのコピー
├── models/
│   ├── best.pth           # 最良モデル
│   └── latest.pth         # 最新モデル
├── logs/
│   └── training.log       # 訓練ログ
├── visualization/
│   ├── training_history.csv
│   ├── loss.png           # 損失グラフ
│   ├── metrics.png        # mIoU/Diceグラフ
│   └── learning_rate.png  # 学習率グラフ
└── paths/
    ├── train.txt          # 訓練画像パスリスト
    └── val.txt            # 検証画像パスリスト
```

### 3. 推論

```bash
# 単一画像
python pochi.py seg-infer \
  --config configs/pochi_seg_config.py \
  --model work_dirs/20251225_001/models/best.pth \
  --data path/to/image.jpg

# ディレクトリ内の全画像
python pochi.py seg-infer \
  --config configs/pochi_seg_config.py \
  --model work_dirs/20251225_001/models/best.pth \
  --data path/to/images/

# パスリストファイル (.txt)
python pochi.py seg-infer \
  --config configs/pochi_seg_config.py \
  --model work_dirs/20251225_001/models/best.pth \
  --data path/to/image_list.txt
```

推論結果は `work_dirs/predictions/YYYYMMDD_XXX/` に保存されます:

```
predictions/20251225_001/
├── image1_mask.png    # カラーマスク
├── image1_vis.png     # 元画像にオーバーレイ
├── image2_mask.png
└── image2_vis.png
```

## データセット形式

VOC形式のディレクトリ構造:

```
dataset/
├── JPEGImages/           # 画像ファイル (.jpg, .png, .bmp)
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── SegmentationClass/    # マスク画像 (.png, グレースケール)
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── ImageSets/
│   └── Segmentation/
│       ├── train.txt     # 訓練用画像ID
│       └── val.txt       # 検証用画像ID
└── class_names.txt       # クラス名 (オプション)
```

### class_names.txt の例

```
background
dog
cat
person
```

## モデル

| モデル | 特徴 | 用途 |
|--------|------|------|
| **Unet** | シンプル・高速・軽量, スキップ接続で細部保持 | 汎用, 医療画像, 小規模データ |
| **DeepLabV3+** | ASPP + デコーダーで高精度, マルチスケール対応 | 高精度が必要な場面 |

## 開発

```bash
# コード整形
uv run black .
uv run isort .

# 静的解析
uv run mypy .
uv run pydocstyle .

# テスト実行
uv run pytest

# pre-commit (全フック実行)
uv run pre-commit run --all-files
```

## ドキュメント

- [ユーザーガイド](docs/user-guide.md) - 詳細な使用方法
- [アーキテクチャ](docs/architecture.md) - 設計思想と構造
- [APIリファレンス](docs/api-reference.md) - クラス/関数リファレンス

## ライセンス

MIT License
