# ユーザーガイド

pochisegmentation の詳細な使用方法を説明します.

## 目次

1. [セットアップ](#セットアップ)
2. [データセットの準備](#データセットの準備)
3. [設定ファイル](#設定ファイル)
4. [訓練](#訓練)
5. [推論](#推論)
6. [出力ファイル](#出力ファイル)
7. [Tips](#tips)

---

## セットアップ

### 動作環境

- Python 3.13
- CUDA 13.0 (GPU使用時)

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/kurorosu/pochisegmentation.git
cd pochisegmentation

# uv で依存関係インストール
uv sync

# 開発用依存関係も含める場合
uv sync --group dev
```

---

## データセットの準備

VOC形式のディレクトリ構造が必要です.

### 基本構造

```
dataset/
├── JPEGImages/              # 入力画像
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── SegmentationClass/       # マスク画像 (グレースケール PNG)
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── ImageSets/
│   └── Segmentation/
│       ├── train.txt        # 訓練用画像ID (拡張子なし)
│       └── val.txt          # 検証用画像ID (拡張子なし)
└── class_names.txt          # クラス名リスト (オプション)
```

### 画像ID ファイル (train.txt / val.txt)

```
image_001
image_002
image_003
```

### マスク画像の形式

- **形式**: グレースケール PNG
- **ピクセル値**: クラスインデックス (0 = 背景, 1 = クラス1, ...)
- **サイズ**: 入力画像と同じサイズ

### サポートされる画像形式

- 入力画像: `.jpg`, `.jpeg`, `.png`, `.bmp`
- マスク画像: `.png`

---

## 設定ファイル

`configs/pochi_seg_config.py` で訓練パラメータを設定します.

### 全設定項目

```python
# =============================================================================
# モデル設定
# =============================================================================
architecture = "Unet"           # モデルアーキテクチャ: "Unet" or "DeepLabV3Plus"
encoder_name = "resnet34"       # エンコーダー (resnet18/34/50, efficientnet-b0〜b7 など)
pretrained = True               # ImageNet事前学習済み重みを使用

# =============================================================================
# データ設定
# =============================================================================
data_root = "path/to/dataset"   # データセットのルートパス
num_classes = 4                 # クラス数 (背景含む)
image_size = 256                # 入力画像サイズ (正方形にリサイズ)
batch_size = 16                 # バッチサイズ
num_workers = 4                 # DataLoader のワーカー数

# =============================================================================
# 訓練設定
# =============================================================================
epochs = 100                    # エポック数
learning_rate = 1e-3            # 学習率
optimizer = "AdamW"             # オプティマイザ: "Adam", "AdamW", "SGD"
weight_decay = 1e-4             # 重み減衰

# 損失関数
loss = "DiceLoss"               # 損失関数: "DiceLoss", "FocalLoss", "JaccardLoss"
loss_params = {                 # 損失関数のパラメータ
    "mode": "multiclass"
}

# スケジューラー
scheduler = "CosineAnnealingLR" # スケジューラー: "CosineAnnealingLR", "StepLR", None
scheduler_params = {
    "T_max": 100                # CosineAnnealingLR 用
}

# =============================================================================
# 層別学習率 (オプション)
# =============================================================================
enable_layer_wise_lr = True     # 層別学習率を有効化
encoder_lr = 1e-4               # エンコーダーの学習率 (事前学習済みなので低め)
decoder_lr = 1e-3               # デコーダーの学習率

# =============================================================================
# 出力設定
# =============================================================================
work_dir = "work_dirs"          # ワークディレクトリ
device = "cuda"                 # デバイス: "cuda" or "cpu"
enable_metrics_export = True    # メトリクスCSV/グラフ出力を有効化
```

### エンコーダーの選択肢

| エンコーダー | パラメータ数 | 特徴 |
|-------------|-------------|------|
| `resnet18` | 11M | 軽量, 高速 |
| `resnet34` | 21M | バランス良好 (推奨) |
| `resnet50` | 25M | 高精度 |
| `efficientnet-b0` | 5M | 非常に軽量 |
| `efficientnet-b4` | 19M | 高精度 |

### 損失関数の選択

| 損失関数 | 特徴 | 用途 |
|---------|------|------|
| `DiceLoss` | クラス不均衡に強い | 医療画像, 小さな領域 |
| `FocalLoss` | 難しいサンプルを重視 | クラス不均衡が激しい場合 |
| `JaccardLoss` | IoU直接最適化 | mIoU改善重視 |

---

## 訓練

### 基本コマンド

```bash
python pochi.py seg-train --config configs/pochi_seg_config.py
```

### オプション

```bash
python pochi.py seg-train --config CONFIG_PATH [OPTIONS]

オプション:
  --config PATH    設定ファイルのパス (必須)
```

### 訓練の流れ

1. 設定ファイルの読み込み
2. ワークスペース作成 (`work_dirs/YYYYMMDD_XXX/`)
3. モデル・データセット・損失関数の初期化
4. 訓練ループ実行
5. エポックごとに検証, ベストモデル保存
6. 訓練完了後, グラフ・CSV出力

### 訓練ログ例

```
Epoch 1/100 - Train Loss: 0.4532, Val mIoU: 0.3215, Val Dice: 0.4123
Epoch 2/100 - Train Loss: 0.3821, Val mIoU: 0.4512, Val Dice: 0.5234
...
Best model saved at epoch 45 (mIoU: 0.7823)
```

---

## 推論

### 単一画像の推論

```bash
python pochi.py seg-infer \
  --config configs/pochi_seg_config.py \
  --model work_dirs/20251225_001/models/best.pth \
  --data path/to/image.jpg
```

### ディレクトリ内の全画像を推論

```bash
python pochi.py seg-infer \
  --config configs/pochi_seg_config.py \
  --model work_dirs/20251225_001/models/best.pth \
  --data path/to/images/
```

対応形式: `.jpg`, `.jpeg`, `.png`, `.bmp`

### パスリストファイルを使った推論

```bash
python pochi.py seg-infer \
  --config configs/pochi_seg_config.py \
  --model work_dirs/20251225_001/models/best.pth \
  --data path/to/image_list.txt
```

`image_list.txt` の形式:

```
/path/to/image1.jpg
/path/to/image2.png
/path/to/image3.bmp
```

### 推論オプション

```bash
python pochi.py seg-infer --config CONFIG --model MODEL --data DATA [OPTIONS]

必須引数:
  --config PATH    設定ファイルのパス
  --model PATH     学習済みモデル (.pth) のパス
  --data PATH      画像/ディレクトリ/.txt ファイルのパス
```

---

## 出力ファイル

### 訓練出力 (`work_dirs/YYYYMMDD_XXX/`)

```
work_dirs/20251225_001/
├── config.py                    # 設定ファイルのコピー
├── models/
│   ├── best.pth                 # 最良モデル (mIoU基準)
│   └── latest.pth               # 最新モデル
├── logs/
│   └── training.log             # 訓練ログ
├── visualization/
│   ├── training_history.csv     # 訓練履歴CSV
│   ├── loss.png                 # 損失グラフ
│   ├── metrics.png              # mIoU/Diceグラフ
│   └── learning_rate.png        # 学習率グラフ
└── paths/
    ├── train.txt                # 訓練画像パスリスト
    └── val.txt                  # 検証画像パスリスト
```

### 推論出力 (`work_dirs/predictions/YYYYMMDD_XXX/`)

```
predictions/20251225_001/
├── image1_mask.png              # カラーマスク画像
├── image1_vis.png               # 元画像にマスクをオーバーレイ
├── image2_mask.png
└── image2_vis.png
```

- `*_mask.png`: クラスごとに色分けされたセグメンテーションマスク
- `*_vis.png`: 元画像に半透明マスクを重ねた可視化画像

---

## Tips

### GPU メモリが不足する場合

1. `batch_size` を小さくする (16 → 8 → 4)
2. `image_size` を小さくする (256 → 224 → 192)
3. 軽量なエンコーダーを選択 (`resnet34` → `resnet18`)

### 訓練が収束しない場合

1. `learning_rate` を調整 (1e-3 → 1e-4)
2. `epochs` を増やす
3. 層別学習率を有効化 (`enable_layer_wise_lr = True`)

### クラス不均衡がある場合

1. `FocalLoss` を使用
2. `DiceLoss` を使用 (小さな領域に効果的)

### 過学習を防ぐ

1. `weight_decay` を増やす
2. データ拡張を強化 (transforms 設定)
3. より軽量なモデルを使用
