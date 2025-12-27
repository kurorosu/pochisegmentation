# ユーザーガイド

pochisegmentation の詳細な使用方法を説明します.

## 目次

1. [対話モード](#対話モード)
2. [設定ファイルモード](#設定ファイルモード)
3. [データセットの準備](#データセットの準備)
4. [高度な機能](#高度な機能)
5. [トラブルシューティング](#トラブルシューティング)

---

## 対話モード

### 訓練 (train)

設定ファイル不要で、質問に答えるだけで訓練を開始できます.

```bash
python pochi.py train
```

#### 質問の流れ

1. **データセットのパス** - VOC 形式のデータセットパス
2. **クラス数** - 背景を含むクラス数
3. **アーキテクチャ** - Unet または DeepLabV3Plus
4. **エンコーダー** - resnet34 など
5. **損失関数** - DiceLoss, FocalLoss, JaccardLoss
6. **オプティマイザ** - AdamW, Adam, SGD
7. **スケジューラ** - CosineAnnealingLR, StepLR, ReduceLROnPlateau
8. **エポック数** - 訓練エポック数
9. **バッチサイズ** - バッチサイズ
10. **学習率** - 学習率
11. **画像サイズ** - リサイズ後の画像サイズ
12. **Early Stopping** - 改善なしで停止するエポック数
13. **AMP** - 混合精度訓練の有効/無効

#### 確認画面

設定後、確認画面が表示されます:

```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 項目             ┃ 値                     ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ データセット     │ data/my_dataset        │
│ クラス数         │ 4                      │
│ アーキテクチャ   │ Unet                   │
│ エンコーダー     │ resnet34               │
│ 損失関数         │ DiceLoss               │
│ オプティマイザ   │ AdamW                  │
│ スケジューラ     │ CosineAnnealingLR      │
│ エポック数       │ 100                    │
│ バッチサイズ     │ 16                     │
│ 学習率           │ 0.001                  │
│ 画像サイズ       │ 256                    │
│ Early Stopping   │ 10 エポック            │
│ AMP              │ 有効                   │
└──────────────────┴────────────────────────┘

? 訓練を開始
  設定を保存して終了
  キャンセル
```

**「設定を保存して終了」** を選ぶと、設定ファイルが `configs/` に保存され、次回以降再利用できます.

### 推論 (infer)

```bash
python pochi.py infer
```

#### 質問の流れ

1. **ワークスペース選択** - 訓練済みワークスペースを選択
2. **データ選択** - 推論対象を選択
   - 訓練データ (`paths/train.txt`)
   - 検証データ (`paths/val.txt`)
   - カスタムパス (手動入力)

---

## 設定ファイルモード

### 訓練

```bash
python pochi.py train --config configs/pochi_seg_config.py
```

### 推論

```bash
python pochi.py infer --model work_dirs/20251225_001/models/best.pth --data path/to/images/
```

#### 推論オプション

| オプション | 説明 |
|-----------|------|
| `--model` | モデルファイル (.pth) のパス |
| `--data` | 画像ファイル, ディレクトリ, または .txt ファイル |
| `--output` | 出力ディレクトリ (省略時は自動生成) |

#### データ指定方法

```bash
# 単一画像
python pochi.py infer --model best.pth --data image.jpg

# ディレクトリ内の全画像
python pochi.py infer --model best.pth --data images/

# パスリストファイル
python pochi.py infer --model best.pth --data image_list.txt
```

---

## データセットの準備

### VOC 形式

```
dataset/
├── JPEGImages/              # 入力画像
│   ├── image_001.jpg
│   └── ...
├── SegmentationClass/       # マスク画像 (グレースケール PNG)
│   ├── image_001.png
│   └── ...
├── ImageSets/
│   └── Segmentation/
│       ├── train.txt        # 訓練用画像 ID (拡張子なし)
│       └── val.txt          # 検証用画像 ID
└── class_names.txt          # クラス名リスト (オプション)
```

### 画像 ID ファイル (train.txt / val.txt)

```
image_001
image_002
image_003
```

### マスク画像

- **形式**: グレースケール PNG
- **ピクセル値**: クラスインデックス (0 = 背景, 1, 2, ...)
- **サイズ**: 入力画像と同じサイズ

### class_names.txt (オプション)

クラス名を定義すると、可視化でクラス名が表示されます.

```
background
dog
cat
person
```

---

## 高度な機能

### Early Stopping

指定したエポック数の間、検証 mIoU が改善しなければ訓練を自動停止します.

```python
# 設定ファイル
early_stopping_patience = 10  # 10 エポック改善なしで停止
```

- `None` または `0` で無効
- ReduceLROnPlateau スケジューラと併用時は無効推奨

### AMP (混合精度訓練)

FP16/FP32 の混合精度で訓練し、GPU メモリを削減・高速化します.

```python
# 設定ファイル
enable_amp = True
```

- **効果**: メモリ約 30-50% 削減、速度 1.5-2 倍向上
- **条件**: CUDA 専用 (CPU では自動無効化)
- **推奨**: resnet50 以上の大きなモデルで効果大

### 層別学習率

事前学習済みエンコーダーには低い学習率、デコーダーには高い学習率を設定します.

```python
# 設定ファイル
enable_layer_wise_lr = True
encoder_lr = 1e-4  # エンコーダー (低め)
decoder_lr = 1e-3  # デコーダー (高め)
```

### スケジューラ

| スケジューラ | 説明 |
|-------------|------|
| `CosineAnnealingLR` | コサインカーブで学習率を減衰 (推奨) |
| `StepLR` | 一定エポックごとに学習率を減衰 |
| `ReduceLROnPlateau` | 検証指標が改善しないとき学習率を減衰 |

---

## トラブルシューティング

### GPU メモリ不足

1. `batch_size` を小さくする (16 → 8 → 4)
2. `image_size` を小さくする (256 → 224 → 192)
3. `enable_amp = True` で混合精度訓練を有効化
4. 軽量なエンコーダーを選択 (`resnet50` → `resnet34` → `resnet18`)

### 訓練が収束しない

1. `learning_rate` を調整 (1e-3 → 1e-4)
2. `epochs` を増やす
3. `enable_layer_wise_lr = True` で層別学習率を有効化
4. `DiceLoss` または `FocalLoss` を試す

### 過学習している

1. `early_stopping_patience` を設定 (例: 10)
2. `weight_decay` を増やす
3. より軽量なモデルを使用

### クラス不均衡がある

1. `DiceLoss` を使用 (小さな領域に効果的)
2. `FocalLoss` を使用 (難しいサンプルを重視)

### Ctrl+C で停止したい

訓練中に Ctrl+C を押すと、安全に停止します.
- 現在のエポックの途中でも、ベストモデルは保存済み
- 可視化ファイルも出力されます
