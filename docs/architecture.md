# アーキテクチャ

pochisegmentation の設計思想とディレクトリ構造を説明します.

## 目次

1. [設計方針](#設計方針)
2. [SOLID原則の適用](#solid原則の適用)
3. [ディレクトリ構造](#ディレクトリ構造)
4. [依存性注入 (DI)](#依存性注入-di)
5. [コンポーネント登録](#コンポーネント登録)
6. [拡張方法](#拡張方法)

---

## 設計方針

### 基本理念

1. **Simplicity First** - 最小限の設定で訓練開始可能
2. **SOLID原則** - 特に DI (依存性注入) と DIP (依存性逆転) を重視
3. **ライブラリ活用** - モデル・損失関数・評価指標は既存ライブラリを最大限活用
4. **段階的実装** - 各フェーズで動作確認可能な状態を維持

### 採用モデル

メンテナンス性とコード品質維持のため, モデルは **2種類** に限定.

| モデル | 特徴 | 用途 |
|--------|------|------|
| **Unet** | シンプル・高速・軽量, スキップ接続で細部保持 | 汎用, 医療画像, 小規模データ |
| **DeepLabV3+** | ASPP + デコーダーで高精度, マルチスケール対応 | 高精度が必要な場面 |

---

## SOLID原則の適用

### 原則と適用箇所

| 原則 | 説明 | 適用箇所 |
|------|------|---------|
| **S** (単一責任) | 1つのクラスは1つの責務のみ持つ | Model, Loss, Metrics, Trainer がそれぞれ1つの責務 |
| **O** (開放閉鎖) | 拡張に開放, 修正に閉鎖 | 新モデル追加時, 既存コード修正不要 (register するだけ) |
| **L** (リスコフ) | 派生型は基底型と置換可能 | ISegmentationModel を実装すれば置換可能 |
| **I** (インターフェース分離) | 依存は必要なインターフェースのみ | Model, Loss, Metrics を別インターフェースに分離 |
| **D** (依存性逆転) | 高レベルモジュールは抽象に依存 | Trainer は具象(Unet)ではなく抽象(ISegmentationModel)に依存 |

### 依存関係図

```
┌─────────────────────────────────────────────────────────┐
│                    PochiSegmentationTrainer             │
│  ┌─────────────────────────────────────────────────┐   │
│  │         抽象インターフェースに依存               │   │
│  │  ISegmentationModel, ISegmentationLoss,         │   │
│  │  ISegmentationMetrics                           │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   ComponentFactory                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │              具象クラスを生成                   │   │
│  │  UnetModel, DeepLabV3PlusModel,                 │   │
│  │  DiceLoss, FocalLoss, SegmentationMetrics       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## ディレクトリ構造

```
pochisegmentation/
├── __init__.py                  # コンポーネント登録
├── seg_trainer.py               # PochiSegmentationTrainer (抽象に依存)
├── seg_predictor.py             # PochiSegmentationPredictor
│
├── interfaces/                  # 抽象インターフェース (DIP)
│   ├── __init__.py
│   ├── model.py                 # ISegmentationModel
│   ├── loss.py                  # ISegmentationLoss
│   ├── metrics.py               # ISegmentationMetrics
│   └── dataset.py               # ISegmentationDataset
│
├── models/                      # 具象モデル (2種類)
│   ├── __init__.py
│   ├── unet.py                  # UnetModel
│   └── deeplabv3plus.py         # DeepLabV3PlusModel
│
├── losses/                      # 損失関数 (smp.losses ラッパー)
│   ├── __init__.py
│   └── seg_losses.py            # DiceLoss, FocalLoss, JaccardLoss
│
├── metrics/                     # 評価指標 (torchmetrics ラッパー)
│   ├── __init__.py
│   └── seg_metrics.py           # SegmentationMetrics
│
├── datasets/                    # データセット
│   ├── __init__.py
│   └── voc_dataset.py           # VOCSegmentationDataset
│
├── transforms/                  # Transform (torchvision.transforms.v2)
│   ├── __init__.py
│   └── seg_transforms.py        # get_train_transform, get_val_transform
│
├── factories/                   # ファクトリー (DI)
│   ├── __init__.py
│   └── component_factory.py     # ComponentFactory
│
├── utils/                       # ユーティリティ
│   ├── __init__.py
│   ├── directory_manager.py     # PochiWorkspaceManager
│   ├── timestamp_utils.py       # タイムスタンプ生成
│   └── layer_wise_lr.py         # 層別学習率ヘルパー
│
├── logging/                     # ログ管理
│   ├── __init__.py
│   └── logger_manager.py        # LoggerManager
│
└── visualization/               # 可視化
    ├── __init__.py
    ├── metrics_exporter.py      # TrainingMetricsExporter
    ├── gradient_tracer.py       # GradientTracer
    └── mask_visualizer.py       # colorize_mask, overlay_mask_on_image
```

---

## 依存性注入 (DI)

### 概要

Trainer は具象クラス (Unet, DiceLoss など) を直接知らない.
代わりに, インターフェース (ISegmentationModel など) を通じて操作する.

### コード例

```python
# pochi.py (エントリーポイント)

from pochisegmentation.factories import ComponentFactory
from pochisegmentation.seg_trainer import PochiSegmentationTrainer

# 設定読み込み
config = load_config("configs/pochi_seg_config.py")

# ファクトリーで具象クラスを生成 (DI)
model = ComponentFactory.create_model(config)       # → UnetModel
criterion = ComponentFactory.create_loss(config)    # → DiceLoss
metrics = ComponentFactory.create_metrics(config)   # → SegmentationMetrics

# Trainer に注入 (Trainer は具象を知らない)
trainer = PochiSegmentationTrainer(
    model=model,          # ISegmentationModel
    criterion=criterion,  # ISegmentationLoss
    metrics=metrics,      # ISegmentationMetrics
    optimizer=optimizer,
    device=config.device,
)
```

### メリット

1. **テスト容易性** - モックを注入してユニットテスト可能
2. **柔軟性** - 設定ファイルでモデル・損失関数を切り替え可能
3. **保守性** - 新しいモデル追加時に Trainer の修正不要

---

## コンポーネント登録

### 登録処理 (`__init__.py`)

```python
# pochisegmentation/__init__.py

from pochisegmentation.factories import ComponentFactory
from pochisegmentation.models import UnetModel, DeepLabV3PlusModel
from pochisegmentation.losses import DiceLoss, FocalLoss, JaccardLoss
from pochisegmentation.metrics import SegmentationMetrics

# モデル登録
ComponentFactory.register_model("Unet", UnetModel)
ComponentFactory.register_model("DeepLabV3Plus", DeepLabV3PlusModel)

# 損失関数登録
ComponentFactory.register_loss("DiceLoss", DiceLoss)
ComponentFactory.register_loss("FocalLoss", FocalLoss)
ComponentFactory.register_loss("JaccardLoss", JaccardLoss)

# 評価指標登録
ComponentFactory.register_metrics("SegmentationMetrics", SegmentationMetrics)
```

### 使用方法

```python
# 設定ファイルで名前を指定
architecture = "Unet"      # → UnetModel が生成される
loss = "DiceLoss"          # → DiceLoss が生成される
```

---

## 拡張方法

### 新しいモデルを追加する場合

1. `ISegmentationModel` を実装したクラスを作成

```python
# models/my_model.py
from pochisegmentation.interfaces import ISegmentationModel

class MyModel(ISegmentationModel):
    def __init__(self, encoder_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        # モデル初期化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 順伝播
        pass

    def get_encoder_params(self) -> list:
        return list(self.encoder.parameters())

    def get_decoder_params(self) -> list:
        return list(self.decoder.parameters())
```

2. ファクトリーに登録

```python
# pochisegmentation/__init__.py
from pochisegmentation.models.my_model import MyModel
ComponentFactory.register_model("MyModel", MyModel)
```

3. 設定ファイルで使用

```python
# configs/my_config.py
architecture = "MyModel"
```

### 新しい損失関数を追加する場合

1. `ISegmentationLoss` を実装

```python
# losses/my_loss.py
from pochisegmentation.interfaces import ISegmentationLoss

class MyLoss(ISegmentationLoss):
    def __init__(self, **kwargs):
        # 初期化

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 損失計算
        pass
```

2. ファクトリーに登録

```python
ComponentFactory.register_loss("MyLoss", MyLoss)
```

---

## インターフェース定義

### ISegmentationModel

```python
class ISegmentationModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播."""
        pass

    @abstractmethod
    def get_encoder_params(self) -> list:
        """エンコーダーのパラメータ (層別学習率用)."""
        pass

    @abstractmethod
    def get_decoder_params(self) -> list:
        """デコーダーのパラメータ (層別学習率用)."""
        pass
```

### ISegmentationLoss

```python
class ISegmentationLoss(ABC):
    @abstractmethod
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失計算."""
        pass
```

### ISegmentationMetrics

```python
class ISegmentationMetrics(ABC):
    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """バッチ結果を蓄積."""
        pass

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """蓄積した結果から指標を計算."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """状態をリセット."""
        pass
```
