# API リファレンス

pochisegmentation の主要クラスと関数のリファレンス.

## 目次

1. [Trainer](#trainer)
2. [Predictor](#predictor)
3. [Models](#models)
4. [Losses](#losses)
5. [Metrics](#metrics)
6. [Datasets](#datasets)
7. [Transforms](#transforms)
8. [Factories](#factories)
9. [Visualization](#visualization)

---

## Trainer

### PochiSegmentationTrainer

セグメンテーションモデルの訓練を行うクラス.

```python
from pochisegmentation import PochiSegmentationTrainer

trainer = PochiSegmentationTrainer(
    model: ISegmentationModel,
    criterion: ISegmentationLoss,
    metrics: ISegmentationMetrics,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
    config: Optional[dict] = None,
    workspace_manager: Optional[PochiWorkspaceManager] = None,
)
```

#### パラメータ

| パラメータ | 型 | 説明 |
|-----------|---|------|
| `model` | ISegmentationModel | セグメンテーションモデル |
| `criterion` | ISegmentationLoss | 損失関数 |
| `metrics` | ISegmentationMetrics | 評価指標 |
| `optimizer` | Optimizer | オプティマイザ |
| `scheduler` | _LRScheduler | 学習率スケジューラー (オプション) |
| `device` | str | デバイス ("cuda" or "cpu") |
| `config` | dict | 設定辞書 (オプション) |
| `workspace_manager` | PochiWorkspaceManager | ワークスペース管理 (オプション) |

#### メソッド

##### train()

```python
trainer.train(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
) -> None
```

訓練ループを実行.

---

## Predictor

### PochiSegmentationPredictor

学習済みモデルを使った推論を行うクラス.

```python
from pochisegmentation import PochiSegmentationPredictor

predictor = PochiSegmentationPredictor(
    model: ISegmentationModel,
    transform: Callable,
    device: str = "cuda",
    num_classes: int = 2,
)
```

#### パラメータ

| パラメータ | 型 | 説明 |
|-----------|---|------|
| `model` | ISegmentationModel | 学習済みモデル |
| `transform` | Callable | 前処理Transform |
| `device` | str | デバイス ("cuda" or "cpu") |
| `num_classes` | int | クラス数 |

#### メソッド

##### predict()

```python
predictor.predict(image_path: Union[str, Path]) -> np.ndarray
```

単一画像を推論. クラスインデックスのマスクを返す.

##### predict_image()

```python
predictor.predict_image(image: np.ndarray) -> np.ndarray
```

numpy配列から推論.

##### predict_batch()

```python
predictor.predict_batch(loader: DataLoader) -> list[np.ndarray]
```

DataLoaderを使ったバッチ推論.

##### from_checkpoint() (クラスメソッド)

```python
predictor = PochiSegmentationPredictor.from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: ISegmentationModel,
    transform: Callable,
    device: str = "cuda",
) -> PochiSegmentationPredictor
```

チェックポイントからPredictorを作成.

---

## Models

### UnetModel

```python
from pochisegmentation.models import UnetModel

model = UnetModel(
    encoder_name: str = "resnet34",
    num_classes: int = 2,
    pretrained: bool = True,
)
```

### DeepLabV3PlusModel

```python
from pochisegmentation.models import DeepLabV3PlusModel

model = DeepLabV3PlusModel(
    encoder_name: str = "resnet34",
    num_classes: int = 2,
    pretrained: bool = True,
)
```

#### 共通パラメータ

| パラメータ | 型 | 説明 |
|-----------|---|------|
| `encoder_name` | str | エンコーダー名 (resnet34, efficientnet-b0 など) |
| `num_classes` | int | 出力クラス数 |
| `pretrained` | bool | ImageNet事前学習済み重みを使用 |

#### 共通メソッド

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `forward(x)` | Tensor | 順伝播 |
| `get_encoder_params()` | list | エンコーダーパラメータ |
| `get_decoder_params()` | list | デコーダーパラメータ |

---

## Losses

### DiceLoss

```python
from pochisegmentation.losses import DiceLoss

loss = DiceLoss(mode: str = "multiclass", **kwargs)
```

### FocalLoss

```python
from pochisegmentation.losses import FocalLoss

loss = FocalLoss(mode: str = "multiclass", gamma: float = 2.0, **kwargs)
```

### JaccardLoss

```python
from pochisegmentation.losses import JaccardLoss

loss = JaccardLoss(mode: str = "multiclass", **kwargs)
```

### CombinedLoss

```python
from pochisegmentation.losses import CombinedLoss

loss = CombinedLoss(
    losses: list[ISegmentationLoss],
    weights: Optional[list[float]] = None,
)
```

複数の損失関数を組み合わせる.

---

## Metrics

### SegmentationMetrics

```python
from pochisegmentation.metrics import SegmentationMetrics

metrics = SegmentationMetrics(
    num_classes: int,
    device: str = "cuda",
)
```

#### メソッド

| メソッド | 説明 |
|---------|------|
| `update(preds, targets)` | バッチ結果を蓄積 |
| `compute()` | 指標を計算して辞書で返す |
| `reset()` | 状態をリセット |

#### 計算される指標

| 指標 | 説明 |
|------|------|
| `mIoU` | Mean Intersection over Union |
| `Dice` | Dice係数 |

---

## Datasets

### VOCSegmentationDataset

```python
from pochisegmentation.datasets import VOCSegmentationDataset

dataset = VOCSegmentationDataset(
    root: Union[str, Path],
    split: str = "train",
    transform: Optional[Callable] = None,
)
```

#### パラメータ

| パラメータ | 型 | 説明 |
|-----------|---|------|
| `root` | Path | データセットルートパス |
| `split` | str | "train" or "val" |
| `transform` | Callable | Transform関数 (オプション) |

#### プロパティ

| プロパティ | 型 | 説明 |
|-----------|---|------|
| `class_names` | list[str] | クラス名リスト |
| `num_classes` | int | クラス数 |

#### メソッド

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `get_image_paths()` | list[Path] | 画像パスリスト |
| `get_mask_paths()` | list[Path] | マスクパスリスト |

---

## Transforms

### get_train_transform()

```python
from pochisegmentation.transforms import get_train_transform

transform = get_train_transform(config: dict) -> v2.Compose
```

訓練用Transform. データ拡張を含む.

### get_val_transform()

```python
from pochisegmentation.transforms import get_val_transform

transform = get_val_transform(config: dict) -> v2.Compose
```

検証/推論用Transform. リサイズと正規化のみ.

### wrap_image_and_mask()

```python
from pochisegmentation.transforms import wrap_image_and_mask

image_tv, mask_tv = wrap_image_and_mask(image: np.ndarray, mask: np.ndarray)
```

画像とマスクを tv_tensors でラップ.

---

## Factories

### ComponentFactory

コンポーネントの登録と生成を行うファクトリークラス.

```python
from pochisegmentation.factories import ComponentFactory
```

#### クラスメソッド

##### register_model()

```python
ComponentFactory.register_model(name: str, model_class: type[ISegmentationModel])
```

##### register_loss()

```python
ComponentFactory.register_loss(name: str, loss_class: type[ISegmentationLoss])
```

##### register_metrics()

```python
ComponentFactory.register_metrics(name: str, metrics_class: type[ISegmentationMetrics])
```

##### create_model()

```python
model = ComponentFactory.create_model(config: dict) -> ISegmentationModel
```

設定辞書からモデルを生成.

##### create_loss()

```python
loss = ComponentFactory.create_loss(config: dict) -> ISegmentationLoss
```

設定辞書から損失関数を生成.

##### create_metrics()

```python
metrics = ComponentFactory.create_metrics(config: dict) -> ISegmentationMetrics
```

設定辞書から評価指標を生成.

---

## Visualization

### mask_visualizer

#### colorize_mask()

```python
from pochisegmentation.visualization import colorize_mask

color_mask = colorize_mask(
    mask: np.ndarray,
    num_classes: Optional[int] = None,
    palette: Optional[np.ndarray] = None,
) -> np.ndarray
```

クラスインデックスマスクをカラー画像に変換.

| パラメータ | 型 | 説明 |
|-----------|---|------|
| `mask` | np.ndarray | クラスインデックスマスク (H, W) |
| `num_classes` | int | クラス数 (Noneの場合, mask.max()+1) |
| `palette` | np.ndarray | カラーパレット (N, 3) |

| 戻り値 | 型 | 説明 |
|--------|---|------|
| `color_mask` | np.ndarray | RGB画像 (H, W, 3) |

#### overlay_mask_on_image()

```python
from pochisegmentation.visualization import overlay_mask_on_image

overlay = overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    num_classes: Optional[int] = None,
    palette: Optional[np.ndarray] = None,
) -> np.ndarray
```

元画像にマスクをオーバーレイ.

| パラメータ | 型 | 説明 |
|-----------|---|------|
| `image` | np.ndarray | 元画像 (H, W, 3) RGB |
| `mask` | np.ndarray | クラスインデックスマスク (H, W) |
| `alpha` | float | オーバーレイの透明度 (0.0〜1.0) |

| 戻り値 | 型 | 説明 |
|--------|---|------|
| `overlay` | np.ndarray | オーバーレイ画像 (H, W, 3) RGB |

#### create_color_palette()

```python
from pochisegmentation.visualization import create_color_palette

palette = create_color_palette(num_classes: int) -> np.ndarray
```

クラス数に応じたカラーパレットを生成.

### TrainingMetricsExporter

訓練メトリクスの CSV 出力とグラフ生成.

```python
from pochisegmentation.visualization import TrainingMetricsExporter

exporter = TrainingMetricsExporter(
    output_dir: Path,
    enable_visualization: bool = True,
)
```

#### メソッド

| メソッド | 説明 |
|---------|------|
| `record_epoch(...)` | エポックのメトリクスを記録 |
| `export_to_csv(filename)` | CSV ファイルに出力 |
| `generate_graphs(base_filename)` | グラフを生成 |
| `export_all()` | CSV とグラフを両方出力 |
| `get_best_epoch(metric)` | 最良エポックを取得 |
| `get_summary()` | 訓練サマリーを取得 |
