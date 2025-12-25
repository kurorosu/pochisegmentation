"""対話型 CLI の選択肢定義.

ユーザーはこのファイルを編集して, 独自の選択肢を追加できる.
"""

from dataclasses import dataclass


@dataclass
class Choice:
    """選択肢の定義."""

    value: str
    label: str
    description: str = ""


@dataclass
class TrainingPreset:
    """訓練パラメータのプリセット."""

    name: str
    label: str
    description: str
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int


# =============================================================================
# データセット
# =============================================================================
DATA_ROOTS: list[Choice] = [
    Choice("data/VOC2012", "VOC2012", "Pascal VOC 2012 データセット"),
    Choice("data/custom", "Custom", "カスタムデータセット"),
]

# =============================================================================
# モデル
# =============================================================================
ARCHITECTURES: list[Choice] = [
    Choice("Unet", "Unet", "軽量で高速, 一般的なセグメンテーション向け"),
    Choice("DeepLabV3Plus", "DeepLabV3+", "高精度, Atrous Convolution ベース"),
]

ENCODERS: list[Choice] = [
    Choice("resnet18", "ResNet-18", "軽量 (11M params)"),
    Choice("resnet34", "ResNet-34", "バランス (21M params) - 推奨"),
    Choice("resnet50", "ResNet-50", "高精度 (25M params)"),
    Choice("efficientnet-b0", "EfficientNet-B0", "効率的 (5M params)"),
]

# =============================================================================
# 損失関数
# =============================================================================
LOSSES: list[Choice] = [
    Choice("DiceLoss", "Dice Loss", "領域の重なりを最適化 - 推奨"),
    Choice("FocalLoss", "Focal Loss", "クラス不均衡に強い"),
    Choice("JaccardLoss", "Jaccard Loss", "IoU を直接最適化"),
]

# =============================================================================
# オプティマイザ
# =============================================================================
OPTIMIZERS: list[Choice] = [
    Choice("AdamW", "AdamW", "重み減衰付き Adam - 推奨"),
    Choice("Adam", "Adam", "標準的な Adam"),
    Choice("SGD", "SGD", "モメンタム付き SGD"),
]

# =============================================================================
# スケジューラ
# =============================================================================
SCHEDULERS: list[Choice] = [
    Choice("CosineAnnealingLR", "Cosine Annealing", "滑らかな学習率減衰 - 推奨"),
    Choice("StepLR", "Step LR", "一定ステップごとに減衰"),
    Choice("ReduceLROnPlateau", "Reduce on Plateau", "停滞時に減衰"),
]

# None 用の特別な選択肢
SCHEDULER_NONE = Choice("None", "None", "スケジューラなし")

# =============================================================================
# 訓練プリセット
# =============================================================================
TRAINING_PRESETS: list[TrainingPreset] = [
    TrainingPreset(
        name="quick",
        label="Quick",
        description="高速確認用 (10 epochs)",
        epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        image_size=256,
    ),
    TrainingPreset(
        name="balanced",
        label="Balanced",
        description="推奨設定 (50 epochs)",
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        image_size=256,
    ),
    TrainingPreset(
        name="thorough",
        label="Thorough",
        description="高精度 (100 epochs)",
        epochs=100,
        batch_size=8,
        learning_rate=5e-4,
        image_size=512,
    ),
]
