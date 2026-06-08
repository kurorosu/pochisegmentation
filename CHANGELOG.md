# Changelog

本プロジェクトの注目すべき変更点をこのファイルに記録する.

フォーマットは [Keep a Changelog](https://keepachangelog.com/) に従い, バージョニングは [Semantic Versioning](https://semver.org/) に準拠する.

## [Unreleased]

### Added

- GitHub Issue / PR テンプレートを整備 ([#29](https://github.com/kurorosu/pochisegmentation/pull/29))
- `CHANGELOG.md` を Keep a Changelog 形式で新設し, アーカイブ用 `changelogs/` ディレクトリと運用ガイドを追加 ([#34](https://github.com/kurorosu/pochisegmentation/pull/34))
- `pochisegmentation/__init__.py` に `__version__` を追加し, README にバージョン / Python / ライセンスバッジを追加 ([#34](https://github.com/kurorosu/pochisegmentation/pull/34))
- pre-commit に gitleaks フックを追加 ([#36](https://github.com/kurorosu/pochisegmentation/pull/36))

### Changed

- Issue / PR テンプレートのフォーマットを整理 ([#30](https://github.com/kurorosu/pochisegmentation/pull/30))
- `.gitignore` に `.claude/` を追加 ([#33](https://github.com/kurorosu/pochisegmentation/pull/33))
- `pyproject.toml` の version を v1.0.0 タグに合わせて `1.0.0` へ修正 ([#34](https://github.com/kurorosu/pochisegmentation/pull/34))
- pre-commit の mypy フックを `local` 化し `uv run mypy` ベースに変更, `[tool.mypy]` overrides を整備 ([#36](https://github.com/kurorosu/pochisegmentation/pull/36))
- pytest フックを `uv run --no-sync pytest -q -n 6 --dist=worksteal` に整理し pytest-xdist を追加 ([#36](https://github.com/kurorosu/pochisegmentation/pull/36))
- ロガー名を `pochiseg` に統一 ([#44](https://github.com/kurorosu/pochisegmentation/pull/44))
- CLI コマンドを `cli/commands/` パッケージに分割し, argparse 構築を `cli/parser.py` に切り出し ([#45](https://github.com/kurorosu/pochisegmentation/pull/45))
- CLI エントリを `pochisegmentation/cli/pochi.py` に移植し `[project.scripts]` で `uv run pochi` を提供 ([#46](https://github.com/kurorosu/pochisegmentation/pull/46))
- pytest `addopts` に `-n auto --dist=worksteal` と `markers` を追加 ([#46](https://github.com/kurorosu/pochisegmentation/pull/46))
- dict ベースの設定を Pydantic v2 の `PochiSegConfig` に置換し, factories / core から dict 直アクセスを排除 ([#49](https://github.com/kurorosu/pochisegmentation/pull/49))
- モノリシックな `seg_trainer.py` を `pochisegmentation/training/` 配下の責務別部品 (EpochRunner / Evaluator / CheckpointStore / MetricsTracker / EarlyStopping / TrainingLoop) に分解し, `PochiSegmentationTrainer` を薄いファサードに縮退. ベスト指標と Early Stopping を `early_stopping_monitor` (mIoU / Dice / val_loss) で統一 ([#50](https://github.com/kurorosu/pochisegmentation/pull/50))
- 単一ファイルの `seg_predictor.py` を `pochisegmentation/inference/` パッケージ (preprocess / sync / checkpoint_loader / postprocess) に分解し, `core/inference.py` をオーケストレーションのみに縮退 ([#51](https://github.com/kurorosu/pochisegmentation/pull/51))
- 開発・実行環境の Python を 3.13 から 3.14 に引き上げ (`.python-version` / `requires-python` / black / mypy). 併せて 3.14 の Windows 公式 wheel を持つ `numpy>=2.3` に更新し, テスト生成 config から非 ASCII コメントを除去 ([#53](https://github.com/kurorosu/pochisegmentation/pull/53))
- ロガーの出力形式を pochidetection に合わせてパイプ区切りに変更 (`asctime|level|module|lineno| message`, レベル名 5 文字固定 / 行番号 3 桁ゼロ埋め). 色付き出力のため `colorlog` を依存に追加 ([#55](https://github.com/kurorosu/pochisegmentation/pull/55))

### Removed

- pytest のカバレッジ測定を撤廃 (`addopts` から `--cov` を削除, dev 依存と `uv.lock` から `pytest-cov` を除去) ([#52](https://github.com/kurorosu/pochisegmentation/pull/52))

### Fixed

- `SegmentationMetrics` の `MeanIoU` / `DiceScore` に `input_format="index"` を指定し, インデックス形式入力での mIoU / Dice 誤計算を修正. ベストモデル選択と Early Stopping が正しい指標で動作するようになった (NA.)
- mypy が検出した既存の型エラー 27 件を解消 (明示的な型注釈 / `cast` / imread の `None` ガード / matplotlib colormap API の更新) ([#37](https://github.com/kurorosu/pochisegmentation/pull/37))

## [1.0.0] - 2025-12-27

### Added

- 初期リリース
- segmentation-models-pytorch ベースの UNet / DeepLabV3Plus モデルラッパー
- DIP に基づくインターフェース層 (`ISegmentationModel`, `ISegmentationLoss`, `ISegmentationMetrics`, `ISegmentationDataset`)
- `ComponentFactory` による model / loss / metrics / optimizer / scheduler のレジストリベース解決
- VOC 形式データセット (`VOCSegmentationDataset`) と torchvision v2 transform パイプライン
- 損失関数: `DiceLoss`, `FocalLoss`, `JaccardLoss`, `CombinedLoss`
- 評価指標: mIoU / Dice / per-class IoU (`SegmentationMetrics`)
- `PochiSegmentationTrainer`: AMP, Early Stopping, 層別学習率, checkpoint 管理
- `PochiSegmentationPredictor`: 学習済みモデルでの推論ラッパー
- 可視化: マスクカラーライズ, オーバーレイ, クラス別精度, Confusion Matrix, 学習履歴 CSV / グラフ
- 対話型 CLI (questionary ベースの学習 / 推論ウィザード)
- 設定の自動保存 / 保存済み設定の選択機能
- `ConfigLoader` による Python 設定ファイル読み込み (サイレントフォールバック排除)
- Ctrl+C による訓練の安全停止機能
- スケジューラごとのデフォルトパラメータ自動設定
- 推論ウィザードでモデルに紐づくデータを自動検出
- 訓練ログに損失関数名と層別学習率を表示
- ドキュメント: README, ユーザーガイド, アーキテクチャ, API リファレンス

## Archived Changelogs

過去バージョンの履歴は [`changelogs/`](changelogs/) ディレクトリにアーカイブする.
