# Changelog

本プロジェクトの注目すべき変更点をこのファイルに記録する.

フォーマットは [Keep a Changelog](https://keepachangelog.com/) に従い, バージョニングは [Semantic Versioning](https://semver.org/) に準拠する.

## [Unreleased]

### Added

- GitHub Issue / PR テンプレートを整備 ([#29](https://github.com/kurorosu/pochisegmentation/pull/29))
- `CHANGELOG.md` を Keep a Changelog 形式で新設し, アーカイブ用 `changelogs/` ディレクトリと運用ガイドを追加 ([#34](https://github.com/kurorosu/pochisegmentation/pull/34))
- `pochisegmentation/__init__.py` に `__version__` を追加し, README にバージョン / Python / ライセンスバッジを追加 ([#34](https://github.com/kurorosu/pochisegmentation/pull/34))
- pre-commit に gitleaks フックを追加 (NA.)

### Changed

- Issue / PR テンプレートのフォーマットを整理 ([#30](https://github.com/kurorosu/pochisegmentation/pull/30))
- `.gitignore` に `.claude/` を追加 ([#33](https://github.com/kurorosu/pochisegmentation/pull/33))
- `pyproject.toml` の version を v1.0.0 タグに合わせて `1.0.0` へ修正 ([#34](https://github.com/kurorosu/pochisegmentation/pull/34))
- pre-commit の mypy フックを `local` 化し `uv run mypy` ベースに変更, `[tool.mypy]` overrides を整備 (NA.)
- pytest フックを `uv run --no-sync pytest -q -n 6 --dist=worksteal` に整理し pytest-xdist を追加 (NA.)

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
