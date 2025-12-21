"""セグメンテーションモデルのインターフェース."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ISegmentationModel(ABC, nn.Module):
    """セグメンテーションモデルのインターフェース.

    すべてのセグメンテーションモデルはこのインターフェースを実装し,
    トレーナーや他のコンポーネントとの互換性を確保する.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播.

        Args:
            x: 入力テンソル, 形状は (B, C, H, W).

        Returns:
            出力テンソル, 形状は (B, num_classes, H, W).
        """
        pass

    @abstractmethod
    def get_encoder_params(self) -> list:
        """層別学習率用のエンコーダーパラメータを取得.

        Returns:
            エンコーダーパラメータのリスト.
        """
        pass

    @abstractmethod
    def get_decoder_params(self) -> list:
        """層別学習率用のデコーダーパラメータを取得.

        Returns:
            デコーダーパラメータのリスト.
        """
        pass
