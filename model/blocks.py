# model/blocks.py
"""
File: model/blocks.py
Description: Temporal Fusion Transformer (TFT) のビルディングブロックとなる層を定義します。
             GLU, GRN, VSNなどの基本モジュールが含まれます。
"""

import torch
import torch.nn as nn
from typing import Optional


class GatedLinearUnit(nn.Module):
    """GLU (Gated Linear Unit).

    入力に対して線形変換を行い、シグモイド関数によるゲートを掛けることで
    情報の通過量を制御します。
    Formula: GLU(x) = (x * W1 + b1) * sigmoid(x * W2 + b2)
    """

    def __init__(self, input_size: int, hidden_size: int = None):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.fc = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """GRN (Gated Residual Network).

    TFTの基本構成ブロック。以下の機能を持つ:
    1. Skip Connection (ResNet構造)
    2. Gating (GLUによる情報選択)
    3. Standard Layer (LayerNorm -> GELU -> Linear)

    非線形な関係性と、単純な線形関係性の両方を学習可能にします。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size

        # コンテキスト情報がある場合は結合用に射影
        if context_size > 0:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_size, output_size)
        self.norm = nn.LayerNorm(output_size)

        # 残差結合用の射影（入出力サイズが異なる場合）
        if input_size != output_size:
            self.skip_proj = nn.Linear(input_size, output_size)
        else:
            self.skip_proj = None

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq, InputDim) or (Batch, InputDim)
            context: (Batch, ContextDim) or None
        """
        residual = self.skip_proj(x) if self.skip_proj else x

        x = self.fc1(x)
        if context is not None:
            # 静的コンテキスト情報を隠れ層に加算
            context_vec = self.context_proj(context)

            # 入力xの次元に合わせてcontext_vecを拡張 (Broadcasting対応)
            # x: (Batch, ..., Hidden), context: (Batch, Hidden) -> (Batch, 1, ..., 1, Hidden)
            while context_vec.dim() < x.dim():
                context_vec = context_vec.unsqueeze(1)

            x = x + context_vec

        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)

        return self.norm(residual + x)


class VariableSelectionNetwork(nn.Module):
    """VSN (Variable Selection Network).

    多数の特徴量から、現在の予測にとって重要な変数を動的に選択し重み付けします。
    Configで定義された Microstructure / Macro Lead-Lag 特徴量の中から、
    相場環境に応じて有効なものを自動選択します。
    """

    def __init__(
        self,
        num_inputs: int,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = 0,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size

        # 各入力変数ごとのGRN
        # 高速化のため、特徴量間で重みを共有する単一のGRNに変更
        # (Batch, Seq, NumInputs, InputDim) -> (Batch, Seq, NumInputs, Hidden)
        self.shared_grn = GatedResidualNetwork(
            input_size, hidden_size, hidden_size, dropout, context_size
        )

        # 変数重要度重みを計算するGRN
        # すべての入力をFlattenして入力し、各変数に対するSoftmax重みを出力
        self.weight_grn = GatedResidualNetwork(
            num_inputs * input_size, hidden_size, num_inputs, dropout, context_size
        )

    def forward(
        self, x_stack: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_stack: (Batch, Seq, NumInputs, Dim)
            context: 静的コンテキスト
        Returns:
            weighted_sum: (Batch, Seq, Hidden)
        """
        # 2. 変数選択の重みを計算
        # 入力をフラット化: (Batch, Seq, NumInputs * Dim) に変形
        batch, seq, num_inputs, dim = x_stack.shape
        flat_input = x_stack.view(batch, seq, num_inputs * dim)

        weights = self.weight_grn(flat_input, context)  # (Batch, Seq, NumInputs)
        weights = torch.softmax(weights, dim=-1)  # (Batch, Seq, NumInputs)

        # 重みをブロードキャスト用に拡張: (Batch, Seq, NumInputs, 1)
        weights = weights.unsqueeze(-1)

        # 1. GRNを一括適用 (Batch*Seq*NumInputs, Dim) とみなして処理される
        # GatedResidualNetworkはLinearを使用しているため、末尾次元が合えば多次元テンソルも処理可能
        processed = self.shared_grn(x_stack, context)  # (Batch, Seq, NumInputs, Hidden)

        # 重み付け和
        weighted_sum = (processed * weights).sum(
            dim=2
        )  # NumInputs次元で和をとる -> (Batch, Seq, Hidden)

        return weighted_sum
