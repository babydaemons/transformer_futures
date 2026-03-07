# model/tft.py
"""
File: model/tft.py
Description: Temporal Fusion Transformer本体および出力ヘッド群を定義します。
"""

import torch
import torch.nn as nn
from model.blocks import GatedResidualNetwork, VariableSelectionNetwork


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer (TFT) - Rich Architecture.
    変更点:
    - LSTMをBidirectional (双方向) に変更し、文脈抽出能力を向上。
    - Deep Layer対応。
    """

    def __init__(
        self,
        num_continuous: int,
        num_static: int,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.4,
        num_classes: int = 3,
        num_layers: int = 2,
        num_trade_classes: int = 2,
        num_dir_classes: int = 2,
        atr_idx: int = -1,
        tp_floor_price: float = 0.0,
        use_tp_floor: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.atr_idx = int(atr_idx)
        self.tp_floor_price = float(tp_floor_price)
        self.use_tp_floor = bool(use_tp_floor)

        self.continuous_proj = nn.Conv1d(
            num_continuous,
            num_continuous * d_model,
            kernel_size=1,
            groups=num_continuous,
        )

        self.static_proj = nn.Linear(num_static, d_model)
        self.static_encoder = GatedResidualNetwork(
            d_model, d_model, d_model, dropout=dropout
        )

        self.vsn = VariableSelectionNetwork(
            num_inputs=num_continuous,
            input_size=d_model,
            hidden_size=d_model,
            dropout=dropout,
            context_size=d_model,
        )

        self.lstm = nn.LSTM(
            d_model,
            d_model,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            num_layers=num_layers,
            bidirectional=True,
        )

        self.lstm_proj = nn.Linear(d_model * 2, d_model)
        self.lstm_gate = GatedResidualNetwork(
            d_model, d_model, d_model, dropout=dropout, context_size=d_model
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.att_gate = GatedResidualNetwork(
            d_model, d_model, d_model, dropout=dropout, context_size=d_model
        )

        self.trade_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_trade_classes),
        )

        self.dir_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_dir_classes),
        )

        self.sltp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
            nn.Softplus(),
        )

    def forward(self, x_cont: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features = x_cont.shape

        c_static = self.static_encoder(self.static_proj(x_static))

        x_cont_t = x_cont.permute(0, 2, 1)
        x_proj = self.continuous_proj(x_cont_t)
        x_proj = x_proj.permute(0, 2, 1).view(
            batch_size, seq_len, num_features, self.d_model
        )

        x_vsn = self.vsn(x_proj, context=c_static)

        try:
            self.lstm.flatten_parameters()
        except Exception:
            pass

        x_lstm_raw, _ = self.lstm(x_vsn)
        x_lstm_proj = self.lstm_proj(x_lstm_raw)
        x_lstm = self.lstm_gate(x_lstm_proj + x_vsn, context=c_static)

        query = x_lstm[:, -1:, :]
        attn_out, _ = self.attention(query, x_lstm, x_lstm)
        attn_out = attn_out.squeeze(1)

        x_out = self.att_gate(attn_out + x_lstm[:, -1, :], context=c_static)

        trade_logits = self.trade_head(x_out)
        dir_logits = self.dir_head(x_out)
        sltp_preds = self.sltp_head(x_out)

        return trade_logits, dir_logits, sltp_preds
