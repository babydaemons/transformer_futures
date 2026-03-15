# core/pnl_loss.py
"""
File: core/pnl_loss.py

ソースコードの役割:
本モジュールは、モデルの学習および検証フェーズにおいて、期待収益(PnL)に基づいた
カスタム損失関数（PnL Optimization Loss）の計算ロジックを提供します。
動的SL/TPの予測値を考慮したリターンのクリッピングや、EMAを用いた大域分散の安定化を行います。
"""

import math
import torch
from typing import Optional, Tuple


def calculate_train_pnl_loss(
    probs_short: torch.Tensor,
    probs_action: torch.Tensor,
    p_exit: torch.Tensor,
    p_curr: torch.Tensor,
    curr_atr: torch.Tensor,
    sltp_preds: Optional[torch.Tensor],
    cost: float,
    slippage_tick: float,
    tp_min_after_cost: float,
    use_dynamic_sl_tp: bool,
    global_pnl_var: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """学習フェーズにおけるPnL Lossを計算します。

    予測された取引方向とアクション確率に基づいて期待リターンを計算し、
    Sharpe Ratioに似た形式でリスク調整後リターンを最大化する損失関数として機能します。

    Args:
        probs_short (torch.Tensor): ショート方向の予測確率。
        probs_action (torch.Tensor): 取引実行（Action）の予測確率。
        p_exit (torch.Tensor): エグジット時の評価価格。
        p_curr (torch.Tensor): 現在の価格（エントリー価格のベース）。
        curr_atr (torch.Tensor): 現在のATR（ボラティリティ）。
        sltp_preds (Optional[torch.Tensor]): SL/TPヘッドの予測値テンソル。
        cost (float): 基本取引コスト。
        slippage_tick (float): スリッページティック数。
        tp_min_after_cost (float): コスト差引後の最低確保利幅。
        use_dynamic_sl_tp (bool): 動的SL/TPを使用するかどうかのフラグ。
        global_pnl_var (float): EMAによる大域的なPnLの分散値。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, float]:
            - pnl_loss (torch.Tensor): 計算されたPnL損失。
            - sltp_reg (torch.Tensor): SL/TP予測の正則化ペナルティ。
            - new_global_pnl_var (float): 更新された大域的PnL分散。
    """
    sltp_reg = probs_action.new_tensor(0.0)
    direction = 1.0 - 2.0 * probs_short
    position = probs_action * direction
    raw_return = p_exit - p_curr
    total_cost = cost + slippage_tick * 5.0

    if use_dynamic_sl_tp and (sltp_preds is not None):
        m_sl = torch.clamp(sltp_preds[:, 0], min=0.5, max=5.0)
        raw_m_tp = sltp_preds[:, 1]

        # フロア（最低限確保すべき利益幅）の計算
        tp_floor_val = total_cost + tp_min_after_cost
        tp_pred_raw = raw_m_tp * curr_atr
        loss_floor = torch.mean(torch.relu(tp_floor_val - tp_pred_raw))

        m_tp_floor = (tp_floor_val / curr_atr.clamp(min=1e-6)).to(raw_m_tp.dtype)
        m_tp = torch.clamp(raw_m_tp + m_tp_floor, min=0.5, max=10.0)

        sl_amt = m_sl * curr_atr
        tp_amt = m_tp * curr_atr

        # リターンをSL/TPの範囲にクリッピング
        directed_return = direction * raw_return
        clipped_return = torch.clamp(directed_return, min=-sl_amt, max=tp_amt)
        expected_pnl = position * clipped_return - probs_action * total_cost

        # SL/TP予測に対する正則化（極端な値の抑制）
        sltp_reg = (
            torch.mean((m_sl - 2.0) ** 2 + (raw_m_tp - 2.0) ** 2) * 0.05
            + 0.15 * loss_floor
        )
    else:
        expected_pnl = position * raw_return - probs_action * total_cost

    scaled_pnl = expected_pnl / 100.0
    batch_var = torch.var(scaled_pnl).detach().item()

    # EMAを用いて大域的な分散を更新（バッチごとのブレを吸収）
    new_global_pnl_var = global_pnl_var
    if not math.isnan(batch_var):
        pnl_ema_alpha = 0.05
        new_global_pnl_var = (
            1.0 - pnl_ema_alpha
        ) * global_pnl_var + pnl_ema_alpha * batch_var

    global_pnl_std = math.sqrt(max(new_global_pnl_var, 1e-8)) + 1e-6
    pnl_loss = -(torch.mean(scaled_pnl) / global_pnl_std)

    return pnl_loss, sltp_reg, new_global_pnl_var


def calculate_eval_pnl_loss(
    probs_short: torch.Tensor,
    probs_action: torch.Tensor,
    f_closes: torch.Tensor,
    p_next_open: torch.Tensor,
    curr_atr: torch.Tensor,
    sltp_preds: Optional[torch.Tensor],
    cost: float,
    slippage_tick: float,
    tp_min_after_cost: float,
    use_dynamic_sl_tp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """検証フェーズにおけるPnL Lossを計算します。

    学習時とは異なり、未来の価格推移（軌跡）を時間減衰重み付きで評価し、
    より保守的かつ現実的な期待リターンを算出します。

    Args:
        probs_short (torch.Tensor): ショート方向の予測確率。
        probs_action (torch.Tensor): 取引実行（Action）の予測確率。
        f_closes (torch.Tensor): 未来の終値の軌跡 (Batch, Horizon)。
        p_next_open (torch.Tensor): 次の足の始値（エントリー価格）。
        curr_atr (torch.Tensor): 現在のATR。
        sltp_preds (Optional[torch.Tensor]): SL/TPヘッドの予測値テンソル。
        cost (float): 基本取引コスト。
        slippage_tick (float): スリッページティック数。
        tp_min_after_cost (float): コスト差引後の最低確保利幅。
        use_dynamic_sl_tp (bool): 動的SL/TPを使用するかどうかのフラグ。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - pnl_loss (torch.Tensor): 計算された検証用PnL損失。
            - sltp_reg (torch.Tensor): SL/TP予測の正則化ペナルティ。
    """
    sltp_reg = probs_action.new_tensor(0.0)
    pred_dir_vec = 1.0 - 2.0 * probs_short
    return_curve = pred_dir_vec.unsqueeze(1) * (f_closes - p_next_open.unsqueeze(1))
    horizon_len = f_closes.size(1)

    # 時間減衰ファクターの適用（遠い未来の利益ほど割り引く）
    time_decay = (
        1.0
        - (torch.arange(horizon_len, device=p_next_open.device).float() / horizon_len)
        * 0.5
    )
    decayed_returns = return_curve * time_decay.unsqueeze(0)

    # 上位パーセンタイルの平均を取得（Soft-Maximum的なアプローチ）
    top_k = max(1, horizon_len // 10)
    raw_return = torch.topk(decayed_returns, k=top_k, dim=1).values.mean(dim=1)

    # スリッページを含めたトータルコストの計算
    total_cost_val = cost + slippage_tick * 5.0

    if use_dynamic_sl_tp and (sltp_preds is not None):
        m_sl = torch.clamp(sltp_preds[:, 0], min=0.5, max=5.0)
        raw_m_tp = sltp_preds[:, 1]

        tp_floor_val = total_cost_val + tp_min_after_cost
        m_tp_floor = (tp_floor_val / curr_atr.clamp(min=1e-6)).to(raw_m_tp.dtype)
        m_tp = torch.clamp(raw_m_tp + m_tp_floor, min=0.5, max=10.0)

        sl_amt = m_sl * curr_atr
        tp_amt = m_tp * curr_atr
        clipped_return = torch.clamp(raw_return, min=-sl_amt, max=tp_amt)
        expected_pnl = probs_action * (clipped_return - total_cost_val)
        sltp_reg = torch.mean((m_sl - 2.0) ** 2 + (raw_m_tp - 2.0) ** 2) * 0.05
    else:
        expected_pnl = probs_action * (raw_return - total_cost_val)

    scaled_pnl = expected_pnl / 100.0
    batch_std = torch.std(scaled_pnl).clamp(min=1e-6)
    pnl_loss = -(torch.mean(scaled_pnl) / batch_std)

    return pnl_loss, sltp_reg
