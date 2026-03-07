# core/pnl_loss.py
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
    """
    学習フェーズにおけるPnL Lossを計算します。
    """
    sltp_reg = probs_action.new_tensor(0.0)
    direction = 1.0 - 2.0 * probs_short
    position = probs_action * direction
    raw_return = p_exit - p_curr
    total_cost = cost + slippage_tick * 5.0

    if use_dynamic_sl_tp and (sltp_preds is not None):
        m_sl = torch.clamp(sltp_preds[:, 0], min=0.5, max=5.0)
        raw_m_tp = sltp_preds[:, 1]

        tp_floor_val = total_cost + tp_min_after_cost
        tp_pred_raw = raw_m_tp * curr_atr
        loss_floor = torch.mean(torch.relu(tp_floor_val - tp_pred_raw))

        m_tp_floor = (tp_floor_val / curr_atr.clamp(min=1e-6)).to(raw_m_tp.dtype)
        m_tp = torch.clamp(raw_m_tp + m_tp_floor, min=0.5, max=10.0)

        sl_amt = m_sl * curr_atr
        tp_amt = m_tp * curr_atr

        directed_return = direction * raw_return
        clipped_return = torch.clamp(directed_return, min=-sl_amt, max=tp_amt)
        expected_pnl = position * clipped_return - probs_action * total_cost
        sltp_reg = (
            torch.mean((m_sl - 2.0) ** 2 + (raw_m_tp - 2.0) ** 2) * 0.05
            + 0.15 * loss_floor
        )
    else:
        expected_pnl = position * raw_return - probs_action * total_cost

    scaled_pnl = expected_pnl / 100.0
    batch_var = torch.var(scaled_pnl).detach().item()

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
    """
    検証フェーズにおけるPnL Lossを計算します。
    """
    sltp_reg = probs_action.new_tensor(0.0)
    pred_dir_vec = 1.0 - 2.0 * probs_short
    return_curve = pred_dir_vec.unsqueeze(1) * (f_closes - p_next_open.unsqueeze(1))
    horizon_len = f_closes.size(1)
    time_decay = (
        1.0
        - (torch.arange(horizon_len, device=p_next_open.device).float() / horizon_len)
        * 0.5
    )
    decayed_returns = return_curve * time_decay.unsqueeze(0)
    top_k = max(1, horizon_len // 10)
    raw_return = torch.topk(decayed_returns, k=top_k, dim=1).values.mean(dim=1)

    if use_dynamic_sl_tp and (sltp_preds is not None):
        m_sl = torch.clamp(sltp_preds[:, 0], min=0.5, max=5.0)
        raw_m_tp = sltp_preds[:, 1]
        total_cost_val = cost + slippage_tick * 5.0
        tp_floor_val = total_cost_val + tp_min_after_cost
        m_tp_floor = (tp_floor_val / curr_atr.clamp(min=1e-6)).to(raw_m_tp.dtype)
        m_tp = torch.clamp(raw_m_tp + m_tp_floor, min=0.5, max=10.0)

        sl_amt = m_sl * curr_atr
        tp_amt = m_tp * curr_atr
        clipped_return = torch.clamp(raw_return, min=-sl_amt, max=tp_amt)
        expected_pnl = probs_action * (clipped_return - cost)
        sltp_reg = torch.mean((m_sl - 2.0) ** 2 + (raw_m_tp - 2.0) ** 2) * 0.05
    else:
        expected_pnl = probs_action * (raw_return - cost)

    scaled_pnl = expected_pnl / 100.0
    batch_std = torch.std(scaled_pnl).clamp(min=1e-6)
    pnl_loss = -(torch.mean(scaled_pnl) / batch_std)

    return pnl_loss, sltp_reg
