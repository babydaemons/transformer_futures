# core/loss_calculator.py
"""
File: core/loss_calculator.py

ソースコードの役割:
本モジュールは、モデルの出力に対する損失計算ロジックを提供します。
不均衡データを調整するための動的クラス重み計算や、方向予測ペナルティ、
PnLベースのカスタム損失の統合計算を行います。
"""

import logging
from typing import Tuple, Optional, Any

import torch
import torch.nn as nn
import numpy as np

from core.losses import FocalLoss
from core.pnl_loss import calculate_train_pnl_loss, calculate_eval_pnl_loss


def split_two_stage_targets(
    y_3cls: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    3クラス分類のターゲットをTrade(取引有無)とDir(方向)に分割します。

    Args:
        y_3cls (torch.Tensor): 3クラスのターゲットラベル (0: Neutral, 1: Long, 2: Short)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - trade_y: 取引の有無 (0 or 1)
            - dir_y: 取引方向 (0: Long, 1: Short)
            - dir_mask: 方向損失を計算するためのブールマスク
    """
    trade_y = (y_3cls != 0).long()
    dir_mask = trade_y.bool()
    dir_y = (y_3cls == 2).long()  # 0:Long, 1:Short
    return trade_y, dir_y, dir_mask


def calculate_directional_penalty(
    trade_logits: torch.Tensor,
    dir_y: torch.Tensor,
    probs_short: torch.Tensor,
    dir_mask_f: torch.Tensor,
    dir_den: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """
    方向性予測に対するペナルティを計算します。

    Args:
        trade_logits (torch.Tensor): 取引のロジット
        dir_y (torch.Tensor): 正解の取引方向
        probs_short (torch.Tensor): ショート方向の予測確率
        dir_mask_f (torch.Tensor): マスク（float型）
        dir_den (torch.Tensor): マスクの合計（ゼロ除算防止用）
        margin (float): ペナルティマージン

    Returns:
        torch.Tensor: 計算されたペナルティ値
    """
    p_true = torch.where(dir_y == 1, probs_short, 1.0 - probs_short)
    if margin > 0.0:
        return (((torch.relu(margin - p_true)) ** 2) * dir_mask_f).sum() / dir_den
    return ((1.0 - p_true) * dir_mask_f).sum() / dir_den


class LossCalculator:
    """
    損失の計算やクラス重みの動的計算を行うクラス。
    """

    def __init__(self, cfg: Any, device: torch.device, logger: logging.Logger):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.global_pnl_var = 1.0

    def calculate_class_weights(
        self, y_labels_tr: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """学習データから動的なクラス重みを計算します。

        不均衡データ（Neutralクラスが多数を占める）に対応するため、
        頻度の低いTradeアクションに対してより大きな重みを動的に割り当てます。

        Args:
            y_labels_tr (np.ndarray): 学習データの正解ラベル配列

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Trade分類用とDirection分類用の重みテンソル
        """
        y_labels_t = torch.tensor(y_labels_tr, dtype=torch.long)
        trade_y, _, _ = split_two_stage_targets(y_labels_t)

        n_total = len(trade_y)
        n_trade = trade_y.sum().item()
        n_neutral = n_total - n_trade

        self.logger.info(
            f"Label Dist -- Neutral: {n_neutral} ({n_neutral/n_total:.1%}), Trade: {n_trade} ({n_trade/n_total:.1%})"
        )
        if n_neutral / n_total < 0.1:
            self.logger.warning(
                "WARNING: Extreme Class Imbalance detected! Trade labels are dominant. Consider shortening predict_horizon or increasing label_min_limit."
            )

        raw_ratio = n_neutral / (n_trade + 1e-8)
        base = raw_ratio**self.cfg.train.class_weight_power

        w_action = float(
            np.clip(
                base * self.cfg.train.trade_pos_weight_scale,
                self.cfg.train.class_weight_floor,
                self.cfg.train.class_weight_cap,
            )
        )
        w_neutral = float(
            np.clip(
                base * self.cfg.train.trade_neg_weight_scale,
                1.0,
                self.cfg.train.class_weight_cap,
            )
        )
        weight_trade = torch.tensor(
            [w_neutral, w_action], device=self.device, dtype=torch.float32
        )

        n_long = (y_labels_tr == 1).sum()
        n_short = (y_labels_tr == 2).sum()

        cap = self.cfg.train.class_weight_cap
        if n_long > 0 and n_short > 0:
            raw_ratio_dir = n_long / (n_short + 1e-8)
            w_short = raw_ratio_dir**self.cfg.train.class_weight_power
            w_long = (1.0 / raw_ratio_dir) ** self.cfg.train.class_weight_power

            w_short = float(np.clip(w_short, 1.0 / cap, cap))
            w_long = float(np.clip(w_long, 1.0 / cap, cap))

            mean_w = 0.5 * (w_long + w_short)
            w_long = float(w_long / max(mean_w, 1e-8))
            w_short = float(w_short / max(mean_w, 1e-8))

            weight_dir = torch.tensor(
                [w_long, w_short], device=self.device, dtype=torch.float32
            )
        else:
            weight_dir = torch.tensor(
                [1.0, 1.0], device=self.device, dtype=torch.float32
            )

        self.logger.info(
            f"Dynamic Weights -- Trade (Neutral/Action): {weight_trade.cpu().numpy()}, Dir: {weight_dir.cpu().numpy()}"
        )
        return weight_trade, weight_dir

    def compute_batch_loss(
        self,
        model: nn.Module,
        batch: Tuple,
        is_train: bool,
        criterion_trade: nn.Module,
        criterion_dir: nn.Module,
    ) -> torch.Tensor:
        """
        バッチデータに対する順伝播および損失関数の計算を行います。
        """
        xc, xs, y, p_curr, p_next_open, f_closes = batch[0:6]
        curr_atr = batch[10]

        # is_train時は p_exit を使用する
        p_exit = f_closes.mean(dim=1) if is_train else None

        xc = xc.to(self.device, non_blocking=True)
        xs = xs.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        p_curr = p_curr.to(self.device, non_blocking=True)
        p_next_open = p_next_open.to(self.device, non_blocking=True)
        f_closes = f_closes.to(self.device, non_blocking=True)
        curr_atr = curr_atr.to(self.device, non_blocking=True)
        if is_train:
            p_exit = p_exit.to(self.device, non_blocking=True)

        trade_y, dir_y, dir_mask = split_two_stage_targets(y)

        # 順伝播
        out = model(xc, xs)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            trade_logits, dir_logits, sltp_preds = out
        else:
            trade_logits, dir_logits = out
            sltp_preds = None

        trade_logit_diff = trade_logits[:, 1] - trade_logits[:, 0]
        dir_logit_diff = dir_logits[:, 1] - dir_logits[:, 0]
        probs_action = torch.sigmoid(trade_logit_diff)
        probs_short = torch.sigmoid(dir_logit_diff)

        # 基本的な損失計算 (Trade & Dir)
        loss_trade = criterion_trade(trade_logits, trade_y)
        dir_mask_f = dir_mask.to(dtype=trade_logits.dtype)
        dir_den = dir_mask_f.sum().clamp_min(1.0)

        if isinstance(criterion_dir, FocalLoss):
            ce = nn.functional.cross_entropy(dir_logits, dir_y, reduction="none")
            pt = torch.exp(-ce)
            # FocalLossのコンストラクタでregister_bufferされているため属性アクセスが可能
            if criterion_dir.alpha is not None:
                alpha_t = criterion_dir.alpha[dir_y]
                dir_loss_vec = alpha_t * (1 - pt) ** criterion_dir.gamma * ce
            else:
                dir_loss_vec = (1 - pt) ** criterion_dir.gamma * ce
        else:
            dir_loss_vec = nn.functional.cross_entropy(
                dir_logits, dir_y, reduction="none"
            )

        loss_dir = (dir_loss_vec * dir_mask_f).sum() / dir_den

        # 方向性ペナルティ計算
        dir_pen = trade_logits.new_tensor(0.0)
        if self.cfg.train.directional_penalty > 0.0:
            dir_pen = calculate_directional_penalty(
                trade_logits,
                dir_y,
                probs_short,
                dir_mask_f,
                dir_den,
                self.cfg.train.dir_conf_margin,
            )

        loss = (
            self.cfg.train.trade_loss_weight * loss_trade
            + self.cfg.train.dir_loss_weight * loss_dir
            + self.cfg.train.directional_penalty * dir_pen
        )

        # PnL Loss計算
        if self.cfg.train.pnl_loss_weight > 0.0:
            if is_train:
                pnl_loss, sltp_reg, self.global_pnl_var = calculate_train_pnl_loss(
                    probs_short=probs_short,
                    probs_action=probs_action,
                    p_exit=p_exit,
                    p_curr=p_curr,
                    curr_atr=curr_atr,
                    sltp_preds=sltp_preds,
                    cost=float(self.cfg.backtest.cost),
                    slippage_tick=float(self.cfg.backtest.slippage_tick),
                    tp_min_after_cost=float(self.cfg.backtest.tp_min_after_cost),
                    use_dynamic_sl_tp=bool(self.cfg.backtest.use_dynamic_sl_tp),
                    global_pnl_var=self.global_pnl_var,
                )
            else:
                pnl_loss, sltp_reg = calculate_eval_pnl_loss(
                    probs_short=probs_short,
                    probs_action=probs_action,
                    f_closes=f_closes,
                    p_next_open=p_next_open,
                    curr_atr=curr_atr,
                    sltp_preds=sltp_preds,
                    cost=float(self.cfg.backtest.cost),
                    slippage_tick=float(self.cfg.backtest.slippage_tick),
                    tp_min_after_cost=float(self.cfg.backtest.tp_min_after_cost),
                    use_dynamic_sl_tp=bool(self.cfg.backtest.use_dynamic_sl_tp),
                )

            loss = loss + self.cfg.train.pnl_loss_weight * pnl_loss + sltp_reg

        return loss
