# core/trainer.py
"""
File: core/trainer.py

ソースコードの役割:
本モジュールは、TemporalFusionTransformerモデルの学習ループおよび検証プロセスを管理するトレーナークラスと関連ヘルパー関数を提供します。
PnL最適化損失、Focal Lossの適用、検証バックテストの実行など、学習に特化した処理をカプセル化しています。
設定オブジェクトによって属性参照の安全性が保証されているため、直接属性へアクセスし可読性とパフォーマンスを向上させています。
"""

import time
import copy
import math
import logging
import os
from typing import Tuple, Optional, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

from core.losses import FocalLoss
from util.utils import (
    PerfTimer,
    _perf_sync_if,
    _PERF_STEP_SAMPLE,
    _PERF_EVAL_SAMPLE,
    _PERF_GPU_STATS,
    PERF_LEVEL_NUM,
    EMA,
)
from trade import run_vectorized_backtest
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


class Trainer:
    """
    モデルの訓練、評価、およびEarly StoppingやEMA等の学習状態管理を行うクラス。
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        device: torch.device,
        logger: logging.Logger,
    ):
        """
        Trainerの初期化。

        Args:
            model (nn.Module): 学習対象のPyTorchモデル
            cfg (Any): 設定オブジェクト (Pydanticモデル等を想定)
            device (torch.device): 実行デバイス (CPU/GPU)
            logger (logging.Logger): ロガーインスタンス
        """
        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger

        # 勾配スケーラーとEMA（Exponential Moving Average）の初期化
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.train.use_amp)
        self.ema = (
            EMA(self.model, decay=self.cfg.train.ema_decay)
            if self.cfg.train.use_ema
            else None
        )
        self.global_pnl_var = 1.0

    def _setup_optimizer_and_scheduler(self, loader_train: DataLoader):
        """AdamWオプティマイザとOneCycleLRスケジューラの初期化を行います。"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.train.learning_rate,
            steps_per_epoch=len(loader_train),
            epochs=self.cfg.train.epochs,
            pct_start=0.3,
            anneal_strategy="cos",
        )

    def _calculate_class_weights(
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

    def _compute_batch_loss(
        self,
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
        out = self.model(xc, xs)
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

    def evaluate_loss(
        self, loader: DataLoader, criterion_trade: nn.Module, criterion_dir: nn.Module
    ) -> float:
        """
        検証用データでのLoss計算を行います。

        Args:
            loader (DataLoader): 検証用データローダー
            criterion_trade (nn.Module): 取引有無のLoss関数
            criterion_dir (nn.Module): 取引方向のLoss関数

        Returns:
            float: 平均損失値
        """
        self.model.eval()
        total_sum = None
        n = 0

        with torch.no_grad():
            for batch in loader:
                loss = self._compute_batch_loss(
                    batch,
                    is_train=False,
                    criterion_trade=criterion_trade,
                    criterion_dir=criterion_dir,
                )
                b = batch[2].size(0)
                total_sum = (
                    loss.detach().float() * b
                    if total_sum is None
                    else total_sum + loss.detach().float() * b
                )
                n += b

        return (total_sum / n).item() if n > 0 else 0.0

    def train_fold(
        self,
        loader_train: DataLoader,
        loader_val: DataLoader,
        loader_test: Optional[DataLoader],
        y_labels_tr: np.ndarray,
        fold_idx: int,
        test_dates: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        1Fold分の学習と評価バックテストを実行します。

        Args:
            loader_train (DataLoader): 学習用データローダー
            loader_val (DataLoader): 検証用データローダー
            loader_test (Optional[DataLoader]): テスト用データローダー
            y_labels_tr (np.ndarray): 学習用ラベル配列
            fold_idx (int): 現在のFold番号
            test_dates (List[Any]): テスト期間の日付リスト

        Returns:
            List[Dict[str, Any]]: OOS(Out of Sample)での取引履歴リスト
        """
        self._setup_optimizer_and_scheduler(loader_train)
        weight_trade, weight_dir = self._calculate_class_weights(y_labels_tr)

        if self.cfg.train.use_focal_loss:
            criterion_trade = FocalLoss(
                alpha=weight_trade, gamma=self.cfg.train.focal_gamma
            )
            criterion_dir = FocalLoss(
                alpha=weight_dir, gamma=self.cfg.train.focal_gamma
            )
        else:
            criterion_trade = nn.CrossEntropyLoss(weight=weight_trade)
            criterion_dir = nn.CrossEntropyLoss(weight=weight_dir)

        best_val_loss = float("inf")
        best_val_score = -float("inf")
        best_model_state = None
        bad_epochs = 0

        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            total_loss = 0.0
            steps = 0

            for batch in loader_train:
                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=self.cfg.train.use_amp):
                    loss = self._compute_batch_loss(
                        batch,
                        is_train=True,
                        criterion_trade=criterion_trade,
                        criterion_dir=criterion_dir,
                    )

                self.scaler.scale(loss).backward()

                if self.cfg.train.grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                if self.ema is not None:
                    self.ema.update(self.model)

                total_loss += float(loss.item())
                steps += 1

            # --- Validation ---
            if self.ema is not None:
                self.ema.apply_shadow(self.model)

            val_loss = self.evaluate_loss(loader_val, criterion_trade, criterion_dir)

            with PerfTimer(self.logger, f"fold{fold_idx}/epoch{epoch}/val_backtest"):
                backtest_res = run_vectorized_backtest(
                    self.model, loader_val, self.device, self.cfg, fold_idx=fold_idx
                )

            if self.ema is not None:
                self.ema.restore(self.model)

            # 辞書からの取得。バックテスト結果の辞書に対するものなので維持
            val_score = (
                float(backtest_res.get("score", -float("inf")))
                if backtest_res
                else -float("inf")
            )
            n_trades_val = int(backtest_res.get("n_trades", 0)) if backtest_res else 0

            self.logger.info(
                f"Fold {fold_idx} Ep {epoch}: TrainLoss={total_loss/max(steps,1):.4f} "
                f"ValLoss={val_loss:.4f} | ValScore={val_score:.4f} (Trades: {n_trades_val})"
            )

            score_improved = val_score > best_val_score + 1e-4
            score_equal = False
            if math.isfinite(val_score) and math.isfinite(best_val_score):
                score_equal = abs(val_score - best_val_score) < 1e-4
            elif math.isinf(val_score) and math.isinf(best_val_score):
                score_equal = val_score == best_val_score

            loss_improved_at_same_score = (
                score_equal
                and val_loss < best_val_loss - self.cfg.train.early_stopping_min_delta
            )

            if score_improved or loss_improved_at_same_score:
                if score_improved:
                    best_val_score = val_score
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if self.ema is not None:
                    self.ema.apply_shadow(self.model)
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    self.ema.restore(self.model)
                else:
                    best_model_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.train.patience and epoch > 5:
                    self.logger.info(
                        f"Early stopping at epoch {epoch} (Best Score: {best_val_score:.4f})"
                    )
                    break

        # --- OOS Test ---
        oos_trades = []
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            best_val_res = run_vectorized_backtest(
                self.model, loader_val, self.device, self.cfg, fold_idx
            )
            self.logger.info(
                f"Restored best model (Val Result) with Score={best_val_score:.4f}, Loss={best_val_loss:.4f}: {best_val_res}"
            )

            if loader_test is not None:
                val_th_trade = best_val_res.get("threshold_trade")
                val_th_dir = best_val_res.get("threshold_dir")
                trade_log_path = os.path.join(
                    self.cfg.output_dir, f"fold{fold_idx:04d}_test_{test_dates[0]}.tsv"
                )
                best_oos = run_vectorized_backtest(
                    self.model,
                    loader_test,
                    self.device,
                    self.cfg,
                    fold_idx,
                    fixed_threshold_trade=val_th_trade,
                    fixed_threshold_dir=val_th_dir,
                    trade_log_path=trade_log_path,
                )
                if "trades" in best_oos:
                    oos_trades = best_oos["trades"]
                self.logger.info(f"OOS Test Result ({test_dates[0]}): {best_oos}")

        return oos_trades
