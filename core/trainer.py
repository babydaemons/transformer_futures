# core/trainer.py
"""
File: core/trainer.py

ソースコードの役割:
本モジュールは、TemporalFusionTransformerモデルの学習ループおよび最適化プロセスを管理するトレーナークラスと関連ヘルパー関数を提供します。
クラス不均衡対策の動的重み計算や、PnL最適化損失、Focal Lossの適用など、モデルのパラメータ更新に特化した処理をカプセル化しています。
デバイス転送や損失計算の一部をヘルパーメソッドに分離し、学習ループの可読性を高めています。
検証(Validation)とバックテスト評価のロジックは `core.evaluator` の `ValidationEngine`,
`ModelCheckpointManager`, `OutofSampleRunner` に分離し、クラスの凝集度を高めています。
"""

import time
import logging
from typing import Tuple, Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

from core.losses import FocalLoss
from core.pnl_loss import calculate_train_pnl_loss, calculate_eval_pnl_loss
from core.evaluator import ValidationEngine, ModelCheckpointManager, OutofSampleRunner
from util.utils import EMA


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
            - dir_y: 取引方向 (0: Long, 1: Short) ※Neutralの場合も計算上の便宜のため値が残るがマスクで無視される
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
    モデルの訓練ループと最適化状態の管理を行うクラス。
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        device: torch.device,
        logger: logging.Logger,
    ):
        """Trainerの初期化。

        Args:
            model (nn.Module): 学習対象のPyTorchモデル
            cfg (Any): 設定オブジェクト
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

        # 検証、チェックポイント管理、およびOOS評価を担当するクラス群を初期化
        self.val_engine = ValidationEngine(
            model=self.model,
            cfg=self.cfg,
            device=self.device,
            logger=self.logger,
            compute_loss_fn=self._compute_batch_loss,
        )
        self.checkpoint_manager = ModelCheckpointManager(
            cfg=self.cfg, logger=self.logger
        )
        self.oos_runner = OutofSampleRunner(
            model=self.model, cfg=self.cfg, device=self.device, logger=self.logger
        )

    def _setup_optimizer_and_scheduler(self, loader_train: DataLoader):
        """AdamWオプティマイザとOneCycleLRスケジューラの初期化を行います。

        Args:
            loader_train (DataLoader): 学習データのデータローダー。1エポックあたりのステップ数計算に使用します。
        """
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
        if n_neutral / n_total < 0.3:
            self.logger.warning(
                "WARNING: Trade labels are dominant (Neutral < 30%). "
                "Consider shortening predict_horizon or increasing label_min_limit."
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

    def _transfer_batch_to_device(self, batch: Tuple, is_train: bool) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        バッチデータを指定されたデバイスに転送します。

        Args:
            batch (Tuple): データローダーからのバッチタプル
            is_train (bool): 訓練モードフラグ

        Returns:
            Tuple: デバイス転送済みの各テンソル (xc, xs, y, p_curr, p_next_open, f_closes, curr_atr, p_exit)
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

        if is_train and p_exit is not None:
            p_exit = p_exit.to(self.device, non_blocking=True)

        return xc, xs, y, p_curr, p_next_open, f_closes, curr_atr, p_exit

    def _compute_pnl_loss_component(
        self,
        is_train: bool,
        probs_short: torch.Tensor,
        probs_action: torch.Tensor,
        p_exit: Optional[torch.Tensor],
        p_curr: torch.Tensor,
        curr_atr: torch.Tensor,
        sltp_preds: Optional[torch.Tensor],
        f_closes: torch.Tensor,
        p_next_open: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        予測確率や価格情報からPnLベースの損失項を計算します。

        Args:
            is_train (bool): 訓練モードフラグ
            probs_short (torch.Tensor): ショート方向の予測確率
            probs_action (torch.Tensor): 取引アクションの予測確率
            p_exit (Optional[torch.Tensor]): 訓練用エグジット価格
            p_curr (torch.Tensor): 現在の価格
            curr_atr (torch.Tensor): 現在のATR
            sltp_preds (Optional[torch.Tensor]): SL/TPの予測値
            f_closes (torch.Tensor): 未来の終値シーケンス
            p_next_open (torch.Tensor): 次の足の始値

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (PnL損失, SL/TP正則化項)
        """
        if is_train:
            # 次の足の始値でエントリーするかどうかの設定に基づいてエントリー価格を決定
            entry_price = (
                p_next_open if bool(self.cfg.backtest.use_next_bar_entry) else p_curr
            )
            pnl_loss, sltp_reg, self.global_pnl_var = calculate_train_pnl_loss(
                probs_short=probs_short,
                probs_action=probs_action,
                p_exit=p_exit,
                p_curr=entry_price,
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

        return pnl_loss, sltp_reg

    def _compute_batch_loss(
        self,
        batch: Tuple,
        is_train: bool,
        criterion_trade: nn.Module,
        criterion_dir: nn.Module,
    ) -> torch.Tensor:
        """バッチデータに対する順伝播および損失関数の計算を行います。

        Trade(取引有無)とDirection(方向)のTwo-Stage分類の損失に加え、
        設定に応じて方向性ペナルティおよびPnLベースの損失を合算して返します。

        Args:
            batch (Tuple): DataLoaderから取得したバッチデータのタプル。
            is_train (bool): 訓練モードかどうかのフラグ。
            criterion_trade (nn.Module): Trade予測用損失関数。
            criterion_dir (nn.Module): Direction予測用損失関数。

        Returns:
            torch.Tensor: 計算された総合損失（スカラー値）。
        """
        # 1. デバイスへのデータ転送
        xc, xs, y, p_curr, p_next_open, f_closes, curr_atr, p_exit = (
            self._transfer_batch_to_device(batch, is_train)
        )

        trade_y, dir_y, dir_mask = split_two_stage_targets(y)

        # 2. 順伝播
        out = self.model(xc, xs)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            trade_logits, dir_logits, sltp_preds = out
        else:
            trade_logits, dir_logits = out
            sltp_preds = None

        # 3. ロジットの差分からSigmoid確率を算出 (0: Neutral/Long, 1: Action/Short を想定)
        trade_logit_diff = trade_logits[:, 1] - trade_logits[:, 0]
        dir_logit_diff = dir_logits[:, 1] - dir_logits[:, 0]
        probs_action = torch.sigmoid(trade_logit_diff)
        probs_short = torch.sigmoid(dir_logit_diff)

        # 4. 基本的な分類損失計算 (Trade & Dir)
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

        # 5. 方向性ペナルティ計算
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

        # 6. PnL Loss計算の適用
        if self.cfg.train.pnl_loss_weight > 0.0:
            pnl_loss, sltp_reg = self._compute_pnl_loss_component(
                is_train=is_train,
                probs_short=probs_short,
                probs_action=probs_action,
                p_exit=p_exit,
                p_curr=p_curr,
                curr_atr=curr_atr,
                sltp_preds=sltp_preds,
                f_closes=f_closes,
                p_next_open=p_next_open,
            )
            loss = loss + self.cfg.train.pnl_loss_weight * pnl_loss + sltp_reg

        return loss

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
        1Fold分の学習ループを実行し、OOSでのトレード履歴を返します。

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

                # Mixed Precision (AMP) による勾配スケーリングと逆伝播
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

            self.logger.info(
                f"Fold {fold_idx} Ep {epoch}: TrainLoss={total_loss / max(steps, 1):.4f}"
            )

            # --- Validation ---
            val_result = self.val_engine.run_validation(
                epoch=epoch,
                fold_idx=fold_idx,
                loader_val=loader_val,
                criterion_trade=criterion_trade,
                criterion_dir=criterion_dir,
                ema=self.ema,
            )

            # --- Early Stopping Check ---
            should_stop = self.checkpoint_manager.update_and_check_early_stopping(
                model=self.model, val_result=val_result, epoch=epoch, ema=self.ema
            )

            if should_stop:
                break

        # --- OOS Test ---
        oos_trades = self.oos_runner.run_oos_test(
            fold_idx=fold_idx,
            loader_val=loader_val,
            loader_test=loader_test,
            test_dates=test_dates,
            checkpoint_manager=self.checkpoint_manager,
        )

        return oos_trades
