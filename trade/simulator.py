# trade/simulator.py
"""
File: trade/simulator.py

ソースコードの役割:
本モジュールは、設定（GlobalConfig）や状態（self.data）を管理し、データ準備から
推論結果の統計処理までの一連のバックテストパイプラインを実行するシミュレーション実行層です。
内部の純粋な計算ロジックについては trade.metrics_core モジュールに委譲しています。
引数オブジェクト（MarketPaths, TradeParams）を利用して、クリーンなインターフェースを保ちます。
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, Any, Tuple

from config import GlobalConfig, BAR_SECONDS
from backtest.fast_sim import simulate_fast
from trade.metrics_core import (
    calculate_position_size,
    evaluate_tp_sl,
    MarketPaths,
    TradeParams,
)


class BacktestSimulator:
    """抽出された推論データに基づいてバックテストシミュレーションを実行する。

    各種パフォーマンス指標（PnL, PF, 勝率など）を計算し、閾値最適化に利用可能な
    スコアを算出します。
    """

    def __init__(self, data: Dict[str, np.ndarray], cfg: GlobalConfig):
        """シミュレータの初期化。

        Args:
            data (Dict[str, np.ndarray]): 推論結果およびマーケットデータを含む辞書。
            cfg (GlobalConfig): システム設定。
        """
        self.data = data
        self.cfg = cfg

        self.cooldown_bars = max(0, int(cfg.backtest.cooldown_bars))
        predict_horizon = max(1, int(cfg.features.predict_horizon))
        max_hold_bars = (
            max(
                1,
                int(
                    math.ceil(float(cfg.backtest.max_holding_sec) / float(BAR_SECONDS))
                ),
            )
            if cfg.backtest.max_holding_sec
            else predict_horizon
        )
        self.hold_horizon_bars = max(1, min(predict_horizon, max_hold_bars))
        self.min_tick_speed = cfg.backtest.min_tick_speed_ratio

        # --- Session-Based Threshold (お昼休みのフィルタリング) ---
        self.sim_probs_action = self.data["probs_action"].copy()
        if self.cfg.backtest.avoid_lunch_break:
            dt_jst = pd.to_datetime(
                self.data["curr_ts"], unit="ns", utc=True
            ).tz_convert("Asia/Tokyo")
            hour = dt_jst.hour.values
            minute = dt_jst.minute.values
            is_lunch = ((hour == 11) & (minute >= 30)) | ((hour == 12) & (minute < 30))
            self.sim_probs_action[is_lunch] = 0.0

    def _prepare_market_paths(
        self, entry_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """バックテストに必要な価格パスとホライゾン情報を準備する。

        Args:
            entry_mask (np.ndarray): エントリー発生箇所を示すマスク。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
            エントリー価格(pe)、高値パス(fh)、安値パス(fl)、決済価格(px)、実効ホライゾン(act_horizon)、
            最小保持バー数(min_hold_bars)、最小エグジットインデックス(min_exit_idx)のタプル。
        """
        pe = (
            self.data["p_next_opens"][entry_mask]
            if self.cfg.backtest.use_next_bar_entry
            else self.data["p_closes"][entry_mask]
        )
        fh = self.data["f_highs"][entry_mask]
        fl = self.data["f_lows"][entry_mask]
        fc = self.data["f_closes"][entry_mask]

        t_rem = np.clip(
            self.data["time_to_closes"][entry_mask].astype(np.int32, copy=False),
            0,
            None,
        )
        max_h = max(1, min(int(fc.shape[1]), int(self.hold_horizon_bars)))
        act_horizon = np.clip(t_rem, 1, max_h)

        min_hold_bars = (
            max(
                0,
                int(
                    math.ceil(
                        float(self.cfg.backtest.min_holding_sec) / float(BAR_SECONDS)
                    )
                ),
            )
            if self.cfg.backtest.min_holding_sec
            else 0
        )
        min_exit_idx = max(0, min_hold_bars - 1)

        row_idx = np.arange(len(fc))
        col_idx = np.minimum(act_horizon, fc.shape[1]) - 1
        px = fc[row_idx, col_idx]

        return pe, fh, fl, px, act_horizon, min_hold_bars, min_exit_idx

    def _build_tp_sl_arrays(
        self, entry_mask: np.ndarray, pe: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """動的または固定のTP/SL配列と総取引コストを構築する。

        Args:
            entry_mask (np.ndarray): エントリー発生箇所を示すマスク。
            pe (np.ndarray): エントリー価格配列。

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: TP配列、SL配列、および計算された総取引コストのタプル。
        """
        if self.cfg.backtest.use_dynamic_sl_tp:
            atr_e = self.data["atrs"][entry_mask]
            m_sl_e = np.clip(self.data["m_sl_arr"][entry_mask], 0.5, 5.0)
            sl_arr = m_sl_e * atr_e

            if self.cfg.backtest.use_take_profit:
                tp_floor_val = (
                    float(self.cfg.backtest.cost)
                    + float(self.cfg.backtest.slippage_tick) * 5.0
                    + float(self.cfg.backtest.tp_min_after_cost)
                )
                m_tp_base = np.maximum(
                    self.data["m_tp_arr"][entry_mask], self.cfg.backtest.tp_min_atr_mult
                )
                m_tp_e = np.clip(
                    m_tp_base + (tp_floor_val / np.maximum(atr_e, 1e-6)), 0.5, 10.0
                )
                tp_arr = m_tp_e * atr_e
            else:
                tp_arr = np.zeros(len(pe), dtype=np.float32)
        else:
            tp_arr = (
                np.full(len(pe), float(self.cfg.backtest.tp_price), dtype=np.float32)
                if self.cfg.backtest.use_take_profit
                else np.zeros(len(pe), dtype=np.float32)
            )
            sl_arr = np.full(
                len(pe), float(self.cfg.backtest.sl_price), dtype=np.float32
            )

        total_cost = self.cfg.backtest.cost + (self.cfg.backtest.slippage_tick * 5.0)
        if (
            self.cfg.backtest.use_take_profit
            and self.cfg.backtest.enforce_tp_min_after_cost
        ):
            tp_min = float(total_cost + float(self.cfg.backtest.tp_min_after_cost))
            tp_arr = np.where(tp_arr > 0.0, np.maximum(tp_arr, tp_min), tp_arr)

        return tp_arr, sl_arr, total_cost

    def _calculate_statistics(
        self,
        pnl: np.ndarray,
        lots: np.ndarray,
        executed_preds: np.ndarray,
        executed_labels: np.ndarray,
        raw_signals_count: int,
        total_cost: float,
        th_trade: float,
        th_dir: float,
        min_dir_conf: float,
        entry_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """バックテスト結果から各種パフォーマンス統計指標を算出する。

        Args:
            pnl (np.ndarray): 各トレードのPnL配列。
            lots (np.ndarray): 各トレードのロット数配列。
            executed_preds (np.ndarray): 実行された予測方向の配列。
            executed_labels (np.ndarray): 実際の正解ラベル配列。
            raw_signals_count (int): 生のシグナル発生回数。
            total_cost (float): 取引コスト。
            th_trade (float): エントリー閾値。
            th_dir (float): 方向性閾値。
            min_dir_conf (float): 最小方向性確信度。
            entry_mask (np.ndarray): エントリー発生箇所を示すマスク。

        Returns:
            Dict[str, Any]: バックテストのパフォーマンス統計指標をまとめた辞書。
        """
        pnl_jpy = pnl * lots * self.cfg.backtest.contract_multiplier

        n_trades = len(pnl)
        win_rate = float((pnl > 0).mean()) if n_trades > 0 else 0.0
        dir_acc = (
            float(
                (
                    executed_preds[executed_labels != 0]
                    == executed_labels[executed_labels != 0]
                ).mean()
            )
            if (executed_labels != 0).any()
            else 0.0
        )

        profits = float(pnl_jpy[pnl_jpy > 0].sum())
        losses = float((-pnl_jpy[pnl_jpy < 0]).sum())
        pf = profits / losses if losses > 0 else (float("inf") if profits > 0 else 0.0)

        # スコア計算
        signal_rate = raw_signals_count / max(len(self.data["probs_action"]), 1)
        pf_eff = (
            min(pf, self.cfg.backtest.pf_cap)
            if math.isfinite(pf)
            else self.cfg.backtest.pf_cap
        )
        pnl_score = (
            0.1
            if n_trades > 0
            and pnl_jpy.mean()
            < (total_cost * lots.mean() * self.cfg.backtest.contract_multiplier * 1.5)
            else 1.0
        )

        # 対数を用いたスコア計算。PFと取引回数、方向性精度を総合的に評価する
        base_score = math.log(pf_eff + 1e-6) * math.log(n_trades + 1) * pnl_score
        score = float(base_score - 0.25 * signal_rate + 0.05 * dir_acc)

        return {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "dir_acc": dir_acc,
            "pf": pf,
            "pnl": int(pnl_jpy.sum() + 0.1),
            "score": score,
            "threshold_trade": th_trade,
            "threshold_dir": th_dir,
            "min_dir_conf": float(min_dir_conf),
            "raw_signals_count": int(raw_signals_count),
            "entry_mask": entry_mask,
            "lots": lots,
        }

    def simulate_thresholds(
        self,
        th_trade: float,
        th_dir: float,
        opt_trailing_act: float = None,
        opt_trailing_drop: float = None,
    ) -> Dict[str, Any]:
        """指定された閾値でシミュレーションを実行し、統計情報を計算する。

        Args:
            th_trade (float): エントリー確率の閾値。
            th_dir (float): 方向性確信度の閾値。
            opt_trailing_act (float, optional): 最適化用トレイリング開始マルチプライヤー。
            opt_trailing_drop (float, optional): 最適化用トレイリングドロップマルチプライヤー。

        Returns:
            Dict[str, Any]: 統計情報（取引数、勝率、PnL、スコア等）を含む辞書。
        """
        min_dir_conf = 0.5
        if self.cfg.backtest.use_conditional_signals:
            min_dir_conf = max(min_dir_conf, float(th_dir))

        # シグナル生成
        entry_mask, raw_signals_count = simulate_fast(
            self.sim_probs_action,
            self.data["probs_short"],
            float(th_trade),
            float(min_dir_conf),
            int(self.cooldown_bars),
            self.data["f_closes"],
            self.data["tick_speeds"],
            self.data["time_to_closes"],
            self.min_tick_speed,
            self.data["spreads"],
            self.data["vol_regimes"],
            float(
                self.cfg.backtest.vol_scaling_intensity
                if self.cfg.backtest.use_vol_regime_scaling
                else 0.0
            ),
            int(self.hold_horizon_bars),
        )

        # トレードが発生しなかった場合の早期リターン
        if entry_mask.sum() == 0:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "dir_acc": 0.0,
                "score": -100.0,
                "pnl": 0,
                "threshold_trade": float(th_trade),
                "threshold_dir": float(th_dir),
                "min_dir_conf": float(min_dir_conf),
                "raw_signals_count": int(raw_signals_count),
                "entry_mask": entry_mask,
                "lots": np.array([]),
            }

        pred_dir = np.where(self.data["probs_short"] > 0.5, 2, 1)
        executed_preds = pred_dir[entry_mask]
        executed_labels = self.data["labels"][entry_mask]

        # 1. パスの準備
        pe, fh, fl, px, act_horizon, min_hold_bars, min_exit_idx = (
            self._prepare_market_paths(entry_mask)
        )

        # 2. TP/SL配列の構築
        tp_arr, sl_arr, total_cost = self._build_tp_sl_arrays(entry_mask, pe)

        # 3. 利確・損切・PnLの評価のためのデータクラス構築
        is_short = executed_preds == 2

        paths = MarketPaths(
            pe=pe,
            fh=fh,
            fl=fl,
            px=px,
            act_horizon=act_horizon,
            tp_arr=tp_arr,
            sl_arr=sl_arr,
            atr_e=self.data["atrs"][entry_mask],
        )

        params = TradeParams(
            total_cost=total_cost,
            min_hold_bars=min_hold_bars,
            min_exit_idx=min_exit_idx,
            horizon=int(self.cfg.features.predict_horizon),
            trailing_act_mult=opt_trailing_act,
            trailing_drop_mult=opt_trailing_drop,
        )

        pnl = evaluate_tp_sl(self.cfg, paths, params, is_short)

        # 4. ロット計算と統計処理
        lots = calculate_position_size(
            self.cfg, self.data["probs_action"], th_trade, self.data["atrs"], entry_mask
        )

        return self._calculate_statistics(
            pnl,
            lots,
            executed_preds,
            executed_labels,
            raw_signals_count,
            total_cost,
            th_trade,
            th_dir,
            min_dir_conf,
            entry_mask,
        )
