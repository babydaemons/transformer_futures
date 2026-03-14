# trade/simulator.py
"""
File: trade/simulator.py

ソースコードの役割:
本モジュールは、Transformerモデルの推論結果（売買確率）に基づき、日経平均先物ミニの
バックテストシミュレーションを実行します。ボラティリティ（ATR）やボラティリティ・レジームに応じた
動的なロットサイズ計算、TP/SL（利確・損切）判定、およびトレイリングストップ機能を備えています。
外部環境認識としてUSDJPYやS&P500のレートを考慮したモデル出力を前提としています。
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, Any, Optional, Tuple

from config import GlobalConfig, BAR_SECONDS

# リネームしたモジュールからインポート
from backtest.fast_sim import simulate_fast


def calculate_position_size(
    cfg: GlobalConfig,
    probs_action: np.ndarray,
    th_trade: float,
    atrs: np.ndarray,
    entry_mask: np.ndarray,
) -> np.ndarray:
    """シグナルの確信度やボラティリティ(ATR)に基づいた動的なロットサイズ(Lots)の計算を行う。

    Args:
        cfg (GlobalConfig): システム設定オブジェクト。
        probs_action (np.ndarray): アクション（エントリー）の予測確率配列。
        th_trade (float): エントリーを実行する確率閾値。
        atrs (np.ndarray): ATR（Average True Range）の配列。
        entry_mask (np.ndarray): エントリーが発生した箇所を示すマスク。

    Returns:
        np.ndarray: 各エントリーにおける計算済みロット数（float32）。
    """
    base_lots = cfg.backtest.base_lots
    max_lots = cfg.backtest.max_lots
    conf_scale = cfg.backtest.confidence_scale

    # 閾値を超えた分の確率をベースに追加ロットを計算
    excess_prob = probs_action[entry_mask] - th_trade
    extra_lots_conf = np.floor(np.maximum(0, excess_prob) / conf_scale)

    # ボラティリティ(ATR)が閾値を超えている場合に追加ロットを付与
    atr_e = atrs[entry_mask]
    atr_th = cfg.backtest.atr_threshold
    extra_lots_atr = np.where(atr_e > atr_th, 1.0, 0.0)

    # 計算されたロット数をbase_lotsとmax_lotsの間にクリップする
    lots = np.clip(base_lots + extra_lots_conf + extra_lots_atr, 1.0, max_lots).astype(
        np.float32
    )
    return lots


def _calculate_dynamic_sl(
    pe: np.ndarray,
    fav_px_path: np.ndarray,
    initial_sl_px: np.ndarray,
    atr_e: np.ndarray,
    horizon: int,
    act_mult: float,
    drop_mult: float,
    is_short: bool = False,
) -> np.ndarray:
    """トレイリングストップ（動的SL）の価格パスを計算する。

    Args:
        pe (np.ndarray): エントリー価格。
        fav_px_path (np.ndarray): 有利な方向の価格推移（LongならHigh, ShortならLow）。
        initial_sl_px (np.ndarray): 初期SL価格。
        atr_e (np.ndarray): エントリー時点のATR。
        horizon (int): 最大保持バー数。
        act_mult (float): トレイリング開始マルチプライヤー。
        drop_mult (float): トレイリングドロップマルチプライヤー。
        is_short (bool): ショートポジションフラグ。

    Returns:
        np.ndarray: 各バーにおける動的SL価格の行列。
    """
    act_amt = act_mult * atr_e
    drop_amt = drop_mult * atr_e

    # 時間減衰（Time-Decay）の適用: 後半になるほどドロップ幅を狭める
    time_decay_factor = 1.0 - (np.arange(fav_px_path.shape[1]) / horizon) * 0.5
    drop_amt_decayed = drop_amt[:, None] * time_decay_factor[None, :]

    if not is_short:
        # Long: 高値更新（MFE）を追跡
        fav_cum = np.maximum.accumulate(fav_px_path, axis=1)
        act_mask = (fav_cum - pe[:, None]) >= act_amt[:, None]
        dynamic_sl = np.where(
            act_mask, fav_cum - drop_amt_decayed, initial_sl_px[:, None]
        )
        dynamic_sl = np.maximum.accumulate(dynamic_sl, axis=1)
        dynamic_sl = np.maximum(dynamic_sl, initial_sl_px[:, None])
    else:
        # Short: 安値更新（MFE）を追跡
        fav_cum = np.minimum.accumulate(fav_px_path, axis=1)
        act_mask = (pe[:, None] - fav_cum) >= act_amt[:, None]
        dynamic_sl = np.where(
            act_mask, fav_cum + drop_amt_decayed, initial_sl_px[:, None]
        )
        dynamic_sl = np.minimum.accumulate(dynamic_sl, axis=1)
        dynamic_sl = np.minimum(dynamic_sl, initial_sl_px[:, None])

    return dynamic_sl


def _calculate_directional_pnl(
    cfg: GlobalConfig,
    is_short: bool,
    pe: np.ndarray,
    fh: np.ndarray,
    fl: np.ndarray,
    px: np.ndarray,
    act_horizon: np.ndarray,
    tp_arr: np.ndarray,
    sl_arr: np.ndarray,
    atr_e: np.ndarray,
    total_cost: float,
    min_hold_bars: int,
    min_exit_idx: int,
    horizon: int,
    trailing_act_mult: Optional[float] = None,
    trailing_drop_mult: Optional[float] = None,
) -> np.ndarray:
    """特定の方向（ロングまたはショート）のトレードPnLを計算する共通コアロジック。"""
    n_samples = len(pe)
    h = fh.shape[1]
    idx = np.arange(h, dtype=np.int32)[None, :]
    inf = h + 1

    if not is_short:
        # Long
        upper_px = pe + tp_arr
        lower_px = pe - sl_arr
        fav_px_path = fh
        unfav_px_path = fl
    else:
        # Short
        upper_px = pe + sl_arr
        lower_px = pe - tp_arr
        fav_px_path = fl
        unfav_px_path = fh

    # ストップロスの動的計算
    use_ts = cfg.backtest.use_trailing_stop and cfg.backtest.use_dynamic_sl_tp
    if use_ts:
        act_m = (
            trailing_act_mult
            if trailing_act_mult is not None
            else cfg.backtest.trailing_act_mult
        )
        drop_m = (
            trailing_drop_mult
            if trailing_drop_mult is not None
            else cfg.backtest.trailing_drop_mult
        )
        dynamic_sl = _calculate_dynamic_sl(
            pe,
            fav_px_path,
            (lower_px if not is_short else upper_px),
            atr_e,
            horizon,
            act_m,
            drop_m,
            is_short=is_short,
        )
        if not is_short:
            hit_sl = unfav_px_path <= dynamic_sl
        else:
            hit_sl = unfav_px_path >= dynamic_sl
    else:
        if not is_short:
            hit_sl = unfav_px_path <= lower_px[:, None]
        else:
            hit_sl = unfav_px_path >= upper_px[:, None]

    # 利確判定
    if not is_short:
        hit_tp = fav_px_path >= upper_px[:, None]
    else:
        hit_tp = fav_px_path <= lower_px[:, None]

    tp_mat = np.where(hit_tp, idx, inf)
    sl_mat = np.where(hit_sl, idx, inf)

    # 指定された最小ホールド期間より前の決済を無効化 (インデックスベース)
    if min_exit_idx > 0:
        tp_mat = np.where(idx < min_exit_idx, inf, tp_mat)
        sl_mat = np.where(idx < min_exit_idx, inf, sl_mat)

    idx_tp = tp_mat.min(axis=1)
    idx_sl = sl_mat.min(axis=1)

    # ホライゾン制約の適用
    h_eff = act_horizon.astype(np.int32, copy=False)
    idx_tp = np.where(idx_tp < h_eff, idx_tp, inf)
    idx_sl = np.where(idx_sl < h_eff, idx_sl, inf)

    tp_first = idx_tp < idx_sl
    sl_first = idx_sl < idx_tp

    # PnLの算出（決済価格の決定）
    if not is_short:
        pnl_base = (px - pe) - total_cost
        sl_exit_val = (
            (lambda idx_s: dynamic_sl[np.arange(n_samples), idx_s] - pe)
            if use_ts
            else (lambda _: -sl_arr)
        )
        tp_val = tp_arr
    else:
        pnl_base = (pe - px) - total_cost
        sl_exit_val = (
            (lambda idx_s: pe - dynamic_sl[np.arange(n_samples), idx_s])
            if use_ts
            else (lambda _: -sl_arr)
        )
        tp_val = tp_arr

    pnl = pnl_base
    pnl[tp_first] = tp_val[tp_first] - total_cost
    if sl_first.any():
        idx_for_sl = np.where(sl_first, idx_sl, 0)
        pnl[sl_first] = sl_exit_val(idx_for_sl)[sl_first] - total_cost

    return pnl


def evaluate_tp_sl(
    cfg: GlobalConfig,
    pe: np.ndarray,
    fh: np.ndarray,
    fl: np.ndarray,
    px: np.ndarray,
    act_horizon: np.ndarray,
    tp_arr: np.ndarray,
    sl_arr: np.ndarray,
    atrs: np.ndarray,
    entry_mask: np.ndarray,
    is_short: np.ndarray,
    total_cost: float,
    min_hold_bars: int,
    min_exit_idx: int,
    horizon: int,
    trailing_act_mult: float = None,
    trailing_drop_mult: float = None,
) -> np.ndarray:
    """Take Profit (TP) と Stop Loss (SL) の判定を行い、PnLを計算する。

    内部でロングとショートの個別処理を _calculate_directional_pnl に委譲する。
    """
    pnl = np.zeros(len(pe), dtype=np.float32)
    h = fh.shape[1] if fh.ndim == 2 else 0

    if h > 0 and ((tp_arr > 0.0).any() or (sl_arr > 0.0).any()):
        # Long positions
        m_long = ~is_short
        if m_long.any():
            pnl[m_long] = _calculate_directional_pnl(
                cfg,
                False,
                pe[m_long],
                fh[m_long],
                fl[m_long],
                px[m_long],
                act_horizon[m_long],
                tp_arr[m_long],
                sl_arr[m_long],
                atrs[entry_mask][m_long],
                total_cost,
                min_hold_bars,
                min_exit_idx,
                horizon,
                trailing_act_mult,
                trailing_drop_mult,
            )

        # Short positions
        m_short = is_short
        if m_short.any():
            pnl[m_short] = _calculate_directional_pnl(
                cfg,
                True,
                pe[m_short],
                fh[m_short],
                fl[m_short],
                px[m_short],
                act_horizon[m_short],
                tp_arr[m_short],
                sl_arr[m_short],
                atrs[entry_mask][m_short],
                total_cost,
                min_hold_bars,
                min_exit_idx,
                horizon,
                trailing_act_mult,
                trailing_drop_mult,
            )
    else:
        # TP/SL設定が無効な場合: horizon到達時のclose価格で決済
        pnl = np.where(is_short, (pe - px) - total_cost, (px - pe) - total_cost).astype(
            np.float32, copy=False
        )

    return pnl


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

        self.cooldown_bars = int(cfg.features.predict_horizon)
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
        """バックテストに必要な価格パスとホライゾン情報を準備する"""
        pe = (
            self.data["p_next_opens"][entry_mask]
            if self.cfg.backtest.use_next_bar_entry
            else self.data["p_closes"][entry_mask]
        )
        fh = self.data["f_highs"][entry_mask]
        fl = self.data["f_lows"][entry_mask]
        fc = self.data["f_closes"][entry_mask]

        t_rem = self.data["time_to_closes"][entry_mask]
        act_horizon = np.clip(t_rem.astype(np.int32), 1, self.cooldown_bars)

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
        """動的または固定のTP/SL配列と総取引コストを構築する"""
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
        """バックテスト結果から各種パフォーマンス統計指標を算出する"""
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
            int(self.cooldown_bars),
        )

        # トレードが発生しなかった場合の早期リターン
        # 完全に -inf にすると最適化が停滞するため、微小な負のスコアを与えて区別可能にする
        if entry_mask.sum() == 0:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "dir_acc": 0.0,
                "score": -100.0,  # 変更: -inf から適度なペナルティ値へ
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

        # 3. 利確・損切・PnLの評価
        is_short = executed_preds == 2
        pnl = evaluate_tp_sl(
            self.cfg,
            pe,
            fh,
            fl,
            px,
            act_horizon,
            tp_arr,
            sl_arr,
            self.data["atrs"],
            entry_mask,
            is_short,
            total_cost,
            min_hold_bars,
            min_exit_idx,
            int(self.cfg.features.predict_horizon),
            opt_trailing_act,
            opt_trailing_drop,
        )

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
