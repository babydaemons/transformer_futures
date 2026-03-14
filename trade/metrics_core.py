# trade/metrics_core.py
"""
File: trade/metrics_core.py

ソースコードの役割:
本モジュールは、バックテストにおける純粋な計算ロジック（ポジションサイズの算出、
動的トレイリングストップの計算、方向別のPnL算出、TP/SL判定など）を提供する、
状態を持たないNumpyベースの関数群です。
データクラス（引数オブジェクト）を用いて、関数の引数過多を緩和しています。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from config import GlobalConfig


@dataclass(frozen=True)
class MarketPaths:
    """トレード計算に必要な価格パスや指標の配列データをまとめたデータクラス"""

    pe: np.ndarray  # エントリー価格 (Price Entry)
    fh: np.ndarray  # 高値パス (Future Highs)
    fl: np.ndarray  # 安値パス (Future Lows)
    px: np.ndarray  # 決済価格 (Price Exit)
    act_horizon: np.ndarray  # 実効ホライゾン期間
    tp_arr: np.ndarray  # 利確(TP)幅
    sl_arr: np.ndarray  # 損切(SL)幅
    atr_e: np.ndarray  # エントリー時点のATR


@dataclass(frozen=True)
class TradeParams:
    """トレード計算に必要なスカラー値のパラメータ群"""

    total_cost: float
    min_hold_bars: int
    min_exit_idx: int
    horizon: int
    trailing_act_mult: Optional[float] = None
    trailing_drop_mult: Optional[float] = None


def calculate_position_size(
    cfg: GlobalConfig,
    probs_action: np.ndarray,
    th_trade: float,
    atrs: np.ndarray,
    entry_mask: np.ndarray,
) -> np.ndarray:
    """シグナルの確信度やボラティリティ(ATR)に基づいた動的なロットサイズ(Lots)の計算を行う。

    Args:
        cfg (GlobalConfig): グローバル設定オブジェクト。
        probs_action (np.ndarray): モデルが予測した各アクションの確率配列。
        th_trade (float): トレードを実行するための確率の閾値。
        atrs (np.ndarray): エントリー時点のATR配列。
        entry_mask (np.ndarray): エントリー条件を満たしているかを示すブール配列。

    Returns:
        np.ndarray: 計算された動的なポジションサイズ（ロット数）の配列。
    """
    base_lots = cfg.backtest.base_lots
    max_lots = cfg.backtest.max_lots
    conf_scale = cfg.backtest.confidence_scale

    # 確信度による追加ロットの計算
    excess_prob = probs_action[entry_mask] - th_trade
    extra_lots_conf = np.floor(np.maximum(0, excess_prob) / conf_scale)

    # ボラティリティ（ATR）による追加ロットの計算
    atr_e = atrs[entry_mask]
    atr_th = cfg.backtest.atr_threshold
    extra_lots_atr = np.where(atr_e > atr_th, 1.0, 0.0)

    # ロット数を制限内にクリップ
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
        pe (np.ndarray): エントリー価格の配列。
        fav_px_path (np.ndarray): 有利な方向の価格パス（ロングなら高値、ショートなら安値）。
        initial_sl_px (np.ndarray): 初期ストップロス価格の配列。
        atr_e (np.ndarray): エントリー時点のATR配列。
        horizon (int): トレードの最大ホライゾン期間。
        act_mult (float): トレイリングストップを発動させるためのATR乗数。
        drop_mult (float): トレイリングストップのドロップ幅（許容下落幅）を決めるATR乗数。
        is_short (bool, optional): ショートポジションの場合はTrue。デフォルトはFalse。

    Returns:
        np.ndarray: 各ステップにおける動的ストップロスの価格パス。
    """
    act_amt = act_mult * atr_e
    drop_amt = drop_mult * atr_e

    # 時間経過による許容ドロップ幅の減衰（終盤ほどタイトにする）
    time_decay_factor = 1.0 - (np.arange(fav_px_path.shape[1]) / horizon) * 0.5
    drop_amt_decayed = drop_amt[:, None] * time_decay_factor[None, :]

    if not is_short:
        fav_cum = np.maximum.accumulate(fav_px_path, axis=1)
        act_mask = (fav_cum - pe[:, None]) >= act_amt[:, None]
        dynamic_sl = np.where(
            act_mask, fav_cum - drop_amt_decayed, initial_sl_px[:, None]
        )
        # SLは切り上がりのみ（下がることはない）
        dynamic_sl = np.maximum.accumulate(dynamic_sl, axis=1)
        dynamic_sl = np.maximum(dynamic_sl, initial_sl_px[:, None])
    else:
        fav_cum = np.minimum.accumulate(fav_px_path, axis=1)
        act_mask = (pe[:, None] - fav_cum) >= act_amt[:, None]
        dynamic_sl = np.where(
            act_mask, fav_cum + drop_amt_decayed, initial_sl_px[:, None]
        )
        # SLは切り下がりのみ（上がることはない）
        dynamic_sl = np.minimum.accumulate(dynamic_sl, axis=1)
        dynamic_sl = np.minimum(dynamic_sl, initial_sl_px[:, None])

    return dynamic_sl


def _calculate_directional_pnl(
    cfg: GlobalConfig,
    is_short: bool,
    paths: MarketPaths,
    params: TradeParams,
) -> np.ndarray:
    """特定の方向（ロングまたはショート）のトレードPnLを計算する共通コアロジック。

    Args:
        cfg (GlobalConfig): グローバル設定オブジェクト。
        is_short (bool): ショート方向の計算を行う場合はTrue。
        paths (MarketPaths): 価格パスや指標をまとめたデータクラス。
        params (TradeParams): トレードに関するスカラーパラメータ群。

    Returns:
        np.ndarray: 計算されたPnLの配列。
    """
    n_samples = len(paths.pe)
    h = paths.fh.shape[1]
    idx = np.arange(h, dtype=np.int32)[None, :]
    inf = h + 1

    if not is_short:
        upper_px = paths.pe + paths.tp_arr
        lower_px = paths.pe - paths.sl_arr
        fav_px_path = paths.fh
        unfav_px_path = paths.fl
    else:
        upper_px = paths.pe + paths.sl_arr
        lower_px = paths.pe - paths.tp_arr
        fav_px_path = paths.fl
        unfav_px_path = paths.fh

    use_ts = cfg.backtest.use_trailing_stop and cfg.backtest.use_dynamic_sl_tp
    if use_ts:
        act_m = (
            params.trailing_act_mult
            if params.trailing_act_mult is not None
            else cfg.backtest.trailing_act_mult
        )
        drop_m = (
            params.trailing_drop_mult
            if params.trailing_drop_mult is not None
            else cfg.backtest.trailing_drop_mult
        )

        dynamic_sl = _calculate_dynamic_sl(
            paths.pe,
            fav_px_path,
            (lower_px if not is_short else upper_px),
            paths.atr_e,
            params.horizon,
            act_m,
            drop_m,
            is_short=is_short,
        )
        hit_sl = (
            unfav_px_path <= dynamic_sl if not is_short else unfav_px_path >= dynamic_sl
        )
    else:
        hit_sl = (
            unfav_px_path <= lower_px[:, None]
            if not is_short
            else unfav_px_path >= upper_px[:, None]
        )

    hit_tp = (
        fav_px_path >= upper_px[:, None]
        if not is_short
        else fav_px_path <= lower_px[:, None]
    )

    tp_mat = np.where(hit_tp, idx, inf)
    sl_mat = np.where(hit_sl, idx, inf)

    # 最小ホールド期間の適用
    if params.min_exit_idx > 0:
        tp_mat = np.where(idx < params.min_exit_idx, inf, tp_mat)
        sl_mat = np.where(idx < params.min_exit_idx, inf, sl_mat)

    idx_tp = tp_mat.min(axis=1)
    idx_sl = sl_mat.min(axis=1)

    h_eff = paths.act_horizon.astype(np.int32, copy=False)
    idx_tp = np.where(idx_tp < h_eff, idx_tp, inf)
    idx_sl = np.where(idx_sl < h_eff, idx_sl, inf)

    tp_first = idx_tp < idx_sl
    sl_first = idx_sl < idx_tp

    # 基本PnL（ホライゾン到達時）の計算
    if not is_short:
        pnl_base = (paths.px - paths.pe) - params.total_cost
        sl_exit_val = (
            (lambda idx_s: dynamic_sl[np.arange(n_samples), idx_s] - paths.pe)
            if use_ts
            else (lambda _: -paths.sl_arr)
        )
    else:
        pnl_base = (paths.pe - paths.px) - params.total_cost
        sl_exit_val = (
            (lambda idx_s: paths.pe - dynamic_sl[np.arange(n_samples), idx_s])
            if use_ts
            else (lambda _: -paths.sl_arr)
        )

    pnl = pnl_base
    # TP到達時のPnL上書き
    pnl[tp_first] = paths.tp_arr[tp_first] - params.total_cost
    # SL到達時のPnL上書き
    if sl_first.any():
        idx_for_sl = np.where(sl_first, idx_sl, 0)
        pnl[sl_first] = sl_exit_val(idx_for_sl)[sl_first] - params.total_cost

    return pnl


def evaluate_tp_sl(
    cfg: GlobalConfig,
    paths: MarketPaths,
    params: TradeParams,
    is_short: np.ndarray,
) -> np.ndarray:
    """Take Profit (TP) と Stop Loss (SL) の判定を行い、最終的なPnLを計算する。

    Args:
        cfg (GlobalConfig): グローバル設定オブジェクト。
        paths (MarketPaths): 価格パスや指標をまとめたデータクラス。
        params (TradeParams): トレードに関するスカラーパラメータ群。
        is_short (np.ndarray): 各トレードがショートであるかを示すブール配列。

    Returns:
        np.ndarray: 各トレードの計算済みPnL配列。
    """
    pnl = np.zeros(len(paths.pe), dtype=np.float32)
    h = paths.fh.shape[1] if paths.fh.ndim == 2 else 0

    if h > 0 and ((paths.tp_arr > 0.0).any() or (paths.sl_arr > 0.0).any()):
        m_long = ~is_short
        if m_long.any():
            long_paths = MarketPaths(
                paths.pe[m_long],
                paths.fh[m_long],
                paths.fl[m_long],
                paths.px[m_long],
                paths.act_horizon[m_long],
                paths.tp_arr[m_long],
                paths.sl_arr[m_long],
                paths.atr_e[m_long],
            )
            pnl[m_long] = _calculate_directional_pnl(cfg, False, long_paths, params)

        m_short = is_short
        if m_short.any():
            short_paths = MarketPaths(
                paths.pe[m_short],
                paths.fh[m_short],
                paths.fl[m_short],
                paths.px[m_short],
                paths.act_horizon[m_short],
                paths.tp_arr[m_short],
                paths.sl_arr[m_short],
                paths.atr_e[m_short],
            )
            pnl[m_short] = _calculate_directional_pnl(cfg, True, short_paths, params)
    else:
        # TP/SLが設定されていない場合は単純なExit価格でのPnL計算
        pnl = np.where(
            is_short,
            (paths.pe - paths.px) - params.total_cost,
            (paths.px - paths.pe) - params.total_cost,
        ).astype(np.float32, copy=False)

    return pnl
