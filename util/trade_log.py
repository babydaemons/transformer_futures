# util/trade_log.py
"""
File: util/trade_log.py

ソースコードの役割:
本モジュールは、バックテスト実行後のトレードごとの詳細な損益（PnL）、エントリー・エグジットのタイミング、
および決済理由（TP, SL, HORIZONなど）を計算し、TSV形式のログファイルとして出力する機能を提供します。
ミリ秒オーダーの出来高付き歩み値を基にしたスキャルピング〜デイトレード戦略において、
ポジションホールド中の動的なストップロス(SL)やテイクプロフィット(TP)の判定を行います。
"""

import os
import math
import logging
import datetime
import numpy as np
from typing import Dict, List, Any, Tuple
from config import GlobalConfig, BAR_SECONDS

logger = logging.getLogger(__name__)


def log_backtest_summary(
    best: Dict[str, Any], fold_idx: int, probs_action: np.ndarray
) -> None:
    """
    バックテストのパフォーマンスサマリーを標準出力(ログ)に記録する。

    Args:
        best (Dict[str, Any]): 最適化されたバックテスト結果
        fold_idx (int): クロスバリデーションのフォールドインデックス
        probs_action (np.ndarray): アクション予測確率の配列（統計情報出力用）
    """
    p_stats = np.percentile(probs_action, [50, 75, 90, 95, 99])
    stats_msg = (
        f"Max: {probs_action.max():.4f} | "
        f"p50: {p_stats[0]:.4f}, p90: {p_stats[2]:.4f}, p99: {p_stats[4]:.4f}"
    )

    # トレードが0回の場合は簡易ログを出力して終了
    if best["n_trades"] == 0:
        logging.info(
            f"[Fold {fold_idx} OOS] No trades executed. "
            f"(Raw Signals > {best.get('threshold_trade', 0.0):.3f}: {best.get('raw_signals_count', 0)}) | "
            f"Th_trade: {best.get('threshold_trade', 0.0):.3f} | Th_dir: {best.get('threshold_dir', 0.0):.3f} | "
            f"DirConf>={best.get('min_dir_conf', 0.0):.2f} | [Stats] {stats_msg}"
        )
        return

    # 通常のパフォーマンスメトリクスを出力
    logging.info(
        f"[Fold {fold_idx} OOS] Trades: {best['n_trades']:4d} (Raw: {best.get('raw_signals_count', 0)}) | "
        f"Win Rate: {best.get('win_rate', 0.0):.2%} | PF: {best.get('pf', 0.0):.3f} | "
        f"PnL: {best.get('pnl', 0.0):.2f} | Avg PnL: {best.get('avg_pnl', 0.0):.2f} | "
        f"Moved: {best.get('moved_count', 0):4d} | Neutral Rate: {best.get('neutral_rate', 0.0):.2%} | "
        f"TP<=Cost: {best.get('tp_under_cost_rate_raw', 0.0):.2%} | TPmin: {best.get('tp_min', 0.0):.1f} | "
        f"MinHold: {best.get('min_hold_bars', 0):d}b | Dir Acc: {best.get('dir_acc', 0.0):.2%} (Excl. Neutral) | "
        f"Avg Prob: {probs_action[best['entry_mask']].mean():.3f} | Th_trade: {best.get('threshold_trade', 0.0):.3f} | "
        f"Th_dir: {best.get('threshold_dir', 0.0):.3f} | DirConf>={best.get('min_dir_conf', 0.0):.2f} | "
        f"TS_Act: {best.get('trailing_act_mult', 0.0):.1f} | TS_Drop: {best.get('trailing_drop_mult', 0.0):.1f} | "
        f"Score: {best.get('score', 0.0):.4f} | [Stats] {stats_msg}"
    )


def _calculate_sl_tp(
    cfg: GlobalConfig, data: Dict[str, np.ndarray], idx_entry: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    設定に基づいてストップロス(SL)とテイクプロフィット(TP)、および合計コストを計算する。

    Args:
        cfg (GlobalConfig): グローバル設定オブジェクト。
        data (Dict[str, np.ndarray]): 各種推論・時系列データを含む辞書。
        idx_entry (np.ndarray): エントリーインデックスの配列。

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: TPの配列、SLの配列、合計コスト。
    """
    use_dyn = bool(cfg.backtest.use_dynamic_sl_tp)
    use_tp = bool(cfg.backtest.use_take_profit)

    # 動的SL/TPの計算 (ATRベース)
    if use_dyn:
        m_sl_e = np.clip(data["m_sl_arr"][idx_entry], 0.5, 5.0)
        atr_e = data["atrs"][idx_entry]
        sl_e = (m_sl_e * atr_e).astype(np.float32, copy=False)

        if use_tp:
            m_tp_base = np.maximum(
                data["m_tp_arr"][idx_entry], cfg.backtest.tp_min_atr_mult
            )
            m_tp_e = np.clip(m_tp_base, 0.5, 10.0)
            tp_e = (m_tp_e * atr_e).astype(np.float32, copy=False)
        else:
            tp_e = np.zeros(len(idx_entry), dtype=np.float32)
    else:
        # 固定値のSL/TP
        sl0 = float(cfg.backtest.sl_price)
        sl_e = np.full(len(idx_entry), sl0, dtype=np.float32)

        if use_tp:
            tp0 = float(cfg.backtest.tp_price)
            tp_e = np.full(len(idx_entry), tp0, dtype=np.float32)
        else:
            tp_e = np.zeros(len(idx_entry), dtype=np.float32)

    # 取引コストの算出 (手数料 + スリッページ)
    cost = float(cfg.backtest.cost)
    slippage_val = float(cfg.backtest.slippage_tick * 5.0)
    total_cost = cost + slippage_val

    # TPが設定されている場合、コストを上回る最低TP幅を強制適用
    if use_tp and cfg.backtest.enforce_tp_min_after_cost:
        tp_floor = float(total_cost + float(cfg.backtest.tp_min_after_cost))
        tp_e = np.where(tp_e > 0.0, np.maximum(tp_e, tp_floor), tp_e)

    return tp_e, sl_e, total_cost


def _evaluate_long_positions(
    preds_e: np.ndarray,
    pe: np.ndarray,
    fh: np.ndarray,
    fl: np.ndarray,
    act_h: np.ndarray,
    tp_e: np.ndarray,
    sl_e: np.ndarray,
    atr_e: np.ndarray,
    cfg: GlobalConfig,
    best: Dict[str, Any],
    horizon: int,
    min_exit_idx: int,
    min_hold_bars: int,
    exit_off: np.ndarray,
    exit_reason: np.ndarray,
    exit_px: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Longポジションの出口（TP/SL/Horizon到達）と決済価格を計算する。副作用を防ぐため新しい配列を返す。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 更新された (exit_off, exit_reason, exit_px)
    """
    out_off = exit_off.copy()
    out_reason = exit_reason.copy()
    out_px = exit_px.copy()

    # ロングポジションのマスク
    m_pos = preds_e == 1
    if not m_pos.any():
        return out_off, out_reason, out_px

    pe_pos = pe[m_pos]
    fh_pos = fh[m_pos]
    fl_pos = fl[m_pos]
    h_eff = act_h[m_pos]
    tp_pos = tp_e[m_pos]
    sl_pos = sl_e[m_pos]
    atr_pos = atr_e[m_pos]

    use_dyn = bool(cfg.backtest.use_dynamic_sl_tp)
    use_ts = cfg.backtest.use_trailing_stop

    # TP到達判定
    hit_tp = (fh_pos > (pe_pos[:, None] + tp_pos[:, None])) & (tp_pos[:, None] > 0.0)

    # トレーリングストップ(TS)または通常のSL判定
    if use_ts and use_dyn:
        act_m = best.get("trailing_act_mult", cfg.backtest.trailing_act_mult)
        drop_m = best.get("trailing_drop_mult", cfg.backtest.trailing_drop_mult)
        act_amt = act_m * atr_pos
        drop_amt = drop_m * atr_pos

        time_decay_factor = 1.0 - (np.arange(horizon) / horizon) * 0.5
        drop_amt_decayed = drop_amt[:, None] * time_decay_factor[None, :]

        fh_cummax = np.maximum.accumulate(fh_pos, axis=1)
        act_mask = (fh_cummax - pe_pos[:, None]) >= act_amt[:, None]

        # TS発動条件を満たせば価格を引き上げ
        dyn_sl = np.where(
            act_mask,
            fh_cummax - drop_amt_decayed,
            pe_pos[:, None] - sl_pos[:, None],
        )
        dyn_sl = np.maximum.accumulate(dyn_sl, axis=1)
        dyn_sl = np.maximum(dyn_sl, pe_pos[:, None] - sl_pos[:, None])
        hit_sl = (fl_pos < dyn_sl) & (sl_pos[:, None] > 0.0)
    else:
        hit_sl = (fl_pos < (pe_pos[:, None] - sl_pos[:, None])) & (
            sl_pos[:, None] > 0.0
        )

    # 決済タイミングの評価
    idx = np.arange(horizon, dtype=np.int32)[None, :]
    inf = horizon + 1
    tp_mat = np.where(hit_tp, idx, inf)
    sl_mat = np.where(hit_sl, idx, inf)

    if min_exit_idx > 0:
        tp_mat = np.where(idx < min_exit_idx, inf, tp_mat)
        sl_mat = np.where(idx < min_exit_idx, inf, sl_mat)

    idx_tp = tp_mat.min(axis=1)
    idx_sl = sl_mat.min(axis=1)

    idx_tp = np.where(idx_tp < h_eff, idx_tp, inf)
    idx_sl = np.where(idx_sl < h_eff, idx_sl, inf)

    if min_hold_bars > 0:
        idx_tp = np.where(idx_tp >= min_hold_bars, idx_tp, inf)
        idx_sl = np.where(idx_sl >= min_hold_bars, idx_sl, inf)

    tp_first = idx_tp < idx_sl
    sl_first = idx_sl < idx_tp

    pos_idx = np.flatnonzero(m_pos)

    # TP到達が先の場合の更新
    if tp_first.any():
        j = pos_idx[tp_first]
        out_off[j] = idx_tp[tp_first]
        out_reason[j] = "TP"
        out_px[j] = pe_pos[tp_first] + tp_pos[tp_first]

    # SL到達が先の場合の更新
    if sl_first.any():
        j = pos_idx[sl_first]
        out_off[j] = idx_sl[sl_first]
        out_reason[j] = "TRAILING_SL" if use_ts and use_dyn else "SL"
        if use_ts and use_dyn:
            idx_for_sl = idx_sl[sl_first]
            row_idx_m = np.arange(len(pe_pos))[sl_first]
            out_px[j] = dyn_sl[row_idx_m, idx_for_sl]
        else:
            out_px[j] = pe_pos[sl_first] - sl_pos[sl_first]

    return out_off, out_reason, out_px


def _evaluate_short_positions(
    preds_e: np.ndarray,
    pe: np.ndarray,
    fh: np.ndarray,
    fl: np.ndarray,
    act_h: np.ndarray,
    tp_e: np.ndarray,
    sl_e: np.ndarray,
    atr_e: np.ndarray,
    cfg: GlobalConfig,
    best: Dict[str, Any],
    horizon: int,
    min_exit_idx: int,
    min_hold_bars: int,
    exit_off: np.ndarray,
    exit_reason: np.ndarray,
    exit_px: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shortポジションの出口（TP/SL/Horizon到達）と決済価格を計算する。副作用を防ぐため新しい配列を返す。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 更新された (exit_off, exit_reason, exit_px)
    """
    out_off = exit_off.copy()
    out_reason = exit_reason.copy()
    out_px = exit_px.copy()

    # ショートポジションのマスク
    m_pos = preds_e == 2
    if not m_pos.any():
        return out_off, out_reason, out_px

    pe_pos = pe[m_pos]
    fh_pos = fh[m_pos]
    fl_pos = fl[m_pos]
    h_eff = act_h[m_pos]
    tp_pos = tp_e[m_pos]
    sl_pos = sl_e[m_pos]
    atr_pos = atr_e[m_pos]

    use_dyn = bool(cfg.backtest.use_dynamic_sl_tp)
    use_ts = cfg.backtest.use_trailing_stop

    # TP到達判定
    hit_tp = (fl_pos < (pe_pos[:, None] - tp_pos[:, None])) & (tp_pos[:, None] > 0.0)

    # トレーリングストップ(TS)または通常のSL判定
    if use_ts and use_dyn:
        act_m = best.get("trailing_act_mult", cfg.backtest.trailing_act_mult)
        drop_m = best.get("trailing_drop_mult", cfg.backtest.trailing_drop_mult)
        act_amt = act_m * atr_pos
        drop_amt = drop_m * atr_pos

        time_decay_factor = 1.0 - (np.arange(horizon) / horizon) * 0.5
        drop_amt_decayed = drop_amt[:, None] * time_decay_factor[None, :]

        fl_cummin = np.minimum.accumulate(fl_pos, axis=1)
        act_mask = (pe_pos[:, None] - fl_cummin) >= act_amt[:, None]

        # TS発動条件を満たせば価格を引き下げ
        dyn_sl = np.where(
            act_mask,
            fl_cummin + drop_amt_decayed,
            pe_pos[:, None] + sl_pos[:, None],
        )
        dyn_sl = np.minimum.accumulate(dyn_sl, axis=1)
        dyn_sl = np.minimum(dyn_sl, pe_pos[:, None] + sl_pos[:, None])
        hit_sl = (fh_pos > dyn_sl) & (sl_pos[:, None] > 0.0)
    else:
        hit_sl = (fh_pos > (pe_pos[:, None] + sl_pos[:, None])) & (
            sl_pos[:, None] > 0.0
        )

    # 決済タイミングの評価
    idx = np.arange(horizon, dtype=np.int32)[None, :]
    inf = horizon + 1
    tp_mat = np.where(hit_tp, idx, inf)
    sl_mat = np.where(hit_sl, idx, inf)

    if min_exit_idx > 0:
        tp_mat = np.where(idx < min_exit_idx, inf, tp_mat)
        sl_mat = np.where(idx < min_exit_idx, inf, sl_mat)

    idx_tp = tp_mat.min(axis=1)
    idx_sl = sl_mat.min(axis=1)

    idx_tp = np.where(idx_tp < h_eff, idx_tp, inf)
    idx_sl = np.where(idx_sl < h_eff, idx_sl, inf)

    if min_hold_bars > 0:
        idx_tp = np.where(idx_tp >= min_hold_bars, idx_tp, inf)
        idx_sl = np.where(idx_sl >= min_hold_bars, idx_sl, inf)

    tp_first = idx_tp < idx_sl
    sl_first = idx_sl < idx_tp

    pos_idx = np.flatnonzero(m_pos)

    # TP到達が先の場合の更新
    if tp_first.any():
        j = pos_idx[tp_first]
        out_off[j] = idx_tp[tp_first]
        out_reason[j] = "TP"
        out_px[j] = pe_pos[tp_first] - tp_pos[tp_first]

    # SL到達が先の場合の更新
    if sl_first.any():
        j = pos_idx[sl_first]
        out_off[j] = idx_sl[sl_first]
        out_reason[j] = "TRAILING_SL" if use_ts and use_dyn else "SL"
        if use_ts and use_dyn:
            idx_for_sl = idx_sl[sl_first]
            row_idx_m = np.arange(len(pe_pos))[sl_first]
            out_px[j] = dyn_sl[row_idx_m, idx_for_sl]
        else:
            out_px[j] = pe_pos[sl_first] + sl_pos[sl_first]

    return out_off, out_reason, out_px


def _export_trades_to_tsv(
    trade_log_path: str,
    preds_e: np.ndarray,
    pe: np.ndarray,
    exit_px: np.ndarray,
    entry_ts: np.ndarray,
    exit_ts: np.ndarray,
    exit_off: np.ndarray,
    exit_reason: np.ndarray,
    lots: np.ndarray,
    total_cost: float,
    multiplier: float,
) -> List[Dict[str, Any]]:
    """
    最終的な損益（PNL）の計算と、TSVファイルへの書き込み処理を担当する。

    Returns:
        List[Dict[str, Any]]: 各トレードの詳細を格納した辞書のリスト。
    """
    pnl = np.zeros(len(preds_e), dtype=np.float32)
    m_long = preds_e == 1
    m_short = preds_e == 2

    # 方向ごとのPnLを計算 (コスト控除済み)
    pnl[m_long] = (exit_px[m_long] - pe[m_long]) - total_cost
    pnl[m_short] = (pe[m_short] - exit_px[m_short]) - total_cost
    pnl = pnl * lots * multiplier

    trades = []
    for i in range(len(preds_e)):
        d = "LONG" if preds_e[i] == 1 else "SHORT"
        hold_sec_ts = int((exit_ts[i] - entry_ts[i]) / 1e9)
        hold_sec_bar = int((int(exit_off[i]) + 1) * int(BAR_SECONDS))
        hold_sec = int(max(hold_sec_ts, hold_sec_bar))
        trades.append(
            {
                "entry_ts_ns": entry_ts[i],
                "entry_price": float(pe[i]),
                "hold_sec": hold_sec,
                "exit_price": float(exit_px[i]),
                "lots": int(lots[i]),
                "pnl": float(pnl[i]),
                "dir": d,
                "reason": exit_reason[i],
            }
        )

    def _ts_jst_iso(x: int) -> str:
        try:
            ts_sec = int(x) / 1e9
            dt_utc = datetime.datetime.utcfromtimestamp(ts_sec)
            dt_jst = dt_utc + datetime.timedelta(hours=9)
            return dt_jst.strftime("%Y-%m-%dT%H:%M:%S+09:00")
        except Exception:
            return str(int(x))

    cum_pnl = 0.0
    with open(trade_log_path, "w", encoding="utf-8") as f:
        f.write(
            "entry_ts_jst\tentry_price\thold_sec\texit_price\tlots\tpnl\tdir\treason\tcum_pnl\n"
        )
        for t in trades:
            cum_pnl += t["pnl"]
            f.write(
                f"{_ts_jst_iso(t['entry_ts_ns'])}\t{t['entry_price']:.1f}\t{t['hold_sec']}\t{t['exit_price']:.1f}\t{t['lots']}\t{t['pnl']:.1f}\t{t['dir']}\t{t['reason']}\t{cum_pnl:.1f}\n"
            )

    return trades


def write_trade_log(
    best: Dict[str, Any],
    trade_log_path: str,
    cfg: GlobalConfig,
    data: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """
    詳細な決済理由を含むTSVトレードログを出力する。
    メイン関数として各ヘルパー関数を呼び出し、処理をコーディネートする。

    Args:
        best (Dict[str, Any]): 最適化されたバックテスト結果とエントリー情報（最適化パラメータ含む）の辞書。
        trade_log_path (str): 出力先となるTSVファイルのパス。
        cfg (GlobalConfig): グローバル設定オブジェクト。
        data (Dict[str, np.ndarray]): 各種推論・時系列データを含む辞書。

    Returns:
        List[Dict[str, Any]]: 各トレードの詳細を格納した辞書のリスト。
    """
    if (
        not trade_log_path
        or best.get("n_trades", 0) == 0
        or data.get("future_ts") is None
    ):
        return []

    try:
        os.makedirs(os.path.dirname(trade_log_path) or ".", exist_ok=True)
        horizon = int(cfg.features.predict_horizon)

        entry_mask = best["entry_mask"]
        idx_entry = np.flatnonzero(entry_mask).astype(np.int64)
        lots = best.get("lots", np.ones(len(idx_entry)))

        preds = np.where(data["probs_short"] > 0.5, 2, 1)
        preds_e = preds[idx_entry]
        p_entry = (
            data["p_next_opens"]
            if cfg.backtest.use_next_bar_entry
            else data["p_closes"]
        )
        pe = p_entry[idx_entry]

        entry_ts = (
            data["next_ts"][idx_entry]
            if cfg.backtest.use_next_bar_entry
            else data["curr_ts"][idx_entry]
        )

        # -----------------------------------------------------
        # データ配列の取得
        # -----------------------------------------------------
        fh = data["f_highs"][idx_entry]
        fl = data["f_lows"][idx_entry]
        fc = data["f_closes"][idx_entry]
        fts = data["future_ts"][idx_entry]
        atr_e = data["atrs"][idx_entry]

        # -----------------------------------------------------
        # 実効ホライゾン計算の更新 (Diff適用箇所)
        # -----------------------------------------------------
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
        # 実際に保持可能なホライゾン（バー数）を計算
        hold_horizon_bars = max(
            1, min(predict_horizon, max_hold_bars, int(fc.shape[1]))
        )
        t_rem_raw = data["time_to_closes"][idx_entry].astype(np.float32, copy=False)

        # minutes_to_close が 0〜1 系に潰れている場合は、
        # そのまま int 化すると全件 1 バー保持になるため保険を入れる
        if len(idx_entry) == 0:
            act_h = np.empty(0, dtype=np.int32)
        elif float(np.nanmax(t_rem_raw)) <= 1.5 and hold_horizon_bars > 1:
            logger.warning(
                "time_to_closes looks normalized or compressed "
                "(max=%.4f). Falling back to hold_horizon_bars=%d for trade log.",
                float(np.nanmax(t_rem_raw)),
                int(hold_horizon_bars),
            )
            act_h = np.full(len(idx_entry), hold_horizon_bars, dtype=np.int32)
        else:
            t_rem = np.clip(
                np.floor(t_rem_raw).astype(np.int32, copy=False),
                0,
                None,
            )
            # 実効ホライゾンを計算（残り時間と保持可能ホライゾンを考慮）
            act_h = np.clip(t_rem, 1, hold_horizon_bars).astype(np.int32)

        # col_idxはact_hに基づいて算出するため、act_hより後方で評価
        col_idx = np.minimum(act_h, horizon).astype(np.int32) - 1
        row_idx = np.arange(len(idx_entry), dtype=np.int32)

        # デフォルトの決済設定 (ホライゾン到達によるクローズ)
        exit_off = col_idx.copy()
        exit_reason = np.full(len(idx_entry), "HORIZON", dtype=object)
        exit_px = fc[row_idx, col_idx].copy()

        # TP/SLの計算
        tp_e, sl_e, total_cost = _calculate_sl_tp(cfg, data, idx_entry)

        min_hold_bars = (
            int(math.ceil(float(cfg.backtest.min_holding_sec) / float(BAR_SECONDS)))
            if cfg.backtest.min_holding_sec
            else 0
        )
        min_hold_bars = max(min_hold_bars, 0)
        min_exit_idx = max(0, min_hold_bars - 1)

        # Long positions tracking
        exit_off, exit_reason, exit_px = _evaluate_long_positions(
            preds_e,
            pe,
            fh,
            fl,
            act_h,
            tp_e,
            sl_e,
            atr_e,
            cfg,
            best,
            horizon,
            min_exit_idx,
            min_hold_bars,
            exit_off,
            exit_reason,
            exit_px,
        )

        # Short positions tracking
        exit_off, exit_reason, exit_px = _evaluate_short_positions(
            preds_e,
            pe,
            fh,
            fl,
            act_h,
            tp_e,
            sl_e,
            atr_e,
            cfg,
            best,
            horizon,
            min_exit_idx,
            min_hold_bars,
            exit_off,
            exit_reason,
            exit_px,
        )

        exit_ts = fts[row_idx, exit_off]

        # PNL計算およびTSVへエクスポート
        multiplier = float(cfg.backtest.contract_multiplier)
        trades = _export_trades_to_tsv(
            trade_log_path,
            preds_e,
            pe,
            exit_px,
            entry_ts,
            exit_ts,
            exit_off,
            exit_reason,
            lots,
            total_cost,
            multiplier,
        )

        logger.info(f"Wrote trade log: {trade_log_path} ({len(idx_entry)} trades)")
        return trades

    except Exception as e:
        logger.warning(f"Failed to write trade log TSV: {e}")
        return []
