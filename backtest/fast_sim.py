# backtest/fast_sim.py
"""
File: backtest/fast_sim.py

ソースコードの役割:
本モジュールは、日経225ミニなどのデイトレードに対応したバックテストシミュレーターを提供します。
Numbaを用いた高速なシグナル判定（simulate_fast）や、ボラティリティレジームを考慮した動的な閾値の調整、
動的・静的なストップロス(SL)・テイクプロフィット(TP)の計算などのポートフォリオ管理・取引シミュレーションを実行します。
USDJPYやS&P500といった外部環境認識用データを将来的に統合可能な構造を想定しています。
"""
import math
from typing import Dict, Any, Optional
import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def simulate_fast(
    probs_action: np.ndarray,
    probs_short: np.ndarray,
    th_trade: float,
    min_dir_conf: float,
    cooldown_bars: int,
    future_closes: np.ndarray,
    tick_speeds: np.ndarray,
    time_to_closes: np.ndarray,
    min_tick_speed: float,
    spreads: np.ndarray,
    vol_regimes: np.ndarray,
    vol_scaling_intensity: float,
    base_horizon: int,
) -> tuple[np.ndarray, int]:
    """Numbaによる高速シミュレーションロジック。

    PythonループをJITコンパイルして高速化する。板の状況、残り時間、および
    ボラティリティレジームを考慮してエントリーのシグナル（entry_mask）を生成する。

    Args:
        probs_action (np.ndarray): アクション（取引）を行う確率の配列 (N,)。
        probs_short (np.ndarray): ショート方向の確率の配列 (N,)。
        th_trade (float): 取引を実行するための基本閾値。
        min_dir_conf (float): 方向感の確信度の最小値。
        cooldown_bars (int): エントリー後のクールダウンバー数。
        future_closes (np.ndarray): 未来の終値配列 (N, Horizon)。
        tick_speeds (np.ndarray): Tickスピード（板の動意）の配列 (N,)。
        time_to_closes (np.ndarray): セッション終了までの残り時間配列 (N,)。
        min_tick_speed (float): エントリーを許可する最小Tickスピード。
        spreads (np.ndarray): 現在のバーの値幅配列 (N,)。
        vol_regimes (np.ndarray): ボラティリティのレジーム配列 (N,)。
        vol_scaling_intensity (float): ボラティリティに応じた閾値スケーリング強度。
        base_horizon (int): 基本のホライゾン（予測期間）。

    Returns:
        tuple[np.ndarray, int]:
            - entry_mask (np.ndarray): エントリー箇所を示すブール配列。
            - raw_signals_count (int): フィルタ前の生シグナル数。
    """
    n = len(probs_action)
    entry_mask = np.zeros(n, dtype=np.bool_)
    cooldown_counter = 0
    raw_signals_count = 0

    # future_closes の shape 取得 (N, H)
    max_h = future_closes.shape[1]

    for i in range(n):
        # 1. Tick Speed Filter: 板が静かなときは入らない
        if tick_speeds[i] < min_tick_speed:
            continue

        # 1.1 Dead Market Filter: 値幅が極端に小さい(1Tick以下)なら見送る
        if spreads[i] < 5.0:
            continue

        # 4. Dynamic Thresholding: ボラティリティが低いほど閾値を高くする
        # レジームが 1.0 未満（閑散）の時、閾値を押し上げる
        eff_th_trade = th_trade
        if vol_scaling_intensity > 0:
            # regime=0.5, intensity=0.2 の場合: th * (1.0 + 0.2 * 0.5) = th * 1.1 (10%引き上げ)
            regime_penalty = max(0.0, 1.0 - vol_regimes[i])
            eff_th_trade = th_trade * (1.0 + vol_scaling_intensity * regime_penalty)

        # シグナル判定
        if probs_action[i] > eff_th_trade:
            raw_signals_count += 1

            if cooldown_counter == 0:
                # 方向確信度のチェック
                # min_dir_conf が 0.5 (デフォルト) の場合は実質チェックなし
                if min_dir_conf > 0.5:
                    p_short = probs_short[i]
                    p_dir = p_short if p_short > 0.5 else (1.0 - p_short)
                    if p_dir < min_dir_conf:
                        # クールダウン更新処理へスキップ
                        pass
                    else:
                        # 2. Session End Logic: 残り時間が少ない場合はホライゾンを短縮
                        t_rem = time_to_closes[i]

                        # もし残り時間が1分未満ならエントリーしない
                        if t_rem < 1.0:
                            continue

                        entry_mask[i] = True

                        # クールダウンは保持ホライゾンとは独立した固定バー数
                        cooldown_counter = cooldown_bars
                else:
                    entry_mask[i] = True
                    cooldown_counter = cooldown_bars

        # クールダウンの消化
        if cooldown_counter > 0:
            cooldown_counter -= 1

    return entry_mask, raw_signals_count


class Simulator:
    """バックテストシミュレータークラス。

    日経225ミニなどのスキャルピング〜1分足デイトレードに対応し、
    動的・静的なストップロス(SL)・テイクプロフィット(TP)の計算などの
    ポートフォリオ管理・取引シミュレーションを行う。
    外部環境（USDJPY, S&P500等）を考慮したロジック拡張の基盤となる。
    """

    def __init__(self, cfg: Any, data: Dict[str, Any]):
        """シミュレーターの初期化。

        Args:
            cfg (Any): 設定オブジェクト（Pydanticモデルによる型安全なオブジェクトを想定）。
            data (Dict[str, Any]): バックテストに使用するデータコンテナ。
        """
        self.cfg = cfg
        self.data = data
        # ランチタイム等、確率配列の初期化
        self.sim_probs_action = self.data.get("probs_action", np.array([]))

    def _calculate_sl_tp(
        self, entry_mask: np.ndarray, pe: np.ndarray, cost: float, slippage_val: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """エントリーマスクに基づいてSLおよびTPの価格配列を計算する。

        動的SL/TPが有効な場合はATRベースで計算し、
        TPフロア（最低確保したい利益幅）を考慮して利大を担保する。往復コストを
        カバーするための計算もここで行う。

        Args:
            entry_mask (np.ndarray): エントリー条件を満たすインデックスのブール配列。
            pe (np.ndarray): エントリー価格の配列。
            cost (float): 基本取引コスト。
            slippage_val (float): スリッページによる追加コスト。

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - sl_arr (np.ndarray): ストップロス幅の配列。
                - tp_arr (np.ndarray): テイクプロフィット幅の配列。
        """
        # 動的SL/TPの配列準備
        if self.cfg.backtest.use_dynamic_sl_tp:
            m_sl_e = np.clip(self.data["m_sl_arr"][entry_mask], 0.5, 5.0)
            atr_e = self.data["atrs"][entry_mask]

            if self.cfg.backtest.use_take_profit:
                # TPフロア分を上乗せする（往復コスト15円等の克服用）
                tp_floor_val = (
                    float(self.cfg.backtest.cost)
                    + float(self.cfg.backtest.slippage_tick) * 5.0
                    + float(self.cfg.backtest.tp_min_after_cost)
                )
                # モデル出力値と最低ATR乗数(tp_min_atr_mult)の大きい方を採用し、利大を担保
                m_tp_base = np.maximum(
                    self.data["m_tp_arr"][entry_mask], self.cfg.backtest.tp_min_atr_mult
                )
                m_tp_e = np.clip(
                    m_tp_base + (tp_floor_val / np.maximum(atr_e, 1e-6)), 0.5, 10.0
                )
                tp_arr = m_tp_e * atr_e
            else:
                # TP無効時は 0.0 をセットし、First-touch判定から除外する
                tp_arr = np.zeros(len(pe), dtype=np.float32)

            sl_arr = m_sl_e * atr_e
        else:
            if self.cfg.backtest.use_take_profit:
                tp_arr = np.full(
                    len(pe), float(self.cfg.backtest.tp_price), dtype=np.float32
                )
            else:
                tp_arr = np.zeros(len(pe), dtype=np.float32)
            sl_arr = np.full(
                len(pe), float(self.cfg.backtest.sl_price), dtype=np.float32
            )

        total_cost = cost + slippage_val

        tp_floor = 0.0
        if (
            self.cfg.backtest.use_take_profit
            and self.cfg.backtest.enforce_tp_min_after_cost
        ):
            tp_floor = float(total_cost + float(self.cfg.backtest.tp_min_after_cost))
            tp_arr = np.where(tp_arr > 0.0, np.maximum(tp_arr, tp_floor), tp_arr)

        return sl_arr, tp_arr

    def _simulate_positions(
        self,
        min_hold_bars: int,
        min_exit_idx: int,
        horizon: int,
        trailing_act_mult: Optional[float] = None,
        trailing_drop_mult: Optional[float] = None,
    ) -> np.ndarray:
        """エントリー後のポジションの推移（決済、トレールストップ）をシミュレーションする。

        Args:
            min_hold_bars (int): 最小保有バー数。
            min_exit_idx (int): 最小エグジットインデックス。
            horizon (int): 最大保有ホライゾン。
            trailing_act_mult (float, optional): トレール起動のATR乗数。指定がない場合は設定値を利用。
            trailing_drop_mult (float, optional): トレール落ち幅のATR乗数。指定がない場合は設定値を利用。

        Returns:
            np.ndarray: 各トレード結果を格納した配列。
        """
        cfg = self.cfg
        atrs = self.data.get("atrs", np.array([]))
        entry_mask = self.data.get("entry_mask", np.array([], dtype=np.bool_))

        # 実際のループ内処理の一部抜粋（パッチ部分のコンテキスト）
        m = 0  # Dummy index for snippet validation

        use_ts = cfg.backtest.use_trailing_stop
        if use_ts and cfg.backtest.use_dynamic_sl_tp:
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

            # 配列が利用可能な場合のみ計算（保護用）
            if len(atrs) > 0 and len(entry_mask) > 0:
                act_amt = act_m * atrs[entry_mask][m]
                drop_amt = drop_m * atrs[entry_mask][m]

        # (残りのポジションシミュレーションロジック...)
        return np.array([])

    def simulate_thresholds(
        self,
        th_trade: float,
        th_dir: float,
        opt_trailing_act: Optional[float] = None,
        opt_trailing_drop: Optional[float] = None,
    ) -> Dict[str, Any]:
        """指定された閾値でシミュレーションを実行し、統計情報を含む辞書を返す。

        Args:
            th_trade (float): 取引を実行するための基本閾値。
            th_dir (float): 方向感の確信度の閾値。
            opt_trailing_act (float, optional): 探索・最適化用のトレール起動ATR乗数。
            opt_trailing_drop (float, optional): 探索・最適化用のトレール落ち幅ATR乗数。

        Returns:
            Dict[str, Any]: 最適化スコアやトレード回数などを含む辞書。
        """
        # ランチタイムフィルター適用など
        is_lunch = self.data.get("is_lunch", np.array([], dtype=np.bool_))
        if len(self.sim_probs_action) > 0 and len(is_lunch) > 0:
            self.sim_probs_action[is_lunch] = 0.0

        # Pydanticによる型・属性の安全性が担保されているため getattr を排除し直接参照
        min_hold_bars = self.cfg.backtest.min_hold_bars
        min_exit_idx = self.cfg.backtest.min_exit_idx
        horizon = self.cfg.backtest.horizon

        self._simulate_positions(
            min_hold_bars,
            min_exit_idx,
            horizon,
            trailing_act_mult=opt_trailing_act,
            trailing_drop_mult=opt_trailing_drop,
        )

        # 動的ポジションサイジング・スコアリング（ダミー値による実装例）
        pf_eff, n_trades, pnl_score = 1.0, 10, 1.0
        signal_rate, trade_pen, dir_acc = 0.1, 0.1, 0.5
        raw_signals_count, min_dir_conf = 20, 0.5

        score = (
            math.log(pf_eff + 1e-6) * math.log(n_trades + 1) * pnl_score
            - 0.25 * signal_rate
            - 0.15 * trade_pen
            + 0.05 * dir_acc
        )

        act_m = (
            opt_trailing_act
            if opt_trailing_act is not None
            else self.cfg.backtest.trailing_act_mult
        )
        drop_m = (
            opt_trailing_drop
            if opt_trailing_drop is not None
            else self.cfg.backtest.trailing_drop_mult
        )

        return {
            "n_trades": int(n_trades),
            "threshold_trade": float(th_trade),
            "threshold_dir": float(th_dir),
            "trailing_act_mult": float(act_m),
            "trailing_drop_mult": float(drop_m),
            "min_dir_conf": float(min_dir_conf),
            "raw_signals_count": int(raw_signals_count),
        }
