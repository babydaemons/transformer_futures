# trade/position_sizing.py
"""
File: trade/position_sizing.py

ソースコードの役割:
本モジュールは、システムトレードにおけるロットサイズ計算ロジックを専任で扱います。
シグナルの確信度やボラティリティ（ATR）に基づく追加ロットの付与など、
ベースロットから最大ロットまでの動的なポジションサイジングを実行します。
"""

import numpy as np
from config import GlobalConfig


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
