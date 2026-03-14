# core/fallback_strategy.py
"""
File: core/fallback_strategy.py

ソースコードの役割:
本モジュールは、Out-of-Sample (OOS) テスト等のバックテストにおいて、
指定した初期閾値でトレードが一切発生しなかった場合に適用する
ヒューリスティックな閾値緩和（フォールバック）ロジックを提供します。
"""

import logging
from typing import Dict, Any, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from trade.trading import run_vectorized_backtest


def resolve_oos_fallback(
    model: nn.Module,
    loader_test: DataLoader,
    device: Any,
    cfg: Any,
    fold_idx: int,
    initial_trade_th: float,
    initial_dir_th: float,
    trade_log_path: str,
    logger: logging.Logger,
    min_fallback_trades: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    OOSテストでトレードが発生しない場合に、段階的に閾値を緩和して再評価を行います。
    トレードが発生する閾値が見つかった時点で、そのバックテスト結果を返します。

    Args:
        model (nn.Module): 評価対象のPyTorchモデル。
        loader_test (DataLoader): テスト用データローダー。
        device (Any): 実行デバイス。
        cfg (Any): 設定オブジェクト。
        fold_idx (int): 現在のFold番号。
        initial_trade_th (float): 初期の取引有無の閾値。
        initial_dir_th (float): 初期の取引方向の閾値。
        trade_log_path (str): トレードログの出力パス。
        logger (logging.Logger): ロガーインスタンス。
        min_fallback_trades (int): フォールバック採用に必要な最小トレード数。

    Returns:
        Optional[Dict[str, Any]]: フォールバックによりトレードが発生した場合のバックテスト結果。
                                  すべての緩和を試しても発生しなかった場合は None。
    """
    fallback_candidates = []

    logger.warning(
        "OOS fallback triggered: no trades with restored val thresholds."
    )

    # 1. 方向閾値を0.50（ニュートラル）まで緩和
    if initial_dir_th > 0.50:
        fallback_candidates.append((initial_trade_th, 0.50, "relax_dir_to_0.50"))

    # 2. エントリー閾値をわずかに下げ(-0.01)、方向閾値を0.50以下にキャップ
    fallback_candidates.append(
        (
            max(0.45, initial_trade_th - 0.01),
            min(initial_dir_th, 0.50),
            "relax_trade_by_0.01_and_cap_dir_to_0.50",
        )
    )

    # 3. エントリー閾値をさらに下げ(-0.02)、方向閾値を0.50以下にキャップ
    fallback_candidates.append(
        (
            max(0.45, initial_trade_th - 0.02),
            min(initial_dir_th, 0.50),
            "relax_trade_by_0.02_and_cap_dir_to_0.50",
        )
    )

    seen = set()
    best_oos = None

    # フォールバック候補を順に試行
    for fb_th_trade, fb_th_dir, reason in fallback_candidates:
        key = (round(float(fb_th_trade), 6), round(float(fb_th_dir), 6))
        if key in seen:
            continue
        seen.add(key)

        logger.info(
            "Retrying OOS fallback (%s): threshold_trade=%.3f, threshold_dir=%.3f",
            reason,
            float(fb_th_trade),
            float(fb_th_dir),
        )
        candidate_oos = run_vectorized_backtest(
            model,
            loader_test,
            device,
            cfg,
            fold_idx,
            fixed_threshold_trade=float(fb_th_trade),
            fixed_threshold_dir=float(fb_th_dir),
            trade_log_path=trade_log_path,
        )

        candidate_n_trades = int(candidate_oos.get("n_trades", 0))
        if candidate_n_trades <= 0:
            continue

        if candidate_n_trades >= int(min_fallback_trades):
            candidate_oos["fallback_reason"] = reason
            logger.info(
                "OOS fallback adopted: %s (n_trades=%d)",
                reason,
                candidate_n_trades,
            )
            best_oos = candidate_oos
            break

        logger.info(
            "OOS fallback rejected: %s (n_trades=%d < min_fallback_trades=%d)",
            reason,
            candidate_n_trades,
            int(min_fallback_trades),
        )

    if best_oos is None:
        logger.info(
            "OOS fallback abandoned: no candidate satisfied adoption criteria."
        )

    return best_oos
