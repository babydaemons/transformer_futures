# core/fallback_strategy.py
"""
File: core/fallback_strategy.py

ソースコードの役割:
本モジュールは、Out-of-Sample (OOS) テスト等のバックテストにおいて、
指定した初期閾値でトレードが一切発生しなかった場合に適用する
ヒューリスティックな閾値緩和（フォールバック）ロジックを提供します。
初期OOS結果を評価し、トレード機会が極端に少ない場合に段階的な閾値調整を行い、
デイトレードシステムにおける機会損失を防ぐ役割を担います。
"""

import logging
import copy
from typing import Dict, Any, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from trade.trading import run_vectorized_backtest


def should_trigger_fallback(
    initial_oos: Dict[str, Any],
    logger: logging.Logger,
) -> bool:
    """
    初回OOS結果に対して、フォールバックを起動すべきかどうかを判定します。

    現状の方針では「no-trade only（トレードが全く発生しなかった場合）」で起動します。

    Args:
        initial_oos (Dict[str, Any]): 初回OOSバックテスト結果。
        logger (logging.Logger): ロガーインスタンス。

    Returns:
        bool: フォールバックを起動する場合は True、それ以外は False。
    """
    n_trades = int(initial_oos.get("n_trades", 0))

    if n_trades == 0:
        logger.warning(
            "OOS fallback triggered: no trades with restored val thresholds."
        )
        return True

    logger.info(
        "OOS fallback skipped: initial OOS already has trades (n_trades=%d)", n_trades
    )
    return False


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
    min_fallback_trades: int = 3,
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
        min_fallback_trades (int): 採用に必要な最小トレード数。

    Returns:
        Optional[Dict[str, Any]]: フォールバックによりトレードが発生した場合のバックテスト結果。
                                  すべての緩和を試しても発生しなかった場合は None。
    """
    fallback_candidates = []

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

        if candidate_n_trades >= min_fallback_trades:
            adopted_oos = copy.deepcopy(candidate_oos)
            adopted_oos["fallback_reason"] = reason
            logger.info(
                "OOS fallback adopted: %s (n_trades=%d)",
                reason,
                candidate_n_trades,
            )
            return adopted_oos

        if candidate_n_trades > 0:
            logger.info(
                "OOS fallback rejected: %s (n_trades=%d < min_fallback_trades=%d)",
                reason,
                candidate_n_trades,
                min_fallback_trades,
            )

    logger.info("OOS fallback abandoned: no candidate satisfied adoption criteria.")
    return None
