# trade/trading.py
"""
File: trade/trading.py

ソースコードの役割:
本モジュールは、学習済みモデルを用いた推論結果に基づき、バックテストの実行および
パラメータ（エントリー閾値、方向判定閾値、トレイリングストップ等）の最適化を制御します。
シミュレーション結果の評価、最良パラメータの選定、および詳細なトレードログの出力をオーケストレーションします。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Optional, Dict, Any

# Configおよびユーティリティ
from config import GlobalConfig
from util.utils import PerfTimer

# 分割された各コンポーネントのインポート
from data.inference import extract_inference_data
from trade.simulator import BacktestSimulator
from util.trade_log import write_trade_log, log_backtest_summary


def _evaluate_and_update_best(
    best: Optional[Dict[str, Any]],
    current_result: Dict[str, Any],
    min_trades: int,
    max_signal_rate: float,
    total_samples: int,
) -> Optional[Dict[str, Any]]:
    """
    シミュレーション結果を評価し、各種制約を満たしつつスコアが向上していれば最良結果を更新する。

    Args:
        best (Optional[Dict[str, Any]]): 現在の最良結果の辞書（未設定の場合はNone）
        current_result (Dict[str, Any]): 今回のシミュレーション結果
        min_trades (int): 許容する最小取引回数
        max_signal_rate (float): 許容する最大シグナル発生率
        total_samples (int): 評価対象の全サンプル数（シグナル発生率の計算に使用）

    Returns:
        Optional[Dict[str, Any]]: 更新された最良結果の辞書（条件未達やスコアが低い場合は元のbestを返す）
    """
    # 最小取引数を満たさない場合はスキップ
    if current_result["n_trades"] < min_trades:
        return best

    # シグナル頻度が設定上限を超える場合はスキップ
    # get()を使用してキー不在時の実行時エラーを防止
    signal_rate = current_result.get("raw_signals_count", 0) / max(total_samples, 1)
    if signal_rate > max_signal_rate:
        return best

    # スコアが最も高い設定を保持
    if best is None or current_result["score"] > best["score"]:
        return current_result

    return best


def optimize_backtest_parameters(
    simulator: BacktestSimulator,
    data: Dict[str, np.ndarray],
    cfg: GlobalConfig,
    fold_idx: int = 0,
    fixed_threshold_trade: Optional[float] = None,
    fixed_threshold_dir: Optional[float] = None,
) -> Dict[str, Any]:
    """
    トレード閾値、方向判定閾値、トレイリングストップ等のバックテストパラメータを探索・最適化する。

    Args:
        simulator (BacktestSimulator): バックテスト実行用シミュレータ
        data (Dict[str, np.ndarray]): 推論結果を含むデータ辞書
        cfg (GlobalConfig): 全体設定オブジェクト
        fold_idx (int): クロスバリデーションのフォールド番号
        fixed_threshold_trade (Optional[float]): 固定のエントリー閾値（指定時は最適化スキップ）
        fixed_threshold_dir (Optional[float]): 固定の方向判定閾値（指定時は最適化スキップ）

    Returns:
        Dict[str, Any]: 最適化された（あるいは指定された）パラメータでのシミュレーション結果
    """
    logger = logging.getLogger(__name__)
    probs_action = data["probs_action"]
    total_samples = len(probs_action)

    # 固定閾値が指定されている場合は探索をスキップ
    if fixed_threshold_trade is not None and fixed_threshold_dir is not None:
        return simulator.simulate_thresholds(
            float(fixed_threshold_trade), float(fixed_threshold_dir)
        )

    # 自動チューニングを行わない場合
    if not cfg.backtest.auto_tune_threshold:
        return simulator.simulate_thresholds(
            cfg.backtest.threshold_trade, cfg.backtest.threshold_dir
        )

    # --- 以下、自動チューニング処理 ---
    min_trades = max(
        int(total_samples * cfg.backtest.min_trades_frac),
        cfg.backtest.min_trades_floor,
        cfg.backtest.min_trades_for_tuning,
    )

    # エントリー閾値のチューニング候補生成（パーセンタイルに基づく）
    cand_trade_percentiles = [70, 75, 80, 85, 90, 92, 94, 96, 98, 99]
    cand_trade_thresholds = sorted(
        set(float(np.percentile(probs_action, p)) for p in cand_trade_percentiles)
    )
    cand_trade_thresholds.append(cfg.backtest.threshold_trade)
    cand_trade_thresholds = sorted(set(cand_trade_thresholds))

    # 方向判定閾値のチューニング候補生成
    cand_dir_thresholds = sorted(
        set(
            [
                0.50,
                0.52,
                0.55,
                float(cfg.backtest.threshold_dir),
            ]
        )
    )

    # トレイリングストップ(TS)パラメータの最適化候補設定
    cand_ts_params = [(cfg.backtest.trailing_act_mult, cfg.backtest.trailing_drop_mult)]
    if cfg.backtest.use_trailing_stop:
        cand_ts_params = sorted(
            list(set(cand_ts_params + [(0.8, 1.2), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]))
        )

    best = None
    n_eval = 0
    max_signal_rate = cfg.backtest.max_signal_rate_for_tuning

    with PerfTimer(
        logger,
        f"fold{fold_idx}/backtest/threshold_scan",
        extras={
            "cand_trade": int(len(cand_trade_thresholds)),
            "cand_dir": int(len(cand_dir_thresholds)),
            "cand_ts": int(len(cand_ts_params)),
            "min_trades": int(min_trades),
        },
        do_sync=False,
    ):
        # 探索ループ
        for th_t in cand_trade_thresholds:
            for th_d in cand_dir_thresholds:
                for ts_act, ts_drop in cand_ts_params:
                    n_eval += 1
                    r = simulator.simulate_thresholds(
                        th_t,
                        th_d,
                        opt_trailing_act=ts_act,
                        opt_trailing_drop=ts_drop,
                    )

                    best = _evaluate_and_update_best(
                        best, r, min_trades, max_signal_rate, total_samples
                    )

    logger.perf(f"[perf] fold{fold_idx}/backtest/threshold_search: evaluated={n_eval}")

    # 条件に合致する設定が見つからなかった場合のフォールバック
    if best is None:
        best = simulator.simulate_thresholds(
            cfg.backtest.threshold_trade, cfg.backtest.threshold_dir
        )

    return best


def run_vectorized_backtest(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: GlobalConfig,
    fold_idx: int = 0,
    fixed_threshold_trade: Optional[float] = None,
    fixed_threshold_dir: Optional[float] = None,
    trade_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    検証データ全体に対して推論を行い、簡易的なPnLパフォーマンス（勝率・取引数）を計算する。
    推論、バックテストシミュレーション、ログ出力をオーケストレーションするメイン関数。

    Args:
        model (nn.Module): 評価対象のPyTorchモデル
        loader (DataLoader): 検証用データローダー
        device (torch.device): 推論を実行するデバイス
        cfg (GlobalConfig): トレードルール・バックテスト設定を含む全体設定
        fold_idx (int, optional): クロスバリデーションのフォールドインデックス (デフォルト: 0)
        fixed_threshold_trade (Optional[float], optional): 固定の取引エントリー閾値 (デフォルト: None)
        fixed_threshold_dir (Optional[float], optional): 固定の方向判定閾値 (デフォルト: None)
        trade_log_path (Optional[str], optional): トレード詳細ログの出力先パス (デフォルト: None)

    Returns:
        Dict[str, Any]: バックテストのパフォーマンス指標とトレード履歴を含む辞書
    """
    # 1. DataLoaderからの推論データ抽出と集約 (推論フェーズ)
    data = extract_inference_data(model, loader, device, cfg, fold_idx)

    # ブロードキャストの次元爆発 (N, 1) * (N,) -> (N, N) を防ぐため、
    # 推論結果の 2次元カラムベクトルをすべて 1次元配列に平坦化(Flatten)する
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
            data[k] = v.flatten()

    # 2. シミュレーターの初期化
    simulator = BacktestSimulator(data, cfg)

    # 3. 閾値の探索・評価 (バックテストフェーズ)
    best = optimize_backtest_parameters(
        simulator, data, cfg, fold_idx, fixed_threshold_trade, fixed_threshold_dir
    )

    # 4. パフォーマンスサマリーの標準出力ロギング
    log_backtest_summary(best, fold_idx, data["probs_action"])

    # 5. トレードログの詳細出力 (I/Oフェーズ)
    trades = write_trade_log(best, trade_log_path, cfg, data)

    # best["n_trades"] が 0 の場合の早期リターンハンドリング
    if best["n_trades"] == 0:
        return {"n_trades": 0, "win_rate": 0.0, "score": -float("inf")}

    return {
        "n_trades": int(best["n_trades"]),
        "win_rate": float(best.get("win_rate", 0.0)),
        "pf": float(best.get("pf", 0.0)),
        "pnl": float(best.get("pnl", 0.0)),
        "avg_pnl": float(best.get("avg_pnl", 0.0)),
        "score": float(best.get("score", 0.0)),
        "threshold_trade": float(best.get("threshold_trade", 0.0)),
        "threshold_dir": float(best.get("threshold_dir", 0.0)),
        "min_dir_conf": float(best.get("min_dir_conf", 0.0)),
        "raw_signals_count": int(best.get("raw_signals_count", 0)),
        "trades": trades,
    }
