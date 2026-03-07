# data/inference.py
"""
File: data/inference.py

ソースコードの役割:
本モジュールは、学習済みTransformerモデルを用いた推論処理の実行、および推論結果（予測確率）と
バックテストに必要なメタデータを効率的に集約するためのバッファ管理、プロファイリング機能を提供します。
GPUからCPUへの非同期転送を活用し、推論スループットの最大化を図ります。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from typing import Dict, Tuple, Optional, NamedTuple, ContextManager
from contextlib import contextmanager

from config import GlobalConfig
from util.utils import _perf_sync_if, PerfTimer, PERF_LEVEL_NUM, _PERF_INFER_SAMPLE


class InferenceBatch(NamedTuple):
    """DataLoaderから返されるバッチデータの構造を定義するNamedTuple。
    インデックスによるマジックナンバー参照を防ぎ、可読性を向上させます。
    """

    xc: torch.Tensor
    xs: torch.Tensor
    y: torch.Tensor
    p_curr: torch.Tensor
    p_next_open: torch.Tensor
    f_closes: torch.Tensor
    f_highs: torch.Tensor
    f_lows: torch.Tensor
    tick_spd: torch.Tensor
    t_close: torch.Tensor
    curr_atr: Optional[torch.Tensor] = None
    spread: Optional[torch.Tensor] = None
    curr_ts: Optional[torch.Tensor] = None
    next_ts: Optional[torch.Tensor] = None
    future_ts: Optional[torch.Tensor] = None
    vol_reg: Optional[torch.Tensor] = None


class InferenceResultBuffer:
    """モデルの推論出力（確率、SL/TP予測）に特化した事前割り当てバッファ。
    Pinned Memoryを使用してGPUからの非同期転送を最適化します。
    """

    def __init__(self, n_samples: int):
        self.probs_action = torch.empty(n_samples, dtype=torch.float16, pin_memory=True)
        self.probs_short = torch.empty(n_samples, dtype=torch.float16, pin_memory=True)
        self.m_sl = torch.empty(n_samples, dtype=torch.float16, pin_memory=True)
        self.m_tp = torch.empty(n_samples, dtype=torch.float16, pin_memory=True)

    def update(
        self,
        cursor: int,
        end_cursor: int,
        t_probs: torch.Tensor,
        d_probs: torch.Tensor,
        m_sl: torch.Tensor,
        m_tp: torch.Tensor,
    ):
        """モデル出力を非同期コピーでバッファに書き込む。"""
        self.probs_action[cursor:end_cursor].copy_(
            t_probs.detach().half(), non_blocking=True
        )
        self.probs_short[cursor:end_cursor].copy_(
            d_probs.detach().half(), non_blocking=True
        )
        self.m_sl[cursor:end_cursor].copy_(m_sl.detach().half(), non_blocking=True)
        self.m_tp[cursor:end_cursor].copy_(m_tp.detach().half(), non_blocking=True)


class MarketMetaBuffer:
    """バックテスト用の市場メタデータ（価格、指標、時間軸）に特化したバッファ。"""

    def __init__(self, n_samples: int, horizon: int):
        self.horizon = horizon
        self.labels = np.empty(n_samples, dtype=np.int64)
        self.p_closes = np.empty(n_samples, dtype=np.float32)
        self.p_next_opens = np.empty(n_samples, dtype=np.float32)
        self.tick_speeds = np.empty(n_samples, dtype=np.float32)
        self.time_to_closes = np.empty(n_samples, dtype=np.float32)
        self.atrs = np.empty(n_samples, dtype=np.float32)
        self.vol_regimes = np.empty(n_samples, dtype=np.float32)
        self.spreads = np.empty(n_samples, dtype=np.float32)
        self.curr_ts = np.empty(n_samples, dtype=np.int64)
        self.next_ts = np.empty(n_samples, dtype=np.int64)
        self.future_ts = np.empty((n_samples, horizon), dtype=np.int64)

        # 将来価格
        self.f_highs = np.empty((n_samples, horizon), dtype=np.float32)
        self.f_lows = np.empty((n_samples, horizon), dtype=np.float32)
        self.f_closes = np.empty((n_samples, horizon), dtype=np.float32)

    def update(self, cursor: int, end_cursor: int, batch: InferenceBatch):
        """バッチからメタデータを抽出し、Numpy配列として格納する。"""
        self.labels[cursor:end_cursor] = batch.y.numpy()
        self.p_closes[cursor:end_cursor] = batch.p_curr.numpy()
        self.p_next_opens[cursor:end_cursor] = batch.p_next_open.numpy()
        self.tick_speeds[cursor:end_cursor] = batch.tick_spd.numpy()
        self.time_to_closes[cursor:end_cursor] = batch.t_close.numpy()

        # オプショナル項目の処理
        self.atrs[cursor:end_cursor] = (
            batch.curr_atr.numpy() if batch.curr_atr is not None else 0.0
        )
        self.vol_regimes[cursor:end_cursor] = (
            batch.vol_reg.numpy() if batch.vol_reg is not None else 1.0
        )
        self.spreads[cursor:end_cursor] = (
            batch.spread.numpy() if batch.spread is not None else 1e9
        )

        self.curr_ts[cursor:end_cursor] = (
            batch.curr_ts.numpy() if batch.curr_ts is not None else 0
        )
        self.next_ts[cursor:end_cursor] = (
            batch.next_ts.numpy() if batch.next_ts is not None else 0
        )

        if batch.future_ts is not None:
            self.future_ts[cursor:end_cursor, :] = batch.future_ts.numpy()

        if self.horizon > 0:
            self.f_highs[cursor:end_cursor] = batch.f_highs.numpy()
            self.f_lows[cursor:end_cursor] = batch.f_lows.numpy()
            self.f_closes[cursor:end_cursor] = batch.f_closes.numpy()


class InferenceProfiler:
    """推論処理の各ステップ時間を計測するプロファイラ。
    コンテキストマネージャとして動作し、自動的に時間を集計します。
    """

    def __init__(self, do_perf: bool):
        self.do_perf = do_perf
        self.stats = {"data": 0.0, "dev": 0.0, "fwd": 0.0, "pp": 0.0}
        self.t0_infer_start = time.perf_counter()

    @contextmanager
    def measure(
        self, key: str, should_sync: bool = False, device: Optional[torch.device] = None
    ):
        """指定された区間の実行時間を計測するコンテキストマネージャ。"""
        if not self.do_perf:
            yield
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            if should_sync and device and device.type == "cuda":
                torch.cuda.synchronize()
            self.stats[key] += time.perf_counter() - t0

    def log_stats(self, logger: logging.Logger, fold_idx: int):
        """計測結果をログ出力。"""
        if self.do_perf:
            total_time = (time.perf_counter() - self.t0_infer_start) * 1000.0
            s = self.stats
            logger.perf(
                f"[perf] fold{fold_idx}/backtest/infer: {total_time:.2f} ms | "
                f"data={s['data']*1000:.1f}, dev={s['dev']*1000:.1f}, "
                f"fwd={s['fwd']*1000:.1f}, pp={s['pp']*1000:.1f} (ms)"
            )


def extract_inference_data(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: GlobalConfig,
    fold_idx: int,
) -> Dict[str, np.ndarray]:
    """DataLoaderから推論を実行し、全サンプル分の予測結果とメタデータを集約する。"""
    logger = logging.getLogger(__name__)
    model.eval()

    n_samples = len(loader.dataset)
    horizon = int(cfg.features.predict_horizon)

    # バッファとプロファイラの初期化
    res_buffer = InferenceResultBuffer(n_samples)
    meta_buffer = MarketMetaBuffer(n_samples, horizon)
    profiler = InferenceProfiler(do_perf=logger.isEnabledFor(PERF_LEVEL_NUM))

    cursor = 0
    t_end_prev = time.perf_counter()

    with torch.no_grad():
        for i, raw_batch in enumerate(loader):
            # 1. データ読み込み（待機時間）の計測
            with profiler.measure("data"):
                profiler.stats["data"] += time.perf_counter() - t_end_prev
                # バッチをNamedTupleに展開（構造の抽象化）
                batch = InferenceBatch(*raw_batch)

            should_sync = profiler.do_perf and (i % _PERF_INFER_SAMPLE == 0)

            # 2. デバイス転送
            with profiler.measure("dev", should_sync, device):
                xc = batch.xc.to(device, non_blocking=True)
                xs = batch.xs.to(device, non_blocking=True)

            # 3. フォワードパス
            with profiler.measure("fwd", should_sync, device):
                out = model(xc, xs)
                if isinstance(out, (tuple, list)) and len(out) == 3:
                    trade_logits, dir_logits, sltp_preds = out
                else:
                    trade_logits, dir_logits = out
                    sltp_preds = None

            # 4. ポストプロセス & バッファ格納
            with profiler.measure("pp", should_sync, device):
                batch_size = trade_logits.size(0)
                end_cursor = cursor + batch_size

                # 確率計算
                t_probs = torch.sigmoid(trade_logits[:, 1] - trade_logits[:, 0])
                d_probs = torch.sigmoid(dir_logits[:, 1] - dir_logits[:, 0])

                if sltp_preds is None:
                    m_sl = t_probs.new_full(
                        (batch_size,), float(cfg.backtest.default_m_sl)
                    )
                    m_tp = t_probs.new_full(
                        (batch_size,), float(cfg.backtest.default_m_tp)
                    )
                else:
                    m_sl, m_tp = sltp_preds[:, 0], sltp_preds[:, 1]

                # 各バッファへ更新
                res_buffer.update(cursor, end_cursor, t_probs, d_probs, m_sl, m_tp)
                meta_buffer.update(cursor, end_cursor, batch)

                cursor = end_cursor

            t_end_prev = time.perf_counter()

    # 最終同期とデータ集約
    if device.type == "cuda":
        torch.cuda.synchronize()

    profiler.log_stats(logger, fold_idx)

    # 辞書形式で結合して返す
    return {
        "probs_action": res_buffer.probs_action.float().numpy(),
        "probs_short": res_buffer.probs_short.float().numpy(),
        "m_sl_arr": res_buffer.m_sl.float().numpy(),
        "m_tp_arr": res_buffer.m_tp.float().numpy(),
        "labels": meta_buffer.labels,
        "p_closes": meta_buffer.p_closes,
        "p_next_opens": meta_buffer.p_next_opens,
        "f_highs": meta_buffer.f_highs,
        "f_lows": meta_buffer.f_lows,
        "f_closes": meta_buffer.f_closes,
        "tick_speeds": meta_buffer.tick_speeds,
        "time_to_closes": meta_buffer.time_to_closes,
        "spreads": meta_buffer.spreads,
        "atrs": meta_buffer.atrs,
        "vol_regimes": meta_buffer.vol_regimes,
        "curr_ts": meta_buffer.curr_ts,
        "next_ts": meta_buffer.next_ts,
        "future_ts": meta_buffer.future_ts,
    }
