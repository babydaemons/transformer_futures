# data/dataset.py
"""
ソースコードパス: data/dataset.py
目的・役割:
    日経平均先物ミニの1分足データを用いた時系列予測モデル（Transformer/TFT等）のための
    データセットクラスおよびウォークフォワード・スプリットの定義。
    メモリ効率を考慮し、スライスをオンザフライで行う設計となっている。
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Tuple, Iterator, Optional, Any
from config import cfg


class WalkForwardSplit:
    """
    ウォークフォワード・バリデーション用のデータ分割管理クラス。

    時系列の連続性を維持しながら、学習・検証・テスト期間をスライドさせていく。
    """

    def __init__(
        self, train_days: int, val_days: int, test_days: int, step_days: int = 1
    ):
        """
        Args:
            train_days (int): 学習期間の日数
            val_days (int): 検証期間の日数
            test_days (int): テスト期間の日数
            step_days (int): ウィンドウをスライドさせるステップ（日数）
        """
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days
        self.step_days = step_days

    def split(
        self, dates: List[pd.Timestamp]
    ) -> Iterator[Tuple[List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]]]:
        """
        日付リストに基づいて期間を分割するジェネレータ。

        Yields:
            Tuple[List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]]: (train, val, test) の日付リスト
        """
        sorted_dates = sorted(list(set(dates)))
        n_dates = len(sorted_dates)
        start_idx = 0
        while True:
            train_end_idx = start_idx + self.train_days
            val_end_idx = train_end_idx + self.val_days
            test_end_idx = val_end_idx + self.test_days
            if test_end_idx > n_dates:
                break
            yield (
                sorted_dates[start_idx:train_end_idx],
                sorted_dates[train_end_idx:val_end_idx],
                sorted_dates[val_end_idx:test_end_idx],
            )
            start_idx += self.step_days


# ============================================================
# Dataset
# ============================================================
class TFTDataset(Dataset):
    """
    メモリ効率のために On-the-fly でデータをスライスする Dataset クラス。

    Triple Barrier Method によるラベリングと、バックテストに必要な
    メタデータ（価格、タイムスタンプ、ATR等）を同時に返す。
    """

    def __init__(
        self,
        cont_data: np.ndarray,
        static_data: np.ndarray,
        target_data: np.ndarray,
        high_data: np.ndarray,
        low_data: np.ndarray,
        raw_prices: np.ndarray,
        atr_data: np.ndarray,
        ts_ns_data: np.ndarray,
        seq_len: int,
        predict_horizon: int,
        stride: int = 10,
        label_threshold: float = 0.0005,
        precomputed_labels: Optional[np.ndarray] = None,
    ):
        """
        Args:
            cont_data: 連続変数特徴量
            static_data: 静的特徴量（TOD等）
            target_data: 目的変数関連データ [Close, High, Low, TickSpeed, MinutesToClose, Open]
            high_data: 高値
            low_data: 安値
            raw_prices: 生価格（エントリー計算用）
            atr_data: ATR
            ts_ns_data: ナノ秒タイムスタンプ
            seq_len: 入力シーケンス長
            predict_horizon: 予測ホライゾン
            stride: サンプリングの間隔
            label_threshold: ラベル生成時の閾値（予備）
            precomputed_labels: 事前計算済みラベル
        """
        self.cont_data = cont_data
        self.static_data = static_data
        self.target_data = target_data
        self.high_data = high_data
        self.low_data = low_data
        self.raw_prices = raw_prices
        self.atr_data = atr_data
        self.ts_ns = ts_ns_data
        self.seq_len = seq_len
        self.predict_horizon = predict_horizon
        self.stride = stride
        self.label_threshold = label_threshold
        self.precomputed_labels = precomputed_labels

        total_len = len(target_data)

        # ターゲット配列の列インデックス定義
        # target_data[:, 3] -> tick_speed_ratio, [:, 4] -> minutes_to_close
        self.meta_tick_speed = target_data[:, 3]
        self.meta_time_to_close = target_data[:, 4]

        # 価格データのキャッシュ
        self.close_arr = target_data[:, 0]

        self.num_samples = (total_len - seq_len - predict_horizon - 1) // stride + 1

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        """
        指定されたインデックスに対応するデータスライスを取得する。
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len

        x_cont = self.cont_data[start_idx:end_idx]

        # staticはシーケンス最後の時点を使う（TODなど時変）
        x_stat = self.static_data[end_idx - 1]

        p_curr = self.raw_prices[end_idx - 1]

        # Next Bar Open (for backtest simulation)
        # target_data[:, 5] is open
        if end_idx < len(self.target_data):
            p_next_open = self.target_data[end_idx, 5]
        else:
            p_next_open = p_curr

        # 未来のHigh/Low系列（TP/SLのFirst Touch判定に使用）
        h_start = end_idx
        h_end = min(end_idx + self.predict_horizon, len(self.target_data))

        # 未来系列の切り出し
        future_closes = self.close_arr[h_start:h_end]
        future_highs = self.high_data[h_start:h_end]
        future_lows = self.low_data[h_start:h_end]

        # 現在のメタデータ
        curr_tick_speed = self.meta_tick_speed[end_idx - 1]
        curr_time_to_close = self.meta_time_to_close[end_idx - 1]

        # 現在バーのスプレッド（High-Low）。シグナル品質フィルタに使用
        curr_spread = float(self.high_data[end_idx - 1] - self.low_data[end_idx - 1])

        # タイムスタンプ（ns）。ログ/TSV出力用
        curr_ts_ns = int(self.ts_ns[end_idx - 1])
        next_ts_ns = (
            int(self.ts_ns[end_idx]) if end_idx < len(self.ts_ns) else curr_ts_ns
        )

        # 未来のタイムスタンプ系列
        future_ts_ns = self.ts_ns[h_start:h_end]

        curr_atr = self.atr_data[end_idx - 1]

        # ボラティリティ・レジームの取得
        # cfg.features.continuous_cols の定義順に基づいて continuous 特徴量から抽出
        curr_vol_regime = self.cont_data[
            end_idx - 1, cfg.features.continuous_cols.index("vol_regime")
        ]

        # 事前計算済みラベルがある場合
        if self.precomputed_labels is not None:
            return (
                x_cont,
                x_stat,
                np.array(self.precomputed_labels[end_idx - 1], dtype=np.int64),
                p_curr,
                p_next_open,
                future_closes,
                future_highs,
                future_lows,
                curr_tick_speed,
                curr_time_to_close,
                np.array(curr_atr, dtype=np.float32),
                np.array(curr_spread, dtype=np.float32),
                np.array(curr_ts_ns, dtype=np.int64),
                np.array(next_ts_ns, dtype=np.int64),
                future_ts_ns.astype(np.int64),
                np.array(curr_vol_regime, dtype=np.float32),
            )

        # --- ラベル生成ロジック: Triple Barrier Method ---
        scale = cfg.features.label_threshold_scale
        min_val = cfg.features.label_min_limit
        abs_threshold = max(curr_atr * scale, min_val)

        label = 0
        if h_end > h_start:
            upper_barrier = p_curr + abs_threshold
            lower_barrier = p_curr - abs_threshold

            # First Touch: どちらに先に触れたか
            up_hits = future_highs > upper_barrier
            dn_hits = future_lows < lower_barrier

            if up_hits.any() or dn_hits.any():
                idx_up = int(np.argmax(up_hits)) if up_hits.any() else 10**9
                idx_dn = int(np.argmax(dn_hits)) if dn_hits.any() else 10**9

                if idx_up < idx_dn:
                    label = 1  # Long
                elif idx_dn < idx_up:
                    label = 2  # Short

        return (
            x_cont,
            x_stat,
            np.array(label, dtype=np.int64),
            p_curr,
            p_next_open,
            future_closes,
            future_highs,
            future_lows,
            curr_tick_speed,
            curr_time_to_close,
            np.array(curr_atr, dtype=np.float32),
            np.array(curr_spread, dtype=np.float32),
            np.array(curr_ts_ns, dtype=np.int64),
            np.array(next_ts_ns, dtype=np.int64),
            future_ts_ns.astype(np.int64),
            np.array(curr_vol_regime, dtype=np.float32),  # ★追加
        )


class TimeSeriesDataset(Dataset):
    """
    固定シーケンス長でデータを供給する Dataset クラス。

    TFTDatasetとは異なり、初期化時にベクトル演算で全ラベルを一括生成するため、
    学習時のオーバーヘッドが少ない。
    """

    def __init__(
        self,
        x_cont: np.ndarray,
        x_static: np.ndarray,
        y: np.ndarray,
        atr_data: np.ndarray,
        seq_len: int = 120,
        prediction_horizon: int = 12,
        stride: int = 1,
    ):
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.x_static = torch.tensor(x_static, dtype=torch.float32)

        # target (y) の分解: [Close, High, Low, TickSpeed, MinutesToClose, Open]
        target_arr = np.asarray(y)
        if target_arr.ndim == 2 and target_arr.shape[1] >= 6:
            self.close = torch.tensor(target_arr[:, 0], dtype=torch.float32)
            self.high = torch.tensor(target_arr[:, 1], dtype=torch.float32)
            self.low = torch.tensor(target_arr[:, 2], dtype=torch.float32)
            self.tick_speed = torch.tensor(target_arr[:, 3], dtype=torch.float32)
            self.minutes_to_close = torch.tensor(target_arr[:, 4], dtype=torch.float32)
            self.open_price = torch.tensor(target_arr[:, 5], dtype=torch.float32)
        else:
            self.close = torch.tensor(target_arr[:, 0], dtype=torch.float32)
            self.high = self.close
            self.low = self.close
            self.tick_speed = torch.zeros_like(self.close)
            self.minutes_to_close = torch.zeros_like(self.close)
            self.open_price = self.close

        self.atr_data = torch.tensor(atr_data, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = prediction_horizon
        self.stride = stride

        # --- ラベルの一括計算 (初期化時に実行して高速化) ---
        _prices = self.close.numpy()
        _highs = self.high.numpy()
        _lows = self.low.numpy()
        _atrs = self.atr_data.numpy()

        # バックテストコストを考慮した閾値フロアの計算
        _slip = float(cfg.backtest.slippage_tick) * 5.0
        _cost_floor = (
            float(cfg.backtest.cost) + _slip + float(cfg.features.label_cost_buffer)
        )
        _mode = str(cfg.features.label_min_limit_mode)

        labels_np = generate_labels_numpy(
            prices=_prices,
            highs=_highs,
            lows=_lows,
            atrs=_atrs,
            horizon=self.horizon,
            threshold_factor=float(cfg.features.label_threshold_scale),
            min_limit=float(cfg.features.label_min_limit),
            cost_floor=_cost_floor,
            min_limit_mode=_mode,
        )
        self.labels = torch.tensor(labels_np, dtype=torch.long)

    def __len__(self) -> int:
        return (len(self.close) - self.seq_len - self.horizon) // self.stride

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start = idx * self.stride
        end = start + self.seq_len

        x_c = self.x_cont[start:end]
        x_s = self.x_static[start:end]

        curr_idx = end - 1
        label = self.labels[curr_idx]

        # エントリー価格の決定 (Next Bar Open)
        if end < len(self.open_price):
            p_next_open_val = self.open_price[end]
        else:
            p_next_open_val = self.close[curr_idx]

        p_signal_close = self.close[curr_idx]

        # 未来データの切り出し
        h_start = end
        h_end = end + self.horizon
        f_closes = self.close[h_start:h_end]
        f_highs = self.high[h_start:h_end]
        f_lows = self.low[h_start:h_end]

        tick_spd = self.tick_speed[curr_idx]
        t_close = self.minutes_to_close[curr_idx]
        curr_atr = self.atr_data[curr_idx]

        return (
            x_c,
            x_s[-1],
            label,
            p_signal_close,
            p_next_open_val,
            f_closes,
            f_highs,
            f_lows,
            tick_spd,
            t_close,
            curr_atr,
        )


# ============================================================
# Label Generation Logic (Common)
# ============================================================
def generate_labels_numpy(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atrs: np.ndarray,
    horizon: int,
    threshold_factor: float = 1.0,
    min_limit: float = 10.0,
    cost_floor: float = 0.0,
    min_limit_mode: str = "max",
) -> np.ndarray:
    """
    Triple Barrier Method によるラベリング (NumPyベクトル化実装)。

    期間(horizon)内で「最初に」バリアに触れた方向を判定する。

    Args:
        prices: エントリー判定価格(Close等)
        highs: 高値系列
        lows: 安値系列
        atrs: ATR系列
        horizon: 保持期間
        threshold_factor: ATRにかける係数
        min_limit: 最小閾値
        cost_floor: コスト（手数料・スリップ）に基づく最小閾値
        min_limit_mode: "fixed", "cost", または "max"

    Returns:
        np.ndarray: ラベル配列 (0: Neutral, 1: Long, 2: Short)
    """
    n = int(len(prices))
    labels = np.zeros(n, dtype=np.int64)

    if n <= 0 or horizon <= 0:
        return labels

    # 閾値の計算
    th_atr = atrs * float(threshold_factor)
    mode = str(min_limit_mode or "max").lower()
    if mode == "fixed":
        thresholds = np.maximum(th_atr, float(min_limit))
    elif mode == "cost":
        thresholds = np.maximum(th_atr, float(cost_floor))
    else:
        thresholds = np.maximum(th_atr, max(float(min_limit), float(cost_floor)))

    # スライディングウィンドウを用いて未来期間を一括抽出
    w = horizon + 1
    if n < w:
        return labels

    highs_w = sliding_window_view(highs, window_shape=w)
    lows_w = sliding_window_view(lows, window_shape=w)

    valid_len = min(len(highs_w), len(lows_w), n - horizon)
    if valid_len <= 0:
        return labels

    p0 = prices[:valid_len].astype(np.float32, copy=False)
    th0 = thresholds[:valid_len].astype(np.float32, copy=False)

    # 0番目は現在の足なので除外、1..horizon番目の未来を見る
    future_highs = highs_w[:valid_len, 1:]
    future_lows = lows_w[:valid_len, 1:]

    upper = p0[:, None] + th0[:, None]
    lower = p0[:, None] - th0[:, None]

    hit_up = future_highs > upper
    hit_dn = future_lows < lower

    # 最初に True になった位置（インデックス）を取得
    idx_grid = np.arange(horizon, dtype=np.int32)[None, :]
    inf = horizon + 1
    idx_up = np.where(hit_up, idx_grid, inf).min(axis=1)
    idx_dn = np.where(hit_dn, idx_grid, inf).min(axis=1)

    long_mask = idx_up < idx_dn
    short_mask = idx_dn < idx_up

    labels[:valid_len][long_mask] = 1
    labels[:valid_len][short_mask] = 2
    return labels
