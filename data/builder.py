# data/builder.py
"""
File: data/builder.py

ソースコードの役割:
本モジュールは、データセット構築、正規化（RankGauss等）、動的ラベルスケール調整、および
PyTorch DataLoaderの生成を担当します。データロード、特徴量計算、データ準備の各責務を
専用のクラス（MarketDataLoader, FeaturePipeline, DatasetBuilder）に委譲することで、
柔軟なパイプライン構築を可能にします。
"""

import logging
from typing import Tuple, List, Any, Optional

import numpy as np
import polars as pl
from torch.utils.data import DataLoader

from util.utils import PerfTimer, RankGaussScaler
from data.dataset import TFTDataset, generate_labels_numpy
from features.pipeline import FeaturePipeline
from data.data_loader import MarketDataLoader
from data.dataset_builder import DatasetBuilder


def auto_tune_label_threshold_scale(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atrs: np.ndarray,
    horizon: int,
    base_scale: float,
    min_limit: float,
    min_neutral_ratio: float,
    max_neutral_ratio: float,
) -> float:
    """ラベル分布（Neutral/Trade）の偏りを抑えるため、Foldごとに label_threshold_scale を自動調整します。

    Args:
        prices (np.ndarray): 終値の配列。
        highs (np.ndarray): 高値の配列。
        lows (np.ndarray): 安値の配列。
        atrs (np.ndarray): ATR（Average True Range）の配列。
        horizon (int): 予測ホライズン（期間）。
        base_scale (float): 基準となるラベル閾値のスケール。
        min_limit (float): 最小の閾値制限。
        min_neutral_ratio (float): Neutral（静観）ラベルの最小許容割合。
        max_neutral_ratio (float): Neutral（静観）ラベルの最大許容割合。

    Returns:
        float: 最適化された閾値スケール値。条件を満たすものがない場合は base_scale を返します。
    """
    if len(prices) == 0:
        return float(base_scale)

    target = 0.5 * (min_neutral_ratio + max_neutral_ratio)
    candidates = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]

    best_scale = float(base_scale)
    best_dist = float("inf")
    best_in_range = False

    for mult in candidates:
        scale = float(base_scale) * float(mult)
        scaled_limit = max(float(min_limit) * float(mult), 30.0)

        labels = generate_labels_numpy(
            prices,
            highs,
            lows,
            atrs,
            horizon,
            threshold_factor=scale,
            min_limit=scaled_limit,
        )
        if len(labels) == 0:
            continue

        neutral_ratio = float((labels == 0).mean())
        dist = abs(neutral_ratio - target)
        in_range = (neutral_ratio >= min_neutral_ratio) and (
            neutral_ratio <= max_neutral_ratio
        )

        if (in_range and not best_in_range) or (
            in_range == best_in_range and dist < best_dist
        ):
            best_scale = float(scale)
            best_dist = float(dist)
            best_in_range = bool(in_range)

    return float(best_scale)


def _generate_fold_labels(
    cfg: Any,
    p_close_tr: np.ndarray,
    p_high_tr: np.ndarray,
    p_low_tr: np.ndarray,
    raw_atr_tr: np.ndarray,
    p_close_val: np.ndarray,
    p_high_val: np.ndarray,
    p_low_val: np.ndarray,
    raw_atr_val: np.ndarray,
    p_close_test: np.ndarray,
    p_high_test: np.ndarray,
    p_low_test: np.ndarray,
    raw_atr_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train, Val, Testの各データセットに対するラベルを生成します。必要に応じて閾値の自動調整を行います。

    Args:
        cfg (Any): 設定オブジェクト。
        p_close_tr (np.ndarray): 訓練データの終値。
        p_high_tr (np.ndarray): 訓練データの高値。
        p_low_tr (np.ndarray): 訓練データの安値。
        raw_atr_tr (np.ndarray): 訓練データのATR。
        p_close_val, p_high_val, p_low_val, raw_atr_val: 検証データ。
        p_close_test, p_high_test, p_low_test, raw_atr_test: テストデータ。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 訓練、検証、テスト用のラベル配列。
    """
    tuned_scale = cfg.features.label_threshold_scale
    if cfg.train.auto_tune_label:
        tuned_scale = auto_tune_label_threshold_scale(
            p_close_tr,
            p_high_tr,
            p_low_tr,
            raw_atr_tr,
            cfg.features.predict_horizon,
            cfg.features.label_threshold_scale,
            cfg.features.label_min_limit,
            cfg.train.min_neutral_ratio,
            cfg.train.max_neutral_ratio,
        )

    tuned_min_limit = max(
        cfg.features.label_min_limit
        * (tuned_scale / cfg.features.label_threshold_scale),
        30.0,
    )

    y_labels_tr = generate_labels_numpy(
        p_close_tr,
        p_high_tr,
        p_low_tr,
        raw_atr_tr,
        cfg.features.predict_horizon,
        tuned_scale,
        tuned_min_limit,
    )
    y_labels_val = generate_labels_numpy(
        p_close_val,
        p_high_val,
        p_low_val,
        raw_atr_val,
        cfg.features.predict_horizon,
        tuned_scale,
        tuned_min_limit,
    )
    y_labels_test = (
        generate_labels_numpy(
            p_close_test,
            p_high_test,
            p_low_test,
            raw_atr_test,
            cfg.features.predict_horizon,
            tuned_scale,
            tuned_min_limit,
        )
        if len(p_close_test) > 0
        else np.array([])
    )

    return y_labels_tr, y_labels_val, y_labels_test


def _scale_fold_features(
    c_tr: np.ndarray,
    c_val: np.ndarray,
    c_test: np.ndarray,
    s_tr: np.ndarray,
    s_val: np.ndarray,
    s_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """連続値特徴量および状態特徴量に対するスケーリング（RankGauss、標準化）を適用します。

    Args:
        c_tr, c_val, c_test (np.ndarray): 連続値特徴量（Continuous features）の配列。
        s_tr, s_val, s_test (np.ndarray): 状態特徴量（State features）の配列。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            スケーリング済みの c_tr, c_val, c_test, s_tr, s_val, s_test
    """
    # Scaling for continuous features
    rg_scaler = RankGaussScaler()
    rg_scaler.fit(c_tr)
    c_tr = rg_scaler.transform(c_tr).astype(np.float32)
    c_val = rg_scaler.transform(c_val).astype(np.float32)
    if len(c_test) > 0:
        c_test = rg_scaler.transform(c_test).astype(np.float32)

    # Scaling for state features
    s_mean, s_std = s_tr.mean(axis=0), np.maximum(s_tr.std(axis=0), 1.0)
    s_tr = ((s_tr - s_mean) / s_std).astype(np.float32)
    s_val = ((s_val - s_mean) / s_std).astype(np.float32)
    if len(s_test) > 0:
        s_test = ((s_test - s_mean) / s_std).astype(np.float32)

    return c_tr, c_val, c_test, s_tr, s_val, s_test


def _create_dataloader(
    c: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    p_high: np.ndarray,
    p_low: np.ndarray,
    p_close: np.ndarray,
    raw_atr: np.ndarray,
    ts: np.ndarray,
    labels: np.ndarray,
    cfg: Any,
    device_type: str,
    is_train: bool = False,
) -> Optional[DataLoader]:
    """TFTDatasetをインスタンス化し、PyTorchのDataLoaderを構築します。

    Args:
        c, s, y, p_high, p_low, p_close, raw_atr, ts, labels (np.ndarray): データセット構築に必要な各配列。
        cfg (Any): 設定オブジェクト。
        device_type (str): デバイスタイプ ("cuda" or "cpu")。
        is_train (bool, optional): 訓練用データローダーかどうか。デフォルトはFalse。

    Returns:
        Optional[DataLoader]: 構築されたDataLoader。入力配列が空の場合はNoneを返します。
    """
    if len(y) == 0:
        return None

    dataset = TFTDataset(
        c,
        s,
        y,
        p_high,
        p_low,
        p_close.copy(),
        raw_atr,
        ts,
        cfg.features.seq_len,
        cfg.features.predict_horizon,
        stride=cfg.features.dataset_stride,
        precomputed_labels=labels,
    )

    if is_train:
        return DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.train.num_workers,
            prefetch_factor=(
                cfg.train.prefetch_factor if cfg.train.num_workers > 0 else None
            ),
            persistent_workers=(
                cfg.train.persistent_workers if cfg.train.num_workers > 0 else False
            ),
            pin_memory=(device_type == "cuda"),
        )
    else:
        return DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device_type == "cuda"),
        )


def build_fold_dataloaders(
    data_loader: MarketDataLoader,
    pipeline: FeaturePipeline,
    ds_builder: DatasetBuilder,
    train_dates: List[Any],
    val_dates: List[Any],
    test_dates: List[Any],
    cfg: Any,
    logger: logging.Logger,
    fold: int,
    device_type: str,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], np.ndarray]:
    """指定された日付リストに基づいてデータを切り出し、スケーリングとラベル付けを行い、DataLoaderを構築します。
    複雑な処理はプライベート関数に委譲し、オーケストレーターとして機能します。

    Args:
        data_loader (MarketDataLoader): マーケットデータのロードを担当するインスタンス。
        pipeline (FeaturePipeline): 特徴量エンジニアリングを担当するインスタンス。
        ds_builder (DatasetBuilder): NumPy配列への変換とデータセット準備を担当するインスタンス。
        train_dates (List[Any]): 訓練データとして使用する日付のリスト。
        val_dates (List[Any]): 検証データとして使用する日付のリスト。
        test_dates (List[Any]): テストデータとして使用する日付のリスト。
        cfg (Any): 設定情報（Hydra Configなどを想定）。
        logger (logging.Logger): ロガーインスタンス。
        fold (int): 現在の交差検証のフォールド番号。
        device_type (str): 使用するデバイス（"cuda" または "cpu"）。

    Returns:
        Tuple[DataLoader, DataLoader, Optional[DataLoader], np.ndarray]:
            - 訓練用DataLoader
            - 検証用DataLoader
            - テスト用DataLoader（データが存在しない場合はNone）
            - 訓練用の生成済みラベル配列
    """
    import datetime

    # 1. データロードと特徴量計算
    start_dt = train_dates[0] - datetime.timedelta(days=2)
    end_dt = (test_dates[-1] if test_dates else val_dates[-1]) + datetime.timedelta(
        days=1
    )

    with PerfTimer(
        logger,
        f"fold{fold}/load_chunk.collect",
        {"start": str(start_dt), "end": str(end_dt)},
    ):
        lf = data_loader.load_lazy_chunk(start_dt, end_dt)
        df_raw = lf.collect()

    with PerfTimer(logger, f"fold{fold}/compute_features"):
        df_chunk = pipeline.compute_features(df_raw)

    # 2. データのフィルタリングとNumPy配列化
    df_tr = df_chunk.filter(pl.col("timestamp").dt.date().is_in(train_dates))
    df_val = df_chunk.filter(pl.col("timestamp").dt.date().is_in(val_dates))
    df_test = df_chunk.filter(pl.col("timestamp").dt.date().is_in(test_dates))

    ts_tr = df_tr["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
    ts_val = df_val["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
    ts_test = (
        df_test["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
        if df_test.height > 0
        else np.array([], dtype=np.int64)
    )

    c_tr, s_tr, y_tr = ds_builder.prepare_numpy_data(df_tr)
    c_val, s_val, y_val = ds_builder.prepare_numpy_data(df_val)
    c_test, s_test, y_test = ds_builder.prepare_numpy_data(df_test)

    p_close_tr, p_high_tr, p_low_tr = y_tr[:, 0], y_tr[:, 1], y_tr[:, 2]
    p_close_val, p_high_val, p_low_val = y_val[:, 0], y_val[:, 1], y_val[:, 2]
    p_close_test = y_test[:, 0] if len(y_test) > 0 else np.array([])
    p_high_test = y_test[:, 1] if len(y_test) > 0 else np.array([])
    p_low_test = y_test[:, 2] if len(y_test) > 0 else np.array([])

    raw_atr_tr = df_tr["atr"].to_numpy().astype(np.float32)
    raw_atr_val = df_val["atr"].to_numpy().astype(np.float32)
    raw_atr_test = (
        df_test["atr"].to_numpy().astype(np.float32)
        if len(df_test) > 0
        else np.array([])
    )

    # 3. ラベル生成
    y_labels_tr, y_labels_val, y_labels_test = _generate_fold_labels(
        cfg,
        p_close_tr,
        p_high_tr,
        p_low_tr,
        raw_atr_tr,
        p_close_val,
        p_high_val,
        p_low_val,
        raw_atr_val,
        p_close_test,
        p_high_test,
        p_low_test,
        raw_atr_test,
    )

    # 4. スケーリング
    c_tr, c_val, c_test, s_tr, s_val, s_test = _scale_fold_features(
        c_tr, c_val, c_test, s_tr, s_val, s_test
    )

    # 5. DataLoader構築
    loader_train = _create_dataloader(
        c_tr,
        s_tr,
        y_tr,
        p_high_tr,
        p_low_tr,
        p_close_tr,
        raw_atr_tr,
        ts_tr,
        y_labels_tr,
        cfg,
        device_type,
        is_train=True,
    )
    loader_val = _create_dataloader(
        c_val,
        s_val,
        y_val,
        p_high_val,
        p_low_val,
        p_close_val,
        raw_atr_val,
        ts_val,
        y_labels_val,
        cfg,
        device_type,
        is_train=False,
    )
    loader_test = _create_dataloader(
        c_test,
        s_test,
        y_test,
        p_high_test,
        p_low_test,
        p_close_test,
        raw_atr_test,
        ts_test,
        y_labels_test,
        cfg,
        device_type,
        is_train=False,
    )

    return loader_train, loader_val, loader_test, y_labels_tr
