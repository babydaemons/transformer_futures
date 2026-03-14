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
from typing import Tuple, List, Any, Optional, Union
import datetime

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

    イテレーションを用いてニュートラル割合が指定範囲内に収まるよう、動的にスケール値を探索します。

    Args:
        prices (np.ndarray): 終値の配列。
        highs (np.ndarray): 高値の配列。
        lows (np.ndarray): 安値の配列。
        atrs (np.ndarray): ATR（Average True Range）の配列。
        horizon (int): 予測ホライズン（期間）。
        base_scale (float): 探索の基準となるラベル閾値のスケール。
        min_limit (float): 最小の閾値制限。
        min_neutral_ratio (float): Neutral（静観）ラベルの最小許容割合。
        max_neutral_ratio (float): Neutral（静観）ラベルの最大許容割合。

    Returns:
        float: 最適化された閾値スケール値。条件を満たすものが見つからない場合は最も近かった値を返します。
    """
    logger = logging.getLogger(__name__)
    scale = base_scale
    step = 0.1
    max_iter = 20
    best_scale = scale
    best_diff = float("inf")
    ratio = 0.0

    for _ in range(max_iter):
        labels = generate_labels_numpy(
            prices=prices,
            highs=highs,
            lows=lows,
            atrs=atrs,
            horizon=horizon,
            threshold_scale=scale,
            min_limit=min_limit,
            cost_buffer=0.0,
            mode="cost",
        )

        n_total = len(labels)
        if n_total == 0:
            break

        n_neutral = np.sum(labels == 0)
        ratio = n_neutral / n_total

        diff = 0.0
        if ratio < min_neutral_ratio:
            diff = min_neutral_ratio - ratio
            scale += step
        elif ratio > max_neutral_ratio:
            diff = ratio - max_neutral_ratio
            scale -= step
        else:
            best_scale = scale
            break

        if diff < best_diff:
            best_diff = diff
            best_scale = scale

        step *= 0.8

    logger.info(
        f"Auto-tuned threshold_scale: {base_scale:.3f} -> {best_scale:.3f} (Neutral Ratio: {ratio:.1%})"
    )
    return best_scale


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
    """Train, Val, Testの各データセットに対するラベルを生成します。

    設定により、訓練データのラベル分布から閾値の自動調整（auto-tuning）を実行します。

    Args:
        cfg (Any): アプリケーション全体の設定オブジェクト。
        p_close_tr (np.ndarray): 訓練データの終値。
        p_high_tr (np.ndarray): 訓練データの高値。
        p_low_tr (np.ndarray): 訓練データの安値。
        raw_atr_tr (np.ndarray): 訓練データのATR。
        p_close_val, p_high_val, p_low_val, raw_atr_val: 検証データ。
        p_close_test, p_high_test, p_low_test, raw_atr_test: テストデータ。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 訓練、検証、テスト用のラベル配列。
    """
    t_scale = cfg.features.label_threshold_scale
    if cfg.train.auto_tune_label and len(p_close_tr) > 0:
        t_scale = auto_tune_label_threshold_scale(
            prices=p_close_tr,
            highs=p_high_tr,
            lows=p_low_tr,
            atrs=raw_atr_tr,
            horizon=cfg.features.predict_horizon,
            base_scale=t_scale,
            min_limit=cfg.features.label_min_limit,
            min_neutral_ratio=cfg.train.min_neutral_ratio,
            max_neutral_ratio=cfg.train.max_neutral_ratio,
        )

    def _gen(c, h, l, a):
        if len(c) == 0:
            return np.array([])
        return generate_labels_numpy(
            prices=c,
            highs=h,
            lows=l,
            atrs=a,
            horizon=cfg.features.predict_horizon,
            threshold_scale=t_scale,
            min_limit=cfg.features.label_min_limit,
            cost_buffer=cfg.features.label_cost_buffer,
            mode=cfg.features.label_min_limit_mode,
        )

    return (
        _gen(p_close_tr, p_high_tr, p_low_tr, raw_atr_tr),
        _gen(p_close_val, p_high_val, p_low_val, raw_atr_val),
        _gen(p_close_test, p_high_test, p_low_test, raw_atr_test),
    )


def _scale_fold_features(
    c_tr: np.ndarray,
    c_val: np.ndarray,
    c_test: np.ndarray,
    s_tr: np.ndarray,
    s_val: np.ndarray,
    s_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """連続値特徴量および状態特徴量に対するスケーリング（RankGauss等）を適用します。

    Args:
        c_tr, c_val, c_test (np.ndarray): 連続値特徴量（Continuous features）の配列。
        s_tr, s_val, s_test (np.ndarray): 状態特徴量（State features）の配列。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            スケーリング済みの c_tr, c_val, c_test, s_tr, s_val, s_test
    """
    if c_tr.shape[1] > 0 and len(c_tr) > 0:
        scaler = RankGaussScaler()
        scaler.fit(c_tr)
        c_tr = scaler.transform(c_tr)
        if len(c_val) > 0:
            c_val = scaler.transform(c_val)
        if len(c_test) > 0:
            c_test = scaler.transform(c_test)

    return c_tr, c_val, c_test, s_tr, s_val, s_test


def _create_dataloader(
    c_feat: np.ndarray,
    s_feat: np.ndarray,
    y: np.ndarray,
    p_high: np.ndarray,
    p_low: np.ndarray,
    p_close: np.ndarray,
    raw_atr: np.ndarray,
    ts: np.ndarray,
    y_labels: np.ndarray,
    cfg: Any,
    device_type: str,
    is_train: bool = False,
) -> Optional[DataLoader]:
    """TFTDatasetをインスタンス化し、PyTorchのDataLoaderを構築します。

    Args:
        c_feat (np.ndarray): 連続値特徴量の配列。
        s_feat (np.ndarray): 状態特徴量の配列。
        y (np.ndarray): ターゲット変数のベース配列。
        p_high, p_low, p_close (np.ndarray): 高値、安値、終値の配列。
        raw_atr (np.ndarray): ATRの配列。
        ts (np.ndarray): タイムスタンプ配列。
        y_labels (np.ndarray): 計算済みの目的変数（ラベル）配列。
        cfg (Any): 設定オブジェクト。
        device_type (str): デバイスタイプ ("cuda" or "cpu")。
        is_train (bool, optional): 訓練用データローダーかどうか。デフォルトはFalse。

    Returns:
        Optional[DataLoader]: 構築されたDataLoader。入力配列が空の場合はNoneを返します。
    """
    if len(c_feat) == 0:
        return None

    dataset = TFTDataset(
        c_feat=c_feat,
        s_feat=s_feat,
        y=y,
        p_high=p_high,
        p_low=p_low,
        p_close=p_close,
        raw_atr=raw_atr,
        ts=ts,
        labels=y_labels,
        seq_len=cfg.features.seq_len,
        predict_horizon=cfg.features.predict_horizon,
        stride=cfg.features.dataset_stride if is_train else 1,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=is_train,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory if device_type == "cuda" else False,
        prefetch_factor=(
            cfg.train.prefetch_factor if cfg.train.num_workers > 0 else None
        ),
        persistent_workers=(
            cfg.train.persistent_workers if cfg.train.num_workers > 0 else False
        ),
        drop_last=is_train,
    )


def build_fold_dataloaders(
    data_loader: MarketDataLoader,
    pipeline: FeaturePipeline,
    ds_builder: DatasetBuilder,
    train_dates: List[datetime.date],
    val_dates: List[datetime.date],
    test_dates: List[datetime.date],
    cfg: Any,
    logger: logging.Logger,
    fold: int,
    device_type: str,
) -> Tuple[
    Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], np.ndarray
]:
    """1Fold分のデータをロード、特徴量計算、分割、スケーリング、ラベル生成を行い、PyTorch DataLoaderを構築します。

    複雑な処理はプライベート関数に委譲し、オーケストレーターとして機能します。

    Args:
        data_loader (MarketDataLoader): データロードを担当するインスタンス。
        pipeline (FeaturePipeline): 特徴量計算を担当するパイプライン。
        ds_builder (DatasetBuilder): Dataset変換用ビルダー。
        train_dates (List[datetime.date]): 訓練データとして使用する日付リスト。
        val_dates (List[datetime.date]): 検証データとして使用する日付リスト。
        test_dates (List[datetime.date]): テストデータとして使用する日付リスト。
        cfg (Any): 設定情報。
        logger (logging.Logger): ロガーインスタンス。
        fold (int): 現在のフォールド番号。
        device_type (str): 使用するデバイス（"cuda" または "cpu"）。

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], np.ndarray]:
            訓練用、検証用、テスト用の各DataLoader、および訓練用のラベル配列。
    """
    # 1. データロードと特徴量計算
    # 結合時の欠損を防ぐため、開始日を2日前、終了日を1日後に拡張してチャンクを取得します。
    start_dt = train_dates[0] - datetime.timedelta(days=2)
    end_dt = (test_dates[-1] if test_dates else val_dates[-1]) + datetime.timedelta(
        days=1
    )

    with PerfTimer(logger, f"Fold {fold}: load_and_compute_features"):
        # 1) 生データをロード (LazyFrame)
        df_raw = data_loader.load_lazy_chunk(start_dt, end_dt)

        # 2) chunk内で特徴量を計算 (LazyFrameのままパイプラインを通す)
        df_chunk = pipeline.compute_features(df_raw)

        # 3) Train/Val/Test に分割 (ts_col を使用してフィルタリング)
        ts_col = cfg.features.ts_col

        df_tr = df_chunk.filter(pl.col(ts_col).dt.date().is_in(train_dates))
        df_v = df_chunk.filter(pl.col(ts_col).dt.date().is_in(val_dates))
        df_te = df_chunk.filter(pl.col(ts_col).dt.date().is_in(test_dates))

        # 4) メモリ展開 (ここで特徴量の計算とフィルタリングが実行される)
        df_tr = df_tr.collect()
        df_v = df_v.collect()
        df_te = df_te.collect()

    # DataFrameからNumPy配列への変換準備
    c_tr, s_tr, y_tr, p_high_tr, p_low_tr, p_close_tr, ts_tr = ds_builder.prepare_data(
        df_tr
    )
    c_val, s_val, y_val, p_high_val, p_low_val, p_close_val, ts_val = (
        ds_builder.prepare_data(df_v)
    )
    c_test, s_test, y_test, p_high_test, p_low_test, p_close_test, ts_test = (
        ds_builder.prepare_data(df_te)
    )

    raw_atr_tr = (
        df_tr["atr"].to_numpy().astype(np.float32) if len(df_tr) > 0 else np.array([])
    )
    raw_atr_val = (
        df_v["atr"].to_numpy().astype(np.float32) if len(df_v) > 0 else np.array([])
    )
    raw_atr_test = (
        df_te["atr"].to_numpy().astype(np.float32) if len(df_te) > 0 else np.array([])
    )

    # ラベル生成
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

    # スケーリング
    c_tr, c_val, c_test, s_tr, s_val, s_test = _scale_fold_features(
        c_tr, c_val, c_test, s_tr, s_val, s_test
    )

    # DataLoader構築
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
