# data/dataset_builder.py
"""
File: data/dataset_builder.py

ソースコードの役割:
本モジュールは、特徴量計算済みのDataFrameを受け取り、モデルに入力可能なNumPy配列への変換、
ウォークフォワード検証用の期間分割、およびPyTorchの TimeSeriesDataset を構築する
DatasetBuilderクラスを提供します。
"""

import logging
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from typing import List, Tuple, Iterator
from torch.utils.data import Dataset

from config import cfg
from data.dataset import TimeSeriesDataset
from data.data_loader import MarketDataLoader
from features.pipeline import FeaturePipeline


class DatasetBuilder:
    """
    特徴量からPyTorch Datasetへの変換と、学習/検証期間の分割を行うクラス。
    """

    def __init__(self):
        """
        DatasetBuilderの初期化。
        設定ファイル（cfg）から特徴量カラムや分割期間パラメータを読み込みます。
        """
        self.continuous_cols = cfg.features.continuous_cols
        self.static_cols = cfg.features.static_cols

        # 期間分割の設定
        self.train_days = cfg.train_days
        self.val_days = cfg.val_days
        self.step_days = cfg.step_days

    def prepare_numpy_data(
        self, df: pl.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        データフレームをNumPy配列に変換し、モデルの入力形式に整えます。

        Args:
            df (pl.DataFrame): 特徴量計算済みのデータフレーム

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - cont: 連続値特徴量のNumPy配列
                - static: 静的特徴量のNumPy配列
                - target: ターゲット変数のNumPy配列
        """
        missing = [c for c in self.continuous_cols if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(0.0).alias(c) for c in missing])

        cont = np.nan_to_num(
            df.select(self.continuous_cols).to_numpy().astype(np.float32),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        static = np.nan_to_num(
            df.select(self.static_cols).to_numpy().astype(np.float32), nan=0.0
        )

        # Target列: [Close, High, Low, TickSpeed, MinutesToClose, Open]
        target = (
            df.select(
                [
                    pl.col("close"),
                    pl.col("high"),
                    pl.col("low"),
                    pl.col("tick_speed_ratio"),
                    pl.col("minutes_to_close"),
                    pl.col("open"),
                ]
            )
            .to_numpy()
            .astype(np.float32)
        )

        return cont, static, target

    def walk_forward_split(
        self, dates: List[datetime]
    ) -> Iterator[Tuple[List[datetime], List[datetime], List[datetime]]]:
        """
        ウォークフォワードテスト用の日付分割ジェネレーター。

        Args:
            dates (List[datetime]): 利用可能な取引日のリスト

        Yields:
            Tuple[List[datetime], List[datetime], List[datetime]]:
                - train_dates: 学習用期間の日付リスト
                - val_dates: 検証用期間の日付リスト
                - test_dates: テスト用期間の日付リスト（今回は空リストを返す仕様）
        """
        dates = sorted(dates)
        total_days = len(dates)
        window_size = self.train_days + self.val_days

        if total_days < window_size:
            logging.warning("Not enough data for a full walk-forward split.")
            return

        for i in range(0, total_days - window_size + 1, self.step_days):
            train_end = i + self.train_days
            val_end = train_end + self.val_days
            yield (dates[i:train_end], dates[train_end:val_end], [])

    def build_dataset(
        self,
        dates: List[datetime],
        data_loader: MarketDataLoader,
        pipeline: FeaturePipeline,
    ) -> Dataset:
        """
        指定された日付のデータからPyTorch用データセットを構築します。
        内部でDataLoaderとPipelineをオーケストレーションします。

        Args:
            dates (List[datetime]): 抽出対象の取引日リスト
            data_loader (MarketDataLoader): データ読み込みインスタンス
            pipeline (FeaturePipeline): 特徴量計算インスタンス

        Returns:
            Dataset: PyTorchのTimeSeriesDatasetインスタンス

        Raises:
            ValueError: 空の日付リストが渡された場合
        """
        if not dates:
            raise ValueError("Empty date list provided for dataset build.")

        start_dt = dates[0]
        end_dt = dates[-1] + timedelta(days=1)

        lf = data_loader.load_lazy_chunk(start_dt, end_dt)
        df = lf.collect()

        df_features = pipeline.compute_features(df)
        cont, static, target = self.prepare_numpy_data(df_features)

        # ATR抽出 (推論用Datasetビルド時)
        atr_data = df_features["atr"].to_numpy().astype(np.float32)

        return TimeSeriesDataset(
            cont,
            static,
            target,
            atr_data,
            seq_len=cfg.features.seq_len,
            prediction_horizon=cfg.features.predict_horizon,
            stride=cfg.features.dataset_stride,
        )
