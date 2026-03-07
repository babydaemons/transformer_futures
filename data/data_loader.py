# data/data_loader.py
"""
File: data/data_loader.py

ソースコードの役割:
本モジュールは、PolarsのLazyFrameを用いたParquetファイルの効率的な読み込みと、
メインデータ（NK225）およびマクロデータ（USDJPY、S&P500）の時間軸ベースの
結合（join_asof）を担当するデータローダークラスを提供します。
"""

import logging
import polars as pl
from datetime import timedelta, datetime
from typing import List

from config import cfg


class MarketDataLoader:
    """
    市場データ（日経225先物、USD/JPY、S&P500）の読み込みと結合を行うクラス。
    """

    def __init__(self):
        """
        MarketDataLoaderの初期化。
        設定ファイル（cfg）から各種Parquetファイルのパスを読み込みます。
        """
        self.nk225_path = cfg.features.nk225_file
        self.usdjpy_path = cfg.features.usdjpy_file
        self.sp500_path = cfg.features.sp500_file

    def get_trading_dates(self) -> List[datetime]:
        """
        メインデータ（NK225）から取引日のリストを抽出して返します。

        Returns:
            List[datetime]: 重複のない取引日のリスト（昇順）
        """
        logging.info(f"Scanning trading dates from {self.nk225_path}...")

        dates_df = (
            pl.scan_parquet(self.nk225_path)
            .select(pl.col("trade_ts").dt.date().alias("date"))
            .unique()
            .collect()
            .sort("date")
        )

        dates = dates_df["date"].to_list()
        logging.info(f"Found {len(dates)} trading days.")
        return dates

    def load_lazy_chunk(self, start_dt: datetime, end_dt: datetime) -> pl.LazyFrame:
        """
        指定された期間のメインデータとマクロデータ（USD/JPY、S&P500）を遅延評価で読み込み、結合します。

        Args:
            start_dt (datetime): 取得開始日時
            end_dt (datetime): 取得終了日時

        Returns:
            pl.LazyFrame: メインデータとマクロデータが結合されたLazyFrame
        """
        lf_nk = pl.scan_parquet(self.nk225_path).filter(
            pl.col("trade_ts").is_between(start_dt, end_dt)
        )

        lf_usd = self._prepare_macro_data(self.usdjpy_path, "usdjpy", start_dt, end_dt)
        lf_spx = self._prepare_macro_data(self.sp500_path, "sp500", start_dt, end_dt)

        return (
            lf_nk.sort("trade_ts")
            .join_asof(lf_usd, on="trade_ts", strategy="backward")
            .join_asof(lf_spx, on="trade_ts", strategy="backward")
        )

    def _prepare_macro_data(
        self, path: str, prefix: str, start_dt: datetime, end_dt: datetime
    ) -> pl.LazyFrame:
        """
        マクロデータを準備します。結合時の欠損を防ぐため、開始時刻より少し前のデータも含めて読み込みます。

        Args:
            path (str): マクロデータのファイルパス
            prefix (str): 列名に付与するプレフィックス（例: 'usdjpy'）
            start_dt (datetime): 取得開始日時
            end_dt (datetime): 取得終了日時

        Returns:
            pl.LazyFrame: 前処理されたマクロデータのLazyFrame
        """
        try:
            # 結合前のラグマージンとして60分確保
            s_margin = start_dt - timedelta(minutes=60)
            lf = pl.scan_parquet(path)
            schema = lf.collect_schema()

            ts_col = "available_ts" if "available_ts" in schema.names() else "trade_ts"
            price_col = "close"

            return (
                lf.filter(pl.col(ts_col).is_between(s_margin, end_dt))
                .select(
                    [
                        pl.col(ts_col).alias("trade_ts"),
                        pl.col(price_col).alias(f"{prefix}_close"),
                    ]
                )
                .sort("trade_ts")
            )
        except Exception as e:
            logging.warning(f"Failed to load macro data {prefix}: {e}")
            # エラー時はゼロ埋めのダミーフレームを返す
            return pl.LazyFrame(
                {
                    "trade_ts": [start_dt],
                    f"{prefix}_close": [0.0],
                }
            ).select(
                [
                    pl.col("trade_ts").cast(pl.Datetime),
                    pl.col(f"{prefix}_close").cast(pl.Float64),
                ]
            )
