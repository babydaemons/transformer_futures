# data/data_loader.py
"""
File: data/data_loader.py

ソースコードの役割:
本モジュールは、PolarsのLazyFrameを用いたParquetおよびCSVファイルの効率的な読み込みと、
メインデータ（NK225）に対して各種マクロデータ（USDJPY、S&P500、XAUUSD、XTIUSD）や
需給データ（空売り比率、投資主体別売買動向）を時間軸ベースで結合（join_asof）する
データローダークラスを提供します。
tree.txt に示される階層化されたディレクトリ構造（年別フォルダごとのParquet分割）に
対応し、ワイルドカードによる複数ファイルの一括遅延ロードを行います。
"""

import os
import logging
import polars as pl
from datetime import timedelta, datetime
from typing import List

from config import cfg


class MarketDataLoader:
    """
    市場データ（日経225先物、各種マクロ指標、需給データ）の読み込みと結合を行うクラス。
    """

    def __init__(self):
        """
        MarketDataLoaderの初期化。
        設定ファイル（cfg）から各種データのパス（ディレクトリまたはファイル）を読み込みます。
        tree.txt に示される階層化されたディレクトリ構造（例: parquet/NK225/2018/*.parquet）
        に対応するため、指定されたパスがディレクトリの場合はワイルドカードを自動補完します。
        """
        # Parquet形式のデータパス解決
        self.nk225_path = self._resolve_path(cfg.features.nk225_file)
        self.usdjpy_path = self._resolve_path(cfg.features.usdjpy_file)
        self.sp500_path = self._resolve_path(cfg.features.sp500_file)
        self.xauusd_path = self._resolve_path(cfg.features.xauusd_file)
        self.xtiusd_path = self._resolve_path(cfg.features.xtiusd_file)

        # CSV形式のデータはワイルドカード補完を適用せずそのまま読み込む
        self.short_selling_path = cfg.features.short_selling_file
        self.investor_path = cfg.features.investor_file

    def _resolve_path(self, path_str: str) -> str:
        """
        指定されたパスがディレクトリである場合、年別サブディレクトリ内の
        Parquetファイルをすべて読み込めるようにワイルドカードパスを構築します。

        Args:
            path_str (str): 設定ファイルで指定されたパス文字列

        Returns:
            str: 解決済みのパス文字列（ワイルドカードを含む場合あり）
        """
        if os.path.isdir(path_str):
            # 例: C:/TRANSFORMER_FUTURES_DATA/parquet/NK225 -> C:/TRANSFORMER_FUTURES_DATA/parquet/NK225/*/*.parquet
            return os.path.join(path_str, "*", "*.parquet")
        elif not path_str.endswith(".parquet") and "*" not in path_str:
            # 存在しないパスだがディレクトリを意図していると推測される場合へのフォールバック
            return os.path.join(path_str, "*", "*.parquet")
        return path_str

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
        指定された期間のメインデータ、マクロデータ、および需給データを遅延評価で読み込み、結合します。

        Args:
            start_dt (datetime): 取得開始日時
            end_dt (datetime): 取得終了日時

        Returns:
            pl.LazyFrame: 全てのデータが時間軸ベースで結合されたLazyFrame
        """
        # メインデータの読み込みとフィルタリング
        lf_nk = pl.scan_parquet(self.nk225_path).filter(
            pl.col("trade_ts").is_between(start_dt, end_dt)
        )

        # マクロデータの準備
        lf_usd = self._prepare_macro_data(self.usdjpy_path, "usdjpy", start_dt, end_dt)
        lf_spx = self._prepare_macro_data(self.sp500_path, "sp500", start_dt, end_dt)
        lf_xau = self._prepare_macro_data(self.xauusd_path, "xauusd", start_dt, end_dt)
        lf_xti = self._prepare_macro_data(self.xtiusd_path, "xtiusd", start_dt, end_dt)

        # 需給データ（日次・週次データ）の準備
        lf_ss = self._prepare_short_selling_data(self.short_selling_path)
        lf_inv = self._prepare_investor_data(self.investor_path)

        # 時間軸ベースでの結合（ASOF JOIN）
        # backward方向のasof joinを使用し、未来情報の漏洩を防ぐ
        return (
            lf_nk.sort("trade_ts")
            .join_asof(lf_usd, on="trade_ts", strategy="backward")
            .join_asof(lf_spx, on="trade_ts", strategy="backward")
            .join_asof(lf_xau, on="trade_ts", strategy="backward")
            .join_asof(lf_xti, on="trade_ts", strategy="backward")
            .join_asof(lf_ss, on="trade_ts", strategy="backward")
            .join_asof(lf_inv, on="trade_ts", strategy="backward")
        )

    def _prepare_macro_data(
        self, path: str, prefix: str, start_dt: datetime, end_dt: datetime
    ) -> pl.LazyFrame:
        """
        マクロ価格データを準備します。結合時の欠損を防ぐため、開始時刻より少し前のデータも含めて読み込みます。

        Args:
            path (str): マクロデータのファイルパスまたはワイルドカードパス
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
            logging.warning(f"Failed to load macro data {prefix} from {path}: {e}")
            # エラー時はゼロ埋めのダミーフレームを返す
            return pl.LazyFrame(
                {
                    "trade_ts": [start_dt],
                    f"{prefix}_close": [0.0],
                }
            ).select(
                [
                    pl.col("trade_ts").cast(pl.Datetime("ns")),
                    pl.col(f"{prefix}_close").cast(pl.Float64),
                ]
            )

    def _prepare_short_selling_data(self, path: str) -> pl.LazyFrame:
        """
        空売り比率データ（日次）を準備します。
        未来の情報漏洩（Look-ahead bias）を防ぐため、発表日の翌日からデータが適用されるように日付を1日シフトします。

        Args:
            path (str): 空売り比率データのCSVファイルパス

        Returns:
            pl.LazyFrame: 前処理された空売り比率データのLazyFrame
        """
        try:
            lf = pl.scan_csv(path, separator="\t")
            return lf.select(
                [
                    (
                        pl.col("年月日")
                        .str.strptime(pl.Datetime, "%Y-%m-%d")
                        .dt.replace_time_zone(None)
                        + pl.duration(days=1)
                    ).alias("trade_ts"),
                    pl.col("比率_a_d")
                    .str.replace("%", "")
                    .cast(pl.Float64)
                    .alias("short_selling_ratio_raw"),
                ]
            ).sort("trade_ts")
        except Exception as e:
            logging.warning(f"Failed to load short selling data: {e}")
            return pl.LazyFrame(
                {"trade_ts": [datetime(2000, 1, 1)], "short_selling_ratio_raw": [0.0]}
            ).select(
                [
                    pl.col("trade_ts").cast(pl.Datetime),
                    pl.col("short_selling_ratio_raw").cast(pl.Float64),
                ]
            )

    def _prepare_investor_data(self, path: str) -> pl.LazyFrame:
        """
        投資主体別売買動向データ（週次/月次）を準備します。
        ファイル名に含まれる日付（YYYYMMDD）を基準とし、適用を1日遅らせてタイムスタンプとします。

        Args:
            path (str): 投資主体別売買動向データのCSVファイルパス

        Returns:
            pl.LazyFrame: 前処理された投資主体別売買動向データのLazyFrame
        """
        try:
            lf = pl.scan_csv(path, separator="\t")
            return (
                lf.with_columns(
                    [
                        # ファイル名 (例: 20240101.xls) から日付を抽出し、適用を1日遅らせる
                        (
                            pl.col("file_name")
                            .str.extract(r"(\d{8})")
                            .str.strptime(pl.Datetime, "%Y%m%d")
                            + pl.duration(days=1)
                        ).alias("trade_ts"),
                        # 外国人の差引額を抽出
                        pl.col("foreigners_balance")
                        .cast(pl.Float64)
                        .fill_null(0.0)
                        .alias("foreigners_balance_raw"),
                    ]
                )
                .select(["trade_ts", "foreigners_balance_raw"])
                .drop_nulls("trade_ts")
                .sort("trade_ts")
            )
        except Exception as e:
            logging.warning(f"Failed to load investor data: {e}")
            return pl.LazyFrame(
                {"trade_ts": [datetime(2000, 1, 1)], "foreigners_balance_raw": [0.0]}
            ).select(
                [
                    pl.col("trade_ts").cast(pl.Datetime),
                    pl.col("foreigners_balance_raw").cast(pl.Float64),
                ]
            )
