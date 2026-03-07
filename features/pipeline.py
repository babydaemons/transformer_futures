# features/pipeline.py
"""
File: features/pipeline.py

ソースコードの役割:
本モジュールは、Polars DataFrameに対する特徴量計算に特化したクラスを提供します。
基本価格指標、VWAP、マクロ要因、カレンダー特徴量などの計算をプライベートメソッドに分割し、
外部のテクニカル/マクロ計算モジュール群と統合して最終的な特徴量群を構築します。
デイトレードのタイムフレームに適応させるため、1分足ベースの微小ノイズに対するEMA平滑化を含みます。
"""

import polars as pl
import numpy as np
import math

from config import cfg

# 各ドメインからの関数インポート
from features.technicals import compute_technicals, compute_flow_imbalance_features
from features.macro import compute_macro_features
from features.volume_profile import compute_volume_profile


class FeaturePipeline:
    """
    Polars DataFrameに対して各種特徴量を計算・追加するパイプラインクラス。
    """

    def __init__(self):
        """
        FeaturePipelineの初期化。
        設定ファイル（cfg）からテクニカル指標の計算パラメータを読み込みます。
        """
        self.rsi_period = cfg.features.rsi_period
        self.macd_fast = cfg.features.macd_fast
        self.macd_slow = cfg.features.macd_slow
        self.macd_signal = cfg.features.macd_signal
        self.bb_window = cfg.features.bb_window
        self.adx_period = cfg.features.adx_period

    def compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        入力データフレームに対してすべてのドメインの特徴量エンジニアリングを統合適用します。

        Args:
            df (pl.DataFrame): 入力となる生データフレーム

        Returns:
            pl.DataFrame: 特徴量が追加・計算されたデータフレーム（欠損値除去済み）
        """
        df = self._prepare_base_columns(df)
        df = self._compute_price_volume_basics(df)
        df = self._compute_session_flags(df)

        # 分割された各外部ドメインのロジックを呼び出し
        df = compute_flow_imbalance_features(df)
        df = compute_technicals(
            df,
            self.rsi_period,
            self.macd_fast,
            self.macd_slow,
            self.bb_window,
            self.adx_period,
        )
        df = compute_macro_features(df)
        df = compute_volume_profile(df)

        # 内部での高度な特徴量計算
        df = self._compute_vwap_and_skew(df)
        df = self._compute_beta_and_distance(df)
        df = self._compute_cyclic_time_features(df)

        return df.drop_nulls()

    def _prepare_base_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        基本カラム（close, open, timestamp等）の準備を行います。

        Args:
            df (pl.DataFrame): 対象データフレーム

        Returns:
            pl.DataFrame: カラム名が標準化されたデータフレーム
        """
        if "trade_price" in df.columns and "close" not in df.columns:
            df = df.with_columns(
                [
                    pl.col("trade_price").alias("close"),
                    pl.col("high_price").alias("high"),
                    pl.col("low_price").alias("low"),
                    pl.col("trade_price")
                    .shift(1)
                    .fill_null(pl.col("trade_price"))
                    .alias("open"),
                ]
            )

        if "trade_ts" in df.columns:
            df = df.with_columns(pl.col("trade_ts").alias("timestamp"))

        return df

    def _compute_price_volume_basics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        基本的な価格・出来高の変換やボラティリティ指標を計算します。
        微小なノイズを排除するため、1分足リターンや出来高などの一部指標に対してEMAによる平滑化を行っています。

        Args:
            df (pl.DataFrame): 対象データフレーム

        Returns:
            pl.DataFrame: 基本特徴量が追加されたデータフレーム
        """
        return df.with_columns(
            [
                # 生の1分リターンと出来高にEMA(span=3)をかけ、ノイズを抑制
                pl.col("close")
                .log()
                .diff()
                .fill_null(0)
                .ewm_mean(span=3, adjust=False)
                .alias("log_ret"),
                ((pl.col("buy_volume") + pl.col("sell_volume") + 1).log())
                .ewm_mean(span=3, adjust=False)
                .alias("log_vol"),
                (pl.col("buy_volume") + 1).log().alias("log_buy_vol"),
                (pl.col("sell_volume") + 1).log().alias("log_sell_vol"),
                (
                    pl.col("close").log().diff().fill_null(0)
                    / (
                        pl.col("close").log().diff().rolling_std(12).fill_null(1.0)
                        + 1e-8
                    )
                ).alias("ret_div_vol"),
                (pl.col("buy_volume") - pl.col("sell_volume"))
                .rolling_sum(6)
                .fill_null(0)
                .alias("ofi_signal"),
                (
                    (pl.col("buy_volume") - pl.col("sell_volume"))
                    / (pl.col("buy_volume") + pl.col("sell_volume") + 1e-8)
                )
                .fill_nan(0)
                .ewm_mean(span=3, adjust=False)  # 需給圧力のジッターを平滑化
                .alias("volume_pressure"),
                pl.col("tick_count")
                .diff()
                .fill_null(0)
                .ewm_mean(span=5, adjust=False)
                .alias("trade_freq_accel"),
                (pl.col("high") - pl.col("low")).fill_null(0).alias("price_spread"),
                (((pl.col("high") / pl.col("low")).log() ** 2) / (4 * np.log(2))).alias(
                    "parkinson_vol"
                ),
                pl.col("close")
                .log()
                .diff()
                .rolling_std(12)
                .fill_null(0)
                .alias("realized_vol"),
                (
                    (pl.col("high") - pl.col("close"))
                    + (pl.col("close") - pl.col("low"))
                ).alias("shadow_range"),
            ]
        )

    def _compute_session_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        セッション情報（日中/夜間フラグ）や引けまでの時間を計算します。

        Args:
            df (pl.DataFrame): 対象データフレーム

        Returns:
            pl.DataFrame: セッション関連フラグが追加されたデータフレーム
        """
        total_minutes = df["timestamp"].dt.hour().cast(pl.Int32) * 60 + df[
            "timestamp"
        ].dt.minute().cast(pl.Int32)

        df = df.with_columns(
            pl.when((total_minutes >= 525) & (total_minutes < 915))
            .then(915 - total_minutes)
            .when(total_minutes >= 990)
            .then((24 * 60 + 360) - total_minutes)
            .when(total_minutes < 360)
            .then(360 - total_minutes)
            .otherwise(0)
            .cast(pl.Float32)
            .alias("minutes_to_close")
        )

        return df.with_columns(
            [
                ((total_minutes >= 525) & (total_minutes < 915))
                .cast(pl.Float32)
                .alias("is_day_session"),
                ((total_minutes >= 990) | (total_minutes < 360))
                .cast(pl.Float32)
                .alias("is_night_session"),
                ((total_minutes >= 990) & (total_minutes < 1050))
                .cast(pl.Float32)
                .alias("is_night_open"),
                ((total_minutes >= 1380) | (total_minutes < 360))
                .cast(pl.Float32)
                .alias("is_night_late"),
            ]
        )

    def _compute_vwap_and_skew(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        15分、4時間、日次のVWAPおよび出来高の偏り(Skew)を計算します。

        Args:
            df (pl.DataFrame): 対象データフレーム

        Returns:
            pl.DataFrame: VWAP関連指標が追加されたデータフレーム
        """
        pv = pl.col("close") * pl.col("vol_total_1bar")
        v = pl.col("vol_total_1bar")

        vwap_1d = pv.sum().over(pl.col("timestamp").dt.date()) / (
            v.sum().over(pl.col("timestamp").dt.date()) + 1e-8
        )
        vwap_1m = pv.rolling_sum(15) / (v.rolling_sum(15) + 1e-8)
        vwap_4h = pv.rolling_sum(240) / (v.rolling_sum(240) + 1e-8)

        df = df.with_columns(
            [
                ((pl.col("close") - vwap_1m) / (pl.col("close") + 1e-8))
                .fill_null(0)
                .alias("dist_vwap_1m"),
                ((pl.col("close") - vwap_4h) / (pl.col("close") + 1e-8))
                .fill_null(0)
                .alias("dist_vwap_4h"),
                ((pl.col("close") - vwap_1d) / (pl.col("close") + 1e-8))
                .fill_null(0)
                .alias("dist_vwap_1d"),
            ]
        )

        # VWAP Skew
        vwap_1h_calc = pv.rolling_sum(60) / (v.rolling_sum(60) + 1e-8)
        vol_above_vwap_1h = (
            v * (pl.col("close") > vwap_1h_calc).cast(pl.Float32)
        ).rolling_sum(60)
        vol_skew_1h = (vol_above_vwap_1h / (v.rolling_sum(60) + 1e-8)).fill_null(0.5)

        vol_above_vwap_1d = (
            (v * (pl.col("close") > vwap_1d).cast(pl.Float32))
            .sum()
            .over(pl.col("timestamp").dt.date())
        )
        vol_skew_1d = (
            vol_above_vwap_1d / (v.sum().over(pl.col("timestamp").dt.date()) + 1e-8)
        ).fill_null(0.5)

        return df.with_columns(
            [vol_skew_1h.alias("vol_skew_1h"), vol_skew_1d.alias("vol_skew_1d")]
        )

    def _compute_beta_and_distance(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        対S&P500のローリング・ベータと、移動平均からの乖離を計算します。

        Args:
            df (pl.DataFrame): 対象データフレーム

        Returns:
            pl.DataFrame: ベータ値と移動平均乖離率が追加されたデータフレーム
        """
        # Beta calculation
        spx_ret = pl.col("sp500_ret_lag1").rolling_mean(60)
        nk_ret = pl.col("log_ret").rolling_mean(60)
        df = df.with_columns((nk_ret - spx_ret).fill_null(0).alias("rs_sp500_1h"))

        spx_var = pl.col("sp500_ret_lag1").rolling_var(60) + 1e-8
        cov_nk_spx = (pl.col("log_ret") * pl.col("sp500_ret_lag1")).rolling_mean(60) - (
            nk_ret * spx_ret
        )
        beta_sp500_1h = (cov_nk_spx / spx_var).fill_null(0).clip(-5.0, 5.0)

        df = df.with_columns(beta_sp500_1h.alias("beta_sp500_1h"))

        # Moving Average Distance
        p = pl.col("close")
        return df.with_columns(
            [
                ((p - p.rolling_mean(15)) / (p + 1e-8)).fill_null(0).alias("dist_15m"),
                ((p - p.rolling_mean(60)) / (p + 1e-8)).fill_null(0).alias("dist_1h"),
                ((p - p.rolling_mean(240)) / (p + 1e-8)).fill_null(0).alias("dist_4h"),
                ((p - p.rolling_mean(1440)) / (p + 1e-8)).fill_null(0).alias("dist_1d"),
                (p.rolling_std(60) / (p.rolling_std(1440) + 1e-8))
                .fill_null(1.0)
                .alias("vol_regime"),
            ]
        )

    def _compute_cyclic_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        時刻や曜日などの周期的な時間特徴量と、イベントフラグを計算します。

        Args:
            df (pl.DataFrame): 対象データフレーム

        Returns:
            pl.DataFrame: 時間周期特徴量とフラグが追加されたデータフレーム
        """
        pi = np.pi
        df = df.with_columns(
            [
                (2 * pi * pl.col("timestamp").dt.weekday() / 7)
                .sin()
                .alias("day_of_week_sin"),
                (2 * pi * pl.col("timestamp").dt.weekday() / 7)
                .cos()
                .alias("day_of_week_cos"),
                (2 * pi * pl.col("timestamp").dt.hour() / 24)
                .sin()
                .alias("hour_of_day_sin"),
                (2 * pi * pl.col("timestamp").dt.hour() / 24)
                .cos()
                .alias("hour_of_day_cos"),
                # Session Flags
                (
                    (pl.col("timestamp").dt.hour() == 9)
                    & (pl.col("timestamp").dt.minute() < 30)
                )
                .cast(pl.Float32)
                .alias("is_market_open"),
                (
                    (pl.col("timestamp").dt.hour() == 11)
                    & (pl.col("timestamp").dt.minute() >= 30)
                    | (pl.col("timestamp").dt.hour() == 12)
                    & (pl.col("timestamp").dt.minute() < 30)
                )
                .cast(pl.Float32)
                .alias("is_lunch_break"),
                (
                    (pl.col("timestamp").dt.hour() == 14)
                    & (pl.col("timestamp").dt.minute() >= 30)
                    | (pl.col("timestamp").dt.hour() == 15)
                )
                .cast(pl.Float32)
                .alias("is_closing_auction"),
            ]
        )

        total_minutes = df["timestamp"].dt.hour().cast(pl.Int32) * 60 + df[
            "timestamp"
        ].dt.minute().cast(pl.Int32)
        minute_of_day_f = total_minutes.cast(pl.Float64)

        return df.with_columns(
            [
                (2 * math.pi * minute_of_day_f / 1440).sin().alias("minute_of_day_sin"),
                (2 * math.pi * minute_of_day_f / 1440).cos().alias("minute_of_day_cos"),
                (2 * math.pi * minute_of_day_f / 1440).sin().alias("tod_sin"),
                (2 * math.pi * minute_of_day_f / 1440).cos().alias("tod_cos"),
                pl.when((total_minutes >= 525) & (total_minutes < 915))
                .then(total_minutes - 525)
                .when(total_minutes >= 990)
                .then(total_minutes - 990)
                .when(total_minutes < 360)
                .then(total_minutes + 450)
                .otherwise(0)
                .cast(pl.Float32)
                .alias("minutes_from_open"),
                (
                    ((total_minutes >= 525) & (total_minutes <= 555))
                    | ((total_minutes >= 1290) & (total_minutes <= 1410))
                )
                .cast(pl.Float32)
                .alias("is_high_vol_window"),
            ]
        )
