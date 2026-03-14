# features/pipeline.py
"""
File: features/pipeline.py

ソースコードの役割:
生データ(OHLCV)を入力とし、Polarsの遅延評価(Lazy API)を用いて
各種特徴量(Volume Profile, Technicals, Statistical, Macroなど)を
一括計算・結合するパイプラインクラスを提供します。
"""

import polars as pl
from config import GlobalConfig

from features.volume_profile import VolumeProfileFeature
from features.technicals import TechnicalFeature
from features.statistical import StatisticalFeature
from features.macro import MacroFeature
from features.calendar import CalendarFeature


class FeaturePipeline:
    """
    複数ドメインの特徴量を統合して計算するパイプライン。
    """

    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self.vp = VolumeProfileFeature(cfg)
        self.tech = TechnicalFeature(cfg)
        self.stat = StatisticalFeature(cfg)
        self.macro = MacroFeature(cfg)
        self.cal = CalendarFeature(cfg)

    def compute_features(self, df_raw: pl.LazyFrame) -> pl.LazyFrame:
        """
        全特徴量の計算パイプラインを実行します。
        """
        # 前処理として、必要なカラムが欠落している場合の補完を行う
        df = self._ensure_required_columns(df_raw)

        df = self._compute_price_volume_basics(df)

        df = self.vp.compute(df)
        df = self.tech.compute(df)
        df = self.stat.compute(df)
        df = self.cal.compute(df)

        # 外部データ/マクロ連携
        df = self.macro.compute(df)

        df = self._compute_cross_asset(df)

        # 不要な一時カラムを落とす
        df = (
            df.drop(["timestamp"]) if "timestamp" in df.collect_schema().names() else df
        )

        return df

    def _ensure_required_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        入力データに不足している必須カラム（tick_count等）があればデフォルト値で補完します。
        """
        schema = df.collect_schema()
        cols = schema.names()

        exprs = []
        # tick_count が存在しない場合、volume で代替するか、常に 1 を設定する
        if "tick_count" not in cols:
            if "volume" in cols:
                # 出来高をティック数の代替として扱う（より安全なフォールバック）
                exprs.append(pl.col("volume").alias("tick_count"))
            else:
                exprs.append(pl.lit(1).alias("tick_count"))

        # その他の必須カラムが欠落している場合の安全策
        if "buy_volume" not in cols and "volume" in cols:
            exprs.append((pl.col("volume") / 2).alias("buy_volume"))
        if "sell_volume" not in cols and "volume" in cols:
            exprs.append((pl.col("volume") / 2).alias("sell_volume"))

        if exprs:
            return df.with_columns(exprs)
        return df

    def _compute_price_volume_basics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        価格・出来高の基礎的な指標（リターン、対数出来高など）を計算します。
        """
        price_col = self.cfg.features.price_col
        # 生データが "close" を持っている場合、それを設定の price_col にリネームまたは複製する
        schema_names = df.collect_schema().names()

        # 設定された price_col (例: trade_price) が無い場合、"close" から作成する
        if price_col not in schema_names and "close" in schema_names:
            df = df.with_columns(pl.col("close").alias(price_col))

        return df.with_columns(
            # Log Returns
            pl.col(price_col).log().diff().alias("log_ret"),
            # Log Volume
            (pl.col("volume") + 1.0).log().alias("log_vol"),
            (pl.col("buy_volume") + 1.0).log().alias("log_buy_vol"),
            (pl.col("sell_volume") + 1.0).log().alias("log_sell_vol"),
            # Return / Volume (Sharpe-like momentum)
            (pl.col(price_col).log().diff() / ((pl.col("volume") + 1.0).log()))
            .fill_nan(0.0)
            .alias("ret_div_vol"),
            # Relative Volume (対10本移動平均)
            (pl.col("volume") / (pl.col("volume").rolling_mean(10) + 1e-8))
            .fill_nan(1.0)
            .alias("rel_vol"),
            # 需給圧力 (buy - sell) / (buy + sell)
            ((pl.col("buy_volume") - pl.col("sell_volume")) / (pl.col("volume") + 1e-8))
            .fill_nan(0.0)
            .alias("buy_pressure"),
            # Order Flow Imbalance (簡易版)
            (pl.col("buy_volume") - pl.col("sell_volume")).alias("ofi_signal"),
            # Volume Pressure
            (
                (pl.col("close") - pl.col("open"))
                / (pl.col("high") - pl.col("low") + 1e-8)
                * pl.col("volume")
            )
            .fill_nan(0.0)
            .alias("volume_pressure"),
            # Price Spread
            (pl.col("high") - pl.col("low")).alias("price_spread"),
            # Trade Frequency Acceleration (Tick Countの変化率)
            (pl.col("tick_count").pct_change()).fill_nan(0.0).alias("trade_freq_accel"),
            # Amihud Illiquidity ( |Return| / Volume )
            (pl.col(price_col).log().diff().abs() / (pl.col("volume") + 1e-8))
            .fill_nan(0.0)
            .alias("amihud_illiquidity"),
            # Realized Volatility (簡易版: 直近10本の標準偏差)
            pl.col(price_col)
            .log()
            .diff()
            .rolling_std(10)
            .fill_nan(0.0)
            .alias("realized_vol"),
            # Parkinson Volatility
            (((pl.col("high") / pl.col("low")).log() ** 2) / (4 * 0.693147))
            .rolling_mean(10)
            .sqrt()
            .fill_nan(0.0)
            .alias("parkinson_vol"),
            # Garman-Klass Volatility
            (
                0.5 * ((pl.col("high") / pl.col("low")).log() ** 2)
                - (2 * 0.693147 - 1) * ((pl.col("close") / pl.col("open")).log() ** 2)
            )
            .rolling_mean(10)
            .sqrt()
            .fill_nan(0.0)
            .alias("garman_klass_vol"),
            # Shadow Range (ヒゲの長さ)
            (
                (pl.col("high") - pl.max_horizontal(["open", "close"]))
                + (pl.min_horizontal(["open", "close"]) - pl.col("low"))
            ).alias("shadow_range"),
            # Size Imbalance 1bar
            (
                (pl.col("buy_volume") - pl.col("sell_volume"))
                / (pl.col("tick_count") + 1e-8)
            )
            .fill_nan(0.0)
            .alias("size_imb_1bar"),
            # Vol/Size Std
            (pl.col("volume") / (pl.col("tick_count") + 1e-8))
            .rolling_std(10)
            .fill_nan(0.0)
            .alias("vol_size_std"),
            # Max Trade Size (代替としてVolumeを使用)
            pl.col("volume").rolling_max(10).alias("max_trade_size"),
            # Tick Speed Ratio (直近10本の平均対比)
            (pl.col("tick_count") / (pl.col("tick_count").rolling_mean(10) + 1e-8))
            .fill_nan(1.0)
            .alias("tick_speed_ratio"),
            # Delta Volume
            (pl.col("volume").diff()).fill_nan(0.0).alias("delta_volume"),
        )

    def _compute_cross_asset(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        他資産(US500など)との相関・ベータなどを計算するプレースホルダー。
        外部データ結合処理(MacroFeature等)で取得したカラムを利用する想定。
        """
        # 実際には cfg.features のカラム名に合わせて計算する
        schema = df.collect_schema().names()
        exprs = []

        # 例: usdjpy_ret_lag1 等がまだなければダミーを入れる (エラー回避のため)
        dummy_cols = [
            "dist_15m",
            "dist_1h",
            "dist_4h",
            "dist_1d",
            "dist_vwap_1m",
            "dist_vwap_4h",
            "dist_vwap_1d",
            "rs_sp500_1h",
            "beta_sp500_1h",
            "vol_regime",
            "dist_prev_poc_1d",
            "dist_prev_poc_1w",
            "vol_skew_1h",
            "vol_skew_1d",
            "usdjpy_ret_lag1",
            "sp500_ret_lag1",
            "corr_usdjpy",
            "usdjpy_lead_spread",
            "usdjpy_bb_score",
            "usdjpy_cum_divergence_1h",
            "xauusd_ret_lag1",
            "xtiusd_ret_lag1",
            "short_selling_ratio",
            "foreigners_balance_norm",
        ]

        for col in dummy_cols:
            if col not in schema:
                exprs.append(pl.lit(0.0).alias(col))

        if exprs:
            df = df.with_columns(exprs)

        return df
