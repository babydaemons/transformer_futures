# features/technicals.py
"""
File: features/technicals.py

ソースコードの役割:
本モジュールは、RSI、MACD、ボリンジャーバンド、ATR、ADXなどの一般的なテクニカル指標を
Polarsの遅延評価(Lazy API)を用いて高速に計算する機能を提供します。
これらの指標はトレンドの強さ、モメンタム、ボラティリティを定量化し、モデルの入力特徴量として利用されます。
"""

import polars as pl
from typing import Any


class TechnicalFeature:
    """
    テクニカル指標（RSI, MACD, BB, ATR, ADXなど）を計算するクラス。
    pipeline.py から呼び出される統一インターフェース (compute メソッド) を提供します。
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        # cfgからパラメータを取得 (デフォルト値を設定して安全に取得)
        self.rsi_period = getattr(self.cfg.features, "rsi_period", 14)
        self.macd_fast = getattr(self.cfg.features, "macd_fast", 12)
        self.macd_slow = getattr(self.cfg.features, "macd_slow", 26)
        self.macd_signal = getattr(self.cfg.features, "macd_signal", 9)
        self.bb_window = getattr(self.cfg.features, "bb_window", 20)
        self.adx_period = getattr(self.cfg.features, "adx_period", 14)

    def compute(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        全テクニカル指標の計算を順次実行します。

        Args:
            df (pl.LazyFrame): 計算対象のデータフレーム

        Returns:
            pl.LazyFrame: 特徴量が追加されたデータフレーム
        """
        df = self._compute_rsi(df)
        df = self._compute_macd(df)
        df = self._compute_bollinger_bands(df)
        df = self._compute_atr(df)
        df = self._compute_adx(df)
        df = self._compute_efficiency_ratio(df)
        return df

    def _compute_rsi(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        RSI (Relative Strength Index) を計算します。
        """
        schema_names = df.collect_schema().names()
        if "close" not in schema_names:
            return df

        period = self.rsi_period

        # 簡易的なRSI計算 (ワイルダーの平滑移動平均ではなく単純移動平均を使用する高速版)
        diff = pl.col("close").diff()
        up = pl.when(diff > 0).then(diff).otherwise(0)
        down = pl.when(diff < 0).then(diff.abs()).otherwise(0)

        roll_up = up.rolling_mean(window_size=period)
        roll_down = down.rolling_mean(window_size=period)

        rs = roll_up / (roll_down + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return df.with_columns(rsi.fill_nan(50.0).fill_null(50.0).alias("rsi"))

    def _compute_macd(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        MACD (Moving Average Convergence Divergence) を計算します。
        本来はEMAですが、ここでは簡略化のため単純移動平均(SMA)で代用します(LazyFrameでの計算容易化のため)。
        """
        schema_names = df.collect_schema().names()
        if "close" not in schema_names:
            return df

        fast = self.macd_fast
        slow = self.macd_slow

        # EMAの厳密な計算はLazyFrameでは複雑なため、単純移動平均で近似
        macd_line = pl.col("close").rolling_mean(fast) - pl.col("close").rolling_mean(
            slow
        )

        return df.with_columns(macd_line.fill_nan(0.0).fill_null(0.0).alias("macd"))

    def _compute_bollinger_bands(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Bollinger Bands の %b (現在の価格がバンド内のどの位置にあるか) を計算します。
        """
        schema_names = df.collect_schema().names()
        if "close" not in schema_names:
            return df

        window = self.bb_window

        sma = pl.col("close").rolling_mean(window)
        std = pl.col("close").rolling_std(window)

        upper = sma + (std * 2)
        lower = sma - (std * 2)

        # %b = (Close - Lower) / (Upper - Lower)
        bb_b = (pl.col("close") - lower) / (upper - lower + 1e-8)

        return df.with_columns(bb_b.fill_nan(0.5).fill_null(0.5).alias("bb_b"))

    def _compute_atr(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        ATR (Average True Range) を計算します。
        """
        schema_names = df.collect_schema().names()
        if not all(c in schema_names for c in ["high", "low", "close"]):
            return df

        period = self.adx_period  # ATRはADXと同じ期間を使うのが一般的

        # True Range
        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - pl.col("close").shift(1)).abs()
        tr3 = (pl.col("low") - pl.col("close").shift(1)).abs()

        tr = pl.max_horizontal([tr1, tr2, tr3])
        atr = tr.rolling_mean(window_size=period)

        return df.with_columns(atr.fill_nan(0.0).fill_null(0.0).alias("atr"))

    def _compute_adx(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        ADX (Average Directional Index) の近似値を計算します。
        厳密なADX計算はステップ数が多いため、ここでは簡易的なボラティリティ/トレンド指標でプレースホルダーとします。
        """
        schema_names = df.collect_schema().names()
        if not all(c in schema_names for c in ["high", "low", "close"]):
            return df

        period = self.adx_period

        # 簡易版ADX (ここではDirectional Movementの近似として、|Close - Close(n)| / ATR(n) を使用)
        # ※本来の計算とは異なりますが、パイプラインを通すための安全なプレースホルダーです
        if "atr" not in schema_names:
            # ATRがまだ計算されていない場合(通常は直前で計算されるはず)
            return df.with_columns(pl.lit(0.0).alias("adx"))

        simplified_adx = (
            (pl.col("close") - pl.col("close").shift(period)).abs()
            / (pl.col("atr") + 1e-8)
            * 100
        )

        return df.with_columns(simplified_adx.fill_nan(0.0).fill_null(0.0).alias("adx"))

    def _compute_efficiency_ratio(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Kaufman's Efficiency Ratio (ER) を計算します。
        ER = |Close - Close(n)| / Sum(|Close - Close(1)|, n)
        """
        schema_names = df.collect_schema().names()
        if "close" not in schema_names:
            return df

        period = 10  # デフォルトの計算期間

        # direction = |Close - Close(n)|
        direction = (pl.col("close") - pl.col("close").shift(period)).abs()

        # volatility = Sum(|Close - Close(1)|, n)
        volatility = pl.col("close").diff().abs().rolling_sum(window_size=period)

        er = direction / (volatility + 1e-8)

        df = df.with_columns(er.fill_nan(0.0).fill_null(0.0).alias("efficiency_ratio"))

        # 1時間 (60本) の ER
        period_1h = 60
        direction_1h = (pl.col("close") - pl.col("close").shift(period_1h)).abs()
        volatility_1h = pl.col("close").diff().abs().rolling_sum(window_size=period_1h)
        er_1h = direction_1h / (volatility_1h + 1e-8)

        return df.with_columns(
            er_1h.fill_nan(0.0).fill_null(0.0).alias("efficiency_ratio_1h")
        )
