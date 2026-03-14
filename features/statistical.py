# features/statistical.py
"""
File: features/statistical.py

ソースコードの役割:
本モジュールは、過去の価格変動に基づく統計的特徴量（自己相関、ローリング尖度、ローリング歪度、
1日のボラティリティ・レジーム、ヒストリカル・ボラティリティの分位数など）をPolarsの遅延評価(Lazy API)を用いて計算します。
これにより、市場のランダム性やテールリスクの状態をモデルに学習させます。
"""

import polars as pl
from typing import Any


class StatisticalFeature:
    """
    統計的特徴量（Skewness, Kurtosis, Volatility Regimeなど）を計算するクラス。
    pipeline.py から呼び出される統一インターフェース (compute メソッド) を提供します。
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        # cfgからパラメータを取得 (デフォルト値を設定)
        self.window_1h = getattr(self.cfg.features, "stat_window_1h", 60)
        self.window_1d = getattr(
            self.cfg.features, "stat_window_1d", 1440
        )  # 簡易的な1日の分数

    def compute(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        全統計的特徴量の計算を順次実行します。

        Args:
            df (pl.LazyFrame): 計算対象のデータフレーム

        Returns:
            pl.LazyFrame: 特徴量が追加されたデータフレーム
        """
        df = self._compute_rolling_moments(df)
        df = self._compute_volatility_regime(df)
        return df

    def _compute_rolling_moments(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        ローリングベースの歪度 (Skewness) と尖度 (Kurtosis) を近似計算します。
        """
        schema_names = df.collect_schema().names()
        if "log_ret" not in schema_names:
            return df

        window = self.window_1h

        # 平均と標準偏差
        roll_mean = pl.col("log_ret").rolling_mean(window)
        roll_std = pl.col("log_ret").rolling_std(window)

        # 中心化されたリターン
        centered = pl.col("log_ret") - roll_mean

        # 歪度の近似 (3次モーメント / std^3)
        skew_approx = (centered**3).rolling_mean(window) / (roll_std**3 + 1e-8)

        # 尖度の近似 (4次モーメント / std^4) - 3 (超過尖度)
        kurt_approx = (centered**4).rolling_mean(window) / (roll_std**4 + 1e-8) - 3.0

        return df.with_columns(
            [
                skew_approx.fill_nan(0.0).fill_null(0.0).alias("log_ret_skew_1h"),
                kurt_approx.fill_nan(0.0).fill_null(0.0).alias("log_ret_kurt_1h"),
            ]
        )

    def _compute_volatility_regime(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        ボラティリティ・レジームを計算します。
        (現在のボラティリティが過去N本の中でどのパーセンタイルに位置するか等)
        """
        schema_names = df.collect_schema().names()
        # どのボラティリティ指標を使うか
        vol_col = "realized_vol"
        if vol_col not in schema_names:
            if "atr" in schema_names:
                vol_col = "atr"
            else:
                return df.with_columns(pl.lit(0.5).alias("vol_regime"))

        window = self.window_1d

        # 簡易的なレジーム判定: (現在のVol - Min(Vol)) / (Max(Vol) - Min(Vol))
        # (厳密な分位数計算はLazyFrameでは重いため、Min-Maxスケーリングで近似)
        roll_min = pl.col(vol_col).rolling_min(window)
        roll_max = pl.col(vol_col).rolling_max(window)

        regime = (pl.col(vol_col) - roll_min) / (roll_max - roll_min + 1e-8)

        return df.with_columns(regime.fill_nan(0.5).fill_null(0.5).alias("vol_regime"))
