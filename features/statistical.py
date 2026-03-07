# features/statistical.py
"""
File: features/statistical.py

ソースコードの役割:
本モジュールは、外部指標（S&P500など）との相関・ベータ値や、移動平均からの乖離率など、
統計的裁定や相対強度の観点からの特徴量エンジニアリングを提供します。
"""

import polars as pl


def compute_statistical_features(df: pl.DataFrame) -> pl.DataFrame:
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

