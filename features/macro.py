# features/macro.py
"""
File: features/macro.py

ソースコードの役割:
本モジュールは、外部資産（USD/JPY、S&P500など）のLag結合やLead-Lagスプレッド、
Bollinger Bandスコアなどのマクロ経済関連の特徴量計算を提供します。
"""

import polars as pl


def compute_macro_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    外部資産（USD/JPY、S&P500）のLag等を用いたマクロ関連の特徴量を計算します。

    Args:
        df (pl.DataFrame): メインの価格データと結合済みのマクロデータを含むデータフレーム。

    Returns:
        pl.DataFrame: マクロ特徴量が追加されたデータフレーム。
    """
    # 1. まずLag特徴量を作成 (第1段階)
    lag_cols = [
        (
            pl.col(f"{asset}_close")
            .pct_change()
            .fill_null(0)
            .shift(1)
            .fill_null(0)
            .alias(f"{asset}_ret_lag1")
            if f"{asset}_close" in df.columns
            else pl.lit(0.0).alias(f"{asset}_ret_lag1")
        )
        for asset in ["usdjpy", "sp500"]
    ]

    df = df.with_columns(lag_cols)

    # 2. 確定したLag特徴量を使って派生特徴量を作成 (第2段階)
    deriv_cols = []
    if "usdjpy_close" in df.columns:
        deriv_cols.append(
            pl.rolling_corr(pl.col("close"), pl.col("usdjpy_close"), window_size=60)
            .fill_nan(0)
            .fill_null(0)
            .alias("corr_usdjpy")
        )
        # --- Lead-Lag Spread ---
        deriv_cols.append(
            (pl.col("log_ret") - pl.col("usdjpy_ret_lag1"))
            .fill_null(0)
            .alias("usdjpy_lead_spread")
        )
        # マクロ乖離の累積 (60分)
        deriv_cols.append(
            (pl.col("log_ret") - pl.col("usdjpy_ret_lag1"))
            .rolling_sum(60)
            .fill_null(0)
            .alias("usdjpy_cum_divergence_1h")
        )

        # USDJPY Bollinger Band Score (Z-score)
        u_mean = pl.col("usdjpy_close").rolling_mean(20)
        u_std = pl.col("usdjpy_close").rolling_std(20) + 1e-5
        deriv_cols.append(
            ((pl.col("usdjpy_close") - u_mean) / u_std)
            .fill_null(0)
            .alias("usdjpy_bb_score")
        )

    else:
        deriv_cols.append(pl.lit(0.0).alias("corr_usdjpy"))
        deriv_cols.append(pl.lit(0.0).alias("usdjpy_lead_spread"))
        deriv_cols.append(pl.lit(0.0).alias("usdjpy_cum_divergence_1h"))
        deriv_cols.append(pl.lit(0.0).alias("usdjpy_bb_score"))

    return df.with_columns(deriv_cols)
