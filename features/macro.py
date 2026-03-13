# features/macro.py
"""
File: features/macro.py

ソースコードの役割:
本モジュールは、外部資産（USD/JPY、S&P500、XAU/USD、XTI/USDなど）のLag結合やLead-Lagスプレッド、
Bollinger Bandスコア、および需給データ（空売り比率、外国人投資家動向）の正規化などのマクロ経済関連の
特徴量計算を提供します。1分足のデイトレードにおいて、外部環境の短期的なトレンドと乖離を数値化します。
"""

import polars as pl


def compute_macro_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    外部資産や需給データのLag等を用いたマクロ関連の特徴量を計算します。

    Args:
        df (pl.DataFrame): メインの価格データと結合済みのマクロデータを含むデータフレーム。

    Returns:
        pl.DataFrame: マクロ・需給特徴量が追加されたデータフレーム。
    """
    # 1. 各アセットのLag特徴量を作成 (USDJPY, S&P500, XAUUSD, XTIUSD)
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
        for asset in ["usdjpy", "sp500", "xauusd", "xtiusd"]
    ]

    df = df.with_columns(lag_cols)

    # 2. 確定したLag特徴量を使って派生特徴量を作成 (第2段階)
    deriv_cols = []

    # --- USD/JPY 関連指標 ---
    if "usdjpy_close" in df.columns:
        # 相関指標 (60分ウィンドウ)
        deriv_cols.append(
            pl.rolling_corr(pl.col("close"), pl.col("usdjpy_close"), window_size=60)
            .fill_nan(0)
            .fill_null(0)
            .alias("corr_usdjpy")
        )

        # Lead-Lag Spread: 自銘柄の対数リターンと外部資産のラグ付きリターンの差分
        deriv_cols.append(
            (pl.col("log_ret") - pl.col("usdjpy_ret_lag1"))
            .fill_null(0)
            .alias("usdjpy_lead_spread")
        )

        # マクロ乖離の累積 (60分合計)
        deriv_cols.append(
            (pl.col("log_ret") - pl.col("usdjpy_ret_lag1"))
            .rolling_sum(60)
            .fill_null(0)
            .alias("usdjpy_cum_divergence_1h")
        )

        # USDJPY Bollinger Band Score (Z-scoreによる割安・割高判定)
        u_mean = pl.col("usdjpy_close").rolling_mean(20)
        u_std = pl.col("usdjpy_close").rolling_std(20) + 1e-5
        deriv_cols.append(
            ((pl.col("usdjpy_close") - u_mean) / u_std)
            .fill_null(0)
            .alias("usdjpy_bb_score")
        )

    else:
        # 外部資産データが存在しない場合のフォールバック
        deriv_cols.append(pl.lit(0.0).alias("corr_usdjpy"))
        deriv_cols.append(pl.lit(0.0).alias("usdjpy_lead_spread"))
        deriv_cols.append(pl.lit(0.0).alias("usdjpy_cum_divergence_1h"))
        deriv_cols.append(pl.lit(0.0).alias("usdjpy_bb_score"))

    # --- 空売り比率の正規化 ---
    if "short_selling_ratio_raw" in df.columns:
        ssr = pl.col("short_selling_ratio_raw")
        # 過去約1週間の移動平均からの乖離をZスコアとして算出
        deriv_cols.append(
            (
                (ssr - ssr.rolling_mean(5 * 24 * 60))
                / (ssr.rolling_std(5 * 24 * 60) + 1e-8)
            )
            .fill_null(0)
            .alias("short_selling_ratio")
        )
    else:
        deriv_cols.append(pl.lit(0.0).alias("short_selling_ratio"))

    # --- 外国人投資家買い越し額の正規化 ---
    if "foreigners_balance_raw" in df.columns:
        fb = pl.col("foreigners_balance_raw")
        # 過去約1ヶ月の移動平均からの乖離をZスコアとして算出
        deriv_cols.append(
            (
                (fb - fb.rolling_mean(20 * 24 * 60))
                / (fb.rolling_std(20 * 24 * 60) + 1e-8)
            )
            .fill_null(0)
            .alias("foreigners_balance_norm")
        )
    else:
        deriv_cols.append(pl.lit(0.0).alias("foreigners_balance_norm"))

    return df.with_columns(deriv_cols)
