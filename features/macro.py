# features/macro.py
"""
File: features/macro.py

ソースコードの役割:
本モジュールは、外部資産（USD/JPY、S&P500、XAU/USD、XTI/USDなど）のLag結合やLead-Lagスプレッド、
Bollinger Bandスコア、および需給データ（空売り比率、外国人投資家動向）の正規化などのマクロ経済関連の
特徴量計算を提供します。1分足のデイトレードにおいて、外部環境の短期的なトレンドと乖離を数値化します。
"""

import polars as pl
from typing import List


def _compute_asset_lags(df: pl.DataFrame) -> List[pl.Expr]:
    """
    各アセットのLag特徴量リストを生成します。

    Args:
        df (pl.DataFrame): 入力データフレーム。

    Returns:
        List[pl.Expr]: Lag特徴量のPolars式リスト。
    """
    return [
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


def _compute_usdjpy_derivatives(df: pl.DataFrame) -> List[pl.Expr]:
    """
    USD/JPY関連の相関やボリンジャーバンドスコアなどの派生特徴量リストを生成します。

    Args:
        df (pl.DataFrame): 入力データフレーム。

    Returns:
        List[pl.Expr]: USD/JPY派生特徴量のPolars式リスト。
    """
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

    return deriv_cols


def _compute_supply_demand_norms(df: pl.DataFrame) -> List[pl.Expr]:
    """
    空売り比率・外国人動向などの需給正規化特徴量リストを生成します。

    Args:
        df (pl.DataFrame): 入力データフレーム。

    Returns:
        List[pl.Expr]: 需給関連の正規化特徴量のPolars式リスト。
    """
    deriv_cols = []

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

    return deriv_cols


def compute_macro_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    外部資産や需給データのLag等を用いたマクロ関連の特徴量を計算します。

    Args:
        df (pl.DataFrame): メインの価格データと結合済みのマクロデータを含むデータフレーム。

    Returns:
        pl.DataFrame: マクロ・需給特徴量が追加されたデータフレーム。
    """
    # 1. 各アセットのLag特徴量を追加 (後続の派生指標の計算に必要)
    df = df.with_columns(_compute_asset_lags(df))

    # 2. Lagに依存する派生特徴量と、需給正規化特徴量をまとめて適用
    deriv_exprs = _compute_usdjpy_derivatives(df) + _compute_supply_demand_norms(df)

    return df.with_columns(deriv_exprs)
