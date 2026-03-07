# features/volume_profile.py
"""
File: features/volume_profile.py

ソースコードの役割:
本モジュールは、価格帯別出来高（Volume Profile）やPOC（Point of Control）に
関連する特徴量の計算機能を提供します。
"""

import polars as pl


def compute_volume_profile(df: pl.DataFrame) -> pl.DataFrame:
    """
    価格帯別出来高（Volume Profile）やPOC（Point of Control）に関する特徴量を計算します。

    Args:
        df (pl.DataFrame): 処理対象のデータフレーム。

    Returns:
        pl.DataFrame: Volume Profile特徴量が追加されたデータフレーム。
    """
    if "close" not in df.columns or "vol_total_1bar" not in df.columns:
        return df

    # 10円刻みの価格ビンを作成
    temp_df = df.select(
        [
            pl.col("timestamp"),
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("close"),
            pl.col("vol_total_1bar"),
            ((pl.col("close") / 10).round().cast(pl.Int32) * 10).alias("price_bin"),
        ]
    )

    # 日ごとの価格帯別出来高
    daily_profile = temp_df.group_by(["date", "price_bin"]).agg(
        pl.col("vol_total_1bar").sum().alias("bin_vol")
    )

    # 日次POC (その日で最大の出来高を持つ価格帯)
    daily_poc = (
        daily_profile.sort(["date", "bin_vol"], descending=[False, True])
        .group_by("date")
        .head(1)
        .select(pl.col("date"), pl.col("price_bin").alias("daily_poc_raw"))
    )

    # 過去1週間(5営業日)のPOCを計算して翌日にシフト適用
    unique_dates = temp_df.select("date").unique().sort("date")["date"].to_list()
    poc_records = []
    for i in range(len(unique_dates)):
        curr_date = unique_dates[i]
        next_date = unique_dates[i + 1] if i + 1 < len(unique_dates) else None

        prev_poc_val = daily_poc.filter(pl.col("date") == curr_date)["daily_poc_raw"]
        prev_poc_val = prev_poc_val[0] if len(prev_poc_val) > 0 else None

        start_idx = max(0, i - 4)
        window_dates = unique_dates[start_idx : i + 1]
        window_profile = daily_profile.filter(pl.col("date").is_in(window_dates))

        if len(window_profile) > 0:
            weekly_poc_val = (
                window_profile.group_by("price_bin")
                .agg(pl.col("bin_vol").sum())
                .sort("bin_vol", descending=True)["price_bin"][0]
            )
        else:
            weekly_poc_val = prev_poc_val

        if next_date is not None:
            poc_records.append(
                {
                    "date": next_date,
                    "prev_daily_poc": float(prev_poc_val) if prev_poc_val else None,
                    "prev_weekly_poc": (
                        float(weekly_poc_val) if weekly_poc_val else None
                    ),
                }
            )

    if poc_records:
        poc_df = pl.DataFrame(poc_records)
        df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))
        df = df.join(poc_df, on="date", how="left")
        df = df.with_columns(
            [
                pl.col("prev_daily_poc").forward_fill().fill_null(pl.col("close")),
                pl.col("prev_weekly_poc").forward_fill().fill_null(pl.col("close")),
            ]
        )
        df = df.with_columns(
            [
                (
                    (pl.col("close") - pl.col("prev_daily_poc"))
                    / (pl.col("close") + 1e-8)
                ).alias("dist_prev_poc_1d"),
                (
                    (pl.col("close") - pl.col("prev_weekly_poc"))
                    / (pl.col("close") + 1e-8)
                ).alias("dist_prev_poc_1w"),
            ]
        )
        df = df.drop(["date", "prev_daily_poc", "prev_weekly_poc"])
    else:
        df = df.with_columns(
            [
                pl.lit(0.0).alias("dist_prev_poc_1d"),
                pl.lit(0.0).alias("dist_prev_poc_1w"),
            ]
        )

    return df


def compute_vwap_and_skew(df: pl.DataFrame) -> pl.DataFrame:
    """
    15分、4時間、日次のVWAPおよび出来高の偏り(Skew)を計算します。

    Args:
        df (pl.DataFrame): 対象データフレーム

    Returns:
        pl.DataFrame: VWAPおよびSkew指標が追加されたデータフレーム
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
