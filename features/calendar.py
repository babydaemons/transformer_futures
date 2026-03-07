# features/calendar.py
"""
File: features/calendar.py

ソースコードの役割:
本モジュールは、Polars DataFrameに対して時刻、曜日、セッション情報（日中/夜間フラグ）などの
カレンダーおよび時間に関連する周期的な特徴量エンジニアリングを提供します。
"""

import polars as pl
import numpy as np
import math


def compute_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    データフレームに対してカレンダー特徴量とセッションフラグを追加します。

    Args:
        df (pl.DataFrame): 入力となるデータフレーム（`timestamp` カラムが必須）

    Returns:
        pl.DataFrame: カレンダー特徴量とセッションフラグが追加されたデータフレーム
    """
    pi = np.pi

    # 時刻の分換算（0〜1440）
    total_minutes = df["timestamp"].dt.hour().cast(pl.Int32) * 60 + df[
        "timestamp"
    ].dt.minute().cast(pl.Int32)
    minute_of_day_f = total_minutes.cast(pl.Float64)

    # 1. セッションフラグと引けまでの時間
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

    df = df.with_columns(
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

    # 2. 周期的な時間特徴量と特定イベントフラグ
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

    return df
