# features/calendar.py
"""
File: features/calendar.py

ソースコードの役割:
本モジュールは、タイムスタンプからカレンダー情報（曜日、時間帯など）や
セッション情報（日中、夜間、昼休みなど）を抽出・計算し、Polarsの遅延評価(Lazy API)を用いて
モデルが時間的文脈を理解するための特徴量を生成します。
"""

import polars as pl
import numpy as np
from typing import Any


class CalendarFeature:
    """
    カレンダー情報（曜日、時間、セッション状態など）の特徴量を計算するクラス。
    pipeline.py から呼び出される統一インターフェース (compute メソッド) を提供します。
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg

    def compute(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        カレンダー特徴量の計算を順次実行します。

        Args:
            df (pl.LazyFrame): 計算対象のデータフレーム

        Returns:
            pl.LazyFrame: 特徴量が追加されたデータフレーム
        """
        df = self._compute_calendar_features(df)
        df = self._compute_session_features(df)
        return df

    def _compute_calendar_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        タイムスタンプから曜日や時間のサイン/コサイン特徴量を計算します。
        （周期的な時間情報を連続値としてモデルに与えるため）
        """
        # Pydanticで属性参照の安全性が保障されているため直接参照に変更
        ts_col = self.cfg.features.ts_col
        schema_names = df.collect_schema().names()

        if ts_col not in schema_names:
            if "timestamp" in schema_names:
                ts_col = "timestamp"
            else:
                return df

        # 曜日 (月曜=1, 日曜=7)
        day_of_week = pl.col(ts_col).dt.weekday()
        # 時間 (0-23)
        hour = pl.col(ts_col).dt.hour()
        # 分 (0-59)
        minute = pl.col(ts_col).dt.minute()

        # 1日のうちの分数 (0-1439)
        minute_of_day = hour * 60 + minute

        return df.with_columns(
            [
                # 曜日の周期特徴量
                (np.sin(2 * np.pi * day_of_week / 7)).alias("day_of_week_sin"),
                (np.cos(2 * np.pi * day_of_week / 7)).alias("day_of_week_cos"),
                # 時間の周期特徴量
                (np.sin(2 * np.pi * hour / 24)).alias("hour_of_day_sin"),
                (np.cos(2 * np.pi * hour / 24)).alias("hour_of_day_cos"),
                # Time of Day (1日の周期)
                (np.sin(2 * np.pi * minute_of_day / 1440)).alias("tod_sin"),
                (np.cos(2 * np.pi * minute_of_day / 1440)).alias("tod_cos"),
            ]
        )

    def _compute_session_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        日本の取引時間に基づくセッション情報（フラグ）を計算します。
        """
        # Pydanticで属性参照の安全性が保障されているため直接参照に変更
        ts_col = self.cfg.features.ts_col
        schema_names = df.collect_schema().names()

        if ts_col not in schema_names:
            if "timestamp" in schema_names:
                ts_col = "timestamp"
            else:
                return df

        hour = pl.col(ts_col).dt.hour()
        minute = pl.col(ts_col).dt.minute()
        time_hm = hour * 100 + minute  # HHMM形式 (例: 8時45分 -> 845)
        minute_of_day = hour * 60 + minute

        # セッション終了までの残り分数
        # 現行システムのセッション定義に合わせて、
        # 日中: 08:45 - 15:15
        # 夜間: 16:30 - 翌 06:00
        # を採用する。
        #
        # BAR_SECONDS=60 の現行構成では「残り分数」と「残りバー数」が一致するため、
        # バックテスト側の time_to_closes としてそのまま利用できる。
        minutes_to_close = (
            pl.when((time_hm >= 845) & (time_hm < 1515))
            .then((15 * 60 + 15) - minute_of_day)
            .when(time_hm < 600)
            .then((6 * 60) - minute_of_day)
            .when(time_hm >= 1630)
            .then(((24 * 60) - minute_of_day) + (6 * 60))
            .otherwise(0)
            .cast(pl.Float32)
            .alias("minutes_to_close")
        )

        return df.with_columns(
            [
                minutes_to_close,
                # 日中セッション (08:45 - 15:15)
                pl.when((time_hm >= 845) & (time_hm < 1515))
                .then(1.0)
                .otherwise(0.0)
                .alias("is_day_session"),
                # 夜間セッション (16:30 - 翌06:00)
                pl.when((time_hm >= 1630) | (time_hm < 600))
                .then(1.0)
                .otherwise(0.0)
                .alias("is_night_session"),
                # 夜間オープン付近 (16:30 - 17:30)
                pl.when((time_hm >= 1630) & (time_hm < 1730))
                .then(1.0)
                .otherwise(0.0)
                .alias("is_night_open"),
                # 夜間遅く (00:00 - 06:00)
                pl.when(time_hm < 600).then(1.0).otherwise(0.0).alias("is_night_late"),
                # 昼休み (11:30 - 12:30) 現物市場の昼休み
                pl.when((time_hm >= 1130) & (time_hm < 1230))
                .then(1.0)
                .otherwise(0.0)
                .alias("is_lunch_break"),
                # オープン直後の高ボラティリティ時間 (08:45 - 09:30)
                pl.when((time_hm >= 845) & (time_hm < 930))
                .then(1.0)
                .otherwise(0.0)
                .alias("is_high_vol_window"),
                # クロージングオークション付近 (15:00 - 15:15)
                pl.when((time_hm >= 1500) & (time_hm < 1515))
                .then(1.0)
                .otherwise(0.0)
                .alias("is_closing_auction"),
                # オープンからの経過分数 (08:45を起点とする、日中セッションのみ計算)
                pl.when((time_hm >= 845) & (time_hm < 1515))
                .then((hour - 8) * 60 + minute - 45)
                .otherwise(0.0)
                .alias("minutes_from_open"),
                # マーケットオープンフラグ (日中または夜間セッション中)
                pl.when(
                    ((time_hm >= 845) & (time_hm < 1515))
                    | ((time_hm >= 1630) | (time_hm < 600))
                )
                .then(1.0)
                .otherwise(0.0)
                .alias("is_market_open"),
            ]
        )
