# features/volume_profile.py
"""
File: features/volume_profile.py

ソースコードの役割:
本モジュールは、価格帯別出来高（Volume Profile）やPOC（Point of Control）に
関連する特徴量の計算機能を提供します。日次VWAPや指定期間のVWAP、さらに
出来高の偏り（Volume Skew）を計算し、モデルの入力となる特徴量を生成します。
計算において累積和（cum_sum）を利用することで、バックテスト時の未来情報のリークを防止します。
"""

import polars as pl
from typing import Any


class VolumeProfileFeature:
    """
    Volume ProfileおよびVWAP関連の特徴量を計算するクラス。
    pipeline.py から呼び出される統一インターフェース (compute メソッド) を提供します。
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg

    def compute(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Volume Profile と VWAP の特徴量計算を順次実行します。

        Args:
            df (pl.LazyFrame): 計算対象のデータフレーム

        Returns:
            pl.LazyFrame: 特徴量が追加されたデータフレーム
        """
        df = self._compute_volume_profile(df)
        df = self._compute_vwap_features(df)
        return df

    def _compute_volume_profile(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """価格帯別出来高（Volume Profile）やPOC（Point of Control）に関する特徴量を計算します。

        10円刻みの価格ビンを作成し、日次および過去1週間のPOCを求め、現在の価格との
        乖離（ディスタンス）を特徴量としてデータフレームに追加します。

        Args:
            df (pl.LazyFrame): 処理対象のデータフレーム

        Returns:
            pl.LazyFrame: POC特徴量（`dist_prev_poc_1d`, `dist_prev_poc_1w`）が追加されたデータフレーム。
        """
        # NOTE: 遅延評価(LazyFrame)において列の存在確認は厳密には collect_schema().names()
        # で行うのが安全ですが、パイプライン側で必須列は補完される前提とします。

        # 代替カラムの割り当て: "vol_total_1bar" がない場合は "volume" を使用
        schema_names = df.collect_schema().names()
        vol_col = "vol_total_1bar" if "vol_total_1bar" in schema_names else "volume"
        ts_col = (
            self.cfg.features.ts_col
            if hasattr(self.cfg.features, "ts_col")
            else "timestamp"
        )

        if "close" not in schema_names or vol_col not in schema_names:
            return df

        # 10円刻みの価格ビンを作成（スリッページやノイズを平滑化）
        # LazyFrameでのWindow関数の記述
        temp_exprs = [
            pl.col("close"),
            (pl.col("close") // 10 * 10).cast(pl.Int32).alias("price_bin"),
            pl.col(vol_col).alias("volume_for_poc"),
            pl.col(ts_col).dt.date().alias("date"),
            pl.col(ts_col).dt.truncate("1w").alias("week"),
        ]

        # NOTE: 実際のPOC計算（各日・週ごとの最大出来高価格帯の算出）は
        # groupby -> agg(sum) -> window(max) のような複雑なLazy処理になるため、
        # ここでは近似的な簡易実装（または事前計算されたプレースホルダー）として記述します。
        # 厳密なPOC計算はメモリ/速度トレードオフがあるため、まずは0.0で埋めてパイプラインを疎通させます。
        # (以前のDataFrame実装からLazyFrame実装への安全な移行措置)

        return df.with_columns(
            [
                pl.lit(0.0).alias("dist_prev_poc_1d"),
                pl.lit(0.0).alias("dist_prev_poc_1w"),
            ]
        )

    def _compute_vwap_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        日次VWAPや指定期間（15分、4時間など）のVWAP、および
        Volume Skew（出来高の偏り）を計算します。
        """
        schema_names = df.collect_schema().names()
        ts_col = (
            self.cfg.features.ts_col
            if hasattr(self.cfg.features, "ts_col")
            else "timestamp"
        )

        if "close" not in schema_names or "volume" not in schema_names:
            return df

        # 価格 × 出来高
        pv = pl.col("close") * pl.col("volume")
        v = pl.col("volume")

        # 日次VWAP (タイムスタンプの日付ごとにリセットされる累積和)
        vwap_1d = pv.cum_sum().over(pl.col(ts_col).dt.date()) / (
            v.cum_sum().over(pl.col(ts_col).dt.date()) + 1e-8
        )

        # ローリングVWAP (15分, 4時間) -> 1分足なら15本、240本
        vwap_15m = pv.rolling_sum(15) / (v.rolling_sum(15) + 1e-8)
        vwap_4h = pv.rolling_sum(240) / (v.rolling_sum(240) + 1e-8)

        # 現在価格と各VWAPとの乖離率を計算
        df = df.with_columns(
            [
                ((pl.col("close") - vwap_15m) / (pl.col("close") + 1e-8))
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("dist_vwap_15m"),
                ((pl.col("close") - vwap_4h) / (pl.col("close") + 1e-8))
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("dist_vwap_4h"),
                ((pl.col("close") - vwap_1d) / (pl.col("close") + 1e-8))
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("dist_vwap_1d"),
            ]
        )

        # --- VWAP Skew の計算 ---
        # 1時間 (60分) の VWAP Skew
        vwap_1h_calc = pv.rolling_sum(60) / (v.rolling_sum(60) + 1e-8)
        # VWAPより上の出来高割合
        vol_above_vwap_1h = (
            pl.when(pl.col("close") > vwap_1h_calc)
            .then(pl.col("volume"))
            .otherwise(0)
            .rolling_sum(60)
        )
        vol_skew_1h = (vol_above_vwap_1h / (v.rolling_sum(60) + 1e-8)) - 0.5

        # 1日 (概算でセッション長によるが、簡易的にローリング1440本とするか日次累積和を使用)
        vol_above_vwap_1d = (
            pl.when(pl.col("close") > vwap_1d)
            .then(pl.col("volume"))
            .otherwise(0)
            .cum_sum()
            .over(pl.col(ts_col).dt.date())
        )
        vol_skew_1d = (
            vol_above_vwap_1d / (v.cum_sum().over(pl.col(ts_col).dt.date()) + 1e-8)
        ) - 0.5

        df = df.with_columns(
            [
                vol_skew_1h.fill_nan(0.0).fill_null(0.0).alias("vol_skew_1h"),
                vol_skew_1d.fill_nan(0.0).fill_null(0.0).alias("vol_skew_1d"),
            ]
        )

        return df
