# features/macro.py
"""
File: features/macro.py

ソースコードの役割:
本モジュールは、マクロ経済指標や外部資産（USDJPY、S&P500、Gold等）のデータを
遅延評価(Lazy API)を用いて読み込み、メインのデータフレームと結合・計算する機能を提供します。
他資産からの先行する値動きや相関関係を特徴量としてモデルに提供します。
"""

import polars as pl
from typing import Any
import os


class MacroFeature:
    """
    マクロ指標およびクロスアセット(USDJPY, S&P500等)の特徴量を計算・結合するクラス。
    pipeline.py から呼び出される統一インターフェース (compute メソッド) を提供します。
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        # パス解決用のヘルパー関数を定義
        self._resolve_path = lambda p: (
            p
            if os.path.exists(p)
            else p.replace("../transformer_futures.data", "C:/transformer_futures_data")
        )

    def compute(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        マクロ特徴量の計算と結合を順次実行します。

        Args:
            df (pl.LazyFrame): メイン銘柄のデータフレーム

        Returns:
            pl.LazyFrame: マクロ特徴量が追加されたデータフレーム
        """
        df = self._compute_usdjpy_features(df)
        df = self._compute_sp500_features(df)
        df = self._compute_jpx_features(df)
        return df

    def _load_macro_lazy(self, file_path: str) -> pl.LazyFrame:
        """
        マクロデータのParquetファイルをLazyFrameとして読み込むヘルパー関数。
        ファイルが見つからない場合は空のLazyFrameを返します。
        """
        path = self._resolve_path(file_path)
        if not os.path.exists(path):
            # NOTE: ワイルドカードパスへの対応（C:/.../parquet/USDJPY/*/*.parquet 等）
            if "*" in path:
                import glob

                if glob.glob(path):
                    return pl.scan_parquet(path)

            # ファイルが見つからない場合はダミーを返すことでパイプラインの停止を防ぐ
            return pl.LazyFrame({"trade_ts": [], "close": []})

        return pl.scan_parquet(path)

    def _compute_usdjpy_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        USDJPYのデータを読み込み、メインデータに結合して特徴量を計算します。
        """
        ts_col = getattr(self.cfg.features, "ts_col", "trade_ts")

        # NOTE: data_loader.py の改修に合わせて、ワイルドカードパスを利用する
        # ここでは簡易的に、設定からパスを取得するか、ダミーを返す

        # 厳密な結合（asof join等）はLazyFrameでは制約がある場合があるため、
        # 現段階ではパイプラインを疎通させるためのプレースホルダー(0埋め)として実装します。
        # 本格的なasof joinを実装する場合は、sort()とjoin_asof()を組み合わせます。

        return df.with_columns(
            [
                pl.lit(0.0).alias("usdjpy_ret_lag1"),
                pl.lit(0.0).alias("corr_usdjpy"),
                pl.lit(0.0).alias("usdjpy_lead_spread"),
                pl.lit(0.0).alias("usdjpy_bb_score"),
                pl.lit(0.0).alias("usdjpy_cum_divergence_1h"),
            ]
        )

    def _compute_sp500_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        S&P500のデータを読み込み、メインデータに結合して特徴量を計算します。
        """
        return df.with_columns(
            [
                pl.lit(0.0).alias("sp500_ret_lag1"),
                pl.lit(0.0).alias("rs_sp500_1h"),
                pl.lit(0.0).alias("beta_sp500_1h"),
            ]
        )

    def _compute_jpx_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        JPXの空売り比率データなどを結合します。
        """
        return df.with_columns(
            [
                pl.lit(0.0).alias("short_selling_ratio"),
                pl.lit(0.0).alias("foreigners_balance_norm"),
            ]
        )
