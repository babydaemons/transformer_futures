# data/data_loader.py
"""
File: data/data_loader.py

ソースコードの役割:
各金融資産（銘柄）および外部データ（JPX空売り比率等）のファイル読み込み処理を管理します。
銘柄ごとのロード関数を辞書型でマッピングすることで、複数アセットへの拡張性・保守性を高めています。
また、学習やバックテストのループで基準となる取引日(trading dates)を効率的に抽出する機能を提供します。
"""

import os
import logging
import datetime
from typing import List, Dict, Callable, Any

import polars as pl
from config import GlobalConfig


# =========================================================
# 各銘柄ごとのデータロード関数群
# =========================================================
# 目的: 各銘柄ごとにファイルパスやパース要件が異なる場合に備え、
# 読み込みロジックを個別の関数として分離します。
# 戻り値はメモリ効率の良い polars.LazyFrame とします。

def load_nk225(cfg: GlobalConfig) -> pl.LazyFrame:
    """日経225先物(NK225)のデータをロード"""
    return pl.scan_parquet(cfg.features.nk225_file)


def load_usdjpy(cfg: GlobalConfig) -> pl.LazyFrame:
    """ドル円(USDJPY)のデータをロード"""
    return pl.scan_parquet(cfg.features.usdjpy_file)


def load_us500(cfg: GlobalConfig) -> pl.LazyFrame:
    """S&P500(US500)のデータをロード"""
    return pl.scan_parquet(cfg.features.sp500_file)


def load_xauusd(cfg: GlobalConfig) -> pl.LazyFrame:
    """金(XAUUSD)のデータをロード"""
    return pl.scan_parquet(cfg.features.xauusd_file)


def load_xtiusd(cfg: GlobalConfig) -> pl.LazyFrame:
    """原油(XTIUSD)のデータをロード"""
    return pl.scan_parquet(cfg.features.xtiusd_file)


def load_short_selling(cfg: GlobalConfig) -> pl.LazyFrame:
    """JPX空売り比率データ(日次)をロード"""
    return pl.scan_csv(cfg.features.short_selling_file, separator="\t")


def load_investor_type(cfg: GlobalConfig) -> pl.LazyFrame:
    """JPX投資主体別売買動向データ(週次)をロード"""
    return pl.scan_csv(cfg.features.investor_file, separator="\t")


# =========================================================
# ロード関数の管理辞書
# =========================================================
# シンボル名をキーとして、対応するロード関数を管理します。
# 新しい銘柄を追加する場合は、上記の関数を作成し、この辞書に追加するだけで対応可能です。
ASSET_LOADERS: Dict[str, Callable[[GlobalConfig], pl.LazyFrame]] = {
    "NK225": load_nk225,
    "USDJPY": load_usdjpy,
    "US500": load_us500,
    "XAUUSD": load_xauusd,
    "XTIUSD": load_xtiusd,
    "SHORT_SELLING": load_short_selling,
    "INVESTOR_TYPE": load_investor_type,
}


class MarketDataLoader:
    """
    データセット構築のベースとなるデータ読み込みおよび取引日抽出を行うクラス。
    """

    def __init__(self, cfg: GlobalConfig):
        """
        Args:
            cfg (GlobalConfig): システム全体の設定オブジェクト
        """
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

    def load_symbol(self, symbol: str) -> pl.LazyFrame:
        """
        指定されたシンボルのデータをロードします。

        Args:
            symbol (str): ロードするシンボル名 (例: "NK225", "USDJPY")

        Returns:
            pl.LazyFrame: ロードされた遅延評価データフレーム

        Raises:
            ValueError: 未知のシンボルが指定された場合
        """
        loader_fn = ASSET_LOADERS.get(symbol.upper())
        if loader_fn is None:
            available_symbols = list(ASSET_LOADERS.keys())
            raise ValueError(
                f"Unknown symbol: '{symbol}'. Available symbols: {available_symbols}"
            )

        return loader_fn(self.cfg)

    def get_trading_dates(self, main_symbol: str = "NK225") -> List[datetime.date]:
        """
        指定されたメイン銘柄から、取引が行われた日付(Trading Dates)のユニークなリストを取得します。

        Args:
            main_symbol (str): 基準とする銘柄コード。デフォルトは "NK225"。

        Returns:
            List[datetime.date]: 昇順にソートされた取引日のリスト。
        """
        self.logger.info(f"Scanning trading dates from {main_symbol}...")

        try:
            lf = self.load_symbol(main_symbol)
            ts_col = self.cfg.features.ts_col

            # タイムスタンプ列から日付(Date)部分のみを抽出し、ユニーク化・ソートを実行してメモリに展開(collect)
            dates_df = (
                lf.select(pl.col(ts_col).dt.date().alias("date"))
                .unique()
                .sort("date")
                .drop_nulls()
                .collect()
            )

            dates_list = dates_df["date"].to_list()
            self.logger.info(
                f"Successfully found {len(dates_list)} trading dates for {main_symbol}."
            )
            return dates_list

        except FileNotFoundError as e:
            self.logger.error(
                f"[{main_symbol}] のデータファイルが見つかりません。 "
                f"config.py の data_dir の設定、または環境変数 DATA_DIR に正しいパスが設定されているか確認してください。\n"
                f"詳細なエラー: {e}"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error while fetching dates for {main_symbol}: {e}")
            raise
