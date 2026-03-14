# data/data_loader.py
"""
File: data/data_loader.py

ソースコードの役割:
各金融資産（銘柄）および外部データ（JPX空売り比率等）のファイル読み込み処理を管理します。
銘柄ごとのロード関数を辞書型でマッピングすることで、複数アセットへの拡張性・保守性を高めています。
また、学習やバックテストのループで基準となる取引日(trading dates)を効率的に抽出する機能を提供します。

【機能改修】
全期間単一ファイルからではなく、「基準ディレクトリ/parquet/銘柄名/西暦/*.parquet」の
ディレクトリ階層ルールに従い、ワイルドカードを用いて分割されたParquetファイルを一括連結ロードします。
"""

import os
import glob
import logging
import datetime
from typing import List, Dict, Callable, Any, Optional

import polars as pl
from config import GlobalConfig


# =========================================================
# ファイルパスの自動修復・フォールバック機能
# =========================================================
def _resolve_path(configured_path: str) -> str:
    """
    設定されたパスが存在しない場合、正しいディレクトリパスへ
    自動フォールバックして探索するヘルパー関数です。(主にTSVファイル等用)
    """
    if os.path.exists(configured_path):
        return configured_path

    fallback_dirs = ["C:/transformer_futures_data", "C:/transformer_features_data"]
    file_name = os.path.basename(configured_path)

    for fallback_dir in fallback_dirs:
        fallback_path = os.path.join(fallback_dir, file_name).replace("\\", "/")
        if os.path.exists(fallback_path):
            return fallback_path

    return configured_path


def _get_parquet_glob_path(cfg: GlobalConfig, symbol: str) -> str:
    """
    指定された銘柄の分割Parquetファイルをワイルドカードで一括ロードするためのパスを生成します。
    例: C:/transformer_futures_data/parquet/NK225/*/*.parquet
    """
    data_dir = cfg.features.data_dir

    # ベースディレクトリが存在しない場合の自動フォールバック
    if not os.path.exists(data_dir):
        fallback_dirs = ["C:/transformer_futures_data", "C:/transformer_features_data"]
        for fallback in fallback_dirs:
            if os.path.exists(fallback):
                data_dir = fallback
                break

    # ワイルドカードパスの構築
    # 構造: 基準ディレクトリ / parquet / 銘柄名 / 西暦(ワイルドカード) / ファイル名(ワイルドカード)
    glob_path = os.path.join(data_dir, "parquet", symbol, "*", "*.parquet").replace(
        "\\", "/"
    )

    # globで実際にファイルが存在するかチェック（Polars側でエラーになる前にキャッチするため）
    if not glob.glob(glob_path):
        raise FileNotFoundError(
            f"ワイルドカードに一致するデータが見つかりません: {glob_path}"
        )

    return glob_path


# =========================================================
# 各銘柄ごとのデータロード関数群
# =========================================================
# 目的: 各銘柄ごとにファイルパスやパース要件が異なる場合に備え、
# 読み込みロジックを個別の関数として分離します。
# 戻り値はメモリ効率の良い polars.LazyFrame とします。


def load_nk225(cfg: GlobalConfig) -> pl.LazyFrame:
    """日経225先物(NK225)のデータを一括ロード"""
    return pl.scan_parquet(_get_parquet_glob_path(cfg, "NK225"))


def load_usdjpy(cfg: GlobalConfig) -> pl.LazyFrame:
    """ドル円(USDJPY)のデータを一括ロード"""
    return pl.scan_parquet(_get_parquet_glob_path(cfg, "USDJPY"))


def load_us500(cfg: GlobalConfig) -> pl.LazyFrame:
    """S&P500(US500)のデータを一括ロード"""
    return pl.scan_parquet(_get_parquet_glob_path(cfg, "US500"))


def load_xauusd(cfg: GlobalConfig) -> pl.LazyFrame:
    """金(XAUUSD)のデータを一括ロード"""
    return pl.scan_parquet(_get_parquet_glob_path(cfg, "XAUUSD"))


def load_xtiusd(cfg: GlobalConfig) -> pl.LazyFrame:
    """原油(XTIUSD)のデータを一括ロード"""
    return pl.scan_parquet(_get_parquet_glob_path(cfg, "XTIUSD"))


def load_short_selling(cfg: GlobalConfig) -> pl.LazyFrame:
    """JPX空売り比率データ(日次)をロード"""
    return pl.scan_csv(_resolve_path(cfg.features.short_selling_file), separator="\t")


def load_investor_type(cfg: GlobalConfig) -> pl.LazyFrame:
    """JPX投資主体別売買動向データ(週次)をロード"""
    return pl.scan_csv(_resolve_path(cfg.features.investor_file), separator="\t")


# =========================================================
# ロード関数の管理辞書
# =========================================================
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

    def get_trading_dates(
        self, main_symbol: Optional[str] = None
    ) -> List[datetime.date]:
        """
        取引が行われた日付(Trading Dates)のユニークなリストを取得します。
        指定された銘柄が見つからない場合は、他の主要な銘柄から自動的にフォールバックして抽出します。

        Args:
            main_symbol (Optional[str]): 基準とする銘柄コード。省略された場合は自動探索。

        Returns:
            List[datetime.date]: 昇順にソートされた取引日のリスト。

        Raises:
            FileNotFoundError: 利用可能なデータファイルが1つも見つからなかった場合
        """
        # 探索するシンボルの優先順位
        symbols_to_try = (
            [main_symbol]
            if main_symbol
            else ["NK225", "US500", "USDJPY", "XAUUSD", "XTIUSD"]
        )

        for sym in symbols_to_try:
            self.logger.info(f"Scanning trading dates from {sym}...")
            try:
                lf = self.load_symbol(sym)
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

                if not dates_list:
                    self.logger.warning(
                        f"[{sym}] から日付データが取得できませんでした。"
                    )
                    continue

                self.logger.info(
                    f"Successfully found {len(dates_list)} trading dates from {sym}."
                )
                return dates_list

            except FileNotFoundError as e:
                # どのパスを探して見つからなかったのかを詳細にログ出力
                self.logger.warning(
                    f"[{sym}] のデータファイルが見つかりません。別の銘柄を探索します。(詳細: {e})"
                )
                continue
            except Exception as e:
                self.logger.warning(
                    f"[{sym}] の日付抽出中にエラーが発生しました: {type(e).__name__} - {e}"
                )
                continue

        # すべてのシンボルで失敗した場合
        error_msg = (
            "データの読み込みに失敗しました。\n"
            f"config.py で指定されたパス ({self.cfg.features.data_dir})、\n"
            "および C:/transformer_futures_data の両方を探索しましたがファイルが見つかりません。\n"
            "指定されたディレクトリ構造(例: .../parquet/NK225/*/*.parquet)にデータが存在するか確認してください。"
        )
        self.logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    def load_lazy_chunk(
        self, start_dt: datetime.date, end_dt: datetime.date, main_symbol: str = "NK225"
    ) -> pl.LazyFrame:
        """
        指定された日付範囲の基準銘柄(メインデータ)の遅延評価データフレーム(LazyFrame)を取得します。

        Args:
            start_dt (datetime.date): 取得開始日
            end_dt (datetime.date): 取得終了日
            main_symbol (str): 基準とする銘柄コード (デフォルト: "NK225")

        Returns:
            pl.LazyFrame: 日付でフィルタリングされたLazyFrame
        """
        lf = self.load_symbol(main_symbol)
        ts_col = self.cfg.features.ts_col

        # タイムスタンプでフィルタリング (datetime型に変換して比較)
        start_datetime = datetime.datetime.combine(start_dt, datetime.time.min)
        end_datetime = datetime.datetime.combine(end_dt, datetime.time.max)

        lf = lf.filter(
            (pl.col(ts_col) >= start_datetime) & (pl.col(ts_col) <= end_datetime)
        )

        return lf
