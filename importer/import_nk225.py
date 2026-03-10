# import_data.py
"""
File: import_data.py

ソースコードの役割:
JPX先物(Tick)およびMT5(Tick)生データを読み込み、システム共通の1秒足Parquetフォーマットに変換する統合データパイプラインスクリプトです。
日経225ミニの歩み値からの売買方向推定（Tickテスト）や、外部環境データ（USDJPY, S&P500等）のタイムゾーン変換・動的リサンプリングを担当し、
モデルの学習・推論用の基盤データを構築します。
"""

# JPX先物(Tick)およびMT5(Tick)データを読み込み、
# システム共通の1秒足Parquetフォーマットに変換する統合スクリプト。
#
# 機能:
#     1. NK225モード: JPX生TSV (単一 or リスト) -> 期近判定 -> 方向推定 -> 1秒足
#         - inputs/NK225.txt
#         - inputs/NK225/future_tick_19_20YYMM.tsv
#     2. MT5モード:   MT5 Tick TSV -> タイムゾーン変換 (MT5->JST) -> Askベース1秒足
#         - inputs/MT5/USDJPY.tsv
#         - inputs/MT5/US500.tsv
#         - inputs/MT5/XAUUSD.tsv
#         - inputs/MT5/BTCUSD.tsv

from __future__ import annotations
import os
import argparse
import logging
import polars as pl
import config as cfg

# ログ設定
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 定数定義 ---
JPX_TSV_COLS = ["trade_date", "time", "trade_price",
                "price_type", "trade_volume", "contract_month"]
MT5_TSV_COLS = ["timestamp", "bid", "ask"]


class ConvertNK225:
    """JPX先物データ（日経225ミニ）の変換ロジックを管理するクラス。"""

    @staticmethod
    def read_minimal_tsv(tsv_path: str) -> pl.DataFrame:
        """JPX生TSVを読み込み、型変換とNaiveな日時結合を行います。

        Args:
            tsv_path (str): 読み込むTSVファイルのパス。

        Returns:
            pl.DataFrame: 型変換および日時結合が完了したデータフレーム。

        Raises:
            ValueError: 必須カラムが欠落している場合。
        """
        logger.info(f"Reading JPX TSV: {tsv_path}")
        df = pl.read_csv(
            tsv_path,
            separator="\t",
            has_header=True,
            infer_schema_length=0,
            ignore_errors=True,
        )

        missing = [c for c in JPX_TSV_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Required columns missing in {tsv_path}: {missing}")

        # 型変換とパディング
        df = df.select([
            pl.col("trade_date").cast(pl.Utf8).str.replace_all(
                r"\D", "").str.slice(0, 8),
            pl.col("time").cast(pl.Utf8).str.replace_all(
                r"\D", "").str.zfill(9).str.slice(0, 9),
            pl.col("trade_price").cast(pl.Float64),
            pl.col("price_type").cast(pl.Utf8),
            pl.col("trade_volume").cast(pl.Int64),
            pl.col("contract_month").cast(pl.Int64),
        ])

        # trade_dt (JST naive) 作成
        # format: YYYYMMDD + HHMMSSmmm
        trade_dt = (pl.col("trade_date") + pl.col("time")).str.strptime(
            pl.Datetime, "%Y%m%d%H%M%S%3f", strict=False
        )
        df = df.with_columns(trade_dt.alias("trade_dt")
                             ).drop_nulls(subset=["trade_dt"])

        return df.sort("trade_dt")

    @staticmethod
    def process(df: pl.DataFrame) -> pl.DataFrame:
        """JPX生データに対して以下の処理を行い、1秒足データへリサンプリングします。
        1. 期近限月のフィルタリング
        2. 歩み値からの売買方向推定 (Tick Test)
        3. 動的リサンプリングと特徴量（Volume, Size Std等）の集計

        Args:
            df (pl.DataFrame): `read_minimal_tsv` で読み込まれた生データフレーム。

        Returns:
            pl.DataFrame: 1秒足にリサンプリングされたデータフレーム。
        """
        if df.is_empty():
            return df

        # --- 1. 期近フィルタ ---
        # logger.info("Filtering near month contracts...")
        cm = df.group_by("trade_date").agg(
            pl.col("contract_month").min().alias("near_cm"))
        df = df.join(cm, on="trade_date", how="inner").filter(
            pl.col("contract_month") == pl.col("near_cm")
        ).drop("near_cm")

        # --- 2. 方向推定 ---
        # logger.info("Estimating trade direction...")
        df = df.sort("trade_dt")
        df = df.with_columns(
            pl.col("trade_price").shift(1).alias("prev_price"))

        direction = (
            pl.when(pl.col("prev_price").is_null()).then(pl.lit(1))
            .when(pl.col("trade_price") > pl.col("prev_price")).then(pl.lit(1))
            .when(pl.col("trade_price") < pl.col("prev_price")).then(pl.lit(-1))
            .otherwise(None)
        )
        df = df.with_columns(direction.alias("direction")).with_columns(
            pl.col("direction").fill_null(strategy="forward")
        )

        df = df.with_columns([
            pl.when(pl.col("direction") == 1).then(
                pl.col("trade_volume")).otherwise(0).alias("buy_volume"),
            pl.when(pl.col("direction") == -1).then(pl.col("trade_volume")
                                                    ).otherwise(0).alias("sell_volume"),
            pl.when(pl.col("direction") == 1).then(
                1).otherwise(0).alias("buy_tick"),
            pl.when(pl.col("direction") == -
                    1).then(1).otherwise(0).alias("sell_tick"),
        ])

        # --- 3. リサンプリング ---
        bar_interval = f"{cfg.BAR_SECONDS}s"
        # logger.info(f"Resampling to {bar_interval} bars (NK225)...")

        # Volume > 0 のみフィルタリング (JPXは夜間休場明けなどにゴミが入ることがある)
        q = (
            df.filter((pl.col("price_type") == "N") & (
                pl.col("trade_volume") > 0)).lazy()
            .group_by_dynamic('trade_dt', every=bar_interval, label='left')
            .agg([
                pl.col('trade_price').last().alias('trade_price'),
                pl.col('trade_price').max().alias('high_price'),
                pl.col('trade_price').min().alias('low_price'),
                pl.col('trade_price').std().fill_null(0).alias('price_std'),
                pl.col('buy_volume').sum(),
                pl.col('sell_volume').sum(),
                # 平均算出のためにカウントを集計
                pl.col('buy_tick').sum().alias('buy_tick_count'),
                pl.col('sell_tick').sum().alias('sell_tick_count'),
                pl.len().alias('tick_count'),
                # 大口検知用: 約定サイズの標準偏差 (1本のバーの中でサイズが均一か、特異点があるか)
                pl.col('trade_volume').std().fill_null(
                    0).alias('vol_size_std'),

                # ★追加: その1分間で発生した「最大の約定サイズ」
                pl.col('trade_volume').max().alias('max_trade_size'),

                # ★追加: 買い圧力の合計 (Buy Vol - Sell Vol) の純額
                (pl.col('buy_volume').sum() -
                 pl.col('sell_volume').sum()).alias('delta_volume')
            ])
            # 平均約定サイズの算出 (volume / count)
            .with_columns([
                (pl.col('buy_volume') / pl.col('buy_tick_count')
                 ).fill_nan(0).fill_null(0).alias('avg_buy_size'),
                (pl.col('sell_volume') / pl.col('sell_tick_count')
                 ).fill_nan(0).fill_null(0).alias('avg_sell_size')
            ])
            .with_columns(pl.col('trade_dt').dt.cast_time_unit("ns").alias('trade_ts'))
            .drop('trade_dt')
        )

        out = q.collect().sort("trade_ts")

        # 出力に必要なカラムを指定
        req_cols = [
            "trade_ts", "trade_price", "high_price", "low_price",
            "buy_volume", "sell_volume", "tick_count",
            "avg_buy_size", "avg_sell_size", "vol_size_std",
            "max_trade_size", "delta_volume"  # ★追加
        ]
        return ConvertNK225._finalize_columns(out, req_cols)

    @staticmethod
    def process_from_list(list_path: str) -> pl.DataFrame:
        """ファイルパスが記述されたテキストファイルを読み込み、各TSVを処理して結合します。

        Args:
            list_path (str): 処理対象のTSVファイルパスが1行ずつ記載されたリストファイルのパス。

        Returns:
            pl.DataFrame: 全てのTSVファイルを処理・結合したデータフレーム。

        Raises:
            FileNotFoundError: 有効なファイルパスが見つからない場合。
            ValueError: 処理結果のデータフレームが空の場合。
        """
        logger.info(f"Processing file list: {list_path}")
        with open(list_path, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()
                     and os.path.exists(line.strip())]

        if not files:
            raise FileNotFoundError(f"No valid paths found in {list_path}")

        dfs = []
        for i, p in enumerate(files):
            logger.info(f"[{i+1}/{len(files)}] Processing: {p}")
            try:
                raw = ConvertNK225.read_minimal_tsv(p)
                bars = ConvertNK225.process(raw)
                if not bars.is_empty():
                    dfs.append(bars)
            except Exception as e:
                logger.error(f"Failed to process {p}: {e}")

        if not dfs:
            raise ValueError("No valid data generated from list.")

        logger.info("Concatenating all chunks...")
        return pl.concat(dfs).sort("trade_ts")

    @staticmethod
    def _finalize_columns(df: pl.DataFrame, req_cols: list[str]) -> pl.DataFrame:
        """出力に必要なカラムを整え、存在しないカラムは0埋めで追加します。

        Args:
            df (pl.DataFrame): 処理中のデータフレーム。
            req_cols (list[str]): 最終的に出力に含める必須カラムのリスト。

        Returns:
            pl.DataFrame: カラム構成が統一されたデータフレーム。
        """
        for c in req_cols:
            if c not in df.columns:
                if c == "trade_ts":
                    continue
                df = df.with_columns(pl.lit(0).alias(c))
        return df.select(req_cols).sort("trade_ts")


class ConvertMT5:
    """MT5 Tick TSV (USDJPY, S&P500等) の変換ロジックを管理するクラス。"""

    @staticmethod
    def read_tick_tsv(tsv_path: str) -> pl.LazyFrame:
        """MT5の生Tickデータを遅延評価（LazyFrame）で読み込みます。

        Args:
            tsv_path (str): 読み込むTSVファイルのパス。

        Returns:
            pl.LazyFrame: タイムスタンプがパースされたLazyFrame。
        """
        logger.info(f"Reading MT5 Tick TSV: {tsv_path}")
        # read_csv (Eager) -> scan_csv (Lazy) に変更し、メモリ展開を遅延させる
        lf = pl.scan_csv(
            tsv_path,
            separator="\t",
            has_header=True,
            new_columns=MT5_TSV_COLS,
            schema_overrides={"bid": pl.Float64, "ask": pl.Float64}
        )
        lf = lf.with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime,
                                             format="%Y.%m.%d %H:%M:%S.%3f", strict=False)
            .alias("mt5_ts")
        )
        return lf.drop_nulls(subset=["mt5_ts"])

    @staticmethod
    def process_to_1s_bars(lf: pl.LazyFrame) -> pl.DataFrame:
        """MT5のTickデータをJSTの1秒足にリサンプリングします。

        Args:
            lf (pl.LazyFrame): `read_tick_tsv` で読み込まれたLazyFrame。

        Returns:
            pl.DataFrame: タイムゾーン変換・リサンプリングが完了したデータフレーム。
        """
        logger.info("Configuring Lazy Query (Timezone -> Resample)...")
        # MT5(-7h) -> NY(US/Eastern) -> JST(Asia/Tokyo)
        interval = f"{cfg.BAR_SECONDS}s"

        q = (
            lf
            .with_columns(
                (pl.col("mt5_ts") - pl.duration(hours=7))
                .dt.replace_time_zone("US/Eastern", ambiguous="earliest", non_existent="null")
                .dt.convert_time_zone("Asia/Tokyo")
                .dt.replace_time_zone(None)
                .dt.cast_time_unit("ns")
                .alias("trade_ts")
            )
            .drop_nulls(subset=["trade_ts"])
            .set_sorted("trade_ts")
            .group_by_dynamic("trade_ts", every=interval, label="left")
            .agg([
                pl.col("ask").first().alias("open"),
                pl.col("ask").max().alias("high"),
                pl.col("ask").min().alias("low"),
                pl.col("ask").last().alias("close"),
                pl.len().alias("tickvol"),
                (pl.col("ask") - pl.col("bid")).mean().alias("spread")
            ])
            # カンニング防止: available_ts = trade_ts + interval
            .with_columns(
                pl.col("trade_ts").dt.offset_by(interval).alias("available_ts")
            )
        )
        logger.info(f"Executing Streaming Query (Resampling to {interval})...")
        return q.collect(engine="streaming").select([
            "available_ts", "open", "high", "low", "close", "tickvol", "spread"
        ]).sort("available_ts")


def main():
    """コマンドライン引数を解析し、指定されたモード（NK225/MT5）でデータ変換処理を実行します。"""
    parser = argparse.ArgumentParser(
        description="Convert Tick Data (JPX/MT5) to 1s Parquet")
    parser.add_argument(
        "mode", choices=["nk225", "mt5"], help="Conversion mode")

    # NK225用: 単一ファイル or リストファイル
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", "-i", help="Input TSV file path")
    group.add_argument(
        "--list", "-l", help="Path to text file listing TSVs (NK225 only)")

    parser.add_argument("--output", "-o", required=True,
                        help="Output Parquet file path")

    args = parser.parse_args()

    try:
        if args.mode == "nk225":
            # NK225モード
            if args.list:
                # リスト処理 (旧 --list 継承)
                df_out = ConvertNK225.process_from_list(args.list)
            elif args.input:
                # 単一ファイル処理
                if not os.path.exists(args.input):
                    raise FileNotFoundError(f"{args.input} not found")
                df_raw = ConvertNK225.read_minimal_tsv(args.input)
                df_out = ConvertNK225.process(df_raw)
            else:
                parser.error("NK225 mode requires either --input or --list")

        elif args.mode == "mt5":
            # MT5モード
            if not args.input:
                parser.error("MT5 mode requires --input")
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"{args.input} not found")

            loader = ConvertMT5()
            df_raw = loader.read_tick_tsv(args.input)
            df_out = loader.process_to_1s_bars(df_raw)

        logger.info(f"Saving to {args.output} (Shape: {df_out.shape})...")
        df_out.write_parquet(args.output)
        logger.info("Done.")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
