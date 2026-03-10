# importer/import_nk225.py
"""
File: importer/import_nk225.py

ソースコードの役割:
JPXデータクラウド形式の日経平均先物（NK225）歩み値データをインポートします。
期近限月の選別、Tick Testによる売買方向推定を行い、階層化された Parquet 形式で保存します。
"""

import glob
import os
import re
import sys
from typing import List

import polars as pl

# 親ディレクトリの config.py を読み込むためのパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import cfg, BAR_SECONDS


def load_jpx_ticks_from_tsv(file_path: str) -> pl.DataFrame:
    """JPX形式のTSVから歩み値を読み込み、売買方向を推定して返します。"""
    if not os.path.exists(file_path):
        return pl.DataFrame()

    try:
        ldf = pl.read_csv(
            file_path,
            separator="\t",
            has_header=True,
            dtypes={
                "trade_date": pl.Utf8,
                "time": pl.Utf8,
                "trade_price": pl.Float64,
                "trade_volume": pl.Int64,
                "contract_month": pl.Int64,
                "price_type": pl.Utf8,
            },
        )

        ldf = ldf.with_columns(
            (pl.col("trade_date") + pl.col("time").str.zfill(9))
            .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S%3f", strict=False)
            .alias("trade_ts")
        ).drop_nulls(subset=["trade_ts"])

        # 期近フィルタリング
        near_cm_df = ldf.group_by("trade_date").agg(
            pl.col("contract_month").min().alias("min_cm")
        )
        ldf = ldf.join(near_cm_df, on="trade_date").filter(
            pl.col("contract_month") == pl.col("min_cm")
        )

        # Tick Testによる売買方向推定
        ldf = ldf.filter(pl.col("price_type") == "N").sort("trade_ts")
        ldf = ldf.with_columns(pl.col("trade_price").diff().alias("price_diff"))
        ldf = ldf.with_columns(
            pl.when(pl.col("price_diff") > 0)
            .then(1)
            .when(pl.col("price_diff") < 0)
            .then(-1)
            .otherwise(None)
            .alias("direction")
        ).with_columns(pl.col("direction").fill_null(strategy="forward").fill_null(1))

        ldf = ldf.with_columns(
            [
                pl.when(pl.col("direction") == 1)
                .then(pl.col("trade_volume"))
                .otherwise(0)
                .alias("buy_vol"),
                pl.when(pl.col("direction") == -1)
                .then(pl.col("trade_volume"))
                .otherwise(0)
                .alias("sell_vol"),
            ]
        )

        return ldf.select(
            ["trade_ts", pl.col("trade_price").alias("price"), "buy_vol", "sell_vol"]
        )

    except Exception as e:
        print(f"  [Error] {os.path.basename(file_path)} の処理中にエラー: {e}")
        return pl.DataFrame()


def resample_to_bars(tick_df: pl.DataFrame, interval_sec: int) -> pl.DataFrame:
    """指定秒足に集約します。"""
    interval_str = f"{interval_sec}s"
    return (
        tick_df.with_columns(
            pl.col("trade_ts").dt.truncate(interval_str).alias("trade_ts")
        )
        .group_by("trade_ts")
        .agg(
            [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("buy_vol").sum().alias("buy_volume"),
                pl.col("sell_vol").sum().alias("sell_volume"),
                (pl.col("buy_vol").sum() + pl.col("sell_vol").sum()).alias("volume"),
            ]
        )
        .sort("trade_ts")
    )


def main():
    root_tsv_dir = "C:/transformer_futures_data/tsv"
    output_base_dir = "C:/transformer_futures_data/parquet"
    symbol = "NK225"

    print(f"NK225 インポート開始: Root={root_tsv_dir}, Output={output_base_dir}")

    files = sorted(
        glob.glob(
            os.path.join(root_tsv_dir, symbol, "**", "future_tick_*.tsv.gz"),
            recursive=True,
        )
    )

    for file_path in files:
        # ファイル名から YYYYMM を抽出 (例: future_tick_19_201801.tsv.gz -> 201801)
        match = re.search(r"_(\d{6,8})\.tsv", os.path.basename(file_path))
        target_period = match.group(1) if match else "000000"

        raw_ticks = load_jpx_ticks_from_tsv(file_path)
        if raw_ticks.is_empty():
            continue

        bars_df = resample_to_bars(raw_ticks, BAR_SECONDS)

        # --- 保存先パスの階層化 ---
        year_str = target_period[:4]
        out_dir = os.path.join(output_base_dir, symbol, year_str)
        out_filename = f"{symbol}-{BAR_SECONDS}-{target_period}.parquet"
        out_path = os.path.join(out_dir, out_filename)

        try:
            os.makedirs(out_dir, exist_ok=True)
            bars_df.write_parquet(out_path)
            print(f"  => Saved: {out_path} ({len(bars_df)} rows)")
        except Exception as e:
            print(f"  [Save Error] {out_filename}: {e}")

    print("\nすべてのNK225データ処理が完了しました。")


if __name__ == "__main__":
    main()
