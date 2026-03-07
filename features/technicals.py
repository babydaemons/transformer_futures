# features/technicals.py
"""
File: features/technicals.py

ソースコードの役割:
本モジュールは、データフレームに対するテクニカル指標（RSI, MACD, Bollinger Bands, ATR, ADX,
Efficiency Ratio, Garman-Klass Volatilityなど）の計算処理、および歩み値や出来高に基づく
Microstructure（微細構造）特徴量の計算処理を提供します。
"""

import polars as pl
import numpy as np


def compute_flow_imbalance_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    出来高や約定回数に基づくMicrostructure（微細構造）特徴量を計算します。

    Args:
        df (pl.DataFrame): 基本的な価格・出来高・歩み値カラムを含むデータフレーム。

    Returns:
        pl.DataFrame: Microstructure特徴量が追加されたデータフレーム。
    """
    # 1. 1分間の総出来高（他モジュールでも参照される基本指標）
    if "vol_total_1bar" not in df.columns:
        df = df.with_columns(
            (pl.col("buy_volume") + pl.col("sell_volume")).alias("vol_total_1bar")
        )

    # 2. 日次累積デルタ（CVD: Cumulative Volume Delta）の生値計算
    # 日ごとにリセットして delta_volume を累積する（technicals.py等で正規化される前提）
    if "delta_volume" in df.columns:
        df = df.with_columns(
            pl.col("delta_volume")
            .cum_sum()
            .over(pl.col("timestamp").dt.date())
            .alias("day_cum_delta_raw")
        )

    # 3. 各種微細構造指標の計算
    df = df.with_columns(
        [
            # Amihud Illiquidity Proxy: |Return| / Volume (非流動性の代理指標)
            (pl.col("log_ret").abs() / (pl.col("vol_total_1bar") + 1e-8))
            .fill_null(0.0)
            .ewm_mean(span=5, adjust=False)  # 突発的なスパイクを均す
            .alias("amihud_illiquidity"),
            
            # 相対出来高 (Relative Volume): 過去15本平均に対する現在の出来高の比率
            (
                pl.col("vol_total_1bar")
                / (pl.col("vol_total_1bar").rolling_mean(15) + 1e-8)
            )
            .fill_null(1.0)
            .alias("rel_vol"),
            
            # Tick Speed Ratio: 過去15本平均に対する現在のTick（約定回数）の比率
            (pl.col("tick_count") / (pl.col("tick_count").rolling_mean(15) + 1e-8))
            .fill_null(1.0)
            .alias("tick_speed_ratio"),
            
            # Buy Pressure: 総出来高に対する買い出来高の割合
            (pl.col("buy_volume") / (pl.col("vol_total_1bar") + 1e-8))
            .fill_null(0.5)
            .alias("buy_pressure"),
            
            # Size Imbalance: 買い平均サイズと売り平均サイズの不均衡
            (
                (pl.col("avg_buy_size") - pl.col("avg_sell_size"))
                / (pl.col("avg_buy_size") + pl.col("avg_sell_size") + 1e-8)
            )
            .fill_null(0.0)
            .ewm_mean(span=3, adjust=False)  # 突発的な大口約定のノイズを均す
            .alias("size_imb_1bar"),
        ]
    )

    return df


# --- テクニカル指標計算用 プライベート関数群 ---

def _compute_rsi(pr: pl.Expr, period: int) -> pl.Expr:
    """RSIを計算するExprを返します。"""
    delta = pr.diff()
    up = delta.clip(lower_bound=0)
    down = -delta.clip(upper_bound=0)

    rsi = 100 - 100 / (
        1
        + up.ewm_mean(span=period, adjust=False)
        / (down.ewm_mean(span=period, adjust=False) + 1e-8)
    )
    return rsi.fill_null(50).alias("rsi")


def _compute_macd(pr: pl.Expr, fast: int, slow: int) -> pl.Expr:
    """MACDを計算するExprを返します。"""
    ema_fast = pr.ewm_mean(span=fast, adjust=False)
    ema_slow = pr.ewm_mean(span=slow, adjust=False)
    macd = ema_fast - ema_slow
    return macd.fill_null(0).alias("macd")


def _compute_bb(pr: pl.Expr, window: int) -> pl.Expr:
    """ボリンジャーバンドの％Bを計算するExprを返します。"""
    bb_mean = pr.rolling_mean(window)
    bb_std = pr.rolling_std(window)
    bb_b = (pr - (bb_mean - 2 * bb_std)) / (4 * bb_std + 1e-8)
    return bb_b.fill_null(0.5).alias("bb_b")


def _compute_atr(tr: pl.Expr, span: int) -> pl.Expr:
    """ATR (Average True Range) を計算するExprを返します。"""
    atr = tr.ewm_mean(span=span, adjust=False)
    return atr.fill_null(0).alias("atr")


def _compute_adx(tr: pl.Expr, period: int) -> pl.Expr:
    """ADX (Average Directional Index) を計算するExprを返します。"""
    up_move = pl.col("high") - pl.col("high").shift(1)
    down_move = pl.col("low").shift(1) - pl.col("low")

    plus_dm = pl.max_horizontal(
        pl.lit(0.0), pl.when(up_move > down_move).then(up_move).otherwise(0.0)
    )
    minus_dm = pl.max_horizontal(
        pl.lit(0.0), pl.when(down_move > up_move).then(down_move).otherwise(0.0)
    )

    tr_s = tr.ewm_mean(span=period, adjust=False)
    plus_di = 100 * (plus_dm.ewm_mean(span=period, adjust=False) / (tr_s + 1e-8))
    minus_di = 100 * (minus_dm.ewm_mean(span=period, adjust=False) / (tr_s + 1e-8))

    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di).abs() + 1e-8)
    adx = dx.ewm_mean(span=period, adjust=False)
    return adx.fill_null(0).alias("adx")


def _compute_er(period: int, alias_name: str) -> pl.Expr:
    """KaufmanのEfficiency Ratioを計算するExprを返します。"""
    change = (pl.col("close") - pl.col("close").shift(period)).abs()
    volatility = pl.col("close").diff().abs().rolling_sum(period)
    er = (change / (volatility + 1e-8)).fill_null(0)
    return er.alias(alias_name)


def _compute_gk_vol(has_open: bool) -> pl.Expr:
    """Garman-Klass Volatilityを計算するExprを返します。"""
    log_hl = (pl.col("high") / pl.col("low")).log()

    if has_open:
        log_co = (pl.col("close") / pl.col("open")).log()
    else:
        log_co = pl.lit(0.0)

    # 負の値に対する平方根計算（NaN発生）を防ぐためにclipを追加
    gk_vol = (
        (0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2))
        .clip(lower_bound=0.0)
        .sqrt()
    )
    return gk_vol.fill_nan(0).fill_null(0).alias("garman_klass_vol")


def _compute_cvd_norm() -> pl.Expr:
    """CVDの正規化計算を行うExprを返します。"""
    # CVDの正規化: 単純に過去60分の出来高移動平均で割って標準化する簡易手法
    vol_ma = pl.col("vol_total_1bar").rolling_mean(60) + 1e-8
    cum_delta_norm = (
        (pl.col("day_cum_delta_raw") / (vol_ma * 100))
        .clip(-5.0, 5.0)
        .fill_null(0)
        .alias("day_cum_delta_norm")
    )
    return cum_delta_norm


# --- メイン計算関数 ---

def compute_technicals(
    df: pl.DataFrame,
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    bb_window: int,
    adx_period: int,
) -> pl.DataFrame:
    """テクニカル指標（RSI, MACD, Bollinger Bands, ATRなど）を計算します。

    Args:
        df (pl.DataFrame): 価格や出来高などの基本情報を含むデータフレーム。
        rsi_period (int): RSIの計算期間。
        macd_fast (int): MACDの短期EMA期間。
        macd_slow (int): MACDの長期EMA期間。
        bb_window (int): ボリンジャーバンドの計算期間。
        adx_period (int): ADXの計算期間。

    Returns:
        pl.DataFrame: テクニカル指標が追加されたデータフレーム。
    """
    pr = pl.col("close")
    has_open = "open" in df.columns

    # --- 共通で利用する計算 (True Range) ---
    # ATRおよびADXの両方で使用するため、事前定義して各関数に渡す
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )

    # 各種テクニカル指標のExprリストを構築
    exprs = [
        _compute_rsi(pr, rsi_period),
        _compute_macd(pr, macd_fast, macd_slow),
        _compute_bb(pr, bb_window),
        _compute_atr(tr, span=60), # 1分足なので期間を長めにとってノイズを除去 (60分ATR)
        _compute_adx(tr, adx_period),
        _compute_er(period=15, alias_name="efficiency_ratio"),
        _compute_er(period=240, alias_name="efficiency_ratio_1h"), # 長期トレンド効率 (4時間 = 240 bars @ 1min)
        _compute_gk_vol(has_open),
        _compute_cvd_norm(),
    ]

    # リスト化したExprを一括適用（遅延評価/並列処理の最適化を維持）
    return df.with_columns(exprs)
