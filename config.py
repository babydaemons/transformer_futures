# config.py
"""
File: config.py

ソースコードの役割:
日経225先物ミニ・マイクロの1分足デイトレードシステム向け統合設定モジュール。
特徴量生成、モデル構造(Transformer系)、学習パラメータ、およびバックテストロジックの全設定を一元管理します。
往復コスト(15円)を克服し、スキャルピングではなく最大数時間のポジションホールドを想定した
デイトレードシステムの挙動を制御します。
"""

from datetime import datetime
import os
from dataclasses import dataclass, field
from typing import List

# --- 基本時間枠設定 ---
# デフォルトは1分足 (60秒)
BAR_SECONDS: int = int(os.getenv("BAR_SECONDS", 60))


@dataclass(frozen=True)
class FeatureConfig:
    """特徴量生成およびデータ入力に関する設定。

    Attributes:
        seq_len (int): コンテキスト長 (120分 = 2時間)。VRAM制限のため調整。
        predict_horizon (int): 予測ホライゾン (30分)。
        dataset_stride (int): データセット作成時のストライド幅。
        nk225_file (str): 日経225先物の入力ファイルパス。
        usdjpy_file (str): USDJPYの入力ファイルパス。
        sp500_file (str): S&P500の入力ファイルパス。
        ts_col (str): タイムスタンプのカラム名。
        price_col (str): 価格のカラム名。
        label_threshold_scale (float): ラベル付与時のノイズ耐性スケール。
        label_min_limit (float): コスト負けを防ぐための最小要求変動幅。
        label_min_limit_mode (str): ラベル最小値の適用モード ("fixed", "cost", "max")。
        label_cost_buffer (float): コスト負け回避のための追加エッジバッファ。
    """

    seq_len: int = 120
    predict_horizon: int = 30  # 60から30に短縮し、より直近のモメンタムを狙う

    # Dataset stride
    dataset_stride: int = int(os.getenv("DATASET_STRIDE", 1))

    # ファイルパス
    data_dir: str = "../transformer_futures.data"
    nk225_file: str = f"{data_dir}/NK225-{BAR_SECONDS}.parquet"
    usdjpy_file: str = f"{data_dir}s/USDJPY-{BAR_SECONDS}.parquet"
    sp500_file: str = f"{data_dir}/US500-{BAR_SECONDS}.parquet"

    # カラム定義
    ts_col: str = "trade_ts"
    price_col: str = "trade_price"
    label_threshold_scale: float = 1.2  # 1.5から1.2へ下げてハードルを緩和
    label_min_limit: float = 40.0  # 75.0から40.0へ下げ、より小さな波をターゲットにする

    # ラベルの最小値適用ロジック
    label_min_limit_mode: str = "cost"
    label_cost_buffer: float = 20.0  # コスト負け回避バッファ

    # ===== 特徴量定義 =====
    continuous_cols: List[str] = field(
        default_factory=lambda: [
            "log_ret",
            "log_vol",
            "ret_div_vol",  # Sharpe-like momentum
            "rel_vol",  # 相対出来高
            "buy_pressure",  # 需給圧力
            "log_buy_vol",
            "log_sell_vol",
            "day_cum_delta_norm",  # 日次累積デルタ(正規化)
            "ofi_signal",
            "volume_pressure",
            "price_spread",
            "trade_freq_accel",
            "amihud_illiquidity",
            "realized_vol",
            "parkinson_vol",
            "garman_klass_vol",
            "shadow_range",
            # --- Size Imbalance ---
            "size_imb_1bar",
            "vol_size_std",
            "max_trade_size",
            "tick_speed_ratio",
            "delta_volume",
            # --- MTF / Cross Asset ---
            "dist_15m",
            "dist_1h",
            "dist_4h",
            "dist_1d",
            "dist_vwap_1m",
            "dist_vwap_4h",
            "dist_vwap_1d",
            "rs_sp500_1h",
            "beta_sp500_1h",  # 対S&P500 ローリング・ベータ
            "vol_regime",
            "dist_prev_poc_1d",
            "dist_prev_poc_1w",
            "vol_skew_1h",
            "vol_skew_1d",
            # --- Technical ---
            "rsi",
            "macd",
            "bb_b",
            "atr",
            "adx",
            "efficiency_ratio",
            "efficiency_ratio_1h",
            # --- Macro ---
            "usdjpy_ret_lag1",
            "sp500_ret_lag1",
            "corr_usdjpy",
            "usdjpy_lead_spread",
            "usdjpy_bb_score",
            "usdjpy_cum_divergence_1h",
        ]
    )

    static_cols: List[str] = field(
        default_factory=lambda: [
            "day_of_week_sin",
            "day_of_week_cos",
            "hour_of_day_sin",
            "hour_of_day_cos",
            "minute_of_day_sin",
            "minute_of_day_cos",
            "is_market_open",
            "is_lunch_break",
            "is_closing_auction",
            "minutes_from_open",
            "is_high_vol_window",
            "tod_sin",
            "tod_cos",
            "is_day_session",
            "is_night_session",
            "is_night_open",
            "is_night_late",
        ]
    )

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    adx_period: int = 14


@dataclass(frozen=True)
class ModelConfig:
    """Temporal Fusion Transformer モデル構造設定。

    Attributes:
        d_model (int): モデルの隠れ層の次元数。
        hidden_size (int): 内部フィードフォワード層のサイズ。
        n_heads (int): アテンションヘッド数。
        dropout (float): ドロップアウト率。
        num_layers (int): Transformer層の数。
        num_classes (int): 分類クラス数(Up/Down/Neutral)。
        num_trade_classes (int): 取引有無クラス数。
        num_dir_classes (int): トレンド方向クラス数。
    """

    d_model: int = 64
    hidden_size: int = 64
    n_heads: int = 2
    dropout: float = 0.50  # 過学習(Overfitting)対策でさらに強化

    num_layers: int = 2

    num_classes: int = 3
    num_trade_classes: int = 2
    num_dir_classes: int = 2


@dataclass(frozen=True)
class TrainConfig:
    """学習プロセスおよび損失関数設定。"""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-3  # L2正則化を強める

    batch_size: int = 256
    epochs: int = 30
    patience: int = 3  # Early Stoppingを早くして過剰なカーブフィットを防ぐ
    early_stopping_min_delta: float = 1e-4

    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-7

    use_ema: bool = True
    ema_decay: float = 0.999
    feature_noise_std: float = 0.05

    use_temperature_scaling: bool = False
    temperature_max_iter: int = 200

    grad_clip: float = 1.0
    label_smoothing: float = 0.05

    # Loss Weights
    trade_loss_weight: float = 5.0  # 分類精度(エントリーすべきか否か)をより重視
    dir_loss_weight: float = 10.0  # 方向予測を最重視

    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    directional_penalty: float = 1.0

    dir_conf_margin: float = 0.55
    false_action_penalty: float = 0.00
    false_action_margin: float = 0.15

    neutral_logit_margin: float = 1.0
    neutral_logit_penalty: float = 0.30

    pnl_loss_weight: float = (
        0.10  # PnL Lossによる未来情報の過剰学習を防ぐため比重を下げる
    )

    action_logit_margin: float = 0.75
    action_logit_penalty: float = 0.10

    use_amp: bool = True

    # DataLoader
    num_workers: int = 2
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory: bool = True

    # Class Weighting
    class_weight_power: float = 1.0
    class_weight_cap: float = 8.0
    class_weight_floor: float = 1.0
    trade_pos_weight_scale: float = 5.0
    trade_neg_weight_scale: float = 3.50

    auto_tune_label: bool = True
    min_neutral_ratio: float = 0.60
    max_neutral_ratio: float = 0.85

    use_triple_barrier: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    """バックテストおよびシグナルフィルタ設定。

    Attributes:
        cost (float): 片道手数料+スリッページ。
        use_next_bar_entry (bool): 次足始値エントリーの有効化。
        use_vol_regime_scaling (bool): レジームによる閾値の動的変更。
        vol_scaling_intensity (float): 閑散時にどれだけ閾値を厳しくするか。
    """

    cost: float = 15.0
    use_next_bar_entry: bool = True
    slippage_tick: float = 1.0

    enforce_tp_min_after_cost: bool = True
    tp_min_after_cost: float = 10.0

    min_holding_sec: int = 2 * 60  # 最低ホールド時間を5分から2分へ短縮
    max_holding_sec: int = 2 * 60 * 60  # 最大ホールド時間を4時間から2時間へ短縮
    cooldown_bars: int = 30  # predict_horizonに合わせて30バーへ変更

    sl_price: float = 100.0
    tp_price: float = 0.0

    use_dynamic_sl_tp: bool = True
    use_take_profit: bool = True  # 微益撤退を防ぎつつ確実に利益を確保するためTPを復活
    tp_min_atr_mult: float = 2.0  # ATRの2倍を最低利確ラインに設定

    default_m_sl: float = 2.0
    default_m_tp: float = 2.0

    use_trailing_stop: bool = True
    trailing_act_mult: float = 2.0  # トレイリング開始を遅らせる(ノイズでの狩られ対策)
    trailing_drop_mult: float = 2.5  # 許容する戻り幅を広くする

    adx_filter: float = 0.0
    atr_threshold: float = 15.0
    adx_quantile_floor: float = 0.0

    min_tick_speed_ratio: float = 1.2
    use_edge_threshold: bool = False
    edge_threshold: float = 0.02
    edge_margin: float = 0.01

    prob_threshold: float = 0.505

    # 2段階しきい値
    threshold_trade: float = 0.505
    threshold_dir: float = 0.55

    # --- Dynamic Thresholding ---
    use_vol_regime_scaling: bool = True  # ★追加: レジームによる閾値の動的変更
    vol_scaling_intensity: float = (
        0.2  # ★追加: 閑散時にどれだけ閾値を厳しくするか(0.0~1.0)
    )

    # Tuning
    max_signal_rate_for_tuning: float = 0.50
    auto_tune_threshold: bool = True
    min_trades_for_tuning: int = 8
    tune_metric: str = "pnl_dd"

    use_conditional_signals: bool = True
    min_dir_prob: float = 0.05
    min_trades_frac: float = 0.001
    min_trades_floor: int = 3

    min_pf_to_trade: float = 1.10
    no_trade_threshold: float = 1.05

    horizon_hold_mult: float = 3.0
    target_trades: int = 40
    max_trades: int = 200
    pf_cap: float = 6.0

    avoid_lunch_break: bool = True

    # Position Sizing
    contract_multiplier: float = 10.0
    base_lots: int = 1
    max_lots: int = 3
    confidence_scale: float = 0.05


@dataclass(frozen=True)
class GlobalConfig:
    """システム全体の統合設定。"""

    output_dir: str = datetime.now().strftime("%Y%m%d-%H%M")

    n_folds: int = 20 * 12 * 7
    train_days: int = 30
    val_days: int = 5
    test_days: int = 1
    step_days: int = 1

    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


# インスタンス化
cfg = GlobalConfig()
