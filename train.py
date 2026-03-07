# train.py
"""
File: train.py

ソースコードの役割:
本モジュールは、日経平均先物ミニのモデル学習メインエントリーポイントです。
引数処理、データセット全体の分割（Walk Forward/Target Year）、各Foldでのモジュール呼び出し、および最終結果の集計を行います。
"""

import random
import logging
import gc
import os
import argparse
from typing import Optional, List, Tuple

import torch
import numpy as np
import datetime

from config import cfg
from model.tft import TemporalFusionTransformer
from features.pipeline import FeaturePipeline
from data.data_loader import MarketDataLoader
from data.dataset_builder import DatasetBuilder
from util.utils import setup_logging, PerfTimer
from data.dataset import WalkForwardSplit
from data.builder import build_fold_dataloaders
from core.trainer import Trainer


def set_seed(seed: int = 42) -> None:
    """乱数シードの固定化を行います。

    再現性を確保するため、Python、NumPy、およびPyTorchの各乱数生成器のシードを固定し、
    cuDNNの決定論的振る舞いを有効化します。

    Args:
        seed (int): 固定するシード値。デフォルトは42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _generate_target_year_splits(dates: List[datetime.date], target_year: int) -> List[Tuple[List[datetime.date], List[datetime.date], List[datetime.date]]]:
    """指定された対象年のための Walk-Forward 分割を生成します。

    指定した年に該当する取引日をテストデータとし、そのテストデータの前方を
    検証データ（val）および学習データ（train）として分割したリストを返します。

    Args:
        dates (List[datetime.date]): データセット全体に存在する取引日のリスト。
        target_year (int): テスト対象の年。

    Returns:
        List[Tuple[List[datetime.date], List[datetime.date], List[datetime.date]]]:
            (train_dates, val_dates, test_dates) のタプルのリスト。
    """
    splits = []
    target_dates = [d for d in dates if d.year == target_year]
    if not target_dates:
        logging.error(f"No trading dates found for year {target_year}")
        return splits

    first_test_idx = dates.index(target_dates[0])
    val_days, train_days, test_days, step_days = 5, 30, 1, 1

    if first_test_idx < val_days + train_days:
        logging.error("Not enough historical data before the target test date.")
        return splits

    current_test_idx = first_test_idx
    while (
        current_test_idx < len(dates)
        and dates[current_test_idx].year == target_year
    ):
        val_start_idx = current_test_idx - val_days
        train_start_idx = val_start_idx - train_days
        if train_start_idx < 0:
            break

        test_end_idx = min(current_test_idx + test_days, len(dates))
        actual_test_dates = [
            d for d in dates[current_test_idx:test_end_idx] if d.year == target_year
        ]
        if not actual_test_dates:
            break

        splits.append(
            (
                dates[train_start_idx:val_start_idx],
                dates[val_start_idx:current_test_idx],
                actual_test_dates,
            )
        )
        current_test_idx += step_days

    logging.info(
        f"Target Year Continuous Mode: Generated {len(splits)} rolling splits for {target_year}."
    )
    return splits


def save_trades_to_tsv(trades: list, output_path: str) -> None:
    """トレード履歴をTSVファイルとして保存します。

    Args:
        trades (list): 個々のトレード情報を含む辞書のリスト。
        output_path (str): 保存先のファイルパス。
    """
    if not trades:
        return
    try:
        trades.sort(key=lambda x: x["entry_ts_ns"])

        def _ts_jst_iso(x: int) -> str:
            try:
                dt_utc = datetime.datetime.fromtimestamp(int(x) / 1e9, tz=datetime.timezone.utc)
                dt_jst = dt_utc.astimezone(datetime.timezone(datetime.timedelta(hours=9)))
                return dt_jst.strftime("%Y-%m-%dT%H:%M:%S+09:00")
            except Exception:
                return str(int(x))

        cum_pnl = 0.0
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                "entry_ts_jst\tentry_price\thold_sec\texit_price\tpnl\tdir\treason\tcum_pnl\n"
            )
            for t in trades:
                cum_pnl += t["pnl"]
                f.write(
                    f"{_ts_jst_iso(t['entry_ts_ns'])}\t{t['entry_price']:.1f}\t{t['hold_sec']}\t{t['exit_price']:.1f}\t{t['pnl']:.1f}\t{t['dir']}\t{t['reason']}\t{cum_pnl:.1f}\n"
                )
        logging.info(
            f"Successfully saved all trades to {output_path} (Total trades: {len(trades)})"
        )
    except Exception as e:
        logging.error(f"Failed to save all_trades.tsv: {e}")


def train_main(target_year: Optional[int] = None) -> None:
    """メイン実行処理: Foldの分割から学習ループまでの進行管理を行います。

    Args:
        target_year (Optional[int]): 指定された場合、その年のデータをテスト期間として Walk-Forward 分割を行います。
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    logger = logging.getLogger(__name__)
    
    # 分割された各コンポーネントを初期化
    data_loader = MarketDataLoader()
    pipeline = FeaturePipeline()
    ds_builder = DatasetBuilder()

    logging.info("Scanning dataset for dates...")
    with PerfTimer(logger, "get_trading_dates"):
        dates = data_loader.get_trading_dates()
    logging.info(f"Found {len(dates)} days.")

    # Fold分割ロジックの呼び出し
    if target_year is not None:
        splits = _generate_target_year_splits(dates, target_year)
        if not splits:
            return
    else:
        splitter = WalkForwardSplit(
            cfg.train_days, cfg.val_days, cfg.test_days, cfg.step_days
        )
        splits = list(splitter.split(dates))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    all_oos_trades = []

    # 全Fold実行ループ
    for fold, (train_dates, val_dates, test_dates) in enumerate(splits):
        if fold >= cfg.n_folds:
            break

        logging.info(
            f"=== Fold {fold} === Train: {train_dates[0]}~{train_dates[-1]} | Val: {val_dates[0]}~{val_dates[-1]} | Test: {test_dates[0] if test_dates else 'N/A'}~{test_dates[-1] if test_dates else 'N/A'}"
        )

        # 1. DataLoaderの構築
        loader_train, loader_val, loader_test, y_labels_tr = build_fold_dataloaders(
            data_loader, pipeline, ds_builder, train_dates, val_dates, test_dates, cfg, logger, fold, device.type
        )

        # 2. Modelの初期化
        atr_idx = (
            cfg.features.continuous_cols.index("atr")
            if "atr" in cfg.features.continuous_cols
            else -1
        )
        total_cost = float(cfg.backtest.cost) + float(cfg.backtest.slippage_tick) * 5.0
        tp_floor_price = (
            total_cost + float(cfg.backtest.tp_min_after_cost)
            if cfg.backtest.enforce_tp_min_after_cost
            else 0.0
        )

        model = TemporalFusionTransformer(
            num_continuous=len(cfg.features.continuous_cols),
            num_static=len(cfg.features.static_cols),
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            num_layers=cfg.model.num_layers,
            atr_idx=atr_idx,
            tp_floor_price=tp_floor_price,
            use_tp_floor=True,
        ).to(device)

        # 3. Trainerによる学習の実行
        trainer = Trainer(model, cfg, device, logger)
        oos_trades = trainer.train_fold(
            loader_train, loader_val, loader_test, y_labels_tr, fold, test_dates
        )
        all_oos_trades.extend(oos_trades)

        # 4. クリーンアップ
        with PerfTimer(logger, f"fold{fold}/cleanup"):
            del model, trainer, loader_train, loader_val, loader_test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # === 結果集計 ===
    if all_oos_trades:
        all_trades_path = os.path.join(cfg.output_dir, "all_trades.tsv")
        save_trades_to_tsv(all_oos_trades, all_trades_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NK225-TFT Training")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Target year for out-of-sample test on its first trading day.",
    )
    args = parser.parse_args()

    setup_logging()
    set_seed(42)
    torch.set_float32_matmul_precision("high")

    train_main(target_year=args.year)
