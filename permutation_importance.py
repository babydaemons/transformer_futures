# permutation_importance.py
"""
File: permutation_importance.py

ソースコードの役割:
本スクリプトは、学習済みのモデルと検証(Validation)データを用いて、
Permutation Feature Importance (シャッフル法) による特徴量の寄与度を計算します。
各特徴量の列をランダムにシャッフルし、ベースラインのLossからどれだけ悪化するかを測定することで、
「どの特徴量が予測に不可欠か」および「どの特徴量がノイズになっているか」を機械的に抽出します。
"""

import os
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

# プロジェクト内の必要なモジュールをインポート
from config import GlobalConfig
from core.loss_calculator import LossCalculator
from core.losses import FocalLoss
from model.tft import TemporalFusionTransformer

# データローダーとパイプライン関連のインポート
from data.data_loader import MarketDataLoader
from features.pipeline import FeaturePipeline
from data.dataset_builder import DatasetBuilder
from data.dataset import WalkForwardSplit
from data.builder import build_fold_dataloaders

# ロガーの初期化
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_permutation_importance(
    model: torch.nn.Module,
    loader_val: torch.utils.data.DataLoader,
    device: torch.device,
    loss_calc: LossCalculator,
    criterion_trade: torch.nn.Module,
    criterion_dir: torch.nn.Module,
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """Validationデータセットを用いてPermutation Importanceを計算する。

    各特徴量をランダムにシャッフルすることで特徴量とターゲットの関係性を破壊し、
    ベースラインのLossからどれだけ悪化するかを計測することで重要度を評価する。

    Args:
        model (torch.nn.Module): 評価対象の学習済みモデル。
        loader_val (DataLoader): 検証用データローダー。
        device (torch.device): 実行デバイス (CPU/GPU)。
        loss_calc (LossCalculator): 損失計算オーケストレーター。
        criterion_trade (torch.nn.Module): トレード用の分類Loss。
        criterion_dir (torch.nn.Module): 方向用の分類Loss。
        feature_names (List[str]): 評価する特徴量の名前リスト。

    Returns:
        List[Tuple[str, float]]: 特徴量名とその重要度(Lossの悪化幅)のリスト。降順ソート済み。
    """
    model.eval()

    # 1. ベースラインLossの計算
    logger.info("Calculating baseline loss...")
    base_loss_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader_val:
            # LossCalculatorがデバイス転送も内部で行います
            loss = loss_calc.compute_batch_loss(
                model, batch, False, criterion_trade, criterion_dir
            )

            y = batch[2]
            b_size = y.size(0)
            base_loss_sum += loss.item() * b_size
            total_samples += b_size

    base_loss = base_loss_sum / total_samples
    logger.info(f"Baseline Loss: {base_loss:.4f}")

    # 2. 各特徴量をシャッフルしてLossの悪化を測定
    importances = {}

    # 最後のバッチのテンソル形状から特徴量の数を取得
    num_features = batch[0].size(2)  # 連続特徴量の数

    # 特徴量名リストの長さチェック
    if len(feature_names) != num_features:
        logger.warning(
            f"Feature names list length ({len(feature_names)}) does not match "
            f"model input features ({num_features}). Using generic names."
        )
        feature_names = [f"Feature_{i}" for i in range(num_features)]

    logger.info(f"Starting permutation importance for {num_features} features...")

    for feat_idx, feat_name in enumerate(feature_names):
        shuffled_loss_sum = 0.0

        with torch.no_grad():
            for batch in loader_val:
                # シャッフル対象のテンソル(CPU上)
                x_num = batch[0]

                # 特徴量列のオリジナルデータを退避
                orig_col = x_num[:, :, feat_idx].clone()

                # バッチ内でその特徴量だけをランダムシャッフルして関係性を破壊
                perm_idx = torch.randperm(x_num.size(0))
                x_num[:, :, feat_idx] = x_num[perm_idx, :, feat_idx]

                # LossCalculatorを使って計算 (batch内のテンソルはインプレースで書き換えられているためそのまま渡す)
                loss = loss_calc.compute_batch_loss(
                    model, batch, False, criterion_trade, criterion_dir
                )

                y = batch[2]
                b_size = y.size(0)
                shuffled_loss_sum += loss.item() * b_size

                # 次のバッチの処理に影響が出ないようオリジナルに戻す
                x_num[:, :, feat_idx] = orig_col

        shuffled_loss = shuffled_loss_sum / total_samples

        # 悪化度合い (Lossの増加分) が重要度
        importance = shuffled_loss - base_loss
        importances[feat_name] = importance

        logger.info(
            f"Evaluated {feat_name}: shuffled_loss={shuffled_loss:.4f}, importance={importance:.4f}"
        )

    # 3. 寄与度の高い順（Loss悪化幅が大きい順）にソートして返す
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_importances


def main() -> None:
    """Permutation Feature Importanceの実行エントリーポイント。

    モデルの推論環境を構築し、Walk-Forward Splitで切り出した
    検証データを用いて各特徴量の重要度を算出・出力する。
    """
    # 設定の読み込みとデバイスの決定
    cfg = GlobalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Building data loaders...")

    # 必要なコンポーネントの初期化
    data_loader = MarketDataLoader(cfg)
    pipeline = FeaturePipeline(cfg)
    ds_builder = DatasetBuilder()

    # Fold 0 の日付分割を取得
    dates = data_loader.get_trading_dates()
    splitter = WalkForwardSplit(
        cfg.train_days, cfg.val_days, cfg.test_days, cfg.step_days
    )
    splits = list(splitter.split(dates))
    train_dates, val_dates, test_dates = splits[0]

    # データローダーの構築
    loader_train, loader_val, loader_test, y_labels_tr = build_fold_dataloaders(
        data_loader=data_loader,
        pipeline=pipeline,
        ds_builder=ds_builder,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        cfg=cfg,
        logger=logger,
        fold=0,
        device_type=device.type,
    )

    logger.info("Initializing model and loading best weights...")

    # train.py と同じ方法でモデルを初期化するための変数を準備
    atr_idx = (
        cfg.features.continuous_cols.index("atr")
        if "atr" in cfg.features.continuous_cols
        else -1
    )

    # コスト計算 (15円の往復コスト + スリッページ)
    total_cost = float(cfg.backtest.cost) + float(cfg.backtest.slippage_tick) * 5.0

    # TPのフロア価格（最低限確保したい利益幅）の計算
    tp_floor_price = (
        total_cost + float(cfg.backtest.tp_min_after_cost)
        if cfg.backtest.enforce_tp_min_after_cost
        else 0.0
    )

    # モデルのインスタンス化
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

    # ベストモデルの重みをロード (パスは実際の出力ディレクトリに合わせて変更してください)
    best_model_path = os.path.join("20260315-0847", "best_model_fold0.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"Loaded weights from {best_model_path}")
    else:
        logger.error(
            f"Best model not found at {best_model_path}. Please check the path."
        )
        return

    # LossCalculator と評価用のLoss関数群の準備
    loss_calc = LossCalculator(cfg, device, logger)
    weight_trade, weight_dir = loss_calc.calculate_class_weights(y_labels_tr)

    # configの指定に従ってLoss関数を選択
    if cfg.train.use_focal_loss:
        criterion_trade = FocalLoss(alpha=weight_trade, gamma=cfg.train.focal_gamma).to(
            device
        )
        criterion_dir = FocalLoss(alpha=weight_dir, gamma=cfg.train.focal_gamma).to(
            device
        )
    else:
        criterion_trade = nn.CrossEntropyLoss(weight=weight_trade).to(device)
        criterion_dir = nn.CrossEntropyLoss(weight=weight_dir).to(device)

    # config.py から入力特徴量（連続値シーケンス）のリストを動的に取得します
    # ※ TFTモデルの連続値入力 (x_num) に対応する特徴量名群です
    feature_names = cfg.features.continuous_cols

    # 実行
    results = calculate_permutation_importance(
        model,
        loader_val,
        device,
        loss_calc,
        criterion_trade,
        criterion_dir,
        feature_names,
    )

    # 結果の標準出力への表示
    print("\n" + "=" * 50)
    print("Permutation Feature Importance Results")
    print("=" * 50)
    print("※ 値が大きいほど重要 (シャッフルするとLossが悪化する)")
    print(
        "※ 値がマイナスの場合は、その特徴量が無い方がLossが改善する(＝有害なノイズ)ことを意味します\n"
    )

    for rank, (name, imp) in enumerate(results, 1):
        print(f"{rank:2d}. {name:30s} : {imp:+.6f}")

    print("=" * 50)


if __name__ == "__main__":
    main()
