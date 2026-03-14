# core/evaluator.py
"""
File: core/evaluator.py

ソースコードの役割:
本モジュールは、モデルの検証(Validation)およびOut-of-Sample(OOS)テストの評価プロセスを担当する `Evaluator` クラスを提供します。
Lossの計算や、バックテストシミュレーションを通じたスコアの算出、ベストモデルの追跡・管理を行います。
`Trainer` クラスから評価ロジックを分離し、各クラスの責務を明確化することで可読性と保守性を高めます。
"""

import math
import copy
import logging
import os
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from util.utils import PerfTimer
from trade import run_vectorized_backtest


class Evaluator:
    """
    モデルの評価、バックテストの実行、およびベストスコア・ベストモデルの管理を行うクラス。
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        device: torch.device,
        logger: logging.Logger,
        compute_loss_fn: callable,
    ):
        """
        Evaluatorの初期化。

        Args:
            model (nn.Module): 評価対象のPyTorchモデル
            cfg (Any): 設定オブジェクト
            device (torch.device): 実行デバイス (CPU/GPU)
            logger (logging.Logger): ロガーインスタンス
            compute_loss_fn (callable): 損失を計算するためのコールバック関数（Trainerから渡される）
        """
        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.compute_loss_fn = compute_loss_fn

        # 評価状態の追跡
        self.best_val_loss = float("inf")
        self.best_val_score = -float("inf")
        self.best_model_state = None
        self.bad_epochs = 0

    def evaluate_loss(
        self, loader: DataLoader, criterion_trade: nn.Module, criterion_dir: nn.Module
    ) -> float:
        """
        検証用データでのLoss計算を行います。

        Args:
            loader (DataLoader): 検証用データローダー
            criterion_trade (nn.Module): 取引有無のLoss関数
            criterion_dir (nn.Module): 取引方向のLoss関数

        Returns:
            float: 平均損失値
        """
        self.model.eval()
        total_sum = None
        n = 0

        with torch.no_grad():
            for batch in loader:
                loss = self.compute_loss_fn(
                    batch=batch,
                    is_train=False,
                    criterion_trade=criterion_trade,
                    criterion_dir=criterion_dir,
                )
                b = batch[2].size(0)
                if total_sum is None:
                    total_sum = loss.detach().float() * b
                else:
                    total_sum += loss.detach().float() * b
                n += b

        return (total_sum / n).item() if n > 0 else 0.0

    def evaluate_and_track_best(
        self,
        epoch: int,
        fold_idx: int,
        loader_val: DataLoader,
        criterion_trade: nn.Module,
        criterion_dir: nn.Module,
        ema: Optional[Any] = None,
    ) -> bool:
        """
        検証セットの評価とバックテストを実行し、ベストモデルの更新とEarly Stoppingの判定を行います。

        Args:
            epoch (int): 現在のエポック数
            fold_idx (int): 現在のFold番号
            loader_val (DataLoader): 検証用データローダー
            criterion_trade (nn.Module): 取引有無のLoss関数
            criterion_dir (nn.Module): 取引方向のLoss関数
            ema (Optional[Any]): EMAオブジェクト（適用・復元用）

        Returns:
            bool: Early Stopping条件に達した場合は True、それ以外は False を返します。
        """
        if ema is not None:
            ema.apply_shadow(self.model)

        val_loss = self.evaluate_loss(loader_val, criterion_trade, criterion_dir)

        # 高速化されたベクトル化バックテストを実行してスコアを取得
        with PerfTimer(self.logger, f"fold{fold_idx}/epoch{epoch}/val_backtest"):
            backtest_res = run_vectorized_backtest(
                self.model, loader_val, self.device, self.cfg, fold_idx=fold_idx
            )

        if ema is not None:
            ema.restore(self.model)

        val_score = (
            float(backtest_res.get("score", -float("inf")))
            if backtest_res
            else -float("inf")
        )
        n_trades_val = int(backtest_res.get("n_trades", 0)) if backtest_res else 0

        self.logger.info(
            f"Fold {fold_idx} Ep {epoch}: ValLoss={val_loss:.4f} | ValScore={val_score:.4f} (Trades: {n_trades_val})"
        )

        # スコアの改善判定
        score_improved = val_score > self.best_val_score + 1e-4
        score_equal = False

        if math.isfinite(val_score) and math.isfinite(self.best_val_score):
            score_equal = abs(val_score - self.best_val_score) < 1e-4
        elif math.isinf(val_score) and math.isinf(self.best_val_score):
            score_equal = val_score == self.best_val_score

        # スコアが同等ならLossが大きく改善しているかをチェック
        loss_improved_at_same_score = (
            score_equal
            and val_loss < self.best_val_loss - self.cfg.train.early_stopping_min_delta
        )

        if score_improved or loss_improved_at_same_score:
            if score_improved:
                self.best_val_score = val_score
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # ベストモデルの状態を保存
            if ema is not None:
                ema.apply_shadow(self.model)
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                ema.restore(self.model)
            else:
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.cfg.train.patience and epoch > 5:
                self.logger.info(
                    f"Early stopping at epoch {epoch} (Best Score: {self.best_val_score:.4f})"
                )
                return True

        return False

    def run_oos_test(
        self,
        fold_idx: int,
        loader_val: DataLoader,
        loader_test: Optional[DataLoader],
        test_dates: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        学習完了後にベストモデルを復元し、Out-of-Sample（OOS）テストを実行します。

        Args:
            fold_idx (int): 現在のFold番号
            loader_val (DataLoader): 検証用データローダー
            loader_test (Optional[DataLoader]): テスト用データローダー
            test_dates (List[Any]): テスト期間の日付リスト

        Returns:
            List[Dict[str, Any]]: OOS期間でのトレード履歴のリスト
        """
        oos_trades = []
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

            # Validationデータで再度推論し、最適化された最良の閾値等を取得
            best_val_res = run_vectorized_backtest(
                self.model, loader_val, self.device, self.cfg, fold_idx
            )
            self.logger.info(
                f"Restored best model (Val Result) with Score={self.best_val_score:.4f}, Loss={self.best_val_loss:.4f}: {best_val_res}"
            )

            # OOSデータに対するテスト実行
            if loader_test is not None and len(test_dates) > 0:
                val_th_trade = best_val_res.get("threshold_trade")
                val_th_dir = best_val_res.get("threshold_dir")
                trade_log_path = os.path.join(
                    self.cfg.output_dir, f"fold{fold_idx:04d}_test_{test_dates[0]}.tsv"
                )

                best_oos = run_vectorized_backtest(
                    self.model,
                    loader_test,
                    self.device,
                    self.cfg,
                    fold_idx,
                    fixed_threshold_trade=val_th_trade,
                    fixed_threshold_dir=val_th_dir,
                    trade_log_path=trade_log_path,
                )

                if "trades" in best_oos:
                    oos_trades = best_oos["trades"]
                self.logger.info(f"OOS Test Result ({test_dates[0]}): {best_oos}")

        return oos_trades
