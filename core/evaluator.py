# core/evaluator.py
"""
File: core/evaluator.py

ソースコードの役割:
本モジュールは、モデルの検証(Validation)およびOut-of-Sample(OOS)テストの評価プロセスを担当します。
単一責任の原則(SRP)に基づき、以下のクラス群によって構成されています。
1. ValidationEngine: モデルの評価とLoss算出・バックテストシミュレーションの実行。
2. ModelCheckpointManager: スコア・Lossに基づくベストモデルの追跡、状態保存、およびEarly Stoppingの判定。
3. OutofSampleRunner: OOSテストの実行と、トレード未発生時のフォールバック処理の管理。
各クラス間は `ValidationResult` データクラスを通じて疎結合に連携します。
"""

import math
import copy
import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from util.utils import PerfTimer
from trade.trading import run_vectorized_backtest
from core.fallback_strategy import resolve_oos_fallback


@dataclass
class ValidationResult:
    """
    検証(Validation)フェーズの評価結果を保持するデータクラス。

    Attributes:
        val_loss (float): 検証データにおける平均損失。
        val_score (float): バックテストによるスコア。
        n_trades (int): バックテストでの総取引回数。
        threshold_trade (Optional[float]): 取引実行の最適閾値。
        threshold_dir (Optional[float]): 取引方向の最適閾値。
        raw_backtest_res (Optional[Dict[str, Any]]): バックテストの生の実行結果。
    """

    val_loss: float
    val_score: float
    n_trades: int
    threshold_trade: Optional[float] = None
    threshold_dir: Optional[float] = None
    raw_backtest_res: Optional[Dict[str, Any]] = None


class ValidationEngine:
    """
    モデルの評価、Loss算出、および検証データでのバックテストを実行するクラス。

    Attributes:
        model (nn.Module): 評価対象のモデル。
        cfg (Any): 設定オブジェクト。
        device (torch.device): 実行デバイス。
        logger (logging.Logger): ログ出力用インスタンス。
        compute_loss_fn (Callable): 損失を計算するコールバック関数。
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        device: torch.device,
        logger: logging.Logger,
        compute_loss_fn: Callable,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.compute_loss_fn = compute_loss_fn

    def evaluate_loss(
        self, loader: DataLoader, criterion_trade: nn.Module, criterion_dir: nn.Module
    ) -> float:
        """
        検証用データでのLoss計算を行います。

        Args:
            loader (DataLoader): 検証用データローダー。
            criterion_trade (nn.Module): 取引有無のLoss関数。
            criterion_dir (nn.Module): 取引方向のLoss関数。

        Returns:
            float: 平均損失値。
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

    def run_validation(
        self,
        epoch: int,
        fold_idx: int,
        loader_val: DataLoader,
        criterion_trade: nn.Module,
        criterion_dir: nn.Module,
        ema: Optional[Any] = None,
    ) -> ValidationResult:
        """
        Lossの評価とバックテストを実行し、結果を返します。

        Args:
            epoch (int): 現在のエポック数。
            fold_idx (int): 現在のFold番号。
            loader_val (DataLoader): 検証用データローダー。
            criterion_trade (nn.Module): 取引有無のLoss関数。
            criterion_dir (nn.Module): 取引方向のLoss関数。
            ema (Optional[Any]): EMAオブジェクト（適用・復元用）。

        Returns:
            ValidationResult: 評価とバックテストの結果。
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
        n_trades = int(backtest_res.get("n_trades", 0)) if backtest_res else 0
        th_trade = (
            float(backtest_res.get("threshold_trade"))
            if backtest_res and backtest_res.get("threshold_trade") is not None
            else None
        )
        th_dir = (
            float(backtest_res.get("threshold_dir"))
            if backtest_res and backtest_res.get("threshold_dir") is not None
            else None
        )

        self.logger.info(
            f"Fold {fold_idx} Ep {epoch}: ValLoss={val_loss:.4f} | ValScore={val_score:.4f} (Trades: {n_trades})"
        )

        return ValidationResult(
            val_loss=val_loss,
            val_score=val_score,
            n_trades=n_trades,
            threshold_trade=th_trade,
            threshold_dir=th_dir,
            raw_backtest_res=backtest_res,
        )


class ModelCheckpointManager:
    """
    検証結果に基づき、ベストモデルの保持およびEarly Stoppingの判定を管理するクラス。

    Attributes:
        cfg (Any): 設定オブジェクト。
        logger (logging.Logger): ログ出力用インスタンス。
        best_val_loss (float): 記録された最小の検証損失。
        best_val_score (float): 記録された最大のバックテストスコア。
        best_model_state (Optional[Dict]): 最良スコア時のモデル重み。
        bad_epochs (int): スコアが改善しない連続エポック数。
    """

    def __init__(self, cfg: Any, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.best_val_loss = float("inf")
        self.best_val_score = -float("inf")
        self.best_model_state = None
        self.bad_epochs = 0

    def update_and_check_early_stopping(
        self,
        model: nn.Module,
        val_result: ValidationResult,
        epoch: int,
        ema: Optional[Any] = None,
    ) -> bool:
        """
        新しい評価結果を用いてベストモデルの更新判定を行い、Early Stoppingの必要性をチェックします。

        Args:
            model (nn.Module): 対象のモデル。
            val_result (ValidationResult): 現在のエポックの評価結果。
            epoch (int): 現在のエポック数。
            ema (Optional[Any]): EMAオブジェクト（適用・復元用）。

        Returns:
            bool: Early Stopping条件に達した場合は True、それ以外は False。
        """
        val_score = val_result.val_score
        val_loss = val_result.val_loss

        # スコアの改善判定
        score_improved = val_score > self.best_val_score + 1e-4
        score_equal = False

        if math.isfinite(val_score) and math.isfinite(self.best_val_score):
            score_equal = abs(val_score - self.best_val_score) < 1e-4
        elif math.isinf(val_score) and math.isinf(self.best_val_score):
            score_equal = val_score == self.best_val_score

        # スコアが同等なら、Lossが基準値以上に改善しているかをチェック
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
                ema.apply_shadow(model)
                self.best_model_state = copy.deepcopy(model.state_dict())
                ema.restore(model)
            else:
                self.best_model_state = copy.deepcopy(model.state_dict())

            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            # 一定エポック経過後、改善が見られなければEarly Stopping
            if self.bad_epochs >= self.cfg.train.patience and epoch > 5:
                self.logger.info(
                    f"Early stopping at epoch {epoch} (Best Score: {self.best_val_score:.4f})"
                )
                return True

        return False

    def restore_best_model(self, model: nn.Module) -> bool:
        """
        保持しているベストモデルの重みを指定されたモデルに読み込みます。

        Args:
            model (nn.Module): 重みを復元する対象のモデル。

        Returns:
            bool: 復元に成功した場合は True、保存された状態がない場合は False。
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            return True
        return False


class OutofSampleRunner:
    """
    学習完了後のベストモデルを用いて、Out-of-Sample（OOS）テストと
    フォールバック検証の実行を管理するクラス。

    Attributes:
        model (nn.Module): 評価対象のモデル。
        cfg (Any): 設定オブジェクト。
        device (torch.device): 実行デバイス。
        logger (logging.Logger): ログ出力用インスタンス。
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        device: torch.device,
        logger: logging.Logger,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger

    def run_oos_test(
        self,
        fold_idx: int,
        loader_val: DataLoader,
        loader_test: Optional[DataLoader],
        test_dates: List[Any],
        checkpoint_manager: ModelCheckpointManager,
    ) -> List[Dict[str, Any]]:
        """
        ベストモデルを復元し、OOSテストを実行します。
        OOS期間でトレードが発生しなかった場合は、閾値を緩和してフォールバック検証を行います。

        Args:
            fold_idx (int): 現在のFold番号。
            loader_val (DataLoader): 検証用データローダー。
            loader_test (Optional[DataLoader]): テスト用データローダー。
            test_dates (List[Any]): テスト期間の日付リスト。
            checkpoint_manager (ModelCheckpointManager): ベストモデルを保持しているマネージャー。

        Returns:
            List[Dict[str, Any]]: OOS期間でのトレード履歴のリスト。
        """
        oos_trades = []

        # モデル状態の復元
        if not checkpoint_manager.restore_best_model(self.model):
            self.logger.warning("No best model state found to restore for OOS test.")
            return oos_trades

        # Validationデータで再度推論し、最適化された最良の閾値等を取得
        best_val_res = run_vectorized_backtest(
            self.model, loader_val, self.device, self.cfg, fold_idx
        )
        self.logger.info(
            f"Restored best model (Val Result) with Score={checkpoint_manager.best_val_score:.4f}, "
            f"Loss={checkpoint_manager.best_val_loss:.4f}: {best_val_res}"
        )

        # OOSデータに対するテスト実行
        if loader_test is not None and len(test_dates) > 0:
            val_th_trade = best_val_res.get("threshold_trade")
            val_th_dir = best_val_res.get("threshold_dir")
            trade_log_path = os.path.join(
                self.cfg.output_dir, f"fold{fold_idx:04d}_test_{test_dates[0]}.tsv"
            )

            val_th_trade_f = float(val_th_trade) if val_th_trade is not None else 0.50
            val_th_dir_f = float(val_th_dir) if val_th_dir is not None else 0.50

            self.logger.info(
                "Running OOS test with restored val thresholds: threshold_trade=%.3f, threshold_dir=%.3f",
                val_th_trade_f,
                val_th_dir_f,
            )

            best_oos = run_vectorized_backtest(
                self.model,
                loader_test,
                self.device,
                self.cfg,
                fold_idx,
                fixed_threshold_trade=val_th_trade_f,
                fixed_threshold_dir=val_th_dir_f,
                trade_log_path=trade_log_path,
            )

            self.logger.info(
                "Initial OOS result: n_trades=%d, threshold_trade=%.3f, threshold_dir=%.3f",
                int(best_oos.get("n_trades", 0)),
                float(best_oos.get("threshold_trade", val_th_trade_f)),
                float(best_oos.get("threshold_dir", val_th_dir_f)),
            )

            # OOSでトレードが発生しなかった場合のフォールバックロジック
            if best_oos.get("n_trades", 0) == 0:
                self.logger.warning(
                    "OOS fallback triggered: no trades with restored val thresholds."
                )

                min_fallback_trades = 3

                fallback_res = resolve_oos_fallback(
                    model=self.model,
                    loader_test=loader_test,
                    device=self.device,
                    cfg=self.cfg,
                    fold_idx=fold_idx,
                    initial_trade_th=val_th_trade_f,
                    initial_dir_th=val_th_dir_f,
                    trade_log_path=trade_log_path,
                    logger=self.logger,
                )

                if fallback_res is not None:
                    candidate_n_trades = int(fallback_res.get("n_trades", 0))
                    reason = fallback_res.get("fallback_reason", "resolved_by_module")

                    if candidate_n_trades >= min_fallback_trades:
                        self.logger.info(
                            "OOS fallback adopted: %s (n_trades=%d)",
                            reason,
                            candidate_n_trades,
                        )
                        best_oos = fallback_res
                    elif candidate_n_trades > 0:
                        self.logger.info(
                            "OOS fallback rejected: %s (n_trades=%d < min_fallback_trades=%d)",
                            reason,
                            candidate_n_trades,
                            min_fallback_trades,
                        )
                    else:
                        self.logger.info(
                            "OOS fallback abandoned: no candidate satisfied adoption criteria."
                        )
                else:
                    self.logger.info(
                        "OOS fallback abandoned: no candidate satisfied adoption criteria."
                    )

            if "trades" in best_oos:
                oos_trades = best_oos["trades"]
            self.logger.info(f"OOS Test Result ({test_dates[0]}): {best_oos}")

        return oos_trades
