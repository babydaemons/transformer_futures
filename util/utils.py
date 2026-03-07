# tester/core/utils.py
from datetime import datetime
import os
import time
import json
import logging
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from scipy.special import erfinv

# ============================================================
# 0. Logging Setup
# ============================================================
TRACE_LEVEL_NUM = 5
PERF_LEVEL_NUM = 6

logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.addLevelName(PERF_LEVEL_NUM, "PERF")

setattr(logging, "TRACE", TRACE_LEVEL_NUM)
setattr(logging, "PERF", PERF_LEVEL_NUM)

def trace_method(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws, stacklevel=2)

def perf_method(self, message, *args, **kws):
    if self.isEnabledFor(PERF_LEVEL_NUM):
        self._log(PERF_LEVEL_NUM, message, args, **kws, stacklevel=2)

logging.Logger.trace = trace_method
logging.Logger.perf = perf_method

def trace_module_func(msg, *args, **kwargs):
    logging.log(TRACE_LEVEL_NUM, msg, *args, **kwargs)

def perf_module_func(msg, *args, **kwargs):
    logging.log(PERF_LEVEL_NUM, msg, *args, **kwargs)

logging.trace = trace_module_func
logging.perf = perf_module_func

# --- Perf settings (env) ---
_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
_PERF_CUDA_SYNC = int(os.getenv("PERF_CUDA_SYNC", "1")) != 0
_PERF_LOG_JSON = int(os.getenv("PERF_LOG_JSON", "0")) != 0
_PERF_STEP_SAMPLE = int(os.getenv("PERF_STEP_SAMPLE", "200"))
_PERF_EVAL_SAMPLE = int(os.getenv("PERF_EVAL_SAMPLE", "50"))
_PERF_INFER_SAMPLE = int(os.getenv("PERF_INFER_SAMPLE", "50"))
_PERF_GPU_STATS = int(os.getenv("PERF_GPU_STATS", "1")) != 0

def setup_logging():
    """メインプロセスでのみ呼び出すログ設定"""
    log_filename = datetime.now().strftime("%Y%m%d-%H%M.log")
    logging.basicConfig(
        level=_LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )


def _perf_sync_if(do_sync: bool):
    """perf計測用のCUDA同期。

    NOTE:
    - 推論ループ内で毎回 synchronize() すると大きく遅くなるため、サンプリング時のみ同期する。
    """
    if do_sync and _PERF_CUDA_SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()

class PerfTimer:
    """perf() レベルで区間計測ログを出す軽量タイマー。

    - PERF_CUDA_SYNC=1 の場合、GPU計測のために前後で cuda synchronize します。
    - PERF_LOG_JSON=1 の場合、extras を JSON として併記します。
    """
    def __init__(self, logger: logging.Logger, name: str, extras: Optional[dict] = None, do_sync: bool = True):
        self.logger = logger
        self.name = name
        self.extras = extras or {}
        self.do_sync = bool(do_sync)
        self.t0 = 0.0

    def __enter__(self):
        _perf_sync_if(self.do_sync)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _perf_sync_if(self.do_sync)
        dt = (time.perf_counter() - self.t0) * 1000.0
        if _PERF_LOG_JSON and self.extras:
            try:
                payload = json.dumps(self.extras, ensure_ascii=False, default=str)
            except Exception:
                payload = str(self.extras)
            self.logger.perf(f"[perf] {self.name}: {dt:.3f} ms | {payload}")
        else:
            if self.extras:
                self.logger.perf(f"[perf] {self.name}: {dt:.3f} ms | {self.extras}")
            else:
                self.logger.perf(f"[perf] {self.name}: {dt:.3f} ms")
        return False

# --- RankGaussScaler ---
class RankGaussScaler:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.mapping = []

    def fit(self, X):
        self.mapping = []
        N = X.shape[0]
        quantiles = np.linspace(0, 1, N)
        quantiles = np.clip(quantiles, self.epsilon, 1 - self.epsilon)
        gauss_vals = np.sqrt(2) * erfinv(2 * quantiles - 1)
        for i in range(X.shape[1]):
            col = X[:, i]
            sorted_col = np.sort(col)
            self.mapping.append((sorted_col, gauss_vals))
        return self

    def transform(self, X):
        X_new = np.zeros_like(X)
        for i in range(X.shape[1]):
            src, dst = self.mapping[i]
            X_new[:, i] = np.interp(X[:, i], src, dst)
        return X_new

# --- EMA ---
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}