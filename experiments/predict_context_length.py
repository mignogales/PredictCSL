"""
Context-length predictor — dual-objective Patch-Transformer training pipeline.

Pipeline overview
-----------------
Sibling of predict_period.py. Same Patch-Transformer backbone and multi-GPU
random search, but the regression target changes:

    predict_period.py        : scalar log-period of a synthetic series.
    predict_context_length.py: the error-vs-context curve of a TS foundation
                               model — one error value per ablation window.

Given a time series, the model predicts how a TSFM's forecast error varies
with input context length. argmin of the predicted curve is the recommended
("useful") context length. Trained purely on synthetic data labeled offline
by build_context_length_dataset.py, so the predictor is fully zero-shot.

Two simultaneous objectives:

    L = lambda_curve * MSE_curve  +  lambda_recon * MSE_reconstruction

  * MSE_curve : regression of the per-series error-vs-context curve, routed
                from a learnable [CLS] token through an MLP head. The target
                curve is z-scored per series (the decision is argmin, which is
                shift/scale invariant — the model learns curve *shape*).
  * MSE_recon : masked-patch prediction (SimMIM-style) over non-CLS patch
                tokens — an auxiliary representation-learning signal. Disable
                by setting LAMBDA_RECON = 0.

Data
----
Loaded from a build_context_length_dataset.py run directory:
    contexts.npy             (N, CONTEXT_LENGTH)            -- model input
    curves_{mae,mse}.npy     (N, n_windows, n_horizons)     -- regression target
    meta.json                -- window grid, horizon grid, label model
Rows with NaN anywhere in their curve are dropped.

Horizon is a first-class label axis: the useful context length depends on how
far ahead the TSFM is forecasting, so the predictor takes horizon as an extra
input and conditions on it via a learned horizon embedding added to the [CLS]
token. At train time one horizon is sampled uniformly per sample; at eval
time we sweep all horizons and average regret / curve MSE across them.

Multi-GPU execution
-------------------
Process-per-GPU parallelism (preserved from predict_period.py): one worker per
device drains trials from a shared queue; per-device VRAM budgets; auto
batch-size selection; sqrt LR scaling. All workers load the same fixed labeled
dataset and the same train/val split.

Cache layout
------------
logs/experiments/context_length_predictor/<family>/
    trials/trial_<NNN>.json
    sweep_summary.csv / sweep_summary.png
    selection_report.json
    best_model.pt / best_config.json

The <family> subdir matches the basename of --dataset-dir, so each
labeling-model family gets its own deterministic output directory.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
import json
import math
import time
import random
import shutil
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Any
from queue import Empty
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from colorama import Fore

from dotenv import load_dotenv
load_dotenv()


# ==============================================================================
#  EXPERIMENT CONFIGURATION
# ==============================================================================

# -- Data ---------------------------------------------------------------------
CONTEXT_LENGTH = 8192          # must match build_context_length_dataset MAX_WINDOW
CURVE_METRIC   = "mae"         # which label curve to learn: "mae" or "mse"
VAL_FRACTION   = 0.1           # held-out fraction of the labeled dataset

# -- Random search / training loop --------------------------------------------
N_TRIALS                = 60
MAX_EPOCHS              = 40
VAL_EVERY_N_EPOCHS      = 2
EARLY_STOPPING_PATIENCE = 4            # validation events, not epochs
GRAD_CLIP               = 1.0
SEED                    = 42

# Smoke-test mode (PREDICTCSL_TEST=1, set by experiments/run_all.py --test):
# shrink the random search and per-trial training so this stage finishes in a
# couple of minutes. Resolved from the env at import so it also takes effect
# inside the spawned per-GPU trial workers (they re-import this module).
if os.environ.get("PREDICTCSL_TEST") == "1":
    N_TRIALS                = 2
    MAX_EPOCHS              = 3
    VAL_EVERY_N_EPOCHS      = 1
    EARLY_STOPPING_PATIENCE = 2

# -- Dual-objective loss weights ----------------------------------------------
LAMBDA_CURVE = 1.0             # weight of MSE_curve (z-scored curve regression)
LAMBDA_RECON = 0.5             # weight of MSE_reconstruction (aux; 0 disables)
# Selection metric for ranking trials. Options:
#   "regret"   -> val_regret (normalized error penalty of the chosen window)
#   "curve"    -> val_curve_mse
#   "recon"    -> val_recon_mse
#   "combined" -> lambda_curve * val_curve_mse + lambda_recon * val_recon_mse
SELECTION_METRIC = "regret"

# -- Auto batch-size + LR scaling ---------------------------------------------
BS_REFERENCE  = 64
BS_CANDIDATES = [256, 192, 128, 96, 64, 48, 32, 24, 16, 8]
LR_SCALING_RULE = "sqrt"

# -- Multi-GPU configuration --------------------------------------------------
DEVICES                = None          # None -> use all visible CUDA devices
VRAM_BUDGET_GB_PER_DEVICE: Optional[List[float]] = None
VRAM_BUDGET_DEFAULT_GB = 22.0

# -- Hyperparameter search space ----------------------------------------------
# Non-overlapping patches: CONTEXT_LENGTH / patch_length tokens (must divide).
HP_SPACE = {
    "patch_length":        [16, 32, 64, 128],
    "d_model":             [128, 256],
    "num_hidden_layers":   [2, 4, 6, 8],
    "num_attention_heads": [4, 8],
    "dropout":             [0.1, 0.2],
    "mask_ratio":          [0.30, 0.40, 0.50],
    "learning_rate":       [1e-4, 3e-4, 5e-4],
    "weight_decay":        [1e-4, 1e-3],
}

# Run root. Overridable via env so run_all.py --test can redirect predictor
# output into its throwaway tree. Resolved at import so the spawned trial
# workers (which persist per-trial checkpoints under this root) agree with the
# parent process.
CACHE_ROOT = os.environ.get(
    "PREDICTCSL_PREDICTOR_ROOT", "logs/experiments/context_length_predictor")


# ==============================================================================
#  REPRODUCIBILITY / DEVICE RESOLUTION
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_devices() -> List[str]:
    if not torch.cuda.is_available():
        return ["cpu"]
    if DEVICES is not None:
        return list(DEVICES)
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]


def resolve_vram_budgets(devices: List[str]) -> List[float]:
    if VRAM_BUDGET_GB_PER_DEVICE is None:
        return [VRAM_BUDGET_DEFAULT_GB] * len(devices)
    if len(VRAM_BUDGET_GB_PER_DEVICE) != len(devices):
        raise ValueError(
            f"VRAM_BUDGET_GB_PER_DEVICE has {len(VRAM_BUDGET_GB_PER_DEVICE)} "
            f"entries but {len(devices)} devices resolved: {devices}.")
    return list(VRAM_BUDGET_GB_PER_DEVICE)


# ==============================================================================
#  DATASET LOADING
# ==============================================================================

def load_dataset_meta(dataset_dir: str) -> Dict[str, Any]:
    with open(os.path.join(dataset_dir, "meta.json")) as f:
        return json.load(f)


def load_split_tensors(
    dataset_dir: str,
    seed: int = SEED,
    curve_metric: str = CURVE_METRIC,
    val_fraction: float = VAL_FRACTION,
) -> Tuple[torch.Tensor, ...]:
    """Load the labeled dataset and build a fixed train/val split.

    Curves have shape (N, n_windows, n_horizons): one error-vs-context curve
    per (series, horizon). The target is z-scored along the windows axis
    *per horizon* — argmin (the decision) is invariant to shift/scale, so
    the network only has to learn curve *shape*, separately for each horizon.
    The raw (un-normalized) curve is kept for the regret metric.

    NaN entries mark (window, horizon) points that the labeler could not serve
    for a given series — either a model-specific context cap (e.g. Sundial,
    TimeMoE) or a short/left-padded series whose genuine signal is shorter than
    the window. These are *kept* (not dropped) and carried through as NaN; the
    loss and the regret/argmin metrics mask them out per entry. Only series that
    are entirely NaN (never labeled) are dropped.

    Returns:
        x_train:      (n_train, L, 1)
        y_train_norm: (n_train, n_windows, n_horizons)  z-scored target
        y_train_raw:  (n_train, n_windows, n_horizons)  raw error curve
        x_val, y_val_norm, y_val_raw: validation counterparts.
    """
    # contexts.npy lives at the pool root (one level above the model family
    # subdir). Fall back to dataset_dir itself for flat layouts.
    _ctx_candidate = os.path.join(dataset_dir, "contexts.npy")
    if not os.path.isfile(_ctx_candidate):
        _ctx_candidate = os.path.join(os.path.dirname(dataset_dir), "contexts.npy")
    if not os.path.isfile(_ctx_candidate):
        raise FileNotFoundError(
            f"contexts.npy not found in {dataset_dir} or its parent.")
    contexts = np.load(_ctx_candidate)
    curves = np.load(os.path.join(dataset_dir, f"curves_{curve_metric}.npy"))
    if contexts.shape[0] != curves.shape[0]:
        raise ValueError(
            f"contexts ({contexts.shape[0]}) and curves ({curves.shape[0]}) "
            "row counts differ.")
    if curves.ndim != 3:
        raise ValueError(
            f"Expected curves of shape (N, n_windows, n_horizons); got "
            f"{curves.shape}. Re-run build_context_length_dataset.py with "
            "the multi-horizon labeler.")

    # Treat any non-finite curve point (NaN from unservable windows, or inf from
    # overflowed errors in older pools) as an unlabeled point -> NaN, masked
    # downstream. Also drop series with a non-finite context, which would
    # otherwise inject NaN straight into the network input (unmasked).
    curves = curves.astype(np.float32, copy=False)
    curves[~np.isfinite(curves)] = np.nan
    ctx_ok = np.isfinite(contexts).all(axis=1)
    surface_ok = ~np.isnan(curves).all(axis=(1, 2))   # at least one labeled point
    valid = ctx_ok & surface_ok
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"  load_split_tensors: dropping {n_dropped} series "
              f"({int((~ctx_ok).sum())} non-finite context, "
              f"{int((~surface_ok).sum())} all-NaN curve).")
    contexts = contexts[valid].astype(np.float32, copy=False)
    curves = curves[valid]
    if contexts.shape[0] == 0:
        raise RuntimeError("No labeled series in dataset.")

    # Per-(series, horizon) z-score along the windows axis, ignoring NaN points.
    # NaN entries stay NaN in curves_norm and are masked in the loss / metrics.
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(curves, axis=1, keepdims=True)       # (N, 1, n_h)
        sd = np.nanstd(curves, axis=1, keepdims=True)
    curves_norm = (curves - mu) / (sd + 1e-8)

    n = contexts.shape[0]
    perm = np.random.RandomState(seed).permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def to_t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).float()

    x_train = to_t(contexts[train_idx]).unsqueeze(-1)        # (n_train, L, 1)
    x_val   = to_t(contexts[val_idx]).unsqueeze(-1)
    return (
        x_train, to_t(curves_norm[train_idx]), to_t(curves[train_idx]),
        x_val,   to_t(curves_norm[val_idx]),   to_t(curves[val_idx]),
    )


# ==============================================================================
#  MODEL — DUAL-OBJECTIVE PATCH-TRANSFORMER
# ==============================================================================

class PatchTSTContextLength(nn.Module):
    """Patch-based Transformer with a curve-regression head and a recon head.

    Forward pipeline:
        x : (B, L, 1), horizon_idx : (B,) long
        -> non-overlapping patches: (B, N, P), N = L / P
        -> SimMIM-style masking (training only): a random ``mask_ratio`` of
           patches is zeroed in the embedding pathway; originals are the
           reconstruction target.
        -> linear patch embedding -> prepend learnable [CLS] + horizon-embed
        -> learnable positional embedding -> Transformer encoder -> LayerNorm
        -> two heads:
              * curve head: MLP on h[:, 0, :] -> (B, n_windows) error curve
                            *for the requested horizon*
              * recon head: linear on h[:, 1:, :] -> (B, N, P)
    """

    def __init__(
        self,
        context_length: int,
        patch_length: int,
        d_model: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        dropout: float,
        mask_ratio: float,
        n_windows: int,
        n_horizons: int,
    ) -> None:
        super().__init__()
        if context_length % patch_length != 0:
            raise ValueError(
                f"context_length={context_length} must be divisible by "
                f"patch_length={patch_length}.")
        if d_model % num_attention_heads != 0:
            raise ValueError(
                f"d_model={d_model} not divisible by "
                f"num_attention_heads={num_attention_heads}.")

        self.context_length = int(context_length)
        self.patch_length   = int(patch_length)
        self.d_model        = int(d_model)
        self.mask_ratio     = float(mask_ratio)
        self.n_windows      = int(n_windows)
        self.n_horizons     = int(n_horizons)
        self.num_patches    = self.context_length // self.patch_length

        # --- Embedding -------------------------------------------------------
        self.patch_embed = nn.Linear(self.patch_length, self.d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.d_model))
        # Per-horizon additive embedding routed into the CLS token. Conditioning
        # before the encoder lets attention vary with the requested horizon.
        self.horizon_embed = nn.Embedding(self.n_horizons, self.d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.horizon_embed.weight, std=0.02)

        # --- Encoder ---------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_attention_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_hidden_layers)
        self.norm = nn.LayerNorm(self.d_model)

        # --- Heads -----------------------------------------------------------
        self.curve_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.n_windows),
        )
        self.recon_head = nn.Linear(self.d_model, self.patch_length)

    # ------------------------------------------------------------------
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, 1) -> (B, N, P) non-overlapping patches."""
        x = x.squeeze(-1)
        return x.view(x.size(0), self.num_patches, self.patch_length)

    def _build_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Per-sample mask (B, N) bool with exactly round(mask_ratio*N) True."""
        n_mask = int(round(self.mask_ratio * self.num_patches))
        if n_mask == 0:
            return torch.zeros(batch_size, self.num_patches,
                               dtype=torch.bool, device=device)
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, self.num_patches,
                           dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)
        return mask

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        horizon_idx: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: curve_pred (B, n_windows), recon_pred (B, N, P),
        original_patches (B, N, P), mask (B, N).

        horizon_idx: (B,) long, index into HORIZON_GRID from the dataset meta.
        """
        B = x.size(0)
        original_patches = self._patchify(x)

        if mask is None:
            mask = (self._build_mask(B, x.device)
                    if self.training else
                    torch.zeros(B, self.num_patches,
                                dtype=torch.bool, device=x.device))

        embed_input = original_patches.masked_fill(mask.unsqueeze(-1), 0.0)

        h = self.patch_embed(embed_input)
        cls = self.cls_token.expand(B, -1, -1)                       # (B, 1, D)
        h_emb = self.horizon_embed(horizon_idx).unsqueeze(1)         # (B, 1, D)
        cls = cls + h_emb
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embed
        h = self.encoder(h)
        h = self.norm(h)

        curve_pred = self.curve_head(h[:, 0, :])             # (B, n_windows)
        recon_pred = self.recon_head(h[:, 1:, :])            # (B, N, P)
        return curve_pred, recon_pred, original_patches, mask


def compute_dual_loss(
    curve_pred: torch.Tensor,
    recon_pred: torch.Tensor,
    original_patches: torch.Tensor,
    mask: torch.Tensor,
    curve_target: torch.Tensor,
    lambda_curve: float,
    lambda_recon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined dual-objective loss.

    Curve loss is MSE over the z-scored target curve, masking NaN windows
    (points the labeler could not serve for this series). Reconstruction loss is
    MSE over MASKED patches only (visible patches are trivially identity).
    """
    curve_valid = ~torch.isnan(curve_target)
    target_safe = torch.nan_to_num(curve_target, nan=0.0)
    sq = (curve_pred - target_safe).pow(2) * curve_valid.float()
    curve_mse = sq.sum() / curve_valid.float().sum().clamp_min(1.0)

    if mask.any():
        sq_err = (recon_pred - original_patches).pow(2)
        per_patch = sq_err.mean(dim=-1)
        denom = mask.float().sum().clamp_min(1.0)
        recon_mse = (per_patch * mask.float()).sum() / denom
    else:
        recon_mse = torch.zeros((), device=curve_pred.device)

    total = lambda_curve * curve_mse + lambda_recon * recon_mse
    return total, curve_mse, recon_mse


# ==============================================================================
#  RANDOM SEARCH
# ==============================================================================

@dataclass
class TrialConfig:
    patch_length:        int
    d_model:             int
    num_hidden_layers:   int
    num_attention_heads: int
    dropout:             float
    mask_ratio:          float
    learning_rate:       float
    weight_decay:        float

    def __post_init__(self) -> None:
        if self.d_model % self.num_attention_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by "
                f"num_attention_heads={self.num_attention_heads}")
        if CONTEXT_LENGTH % self.patch_length != 0:
            raise ValueError(
                f"context_length={CONTEXT_LENGTH} not divisible by "
                f"patch_length={self.patch_length}")


def sample_trial_configs(n_trials: int, seed: int = SEED) -> List[TrialConfig]:
    """Random search with rejection of d_model/heads/patch incompatibilities."""
    rng = random.Random(seed); seen, configs = set(), []
    max_attempts = n_trials * 50; attempts = 0
    while len(configs) < n_trials and attempts < max_attempts:
        attempts += 1
        cfg = {k: rng.choice(v) for k, v in HP_SPACE.items()}
        if cfg["d_model"] % cfg["num_attention_heads"] != 0:
            continue
        if CONTEXT_LENGTH % cfg["patch_length"] != 0:
            continue
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key); configs.append(TrialConfig(**cfg))
    return configs


# ==============================================================================
#  VRAM PROBING & AUTO BATCH-SIZE
# ==============================================================================

def scale_lr(base_lr: float, bs: int, bs_ref: int = BS_REFERENCE,
             rule: str = LR_SCALING_RULE) -> float:
    if rule == "sqrt":
        return float(base_lr * math.sqrt(bs / bs_ref))
    if rule == "linear":
        return float(base_lr * (bs / bs_ref))
    raise ValueError(f"Unknown LR scaling rule: {rule}")


def _probe_vram(
    trial: TrialConfig,
    batch_size: int,
    device: str,
    n_windows: int,
    n_horizons: int,
) -> Optional[float]:
    """Measure peak VRAM (GB) of one fwd+bwd+step. None on OOM, 0.0 on CPU."""
    if not device.startswith("cuda"):
        return 0.0
    try:
        with torch.cuda.device(torch.device(device)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = PatchTSTContextLength(
                context_length      = CONTEXT_LENGTH,
                patch_length        = trial.patch_length,
                d_model             = trial.d_model,
                num_hidden_layers   = trial.num_hidden_layers,
                num_attention_heads = trial.num_attention_heads,
                dropout             = trial.dropout,
                mask_ratio          = trial.mask_ratio,
                n_windows           = n_windows,
                n_horizons          = n_horizons,
            ).to(device)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

            x = torch.randn(batch_size, CONTEXT_LENGTH, 1, device=device)
            y = torch.randn(batch_size, n_windows, device=device)
            h_idx = torch.randint(0, n_horizons, (batch_size,), device=device)

            curve_pred, recon_pred, orig, mask = model(x, horizon_idx=h_idx)
            loss, _, _ = compute_dual_loss(
                curve_pred, recon_pred, orig, mask, y,
                LAMBDA_CURVE, LAMBDA_RECON)
            loss.backward(); opt.step()

            torch.cuda.synchronize(torch.device(device))
            peak_bytes = torch.cuda.max_memory_allocated()
            del model, opt, x, y, h_idx, curve_pred, recon_pred, orig, mask, loss
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return peak_bytes / (1024 ** 3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        return None


def auto_select_bs_lr(
    trial: TrialConfig,
    device: str,
    budget_gb: float,
    n_windows: int,
    n_horizons: int,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Largest bs in BS_CANDIDATES that fits the per-device budget."""
    for bs in BS_CANDIDATES:
        peak = _probe_vram(trial, bs, device, n_windows, n_horizons)
        if peak is None:
            continue
        if peak <= budget_gb:
            return bs, scale_lr(trial.learning_rate, bs), peak
    return None, None, None


# ==============================================================================
#  EVALUATION
# ==============================================================================

def _evaluate(
    model: PatchTSTContextLength,
    x_val: torch.Tensor,
    y_val_norm: torch.Tensor,
    y_val_raw: torch.Tensor,
    batch_size: int,
    eval_seed: int,
) -> Dict[str, float]:
    """Validation pass — sweeps every horizon for every sample.

    y_val_norm / y_val_raw have shape (n_val, n_windows, n_horizons). We
    iterate horizons inside each batch (so the mask is shared) and average
    curve MSE, regret and window accuracy across (sample, horizon). The
    reconstruction loss is horizon-independent, so we only accumulate it at
    h_idx=0 to avoid double-counting.

    Regret is the headline metric: using the predicted argmin window instead
    of the true-best window, how much extra (relative) error do you incur?
        regret = (err[pred_argmin] - err[true_argmin]) / err[true_argmin].
    win_acc is the fraction whose predicted argmin is within one grid step of
    the true argmin (computed per (sample, horizon)).
    """
    model.eval()
    device = x_val.device
    n_val = x_val.shape[0]
    n_windows  = y_val_norm.shape[1]
    n_horizons = y_val_norm.shape[2]

    curve_sum = torch.zeros((), device=device)
    curve_count = torch.zeros((), device=device)
    recon_sum = torch.zeros((), device=device)
    regret_sum = torch.zeros((), device=device)
    acc_sum   = torch.zeros((), device=device)
    n_recon   = 0

    generator = torch.Generator(device="cpu").manual_seed(eval_seed)

    with torch.no_grad():
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            x  = x_val[start:end]
            yn_all = y_val_norm[start:end]                # (B, n_w, n_h)
            yr_all = y_val_raw[start:end]                 # (B, n_w, n_h)
            B  = x.shape[0]

            # Mask is sampled once per batch and reused for every horizon: the
            # reconstruction target doesn't depend on horizon.
            n_mask = int(round(model.mask_ratio * model.num_patches))
            noise  = torch.rand(B, model.num_patches, generator=generator)
            ids    = torch.argsort(noise, dim=1)
            mask   = torch.zeros(B, model.num_patches, dtype=torch.bool)
            if n_mask > 0:
                mask.scatter_(1, ids[:, :n_mask], True)
            mask = mask.to(device, non_blocking=True)

            idx_b = torch.arange(B, device=device)
            for h_idx_val in range(n_horizons):
                yn = yn_all[:, :, h_idx_val]
                yr = yr_all[:, :, h_idx_val]
                h_idx_batch = torch.full(
                    (B,), h_idx_val, device=device, dtype=torch.long)

                curve_pred, recon_pred, orig, used_mask = model(
                    x, horizon_idx=h_idx_batch, mask=mask)

                # Mask NaN windows (unservable points) in the curve MSE.
                cvalid = ~torch.isnan(yn)
                yn_safe = torch.nan_to_num(yn, nan=0.0)
                curve_sum = curve_sum + (
                    (curve_pred - yn_safe).pow(2) * cvalid.float()).sum()
                curve_count = curve_count + cvalid.float().sum()

                if h_idx_val == 0 and used_mask.any():
                    sq_err = (recon_pred - orig).pow(2).mean(dim=-1)
                    recon_sum = recon_sum + (sq_err * used_mask.float()).sum()
                    n_recon += int(used_mask.float().sum().item())

                # Restrict argmin to servable windows: invalid points get +inf so
                # neither the oracle nor the predictor can select them.
                rvalid = ~torch.isnan(yr)
                yr_inf = torch.where(rvalid, yr, torch.full_like(yr, float("inf")))
                pred_inf = torch.where(
                    rvalid, curve_pred, torch.full_like(curve_pred, float("inf")))
                pred_arg = pred_inf.argmin(dim=1)
                true_arg = yr_inf.argmin(dim=1)
                best_err   = yr_inf[idx_b, true_arg]
                chosen_err = yr_inf[idx_b, pred_arg]
                regret_sum = regret_sum + (
                    (chosen_err - best_err) / best_err.clamp_min(1e-8)).sum()
                acc_sum = acc_sum + (pred_arg - true_arg).abs().le(1).float().sum()

    n_arg   = max(n_val * n_horizons, 1)
    curve_mse = (curve_sum / curve_count.clamp_min(1.0)).item()
    recon_mse = (recon_sum / max(n_recon, 1)).item()
    combined  = LAMBDA_CURVE * curve_mse + LAMBDA_RECON * recon_mse
    return {
        "val_curve_mse": curve_mse,
        "val_recon_mse": recon_mse,
        "val_combined":  combined,
        "val_regret":    (regret_sum / n_arg).item(),
        "val_win_acc":   (acc_sum / n_arg).item(),
    }


def _selection_score(metrics: Dict[str, float]) -> float:
    if SELECTION_METRIC == "regret":
        return metrics["val_regret"]
    if SELECTION_METRIC == "curve":
        return metrics["val_curve_mse"]
    if SELECTION_METRIC == "recon":
        return metrics["val_recon_mse"]
    if SELECTION_METRIC == "combined":
        return metrics["val_combined"]
    raise ValueError(f"Unknown SELECTION_METRIC={SELECTION_METRIC!r}")


# ==============================================================================
#  TRIAL RUNNER
# ==============================================================================

def _failed_result(trial_idx: int, device: str, trial: TrialConfig,
                    reason: str) -> Dict[str, Any]:
    return {
        "trial_idx": trial_idx, "device": device,
        "val_curve_mse": float("nan"), "val_recon_mse": float("nan"),
        "val_combined": float("nan"), "val_regret": float("nan"),
        "val_win_acc": float("nan"),
        "history": {}, "best_state_path": None,
        "failed": True, "skip_reason": reason,
        "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
        "elapsed_seconds": 0.0, "cfg": asdict(trial),
    }


def _run_single_trial(
    trial_idx: int,
    trial: TrialConfig,
    vram_budget_gb: float,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    y_train_raw: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    y_val_raw: torch.Tensor,
    device: str,
    n_windows: int,
    n_horizons: int,
    run_label: str,
) -> Dict[str, Any]:
    """Train one configuration end-to-end; return the best-val checkpoint."""
    bs, lr, peak = auto_select_bs_lr(
        trial, device, vram_budget_gb, n_windows, n_horizons)
    if bs is None:
        return _failed_result(trial_idx, device, trial, "vram_oom")

    tag = f"[{device}] trial {trial_idx:03d}"
    print(Fore.CYAN + f"  {tag}: bs={bs} lr={lr:.2e} peak={peak:.2f}GB  "
          + f"budget={vram_budget_gb:.1f}GB" + Fore.RESET)

    t0 = time.perf_counter()
    n_train = x_train.shape[0]
    steps_per_epoch = max(n_train // bs, 1)

    model = PatchTSTContextLength(
        context_length      = CONTEXT_LENGTH,
        patch_length        = trial.patch_length,
        d_model             = trial.d_model,
        num_hidden_layers   = trial.num_hidden_layers,
        num_attention_heads = trial.num_attention_heads,
        dropout             = trial.dropout,
        mask_ratio          = trial.mask_ratio,
        n_windows           = n_windows,
        n_horizons          = n_horizons,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=trial.weight_decay)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Dict[str, float] = {}
    patience_left = EARLY_STOPPING_PATIENCE
    history: Dict[str, List[float]] = {
        "train_total": [], "train_curve": [], "train_recon": [],
        "val_combined": [], "val_curve": [], "val_recon": [],
        "val_regret": [], "val_win_acc": [], "val_epochs": [],
    }

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            run_total = torch.zeros((), device=device)
            run_curve = torch.zeros((), device=device)
            run_recon = torch.zeros((), device=device)
            n_seen = 0

            for step in range(steps_per_epoch):
                idx = perm[step * bs : (step + 1) * bs]
                x = x_train.index_select(0, idx)
                y = y_train.index_select(0, idx)              # (bs, n_w, n_h)

                B = x.shape[0]
                h_idx = torch.randint(
                    0, n_horizons, (B,), device=device, dtype=torch.long)
                y_at_h = y[torch.arange(B, device=device), :, h_idx]  # (B, n_w)

                optimizer.zero_grad(set_to_none=True)
                curve_pred, recon_pred, orig, mask = model(x, horizon_idx=h_idx)
                loss, l_curve, l_recon = compute_dual_loss(
                    curve_pred, recon_pred, orig, mask, y_at_h,
                    LAMBDA_CURVE, LAMBDA_RECON)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                run_total = run_total + loss.detach()    * bs
                run_curve = run_curve + l_curve.detach()  * bs
                run_recon = run_recon + l_recon.detach()  * bs
                n_seen += bs

            denom = max(n_seen, 1)
            history["train_total"].append((run_total / denom).item())
            history["train_curve"].append((run_curve / denom).item())
            history["train_recon"].append((run_recon / denom).item())

            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == MAX_EPOCHS:
                metrics = _evaluate(
                    model, x_val, y_val, y_val_raw, bs,
                    eval_seed=SEED + 1000 * trial_idx)
                history["val_combined"].append(metrics["val_combined"])
                history["val_curve"].append(metrics["val_curve_mse"])
                history["val_recon"].append(metrics["val_recon_mse"])
                history["val_regret"].append(metrics["val_regret"])
                history["val_win_acc"].append(metrics["val_win_acc"])
                history["val_epochs"].append(epoch)
                print(Fore.YELLOW
                      + f"  {tag} epoch {epoch:3d}  "
                      + f"train_total={history['train_total'][-1]:.4f}  "
                      + f"val: curve={metrics['val_curve_mse']:.4f} "
                      + f"recon={metrics['val_recon_mse']:.4f} "
                      + f"regret={metrics['val_regret']:.4f} "
                      + f"win_acc={metrics['val_win_acc']:.3f}"
                      + Fore.RESET)

                score = _selection_score(metrics)
                if score < best_val:
                    best_val = score
                    best_metrics = dict(metrics)
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    patience_left = EARLY_STOPPING_PATIENCE
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(Fore.MAGENTA
                              + f"  {tag} early stop at epoch {epoch}"
                              + Fore.RESET)
                        break
    except torch.cuda.OutOfMemoryError as exc:
        print(Fore.RED + f"  {tag} OOM during training: {exc}" + Fore.RESET)
        torch.cuda.empty_cache()

    best_state_path: Optional[str] = None
    if best_state is not None:
        # Durable, deterministic location under the run dir (atomic write) so a
        # later resume can still select this trial's checkpoint.
        best_state_path = _trial_best_path(run_label, trial_idx)
        os.makedirs(os.path.dirname(best_state_path), exist_ok=True)
        tmp_path = best_state_path + ".tmp"
        torch.save(best_state, tmp_path)
        os.replace(tmp_path, best_state_path)
        del best_state

    elapsed = time.perf_counter() - t0
    del optimizer, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "trial_idx": trial_idx, "device": device,
        "val_curve_mse": best_metrics.get("val_curve_mse", float("nan")),
        "val_recon_mse": best_metrics.get("val_recon_mse", float("nan")),
        "val_combined":  best_metrics.get("val_combined",  float("nan")),
        "val_regret":    best_metrics.get("val_regret",    float("nan")),
        "val_win_acc":   best_metrics.get("val_win_acc",   float("nan")),
        "history": history,
        "best_state_path": best_state_path,
        "failed": best_state_path is None,
        "skip_reason": None if best_state_path is not None else "no_improvement",
        "auto_batch_size": bs, "auto_lr": lr,
        "peak_vram_gb": round(peak, 3),
        "elapsed_seconds": round(elapsed, 2),
        "cfg": asdict(trial),
    }


# ==============================================================================
#  WORKER PROCESS
# ==============================================================================

def gpu_worker(
    worker_id: int,
    device: str,
    vram_budget_gb: float,
    trial_queue: "mp.Queue",
    result_queue: "mp.Queue",
    dataset_dir: str,
    n_windows: int,
    n_horizons: int,
    run_label: str,
) -> None:
    """One worker per GPU. Loads the labeled dataset, then drains trials."""
    set_seed(SEED + worker_id)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    torch.set_float32_matmul_precision("high")

    try:
        (x_tr, y_tr, y_tr_raw,
         x_va, y_va, y_va_raw) = load_split_tensors(dataset_dir, seed=SEED)
        x_train = x_tr.pin_memory().to(device, non_blocking=True)
        y_train = y_tr.pin_memory().to(device, non_blocking=True)
        y_train_raw = y_tr_raw.pin_memory().to(device, non_blocking=True)
        x_val   = x_va.pin_memory().to(device, non_blocking=True)
        y_val   = y_va.pin_memory().to(device, non_blocking=True)
        y_val_raw = y_va_raw.pin_memory().to(device, non_blocking=True)
        del x_tr, y_tr, y_tr_raw, x_va, y_va, y_va_raw
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        print(Fore.CYAN
              + f"  [{device}] worker {worker_id} ready: "
              + f"x_train={tuple(x_train.shape)} x_val={tuple(x_val.shape)}"
              + Fore.RESET)
    except Exception as exc:
        print(Fore.RED + f"  [{device}] worker {worker_id} setup failed: "
              + f"{type(exc).__name__}: {exc}" + Fore.RESET)
        while True:
            try: msg = trial_queue.get(timeout=5)
            except Empty: break
            if msg is None: break
            trial_idx, trial = msg
            result_queue.put(_failed_result(
                trial_idx, device, trial, "worker_setup_failed"))
        return

    while True:
        try:
            msg = trial_queue.get(timeout=3600)
        except Empty:
            print(Fore.MAGENTA + f"  [{device}] worker {worker_id} queue "
                  + "timeout — exiting." + Fore.RESET)
            break
        if msg is None:
            break
        trial_idx, trial = msg
        try:
            result = _run_single_trial(
                trial_idx, trial, vram_budget_gb,
                x_train, y_train, y_train_raw,
                x_val, y_val, y_val_raw, device, n_windows, n_horizons,
                run_label,
            )
        except Exception as exc:
            print(Fore.RED + f"  [{device}] trial {trial_idx:03d} "
                  + f"CRASHED: {type(exc).__name__}: {exc}" + Fore.RESET)
            result = _failed_result(
                trial_idx, device, trial,
                f"trial_exception:{type(exc).__name__}")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        result_queue.put(result)

    del x_train, y_train, y_train_raw, x_val, y_val, y_val_raw
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(Fore.CYAN + f"  [{device}] worker {worker_id} exited." + Fore.RESET)


# ==============================================================================
#  PLOTTING
# ==============================================================================

def _plot_sweep_summary(trials_df: pd.DataFrame, save_path: str,
                        run_label: str) -> None:
    """Three-panel scatter: val_regret, val_curve_mse, val_recon_mse."""
    df = trials_df.sort_values("trial_idx").reset_index(drop=True)
    panels = [
        ("val_regret",    "val regret (normalized)"),
        ("val_curve_mse", "val curve MSE (z-scored)"),
        ("val_recon_mse", "val reconstruction MSE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (col, label) in zip(axes, panels):
        sub = df.dropna(subset=[col]) if col in df.columns else df.iloc[0:0]
        if sub.empty:
            ax.set_title(f"{label}  (no valid trials)")
            continue
        running_best = np.minimum.accumulate(sub[col].values)
        ax.scatter(sub["trial_idx"], sub[col],
                   s=32, alpha=0.55, color="#1f77b4",
                   edgecolor="white", linewidth=0.5, label="trial")
        ax.plot(sub["trial_idx"], running_best,
                marker="o", markersize=6, linewidth=2.0, color="#d62728",
                label="running best")
        best_row = sub.iloc[sub[col].values.argmin()]
        ax.annotate(
            f"  best: trial {int(best_row['trial_idx'])}\n"
            f"  {label}={best_row[col]:.4f}",
            xy=(best_row["trial_idx"], best_row[col]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="#fff3b0", ec="#999", alpha=0.95),
        )
        ax.set_xlabel("Trial index", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Context-length predictor sweep — {run_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


# ==============================================================================
#  CACHE HELPERS
# ==============================================================================

def _run_dir(run_label: str) -> str:
    return os.path.join(CACHE_ROOT, run_label)

def _trial_json_path(run_label: str, trial_idx: int) -> str:
    return os.path.join(_run_dir(run_label), "trials",
                        f"trial_{trial_idx:03d}.json")

def _trial_best_path(run_label: str, trial_idx: int) -> str:
    """Durable per-trial best-weights checkpoint (survives a pause/resume).

    Replaces the old ephemeral tempfile: keeping the weights in the run dir is
    what lets a resumed run re-select the global best from *cached* trials, not
    just the ones rebuilt in the current process."""
    return os.path.join(_run_dir(run_label), "trials",
                        f"trial_{trial_idx:03d}_best.pt")

def _write_sweep_progress(run_dir: str, done: int, total: int) -> None:
    """Atomically write {done, total} for run_all.py to poll and drive its bar."""
    path = os.path.join(run_dir, "sweep_progress.json")
    tmp  = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"done": done, "total": total}, f)
        os.replace(tmp, path)
    except OSError:
        pass   # non-critical — the bar just won't update


def _load_trial_result(run_label: str, trial_idx: int) -> Optional[Dict]:
    p = _trial_json_path(run_label, trial_idx)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def _save_trial_result(run_label: str, trial_idx: int, result: Dict) -> None:
    p = _trial_json_path(run_label, trial_idx)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    serializable = {k: v for k, v in result.items() if k != "best_state_path"}
    with open(p, "w") as f:
        json.dump(serializable, f, indent=2)


# ==============================================================================
#  SELECTION + ARTIFACTS
# ==============================================================================

_METRIC_KEY = {
    "regret":   "val_regret",
    "curve":    "val_curve_mse",
    "recon":    "val_recon_mse",
    "combined": "val_combined",
}


def _select_final(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pick the final trial by SELECTION_METRIC over those with a checkpoint."""
    metric_key = _METRIC_KEY[SELECTION_METRIC]
    valid = [r for r in results
             if not r.get("failed", False)
             and r.get("best_state_path")
             and os.path.isfile(r["best_state_path"])
             and not (isinstance(r.get(metric_key), float)
                      and math.isnan(r[metric_key]))]
    if not valid:
        return {}, {"selection_metric": metric_key,
                    "note": "no_valid_trials_with_checkpoints"}

    chosen = min(valid, key=lambda r: r[metric_key])
    report = {
        "selection_metric":   metric_key,
        "selected_trial_idx": int(chosen["trial_idx"]),
        "selected_metrics": {
            "val_curve_mse": chosen.get("val_curve_mse"),
            "val_recon_mse": chosen.get("val_recon_mse"),
            "val_combined":  chosen.get("val_combined"),
            "val_regret":    chosen.get("val_regret"),
            "val_win_acc":   chosen.get("val_win_acc"),
        },
        "n_valid": len(valid), "n_total": len(results),
        "lambda_curve": LAMBDA_CURVE, "lambda_recon": LAMBDA_RECON,
    }
    return chosen, report


def _persist_artifacts(results: List[Dict[str, Any]], run_label: str) -> None:
    rdir = _run_dir(run_label)
    os.makedirs(os.path.join(rdir, "trials"), exist_ok=True)
    for r in results:
        _save_trial_result(run_label, r["trial_idx"], r)

    rows = []
    for r in results:
        flat = {"trial_idx": r["trial_idx"]}
        flat.update(r.get("cfg", {}))
        for key in ("val_curve_mse", "val_recon_mse", "val_combined",
                    "val_regret", "val_win_acc", "auto_batch_size", "auto_lr",
                    "peak_vram_gb", "device", "elapsed_seconds",
                    "skip_reason", "failed"):
            if key in r:
                flat[key] = r[key]
        rows.append(flat)
    df = pd.DataFrame(rows)
    keep = ["trial_idx", "val_regret", "val_win_acc", "val_curve_mse",
            "val_recon_mse", "val_combined",
            "patch_length", "d_model", "num_hidden_layers",
            "num_attention_heads", "dropout", "mask_ratio",
            "learning_rate", "weight_decay",
            "auto_batch_size", "auto_lr", "peak_vram_gb",
            "device", "elapsed_seconds", "failed", "skip_reason"]
    df_csv = df[[c for c in keep if c in df.columns]]
    csv_path = os.path.join(rdir, "sweep_summary.csv")
    df_csv.to_csv(csv_path, index=False)
    print(Fore.GREEN + f"  Sweep CSV: {csv_path}" + Fore.RESET)

    if len(df_csv) > 0:
        _plot_sweep_summary(df_csv.copy(),
                            os.path.join(rdir, "sweep_summary.png"), run_label)


# ==============================================================================
#  MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Context-length predictor training.")
    p.add_argument("--dataset-dir", type=str, required=True,
                   help=(
                       "Model-family subdir produced by build_context_length_dataset.py "
                       "(e.g. logs/experiments/context_length_dataset/chronos2). "
                       "Must contain curves_{mae,mse}.npy and meta.json. "
                       "contexts.npy is resolved from this dir or its parent."
                   ))
    return p.parse_args()


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"--dataset-dir not found: {dataset_dir}")

    set_seed(SEED)
    devices      = resolve_devices()
    vram_budgets = resolve_vram_budgets(devices)

    # ---------- Inspect dataset --------------------------------------------
    meta = load_dataset_meta(dataset_dir)
    window_grid = meta["window_grid"]
    n_windows = len(window_grid)
    if "horizon_grid" not in meta:
        raise ValueError(
            "Dataset meta.json is missing 'horizon_grid' — re-run "
            "build_context_length_dataset.py with the multi-horizon labeler.")
    horizon_grid = meta["horizon_grid"]
    n_horizons = len(horizon_grid)
    if meta.get("max_window") != CONTEXT_LENGTH:
        raise ValueError(
            f"Dataset max_window={meta.get('max_window')} != "
            f"CONTEXT_LENGTH={CONTEXT_LENGTH}.")

    # Probe split sizes once (workers reload independently).
    x_tr, y_tr, _, x_va, _, _ = load_split_tensors(dataset_dir, seed=SEED)
    n_train, n_val = x_tr.shape[0], x_va.shape[0]
    del x_tr, y_tr, x_va

    print(Fore.CYAN + f"Devices: {devices}" + Fore.RESET)
    print(Fore.CYAN + f"VRAM budgets (GB): {dict(zip(devices, vram_budgets))}"
          + Fore.RESET)
    print(Fore.CYAN + f"Dataset: {dataset_dir}" + Fore.RESET)
    print(Fore.CYAN + f"  label model={meta.get('model_display')}  "
          + f"curve_metric={CURVE_METRIC}  horizons={horizon_grid}"
          + Fore.RESET)
    print(Fore.CYAN + f"  n_train={n_train}  n_val={n_val}  "
          + f"n_windows={n_windows}  n_horizons={n_horizons}  "
          + f"windows={window_grid}" + Fore.RESET)
    print(Fore.CYAN + f"N_TRIALS={N_TRIALS}  lambda_curve={LAMBDA_CURVE}  "
          + f"lambda_recon={LAMBDA_RECON}  selection={SELECTION_METRIC}"
          + Fore.RESET)

    # Deterministic per-family layout: run_label is the dataset-dir basename
    # (i.e. the model family that produced the labels). Re-running overwrites
    # in place; cached trials are skipped via _load_trial_result.
    run_label = os.path.basename(os.path.normpath(dataset_dir))
    run_dir = _run_dir(run_label)
    os.makedirs(os.path.join(run_dir, "trials"), exist_ok=True)

    trial_configs = sample_trial_configs(N_TRIALS, seed=SEED)
    print(Fore.CYAN + f"Sampled {len(trial_configs)} unique trial configs"
          + Fore.RESET)

    # ---------- Resolve pending trials (cache-aware) ------------------------
    cached_results: List[Dict[str, Any]] = []
    pending: List[Tuple[int, TrialConfig]] = []
    for idx, trial in enumerate(trial_configs):
        cached = _load_trial_result(run_label, idx)
        if (cached is not None
                and "val_curve_mse" in cached
                and not (isinstance(cached["val_curve_mse"], float)
                         and math.isnan(cached["val_curve_mse"]))):
            # Re-attach the durable weights so this cached trial can still win
            # final selection on a resumed run (None if weights were pruned).
            bp = _trial_best_path(run_label, idx)
            cached["best_state_path"] = bp if os.path.isfile(bp) else None
            cached_results.append(cached)
        else:
            pending.append((idx, trial))

    # Write initial progress so run_all.py's bar gets a total right away
    # (before any GPU work starts). Updated after each result below.
    _write_sweep_progress(run_dir, len(cached_results), N_TRIALS)

    # ---------- Spawn workers ----------------------------------------------
    ctx = mp.get_context("spawn")
    trial_queue  = ctx.Queue()
    result_queue = ctx.Queue()
    workers: List[mp.Process] = []
    for i, (device, budget) in enumerate(zip(devices, vram_budgets)):
        p = ctx.Process(
            target=gpu_worker,
            args=(i, device, budget, trial_queue, result_queue,
                  dataset_dir, n_windows, n_horizons, run_label),
            name=f"gpu_worker_{i}_{device.replace(':', '')}",
        )
        p.start(); workers.append(p)

    for trial_idx, trial in pending:
        trial_queue.put((trial_idx, trial))

    fresh_results: List[Dict[str, Any]] = []
    n_expected = len(pending)
    n_received = 0
    t_start = time.perf_counter()
    while n_received < n_expected:
        try:
            r = result_queue.get(timeout=3600)
        except Empty:
            if not any(p.is_alive() for p in workers):
                print(Fore.RED + f"  All workers died with "
                      + f"{n_received}/{n_expected} results." + Fore.RESET)
                break
            print(Fore.YELLOW + f"  No result in 1h "
                  + f"({n_received}/{n_expected})." + Fore.RESET)
            continue
        fresh_results.append(r); n_received += 1
        # Flush this trial's JSON immediately so an interrupted sweep can resume
        # from it. The durable weights (trial_NNN_best.pt) are written by the
        # worker, but _load_trial_result keys resume on the JSON — without this
        # incremental save the JSONs only land at the end (_persist_artifacts),
        # so a crash mid-sweep loses all cached trials and restarts from zero.
        _save_trial_result(run_label, r["trial_idx"], r)
        _write_sweep_progress(run_dir, len(cached_results) + n_received, N_TRIALS)

    print(Fore.MAGENTA + f"  Sweep wall-clock: "
          + f"{time.perf_counter() - t_start:.1f}s  "
          + f"({n_received}/{n_expected} fresh trials)" + Fore.RESET)

    for _ in workers: trial_queue.put(None)
    for p in workers:
        p.join(timeout=120)
        if p.is_alive():
            print(Fore.RED + f"  Worker {p.name} still alive — terminating."
                  + Fore.RESET)
            p.terminate(); p.join(timeout=10)

    all_results = cached_results + fresh_results
    _persist_artifacts(all_results, run_label)

    # ---------- Final selection --------------------------------------------
    # Select over ALL trials (cached + fresh): on a resumed run the winner may
    # be a trial that finished in an earlier session, whose weights now live
    # durably under trials/trial_NNN_best.pt.
    chosen, report = _select_final(all_results)
    if chosen:
        shutil.copyfile(chosen["best_state_path"],
                        os.path.join(run_dir, "best_model.pt"))
        with open(os.path.join(run_dir, "best_config.json"), "w") as f:
            json.dump({
                "trial_idx":        int(chosen["trial_idx"]),
                **chosen.get("cfg", {}),
                "val_curve_mse":    chosen.get("val_curve_mse"),
                "val_recon_mse":    chosen.get("val_recon_mse"),
                "val_combined":     chosen.get("val_combined"),
                "val_regret":       chosen.get("val_regret"),
                "val_win_acc":      chosen.get("val_win_acc"),
                "selection_metric": report["selection_metric"],
                "auto_batch_size":  chosen.get("auto_batch_size"),
                "auto_lr":          chosen.get("auto_lr"),
                "context_length":   CONTEXT_LENGTH,
                "n_windows":        n_windows,
                "window_grid":      window_grid,
                "n_horizons":       n_horizons,
                "horizon_grid":     horizon_grid,
                "curve_metric":     CURVE_METRIC,
                "lambda_curve":     LAMBDA_CURVE,
                "lambda_recon":     LAMBDA_RECON,
                "dataset_dir":      dataset_dir,
                "label_model":      meta.get("model_display"),
            }, f, indent=2)
        with open(os.path.join(run_dir, "selection_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        sk = report["selection_metric"]
        print(Fore.GREEN + f"  FINAL SELECTION: trial "
              + f"{int(chosen['trial_idx']):03d}  ({sk}={chosen.get(sk):.4f}  "
              + f"win_acc={chosen.get('val_win_acc'):.3f})" + Fore.RESET)
    else:
        print(Fore.YELLOW + "  No valid fresh trials — no checkpoint persisted."
              + Fore.RESET)

    # NOTE: per-trial best-weights (trials/trial_NNN_best.pt) are intentionally
    # kept, not deleted — they are the resume cache that lets a later run
    # re-select the global best without retraining. best_model.pt is a separate
    # copy of the winner. Delete the trials/ dir manually to reclaim disk.

    print(Fore.GREEN + "\nContext-length predictor training done." + Fore.RESET)


if __name__ == "__main__":
    main()
