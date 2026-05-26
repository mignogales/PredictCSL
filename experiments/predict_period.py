"""
Dual-objective Patch-Transformer training pipeline.

Pipeline overview
-----------------
Single-stage random search over a Patch-Transformer architecture with two
simultaneous training objectives, evaluated and selected jointly:

    L = lambda1 * MSE_period  +  lambda2 * MSE_reconstruction

where
  * MSE_period         : continuous regression of the dominant period T of a
                         synthetic 1-D time series, routed from a learnable
                         [CLS] token at index 0 through an MLP head.
  * MSE_reconstruction : masked-patch prediction (SimMIM-style) over all
                         non-CLS patch tokens at indices 1..N. A subset of
                         input patches is zeroed in the embedding pathway;
                         the model must reconstruct their original values.

Synthetic data
--------------
A `SyntheticPeriodicDataset` generates 1-D series of fixed length L = 8192
on the fly by composing:
  * 1-3 periodic components (sine/cosine, log-uniformly sampled periods,
    random phases, random amplitudes with a dominant component);
  * (optional) AR(p) process with stable coefficients;
  * (optional) low-order polynomial trend;
  * Gaussian noise;
followed by per-sample standardization. The dominant component's period
defines the regression target (predicted as log T for numerical stability).

Multi-GPU execution
-------------------
Process-per-GPU parallelism is preserved from the original PatchTST script:
  * One worker per device drains trials from a shared queue.
  * Per-device VRAM budgets; auto batch-size selection over BS_CANDIDATES via
    empirical fwd+bwd+step probing; LR rescaled with the sqrt rule.
  * Validation accumulators stay on device — one host sync per pass.
  * Pinned CPU->GPU upload at worker startup.

Cache layout
------------
logs/experiments/patchtst_dual_objective/<run>/
    trials/trial_<NNN>.json
    sweep_summary.csv
    sweep_summary.png
    selection_report.json
    best_model.pt
    best_config.json
"""

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
import tempfile
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from queue import Empty
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore

from dotenv import load_dotenv
load_dotenv()


# ==============================================================================
#  EXPERIMENT CONFIGURATION
# ==============================================================================

# -- Architectural / data constants -------------------------------------------
CONTEXT_LENGTH         = 8192          # hardcoded per task specification
PERIOD_MIN             = 16            # minimum dominant period (samples)
PERIOD_MAX             = 2048          # maximum dominant period (samples)
N_TRAIN_WINDOWS        = 20_000
N_VAL_WINDOWS          = 2_000

# -- Random search / training loop --------------------------------------------
N_TRIALS                = 60
MAX_EPOCHS              = 40
VAL_EVERY_N_EPOCHS      = 2
EARLY_STOPPING_PATIENCE = 4            # validation events, not epochs
WEIGHT_DECAY            = 1e-4
GRAD_CLIP               = 1.0
SEED                    = 42

# -- Dual-objective loss weights (configurable per task spec) -----------------
LAMBDA_PERIOD          = 1.0           # weight of MSE_period (log-T regression)
LAMBDA_RECON           = 0.0           # weight of MSE_reconstruction
# Selection metric for ranking trials. Options:
#   "combined" -> lambda1 * val_period_mse + lambda2 * val_recon_mse
#   "period"   -> val_period_mse
#   "recon"    -> val_recon_mse
SELECTION_METRIC       = "combined"

# -- Auto batch-size + LR scaling ---------------------------------------------
BS_REFERENCE           = 64            # reference batch size for LR scaling
BS_CANDIDATES          = [256, 192, 128, 96, 64, 48, 32, 24, 16, 8]
LR_SCALING_RULE        = "sqrt"

# -- Multi-GPU configuration --------------------------------------------------
DEVICES                = None          # None -> use all visible CUDA devices
VRAM_BUDGET_GB_PER_DEVICE: Optional[List[float]] = [10.0, 10.0, 6.0, 10.0]
VRAM_BUDGET_DEFAULT_GB = 8.0

# -- Hyperparameter search space ----------------------------------------------
# Patches are non-overlapping (stride == patch_length) — yields exactly
# CONTEXT_LENGTH / patch_length tokens, which must divide cleanly.
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

CACHE_ROOT = "logs/experiments/patchtst_dual_objective"


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
#  SYNTHETIC DATASET
# ==============================================================================

def _generate_synthetic_sample(
    rng: np.random.RandomState,
    L: int = CONTEXT_LENGTH,
    period_min: int = PERIOD_MIN,
    period_max: int = PERIOD_MAX,
) -> Tuple[np.ndarray, float]:
    """Generate one synthetic 1-D series and its dominant log-period label.

    Composition (each independent except the periodic block, which is mandatory):
      * 1-3 periodic components, log-uniformly spaced periods in
        [period_min, period_max]. The largest-amplitude component is
        multiplicatively boosted so dominance is unambiguous.
      * (p = 0.5) AR(p) process with p ~ U{1,2,3} and coefficients drawn
        from U[-0.3/p, 0.3/p] to guarantee weak-stationarity in expectation.
      * (p = 0.5) polynomial trend of degree in {1,2,3}, low magnitude.
      * Gaussian noise with std in [0.05, 0.30].
    The final series is standardized to zero mean and unit variance.

    Args:
        rng: numpy RandomState for full determinism per sample.
        L: sequence length (samples).
        period_min, period_max: bounds of the period grid (samples).

    Returns:
        signal: float32 ndarray of shape (L,).
        log_dominant_period: scalar np.float32, natural log of the dominant T.
    """
    t = np.arange(L, dtype=np.float32)
    signal = np.zeros(L, dtype=np.float32)

    # -- (1) Periodic components ---------------------------------------------
    n_periodic = int(rng.randint(1, 4))
    log_lo, log_hi = math.log(period_min), math.log(period_max)
    periods = np.exp(rng.uniform(log_lo, log_hi, size=n_periodic)).astype(np.float32)
    amplitudes = rng.uniform(0.5, 2.0, size=n_periodic).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_periodic).astype(np.float32)

    # Enforce a clear amplitude margin so the dominant component is well-defined.
    dom_idx = int(np.argmax(amplitudes))
    amplitudes[dom_idx] *= float(rng.uniform(1.8, 3.0))
    dominant_period = float(periods[dom_idx])

    for amp, T_p, ph in zip(amplitudes, periods, phases):
        omega = 2.0 * np.pi / float(T_p)
        # Half the time use cosine; the network must be phase-invariant
        # for the period label.
        if rng.uniform() < 0.5:
            signal += amp * np.sin(omega * t + ph)
        else:
            signal += amp * np.cos(omega * t + ph)

    # -- (2) Optional AR(p) ---------------------------------------------------
    if rng.uniform() < 0.5:
        p = int(rng.randint(1, 4))
        coeffs = rng.uniform(-0.3 / p, 0.3 / p, size=p).astype(np.float32)
        innov = rng.normal(0.0, 0.3, size=L).astype(np.float32)
        ar_series = np.zeros(L, dtype=np.float32)
        # Vectorization is awkward for AR with arbitrary p; the loop runs
        # once per sample at dataset-build time so the cost is acceptable.
        for i in range(p, L):
            acc = 0.0
            for k in range(p):
                acc += float(coeffs[k]) * float(ar_series[i - k - 1])
            ar_series[i] = acc + innov[i]
        signal += ar_series

    # -- (3) Optional polynomial trend ---------------------------------------
    if rng.uniform() < 0.5:
        deg = int(rng.randint(1, 4))
        t_norm = (t / float(L)).astype(np.float32)              # in [0, 1)
        poly_coeffs = (rng.uniform(-1.0, 1.0, size=deg + 1) * 0.3).astype(np.float32)
        trend = np.polyval(poly_coeffs[::-1], t_norm).astype(np.float32)
        signal += trend

    # -- (4) Gaussian noise --------------------------------------------------
    noise_std = float(rng.uniform(0.05, 0.30))
    signal += rng.normal(0.0, noise_std, size=L).astype(np.float32)

    # -- Standardize ---------------------------------------------------------
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    signal = signal.astype(np.float32, copy=False)

    return signal, np.float32(math.log(dominant_period))


class SyntheticPeriodicDataset(torch.utils.data.Dataset):
    """Synthetic 1-D periodic dataset with deterministic per-index sampling.

    Each ``__getitem__(idx)`` call seeds an independent RandomState with
    ``base_seed + idx`` so the train/val partitions are fully reproducible
    and worker-local.

    Args:
        n_samples: number of samples in this split.
        base_seed: base seed; per-sample seed = base_seed + idx.
        sequence_length: fixed L per sample.
        period_min, period_max: dominant-period bounds (samples).
    """

    def __init__(
        self,
        n_samples: int,
        base_seed: int,
        sequence_length: int = CONTEXT_LENGTH,
        period_min: int = PERIOD_MIN,
        period_max: int = PERIOD_MAX,
    ) -> None:
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)
        self.sequence_length = int(sequence_length)
        self.period_min = int(period_min)
        self.period_max = int(period_max)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.RandomState(self.base_seed + idx)
        signal, log_T = _generate_synthetic_sample(
            rng, self.sequence_length, self.period_min, self.period_max)
        x = torch.from_numpy(signal).unsqueeze(-1)              # (L, 1)
        y = torch.tensor(log_T, dtype=torch.float32)            # scalar
        return x, y


def build_train_val_tensors(
    n_train: int = N_TRAIN_WINDOWS,
    n_val: int = N_VAL_WINDOWS,
    context_length: int = CONTEXT_LENGTH,
    seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize train/val pools as dense tensors.

    Returns:
        x_train: (n_train, L, 1) float32
        y_train: (n_train,)      float32  (log period)
        x_val:   (n_val,   L, 1) float32
        y_val:   (n_val,)        float32
    """
    train_ds = SyntheticPeriodicDataset(n_train, base_seed=seed,
                                        sequence_length=context_length)
    val_ds   = SyntheticPeriodicDataset(n_val,   base_seed=seed + 10_000_000,
                                        sequence_length=context_length)

    x_train = torch.empty(n_train, context_length, 1, dtype=torch.float32)
    y_train = torch.empty(n_train, dtype=torch.float32)
    for i in range(n_train):
        xi, yi = train_ds[i]; x_train[i] = xi; y_train[i] = yi

    x_val = torch.empty(n_val, context_length, 1, dtype=torch.float32)
    y_val = torch.empty(n_val, dtype=torch.float32)
    for i in range(n_val):
        xi, yi = val_ds[i]; x_val[i] = xi; y_val[i] = yi

    return x_train, y_train, x_val, y_val


# ==============================================================================
#  MODEL — DUAL-OBJECTIVE PATCH-TRANSFORMER
# ==============================================================================

class PatchTSTDualObjective(nn.Module):
    """Patch-based Transformer with two prediction heads.

    Forward pipeline:
        x : (B, L, 1)
        -> non-overlapping patches: (B, N, P) with N = L / P
        -> SimMIM-style masking: a random ``mask_ratio`` of patches is
           zeroed in the embedding pathway (training only); original patches
           are retained as the reconstruction target.
        -> linear patch embedding: (B, N, d_model)
        -> prepend learnable [CLS]: (B, N+1, d_model)
        -> learnable positional embedding -> Transformer encoder
        -> LayerNorm
        -> two heads:
              * period head: MLP on h[:, 0, :] -> scalar log-period
              * recon  head: linear on h[:, 1:, :] -> (B, N, P)

    Args:
        context_length: input length L.
        patch_length: patch size P. Must divide ``context_length`` evenly.
        d_model: embedding dimension.
        num_hidden_layers: number of TransformerEncoder layers.
        num_attention_heads: number of attention heads.
        dropout: dropout used in attention, FFN, and head MLPs.
        mask_ratio: fraction of patches masked during training.
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
        self.num_patches    = self.context_length // self.patch_length

        # --- Embedding -------------------------------------------------------
        self.patch_embed = nn.Linear(self.patch_length, self.d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        self.period_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )
        self.recon_head = nn.Linear(self.d_model, self.patch_length)

    # ------------------------------------------------------------------
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, 1) -> (B, N, P) non-overlapping patches."""
        x = x.squeeze(-1)                                       # (B, L)
        # reshape into N non-overlapping windows of length P.
        return x.view(x.size(0), self.num_patches, self.patch_length)

    def _build_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Bernoulli-style per-sample mask of shape (B, N), dtype bool.

        Uses ``argsort`` over uniform noise so the number of masked patches
        per sample is exactly ``round(mask_ratio * N)``.
        """
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
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, L, 1) input series.
            mask: optional (B, N) bool tensor. If None, a fresh random mask
                is sampled when ``self.training`` is True; in eval the mask
                is all-False unless explicitly provided.

        Returns:
            period_pred:     (B,)        predicted log-period.
            recon_pred:      (B, N, P)   reconstructed patches at every position.
            original_patches:(B, N, P)   pre-masking patches (reconstruction
                                          target).
            mask:            (B, N) bool mask actually used for this pass.
        """
        B = x.size(0)
        original_patches = self._patchify(x)                    # (B, N, P)

        if mask is None:
            mask = (self._build_mask(B, x.device)
                    if self.training else
                    torch.zeros(B, self.num_patches,
                                dtype=torch.bool, device=x.device))

        # Zero-out masked patches in the embedding pathway (target retained).
        # broadcasting: mask (B,N) -> (B,N,1) over the patch_length dim.
        embed_input = original_patches.masked_fill(
            mask.unsqueeze(-1), 0.0)                            # (B, N, P)

        h = self.patch_embed(embed_input)                       # (B, N, d)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, d)
        h = torch.cat([cls, h], dim=1)                          # (B, N+1, d)
        h = h + self.pos_embed
        h = self.encoder(h)
        h = self.norm(h)

        cls_out   = h[:, 0, :]                                  # (B, d)
        patch_out = h[:, 1:, :]                                 # (B, N, d)

        period_pred = self.period_head(cls_out).squeeze(-1)     # (B,)
        recon_pred  = self.recon_head(patch_out)                # (B, N, P)

        return period_pred, recon_pred, original_patches, mask


def compute_dual_loss(
    period_pred: torch.Tensor,
    recon_pred: torch.Tensor,
    original_patches: torch.Tensor,
    mask: torch.Tensor,
    period_target: torch.Tensor,
    lambda_period: float,
    lambda_recon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined dual-objective loss.

    Period loss is MSE in log-T space (matches the target representation).
    Reconstruction loss is MSE averaged over MASKED patches only — the
    visible patches are trivially identity and would otherwise dilute the
    gradient signal.

    Returns:
        total: lambda_period * period_mse + lambda_recon * recon_mse
        period_mse, recon_mse: the two component scalars (for logging).
    """
    period_mse = F.mse_loss(period_pred, period_target)

    if mask.any():
        sq_err = (recon_pred - original_patches).pow(2)         # (B, N, P)
        per_patch = sq_err.mean(dim=-1)                          # (B, N)
        denom = mask.float().sum().clamp_min(1.0)
        recon_mse = (per_patch * mask.float()).sum() / denom
    else:
        recon_mse = torch.zeros((), device=period_pred.device)

    total = lambda_period * period_mse + lambda_recon * recon_mse
    return total, period_mse, recon_mse


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
) -> Optional[float]:
    """Measure peak VRAM (GB) of one fwd+bwd+step on ``device``.

    Returns None on OOM. Returns 0.0 on CPU (no measurement).
    """
    if not device.startswith("cuda"):
        return 0.0
    try:
        with torch.cuda.device(torch.device(device)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = PatchTSTDualObjective(
                context_length      = CONTEXT_LENGTH,
                patch_length        = trial.patch_length,
                d_model             = trial.d_model,
                num_hidden_layers   = trial.num_hidden_layers,
                num_attention_heads = trial.num_attention_heads,
                dropout             = trial.dropout,
                mask_ratio          = trial.mask_ratio,
            ).to(device)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

            x = torch.randn(batch_size, CONTEXT_LENGTH, 1, device=device)
            y = torch.randn(batch_size, device=device)

            period_pred, recon_pred, orig, mask = model(x)
            loss, _, _ = compute_dual_loss(
                period_pred, recon_pred, orig, mask, y,
                LAMBDA_PERIOD, LAMBDA_RECON)
            loss.backward(); opt.step()

            torch.cuda.synchronize(torch.device(device))
            peak_bytes = torch.cuda.max_memory_allocated()
            del model, opt, x, y, period_pred, recon_pred, orig, mask, loss
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
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Largest bs in BS_CANDIDATES that fits the per-device budget."""
    for bs in BS_CANDIDATES:
        peak = _probe_vram(trial, bs, device)
        if peak is None:
            continue
        if peak <= budget_gb:
            return bs, scale_lr(trial.learning_rate, bs), peak
    return None, None, None


# ==============================================================================
#  EVALUATION
# ==============================================================================

def _evaluate_dual(
    model: PatchTSTDualObjective,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    eval_seed: int,
) -> Dict[str, float]:
    """Validation pass: returns {val_period_mse, val_recon_mse, val_combined}.

    Reconstruction error is computed with a deterministic mask (seeded from
    ``eval_seed``) so the metric is comparable across trials and epochs.
    """
    model.eval()
    device = x_val.device
    n_val = x_val.shape[0]

    period_sum = torch.zeros((), device=device)
    recon_sum  = torch.zeros((), device=device)
    n_period   = 0
    n_recon    = 0

    # Pre-generate a fixed set of validation masks so the eval metric does
    # not jitter across calls.
    generator = torch.Generator(device="cpu").manual_seed(eval_seed)

    with torch.no_grad():
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            x   = x_val[start:end]
            yT  = y_val[start:end]
            B   = x.shape[0]

            # Deterministic mask for this batch (CPU rng -> bool tensor).
            n_mask = int(round(model.mask_ratio * model.num_patches))
            noise  = torch.rand(B, model.num_patches, generator=generator)
            ids    = torch.argsort(noise, dim=1)
            mask   = torch.zeros(B, model.num_patches, dtype=torch.bool)
            if n_mask > 0:
                mask.scatter_(1, ids[:, :n_mask], True)
            mask = mask.to(device, non_blocking=True)

            period_pred, recon_pred, orig, used_mask = model(x, mask=mask)

            period_sum = period_sum + F.mse_loss(
                period_pred, yT, reduction="sum")
            n_period += B

            if used_mask.any():
                sq_err = (recon_pred - orig).pow(2).mean(dim=-1)  # (B, N)
                recon_sum = recon_sum + (sq_err * used_mask.float()).sum()
                n_recon  += int(used_mask.float().sum().item())

    period_mse = (period_sum / max(n_period, 1)).item()
    recon_mse  = (recon_sum  / max(n_recon,  1)).item()
    combined   = LAMBDA_PERIOD * period_mse + LAMBDA_RECON * recon_mse
    return {
        "val_period_mse": period_mse,
        "val_recon_mse":  recon_mse,
        "val_combined":   combined,
    }


# ==============================================================================
#  TRIAL RUNNER
# ==============================================================================

def _run_single_trial(
    trial_idx: int,
    trial: TrialConfig,
    vram_budget_gb: float,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val:   torch.Tensor,
    y_val:   torch.Tensor,
    device:  str,
) -> Dict[str, Any]:
    """Train one configuration end-to-end and return the best-val checkpoint."""
    bs, lr, peak = auto_select_bs_lr(trial, device, vram_budget_gb)
    if bs is None:
        return {
            "trial_idx": trial_idx, "device": device,
            "val_period_mse": float("nan"),
            "val_recon_mse":  float("nan"),
            "val_combined":   float("nan"),
            "history": {"train_total": [], "train_period": [],
                        "train_recon": [], "val_combined": [],
                        "val_period": [], "val_recon": [],
                        "val_epochs": []},
            "best_state_path": None,
            "failed": True, "skip_reason": "vram_oom",
            "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
            "elapsed_seconds": 0.0, "cfg": asdict(trial),
        }

    tag = f"[{device}] trial {trial_idx:03d}"
    print(Fore.CYAN
          + f"  {tag}: bs={bs} lr={lr:.2e} peak={peak:.2f}GB  "
          + f"budget={vram_budget_gb:.1f}GB" + Fore.RESET)

    t0 = time.perf_counter()
    n_train = x_train.shape[0]
    steps_per_epoch = n_train // bs

    model = PatchTSTDualObjective(
        context_length      = CONTEXT_LENGTH,
        patch_length        = trial.patch_length,
        d_model             = trial.d_model,
        num_hidden_layers   = trial.num_hidden_layers,
        num_attention_heads = trial.num_attention_heads,
        dropout             = trial.dropout,
        mask_ratio          = trial.mask_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=trial.weight_decay)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Dict[str, float] = {}
    patience_left = EARLY_STOPPING_PATIENCE
    history: Dict[str, List[float]] = {
        "train_total": [], "train_period": [], "train_recon": [],
        "val_combined": [], "val_period": [], "val_recon": [],
        "val_epochs": [],
    }

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            run_total  = torch.zeros((), device=device)
            run_period = torch.zeros((), device=device)
            run_recon  = torch.zeros((), device=device)
            n_seen = 0

            for step in range(steps_per_epoch):
                idx = perm[step * bs : (step + 1) * bs]
                x   = x_train.index_select(0, idx)
                y   = y_train.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)
                period_pred, recon_pred, orig, mask = model(x)
                loss, l_period, l_recon = compute_dual_loss(
                    period_pred, recon_pred, orig, mask, y,
                    LAMBDA_PERIOD, LAMBDA_RECON)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                run_total  = run_total  + loss.detach()    * bs
                run_period = run_period + l_period.detach()* bs
                run_recon  = run_recon  + l_recon.detach() * bs
                n_seen    += bs

            denom = max(n_seen, 1)
            train_total  = (run_total  / denom).item()
            train_period = (run_period / denom).item()
            train_recon  = (run_recon  / denom).item()
            history["train_total"].append(train_total)
            history["train_period"].append(train_period)
            history["train_recon"].append(train_recon)

            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == MAX_EPOCHS:
                metrics = _evaluate_dual(
                    model, x_val, y_val, bs,
                    eval_seed=SEED + 1000 * trial_idx)
                history["val_combined"].append(metrics["val_combined"])
                history["val_period"].append(metrics["val_period_mse"])
                history["val_recon"].append(metrics["val_recon_mse"])
                history["val_epochs"].append(epoch)
                print(Fore.YELLOW
                      + f"  {tag} epoch {epoch:3d}  "
                      + f"train: total={train_total:.4f} "
                      + f"period={train_period:.4f} recon={train_recon:.4f}  "
                      + f"val: combined={metrics['val_combined']:.4f} "
                      + f"period={metrics['val_period_mse']:.4f} "
                      + f"recon={metrics['val_recon_mse']:.4f}"
                      + Fore.RESET)

                # Selection: minimize the chosen metric.
                if SELECTION_METRIC == "combined":
                    score = metrics["val_combined"]
                elif SELECTION_METRIC == "period":
                    score = metrics["val_period_mse"]
                elif SELECTION_METRIC == "recon":
                    score = metrics["val_recon_mse"]
                else:
                    raise ValueError(f"Unknown SELECTION_METRIC={SELECTION_METRIC!r}")

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

    # Persist best checkpoint to a per-trial temp file (path returned through
    # the queue; the parent decides which to keep after selection).
    best_state_path: Optional[str] = None
    if best_state is not None:
        fd, best_state_path = tempfile.mkstemp(
            prefix=f"patchtst_dual_trial{trial_idx:03d}_", suffix=".pt")
        os.close(fd)
        torch.save(best_state, best_state_path)
        del best_state

    elapsed = time.perf_counter() - t0
    del optimizer, model
    torch.cuda.empty_cache()

    return {
        "trial_idx": trial_idx, "device": device,
        "val_period_mse": best_metrics.get("val_period_mse", float("nan")),
        "val_recon_mse":  best_metrics.get("val_recon_mse",  float("nan")),
        "val_combined":   best_metrics.get("val_combined",   float("nan")),
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
    n_train: int,
    n_val: int,
) -> None:
    """One worker per GPU. Builds synthetic data locally, then drains trials."""
    set_seed(SEED + worker_id)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    torch.set_float32_matmul_precision("high")

    try:
        # Workers use disjoint base seeds so the union covers more of the
        # generative manifold than any single worker alone.
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = build_train_val_tensors(
            n_train=n_train, n_val=n_val,
            context_length=CONTEXT_LENGTH,
            seed=SEED + 1_000_000 * (worker_id + 1),
        )
        x_train = x_train_cpu.pin_memory().to(device, non_blocking=True)
        y_train = y_train_cpu.pin_memory().to(device, non_blocking=True)
        x_val   = x_val_cpu  .pin_memory().to(device, non_blocking=True)
        y_val   = y_val_cpu  .pin_memory().to(device, non_blocking=True)
        del x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu
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
            result_queue.put({
                "trial_idx": trial_idx, "device": device,
                "val_period_mse": float("nan"),
                "val_recon_mse":  float("nan"),
                "val_combined":   float("nan"),
                "history": {}, "best_state_path": None,
                "failed": True, "skip_reason": "worker_setup_failed",
                "auto_batch_size": None, "auto_lr": None,
                "peak_vram_gb": None,
                "elapsed_seconds": 0.0, "cfg": asdict(trial),
            })
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
                x_train, y_train, x_val, y_val, device,
            )
        except Exception as exc:
            print(Fore.RED + f"  [{device}] trial {trial_idx:03d} "
                  + f"CRASHED: {type(exc).__name__}: {exc}" + Fore.RESET)
            result = {
                "trial_idx": trial_idx, "device": device,
                "val_period_mse": float("nan"),
                "val_recon_mse":  float("nan"),
                "val_combined":   float("nan"),
                "history": {}, "best_state_path": None,
                "failed": True,
                "skip_reason": f"trial_exception:{type(exc).__name__}",
                "auto_batch_size": None, "auto_lr": None,
                "peak_vram_gb": None,
                "elapsed_seconds": 0.0, "cfg": asdict(trial),
            }
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        result_queue.put(result)

    del x_train, y_train, x_val, y_val
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(Fore.CYAN + f"  [{device}] worker {worker_id} exited." + Fore.RESET)


# ==============================================================================
#  PLOTTING
# ==============================================================================

def _plot_sweep_summary(
    trials_df: pd.DataFrame,
    save_path: str,
    run_label: str,
) -> None:
    """Three-panel scatter: val_combined, val_period_mse, val_recon_mse.

    Each panel shows individual trials (faded scatter) and the running best
    (solid red), with annotation of the best trial.
    """
    df = trials_df.sort_values("trial_idx").reset_index(drop=True)
    panels = [
        ("val_combined",   "val combined loss"),
        ("val_period_mse", "val period MSE (log-T)"),
        ("val_recon_mse",  "val reconstruction MSE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (col, label) in zip(axes, panels):
        sub = df.dropna(subset=[col])
        if sub.empty:
            ax.set_title(f"{label}  (no valid trials)")
            continue
        # Running best
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
    fig.suptitle(f"Dual-objective Patch-Transformer sweep — {run_label}",
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

def _best_model_path(run_label: str) -> str:
    return os.path.join(_run_dir(run_label), "best_model.pt")

def _best_config_path(run_label: str) -> str:
    return os.path.join(_run_dir(run_label), "best_config.json")

def _selection_report_path(run_label: str) -> str:
    return os.path.join(_run_dir(run_label), "selection_report.json")

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
#  SELECTION
# ==============================================================================

def _select_final(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pick the final trial by SELECTION_METRIC over those with a checkpoint."""
    metric_key = {
        "combined": "val_combined",
        "period":   "val_period_mse",
        "recon":    "val_recon_mse",
    }[SELECTION_METRIC]

    valid = [r for r in results
             if not r.get("failed", False)
             and r.get("best_state_path")
             and os.path.isfile(r["best_state_path"])
             and not (isinstance(r.get(metric_key), float)
                      and math.isnan(r[metric_key]))]
    if not valid:
        return {}, {
            "selection_metric": metric_key,
            "note": "no_valid_trials_with_checkpoints",
        }

    chosen = min(valid, key=lambda r: r[metric_key])
    report = {
        "selection_metric":   metric_key,
        "selected_trial_idx": int(chosen["trial_idx"]),
        "selected_metrics": {
            "val_period_mse": chosen.get("val_period_mse"),
            "val_recon_mse":  chosen.get("val_recon_mse"),
            "val_combined":   chosen.get("val_combined"),
        },
        "n_valid":  len(valid),
        "n_total":  len(results),
        "lambda_period": LAMBDA_PERIOD,
        "lambda_recon":  LAMBDA_RECON,
    }
    return chosen, report


# ==============================================================================
#  ARTIFACTS
# ==============================================================================

def _persist_artifacts(
    results: List[Dict[str, Any]],
    run_label: str,
) -> None:
    rdir = _run_dir(run_label)
    os.makedirs(os.path.join(rdir, "trials"), exist_ok=True)
    for r in results:
        _save_trial_result(run_label, r["trial_idx"], r)

    rows = []
    for r in results:
        flat = {"trial_idx": r["trial_idx"]}
        flat.update(r.get("cfg", {}))
        for key in ("val_period_mse", "val_recon_mse", "val_combined",
                    "auto_batch_size", "auto_lr", "peak_vram_gb",
                    "device", "elapsed_seconds", "skip_reason", "failed"):
            if key in r:
                flat[key] = r[key]
        rows.append(flat)
    df = pd.DataFrame(rows)
    keep = ["trial_idx", "val_combined", "val_period_mse", "val_recon_mse",
            "patch_length", "d_model", "num_hidden_layers",
            "num_attention_heads", "dropout", "mask_ratio",
            "learning_rate", "weight_decay",
            "auto_batch_size", "auto_lr", "peak_vram_gb",
            "device", "elapsed_seconds", "failed", "skip_reason"]
    df_csv = df[[c for c in keep if c in df.columns]]
    csv_path = os.path.join(rdir, "sweep_summary.csv")
    df_csv.to_csv(csv_path, index=False)
    print(Fore.GREEN + f"  Sweep CSV: {csv_path}" + Fore.RESET)

    df_plot = df_csv.copy()
    if len(df_plot) > 0:
        _plot_sweep_summary(df_plot,
                            os.path.join(rdir, "sweep_summary.png"),
                            run_label)


# ==============================================================================
#  MAIN
# ==============================================================================

def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    set_seed(SEED)
    devices      = resolve_devices()
    vram_budgets = resolve_vram_budgets(devices)

    print(Fore.CYAN + f"Devices: {devices}" + Fore.RESET)
    print(Fore.CYAN + f"VRAM budgets (GB) per device: "
          + f"{dict(zip(devices, vram_budgets))}" + Fore.RESET)
    print(Fore.CYAN + f"LR scaling: {LR_SCALING_RULE} (bs_ref={BS_REFERENCE})  |  "
          + f"BS candidates: {BS_CANDIDATES}" + Fore.RESET)
    print(Fore.CYAN + f"N_TRIALS={N_TRIALS}   "
          + f"lambda_period={LAMBDA_PERIOD}  lambda_recon={LAMBDA_RECON}   "
          + f"selection={SELECTION_METRIC}" + Fore.RESET)
    print(Fore.CYAN + f"Context length L={CONTEXT_LENGTH}  "
          + f"period range=[{PERIOD_MIN}, {PERIOD_MAX}]" + Fore.RESET)

    run_label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
                and "val_combined" in cached
                and not (isinstance(cached["val_combined"], float)
                         and math.isnan(cached["val_combined"]))):
            print(Fore.WHITE + f"  CACHED trial {idx:03d}: "
                  + f"val_combined={cached['val_combined']:.4f}" + Fore.RESET)
            cached["best_state_path"] = None     # stripped on save; rerun if needed
            cached_results.append(cached)
        else:
            pending.append((idx, trial))

    # ---------- Spawn workers ----------------------------------------------
    ctx = mp.get_context("spawn")
    trial_queue  = ctx.Queue()
    result_queue = ctx.Queue()
    workers: List[mp.Process] = []
    for i, (device, budget) in enumerate(zip(devices, vram_budgets)):
        p = ctx.Process(
            target=gpu_worker,
            args=(i, device, budget, trial_queue, result_queue,
                  N_TRAIN_WINDOWS, N_VAL_WINDOWS),
            name=f"gpu_worker_{i}_{device.replace(':', '')}",
        )
        p.start(); workers.append(p)

    # ---------- Dispatch ----------------------------------------------------
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
            alive = [p for p in workers if p.is_alive()]
            if not alive:
                print(Fore.RED + f"  All workers died with "
                      + f"{n_received}/{n_expected} results." + Fore.RESET)
                break
            print(Fore.YELLOW + f"  No result in 1h "
                  + f"({n_received}/{n_expected}); "
                  + f"{len(alive)} worker(s) alive." + Fore.RESET)
            continue
        fresh_results.append(r); n_received += 1

    elapsed = time.perf_counter() - t_start
    print(Fore.MAGENTA
          + f"  Sweep wall-clock: {elapsed:.1f}s  "
          + f"({n_received}/{n_expected} fresh trials)" + Fore.RESET)

    # ---------- Tear down workers -----------------------------------------
    for _ in workers: trial_queue.put(None)
    for p in workers:
        p.join(timeout=120)
        if p.is_alive():
            print(Fore.RED + f"  Worker {p.name} still alive — terminating."
                  + Fore.RESET)
            p.terminate(); p.join(timeout=10)

    all_results = cached_results + fresh_results
    _persist_artifacts(all_results, run_label)

    # ---------- Final selection (only fresh trials carry checkpoints) -----
    chosen, report = _select_final(fresh_results)
    if chosen:
        shutil.copyfile(chosen["best_state_path"], _best_model_path(run_label))
        with open(_best_config_path(run_label), "w") as f:
            json.dump({
                "trial_idx":         int(chosen["trial_idx"]),
                **chosen.get("cfg", {}),
                "val_period_mse":    chosen.get("val_period_mse"),
                "val_recon_mse":     chosen.get("val_recon_mse"),
                "val_combined":      chosen.get("val_combined"),
                "selection_metric":  report["selection_metric"],
                "auto_batch_size":   chosen.get("auto_batch_size"),
                "auto_lr":           chosen.get("auto_lr"),
                "context_length":    CONTEXT_LENGTH,
                "lambda_period":     LAMBDA_PERIOD,
                "lambda_recon":      LAMBDA_RECON,
                "period_min":        PERIOD_MIN,
                "period_max":        PERIOD_MAX,
            }, f, indent=2)
        with open(_selection_report_path(run_label), "w") as f:
            json.dump(report, f, indent=2)
        sel_metric_key = report["selection_metric"]
        print(Fore.GREEN + f"  FINAL SELECTION: trial "
              + f"{int(chosen['trial_idx']):03d}  "
              + f"({sel_metric_key}={chosen.get(sel_metric_key):.4f})"
              + Fore.RESET)
    else:
        print(Fore.YELLOW + "  No valid fresh trials — no checkpoint persisted."
              + Fore.RESET)

    # Clean up per-trial temp checkpoints.
    for r in fresh_results:
        p = r.get("best_state_path")
        if p and os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    print(Fore.GREEN + "\nDual-objective training pipeline done." + Fore.RESET)


if __name__ == "__main__":
    main()