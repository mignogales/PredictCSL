"""
Synthetic distant-dependency controls (spec §7) — the mandatory control that
distinguishes "the model adaptively ignores uninformative distant context"
from "the architecture cannot use distant context".

Three families, all univariate, all returning (contexts (N, T), targets (N, H))
tail-aligned like every other pool:

  A. local:            X_t = f(X_{t-r..t-1}) + eps                (linear AR /
                       nonlinear / seasonal-local)
  B. local + distant:  X_t = f(local) + strength * g(X_{t-d}) + eps,  d >> r
  C. instance-conditional: the distant term is gated by a per-instance latent
                       G_i ~ Bernoulli(0.5).

Construction notes
------------------
* The linear distant term is STABILITY-RESCALED: local coefficients carry total
  mass ``LOCAL_MASS`` and the distant coefficient is
  ``strength * (STABLE_CAP - LOCAL_MASS)``, keeping the AR polynomial inside
  the unit circle for every configured strength.
* Distant information must be genuinely non-recoverable from the recent
  window; :func:`verify_distant_information` checks this EMPIRICALLY with two
  ridge oracles (recent-window-only vs recent+lag-d features) — run it for
  every generated configuration before interpreting model results.
* Family C rescales each series to unit variance so gated / ungated instances
  have comparable marginal scale and cannot be told apart by amplitude alone
  (valid for the linear ``g``; the ``sin`` variant notes the caveat).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np

LOCAL_MASS = 0.6          # total |coefficient| mass of the local AR part
STABLE_CAP = 0.95         # local + distant mass stays under this
BURN_IN = 512


@dataclasses.dataclass(frozen=True)
class ControlSpec:
    family: str                       # "A" | "B" | "C"
    local_kind: str = "linear"        # linear | nonlinear | seasonal
    local_order: int = 8              # r
    distant_lag: int = 0              # d (0 => no distant term)
    strength: float = 0.0             # dependency strength in [0, 1]
    distant_kind: str = "linear"      # linear | sin
    noise: float = 0.1
    season: int = 24                  # used by local_kind == seasonal

    @property
    def name(self) -> str:
        bits = [f"fam{self.family}", self.local_kind, f"r{self.local_order}"]
        if self.family in ("B", "C"):
            bits += [f"d{self.distant_lag}", f"s{self.strength:g}",
                     self.distant_kind]
        bits.append(f"n{self.noise:g}")
        return "_".join(bits)


def _local_coeffs(spec: ControlSpec, rng: np.random.Generator) -> np.ndarray:
    """Decaying positive AR coefficients with total mass LOCAL_MASS."""
    raw = np.exp(-np.arange(spec.local_order) / 2.0)
    raw = raw * (0.75 + 0.5 * rng.random(spec.local_order))
    return raw / raw.sum() * LOCAL_MASS


def _simulate(spec: ControlSpec, length: int, gated: bool,
              rng: np.random.Generator) -> np.ndarray:
    r, d = spec.local_order, spec.distant_lag
    total = length + BURN_IN
    a = _local_coeffs(spec, rng)
    b = spec.strength * (STABLE_CAP - LOCAL_MASS) if (gated and d > 0) else 0.0
    x = np.zeros(total)
    x[:max(r, d, 1)] = rng.normal(0, spec.noise, size=max(r, d, 1))
    eps = rng.normal(0, spec.noise, size=total)
    phase = rng.uniform(0, 2 * np.pi)
    for t in range(max(r, d, 1), total):
        local = float(a @ x[t - r:t][::-1])
        if spec.local_kind == "nonlinear":
            local += 0.2 * np.sin(x[t - 1])
        elif spec.local_kind == "seasonal":
            local += 0.3 * np.sin(2 * np.pi * t / spec.season + phase)
        distant = 0.0
        if b > 0.0:
            src = x[t - d]
            distant = b * (np.sin(3.0 * src) if spec.distant_kind == "sin"
                           else src)
        x[t] = local + distant + eps[t]
    return x[BURN_IN:]


def generate_control(spec: ControlSpec, n_instances: int, series_length: int,
                     horizon: int, seed: int = 0
                     ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Returns (contexts (N, T), targets (N, H), gates (N,) or None).

    Family A: no distant term. Family B: every instance gated on. Family C:
    ~50% gated (G_i returned) with per-series unit-variance rescaling.
    """
    T = int(series_length)
    total = T + int(horizon)
    contexts = np.empty((n_instances, T), dtype=np.float32)
    targets = np.empty((n_instances, horizon), dtype=np.float32)
    gates: Optional[np.ndarray] = None
    if spec.family == "C":
        gates = (np.random.default_rng(seed).random(n_instances) < 0.5)

    for i in range(n_instances):
        rng = np.random.default_rng((seed, i))
        gated = bool(gates[i]) if gates is not None else (spec.family == "B")
        x = _simulate(spec, total, gated, rng)
        if spec.family == "C":
            sd = x.std()
            if sd > 0:
                x = x / sd            # comparable marginal scales (spec §7.4)
        contexts[i] = x[:T]
        targets[i] = x[T:total]
    return contexts, targets, gates


# ------------------------------------------------------------------------------
#  Oracle verification (spec §7.3): distant info must be genuinely predictive
#  and NOT recoverable from the recent window.
# ------------------------------------------------------------------------------

def _lag_matrix(x: np.ndarray, lags: List[int], t_idx: np.ndarray
                ) -> np.ndarray:
    return np.stack([x[t_idx - l] for l in lags], axis=1)


def _ridge_mse(feats_tr, y_tr, feats_te, y_te, alpha: float) -> float:
    A = feats_tr.T @ feats_tr + alpha * np.eye(feats_tr.shape[1])
    w = np.linalg.solve(A, feats_tr.T @ y_tr)
    return float(np.mean((feats_te @ w - y_te) ** 2))


def verify_distant_information(contexts: np.ndarray, spec: ControlSpec,
                               recent_window: Optional[int] = None,
                               alpha: float = 1.0, seed: int = 0
                               ) -> Dict[str, float]:
    """Compare a recent-window-only ridge oracle with one that also sees lag d.

    Returns oracle MSEs + relative gain. For strength == 0 (or family A) the
    gain should be ~0; for strength > 0 it must be positive — callers treat a
    non-positive gain as a broken configuration, not as a model finding.
    """
    r = spec.local_order
    d = spec.distant_lag
    win = int(recent_window or max(2 * r, 16))
    recent_lags = list(range(1, win + 1))
    rng = np.random.default_rng(seed)

    feats_recent, feats_full, ys = [], [], []
    for i in range(contexts.shape[0]):
        x = contexts[i].astype(np.float64)
        lo = max(win, d) + 1
        t_idx = rng.integers(lo, len(x), size=min(64, len(x) - lo))
        feats_recent.append(_lag_matrix(x, recent_lags, t_idx))
        extra = _lag_matrix(x, [d, d - 1, d + 1], t_idx) if d > 0 else \
            np.zeros((len(t_idx), 0))
        feats_full.append(np.concatenate(
            [_lag_matrix(x, recent_lags, t_idx), extra], axis=1))
        ys.append(x[t_idx])
    Xr = np.concatenate(feats_recent)
    Xf = np.concatenate(feats_full)
    y = np.concatenate(ys)
    n_tr = int(0.7 * len(y))
    perm = rng.permutation(len(y))
    tr, te = perm[:n_tr], perm[n_tr:]

    mse_recent = _ridge_mse(Xr[tr], y[tr], Xr[te], y[te], alpha)
    mse_full = _ridge_mse(Xf[tr], y[tr], Xf[te], y[te], alpha)
    gain = (mse_recent - mse_full) / max(mse_recent, 1e-12)
    return {"oracle_mse_recent": mse_recent, "oracle_mse_full": mse_full,
            "relative_gain": float(gain),
            "distant_predictive": bool(gain > 0.02)}


def gate_detectable(contexts: np.ndarray, gates: np.ndarray,
                    spec: ControlSpec, seed: int = 0) -> float:
    """Family C requirement: G_i must be inferable BEFORE forecasting. Returns
    the held-out accuracy of a lag-d autocorrelation threshold classifier."""
    d = spec.distant_lag
    scores = np.empty(contexts.shape[0])
    for i in range(contexts.shape[0]):
        x = contexts[i].astype(np.float64)
        x0, xd = x[d:], x[:-d]
        x0 = x0 - x0.mean()
        xd = xd - xd.mean()
        denom = np.sqrt((x0 ** 2).sum() * (xd ** 2).sum())
        scores[i] = (x0 * xd).sum() / denom if denom > 0 else 0.0
    thresh = np.median(scores)
    pred = scores > thresh
    acc = max(np.mean(pred == gates), np.mean(pred == ~gates))
    return float(acc)
