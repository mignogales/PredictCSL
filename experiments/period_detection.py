"""GiftEval-cadence period detection without model or dataset dependencies."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


GIFT_EVAL_PERIOD_LABELS: Tuple[str, ...] = (
    "10S", "5T", "10T", "15T", "H", "D", "W", "M", "Q", "A",
)

# Calendar units share one numerical year so M -> Q -> A maps exactly to
# 3 -> 4 -> 12, while daily/sub-daily conversions retain calendar-scale values.
_YEAR_SECONDS = 365.25 * 24.0 * 60.0 * 60.0
_UNIT_SECONDS = {
    "S": 1.0,
    "T": 60.0,
    "H": 60.0 * 60.0,
    "D": 24.0 * 60.0 * 60.0,
    "W": 7.0 * 24.0 * 60.0 * 60.0,
    "M": _YEAR_SECONDS / 12.0,
    "Q": _YEAR_SECONDS / 4.0,
    "A": _YEAR_SECONDS,
}


def _frequency_seconds(freq: str) -> float:
    """Translate a pandas/GluonTS cadence label to representative seconds."""
    head = str(freq).strip().upper().split("-")[0]
    head = head.replace("MIN", "T")
    aliases = {"ME": "M", "QE": "Q", "YE": "A", "Y": "A"}
    digits = "".join(ch for ch in head if ch.isdigit())
    multiplier = int(digits) if digits else 1
    unit = aliases.get(head[len(digits):], head[len(digits):])
    if unit not in _UNIT_SECONDS:
        raise ValueError(f"Unsupported sampling frequency: {freq!r}")
    return multiplier * _UNIT_SECONDS[unit]


def cadence_candidates(
    sampling_freq: str,
    context_length: int,
    labels: Sequence[str] = GIFT_EVAL_PERIOD_LABELS,
) -> List[Tuple[str, int]]:
    """Return GiftEval cadence labels translated to feasible sample counts."""
    sample_seconds = _frequency_seconds(sampling_freq)
    out: List[Tuple[str, int]] = []
    seen = set()
    for label in labels:
        n_samples = int(round(_frequency_seconds(label) / sample_seconds))
        if n_samples < 1 or 2 * n_samples > context_length or n_samples in seen:
            continue
        seen.add(n_samples)
        out.append((label, n_samples))
    return out


def window_similarity(x: np.ndarray, window: int) -> float:
    """Mean Pearson correlation of consecutive, non-overlapping tail chunks."""
    n_chunks = x.size // window
    if n_chunks < 2:
        return float("-inf")
    chunks = x[-n_chunks * window:].reshape(n_chunks, window)
    scores: List[float] = []
    for left, right in zip(chunks[:-1], chunks[1:]):
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.sum() < 2:
            continue
        a = left[finite] - np.mean(left[finite])
        b = right[finite] - np.mean(right[finite])
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 1e-12:
            scores.append(float(np.dot(a, b) / denom))
    return float(np.mean(scores)) if scores else float("-inf")


def detect_period(
    x: np.ndarray,
    sampling_freq: str,
    season_fallback: int,
) -> Tuple[float, str, Dict[str, float]]:
    """Return the most self-similar GiftEval cadence in sample units."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    candidates = cadence_candidates(sampling_freq, x.size)
    scores = {label: window_similarity(x, size) for label, size in candidates}
    usable = [(label, size) for label, size in candidates
              if np.isfinite(scores[label])]
    if not usable:
        return float(max(1, season_fallback)), "fallback_season", scores

    # A multiple of the fundamental can correlate equally well. Prefer the
    # shortest cadence within numerical tolerance of the maximum.
    best_score = max(scores[label] for label, _ in usable)
    label, size = next(
        item for item in usable if scores[item[0]] >= best_score - 1e-12)
    return float(size), label, scores
