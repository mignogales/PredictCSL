"""
Evaluation pools for exp0/1/2/3/5 (exp4 generates its own control data).

Four sources, mirroring the rest of the repo:
  * synthetic — the stage-1 non-stationary generator
    (``experiments.build_context_length_dataset.generate_dataset``). Short /
    left-padded series are EXCLUDED here: interventions on artificial padding
    would confound the block effects (recorded in run_meta by the caller).
  * harmonic — clean random-phase sinusoid mixtures at controlled scales.
  * kernelsynth — Chronos-style random-composite GP samples.
  * gifteval — the v5 ablation cache (server only). Instances shorter than the
    largest requested context are dropped so every sample supports every
    effective context length with genuine data.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from context_interpretability.experiments.common import ExperimentData


def load_synthetic(n_series: int, seed: int, horizon: int,
                   name: str = "synthetic") -> ExperimentData:
    from experiments.build_context_length_dataset import (
        MAX_WINDOW, generate_dataset)
    # Oversample, then keep full-length series only (see module docstring).
    contexts, targets, _nseg, real_lengths = generate_dataset(
        int(n_series * 1.3) + 8, seed)
    keep = np.flatnonzero(real_lengths >= MAX_WINDOW)[:n_series]
    if keep.size < n_series:
        print(f"[loaders] synthetic: only {keep.size}/{n_series} full-length "
              "series available")
    return ExperimentData(
        name=name,
        contexts=contexts[keep].astype(np.float32),
        targets=targets[keep, :horizon].astype(np.float32),
        sample_ids=[f"synth_{seed}_{int(i)}" for i in keep],
        season_length=1,
        metadata={"source": "generate_dataset", "seed": seed},
    )


def load_harmonic(config: dict, max_samples: int, horizon: int
                  ) -> List[ExperimentData]:
    from context_interpretability.data.validation_generators import (
        DEFAULT_PERIODS, load_harmonic_pools)
    n = min(int(config.get("n_series", 48)), int(max_samples))
    return load_harmonic_pools(
        n_instances=n,
        series_length=int(config.get("series_length", 8192)),
        horizon=int(horizon), seed=int(config.get("seed", 42)),
        periods=config.get("periods") or DEFAULT_PERIODS,
        scales=config.get("scales") or (0.5, 1.0, 2.0),
        max_tones=int(config.get("max_tones", 4)),
    )


def load_kernelsynth(config: dict, max_samples: int, horizon: int
                     ) -> ExperimentData:
    from context_interpretability.data.validation_generators import (
        load_kernelsynth_pool)
    n = min(int(config.get("n_series", 32)), int(max_samples))
    return load_kernelsynth_pool(
        n_instances=n,
        series_length=int(config.get("series_length", 1024)),
        horizon=int(horizon), seed=int(config.get("seed", 42)),
        max_kernels=int(config.get("max_kernels", 5)),
        normalize_std=bool(config.get("normalize_std", True)),
        cache_path=config.get("cache_path"),
        scalable=bool(config.get("scalable", False)),
    )


def load_gifteval(dataset_names: List[str], term: str, horizon: int,
                  max_instances: int, min_context: int
                  ) -> List[ExperimentData]:
    """One ExperimentData per GiftEval dataset (SERVER only — needs GIFT_EVAL).

    Contexts come NaN->0-filled exactly as the ablation feeds the models
    (``GiftEvalCache.contexts``); the true horizon labels come from
    ``labels_np``. The gluonts season length is carried for MASE scoring.
    """
    import experiments.test_window_ablation_gifteval_v5 as abl

    out: List[ExperimentData] = []
    for ge_name, ds_term, display, to_univariate in abl.DATASETS:
        if ds_term != term or (display not in dataset_names
                               and ge_name not in dataset_names):
            continue
        ge = abl.GiftEvalDataset(name=ge_name, term=ds_term,
                                 to_univariate=to_univariate)
        cache = abl.GiftEvalCache(ge, display)
        if cache.labels_np.shape[1] < horizon:
            print(f"[loaders] gifteval {display}: native horizon "
                  f"{cache.labels_np.shape[1]} < requested {horizon} — skipped")
            continue
        ctxs, tgts, ids = [], [], []
        for i, c in enumerate(cache.contexts):
            if len(ctxs) >= max_instances:
                break
            if c.shape[0] < min_context:
                continue
            y = cache.labels_np[i]
            if y.shape[0] < horizon:
                continue
            ctxs.append(np.asarray(c[-min_context:], dtype=np.float32))
            tgts.append(np.asarray(y[:horizon], dtype=np.float32))
            ids.append(f"{display}_t{term}_{i}")
        if not ctxs:
            print(f"[loaders] gifteval {display}: no instance with >= "
                  f"{min_context} context — skipped")
            continue
        out.append(ExperimentData(
            name=f"{display.replace('/', '_')}_{term}",
            contexts=np.stack(ctxs), targets=np.stack(tgts), sample_ids=ids,
            season_length=int(getattr(cache, "season_gluonts", 1) or 1),
            metadata={"source": "gifteval", "term": term},
        ))
    return out
