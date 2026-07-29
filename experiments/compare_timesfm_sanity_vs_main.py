"""Compare TimesFM sanity-leaderboard and main-ablation implementations.

This is a diagnostic script, not part of the production pipeline. It runs one
GiftEval config through:

  1. the official-style sanity wrapper (TimesFmPredictor / tfm.forecast), and
  2. the main v5 ablation wrapper (the shared official ``tfm.forecast`` recipe),

then reports where they differ: data selection, effective contexts, model inputs,
forecasts, and leaderboard-style MASE.

Examples
--------
    python -m experiments.compare_timesfm_sanity_vs_main \
        --dataset m4_weekly --term short --n 16

    python -m experiments.compare_timesfm_sanity_vs_main \
        --dataset electricity/15T --term short --max-context 8192 \
        --main-compile-horizon 1024 --n 32

    python -m experiments.compare_timesfm_sanity_vs_main \
        --dataset electricity/15T --term short --main-window-size 8192
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import experiments.sanity_gifteval_leaderboard as sanity


TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
np = None
torch = None


def _require_numpy():
    global np
    if np is None:
        np = sanity._require_numpy("run TimesFM sanity/main comparison")
    return np


def _require_torch():
    global torch
    if torch is None:
        torch = sanity._require_torch("run TimesFM sanity/main comparison")
    return torch


def _wab():
    import experiments.test_window_ablation_gifteval_v5 as wab

    return wab


def _gluonts_leaderboard_mase():
    from experiments.gifteval_mase import gluonts_leaderboard_mase

    return gluonts_leaderboard_mase


def _jsonable(x):
    np = _require_numpy()

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _array_summary(a, max_items: int = 6) -> Dict:
    np = _require_numpy()

    arr = np.asarray(a)
    flat = arr.reshape(-1) if arr.size else arr
    finite = flat[np.isfinite(flat)] if np.issubdtype(arr.dtype, np.number) else flat
    out = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0,
        "head": flat[:max_items].tolist(),
        "tail": flat[-max_items:].tolist() if flat.size else [],
    }
    if finite.size and np.issubdtype(arr.dtype, np.number):
        out.update({
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
        })
    return out


def _diff_summary(a, b) -> Dict:
    np = _require_numpy()

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return {"shape_match": False, "a_shape": list(aa.shape), "b_shape": list(bb.shape)}
    valid = np.isfinite(aa) & np.isfinite(bb)
    if not valid.any():
        return {"shape_match": True, "finite_overlap": 0}
    d = np.abs(aa[valid] - bb[valid])
    return {
        "shape_match": True,
        "finite_overlap": int(valid.sum()),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "p95_abs": float(np.quantile(d, 0.95)),
        "allclose_1e-5": bool(np.allclose(aa[valid], bb[valid], atol=1e-5, rtol=1e-5)),
    }


def _load_dataset(ds_name: str, term: str):
    from gift_eval.data import Dataset

    wab = _wab()
    to_univariate = Dataset(name=ds_name, term=term, to_univariate=False).target_dim > 1
    dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
    cache = wab.GiftEvalCache(dataset, ds_name)
    pairs = []
    for test_input, test_label in dataset.test_data:
        label = test_label["target"]
        if len(label) >= dataset.prediction_length:
            pairs.append((test_input, test_label))
    if len(pairs) != cache.n_total:
        raise RuntimeError(
            f"Dataset pair/cache mismatch: pairs={len(pairs)} cache={cache.n_total}")
    return dataset, cache, pairs


def _select_indices(cache, n: Optional[int], indices: Optional[str]) -> np.ndarray:
    np = _require_numpy()

    if indices:
        vals = [int(x) for x in indices.split(",") if x.strip()]
        idx = np.asarray(vals, dtype=np.int64)
    elif n is not None and n > 0:
        idx = np.arange(min(int(n), cache.n_total), dtype=np.int64)
    else:
        idx = np.arange(cache.n_total, dtype=np.int64)
    if idx.size == 0:
        raise RuntimeError("No selected instances.")
    if idx.min() < 0 or idx.max() >= cache.n_total:
        raise RuntimeError(f"Selected index out of range 0..{cache.n_total - 1}: {idx}")
    return idx


def _effective_contexts_official(pairs, indices, max_context: Optional[int]) -> List[np.ndarray]:
    contexts = []
    for i in indices:
        arr = sanity.TimesFmPredictor._model_context(pairs[int(i)][0]["target"])
        if max_context is not None:
            arr = arr[-int(max_context):]
            arr = sanity.TimesFmPredictor._model_context(arr)
        contexts.append(arr)
    return contexts


def _effective_contexts_main(cache, indices, cap: int) -> List[np.ndarray]:
    np = _require_numpy()

    contexts = []
    for i in indices:
        ctx = np.asarray(cache.contexts_raw[int(i)], dtype=np.float32)
        width = max(1, min(int(cap), int(cache.context_lengths[int(i)])))
        if ctx.size >= width:
            contexts.append(ctx[-width:])
        else:
            contexts.append(np.concatenate([
                np.zeros(width - ctx.size, dtype=np.float32),
                ctx,
            ]))
    return contexts


def _context_report(name: str, contexts: Sequence[np.ndarray]) -> Dict:
    np = _require_numpy()

    lengths = np.asarray([len(c) for c in contexts], dtype=np.int64)
    nan_counts = np.asarray([int(np.isnan(c).sum()) for c in contexts], dtype=np.int64)
    return {
        "name": name,
        "n": int(len(contexts)),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "length_unique_count": int(np.unique(lengths).size),
        "nan_total": int(nan_counts.sum()),
        "nan_contexts": int((nan_counts > 0).sum()),
        "first": _array_summary(contexts[0]),
    }


def _official_forecast(pairs, indices, horizon: int, max_context: Optional[int],
                       batch_size: int):
    np = _require_numpy()

    tfm = sanity._load_timesfm(TIMESFM_MODEL_ID)
    predictor = sanity.TimesFmPredictor(
        tfm=tfm,
        prediction_length=horizon,
        max_context_knob=max_context,
        default_batch_size=batch_size,
    )
    selected_inputs = [pairs[int(i)][0] for i in indices]
    forecasts = predictor.predict(selected_inputs, batch_size=batch_size)
    median = np.stack([np.asarray(f.quantile(0.5), dtype=np.float64) for f in forecasts])
    quantiles = np.stack([
        np.stack([np.asarray(f.quantile(q), dtype=np.float64)
                  for q in sanity.QUANTILE_LEVELS], axis=0)
        for f in forecasts
    ], axis=0)
    return median, quantiles


def _make_main_batches(cache, positions, original_indices, width: int,
                       batch_size: int, pin_memory: bool, device: str):
    np = _require_numpy()
    torch = _require_torch()

    x_np = np.empty((len(original_indices), width), dtype=np.float32)
    for j, i in enumerate(original_indices):
        ctx = np.asarray(cache.contexts_raw[int(i)], dtype=np.float32)
        if ctx.size >= width:
            x_np[j] = ctx[-width:]
        else:
            x_np[j] = np.concatenate([
                np.zeros(width - ctx.size, dtype=np.float32),
                ctx,
            ])
    y_np = cache.labels_np[np.asarray(original_indices, dtype=np.int64)]
    all_x = torch.from_numpy(x_np).unsqueeze(-1)
    all_y = torch.from_numpy(y_np).unsqueeze(-1)
    if pin_memory and device == "cuda":
        ax, ay = all_x.pin_memory(), all_y.pin_memory()
    else:
        ax, ay = all_x, all_y
    batches = []
    for start in range(0, len(original_indices), batch_size):
        stop = min(start + batch_size, len(original_indices))
        batches.append({"x": ax[start:stop], "y": ay[start:stop]})
    return batches


def _main_forecast(cache, indices, horizon: int, cap: int, batch_size: int,
                   device: str, compile_horizon: str, window_size: Optional[int]):
    np = _require_numpy()
    torch = _require_torch()
    wab = _wab()

    selected = np.asarray(indices, dtype=np.int64)

    if window_size is not None:
        served = selected[cache.context_lengths[selected] >= int(window_size)]
        if served.size == 0:
            raise RuntimeError(
                f"No selected instance has context >= main_window_size={window_size}")
        used_indices = served
        groups = [(np.arange(served.size, dtype=np.int64), int(window_size), served)]
        mode = f"window_{int(window_size)}_skip"
    else:
        used_indices = selected
        contexts = [
            np.asarray(cache.contexts_raw[int(i)], dtype=np.float32)[-int(cap):]
            for i in selected
        ]
        model = wab.load_timesfm(
            TIMESFM_MODEL_ID, int(cap), horizon, batch_size)
        fr, _tgts = wab.predict_timesfm_contexts(
            model, contexts, cache.labels_np[selected], horizon, device,
            batch_size=batch_size)
        return f"native_cap_{int(cap)}", fr, used_indices

    results = []
    compile_h = 1024 if compile_horizon == "1024" else horizon
    try:
        for pos, width, orig_idx in groups:
            model = wab.load_timesfm(TIMESFM_MODEL_ID, width, compile_h, batch_size)
            batches = _make_main_batches(
                cache, pos, orig_idx, width, batch_size,
                pin_memory=(device == "cuda"), device=device)
            fr, tgts = wab.predict_timesfm(model, batches, horizon, device)
            results.append((pos, fr, tgts))
            del model
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        fr, tgts = wab._merge_grouped(results, len(used_indices), horizon, device)
        return mode, fr, used_indices
    finally:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="GiftEval dataset name, e.g. m4_weekly or electricity/15T")
    p.add_argument("--term", default="short", choices=["short", "medium", "long"])
    p.add_argument("--n", type=int, default=16,
                   help="First N instances to compare. Use <=0 for all.")
    p.add_argument("--indices", default=None,
                   help="Comma-separated explicit instance indices, overrides --n.")
    p.add_argument("--max-context", type=int, default=None,
                   help="Context cap for official path and native main path. "
                        "Default: official full context, main cap 15360.")
    p.add_argument("--main-window-size", type=int, default=None,
                   help="Also compare current main-style fixed window in skip mode. "
                        "If omitted, compare native full-context grouping.")
    p.add_argument("--main-compile-horizon", choices=["horizon", "1024"],
                   default="horizon",
                   help="Deprecated compatibility option; both paths now use the "
                        "official max_horizon=1024 recipe.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--out", default=None,
                   help="Optional JSON output path.")
    return p.parse_args()


def main():
    args = parse_args()
    np = _require_numpy()
    torch = _require_torch()
    gluonts_leaderboard_mase = _gluonts_leaderboard_mase()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    dataset, cache, pairs = _load_dataset(args.dataset, args.term)
    indices = _select_indices(cache, args.n, args.indices)
    cap = int(args.max_context) if args.max_context is not None else min(15360, cache.max_context)

    starts = [cache.starts[int(i)] for i in indices]
    contexts_raw = [cache.contexts_raw[int(i)] for i in indices]
    labels = cache.labels_np[indices]

    official_contexts = _effective_contexts_official(pairs, indices, args.max_context)

    official_median, official_quantiles = _official_forecast(
        pairs, indices, cache.horizon, args.max_context, args.batch_size)
    main_mode, main_fr, main_indices = _main_forecast(
        cache, indices, cache.horizon, cap, args.batch_size, device,
        args.main_compile_horizon, args.main_window_size)
    main_context_cap = int(args.main_window_size) if args.main_window_size is not None else cap
    main_contexts = _effective_contexts_main(cache, main_indices, main_context_cap)
    main_median = main_fr.median.detach().cpu().float().numpy()
    main_quantiles = (main_fr.quantiles.detach().cpu().float().numpy()
                      if main_fr.quantiles is not None else None)

    official_mase = gluonts_leaderboard_mase(
        official_median, starts, contexts_raw, labels, cache.freq,
        quantiles=official_quantiles, quantile_levels=sanity.QUANTILE_LEVELS)
    main_starts = [cache.starts[int(i)] for i in main_indices]
    main_contexts_raw = [cache.contexts_raw[int(i)] for i in main_indices]
    main_labels = cache.labels_np[main_indices]
    main_mase = gluonts_leaderboard_mase(
        main_median, main_starts, main_contexts_raw, main_labels, cache.freq,
        quantiles=main_quantiles, quantile_levels=main_fr.quantile_levels)

    starts_match = all(
        str(cache.starts[int(i)]) == str(pairs[int(i)][1]["start"])
        for i in indices
    )
    report = {
        "config": {
            "dataset": args.dataset,
            "term": args.term,
            "freq": cache.freq,
            "horizon": cache.horizon,
            "n_selected": int(len(indices)),
            "n_main_served": int(len(main_indices)),
            "indices_head": indices[:20].tolist(),
            "main_indices_head": main_indices[:20].tolist(),
            "device": device,
            "batch_size": args.batch_size,
            "max_context": args.max_context,
            "main_cap": int(cap),
            "main_mode": main_mode,
            "main_compile_horizon": args.main_compile_horizon,
        },
        "data": {
            "cache_n_total": int(cache.n_total),
            "ctx_min": int(cache.min_context),
            "ctx_max": int(cache.max_context),
            "starts_match_cache_vs_dataset": bool(starts_match),
            "labels": _array_summary(labels),
        },
        "inputs": {
            "official": _context_report("official_sanity", official_contexts),
            "main": _context_report("main_wrapper", main_contexts),
            "lengths_match": bool(
                [len(c) for c in official_contexts] == [len(c) for c in main_contexts]
            ),
            "first_context_diff": _diff_summary(official_contexts[0], main_contexts[0]),
        },
        "forecasts": {
            "official_median": _array_summary(official_median),
            "main_median": _array_summary(main_median),
            "median_diff": _diff_summary(official_median, main_median),
            "quantile_diff": (_diff_summary(official_quantiles, main_quantiles)
                              if main_quantiles is not None else None),
        },
        "metrics": {
            "official_gluonts_mase": float(official_mase),
            "main_gluonts_mase": float(main_mase),
            "abs_diff": float(abs(official_mase - main_mase)),
            "rel_diff_pct": float(
                100.0 * (main_mase - official_mase) / official_mase)
                if official_mase == official_mase and official_mase != 0 else None,
        },
    }

    text = json.dumps(_jsonable(report), indent=2, sort_keys=True)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
