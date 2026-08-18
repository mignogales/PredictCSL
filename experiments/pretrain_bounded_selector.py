#!/usr/bin/env python3
"""Train a conservative context selector from GiftEvalPretrain real series.

This is the first real-data counterpart to the synthetic context-curve pipeline.
It intentionally solves a smaller, safer decision problem for Chronos-2 Small:
choose one of the three longest supported windows (4096, 6144, or 8192).

The job is resumable and has three stages:

  prepare  sample rolling-origin examples from selected GiftEvalPretrain sources
  label    forecast every example/window with Chronos-2 Small and store MAE
  train    fit a small Mamba risk model using source-disjoint splits

The official GIFT-Eval benchmark data is never opened by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ID = "Salesforce/GiftEvalPretrain"
DEFAULT_MODEL_ID = "autogluon/chronos-2-small"
DEFAULT_WINDOWS = (4096, 6144, 8192)
DEFAULT_HORIZONS = (24, 96, 192)

# Whole sources, rather than individual examples, are held out.  This avoids
# evaluating on another cutoff/channel from a source observed during training.
DEFAULT_SOURCE_SPLITS: Dict[str, Tuple[str, ...]] = {
    "train": (
        "australian_electricity_demand",
        "oikolab_weather",
        "bdg-2_panther",
        "gfc12_load",
        "pedestrian_counts",
        "PEMS03",
        "wind_power",
        "solar_power",
    ),
    "val": (
        "bdg-2_bear",
        "beijing_air_quality",
    ),
    "test": (
        "bdg-2_rat",
        "traffic_hourly",
    ),
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def source_to_split(source: str, splits: Dict[str, Sequence[str]]) -> str:
    hits = [name for name, members in splits.items() if source in members]
    if len(hits) != 1:
        raise ValueError(f"Source {source!r} belongs to {len(hits)} splits: {hits}")
    return hits[0]


def iter_univariate_targets(target: object) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(channel, values)`` from a GiftEval univariate/multivariate target."""
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim == 1:
        yield 0, arr
    elif arr.ndim == 2:
        for channel in range(arr.shape[0]):
            yield channel, arr[channel]
    else:
        raise ValueError(f"Expected target rank 1 or 2, got shape {arr.shape}")


def rolling_cutoffs(
    length: int,
    max_window: int,
    max_horizon: int,
    count: int,
) -> List[int]:
    """Evenly spread valid historical origins, including the latest origin."""
    first = int(max_window)
    last = int(length - max_horizon)
    if last < first or count <= 0:
        return []
    if count == 1 or first == last:
        return [last]
    return sorted(set(np.linspace(first, last, count, dtype=np.int64).tolist()))


def standardize_example(
    values: np.ndarray,
    cutoff: int,
    max_window: int,
    max_horizon: int,
    min_context_observed: float = 0.80,
    min_future_observed: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Return standardized context/future or ``None`` for an unusable origin.

    Missing context values become zero *after* finite-value standardization,
    i.e. the observed-context mean. Missing future values remain NaN so metrics
    can ignore them rather than silently treating them as correct forecasts.
    """
    context = np.asarray(values[cutoff - max_window:cutoff], dtype=np.float32)
    future = np.asarray(values[cutoff:cutoff + max_horizon], dtype=np.float32)
    if context.size != max_window or future.size != max_horizon:
        return None
    context_ok = np.isfinite(context)
    future_ok = np.isfinite(future)
    if context_ok.mean() < min_context_observed:
        return None
    if future_ok.mean() < min_future_observed:
        return None
    mu = float(np.mean(context[context_ok]))
    sd = float(np.std(context[context_ok]))
    if not math.isfinite(sd) or sd < 1e-6:
        return None
    context_z = (context - mu) / sd
    future_z = (future - mu) / sd
    context_z = np.nan_to_num(context_z, nan=0.0, posinf=0.0, neginf=0.0)
    return context_z.astype(np.float32), future_z.astype(np.float32)


def _all_sources(splits: Dict[str, Sequence[str]]) -> List[str]:
    return [source for split in ("train", "val", "test") for source in splits[split]]


def _download_source(source: str, cache_dir: Path, repo_files: Sequence[str]) -> List[Path]:
    from huggingface_hub import hf_hub_download

    filenames = sorted(
        name for name in repo_files
        if name.startswith(f"{source}/data-") and name.endswith(".arrow")
    )
    if not filenames:
        raise FileNotFoundError(f"No Arrow shards found for GiftEvalPretrain/{source}")
    paths = []
    for filename in filenames:
        paths.append(Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(cache_dir),
        )))
    return paths


def _load_arrow_shards(paths: Sequence[Path]):
    from datasets import Dataset as HFDataset, concatenate_datasets

    shards = [HFDataset.from_file(str(path)) for path in paths]
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


def sample_source(
    source: str,
    rows,
    max_rows: int,
    origins_per_series: int,
    max_window: int,
    max_horizon: int,
    seed: int,
) -> List[dict]:
    """Reservoir-sample rolling origins so large sources cannot dominate."""
    rng = random.Random(f"{seed}:{source}")
    reservoir: List[dict] = []
    seen = 0
    for row_index, row in enumerate(rows):
        freq = str(row.get("freq", "unknown"))
        item_id = str(row.get("item_id", row_index))
        try:
            channels = iter_univariate_targets(row["target"])
            for channel, values in channels:
                for cutoff in rolling_cutoffs(
                    len(values), max_window, max_horizon, origins_per_series
                ):
                    standardized = standardize_example(
                        values, cutoff, max_window, max_horizon)
                    if standardized is None:
                        continue
                    context, future = standardized
                    candidate = {
                        "context": context,
                        "future": future,
                        "source": source,
                        "item_id": item_id,
                        "channel": int(channel),
                        "cutoff": int(cutoff),
                        "freq": freq,
                    }
                    seen += 1
                    if len(reservoir) < max_rows:
                        reservoir.append(candidate)
                    else:
                        replacement = rng.randrange(seen)
                        if replacement < max_rows:
                            reservoir[replacement] = candidate
        except (TypeError, ValueError):
            # A malformed record should not invalidate an otherwise useful source.
            continue
    rng.shuffle(reservoir)
    return reservoir


def prepare(args: argparse.Namespace, splits: Dict[str, Sequence[str]]) -> None:
    from huggingface_hub import HfApi

    root = Path(args.output_dir)
    prepared = root / "prepared"
    if (prepared / "contexts.npy").is_file() and not args.force_prepare:
        print(f"prepare: reusing {prepared}", flush=True)
        return

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    examples: List[dict] = []
    source_counts: Dict[str, int] = {}
    for source in _all_sources(splits):
        print(f"prepare: downloading/loading {source}", flush=True)
        rows = _load_arrow_shards(_download_source(source, cache_dir, repo_files))
        sampled = sample_source(
            source=source,
            rows=rows,
            max_rows=args.rows_per_source,
            origins_per_series=args.origins_per_series,
            max_window=max(args.windows),
            max_horizon=max(args.horizons),
            seed=args.seed,
        )
        source_counts[source] = len(sampled)
        examples.extend(sampled)
        print(f"prepare: {source}: {len(sampled)} usable origins", flush=True)

    empty_splits = [
        split for split, members in splits.items()
        if sum(source_counts[source] for source in members) == 0
    ]
    if empty_splits:
        raise RuntimeError("No >=8192-step examples in splits: " + ", ".join(empty_splits))
    prepared.mkdir(parents=True, exist_ok=True)
    np.save(prepared / "contexts.npy", np.stack([x["context"] for x in examples]))
    np.save(prepared / "targets.npy", np.stack([x["future"] for x in examples]))
    np.save(prepared / "sources.npy", np.asarray([x["source"] for x in examples]))
    np.save(prepared / "splits.npy", np.asarray([
        source_to_split(x["source"], splits) for x in examples]))
    metadata = [{k: v for k, v in x.items() if k not in ("context", "future")}
                for x in examples]
    (prepared / "examples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata))
    atomic_json(prepared / "manifest.json", {
        "repo_id": REPO_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "windows": list(args.windows),
        "horizons": list(args.horizons),
        "source_splits": {k: list(v) for k, v in splits.items()},
        "source_counts": source_counts,
        "n_examples": len(examples),
        "rows_per_source": args.rows_per_source,
        "origins_per_series": args.origins_per_series,
        "standardization": "finite context z-score; missing context -> context mean",
        "benchmark_data_used": False,
    })
    print(f"prepare: wrote {len(examples)} examples to {prepared}", flush=True)


def label(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    prepared = root / "prepared"
    label_dir = root / "labels" / "chronos2_small"
    label_dir.mkdir(parents=True, exist_ok=True)
    contexts = np.load(prepared / "contexts.npy", mmap_mode="r")
    targets = np.load(prepared / "targets.npy", mmap_mode="r")
    n = int(contexts.shape[0])
    curves_path = label_dir / "curves_mae.npy"
    state_path = label_dir / "label_state.json"
    if curves_path.is_file():
        curves = np.load(curves_path)
        expected = (n, len(args.windows), len(args.horizons))
        if curves.shape != expected:
            raise ValueError(f"Existing curves have shape {curves.shape}, expected {expected}")
    else:
        curves = np.full(
            (n, len(args.windows), len(args.horizons)), np.nan, dtype=np.float32)
    completed: List[int] = []
    if state_path.is_file():
        completed = [int(x) for x in json.loads(state_path.read_text()).get(
            "completed_windows", [])]

    # Import the project wrapper only on a labeling worker; preparing and unit
    # testing this module do not require Chronos or a GPU.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments.build_context_length_dataset import _forecast_uniform, setup_model

    print(f"label: loading {args.model_id} on {args.device}", flush=True)
    model = setup_model("chronos2", args.model_id, args.device)
    max_horizon = max(args.horizons)
    for window_index, window in enumerate(args.windows):
        if window_index in completed and not args.force_label:
            print(f"label: W={window} already complete", flush=True)
            continue
        print(f"label: forecasting {n} examples at W={window}", flush=True)
        x = torch.from_numpy(np.asarray(contexts[:, -window:])).unsqueeze(-1)
        forecast = _forecast_uniform(
            "chronos2", model, args.model_id, x,
            width=window,
            horizon=max_horizon,
            batch_size=args.label_batch_size,
            device=args.device,
            dynamic_batching=True,
            max_batch_size=args.label_batch_size,
        ).detach().cpu().numpy()
        truth = np.asarray(targets)
        for horizon_index, horizon in enumerate(args.horizons):
            curves[:, window_index, horizon_index] = np.nanmean(
                np.abs(forecast[:, :horizon] - truth[:, :horizon]), axis=1)
        np.save(curves_path, curves)
        if window_index not in completed:
            completed.append(window_index)
        atomic_json(state_path, {
            "completed_windows": sorted(completed),
            "model_id": args.model_id,
            "windows": list(args.windows),
            "horizons": list(args.horizons),
        })
    atomic_json(label_dir / "meta.json", {
        "model_id": args.model_id,
        "metric": "MAE on context-standardized real continuations",
        "windows": list(args.windows),
        "horizons": list(args.horizons),
        "n_examples": n,
    })
    print(f"label: complete -> {curves_path}", flush=True)


class CurveDataset(Dataset):
    def __init__(self, contexts: np.ndarray, curves: np.ndarray, indices: np.ndarray):
        self.contexts = torch.from_numpy(np.ascontiguousarray(contexts[indices])).float()
        self.curves = torch.from_numpy(np.ascontiguousarray(curves[indices])).float()
        self.n_horizons = curves.shape[2]

    def __len__(self) -> int:
        return len(self.contexts) * self.n_horizons

    def __getitem__(self, index: int):
        row, horizon = divmod(index, self.n_horizons)
        return self.contexts[row].unsqueeze(-1), horizon, self.curves[row, :, horizon]


@dataclass
class ModelConfig:
    context_length: int = 8192
    patch_length: int = 128
    d_model: int = 128
    num_hidden_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    mask_ratio: float = 0.15


def bounded_risk_loss(
    prediction: torch.Tensor,
    error: torch.Tensor,
    policy_weight: float = 1.0,
    native_harm_weight: float = 2.0,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Calibrated curve loss plus differentiable error-aware policy loss."""
    valid = torch.isfinite(error) & (error > 0)
    valid_rows = valid.all(dim=1)
    if not valid_rows.any():
        return prediction.sum() * 0.0
    prediction = prediction[valid_rows]
    error = error[valid_rows]
    native = error[:, -1].clamp_min(1e-8)
    log_relative = torch.log(error.clamp_min(1e-8) / native[:, None]).clamp(-4, 4)
    calibration = F.smooth_l1_loss(prediction, log_relative)
    oracle = error.min(dim=1).values.clamp_min(1e-8)
    regret = (error - oracle[:, None]) / oracle[:, None]
    harm = F.relu((error - native[:, None]) / native[:, None])
    probabilities = torch.softmax(-prediction / max(temperature, 1e-6), dim=1)
    policy = (probabilities * (regret + native_harm_weight * harm)).sum(dim=1).mean()
    return calibration + policy_weight * policy


def balanced_oracle_loss(
    prediction: torch.Tensor,
    error: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Class-balanced auxiliary loss preventing collapse onto one action."""
    return F.cross_entropy(
        -prediction, error.argmin(dim=1), weight=class_weights)


def hierarchical_shortening_loss(
    prediction: torch.Tensor,
    error: torch.Tensor,
    min_gain: float = 0.01,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Two-stage objective: shorten-or-native, then choose a shorter action."""
    native_error = error[:, -1].clamp_min(1e-8)
    best_short_error, best_short_action = error[:, :-1].min(dim=1)
    true_gain = (native_error - best_short_error) / native_error
    should_shorten = (true_gain >= min_gain).float()
    soft_short_score = -temperature * torch.logsumexp(
        -prediction[:, :-1] / temperature, dim=1)
    predicted_gain = prediction[:, -1] - soft_short_score
    decision = F.binary_cross_entropy_with_logits(
        predicted_gain / temperature, should_shorten)
    positive = should_shorten.bool()
    conditional = (
        F.cross_entropy(-prediction[positive, :-1], best_short_action[positive])
        if positive.any() else prediction.sum() * 0.0)
    return decision + 0.5 * conditional


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: str,
    confidence_threshold: float = 0.0,
) -> dict:
    model.eval()
    selected_all, curves_all = [], []
    loss_sum, count = 0.0, 0
    for context, horizon, curve in loader:
        context, horizon, curve = context.to(device), horizon.to(device), curve.to(device)
        prediction, _, _, _ = model(context, horizon)
        loss = bounded_risk_loss(prediction, curve)
        selected = prediction.argmin(dim=1)
        predicted_gain = prediction[:, -1] - prediction.gather(
            1, selected[:, None]).squeeze(1)
        selected = torch.where(
            predicted_gain >= confidence_threshold,
            selected,
            torch.full_like(selected, prediction.shape[1] - 1),
        )
        selected_all.append(selected.cpu())
        curves_all.append(curve.cpu())
        loss_sum += float(loss) * len(context)
        count += len(context)
    selected = torch.cat(selected_all).numpy()
    curves = torch.cat(curves_all).numpy()
    row = np.arange(len(curves))
    chosen = curves[row, selected]
    native = curves[:, -1]
    oracle = curves.min(axis=1)
    regret = (chosen - oracle) / np.maximum(oracle, 1e-8)
    harm = (chosen - native) / np.maximum(native, 1e-8)
    return {
        "loss": loss_sum / max(count, 1),
        "mean_regret": float(np.mean(regret)),
        "p90_regret": float(np.quantile(regret, 0.90)),
        "mean_error_vs_native": float(np.mean(chosen / np.maximum(native, 1e-8))),
        "improvement_rate_vs_native": float(np.mean(chosen < native)),
        "harm_rate_vs_native": float(np.mean(chosen > native)),
        "harm_gt_5pct_rate": float(np.mean(harm > 0.05)),
        "native_action_rate": float(np.mean(selected == curves.shape[1] - 1)),
        "oracle_improvement_vs_native": float(
            1.0 - np.mean(oracle) / max(float(np.mean(native)), 1e-8)),
        "selection_counts": np.bincount(selected, minlength=curves.shape[1]).tolist(),
        "n_tasks": int(len(curves)),
        "confidence_threshold": float(confidence_threshold),
    }


def calibrate_confidence_gate(model, loader: DataLoader, device: str) -> Tuple[float, dict]:
    """Select the native-fallback margin using validation data only."""
    ranked = []
    for threshold in np.linspace(0.0, 0.30, 61):
        metrics = evaluate(model, loader, device, float(threshold))
        score = (
            metrics["mean_regret"]
            + 0.5 * metrics["p90_regret"]
            + metrics["harm_gt_5pct_rate"]
            + 5.0 * max(0.0, metrics["mean_error_vs_native"] - 1.0)
        )
        ranked.append((score, metrics["mean_error_vs_native"], threshold, metrics))
    _, _, threshold, metrics = min(ranked, key=lambda row: row[:3])
    return float(threshold), metrics


def train(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    prepared = root / "prepared"
    train_dir = root / "selector" / args.selector_name
    train_dir.mkdir(parents=True, exist_ok=True)
    contexts = np.load(prepared / "contexts.npy")
    curves = np.load(root / "labels" / "chronos2_small" / "curves_mae.npy")
    splits = np.load(prepared / "splits.npy")
    if not np.isfinite(curves).all():
        raise RuntimeError("Label surface is incomplete; finish the label stage first")
    indices = {name: np.flatnonzero(splits == name) for name in ("train", "val", "test")}
    if any(len(value) == 0 for value in indices.values()):
        raise RuntimeError(f"Empty source split: { {k: len(v) for k, v in indices.items()} }")

    train_contexts = contexts[indices["train"]]
    train_curves = curves[indices["train"]]
    n_synthetic = 0
    if args.synthetic_dir and args.synthetic_rows > 0:
        synthetic_dir = Path(args.synthetic_dir)
        synthetic_contexts = np.load(synthetic_dir.parent / "contexts.npy", mmap_mode="r")
        synthetic_full = np.load(synthetic_dir / "curves_mae.npy", mmap_mode="r")
        synthetic_meta = json.loads((synthetic_dir / "meta.json").read_text())
        source_windows = list(synthetic_meta["window_grid"])
        source_horizons = np.asarray(synthetic_meta["horizon_grid"], dtype=np.float64)
        window_indices = [source_windows.index(window) for window in args.windows]
        selected = np.asarray(synthetic_full[:, window_indices, :])
        interpolated = []
        log_source_horizons = np.log(source_horizons)
        for horizon in args.horizons:
            hi = int(np.searchsorted(source_horizons, horizon))
            hi = min(max(hi, 1), len(source_horizons) - 1)
            lo = hi - 1
            alpha = ((np.log(horizon) - log_source_horizons[lo])
                     / (log_source_horizons[hi] - log_source_horizons[lo]))
            interpolated.append(
                (1.0 - alpha) * selected[:, :, lo] + alpha * selected[:, :, hi])
        synthetic_curves = np.stack(interpolated, axis=2).astype(np.float32)
        valid = np.isfinite(synthetic_curves).all(axis=(1, 2))
        valid &= np.isfinite(synthetic_contexts).all(axis=1)
        available = np.flatnonzero(valid)
        rng = np.random.RandomState(args.seed)
        chosen = rng.choice(
            available, size=min(args.synthetic_rows, len(available)), replace=False)
        train_contexts = np.concatenate([
            train_contexts, np.asarray(synthetic_contexts[chosen])], axis=0)
        train_curves = np.concatenate([train_curves, synthetic_curves[chosen]], axis=0)
        n_synthetic = len(chosen)
        print(f"train: mixed in {n_synthetic} synthetic examples", flush=True)

    datasets = {
        "train": CurveDataset(
            train_contexts, train_curves, np.arange(len(train_contexts))),
        "val": CurveDataset(contexts, curves, indices["val"]),
        "test": CurveDataset(contexts, curves, indices["test"]),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=(name == "train"),
            num_workers=0,
            pin_memory=args.device.startswith("cuda"),
        )
        for name, dataset in datasets.items()
    }
    train_tasks = train_curves.transpose(0, 2, 1).reshape(
        -1, curves.shape[1])
    oracle_counts = np.bincount(
        train_tasks.argmin(axis=1), minlength=curves.shape[1])
    class_weights_np = np.sqrt(
        oracle_counts.sum() / np.maximum(oracle_counts, 1))
    class_weights_np /= class_weights_np.mean()
    class_weights = torch.as_tensor(
        class_weights_np, dtype=torch.float32, device=args.device)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments.predict_context_length import MambaContextLength

    cfg = ModelConfig(context_length=contexts.shape[1])
    model = MambaContextLength(
        context_length=cfg.context_length,
        patch_length=cfg.patch_length,
        d_model=cfg.d_model,
        num_hidden_layers=cfg.num_hidden_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        dropout=cfg.dropout,
        mask_ratio=cfg.mask_ratio,
        n_windows=len(args.windows),
        n_horizons=len(args.horizons),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_score, stale = float("inf"), 0
    history: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, count = 0.0, 0
        for context, horizon, curve in loaders["train"]:
            context, horizon, curve = context.to(args.device), horizon.to(args.device), curve.to(args.device)
            prediction, reconstruction, original, mask = model(context, horizon)
            task_loss = bounded_risk_loss(
                prediction,
                curve,
                policy_weight=args.policy_weight,
                native_harm_weight=args.native_harm_weight,
            )
            balance_loss = balanced_oracle_loss(prediction, curve, class_weights)
            hierarchy_loss = hierarchical_shortening_loss(
                prediction, curve, min_gain=args.min_shortening_gain)
            if mask.any():
                reconstruction_loss = (
                    (reconstruction - original).pow(2).mean(dim=-1) * mask.float()
                ).sum() / mask.float().sum().clamp_min(1.0)
            else:
                reconstruction_loss = prediction.sum() * 0.0
            loss = (
                task_loss
                + args.balance_weight * balance_loss
                + args.hierarchy_weight * hierarchy_loss
                + args.reconstruction_weight * reconstruction_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(context)
            count += len(context)
        validation = evaluate(model, loaders["val"], args.device)
        score = (validation["mean_regret"] + 0.5 * validation["p90_regret"]
                 + args.selection_harm_weight * validation["harm_gt_5pct_rate"])
        record = {
            "epoch": epoch,
            "train_loss": running / max(count, 1),
            "selection_score": score,
            "validation": validation,
        }
        history.append(record)
        atomic_json(train_dir / "history.json", history)
        print(
            f"train: epoch={epoch:03d} loss={record['train_loss']:.5f} "
            f"val_regret={validation['mean_regret']:.4f} "
            f"val_harm5={validation['harm_gt_5pct_rate']:.3f}", flush=True)
        if score < best_score - 1e-6:
            best_score, stale = score, 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_config": asdict(cfg),
                "windows": list(args.windows),
                "horizons": list(args.horizons),
                "model_id": args.model_id,
                "epoch": epoch,
                "validation": validation,
                "source_disjoint": True,
            }, train_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"train: early stop after {epoch} epochs", flush=True)
                break

    checkpoint = torch.load(train_dir / "best_model.pt", map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    raw_validation = evaluate(model, loaders["val"], args.device)
    confidence_threshold, gated_validation = calibrate_confidence_gate(
        model, loaders["val"], args.device)
    report = {
        "version": 2,
        "method": "balanced bounded Mamba risk selector with native confidence gate",
        "model_id": args.model_id,
        "windows": list(args.windows),
        "horizons": list(args.horizons),
        "model_config": asdict(cfg),
        "best_epoch": checkpoint["epoch"],
        "oracle_action_counts_train": oracle_counts.tolist(),
        "class_weights": class_weights_np.tolist(),
        "native_harm_weight": args.native_harm_weight,
        "balance_weight": args.balance_weight,
        "hierarchy_weight": args.hierarchy_weight,
        "min_shortening_gain": args.min_shortening_gain,
        "confidence_threshold": confidence_threshold,
        "source_disjoint": True,
        "split_sizes": {k: int(len(v)) for k, v in indices.items()},
        "synthetic_train_examples": n_synthetic,
        "validation_raw": raw_validation,
        "validation_gated": gated_validation,
        "internal_test_raw": evaluate(model, loaders["test"], args.device),
        "internal_test_gated": evaluate(
            model, loaders["test"], args.device, confidence_threshold),
        "official_gifteval_test_used": False,
    }
    atomic_json(train_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "label", "train", "all"), nargs="?", default="all")
    parser.add_argument("--output-dir", default="logs/experiments/gifteval_pretrain_bounded_v1")
    parser.add_argument("--cache-dir", default=os.environ.get(
        "GIFT_PRETRAIN_SAMPLE", "/home/mnogales/data/GiftEvalPretrainSample"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--selector-name", default="mamba_bounded_v1")
    parser.add_argument("--source-splits-json", default="")
    parser.add_argument("--synthetic-dir", default="")
    parser.add_argument("--synthetic-rows", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--rows-per-source", type=int, default=64)
    parser.add_argument("--origins-per-series", type=int, default=2)
    parser.add_argument("--label-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--native-harm-weight", type=float, default=2.0)
    parser.add_argument("--balance-weight", type=float, default=0.0)
    parser.add_argument("--hierarchy-weight", type=float, default=0.0)
    parser.add_argument("--min-shortening-gain", type=float, default=0.01)
    parser.add_argument("--selection-harm-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-label", action="store_true")
    args = parser.parse_args()
    args.windows = tuple(sorted(set(args.windows)))
    args.horizons = tuple(sorted(set(args.horizons)))
    if tuple(args.windows) != DEFAULT_WINDOWS:
        print(f"warning: v1 was designed for windows={DEFAULT_WINDOWS}; got {args.windows}")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    splits = DEFAULT_SOURCE_SPLITS
    if args.source_splits_json:
        splits = json.loads(Path(args.source_splits_json).read_text())
    if args.stage in ("prepare", "all"):
        prepare(args, splits)
    if args.stage in ("label", "all"):
        label(args)
    if args.stage in ("train", "all"):
        train(args)


if __name__ == "__main__":
    main()
