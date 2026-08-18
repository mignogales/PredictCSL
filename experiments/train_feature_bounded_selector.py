#!/usr/bin/env python3
"""Small-data tree baseline for the GiftEvalPretrain bounded selector."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


def _corr(x, lag):
    if len(x) <= lag + 4:
        return 0.0
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    return float(np.corrcoef(a, b)[0, 1]) if sa > 1e-6 and sb > 1e-6 else 0.0


def series_features(x):
    x = np.asarray(x, dtype=np.float64)
    out = []
    for width in (32, 96, 192, 512, 1024, 2048, 4096, 8192):
        y = x[-width:]
        d = np.diff(y)
        t = np.linspace(-1.0, 1.0, len(y))
        out.extend([
            y.mean(), y.std(), np.median(y), np.quantile(y, .25),
            np.quantile(y, .75), np.dot(t, y) / max(np.dot(t, t), 1e-8),
            np.mean(np.abs(d)), d.std(), y[-1],
        ])
    for lag in (1, 2, 6, 12, 24, 48, 96, 168, 336, 672):
        out.append(_corr(x, lag))
    recent = x[-2048:] - x[-2048:].mean()
    power = np.abs(np.fft.rfft(recent))[1:] ** 2
    power /= max(power.sum(), 1e-12)
    out.extend([
        -float(np.sum(power * np.log(power + 1e-12))) / np.log(len(power)),
        float(np.argmax(power) + 1) / len(power),
        float(np.sort(power)[-5:].sum()),
    ])
    return np.nan_to_num(np.asarray(out, dtype=np.float32))


def metrics(error, prediction, threshold):
    selected = prediction.argmin(1)
    gain = prediction[:, -1] - prediction[np.arange(len(prediction)), selected]
    selected = np.where(gain >= threshold, selected, error.shape[1] - 1)
    chosen = error[np.arange(len(error)), selected]
    native = error[:, -1]
    oracle = error.min(1)
    harm = (chosen - native) / np.maximum(native, 1e-8)
    return {
        "mean_error_vs_native": float(np.mean(chosen / np.maximum(native, 1e-8))),
        "mean_regret": float(np.mean((chosen - oracle) / np.maximum(oracle, 1e-8))),
        "harm_gt_5pct_rate": float(np.mean(harm > .05)),
        "harm_rate": float(np.mean(harm > 0)),
        "improvement_rate": float(np.mean(chosen < native)),
        "native_action_rate": float(np.mean(selected == error.shape[1] - 1)),
        "selection_counts": np.bincount(selected, minlength=error.shape[1]).tolist(),
        "threshold": float(threshold),
    }


def choose_policy(error, prediction, harm_limit):
    feasible = []
    raw_action = prediction.argmin(1)
    max_gain = float(np.max(
        prediction[:, -1] - prediction[np.arange(len(prediction)), raw_action]))
    for threshold in np.linspace(0, max(0.30, max_gain + 1e-4), 501):
        result = metrics(error, prediction, float(threshold))
        if result["harm_gt_5pct_rate"] <= harm_limit:
            feasible.append((result["mean_error_vs_native"], result["mean_regret"], threshold, result))
    return min(feasible, key=lambda row: row[:3])[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs/experiments/gifteval_pretrain_bounded_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "selector" / "feature_trees_v3"
    out.mkdir(parents=True, exist_ok=True)
    contexts = np.load(root / "prepared" / "contexts.npy")
    splits = np.load(root / "prepared" / "splits.npy")
    curves = np.load(root / "labels" / "chronos2_small" / "curves_mae.npy")
    base = np.stack([series_features(x) for x in contexts])
    horizons = np.asarray([24, 96, 192], dtype=np.float32)
    x = np.repeat(base, len(horizons), axis=0)
    h = np.tile(horizons, len(base))
    x = np.column_stack([x, np.log1p(h), h / 8192.0]).astype(np.float32)
    error = curves.transpose(0, 2, 1).reshape(-1, curves.shape[1])
    target = np.log(np.maximum(error, 1e-8) / np.maximum(error[:, -1:], 1e-8))
    task_splits = np.repeat(splits, len(horizons))
    masks = {name: task_splits == name for name in ("train", "val", "test")}
    candidates = []
    specs = []
    for leaf in (2, 5, 10, 20):
        specs.append((f"extra_leaf{leaf}", ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=leaf, max_features=.75,
            n_jobs=-1, random_state=args.seed)))
    specs.append(("rf_leaf5", RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, max_features=.75,
        n_jobs=-1, random_state=args.seed)))
    for name, model in specs:
        model.fit(x[masks["train"]], target[masks["train"]])
        pred = model.predict(x[masks["val"]])
        safe = choose_policy(error[masks["val"]], pred, .01)
        balanced = choose_policy(error[masks["val"]], pred, .03)
        candidates.append((safe["mean_error_vs_native"], safe["mean_regret"], name, model, safe, balanced))
        print(name, "safe", safe, "balanced", balanced, flush=True)
    _, _, name, model, safe_val, balanced_val = min(candidates, key=lambda row: row[:3])
    test_pred = model.predict(x[masks["test"]])
    report = {
        "version": 3,
        "method": "engineered time-series features plus tree ensemble",
        "selected_model": name,
        "n_features": int(x.shape[1]),
        "validation_safe_1pct": safe_val,
        "test_safe_1pct": metrics(error[masks["test"]], test_pred, safe_val["threshold"]),
        "validation_balanced_3pct": balanced_val,
        "test_balanced_3pct": metrics(error[masks["test"]], test_pred, balanced_val["threshold"]),
        "official_gifteval_test_used": False,
    }
    joblib.dump(model, out / "model.joblib")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
