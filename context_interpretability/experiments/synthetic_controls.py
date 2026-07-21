"""
Experiment 4 — synthetic distant-dependency controls (spec §7). MANDATORY.

For each control configuration (family x lag x strength x noise):
  1. verify EMPIRICALLY that the distant lag is (or is not) genuinely
     predictive with the ridge oracles — a strength>0 config whose oracle gain
     is non-positive is marked broken and excluded from model conclusions;
  2. compute the model's context-length error curve + sufficient context;
  3. run the configured intervention methods (perturbation, attention masking,
     activation patching, integrated gradients) on the control data via the
     other experiment modules, capability-gated;
  4. test the §7.6 expectations: sensitivity concentrated near lag d,
     sufficient context growing with d, intervention effect growing with
     strength.

A model that fails to use lag d even at max strength — WITH the oracle
confirming predictive value — is reported as an architectural/training
limitation (``limitation_flag``), never as adaptive ignoring (spec §7.6).
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List

import numpy as np
from tqdm.auto import tqdm

from context_interpretability.adapters.base import (
    CapabilityError, InterpretabilityAdapter)
from context_interpretability.data.synthetic_generators import (
    ControlSpec, gate_detectable, generate_control, verify_distant_information)
from context_interpretability.experiments import (
    activation_patching, attention_masking, integrated_gradients, perturbation)
from context_interpretability.experiments.common import CleanCache, ExperimentData
from context_interpretability.schema import _json_default

METHOD_RUNNERS = {
    "perturbation": (perturbation, None),          # any forecaster
    "attention_masking": (attention_masking, "attention_masking"),
    "activation_patching": (activation_patching, "activation_patching"),
    "integrated_gradients": (integrated_gradients, "integrated_gradients"),
}


def sufficient_context(windows: List[int], mean_curve: np.ndarray,
                       tolerance: float) -> int:
    """Smallest window whose error is within (1+tol) of the curve minimum."""
    best = np.nanmin(mean_curve)
    for w, e in zip(windows, mean_curve):
        if e <= (1.0 + tolerance) * best:
            return int(w)
    return int(windows[-1])


def _control_specs(scfg: dict) -> List[ControlSpec]:
    specs: List[ControlSpec] = []
    r = int(scfg.get("local_order", 8))
    dk = scfg.get("distant_kind", "linear")
    noises = scfg.get("noise_levels", [0.1])
    for kind in scfg.get("local_kinds", ["linear"]):
        specs.append(ControlSpec("A", kind, r, noise=float(noises[0])))
    for d, s, nz in itertools.product(scfg.get("distant_lags", [64]),
                                      scfg.get("dependency_strengths", [0.5]),
                                      noises):
        specs.append(ControlSpec("B", "linear", r, int(d), float(s), dk,
                                 float(nz)))
    mid = scfg.get("distant_lags", [64])
    specs.append(ControlSpec("C", "linear", r, int(mid[len(mid) // 2]),
                             float(max(scfg.get("dependency_strengths",
                                                [0.5]))),
                             dk, float(noises[0])))
    return specs


def run(adapter: InterpretabilityAdapter, config: dict, out_dir: str,
        run_meta=None, seed: int = 0) -> str:
    scfg = config.get("synthetic_controls") or {}
    tol = float(scfg.get("sufficient_context_tolerance", 0.05))
    n_inst = int(scfg.get("n_instances", 128))
    length = min(int(scfg.get("series_length", 8192)), adapter.max_context())
    metric = config.get("primary_metric", "mae")
    run_methods = scfg.get("run_methods", ["perturbation"])
    summary: List[dict] = []

    specs = _control_specs(scfg)
    for spec in tqdm(specs, desc=f"exp4 {adapter.name} controls",
                     unit="dataset", dynamic_ncols=True):
        ds_dir = os.path.join(out_dir, spec.name)
        os.makedirs(ds_dir, exist_ok=True)
        contexts, targets, gates = generate_control(
            spec, n_inst, length, adapter.horizon, seed)
        data = ExperimentData(
            name=spec.name, contexts=contexts,
            targets=targets[:, :adapter.horizon],
            sample_ids=[f"{spec.name}_{i}" for i in range(n_inst)],
            metadata={"spec": dataclass_dict(spec),
                      "gates": None if gates is None else gates.tolist()})

        # -- 1. oracle verification (always, before any model conclusion) ------
        oracle = verify_distant_information(
            contexts, spec, alpha=float(scfg.get("oracle_ridge_alpha", 1.0)),
            seed=seed)
        broken = (spec.family in ("B", "C") and spec.strength > 0
                  and not oracle["distant_predictive"])
        if gates is not None:
            oracle["gate_detect_accuracy"] = gate_detectable(
                contexts, gates, spec, seed)

        # -- 2. context-length error curve + sufficient context ----------------
        cache = CleanCache(adapter, data, metric,
                           cache_dir=os.path.join(ds_dir, "clean_cache"))
        pairs = adapter.effective_context_lengths(config["context_lengths"])
        pairs = [(r_, e) for r_, e in pairs if e <= length]
        eff = [e for _r, e in pairs]
        loss_mat, mean_curve = cache.error_curve(eff)
        suff = sufficient_context(eff, mean_curve, tol)
        np.savez(os.path.join(ds_dir, "error_curve.npz"),
                 windows=eff, loss_matrix=loss_mat, mean_curve=mean_curve,
                 sufficient_context=suff)

        # -- 3. intervention methods on the control data ------------------------
        methods_run: List[str] = []
        for m in run_methods:
            mod, cap = METHOD_RUNNERS[m]
            if cap is not None and not adapter.capabilities.supported(cap):
                if run_meta:
                    run_meta.skip(f"exp4/{spec.name}/{m}",
                                  f"{adapter.name}: capability off")
                continue
            try:
                mod.run(adapter, data, config,
                        os.path.join(ds_dir, m), run_meta=run_meta, seed=seed)
                methods_run.append(m)
            except CapabilityError as exc:
                if run_meta:
                    run_meta.skip(f"exp4/{spec.name}/{m}", str(exc))

        summary.append({
            "spec": dataclass_dict(spec), "dataset": spec.name,
            "oracle": oracle, "config_broken": broken,
            "windows": eff, "mean_error_curve": mean_curve.tolist(),
            "sufficient_context": suff, "methods_run": methods_run,
            "limitation_flag": bool(
                spec.family == "B" and spec.strength >= 0.5 and not broken
                and suff < spec.distant_lag),
        })
        print(f"[exp4][{adapter.name}] {spec.name}: suff_ctx={suff} "
              f"oracle_gain={oracle['relative_gain']:.3f}"
              f"{'  BROKEN-CONFIG' if broken else ''}")

    path = os.path.join(out_dir, "controls_summary.json")
    with open(path, "w") as f:
        json.dump({"model": adapter.name, "tolerance": tol,
                   "controls": summary}, f, indent=2, default=_json_default)
    if run_meta:
        run_meta.note_samples("exp4_synthetic_controls",
                              n_inst * len(summary))
    return path


def dataclass_dict(spec: ControlSpec) -> dict:
    import dataclasses
    return dataclasses.asdict(spec)
