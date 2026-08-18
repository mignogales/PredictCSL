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

import copy
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


def _load_or_generate_control(spec: ControlSpec, n_inst: int,
                              canonical_length: int, horizon: int, seed: int,
                              cache_root: str):
    """Generate each control pool once and share it across model runs."""
    os.makedirs(cache_root, exist_ok=True)
    path = os.path.join(
        cache_root,
        f"{spec.name}_n{n_inst}_t{canonical_length}_h{horizon}_s{seed}.npz")
    if os.path.exists(path):
        z = np.load(path)
        gates = z["gates"] if bool(z["has_gates"]) else None
        return z["contexts"], z["targets"], gates
    contexts, targets, gates = generate_control(
        spec, n_inst, canonical_length, horizon, seed)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        tmp, contexts=contexts, targets=targets,
        gates=np.empty((0,), dtype=np.int8) if gates is None else gates,
        has_gates=np.asarray(gates is not None))
    os.replace(tmp, path)
    return contexts, targets, gates


def _load_or_compute_oracle(contexts: np.ndarray, gates, spec: ControlSpec,
                            alpha: float, seed: int, cache_root: str) -> dict:
    path = os.path.join(
        cache_root,
        f"{spec.name}_oracle_t{contexts.shape[1]}_a{alpha:g}_s{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    oracle = verify_distant_information(contexts, spec, alpha=alpha,
                                        seed=seed)
    if gates is not None:
        oracle["gate_detect_accuracy"] = gate_detectable(
            contexts, gates, spec, seed)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(oracle, f, default=_json_default)
    os.replace(tmp, path)
    return oracle


def sufficient_context(windows: List[int], mean_curve: np.ndarray,
                       tolerance: float) -> int:
    """Smallest window whose error is within (1+tol) of the curve minimum."""
    best = np.nanmin(mean_curve)
    for w, e in zip(windows, mean_curve):
        if e <= (1.0 + tolerance) * best:
            return int(w)
    return int(windows[-1])


def _control_specs(scfg: dict) -> List[tuple[ControlSpec, str]]:
    """Return the compact core design plus explicitly labelled anchor cells."""
    specs: List[tuple[ControlSpec, str]] = []
    r = int(scfg.get("local_order", 8))
    dk = scfg.get("distant_kind", "linear")
    noises = scfg.get("noise_levels", [0.1])
    for kind in scfg.get("local_kinds", ["linear"]):
        specs.append((ControlSpec("A", kind, r, noise=float(noises[0])),
                      "local_control"))
    for d, s, nz in itertools.product(scfg.get("distant_lags", [64]),
                                      scfg.get("dependency_strengths", [0.5]),
                                      noises):
        specs.append((ControlSpec("B", "linear", r, int(d), float(s), dk,
                                  float(nz)), "core"))

    # Noise is a robustness check at one predeclared anchor, not a fully
    # crossed factor. Labelling these cells prevents pseudo-replication in H4.
    robust = scfg.get("noise_robustness") or {}
    if robust:
        anchor_d = int(robust.get("lag", scfg.get("distant_lags", [64])[0]))
        anchor_s = float(robust.get(
            "strength", max(scfg.get("dependency_strengths", [0.5]))))
        existing = {(spec.distant_lag, spec.strength, spec.noise)
                    for spec, _role in specs if spec.family == "B"}
        for nz in robust.get("extra_noise_levels", []):
            key = (anchor_d, anchor_s, float(nz))
            if key not in existing:
                specs.append((ControlSpec("B", "linear", r, anchor_d,
                                          anchor_s, dk, float(nz)),
                              "noise_robustness"))
                existing.add(key)
    mid = scfg.get("distant_lags", [64])
    specs.append((ControlSpec("C", "linear", r, int(mid[len(mid) // 2]),
                              float(max(scfg.get("dependency_strengths",
                                                 [0.5]))),
                              dk, float(noises[0])), "conditional_control"))
    return specs


def _methods_for_spec(spec: ControlSpec, role: str, scfg: dict) -> List[str]:
    methods = list(scfg.get("run_methods", ["perturbation"]))
    sentinel_lags = {int(v) for v in scfg.get("sentinel_lags", [])}
    sentinel_strengths = {
        float(v) for v in scfg.get("sentinel_strengths", [])}
    is_sentinel = (role == "core" and spec.family == "B"
                   and spec.distant_lag in sentinel_lags
                   and spec.strength in sentinel_strengths)
    if is_sentinel:
        methods.extend(scfg.get("sentinel_methods", []))
    # Preserve order while tolerating a method listed in both sets.
    return list(dict.fromkeys(methods))


def run(adapter: InterpretabilityAdapter, config: dict, out_dir: str,
        run_meta=None, seed: int = 0) -> str:
    scfg = config.get("synthetic_controls") or {}
    tol = float(scfg.get("sufficient_context_tolerance", 0.05))
    n_inst = int(scfg.get("n_instances", 128))
    canonical_length = int(scfg.get("series_length", 8192))
    length = min(canonical_length, adapter.max_context())
    metric = config.get("primary_metric", "mae")
    summary: List[dict] = []
    cache_root = str(scfg.get("control_cache_root") or os.path.join(
        config.get("output_root", os.path.dirname(out_dir)),
        "_synthetic_control_cache"))
    reuse_clean_root = scfg.get("reuse_clean_cache_root")

    specs = _control_specs(scfg)
    only_datasets = set(scfg.get("only_datasets") or [])
    if only_datasets:
        specs = [(spec, role) for spec, role in specs
                 if spec.name in only_datasets]
        found = {spec.name for spec, _role in specs}
        missing = sorted(only_datasets - found)
        if missing:
            raise ValueError(
                f"synthetic_controls.only_datasets not in design: {missing}")
    exp_config = copy.deepcopy(config)
    exp_config["context_lengths"] = list(
        scfg.get("context_lengths", config["context_lengths"]))
    for spec, design_role in tqdm(specs,
                     desc=f"exp4 {adapter.name} controls",
                     unit="dataset", dynamic_ncols=True):
        ds_dir = os.path.join(out_dir, spec.name)
        os.makedirs(ds_dir, exist_ok=True)
        contexts, targets, gates = _load_or_generate_control(
            spec, n_inst, canonical_length, adapter.horizon, seed, cache_root)
        # All models see the same forecast origin and target. Models with a
        # smaller context budget consume the aligned suffix of the shared pool.
        contexts = np.ascontiguousarray(contexts[:, -length:])
        data = ExperimentData(
            name=spec.name, contexts=contexts,
            targets=targets[:, :adapter.horizon],
            sample_ids=[f"{spec.name}_{i}" for i in range(n_inst)],
            metadata={"spec": dataclass_dict(spec),
                      "gates": None if gates is None else gates.tolist()})

        # -- 1. oracle verification (always, before any model conclusion) ------
        oracle = _load_or_compute_oracle(
            contexts, gates, spec,
            alpha=float(scfg.get("oracle_ridge_alpha", 1.0)), seed=seed,
            cache_root=cache_root)
        broken = (spec.family in ("B", "C") and spec.strength > 0
                  and not oracle["distant_predictive"])

        # Preserve the full causal source band for all forecast steps even
        # when the ordinary geometric block cap thins distant history.
        spec_exp_config = copy.deepcopy(exp_config)
        if spec.family in ("B", "C") and spec.distant_lag > 0:
            pcfg = spec_exp_config.setdefault("perturbation", {})
            pcfg["force_lookback_ranges"] = [[
                max(0, spec.distant_lag - adapter.horizon + 1),
                spec.distant_lag + 1,
            ]]

        # -- 2. context-length error curve + sufficient context ----------------
        curve_cache_dir = (
            os.path.join(str(reuse_clean_root), adapter.name,
                         "exp4_synthetic_controls", spec.name, "clean_cache")
            if reuse_clean_root else os.path.join(ds_dir, "clean_cache")
        )
        cache = CleanCache(adapter, data, metric,
                           cache_dir=curve_cache_dir)
        pairs = adapter.effective_context_lengths(exp_config["context_lengths"])
        pairs = [(r_, e) for r_, e in pairs if e <= length]
        eff = [e for _r, e in pairs]
        loss_mat, mean_curve = cache.error_curve(eff)
        suff = sufficient_context(eff, mean_curve, tol)
        np.savez(os.path.join(ds_dir, "error_curve.npz"),
                 windows=eff, loss_matrix=loss_mat, mean_curve=mean_curve,
                 sufficient_context=suff)

        # -- 3. intervention methods on the control data ------------------------
        methods_run: List[str] = []
        for m in _methods_for_spec(spec, design_role, scfg):
            if m not in METHOD_RUNNERS:
                raise ValueError(f"Unknown Exp4 method: {m}")
            mod, cap = METHOD_RUNNERS[m]
            if cap is not None and not adapter.capabilities.supported(cap):
                if run_meta:
                    run_meta.skip(f"exp4/{spec.name}/{m}",
                                  f"{adapter.name}: capability off")
                continue
            try:
                mod.run(adapter, data, spec_exp_config,
                        os.path.join(ds_dir, m), run_meta=run_meta, seed=seed)
                methods_run.append(m)
            except CapabilityError as exc:
                if run_meta:
                    run_meta.skip(f"exp4/{spec.name}/{m}", str(exc))

        summary.append({
            "spec": dataclass_dict(spec), "dataset": spec.name,
            "design_role": design_role,
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

    summary_filename = os.path.basename(str(
        scfg.get("summary_filename", "controls_summary.json")))
    path = os.path.join(out_dir, summary_filename)
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
