"""Direct mean/scale controls for attention-masking decompositions.

Unlike the matched-prefix control, these hooks leave every input value intact
and replace only the statistics used by the model's instance normalization.
They intentionally support only the two installed implementations whose
normalization paths have been verified: Chronos-Bolt and PatchTST-FM.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch


def _chronos_bolt_norm(handle):
    inner = getattr(handle, "model", handle)
    norm = getattr(inner, "instance_norm", None)
    if norm is None:
        raise RuntimeError("Could not find Chronos-Bolt instance_norm")
    return norm


def _patchtst_norm(handle):
    for obj in (handle, getattr(handle, "model", None),
                getattr(handle, "backbone", None)):
        if obj is None:
            continue
        norm = getattr(obj, "norm_fn", None)
        if norm is not None:
            return norm
        backbone = getattr(obj, "backbone", None)
        norm = getattr(backbone, "norm_fn", None)
        if norm is not None:
            return norm
    raise RuntimeError("Could not find PatchTST-FM backbone.norm_fn")


def _nan_stats(x):
    loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num(
        (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
    return loc, scale


def _chronos_bolt_override(handle, visible_timesteps, tail_mean, tail_scale):
    norm = _chronos_bolt_norm(handle)
    orig = norm.forward
    state = {"applied": 0}

    def wrapped(x, loc_scale=None):
        if loc_scale is not None:
            return orig(x, loc_scale=loc_scale)
        x32 = x.to(torch.float32)
        full_loc, full_scale = _nan_stats(x32)
        tail = x32[..., -int(visible_timesteps):]
        tail_loc, tail_sd = _nan_stats(tail)
        eps = float(getattr(norm, "eps", 1e-5))
        full_scale = torch.where(full_scale == 0, eps, full_scale)
        tail_sd = torch.where(tail_sd == 0, eps, tail_sd)
        loc = tail_loc if tail_mean else full_loc
        scale = tail_sd if tail_scale else full_scale
        state["applied"] += 1
        return orig(x, loc_scale=(loc, scale))

    norm.forward = wrapped
    return norm, orig, state


def _masked_tail_stats(x, mask, visible_timesteps, std_min):
    valid = torch.ones_like(x, dtype=torch.bool) if mask is None else ~mask.bool()
    means = []
    scales = []
    for row, row_valid in zip(x, valid):
        values = row[row_valid][-int(visible_timesteps):].to(torch.float32)
        if values.numel() == 0:
            raise RuntimeError("PatchTST normalization hook saw no valid history")
        mean = values.mean()
        scale = ((values - mean).square().mean()).sqrt()
        scale = torch.where(scale > std_min, scale, torch.ones_like(scale))
        means.append(mean)
        scales.append(scale)
    return torch.stack(means).unsqueeze(-1), torch.stack(scales).unsqueeze(-1)


def _patchtst_override(handle, visible_timesteps, tail_mean, tail_scale):
    norm = _patchtst_norm(handle)
    orig = norm.fit_transform
    state = {"applied": 0}

    def wrapped(x, mask=None):
        # Use the implementation's own full-stat calculation, then replace
        # exactly the selected statistic before its ordinary asinh transform.
        norm._get_statistics(x, mask)
        full_mean, full_scale = norm.mean.clone(), norm.std.clone()
        tail_loc, tail_sd = _masked_tail_stats(
            x, mask, visible_timesteps, float(norm.std_min))
        norm.mean = tail_loc if tail_mean else full_mean
        norm.std = tail_sd if tail_scale else full_scale
        state["applied"] += 1
        return norm.transform(x)

    norm.fit_transform = wrapped
    return norm, orig, state


def _reference_stats(reference_contexts: Sequence, family: str):
    """Mean/population-scale of each full reference history.

    Chronos-Bolt natively ignores missing observations. PatchTST-FM's official
    wrapper imputes missing values with that row's observed mean before RevIN,
    so reproduce that preprocessing before calculating the reference scale.
    """
    means = []
    scales = []
    for context in reference_contexts:
        values = np.asarray(context, dtype=np.float32).reshape(-1)
        finite = np.isfinite(values)
        if not finite.any():
            values = np.zeros(1, dtype=np.float32)
        elif family == "patchtst_fm" and not finite.all():
            values = values.copy()
            values[~finite] = float(values[finite].mean())
        else:
            values = values[finite]
        mean = float(values.mean())
        scale = float(np.sqrt(np.mean(np.square(values - mean))))
        means.append(mean)
        scales.append(scale)
    return (torch.tensor(means, dtype=torch.float32),
            torch.tensor(scales, dtype=torch.float32))


def _reference_stats_for_call(stats, x, *, scale_floor, unit_below_floor):
    batch = int(x.shape[0])
    reference_batch = int(stats[0].numel())
    if batch % reference_batch != 0:
        raise RuntimeError(
            "Normalization call cannot be aligned to its reference rows: "
            f"model batch={batch}, reference batch={reference_batch}")
    repeats = batch // reference_batch
    shape = (batch,) + (1,) * (x.ndim - 1)
    # Chronos-Bolt expands (B, Q, T) to (B*Q, T) for recursive long-horizon
    # decoding. Each source row's fixed statistics therefore repeat Q times.
    mean = stats[0].repeat_interleave(repeats).to(
        device=x.device).reshape(shape)
    scale = stats[1].repeat_interleave(repeats).to(
        device=x.device).reshape(shape)
    if unit_below_floor:
        scale = torch.where(scale > scale_floor, scale, torch.ones_like(scale))
    else:
        scale = torch.where(scale == 0, scale_floor, scale)
    return mean, scale


def _chronos_bolt_reference_override(handle, reference_contexts):
    norm = _chronos_bolt_norm(handle)
    orig = norm.forward
    stats = _reference_stats(reference_contexts, "chronos_bolt")
    state = {"applied": 0}

    def wrapped(x, loc_scale=None):
        if loc_scale is not None:
            return orig(x, loc_scale=loc_scale)
        loc, scale = _reference_stats_for_call(
            stats, x, scale_floor=float(getattr(norm, "eps", 1e-5)),
            unit_below_floor=False)
        state["applied"] += 1
        return orig(x, loc_scale=(loc, scale))

    norm.forward = wrapped
    return norm, orig, state


def _patchtst_reference_override(handle, reference_contexts):
    norm = _patchtst_norm(handle)
    orig = norm.fit_transform
    stats = _reference_stats(reference_contexts, "patchtst_fm")
    state = {"applied": 0}

    def wrapped(x, mask=None):
        norm.mean, norm.std = _reference_stats_for_call(
            stats, x, scale_floor=float(norm.std_min),
            unit_below_floor=True)
        state["applied"] += 1
        return norm.transform(x)

    norm.fit_transform = wrapped
    return norm, orig, state


@contextmanager
def normalization_stat_override(
    family,
    handle,
    visible_timesteps,
    *,
    tail_mean=False,
    tail_scale=False,
):
    """Temporarily replace the instance-normalization mean and/or scale."""
    if not tail_mean and not tail_scale:
        yield
        return
    if family == "chronos_bolt":
        norm, orig, state = _chronos_bolt_override(
            handle, visible_timesteps, tail_mean, tail_scale)
    elif family == "patchtst_fm":
        norm, orig, state = _patchtst_override(
            handle, visible_timesteps, tail_mean, tail_scale)
    else:
        raise NotImplementedError(
            f"Direct normalization controls are not verified for {family!r}")
    try:
        yield
    finally:
        if family == "chronos_bolt":
            norm.forward = orig
        else:
            norm.fit_transform = orig
        if state["applied"] == 0:
            raise RuntimeError(
                f"{family} normalization override was installed but never used")


@contextmanager
def normalization_reference_override(family, handle, reference_contexts):
    """Use full-history normalization stats while forecasting sliced inputs.

    ``reference_contexts`` must be in the same row order as the sliced batch
    passed to the model. The hook consumes the references across the model's
    ordinary microbatches and verifies that every supplied row was used once.
    """
    references = list(reference_contexts)
    if not references:
        raise ValueError("reference_contexts must contain at least one row")
    if family == "chronos_bolt":
        norm, orig, state = _chronos_bolt_reference_override(
            handle, references)
    elif family == "patchtst_fm":
        norm, orig, state = _patchtst_reference_override(
            handle, references)
    else:
        raise NotImplementedError(
            f"Full-history normalization is not verified for {family!r}")
    try:
        yield
    finally:
        if family == "chronos_bolt":
            norm.forward = orig
        else:
            norm.fit_transform = orig
        if state["applied"] == 0:
            raise RuntimeError(
                f"{family} normalization reference override was never used")
