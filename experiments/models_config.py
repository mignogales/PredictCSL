"""
Single source of truth for the TSFM model set.

Every script that used to carry its own hand-maintained model list now reads from
here, so the catalog and the "what do we actually run" set can't drift apart
again (they had: TimeMoE/ChronosBolt-Small were in some lists and not others).

Two distinct concepts live in one ordered table:

  * **Catalog** — every model with a loader/predict wrapper in
    ``build_context_length_dataset.py``. Order is LOAD-BEARING: stage 1 selects a
    model by ``--model-idx`` into this list, so entries are only ever *appended*,
    never reordered or removed (doing so shifts indices and breaks resume on the
    server). Use :func:`catalog`.

  * **Run set** — the subset the pipeline actually processes end-to-end
    (``run_all`` + variants, the v5 GiftEval ablation, the predictor-overhead
    summary). Flagged per row by ``run``. Use :func:`models_to_run` /
    :func:`run_pairs`.

A model can be in the catalog (loader exists, ``--model-idx`` reachable) but out
of the run set (``run=False``) — e.g. ChronosBolt-Small, superseded by
ChronosBolt-Base.
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple


class ModelSpec(NamedTuple):
    model_id: str   # HuggingFace / loader id passed to the load_* wrappers
    family: str     # loader + arch family key (chronos2, moirai, timesfm, …)
    display: str    # unique run label / output-dir basename / --models selector
    run: bool       # True -> part of the end-to-end pipeline run set


# ORDER = build_context_length_dataset stage-1 ``--model-idx`` order. APPEND ONLY.
CATALOG: List[ModelSpec] = [
    ModelSpec("autogluon/chronos-2-small",       "chronos2",     "Chronos2-Small",     True),
    # ChronosBolt-Small: loader kept (catalog/--model-idx stable) but out of the
    # run set — superseded by ChronosBolt-Base below.
    ModelSpec("amazon/chronos-bolt-small",       "chronos_bolt", "ChronosBolt-Small",  False),
    ModelSpec("Salesforce/moirai-2.0-R-small",   "moirai",       "Moirai2-Small",      True),
    ModelSpec("google/timesfm-2.5-200m-pytorch", "timesfm",      "TimesFM2.5-200M",    True),
    ModelSpec("ibm-research/patchtst-fm-r1",     "patchtst_fm",  "PatchTST-FM-R1",     True),
    ModelSpec("thuml/sundial-base-128m",         "sundial",      "Sundial-Base-128M",  True),
    # TimeMoE: autoregressive token-by-token decode is slow at large horizons over
    # the full window grid — keep an eye on stage-1 wall-clock when it's enabled.
    ModelSpec("Maple728/TimeMoE-200M",           "timemoe",      "TimeMoE-200M",       True),
    # Appended (not inserted) to keep existing --model-idx positions stable.
    ModelSpec("autogluon/chronos-2-synth",       "chronos2",     "Chronos2-Synth",     True),
    ModelSpec("autogluon/chronos-2-base",        "chronos2",     "Chronos2-Base",      True),
    ModelSpec("amazon/chronos-bolt-base",        "chronos_bolt", "ChronosBolt-Base",   True),
    ModelSpec("Datadog/Toto-2.0-313m",           "toto",         "Toto-2.0-313m",      True),
    ModelSpec("ibm-granite/granite-timeseries-flowstate-r1", "flowstate", "FlowState-R1", True),
    ModelSpec("NX-AI/TiRex",                     "tirex",        "TiRex",              True),
]


def catalog() -> List[Tuple[str, str, str]]:
    """Full ``(model_id, family, display)`` catalog in ``--model-idx`` order.

    For ``build_context_length_dataset`` (stage 1), which indexes this list by
    ``--model-idx`` and must see every model with a loader.
    """
    return [(m.model_id, m.family, m.display) for m in CATALOG]


def models_to_run() -> List[Tuple[str, str, str]]:
    """``(model_id, family, display)`` for the run set (``run=True``), in catalog
    order. For ``run_all`` (+ v2/v3/v4) and the v5 GiftEval ablation."""
    return [(m.model_id, m.family, m.display) for m in CATALOG if m.run]


def run_pairs() -> List[Tuple[str, str]]:
    """``(display, family)`` for the run set, in catalog order. For
    ``summarize_predictor_overhead``, which keys per-model predictors on display."""
    return [(m.display, m.family) for m in CATALOG if m.run]


def run_displays() -> List[str]:
    """Display names of the run set (convenience for CLI help / validation)."""
    return [m.display for m in CATALOG if m.run]
