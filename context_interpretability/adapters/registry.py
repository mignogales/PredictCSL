"""
Adapter factory: model display name -> configured :class:`TSFMAdapter`.

Capabilities come from ``configs/models/capabilities.yaml`` (family entry,
optionally overridden per display name); the model catalog is the repo's
single source of truth, ``experiments.models_config``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from context_interpretability.adapters.base import AdapterCapabilities
from context_interpretability.adapters.tsfm import TSFMAdapter

_CAPS_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "models",
                          "capabilities.yaml")

_CAP_FIELDS = {f for f in AdapterCapabilities.__dataclass_fields__}  # type: ignore[attr-defined]


def load_capability_config(path: Optional[str] = None) -> dict:
    import yaml
    with open(path or os.path.abspath(_CAPS_PATH)) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "families" not in cfg:
        raise ValueError("capabilities.yaml must define a 'families' mapping")
    return cfg


def capabilities_for(family: str, display: str,
                     caps_cfg: Optional[dict] = None) -> AdapterCapabilities:
    cfg = caps_cfg or load_capability_config()
    fam = dict(cfg["families"].get(family) or {})
    fam.update((cfg.get("models") or {}).get(display) or {})
    unknown = set(fam) - _CAP_FIELDS
    if unknown:
        raise ValueError(
            f"capabilities.yaml [{family}/{display}]: unknown fields {sorted(unknown)}")
    if not fam:
        raise ValueError(
            f"No capability entry for family {family!r} — add it to "
            "configs/models/capabilities.yaml (unsupported models must be "
            "declared, not guessed)")
    return AdapterCapabilities(**fam)


def available_models() -> List[str]:
    from experiments.models_config import CATALOG
    return [m.display for m in CATALOG]


def build_adapter(display: str, horizon: int, device: str = "cuda:0",
                  batch_size: int = 16,
                  dynamic_batching: bool = False,
                  batch_reference_context: int = 1024,
                  max_batch_size: Optional[int] = None,
                  caps_cfg: Optional[dict] = None) -> TSFMAdapter:
    from experiments.models_config import CATALOG
    match = [m for m in CATALOG if m.display == display]
    if not match:
        raise SystemExit(
            f"Unknown model {display!r}. Choices: {available_models()}")
    spec = match[0]
    caps = capabilities_for(spec.family, spec.display, caps_cfg)
    return TSFMAdapter(spec.model_id, spec.family, spec.display, caps,
                       horizon=horizon, device=device, batch_size=batch_size,
                       dynamic_batching=dynamic_batching,
                       batch_reference_context=batch_reference_context,
                       max_batch_size=max_batch_size)
