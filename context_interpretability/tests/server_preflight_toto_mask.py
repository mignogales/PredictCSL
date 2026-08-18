"""Server-only semantic preflight for Toto's attention-mask intervention."""

from __future__ import annotations

import numpy as np

from context_interpretability.adapters.registry import build_adapter
from context_interpretability.data.validation_generators import (
    load_harmonic_pools)


def main() -> None:
    adapter = build_adapter(
        "Toto-2.0-313m", horizon=64, device="cuda:0", batch_size=2,
        dynamic_batching=False)
    data = load_harmonic_pools(
        2, series_length=256, horizon=64, seed=991,
        periods=[32, 64, 128], scales=[1.0], max_tones=3)[0]
    x = data.contexts
    clean = adapter.forecast(x)
    top = adapter.forecast_attention_masked(x, 256)
    masked = adapter.forecast_attention_masked(x, 128)
    sliced = adapter.forecast(x[:, -128:])

    # Permuting only the hidden prefix must not change a properly masked
    # forecast. Reverse order preserves its value distribution and hence the
    # full-window mean/variance, isolating attention connectivity.
    permuted = x.copy()
    permuted[:, :128] = permuted[:, :128][:, ::-1]
    masked_permuted = adapter.forecast_attention_masked(permuted, 128)
    clean_permuted = adapter.forecast(permuted)

    top_diff = float(np.max(np.abs(top - clean)))
    masked_prefix_diff = float(np.max(np.abs(masked_permuted - masked)))
    clean_prefix_diff = float(np.max(np.abs(clean_permuted - clean)))
    mask_vs_slice = float(np.mean(np.abs(masked - sliced)))
    print({
        "resolved_patch_length": adapter.patch_length,
        "top_grid_max_abs_diff": top_diff,
        "masked_hidden_prefix_max_abs_diff": masked_prefix_diff,
        "clean_hidden_prefix_max_abs_diff": clean_prefix_diff,
        "masked_vs_slice_mean_abs_diff": mask_vs_slice,
        "finite": bool(np.isfinite(masked).all()),
    })
    assert np.isfinite(masked).all()
    assert top_diff <= 1e-6
    assert masked_prefix_diff <= 2e-4, (
        "hidden-prefix permutation changed Toto's masked forecast")
    adapter.close()


if __name__ == "__main__":
    main()
