"""Local self-test for embedding_saturation's novel logic (no TSFM libs needed).

Exercises HiddenCapture, the saturation metrics, and the full
collect_grid_embeddings -> _write_cell path against a dummy nn.Module, by
monkeypatching the model dispatch. Run with a torch+numpy python:

    /usr/bin/python3 -m experiments._selftest_embsat
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from experiments import embedding_saturation as E


def test_saturation_math():
    windows = np.array([32, 64, 128, 256])
    # Case 1: cleanly converging -> small marginals after the first step.
    conv = np.array([[0.5, 0.04, 0.01, 0.0]], dtype=np.float32)  # to_asymp
    marg = np.array([[0.0, 0.04, 0.01, 0.005]], dtype=np.float32)
    curves = {"to_asymp": conv, "marginal": marg}
    s = E.summarize_saturation(curves, windows)
    # marginal[:,0] is masked, so L* must NOT be the smallest window trivially.
    assert s["Lstar_marginal_0.05"][0] == 64, s["Lstar_marginal_0.05"]
    assert s["Lstar_asymp_0.05"][0] == 64, s["Lstar_asymp_0.05"]
    assert s["drift_score"][0] == 0.0 or s["drift_score"][0] < 0.05

    # Case 2: never converges -> L* = max window.
    bad = {"to_asymp": np.array([[0.5, 0.4, 0.3, 0.0]], dtype=np.float32),
           "marginal": np.array([[0.0, 0.3, 0.3, 0.3]], dtype=np.float32)}
    s2 = E.summarize_saturation(bad, windows)
    assert s2["Lstar_marginal_0.05"][0] == 256, s2["Lstar_marginal_0.05"]

    # Case 3: plateau then drift (stale context re-perturbs) -> drift_score>0.
    drift = {"to_asymp": np.array([[0.5, 0.02, 0.3, 0.0]], dtype=np.float32),
             "marginal": np.array([[0.0, 0.02, 0.4, 0.1]], dtype=np.float32)}
    s3 = E.summarize_saturation(drift, windows)
    assert s3["drift_score"][0] >= 0.4 - 1e-6, s3["drift_score"]
    print("ok  saturation math (masking, no-converge, drift)")


def test_curves_shapes():
    rng = np.random.RandomState(0)
    emb = rng.randn(5, 4, 8).astype(np.float32)
    c = E.saturation_curves(emb)
    assert c["to_asymp"].shape == (5, 4)
    assert np.allclose(c["to_asymp"][:, -1], 0.0, atol=1e-5), "asymptote col not 0"
    assert np.allclose(c["marginal"][:, 0], 0.0), "marginal col0 not 0"
    print("ok  saturation_curves shapes + conventions")


class _Dummy(nn.Module):
    """Two-block stack named like an HF ModuleList (matches the hook regex)."""

    def __init__(self, d=8):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, d), nn.Linear(d, d)])

    def forward(self, x):                # x: (B, W, 1)
        h = torch.tanh(self.layers[0](x))
        return self.layers[1](h)          # (B, W, d) — last block to fire


def test_hidden_capture():
    m = _Dummy()
    cap = E.HiddenCapture(m, E.DEFAULT_HOOK_PATTERN)
    assert cap.matched == ["layers.0", "layers.1"], cap.matched
    with torch.no_grad():
        m(torch.randn(3, 10, 1))
    got = cap.read()
    cap.remove()
    assert got is not None
    last, mean = got
    assert last.shape == (3, 8) and mean.shape == (3, 8), (last.shape, mean.shape)
    print("ok  HiddenCapture (regex match, last-token + mean)")


class _ARDummy(nn.Module):
    """Single block invoked TWICE per forward — stands in for a 2-step rollout."""

    def __init__(self, d=8):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, d)])

    def forward(self, x):                       # x: (B, W, 1)
        h1 = torch.tanh(self.layers[0](x))      # fire 1: "prefill" / step 0
        seed = x.mean(dim=1, keepdim=True)
        h2 = torch.tanh(self.layers[0](seed))   # fire 2: "decode" / step 1
        return h2


def test_read_steps_multistep():
    m = _ARDummy()
    cap = E.HiddenCapture(m, E.DEFAULT_HOOK_PATTERN)
    with torch.no_grad():
        m(torch.randn(3, 10, 1))
    steps = cap.read_steps()
    legacy = cap.read()
    cap.remove()
    last, mean = steps
    assert last.shape == (3, 2, 8), last.shape       # (B, S=2, d)
    assert mean.shape == (3, 2, 8), mean.shape
    # read() must equal the FINAL step (legacy single-embedding contract).
    assert torch.allclose(legacy[0], last[:, -1, :]), "read() != final step"
    print("ok  read_steps captures per-step (S=2) and read() == final step")


class _NestedBlock(nn.Module):
    """Block whose inner MLP expands to a DIFFERENT dim (dff != d)."""

    def __init__(self, d, dff):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.layers = nn.ModuleList([nn.Linear(d, dff), nn.Linear(dff, d)])

    def forward(self, x):
        h = torch.relu(self.mlp.layers[0](x))   # (B, W, dff)  <- WRONG dim
        return self.mlp.layers[1](h)            # (B, W, d)    <- block output


class _NestedDummy(nn.Module):
    """Mimics PatchTST/T5 nesting: blocks.N + blocks.N.mlp.layers.M both match."""

    def __init__(self, d=8, dff=32):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.blocks = nn.ModuleList([_NestedBlock(d, dff) for _ in range(3)])

    def forward(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        return h


def test_nested_capture_picks_block_not_mlp():
    """Parent block hook must fire AFTER its inner mlp.layers, so we capture the
    d-dim block output, never the dff-dim MLP intermediate."""
    d, dff = 8, 32
    m = _NestedDummy(d, dff)
    cap = E.HiddenCapture(m, E.DEFAULT_HOOK_PATTERN)
    # Both the blocks and the inner mlp layers match the regex.
    assert "blocks.2" in cap.matched
    assert "blocks.2.mlp.layers.0" in cap.matched
    with torch.no_grad():
        m(torch.randn(2, 6, 1))
    last, mean = cap.read()
    cap.remove()
    assert last.shape == (2, d), f"captured dim {last.shape[1]} != block dim {d} " \
        f"(grabbed an MLP intermediate of dim {dff}?)"
    print("ok  nested capture picks block output (dim %d), not mlp intermediate (%d)"
          % (d, dff))


def test_find_backbone_through_wrapper():
    """find_backbone must locate the trunk buried under a non-nn.Module wrapper
    with an arbitrary attribute name (the TimesFM/Moirai failure mode)."""
    inner = _NestedDummy(d=8, dff=32)

    class _Wrapper:                       # NOT an nn.Module
        def __init__(self, m):
            self._weird_private_name = m
            self.cfg = {"unrelated": 1}

    w = _Wrapper(inner)
    match, fb = E.find_backbone(w, E.DEFAULT_HOOK_PATTERN)
    assert match is inner, f"got {type(match)}"
    # And via a property (not in __dict__).
    class _PropWrapper:
        def __init__(self, m): self.__m = m
        @property
        def decode_model(self): return self.__m
    match2, _ = E.find_backbone(_PropWrapper(inner), E.DEFAULT_HOOK_PATTERN)
    assert match2 is inner
    print("ok  find_backbone reaches trunk through wrapper attr + property")


def test_integration(monkeypatched=True):
    model = _Dummy()
    model.eval()
    # Monkeypatch the two model-specific hooks so the dummy stands in for a TSFM.
    E._resolve_backbone = lambda family, base, mid, dev: base
    E._predict_dispatch = lambda family, runner, xb, h, dev: runner(xb.to(dev))

    N, W = 6, 32
    contexts = np.random.RandomState(1).randn(N, W).astype(np.float32)
    real_lengths = np.full(N, W, dtype=np.int64)
    windows = [4, 8, 16, 32]

    emb_last, emb_mean, matched = E.collect_grid_embeddings(
        "dummy", model, "dummy-id", contexts, real_lengths, horizon=8,
        windows=windows, batch_size=4, device="cpu", pattern=E.DEFAULT_HOOK_PATTERN)
    # _Dummy fires once per forward -> single generation step (S=1).
    assert emb_last.shape == (N, len(windows), 1, 8), emb_last.shape
    assert matched == ["layers.0", "layers.1"]

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "cell")
        meta = {"real_lengths": real_lengths,
                "n_segments": np.ones(N, dtype=np.int32),
                "horizon": 8, "source": "synthetic", "model": "dummy",
                "hooked_modules": len(matched)}
        summary = E._write_cell(out, windows, emb_last, emb_mean, meta,
                                save_embeddings=True, title="dummy")
        assert os.path.exists(os.path.join(out, "saturation.npz"))
        assert os.path.exists(os.path.join(out, "summary.json"))
        assert os.path.exists(os.path.join(out, "embeddings.npz"))
        z = np.load(os.path.join(out, "saturation.npz"))
        assert z["to_asymp"].shape == (N, len(windows))
        assert np.allclose(z["to_asymp"][:, -1], 0.0, atol=1e-5)
        assert "Lstar_marginal_0.05" in z.files
        assert "n_segments" in z.files and "real_lengths" in z.files
        # New per-step / generation arrays and the step count.
        assert z["to_asymp_steps"].shape == (N, len(windows), 1)
        assert z["gen_marginal"].shape == (N, len(windows), 1)
        assert int(z["n_steps"]) == 1
        assert summary["d_model"] == 8 and summary["n_series"] == N
        assert summary["n_steps"] == 1
    print("ok  integration collect_grid_embeddings -> _write_cell")


if __name__ == "__main__":
    test_saturation_math()
    test_curves_shapes()
    test_hidden_capture()
    test_read_steps_multistep()
    test_nested_capture_picks_block_not_mlp()
    test_find_backbone_through_wrapper()
    test_integration()
    print("\nALL SELF-TESTS PASSED")
