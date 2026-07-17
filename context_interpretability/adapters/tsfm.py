"""
Generic TSFM adapter over the repo's existing model machinery.

Reuses (never reimplements):
  * ``experiments.build_context_length_dataset`` — ``setup_model`` +
    ``_forecast_uniform`` (the exact forecast path of the labeling pipeline);
  * ``experiments.context_attention_mask`` — Experiment 0's masking hooks;
  * the repeated-block discovery pattern of ``experiments.embedding_saturation``
    (``DEFAULT_HOOK_PATTERN``) for layer listing.

Runs on the SERVER (TSFM deps + GPU). Import is lazy so the rest of the
framework (tests, analysis, figures) works without any TSFM installed.

Internal-token conventions (verify per family on the server — the same caveat
as context_attention_mask):
  * token axis is dim -2 of a block's output tensor (B, ..., T, d);
  * context tokens LEAD the prefill sequence; patchtst_fm left-NaN-pads to its
    fixed native context, so a length-W input occupies the TRAILING
    ``ceil(W/P)`` tokens (handled via ``_left_pad_tokens``);
  * for generate()-based decoders only the PREFILL firing (the first one per
    forward) is captured/patched — with a KV cache the patched prefill
    propagates through decode.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from context_interpretability.adapters.base import (
    AdapterCapabilities,
    CapabilityError,
    InterpretabilityAdapter,
)

BLOCK_PATTERN = r"(?:^|\.)(?:layers?|blocks?|h|stacked_xf)\.\d+$"


def resolve_token_indices(spec, n_tokens_total: int) -> List[int]:
    """Resolve a token-index spec against the observed sequence length.

    ``spec`` is a list of ints (negatives allowed, python semantics) or
    ``{"trailing_from": k}`` meaning every token position >= k (chronos2's
    appended future tokens, whose count depends on the horizon)."""
    if isinstance(spec, dict) and "trailing_from" in spec:
        return list(range(int(spec["trailing_from"]), n_tokens_total))
    return [i % n_tokens_total for i in spec]


class TSFMAdapter(InterpretabilityAdapter):

    def __init__(self, model_id: str, family: str, display: str,
                 capabilities: AdapterCapabilities, horizon: int,
                 device: str = "cuda:0", batch_size: int = 16):
        super().__init__(display, capabilities, horizon, device, batch_size)
        self.model_id = model_id
        self.family = family
        self._base = None
        self._backbone = None
        self._layer_names: Optional[List[str]] = None
        self._timesfm_cfg = None          # (width, horizon) currently compiled

    # -- lifecycle -------------------------------------------------------------

    def load(self) -> None:
        if self._base is not None:
            return
        if self.family == "timesfm":
            # EAGER load — the pipeline's load_timesfm rebuilds a (potentially
            # torch.compile'd) model per context width, which runtime hooks and
            # the make_attn_mask wrap can't reliably reach. Loading once with
            # torch_compile=False keeps every forward eager and hookable;
            # compile(ForecastConfig) below is just eager-closure setup (the
            # "compiled_decode" name is a misnomer). Sanity-check on the server
            # that eager forecasts match the compiled pipeline's.
            import timesfm
            self._base = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_id, torch_compile=False)
            self._timesfm_cfg = None      # (width, horizon) currently compiled
        else:
            from experiments.build_context_length_dataset import setup_model
            self._base = setup_model(self.family, self.model_id, self.device)
        self._refresh_patch_length()

    def _timesfm_ensure_config(self, width: int) -> None:
        if self._timesfm_cfg == (width, self.horizon):
            return
        import timesfm
        self._base.compile(timesfm.ForecastConfig(
            max_context=int(width), max_horizon=self.horizon,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, per_core_batch_size=self.batch_size,
            infer_is_positive=True, fix_quantile_crossing=True,
        ))
        self._timesfm_cfg = (int(width), self.horizon)

    def close(self) -> None:
        self._base = None
        self._backbone = None
        try:
            import torch
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _refresh_patch_length(self) -> None:
        """Resolve patch length from the loaded config where possible; the yaml
        declaration is the fallback. Both requested + resolved end up in
        run_meta via metadata()."""
        try:
            if self.family == "patchtst_fm":
                from experiments.context_attention_mask import _patchtstfm_patch_len
                self.capabilities.patch_length = _patchtstfm_patch_len(self._base)
            elif self.family == "chronos2":
                from experiments.context_attention_mask import (
                    _chronos2_inner_module, _chronos2_patch_step)
                self.capabilities.patch_length = _chronos2_patch_step(
                    _chronos2_inner_module(self._base))
            elif self.family == "moirai":
                from experiments.context_attention_mask import _moirai2_patch_size
                p = _moirai2_patch_size(self._base)
                if p:
                    self.capabilities.patch_length = p
        except Exception as exc:  # noqa: BLE001 — keep the declared fallback
            print(f"[{self.name}] patch-length resolution failed "
                  f"({exc!r}); using declared {self.capabilities.patch_length}")

    def metadata(self) -> dict:
        meta = super().metadata()
        meta.update({"model_id": self.model_id, "family": self.family})
        try:
            import torch
            bb = self._resolve_backbone()
            meta["precision"] = str(next(bb.parameters()).dtype)
            meta["n_parameters"] = sum(p.numel() for p in bb.parameters())
        except Exception:  # noqa: BLE001
            meta["precision"] = "unknown"
        return meta

    # -- forecasting -------------------------------------------------------------

    def _uniform_forecast(self, x, width: int, batch_size: int):
        """Median forecast for a uniform-width torch batch (n, W, 1) -> (n, H).

        timesfm routes through the persistent EAGER model (see load()) so hooks
        and the attention-mask wrap actually fire; every other family uses the
        pipeline's ``_forecast_uniform`` verbatim."""
        import torch
        from experiments import build_context_length_dataset as bcl
        if self.family == "timesfm":
            self._timesfm_ensure_config(width)
            meds = [bcl.predict_timesfm(self._base, x[s:s + batch_size],
                                        self.horizon, self.device)
                    for s in range(0, x.shape[0], batch_size)]
            return torch.cat(meds, dim=0)
        return bcl._forecast_uniform(self.family, self._base, self.model_id,
                                     x, width, self.horizon, batch_size,
                                     self.device)

    def forecast(self, contexts: np.ndarray) -> np.ndarray:
        import torch
        self.load()
        x = torch.from_numpy(
            np.ascontiguousarray(contexts, dtype=np.float32)).unsqueeze(-1)
        with torch.no_grad():
            med = self._uniform_forecast(x, int(contexts.shape[1]),
                                         self.batch_size)
        return med.float().cpu().numpy()

    # -- Experiment 0: attention masking ------------------------------------------

    def forecast_attention_masked(self, contexts: np.ndarray,
                                  visible_timesteps: int) -> np.ndarray:
        if not self.capabilities.supports_attention_masking:
            raise CapabilityError(f"{self.name}: attention masking unsupported")
        import torch
        from experiments.context_attention_mask import context_attention_mask
        self.load()
        W = int(contexts.shape[1])
        x = torch.from_numpy(
            np.ascontiguousarray(contexts, dtype=np.float32)).unsqueeze(-1)
        with torch.no_grad(), context_attention_mask(
                self.family, self._base, int(visible_timesteps), W):
            med = self._uniform_forecast(x, W, self.batch_size)
        return med.float().cpu().numpy()

    # -- layer access ---------------------------------------------------------------

    def _resolve_backbone(self):
        import torch
        if self._backbone is not None:
            return self._backbone
        self.load()
        obj = self._base
        # unwrap common pipeline wrappers first
        for attr in ("model", "inner_model", "module"):
            inner = getattr(obj, attr, None)
            if isinstance(inner, torch.nn.Module):
                obj = inner
                break
        if not isinstance(obj, torch.nn.Module):
            from experiments.embedding_saturation import find_backbone
            match, _ = find_backbone(self._base, BLOCK_PATTERN)
            if match is None:
                raise CapabilityError(
                    f"{self.name}: no nn.Module backbone found (family "
                    f"{self.family}) — layer-level methods unavailable")
            obj = match
        self._backbone = obj
        return obj

    def get_layer_names(self) -> List[str]:
        if self._layer_names is None:
            rx = re.compile(BLOCK_PATTERN)
            bb = self._resolve_backbone()
            self._layer_names = [n for n, _ in bb.named_modules() if rx.search(n)]
            if not self._layer_names:
                raise CapabilityError(
                    f"{self.name}: no repeated-block modules matched "
                    f"{BLOCK_PATTERN!r} — dump modules on the server")
        return list(self._layer_names)

    def _modules_by_name(self, names: Sequence[str]) -> Dict[str, object]:
        bb = self._resolve_backbone()
        lookup = dict(bb.named_modules())
        missing = [n for n in names if n not in lookup]
        if missing:
            raise CapabilityError(f"{self.name}: unknown layers {missing}")
        return {n: lookup[n] for n in names}

    @staticmethod
    def _out_hidden(out):
        return out[0] if isinstance(out, (tuple, list)) else out

    @staticmethod
    def _replace_hidden(out, new_hidden):
        if isinstance(out, tuple):
            return (new_hidden,) + tuple(out[1:])
        if isinstance(out, list):
            return [new_hidden] + list(out[1:])
        return new_hidden

    @contextmanager
    def _hooks(self, hook_map):
        """Install {module: hook_fn} forward hooks; always removed."""
        handles = [m.register_forward_hook(fn) for m, fn in hook_map]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _forward_chunk(self, x_chunk) -> "object":
        """One uniform forward over a chunk (single internal batch)."""
        import torch
        with torch.no_grad():
            return self._uniform_forecast(x_chunk, int(x_chunk.shape[1]),
                                          int(x_chunk.shape[0]))

    # -- Experiment 2: activation capture / patching --------------------------------

    def capture_activations(self, contexts: np.ndarray, layers: List[str]
                            ) -> Dict[str, "object"]:
        if not self.capabilities.supports_activation_patching:
            raise CapabilityError(f"{self.name}: activation capture unsupported")
        import torch
        mods = self._modules_by_name(layers)
        x = torch.from_numpy(
            np.ascontiguousarray(contexts, dtype=np.float32)).unsqueeze(-1)
        store: Dict[str, List[torch.Tensor]] = {n: [] for n in layers}
        fired: Dict[str, bool] = {}

        def make_hook(name):
            def hook(_m, _inp, out):
                if fired.get(name):          # keep only the prefill firing
                    return
                h = self._out_hidden(out)
                if torch.is_tensor(h):
                    fired[name] = True
                    store[name].append(h.detach().to(torch.float32).cpu())
            return hook

        with self._hooks([(m, make_hook(n)) for n, m in mods.items()]):
            for start in range(0, x.shape[0], self.batch_size):
                fired.clear()
                self._forward_chunk(x[start:start + self.batch_size])
        out: Dict[str, object] = {}
        for n in layers:
            if not store[n]:
                raise CapabilityError(
                    f"{self.name}: layer {n} never fired with a tensor output")
            out[n] = torch.cat(store[n], dim=0)
        return out

    def run_with_activation_patch(self, contexts: np.ndarray, layer_name: str,
                                  token_indices, replacement_activation
                                  ) -> np.ndarray:
        if not self.capabilities.supports_activation_patching:
            raise CapabilityError(f"{self.name}: activation patching unsupported")
        import torch
        mods = self._modules_by_name([layer_name])
        x = torch.from_numpy(
            np.ascontiguousarray(contexts, dtype=np.float32)).unsqueeze(-1)
        repl_all = replacement_activation
        state = {"start": 0, "chunk": 0, "fired": False}
        preds: List[torch.Tensor] = []

        def hook(_m, _inp, out):
            if state["fired"]:
                return
            h = self._out_hidden(out)
            if not torch.is_tensor(h):
                return
            state["fired"] = True
            idx = resolve_token_indices(token_indices, h.shape[-2])
            if not idx:
                return
            if max(idx) >= h.shape[-2]:
                raise CapabilityError(
                    f"{self.name}/{layer_name}: token index {max(idx)} out of "
                    f"range for seq len {h.shape[-2]} — block->token mapping "
                    f"needs a per-family fix (verify on server)")
            repl = repl_all[state["start"]:state["start"] + state["chunk"]]
            repl = repl.to(device=h.device, dtype=h.dtype)
            if repl.shape[-2] != h.shape[-2]:
                raise CapabilityError(
                    f"{self.name}/{layer_name}: clean/corrupted seq lengths "
                    f"differ ({repl.shape[-2]} vs {h.shape[-2]}) — patching "
                    f"requires equal-length inputs (spec §5.3)")
            new_h = h.clone()
            new_h[..., idx, :] = repl[..., idx, :]
            return self._replace_hidden(out, new_h)

        with self._hooks([(mods[layer_name], hook)]):
            for start in range(0, x.shape[0], self.batch_size):
                chunk = x[start:start + self.batch_size]
                state.update(start=start, chunk=int(chunk.shape[0]), fired=False)
                preds.append(self._forward_chunk(chunk).float().cpu())
        return torch.cat(preds, dim=0).numpy()

    def _left_pad_tokens(self, context_length: int) -> int:
        """Leading non-context tokens in the prefill sequence (patchtst_fm's
        NaN pad to its fixed native context; 0 elsewhere)."""
        if self.capabilities.fixed_native_context:
            p = self.capabilities.patch_length or 1
            native = self.capabilities.maximum_context_length
            return max(0, (native - int(context_length))) // p
        return 0

    def input_block_to_token_indices(self, block_start: int, block_end: int,
                                     context_length: int) -> List[int]:
        p = self.capabilities.patch_length or 1
        off = self._left_pad_tokens(context_length)
        n_ctx_tokens = off + math.ceil(context_length / p)
        tok_lo = off + (int(context_length) - int(block_end)) // p
        tok_hi = off + math.ceil((int(context_length) - int(block_start)) / p)
        return list(range(max(0, tok_lo), min(n_ctx_tokens, tok_hi)))

    def forecast_token_indices(self, context_length: int):
        p = self.capabilities.patch_length or 1
        if self.family == "chronos2":
            # future patch tokens are appended AFTER the context patches; their
            # count depends on the horizon, so resolve "trailing" at hook time.
            return {"trailing_from": math.ceil(context_length / p)}
        if self.family in ("sundial", "timemoe"):
            return [-1]                      # last position drives generation
        return None                          # no dedicated readout token

    # -- Experiment 3: forecast lens ---------------------------------------------

    def forecast_from_layer(self, contexts: np.ndarray, layer_name: str
                            ) -> np.ndarray:
        """Identity-skip lens: blocks DEEPER than ``layer_name`` (same stack,
        higher index) pass their input through unchanged, then the model's own
        final norm + frozen head run as usual. Valid for pre/post-norm residual
        stacks where block(x) ~ x + f(x); declared per family in the capability
        yaml and never silently approximated elsewhere (spec §6.2)."""
        if not self.capabilities.supports_forecast_lens:
            raise CapabilityError(f"{self.name}: forecast lens unsupported")
        import torch
        names = self.get_layer_names()
        m = re.match(r"(.*?)(\d+)$", layer_name)
        if layer_name not in names or not m:
            raise CapabilityError(f"{self.name}: bad lens layer {layer_name!r}")
        prefix, idx = m.group(1), int(m.group(2))
        deeper = [n for n in names
                  if n.startswith(prefix) and n[len(prefix):].isdigit()
                  and int(n[len(prefix):]) > idx]
        mods = self._modules_by_name(deeper) if deeper else {}

        def identity_hook(_m, inp, out):
            h = self._out_hidden(out)
            if not torch.is_tensor(h) or not inp or not torch.is_tensor(inp[0]):
                raise CapabilityError(
                    f"{self.name}: identity-skip lens cannot pass through a "
                    "non-tensor block io — mark family unsupported")
            return self._replace_hidden(out, inp[0])

        x = torch.from_numpy(
            np.ascontiguousarray(contexts, dtype=np.float32)).unsqueeze(-1)
        preds: List[torch.Tensor] = []
        with self._hooks([(mod, identity_hook) for mod in mods.values()]):
            for start in range(0, x.shape[0], self.batch_size):
                preds.append(
                    self._forward_chunk(
                        x[start:start + self.batch_size]).float().cpu())
        return torch.cat(preds, dim=0).numpy()

    # -- Experiment 5: integrated gradients -----------------------------------------

    def forecast_differentiable(self, contexts_tensor):
        """Differentiable median forecast. Only families whose predict wrapper
        is a plain forward (no sampling / generate / pipeline no_grad) can
        support this; verified at runtime by checking grad_fn."""
        if not self.capabilities.supports_integrated_gradients:
            raise CapabilityError(
                f"{self.name}: integrated gradients unsupported (declared)")
        import torch
        from experiments import build_context_length_dataset as bcl
        self.load()
        if self.family != "patchtst_fm":
            raise CapabilityError(
                f"{self.name}: no differentiable forward wrapper for family "
                f"{self.family}")
        x = contexts_tensor.unsqueeze(-1)             # (N, W, 1), requires_grad
        with torch.enable_grad():
            pred = bcl.predict_patchtst_fm(self._base, x, self.horizon,
                                           self.device)
        if pred.grad_fn is None:
            raise CapabilityError(
                f"{self.name}: forward produced no grad_fn (an internal "
                "no_grad/detach breaks the gradient path)")
        return pred
