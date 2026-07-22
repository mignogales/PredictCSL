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
import logging
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
                 device: str = "cuda:0", batch_size: int = 16,
                 dynamic_batching: bool = False,
                 batch_reference_context: int = 8192,
                 max_batch_size: Optional[int] = None):
        super().__init__(display, capabilities, horizon, device, batch_size,
                         dynamic_batching, batch_reference_context,
                         max_batch_size)
        self.model_id = model_id
        self.family = family
        self._base = None
        self._backbone = None
        self._layer_names: Optional[List[str]] = None
        self._timesfm_cfg = None   # (width, horizon, batch size) active config
        self._moirai_runner = None
        self._moirai_cfg = None    # (width, horizon) active forecast wrapper
        self._tuned_batch_sizes: Dict[int, int] = {}
        self._batch_search_complete = set()

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
        if self.family == "patchtst_fm":
            # The upstream forward logs context/forecast lengths at INFO on
            # every batch.  Besides being redundant here, those lines break
            # tqdm's in-place rendering.  Silence only this module; warnings
            # and errors, and logging from every other model, remain visible.
            logging.getLogger(
                "tsfm_public.models.patchtst_fm.modeling_patchtst_fm"
            ).setLevel(logging.WARNING)
        self._refresh_patch_length()

    def _timesfm_ensure_config(self, width: int, batch_size: int) -> None:
        if self._timesfm_cfg == (width, self.horizon, batch_size):
            return
        import timesfm
        self._base.compile(timesfm.ForecastConfig(
            max_context=int(width), max_horizon=self.horizon,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, per_core_batch_size=batch_size,
            infer_is_positive=True, fix_quantile_crossing=True,
        ))
        self._timesfm_cfg = (int(width), self.horizon, int(batch_size))

    def close(self) -> None:
        self._moirai_runner = None
        self._moirai_cfg = None
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

    def batch_size_for(self, context_length: int, n_samples: Optional[int] = None
                       ) -> int:
        tuned = self._tuned_batch_sizes.get(int(context_length))
        if tuned is None:
            return super().batch_size_for(context_length, n_samples)
        if n_samples is not None:
            return max(1, min(tuned, int(n_samples)))
        return tuned

    # -- forecasting -------------------------------------------------------------

    def _uniform_forecast(self, x, width: int, batch_size: int):
        """Median forecast for a uniform-width torch batch (n, W, 1) -> (n, H).

        timesfm routes through the persistent EAGER model (see load()) so hooks
        and the attention-mask wrap actually fire; every other family uses the
        pipeline's ``_forecast_uniform`` verbatim."""
        import torch
        from experiments import build_context_length_dataset as bcl
        if self.family == "timesfm":
            self._timesfm_ensure_config(width, batch_size)
            meds = [bcl.predict_timesfm(self._base, x[s:s + batch_size],
                                        self.horizon, self.device)
                    for s in range(0, x.shape[0], batch_size)]
            return torch.cat(meds, dim=0)
        if self.family == "moirai":
            # `_forecast_uniform` constructs a Moirai2Forecast, moves it to
            # CUDA, deletes it, and calls cuda.empty_cache() on every call.
            # Interpretability cells call this path hundreds of times at the
            # same width, so retain the lightweight forecast wrapper and the
            # allocator cache until the width actually changes.
            cfg = (int(width), self.horizon)
            if self._moirai_cfg != cfg:
                self._moirai_runner = bcl._build_moirai(
                    self._base, self.horizon, width, self.device)
                self._moirai_cfg = cfg
            if self.dynamic_batching and str(self.device).startswith("cuda"):
                return self._autotuned_forecast(
                    x, width, batch_size, bcl.predict_moirai,
                    self._moirai_runner)
            meds = [bcl.predict_moirai(
                self._moirai_runner, x[s:s + batch_size], self.horizon,
                self.device) for s in range(0, x.shape[0], batch_size)]
            return torch.cat(meds, dim=0)
        if (self.family == "chronos2" and self.dynamic_batching
                and str(self.device).startswith("cuda")):
            return self._autotuned_forecast(
                x, width, batch_size, bcl.predict_chronos2, self._base)
        return bcl._forecast_uniform(self.family, self._base, self.model_id,
                                     x, width, self.horizon, batch_size,
                                     self.device)

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        try:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:  # noqa: BLE001
            pass
        return (isinstance(exc, RuntimeError)
                and "out of memory" in str(exc).lower())

    def _autotuned_forecast(self, x, width: int, initial_batch: int,
                            predict_fn, runner):
        """Grow to the largest observed-safe CUDA batch for this width.

        Successful chunks are never repeated. During the first sufficiently
        large call the chunk sizes grow geometrically; an OOM retries only the
        failed chunk at the last successful size. The result is cached for all
        subsequent experiment calls at the same context width.
        """
        import torch
        n = int(x.shape[0])
        limit = min(self.max_batch_size, n)
        cached = self._tuned_batch_sizes.get(int(width))
        current = min(cached or max(1, initial_batch), limit)
        probing = (int(width) not in self._batch_search_complete
                   and current < limit)
        last_success = 0
        start = 0
        medians = []
        while start < n:
            size = min(current, n - start)
            try:
                med = predict_fn(runner, x[start:start + size], self.horizon,
                                 self.device)
            except Exception as exc:  # noqa: BLE001 — selective OOM recovery
                if not self._is_cuda_oom(exc) or size <= 1:
                    raise
                del exc
                torch.cuda.empty_cache()
                current = max(1, last_success or size // 2)
                probing = False
                self._tuned_batch_sizes[int(width)] = current
                self._batch_search_complete.add(int(width))
                print(f"[{self.name}] dynamic batch W={width}: OOM at {size}; "
                      f"using {current}")
                continue
            medians.append(med)
            start += size
            if probing and size == current:
                last_success = current
                grown = min(limit, current * 2)
                if grown > current:
                    current = grown
                else:
                    probing = False
                    self._tuned_batch_sizes[int(width)] = current
                    self._batch_search_complete.add(int(width))
            elif not probing:
                self._tuned_batch_sizes[int(width)] = current

        if int(width) not in self._tuned_batch_sizes:
            # The call ended before the next probe fit; retain the largest
            # batch actually exercised rather than an untested candidate.
            self._tuned_batch_sizes[int(width)] = max(1, last_success or
                                                       min(initial_batch, n))
        tuned = self._tuned_batch_sizes[int(width)]
        if cached is None:
            print(f"[{self.name}] dynamic batch W={width}: tuned={tuned}, "
                  f"search_cap={self.max_batch_size}")
        return torch.cat(medians, dim=0)

    def forecast(self, contexts: np.ndarray) -> np.ndarray:
        import torch
        self.load()
        x = torch.from_numpy(
            np.ascontiguousarray(contexts, dtype=np.float32)).unsqueeze(-1)
        with torch.no_grad():
            width = int(contexts.shape[1])
            med = self._uniform_forecast(
                x, width, self.batch_size_for(width, len(contexts)))
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
            med = self._uniform_forecast(
                x, W, self.batch_size_for(W, len(contexts)))
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
            matched = [n for n, _ in bb.named_modules() if rx.search(n)]
            # A broad naming match can also find nested dimension-changing
            # stacks. PatchTST has encoder.blocks.N (the residual blocks we
            # want) and encoder.blocks.N.mlp.layers.K (MLP projections). Group
            # terminal numeric siblings by exact prefix and choose the largest,
            # shallowest coherent stack rather than mixing both kinds.
            groups: Dict[str, List[Tuple[int, str]]] = {}
            for name in matched:
                m = re.match(r"(.*?)(\d+)$", name)
                if m:
                    groups.setdefault(m.group(1), []).append(
                        (int(m.group(2)), name))
            coherent = {p: rows for p, rows in groups.items()
                        if len(rows) >= 2}
            if not coherent:
                raise CapabilityError(
                    f"{self.name}: no repeated-block modules matched "
                    f"{BLOCK_PATTERN!r} — dump modules on the server")
            _prefix, rows = min(
                coherent.items(),
                key=lambda item: (-len(item[1]), item[0].count("."),
                                  len(item[0]), item[0]))
            self._layer_names = [name for _idx, name in sorted(rows)]
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
        width = int(x_chunk.shape[1])
        with torch.no_grad():
            # Keep the configured capacity stable for a short final chunk.
            # TimesFM includes this value in its ForecastConfig, so using the
            # actual final-chunk length would needlessly rebuild the config.
            return self._uniform_forecast(
                x_chunk, width, self.batch_size_for(width))

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
            batch_size = self.batch_size_for(x.shape[1], x.shape[0])
            for start in range(0, x.shape[0], batch_size):
                fired.clear()
                self._forward_chunk(x[start:start + batch_size])
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
            batch_size = self.batch_size_for(x.shape[1], x.shape[0])
            for start in range(0, x.shape[0], batch_size):
                chunk = x[start:start + batch_size]
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
            batch_size = self.batch_size_for(x.shape[1], x.shape[0])
            for start in range(0, x.shape[0], batch_size):
                preds.append(
                    self._forward_chunk(
                        x[start:start + batch_size]).float().cpu())
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
