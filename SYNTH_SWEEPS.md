# Synthetic single-factor sweeps — design notes

Companion to `experiments/synth_param_sweeps.py`. Each experiment generates
**minimal isolated series** (one generative factor + Gaussian noise, nothing
else) and evaluates every TSFM at context lengths on a **ratio grid**
(`context = r × parameter`, `r ∈ 0.25…16`), so the plots are MAE vs
**normalized context** (`context/parameter`) and curves from different
parameter values align on one scale-free axis. Series are z-scored by their
context-pool stats (the stage-1 convention) and seeded per (experiment, bin),
so **every model sees identical series**. Non-`horizon` experiments forecast
once at h=256 and slice the error at h ∈ {16, 64, 256}.

## The core seven

### 1. `period` — sinusoid period *T* → context/T
One sinusoid of period T ∈ {32 … 2048} plus noise (σ=0.2). The baseline
question: **how many cycles does a TSFM need to see** before extra context
stops helping? Expected shape: error falls until context/T reaches some small
k (the model has locked the phase/amplitude), then flat.

### 2. `seasonality` — composite pattern length *S* → context/S
A **structured repeating profile**, not a smooth wave: a random mix of 3–6
harmonics of the S-cycle, tiled with per-cycle amplitude jitter (a "weekly
profile"). Same normalizer as `period`, so the two plots overlay directly:
any rightward shift of the saturation knee = **complex repetition needs more
covered cycles than smooth repetition**.

### 3. `ar_order` — AR(p) at matched timescale τ → context/τ, curves per p
Raw order p ∈ {1..4} has no length scale (context/p is degenerate), so the
dominant pole is pinned to a target ACF timescale τ ∈ {64, 256, 1024}
(pole = 1−1/τ) and the extra poles stay short-memory (|pole| ≤ 0.7). One
curve per order on x = context/τ answers: **at equal memory length, does
higher-order dynamics need more context?**

### 4. `memory` — AR(1) timescale τ = 1/(1−φ) → context/τ
Long-memory *strength* operationalized as literal correlation reach:
τ ∈ {16 … 4096}. The x-axis asks whether the useful context is a fixed
multiple of the correlation time (curves collapsing onto one master curve
would mean "context worth ≈ c·τ, universally").

### 5. `delay` — trigger→response lag *d* → context/d
Random **trigger events** (Hann bumps, width 33 ≈ one patch — deliberately
NOT single-step spikes, which patch-based models can miss) are each followed
`d` steps later by a deterministic response bump (×0.7 amplitude). One
trigger is planted so its response lands **inside the horizon**: only a
context reaching back ≳ d can see the trigger and predict the response.
Sharpest possible **necessity** test — the information is literally absent
below the threshold, so the curve should be a cliff at context/d ≈ 1 for any
model able to exploit lagged structure at all.

### 6. `regime` — regime duration *D* → context/D
Regimes of fixed duration D (shared T0=32 sinusoid; level/amplitude/phase
shift at every boundary), with the last boundary exactly D before forecast
time and the current regime extending through the horizon. context/D ≤ 1
stays inside the current regime; ratios > 1 pull in progressively more
**stale regimes**. Expected: U-shape with minimum near ratio ≈ 1 — the
empirical signature the context-length predictor lives on. NOTE: age and
duration are confounded here by construction (age = D); `break_age`
deconfounds them.

### 7. `horizon` — forecast horizon *h* → context/h
Fixed canonical signal (sin T=128 + AR(1) τ=128 + noise); only h ∈
{16 … 512} varies, with contexts = r × h. Tests the folk rule "context
should be a multiple of horizon" in isolation from signal properties.

## The extension five

### 8. `break_age` — age *A* of a single change point → context/A
**The most on-thesis probe.** One change point (level/amplitude/phase shift)
at distance A ∈ {64 … 4096} before forecast time, preceded by a long
homogeneous regime; the post-break regime extends through the horizon.
Error should be **flat for context/A < 1** (window inside the current
regime) and **rise past 1** as pre-break history enters the window — a
direct measurement of how much each TSFM is hurt by crossing a break, i.e.
exactly the signal the zero-shot context-length predictor is supposed to
exploit. Complements `regime`: that asks "many stale regimes", this asks
"one break, how recent".

### 9. `snr` — seasonal strength at fixed period → context/T, curves per σ
Fixed T=256, noise σ ∈ {0.1 … 2.0} swept (curves per σ). Quantitative
hypothesis for free: estimating a periodic profile under noise σ needs
~σ² cycles of averaging, so the **saturation knee in context/T should shift
right as σ grows**. Turns the `period` result ("saturates at k cycles") into
a law ("k depends on noise like this"). Absolute MAE differs across σ bins
(z-scoring shrinks the signal share as σ grows) — read the knee position,
not the level; the overlay plot normalizes each bin by its own minimum.

### 10. `multiscale` — nested periods, outer cycle k·T → context/(k·T)
Inner sinusoid of period T=64 whose amplitude in inner cycle j is a **fixed
random value a[j mod k]** — an envelope sequence repeating every outer cycle
k·T, k ∈ {2 … 32} (daily-within-weekly structure). The next inner cycle's
amplitude is only predictable from the position-matched cycle **one outer
period back**, so the hypothesis is a knee at context/(k·T) ≈ 1: below it
the model tracks the inner wave but must guess the envelope. This is the one
structure the flat `period`/`seasonality` sweeps cannot reveal.

### 11. `period_drift` — frequency-staleness timescale *M* → context/M
The dominant period **wanders**: log-period follows an OU process with
correlation time M ∈ {256 … 4096} (stationary std 0.25 around T0=64), and
the wave's phase integrates the instantaneous frequency. Context older than
~M carries a **stale estimate of the current period** — the frequency
analog of `regime`/`break_age`'s level staleness. Does old-frequency
context actively hurt, or is it merely useless?

### 12. `missing_gap` — mean-filled gap length *G* → context/G *(appendix)*
Sinusoid (T0=64) with a missing span of length G ∈ {64 … 2048} occupying
[end−1.5G, end−0.5G) of the pool, zeroed in **raw** space before
standardization — the stage-1 left-pad / GiftEval NaN→0 model-input
convention (the gap maps to the constant −mu/σ). Scale-free geometry:
context/G < 0.5 sees only the genuine tail; 0.5–1.5 is inside the gap
(uninformative constant); > 1.5 **bridges** to pre-gap history. Directly
relevant to the `*_with_missing` GiftEval cells, BUT: since the wrappers
can't uniformly ingest NaN, this measures "uninformative constant span",
not native missing-value handling — keep it an appendix result, not a
headline.

## Skipped on purpose

- **Volatility clustering (GARCH)** — the pipeline scores *median* forecasts
  with MAE; volatility structure moves the predictive *uncertainty*, not the
  median, so the sweep would be flat by construction.

## Reading the outputs

- `logs/experiments/synth_param_sweeps/<Model>/<experiment>/results.npz`:
  `curves_mae`/`curves_mse` `(n_bins, n_series, n_ratios, n_h)`, `naive_mae`
  (last-value baseline), `contexts` (actual clamped lengths; −1 = skipped by
  a model's context cap), `ratios`, `norms`, `horizons`, `eval_horizons`.
- `plots/<experiment>/<Model>.png` — lines per bin, panels per eval horizon
  (`ar_order`: panel rows per τ, lines per order).
- `plots/<experiment>/all_models.png` — per model, each bin's mean curve
  divided by its own minimum, averaged over bins: a scale-free "how much
  does context/parameter matter" comparison.
- Caveats: TiRex NaN-pads everything to 2048 internally (flat compute);
  Sundial/TimeMoE/Toto/FlowState context caps truncate the right end of
  their curves; TimesFM recompiles per distinct width (slowest sweeps).
