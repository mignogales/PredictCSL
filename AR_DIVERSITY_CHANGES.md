# AR Diversity Update

## Problem
The synthetic dataset was AR-poor not just in strength, but in **variety**. All AR processes were sampled from a tiny range (±0.3/p), giving little behavioral diversity across the dataset.

## Solution
Added `_sample_ar_coefficients()` function that samples AR roots directly, ensuring:

1. **Much wider coefficient ranges** per order:
   - **AR(1)**: φ ∈ [-0.7, 0.95] (was [-0.3, 0.3])
   - **AR(2)**: Complex pairs OR real roots (was limited uniform)
   - **AR(3)**: Diverse multi-root combinations (was limited uniform)

2. **Different AR dynamics** represented in the dataset:
   - **Persistent processes** (φ=0.8-0.95): slow mean reversion
   - **Weakly dependent** (φ=0.0-0.3): quick reversion
   - **Negative autocorrelation** (φ=-0.7-0.0): oscillatory damping
   - **Oscillatory AR(2)**: Quasi-periodic with different frequencies
   - **Complex AR(3)**: Multi-scale temporal structure

3. **Importance sampling** by order:
   - AR(1): 50% (most interpretable, captures persistence well)
   - AR(2): 35% (oscillatory behavior, mixed real roots)
   - AR(3): 15% (complex dynamics, less common)

## Code Changes

### `build_context_length_dataset.py`

**Added function** (lines 165-232):
```python
def _sample_ar_coefficients(rng: np.random.RandomState) -> Optional[np.ndarray]:
    """Sample AR(p) coefficients with diverse behavior via root sampling."""
    # 50% adoption rate (unchanged)
    if rng.uniform() < 0.5:
        return None

    # Choose order with different probabilities
    p = int(rng.choice([1, 2, 3], p=[0.50, 0.35, 0.15]))

    # For each order, sample roots widely and convert to AR coefficients
    # (ensures stability by construction)
    ...
```

**Updated segment generation** (old lines 280-283):
```python
# OLD:
if rng.uniform() < 0.5:
    p = int(rng.randint(1, 4))
    coeffs = rng.uniform(-0.3 / p, 0.3 / p, size=p)  # Tiny range!
    innov = rng.normal(0.0, 0.3, size=length)
    ...

# NEW:
coeffs = _sample_ar_coefficients(rng)
if coeffs is not None:
    innov = rng.normal(0.0, 0.5, size=length)  # Slightly larger innovation
    ...
```

## Expected Impact

### Spectrum of AR behaviors across dataset:
- **~25% series**: No AR (periodic + noise only)
- **~25% series**: Weak AR (φ ≈ 0.1-0.3, quick reversion)
- **~25% series**: Persistent AR (φ ≈ 0.6-0.9, slow mean reversion)
- **~15% series**: Oscillatory AR (φ ≈ -0.5-0.0, damped cycles)
- **~10% series**: Complex AR(2,3) with multi-scale dynamics

### On the predictor:
- **More temporal variety** → harder learning problem
- **Wider range of useful context lengths** → more diverse labels
- **Better generalization** to real TS foundation models (which encounter all these patterns)
- **More discriminative features** for the context-length predictor

## Testing

Run a small ablation to verify:
```bash
# Generate 5k series with new AR sampler and one model
python experiments/build_context_length_dataset.py \
    --n-series 5000 --model-idx 0 --device cuda
```

Then compare `curves_mae.npy` shape and statistics to the old version.

## Rollback

If needed, revert the AR generation by commenting out the new function and restoring:
```python
if rng.uniform() < 0.5:
    p = int(rng.randint(1, 4))
    coeffs = rng.uniform(-0.3 / p, 0.3 / p, size=p)
    innov = rng.normal(0.0, 0.3, size=length).astype(np.float32)
    a = np.concatenate([[1.0], -coeffs])
    seg += lfilter([1.0], a, innov).astype(np.float32)
```
