# Synthetic Dataset Composition Analysis

## Current Configuration

### Periodicity Generation (Lines 181-203)
```python
n_periodic = int(rng.randint(1, 4))  # 1-3 periodic components per segment
log_lo, log_hi = math.log(PERIOD_MIN), math.log(PERIOD_MAX)
periods = np.exp(rng.uniform(log_lo, log_hi, size=n_periodic))
amplitudes = rng.uniform(0.5, 2.0, size=n_periodic)
```
- **Frequency**: Every series gets 1-3 sinusoidal/sawtooth/square waves
- **Amplitude**: 0.5-2.0 per component
- **Total periodicity contribution**: Usually **dominant** in the segment variance

### AR Component (Lines 205-211)
```python
if rng.uniform() < 0.5:                              # Only 50% adoption
    p = int(rng.randint(1, 4))                       # AR(1) to AR(3)
    coeffs = rng.uniform(-0.3 / p, 0.3 / p, size=p) # Small coefficients!
    innov = rng.normal(0.0, 0.3, size=length)       # Small innovation
    a = np.concatenate([[1.0], -coeffs])
    seg += lfilter([1.0], a, innov).astype(np.float32)
```

**Problem Analysis:**

| Factor | Current | Issue |
|--------|---------|-------|
| **Adoption rate** | 50% | Half the series have no AR at all |
| **Max coefficient** | ±0.1 (p=1) | Very weak persistence; AR(1)=0.1 ≈ white noise |
| **Innovation scale** | σ=0.3 | Small relative to periodic amplitudes (0.5-2.0) |
| **Order diversity** | 1-3 | Good, but weak coefficients waste potential |

### Relative Magnitudes
For a typical segment with 2 periodic components:
- **Periodicity contribution**: `(2 × amp) ≈ 1.0-4.0` (high variance)
- **AR contribution**: Innovation noise with σ=0.3 filtered by weak coefficients ≈ 0.3-0.5 (low variance)
- **Trend**: Weak polynomial (±0.5 coefficients)
- **Noise**: σ=0.05-0.30

**Result**: Periodicity easily dominates the spectral structure.

---

## Diagnosis: Your Intuition is Correct ✓

The synthetic dataset is **periodicity-heavy** with **weak AR**. This happens because:

1. **Periodicity is deterministic**: Guaranteed ≥1 sinusoid, always present, often 2-3
2. **AR is optional AND weak**: Only 50% of the time, and when present, coefficients are tiny
3. **Energy budget**: Periodic amplitudes (0.5-2.0) >> AR filtered noise (≈0.3)

---

## Recommendations to Increase AR Strength

### Option 1: Simple (Minimal Changes)
```python
# Line 206: Increase adoption rate
if rng.uniform() < 0.75:  # Was 0.5 → Now 75%

# Line 208: Larger coefficients
coeffs = rng.uniform(-0.5 / p, 0.5 / p, size=p)  # Was -0.3/p to 0.3/p

# Line 209: Larger innovation
innov = rng.normal(0.0, 0.6, size=length)  # Was 0.3
```
**Effect**: AR component becomes ~2x stronger, more series have AR.

### Option 2: Medium (More Control)
```python
# Add AR adoption rate as a parameter
AR_ADOPTION_RATE = 0.75  # Line ~102
AR_COEF_SCALE = 0.5      # Coefficient magnitude
AR_INNOV_STD = 0.6       # Innovation scale

if rng.uniform() < AR_ADOPTION_RATE:
    p = int(rng.randint(1, 4))
    coeffs = rng.uniform(-AR_COEF_SCALE / p, AR_COEF_SCALE / p, size=p)
    innov = rng.normal(0.0, AR_INNOV_STD, size=length).astype(np.float32)
    ...
```
**Benefit**: Easy to ablate AR strength later.

### Option 3: Advanced (Stochastic Coupling)
Consider making AR adoption **inversely related** to periodicity strength:
```python
# If we have many periodic components, reduce AR adoption
# If we have weak periodicity, increase AR adoption
# This creates more diverse behavior

n_periodic = int(rng.randint(1, 4))
ar_prob = 0.5 + 0.2 * (1 - min(n_periodic / 3.0, 1.0))  # More AR ↔ fewer periodic
```
**Benefit**: Balances the two mechanisms across the dataset.

---

## Expected Impact (Rough Estimates)

If you increase AR adoption from 50% → 75% + coefficient scale 0.3 → 0.5:

- **AR(1) mean**: ~0.1 → ~0.25-0.35 (stronger persistence)
- **Spectral slope**: Flatter high-freq spectrum (redder spectrum)
- **ACF decay**: Slower decay (higher lag-1 to lag-10 correlation)
- **Peak ratio** (periodicity): Unchanged (periodic components are independent of AR strength)
- **Series diversity**: More variance in temporal structure (good for TSFM generalization)

---

## Testing Strategy

1. **Small ablation**: Run just one model with modified parameters to see effect
   ```bash
   python build_context_length_dataset.py --n-series 1000 --model-idx 0
   ```

2. **Analyze new series** with updated `analyze_synth_composition.py`

3. **Compare predictor performance**: Does the predictor learn better with balanced AR/periodicity?

4. **Check error curves**: Look for differences in `curves_mae.npy` shapes
