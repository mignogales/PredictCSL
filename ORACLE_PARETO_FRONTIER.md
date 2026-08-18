# GiftEval oracle MASE/FLOPs Pareto frontier

Updated: 2026-08-10

## Scope

This analysis uses Chronos2-Small on the official 97-cell GiftEval cohort. The
oracle may choose one dataset-shared context action per dataset/term cell. MASE
is the leaderboard-faithful `mase_gluonts_real`, normalized per cell by the
published Seasonal-Naive reference and aggregated by geometric mean. FLOPs use
the same theoretical model and native-context accounting as Stage 4. Two FLOPs
aggregations are reported:

- **cell-balanced:** one per-forecast cost per cell, aligned with the unweighted
  97-cell MASE aggregation and the Stage-4 comparison CSV;
- **benchmark-workload:** per-forecast cost multiplied by the number of series
  in each cell, answering how much total inference work the benchmark uses.

The curve is the exact **supported** Pareto frontier: every point is optimal for
some non-negative global penalty on total FLOPs. Because the action space is
discrete, unsupported nondominated combinations can exist between these points;
they are deliberately not presented as exact solutions to a linear deployment
objective.

## Main result

The unconstrained accuracy oracle is already cheaper as well as more accurate
than full context:

| Oracle constraint | Normalized MASE | FLOPs saved vs full |
|---|---:|---:|
| Full/native baseline | 0.726708 | 0.00% |
| Minimum-MASE oracle | **0.704681** | 53.88% |
| Within 0.1% of minimum-MASE oracle | 0.705385 | 65.12% |
| Within 0.5% of minimum-MASE oracle | 0.708185 | 74.89% |
| Within 1% of minimum-MASE oracle | 0.711619 | 79.25% |
| Within 2% of minimum-MASE oracle | 0.717344 | 83.19% |
| No worse than full/native | 0.723151 | **85.48%** |
| Within 5% of minimum-MASE oracle | 0.739514 | 89.30% |
| Minimum-compute supported endpoint | 1.046050 | 94.89% |

The attractive part of the curve is broad: accuracy changes very little between
roughly 54% and 75% FLOPs saved, remains better than full context through 85.5%,
and then deteriorates sharply. This places the practical oracle knee around
75–85% savings.

### Benchmark-workload sensitivity

Large GiftEval cells benefit especially strongly from shorter contexts. When
FLOPs are weighted by the number of series in each cell, the same MASE policies
have still larger savings:

| Oracle constraint | Normalized MASE | Workload FLOPs saved |
|---|---:|---:|
| Minimum-MASE oracle | **0.704681** | 62.21% |
| Within 0.1% of minimum-MASE oracle | 0.705384 | 76.22% |
| Within 0.5% of minimum-MASE oracle | 0.707920 | 83.69% |
| Within 1% of minimum-MASE oracle | 0.711553 | 88.73% |
| Within 2% of minimum-MASE oracle | 0.718758 | 92.50% |
| No worse than full/native | 0.726391 | **93.99%** |
| Minimum-compute supported endpoint | 1.046050 | 96.59% |

Thus the qualitative conclusion is robust to the compute weighting, and is even
stronger for actual benchmark throughput. The cell-balanced figures should be
used when comparing directly with the existing strategy-comparison CSV; the
workload-weighted figures should be used for total serving-cost claims.

Budget-oriented readings of the same curve:

| Required FLOPs saving | Best supported normalized MASE at or above budget |
|---:|---:|
| 55% | 0.704688 |
| 60% | 0.704909 |
| 65% | 0.705385 |
| 70% | 0.706415 |
| 75% | 0.708360 |
| 80% | 0.712441 |
| 85% | 0.721724 |
| 90% | 0.744920 |
| 92% | 0.771050 |
| 94% | 0.872407 |

## Gap to learned selectors

The current learned policies are well above the oracle frontier:

| Selector | Normalized MASE | FLOPs saved | Oracle MASE at at least that saving | Relative gap |
|---|---:|---:|---:|---:|
| Mamba curve | 0.724344 | 56.49% | 0.704723 | 2.78% |
| Soft top-k classification | 0.722412 | 27.89% | 0.704681 at 53.88% saved | 2.52% |
| Adjacent pairwise | 0.725725 | 54.24% | 0.704682 | 2.99% |
| 3% acceptable set | 0.723630 | 26.39% | 0.704681 at 53.88% saved | 2.69% |

This is oracle headroom, not an attainable zero-shot estimate: the frontier uses
GiftEval outcome labels to choose each cell action. Its value is diagnostic. It
shows that the context grid contains policies with much better accuracy/compute
tradeoffs and that CSL prediction—not the candidate grid—is the current
bottleneck.

## Artifacts

- `logs/experiments/master_recompute/oracle_pareto_frontier/Chronos2-Small/oracle_pareto_frontier.png`
- `logs/experiments/master_recompute/oracle_pareto_frontier/Chronos2-Small/oracle_supported_frontier.csv`
- `logs/experiments/master_recompute/oracle_pareto_frontier/Chronos2-Small/oracle_cell_actions.csv`
- `logs/experiments/master_recompute/oracle_pareto_frontier/Chronos2-Small/key_operating_points.csv`
- `logs/experiments/master_recompute/oracle_pareto_frontier/Chronos2-Small/reference_points.csv`
- `logs/experiments/master_recompute/oracle_pareto_frontier/Chronos2-Small/report.json`
- `logs/experiments/master_recompute/oracle_pareto_frontier/Chronos2-Small_instance_weighted/`

The reusable analysis is `experiments/oracle_pareto_frontier_gifteval.py`.

The forecast-instance/series-wise extension, including the Mamba series-wise
application, is documented in `SERIESWISE_ORACLE_PARETO_FRONTIER.md`.
