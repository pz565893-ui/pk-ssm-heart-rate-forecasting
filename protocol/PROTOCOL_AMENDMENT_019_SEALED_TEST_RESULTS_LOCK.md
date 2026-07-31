# Protocol Amendment 019: Sealed Test Results and Claim Lock

Date: 2026-07-31

## Purpose

This amendment records the first locked aggregate results from the Wearable v4 sealed test roles. All model, seed, calibration, origin-policy, subgroup, and summary-code decisions were frozen before aggregate test inspection.

## Primary transition result

The primary analysis used tagged transition events, 31 held-out participants, and symmetric five-seed ensembles.

| Model | Trajectory MAE | RMSE | 30 s MAE | 60 s MAE | 120 s MAE |
|---|---:|---:|---:|---:|---:|
| PK-SSM | 9.9942 | 14.7100 | 5.7779 | 10.9124 | 16.4700 |
| TCN | 9.9490 | 14.8255 | 5.5534 | 10.7434 | 16.6918 |

All values are in bpm.

The participant-level paired trajectory-MAE difference, defined as PK-SSM minus TCN, was 0.0452 bpm. The prespecified 10,000-replicate participant bootstrap 95% interval was `[-0.2233, 0.3095]`, and the Wilcoxon signed-rank sensitivity p value was 0.7646. PK-SSM had lower participant MAE for 51.6% of held-out participants, compared with 48.4% for TCN.

This is a null practical and statistical result. PK-SSM and TCN must be described as having comparable primary transition accuracy.

## Horizon-specific paired effects

Positive differences favor TCN; negative differences favor PK-SSM.

| Horizon | PK-SSM minus TCN | Participant-bootstrap 95% interval | Wilcoxon p |
|---:|---:|---:|---:|
| 30 s | +0.2246 | `[0.0457, 0.3998]` | 0.0274 |
| 60 s | +0.1690 | `[-0.1444, 0.4741]` | 0.2094 |
| 120 s | -0.2218 | `[-0.7960, 0.3404]` | 0.6083 |

TCN had a small supported advantage at 30 seconds. Neither model had a supported advantage at 60 or 120 seconds.

## Schedule-wide secondary result

| Model | Trajectory MAE | 30 s MAE | 60 s MAE | 120 s MAE |
|---|---:|---:|---:|---:|
| PK-SSM | 8.7891 | 5.5381 | 9.5627 | 13.9294 |
| TCN | 8.4912 | 5.1810 | 9.2846 | 13.5784 |

The paired trajectory-MAE difference was +0.2979 bpm, with bootstrap 95% interval `[0.0808, 0.5013]` and Wilcoxon p = 0.0062. TCN was better for 74.2% of participants. This secondary result supports a modest TCN advantage outside the transition-only boundary.

## Seed stability on tagged transitions

| Seed | PK-SSM MAE | TCN MAE | Difference |
|---:|---:|---:|---:|
| 20260730 | 10.3347 | 10.1573 | +0.1775 |
| 20260731 | 10.1162 | 10.1905 | -0.0743 |
| 20260732 | 10.1855 | 10.3582 | -0.1727 |
| 20260733 | 10.2645 | 9.9854 | +0.2791 |
| 20260734 | 10.6407 | 9.9564 | +0.6842 |

TCN won three seed-level comparisons and PK-SSM won two. The mean seed-specific difference was +0.1788 bpm. A best-seed result must not be selected.

## Dynamic fidelity

| Model | Total-variation ratio | Rapid-change amplitude ratio |
|---|---:|---:|
| PK-SSM ensemble | 0.3634 | 0.1026 |
| TCN ensemble | 3.0181 | 0.1490 |

A value of one is ideal for each ratio. PK-SSM was strongly over-smoothed, while TCN accumulated excessive total variation. Both models reproduced only a small fraction of observed 10-second rapid-change amplitude. Therefore, the physiology-guided model cannot be described as dynamically faithful merely because it is smoother.

## High-HR performance

| Model | Fold-specific 90th-percentile high-HR MAE | Fixed 160 bpm MAE |
|---|---:|---:|
| PK-SSM | 20.1091 | 20.1256 |
| TCN | 19.8604 | 19.1758 |

Both errors are large. The study must not claim reliable high-intensity forecasting or clinical-grade performance.

## Conformal uncertainty on tagged transitions

### Origin-level split conformal

| Model | Coverage at 30 s | Width | Coverage at 60 s | Width | Coverage at 120 s | Width | Simultaneous curve coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| PK-SSM | 0.9432 | 47.58 | 0.9372 | 72.91 | 0.9424 | 87.93 | 0.9582 |
| TCN | 0.9517 | 44.10 | 0.9336 | 72.42 | 0.9257 | 88.31 | 0.9441 |

Widths are mean full interval widths in bpm. Coverage was near the nominal 95% target, but intervals became extremely wide at longer horizons. Calibration alone does not establish practical informativeness.

### Participant-block sensitivity

Participant-block intervals generally over-covered and were wider, reaching mean 120-second widths of 115.63 bpm for PK-SSM and 118.48 bpm for TCN. These intervals are dependence-aware sensitivity results, not evidence of useful precision.

## Fixed-seed secondary baseline ranking

The tagged-transition fixed-seed MAE ranking was:

| Model | MAE |
|---|---:|
| Transformer | 10.0723 |
| First-order kinetics | 10.1356 |
| TCN | 10.1573 |
| LSTM | 10.1954 |
| PK-SSM | 10.3347 |
| Unconstrained residual SSM | 10.4614 |
| GRU | 10.6565 |
| Persistence | 10.7768 |
| Ridge, clipped to `[30, 220]` | 11.1957 |
| Damped trend | 11.7680 |
| Ridge, raw | 65.1040 |

PK-SSM retained a small advantage over the matched unconstrained residual SSM but did not outperform first-order kinetics on the sealed test. The physiology-guided contribution is therefore boundary-dependent and modest.

## Efficiency

| Model | Parameters per member | Median origins per CPU second |
|---|---:|---:|
| PK-SSM | 205,255 | 1,162 |
| TCN | 119,848 | 1,853 |

TCN used approximately 41.6% fewer parameters and achieved approximately 1.59 times the measured throughput. The measured throughput includes locked export overhead and is not a pure kernel benchmark.

## Descriptive subgroup results

- Female participants: PK-SSM 10.8118 bpm, TCN 10.9130 bpm.
- Male participants: PK-SSM 9.4037 bpm, TCN 9.2528 bpm.
- Aerobic protocol: PK-SSM 7.3354 bpm, TCN 7.3671 bpm.
- Anaerobic protocol: PK-SSM 14.0938 bpm, TCN 13.8891 bpm.
- PK-SSM was descriptively better at sprint onset and aerobic stage boundaries.
- TCN was descriptively better at recovery onset and sprint offset.

These subgroup differences are small relative to absolute errors and must not be used to claim demographic or activity-specific superiority without participant-level uncertainty intervals.

## Locked manuscript claims

### Supported

- PK-SSM and TCN had comparable primary transition accuracy on held-out participants.
- TCN had a modest advantage on schedule-wide forecasting and at 30 seconds.
- PK-SSM imposed smoother, bounded dynamics but remained substantially over-smoothed.
- TCN produced excessive total variation despite competitive MAE.
- Split conformal calibration approached nominal coverage, but long-horizon intervals were very wide.
- Historical-record personalization provided no reliable validation benefit.
- Model rankings depended on deployment boundary, horizon, seed, and metric.

### Not supported

- PK-SSM is the most accurate model.
- Historical personalization improves forecasting.
- Physiology guidance guarantees realistic dynamics.
- The uncertainty intervals are practically narrow or clinically useful.
- High-HR forecasting is reliable.
- Learned kinetic parameters are laboratory-validated physiological measurements.
- The proposed training-load proxy equals true physiological load.

## Title consequence

The working title may remain:

`Deployment-Aware Evaluation of Physiology-Guided Heart-Rate Transition Forecasting under User and Activity Shifts`

only if `Personalized` is explicitly defined as context-conditioned individual kinetic inference and the failed historical-record pathway is presented transparently. The title must not be interpreted as a claim that historical personalization improved accuracy.
