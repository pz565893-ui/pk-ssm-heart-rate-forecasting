# Protocol Amendment 011: Complete Wearable v4 Baseline Validation

Date: 2026-07-31

## Purpose

This amendment records the complete fixed-configuration baseline comparison on the five participant-disjoint Wearable v4 validation folds. It supersedes any provisional interpretation based only on persistence and Ridge comparisons.

Calibration and test roles remained sealed. No result below was used to access, inspect, or modify a test-set outcome.

## Frozen comparison boundary

- Cache: `data/processed/wearable_exercise_transition_v2`
- Roles accessed: train and validation only
- Outer folds: 0 through 4
- Seed: `20260730`
- Context: 300 seconds
- Forecast horizon: 120 seconds
- Primary metric: participant-macro transition trajectory MAE in bpm
- Activity identity: available
- Historical personalization: disabled
- PK-SSM candidate in this comparison: `pkssm_64x4_r6`
- Neural-baseline capacity setting: the corresponding fixed 64-unit configuration
- Ridge sensitivity clipping: `[30, 220]` bpm

## Five-fold validation results

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| TCN | 9.3261 | 9.3657 | 9.0453 | 7.6703 | 7.7973 | 8.6409 | 119,848 |
| PK-SSM, no history | 9.3715 | 9.9294 | 9.2187 | 7.8798 | 8.1963 | 8.9191 | 205,255 |
| Transformer | 9.5722 | 9.8714 | 10.4223 | 8.0688 | 7.6278 | 9.1125 | 152,296 |
| First-order kinetics | 9.8726 | 10.3317 | 9.6941 | 7.8324 | 8.4878 | 9.2437 | 193,805 |
| GRU | 10.0847 | 10.3911 | 9.2503 | 8.0721 | 8.6423 | 9.2881 | 68,776 |
| LSTM | 10.3462 | 10.7166 | 9.8942 | 7.6772 | 8.6239 | 9.4516 | 82,472 |
| Unconstrained residual SSM | 10.6615 | 10.3813 | 10.3983 | 8.2458 | 8.7083 | 9.6790 | 105,923 |
| Last-value persistence | 11.1912 | 11.2845 | 10.7958 | 8.3997 | 9.2178 | 10.1778 | 0 |
| Damped trend | 11.5145 | 12.9783 | 11.0461 | 10.3691 | 10.3107 | 11.2437 | 0 |
| Ridge, clipped sensitivity | 11.4604 | 11.0180 | 10.4891 | 14.8777 | 8.6149 | 11.2920 | 8,880 |
| Ridge, raw | 11.4604 | 11.0180 | 10.4891 | 255.5735 | 8.6149 | 59.4312 | 8,880 |

All errors are validation participant-macro transition MAE in bpm. Lower is better. Values are rounded to four decimal places in the table; full-precision values remain in the immutable selection reports.

## Locked interpretation

1. TCN was the strongest fixed-configuration model by mean MAE. It outperformed the current PK-SSM candidate in all five folds by a mean of 0.2782 bpm, equivalent to 3.12% relative to the PK-SSM mean.
2. PK-SSM outperformed GRU, unconstrained residual SSM, persistence, damped trend, and both Ridge variants in all five folds.
3. PK-SSM outperformed LSTM in four of five folds and first-order kinetics in four of five folds.
4. PK-SSM had a lower mean MAE than Transformer, but won only three of five folds head to head.
5. The comparison supports a contribution of constrained physiological state dynamics relative to the matched unconstrained residual SSM. The mean improvement was 0.7599 bpm.
6. The comparison provides only limited support for the dual-timescale formulation relative to first-order kinetics. The mean improvement was 0.3246 bpm, with one fold favoring first-order kinetics.
7. These results do not support a claim that PK-SSM is the most accurate general predictor. They also do not support a benefit from historical personalization.

## Pre-test consequence

The sealed test sets must remain unopened until all of the following are complete:

- Evaluate every prespecified PK-SSM architecture candidate on validation data.
- Freeze one architecture-selection rule without reference to test data.
- Evaluate random-seed stability for the selected architecture.
- Freeze the final baseline and uncertainty-calibration analysis code.
- Record the final test-opening decision in a new protocol amendment.

The working manuscript title may retain the word `Personalized` only if the manuscript clearly treats historical personalization as a tested mechanism with a negative or boundary-dependent result. It must not imply that personalization improved forecast accuracy.

## Reporting requirements

- Report the full five-fold table and identify TCN as the strongest fixed-configuration validation model.
- Separate architecture-selection results from sealed test results.
- Retain the raw Ridge extrapolation failure and the clipped sensitivity result.
- Report fold-level values, parameter counts, and inference cost.
- Do not select a model, seed, calibration rule, subgroup definition, or title claim using sealed test outcomes.
