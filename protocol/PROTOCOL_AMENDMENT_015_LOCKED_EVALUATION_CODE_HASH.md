# Protocol Amendment 015: Locked Evaluation Code and Dry-Run Hashes

Date: 2026-07-31

## Purpose

This amendment freezes the final forecast-export and conformal-scoring implementation before sealed test access. The implementation was dry-run only on validation and calibration roles from outer fold 0. No test target or prediction was accessed.

## Frozen code hashes

| File | SHA256 |
|---|---|
| `scripts/export_locked_forecasts.py` | `2912F0EF5E931C7A6EF1162E0CC3F62213166ACA6844B7172D04E919ADE89B60` |
| `scripts/score_calibrated_forecasts.py` | `F535253EB22D502B35C904768BD2FF6F32807B71689FCDB6489EC210B6F166CF` |

Any later change to either file requires a new amendment, a new hash, and preservation of all original outputs. Test results cannot justify a code change.

## Dry-run inputs

- Outer fold: 0
- Seed: `20260730`
- Models: PK-SSM and TCN
- Primary origin policy: `tagged_events`
- Validation origins: 63 from 4 participants
- Calibration origins: 70 from 5 participants
- Test role: not accessed

## Reproduction check

The locked exporter reproduced the original participant-macro validation trajectory MAE to floating-point tolerance:

| Model | Selection report MAE | Locked exporter MAE |
|---|---:|---:|
| PK-SSM | 9.3715434 | 9.3715430 |
| TCN | 9.3261343 | 9.3261348 |

## Dry-run dynamic fidelity

| Model | Participant-macro total-variation ratio |
|---|---:|
| PK-SSM | 0.4441 |
| TCN | 6.3181 |

The observed target is the denominator, so a value of one indicates matched total variation. PK-SSM was strongly over-smoothed, whereas TCN was excessively variable. This metric is descriptive and must be interpreted together with trajectory examples and event-specific summaries.

## Dry-run uncertainty results

### Raw Student-t point coverage

| Model | Calibration coverage | Validation coverage |
|---|---:|---:|
| PK-SSM | 0.7559 | 0.8658 |
| TCN | 0.9522 | 0.9553 |

PK-SSM raw uncertainty was materially under-covered and cannot be described as calibrated without conformal correction.

### Origin-level conformal validation coverage

| Model | 30 s | 60 s | 120 s | Simultaneous 120 s curve |
|---|---:|---:|---:|---:|
| PK-SSM | 0.9722 | 0.9861 | 0.9514 | 0.9583 |
| TCN | 0.9722 | 0.9306 | 0.9514 | 0.9444 |

These are dry-run values on validation data and are not confirmatory test results.

## Dry-run report hashes

| Report | SHA256 |
|---|---|
| PK-SSM fold-0 dry-run calibrated report | `AA6642B17C28D4079119FCBBD9526126ADF5505FDE0A8D617A2CEFA0B58025C4` |
| TCN fold-0 dry-run calibrated report | `6E0635573CC861CECD2815C665B2C299D8A2270773867A9C14D0707535FB046E` |

## Ensemble interval interpretation

For a single seed, the raw Student-t interval uses the model's exact predicted distribution. For a multi-seed ensemble, the exporter combines within-seed scale and between-seed variation by moment matching. The resulting raw Student-t diagnostic is only an approximation and is excluded from the primary ensemble uncertainty claim. Primary ensemble uncertainty uses calibration-only split conformal intervals.

## Test-opening status

The code dry-run and hashing requirements in Amendment 014 are satisfied. Final calibration bundles must be exported with the frozen scripts before each corresponding test bundle is scored. The explicit test-opening token remains required by the exporter.
