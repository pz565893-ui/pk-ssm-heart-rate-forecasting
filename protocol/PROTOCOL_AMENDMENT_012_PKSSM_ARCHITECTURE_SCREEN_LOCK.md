# Protocol Amendment 012: PK-SSM Architecture Screen Lock

Date: 2026-07-31

## Purpose

This amendment records the completion of the prespecified PK-SSM architecture screen on the five participant-disjoint Wearable v4 validation folds. The screen used the canonical seed `20260730`. Calibration and test roles remained sealed.

## Fixed screening conditions

- Cache: `data/processed/wearable_exercise_transition_v2`
- Roles accessed: train and validation only
- Outer folds: 0 through 4
- Canonical screening seed: `20260730`
- History regime: none
- Activity identity: available
- Primary screen metric: mean participant-macro transition trajectory MAE across five validation folds
- Secondary preference for near-ties: fewer parameters and lower computational cost

## Architecture-screen results

| Candidate | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pkssm_64x4_r6` | 9.3715 | 9.9294 | 9.2187 | 7.8798 | 8.1963 | 8.9191 | 205,255 |
| `pkssm_64x5_r6` | 9.7877 | 10.0300 | 9.2912 | 7.7225 | 7.9436 | 8.9550 | 246,599 |
| `pkssm_96x4_r6` | 9.5064 | 10.0908 | 9.2927 | 7.6786 | 8.2532 | 8.9644 | 455,271 |
| `pkssm_64x4_r8` | 9.4117 | 10.3985 | 9.1894 | 8.1119 | 7.8479 | 8.9919 | 205,255 |
| `pkssm_48x4_r4` | 9.4780 | 9.9586 | 10.1026 | 8.2018 | 8.4016 | 9.2285 | 117,111 |

All values are validation participant-macro transition MAE in bpm. Lower is better.

## Locked screening decision

`pkssm_64x4_r6` is retained as the provisional PK-SSM architecture because it achieved the lowest five-fold mean MAE. Its advantage over the next candidate was only 0.0359 bpm, so this screen does not establish a practically important architecture difference.

The architecture definition is now frozen. No residual bound, hidden width, dilation schedule, dropout rate, or history pathway may be changed after any calibration or test role is opened.

## Required seed-stability stage

Before the architecture is called final, `pkssm_64x4_r6` must be fitted on all five validation folds with the remaining prespecified seeds:

- `20260731`
- `20260732`
- `20260733`
- `20260734`

The report must include fold-by-seed MAE, the mean and standard deviation across seeds, and the frequency with which PK-SSM outperforms the fixed TCN reference. The seed analysis is a stability assessment, not a new architecture-selection opportunity.

If seed variability is large enough to erase the fixed-seed conclusions, the manuscript must report that instability. It must not choose the best seed for sealed test evaluation.

## Test boundary

Calibration and test roles remain sealed. The final test-opening amendment must specify whether predictions are obtained from one prespecified seed, a seed ensemble, or a deterministic refit. That choice must be made before any test predictions are generated.
