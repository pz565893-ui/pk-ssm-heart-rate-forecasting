# Protocol Amendment 020: GoldenCheetah External Test Opening

Date: 2026-07-31

## Purpose

This amendment authorizes calibration and test evaluation on the frozen GoldenCheetah lightweight longitudinal benchmark. It is written before access to any GoldenCheetah calibration or test prediction.

## External-validity interpretation

GoldenCheetah is an independent real-world exercise dataset. The analysis evaluates replication of the same architecture, task definition, leakage controls, and evaluation code after training on GoldenCheetah training sessions. It is not a zero-shot transfer of Wearable model weights and must not be described as such.

## Frozen dataset boundary

- Participants: 30
- Sessions: 600, with 20 sessions per participant
- Train sessions: 300
- Validation sessions: 120
- Calibration sessions: 90
- Test sessions: 90
- Primary origins: causal effort-transition events
- Primary history rule: no historical personalization
- Historical-session budgets 1, 3, 5, and 10 failed the validation gate and will not be tested as positive candidates

## Frozen validation evidence

| Model | Validation participant-macro transition MAE, bpm |
|---|---:|
| Ridge summary | 8.5714 |
| PK-SSM, no history | 8.6813 |
| LSTM | 8.8817 |
| GRU | 8.8858 |
| TCN | 8.9092 |
| Transformer | 8.9268 |
| Unconstrained residual SSM | 8.9485 |
| First-order kinetics | 9.0020 |
| Persistence | 9.4968 |
| Damped trend | 14.1366 |

Ridge was the strongest validation model. PK-SSM must not be described as the validation winner.

## Locked external test scope

- Outer fold: 0
- Seed: `20260730`
- Models: PK-SSM, TCN, GRU, LSTM, Transformer, unconstrained residual SSM, first-order kinetics, persistence, and damped trend
- Primary origin policy: `tagged_events`
- Secondary origin policy: `evaluation_stride`
- Uncertainty calibration: PK-SSM and TCN only, using GoldenCheetah calibration sessions
- Activity analysis: report every frozen activity label with participant and origin counts
- Sex analysis: descriptive only because the lightweight subset is strongly imbalanced

The frozen Ridge evaluator requires a separate GoldenCheetah parameter path and will be run only if its summary-feature schema matches the Wearable evaluator without model refitting.

## Frozen code

The following previously hashed scripts will be reused without modification:

- `scripts/export_locked_forecasts.py`
- `scripts/score_calibrated_forecasts.py`

## Output roots

- Forecast bundles: `outputs/goldencheetah_locked_evaluation_v1`
- Calibrated reports: `outputs/goldencheetah_locked_scoring_v1`
- Summary: `outputs/goldencheetah_locked_summary_v1`

## Claims permitted by this analysis

- Replication or failure of model ranking on an independent real-world dataset
- Boundary-specific accuracy by activity label
- Calibration and interval-width transportability after dataset-specific calibration
- Failure or success of the physiology-guided constraints relative to matched ablations

## Claims not permitted

- Zero-shot cross-dataset generalization
- Clinical validity
- Laboratory-validated physiology
- Benefit from historical personalization
- Population-level sex conclusions from three female participants

The previously frozen test-opening token is required for test export. No GoldenCheetah test outcome may change the Wearable analysis or model definition.
