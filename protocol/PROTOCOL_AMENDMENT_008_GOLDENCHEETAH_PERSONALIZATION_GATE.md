# Protocol Amendment 008: GoldenCheetah Personalization Gate

Date: 2026-07-31

## Purpose

This amendment records the validation-only historical-personalization experiment on
the frozen 30-user, 600-session GoldenCheetah lightweight longitudinal subset. It
tests whether the negative wearable-exercise history result was caused only by sparse
prior training records. Calibration and temporal-test sessions remain sealed.

## Fixed configuration

- Cache: `data/processed/goldencheetah_transition_v1/outer_fold_0`
- Candidate: `pkssm_64x4_r6`
- Seed: `20260730`, set before model initialization
- Context and horizon: 300 and 120 seconds
- Training origins: 3,501
- Validation transition origins: 864
- Validation target sessions: 116
- History source: same-user `support_train` sessions ending strictly before the
  target session
- Selection metric: participant-macro validation transition MAE

## Results

| Model or history budget | Validation MAE (bpm) | Delta versus PK-SSM without history (bpm) |
|---|---:|---:|
| Persistence | 9.496768 | +0.815469 |
| PK-SSM, budget 0 | 8.681299 | reference |
| PK-SSM, budget 1 | 8.745317 | +0.064017 |
| PK-SSM, budget 3 | 8.721481 | +0.040182 |
| PK-SSM, budget 5 | 8.769479 | +0.088180 |
| PK-SSM, budget 10 | 8.708875 | +0.027576 |

Positive deltas indicate worse performance. All 116 validation target sessions had
the full requested history budget for budgets 1, 3, 5, and 10. Budget 10 trained for
55 epochs and selected epoch 40, so its lack of improvement is not attributable to
premature early stopping.

## Interpretation lock

The no-history PK-SSM improved over persistence by 0.815469 bpm, or approximately
8.59%. However, none of the four historical budgets improved over the matched
no-history PK-SSM. The smallest deterioration was 0.027576 bpm at budget 10, which is
both unfavorable and practically negligible.

Together with Protocol Amendment 007, this result shows that the current historical
personalization mechanism is unsupported in both sparse-history and longitudinally
deep settings. No additional history-encoder redesign or hyperparameter search is
permitted.

## Manuscript consequence

1. The no-history PK-SSM remains eligible for locked user-shift, activity-shift,
   uncertainty, efficiency, and external-validity evaluation.
2. Historical personalization must be reported as a negative ablation and deployment
   boundary, not as a performance contribution.
3. The current working title `Deployment-Aware Evaluation of Physiology-Guided Heart-Rate Transition Forecasting under User and Activity Shifts` is not
   supported as a positive-method claim.
4. A title and framing decision must be made before manuscript finalization. A
   defensible positive-method title is `Heart-Rate Transition Forecasting with a
   Physiology-Guided Kinetic State-Space Model under User and Activity Shifts`.
5. If historical personalization remains in the title, it must be framed explicitly
   as a limitation or boundary study rather than an achieved improvement.

No calibration or test metric was accessed for this decision.
