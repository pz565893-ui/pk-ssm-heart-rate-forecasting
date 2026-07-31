# Protocol Amendment 009: Validation Baseline and Deployment Boundary

Date: 2026-07-31

## Scope

This amendment records validation-only comparator results after repairing model
initialization seeding. No calibration or test outcomes were accessed. All neural
models use seed `20260730`, the same 300-second context, 120-second trajectory target,
training origins, validation transition events, and participant-macro MAE.

## GoldenCheetah same-user temporal validation

| Model | Parameters | Validation MAE (bpm) | Delta versus PK-SSM (bpm) |
|---|---:|---:|---:|
| Ridge context summary | 8,880 | 8.571385 | -0.109914 |
| PK-SSM without history | 205,255 | 8.681299 | reference |
| LSTM | 82,472 | 8.881693 | +0.200394 |
| GRU | 68,776 | 8.885823 | +0.204523 |
| TCN | 119,848 | 8.909232 | +0.227933 |
| Transformer | 152,296 | 8.926787 | +0.245488 |
| Unconstrained residual SSM | 105,923 | 8.948472 | +0.267173 |
| First-order kinetics | 193,805 | 9.001969 | +0.320670 |
| Persistence | 2 | 9.496768 | +0.815469 |
| Damped trend | 2 | 14.136579 | +5.455280 |

Negative deltas indicate better performance than PK-SSM. Ridge therefore achieved
the best same-user temporal validation MAE, outperforming PK-SSM by 0.109914 bpm, or
approximately 1.27%. PK-SSM nevertheless outperformed every tested deep-learning
baseline. Its gain over the best deep comparator, LSTM, was 0.200394 bpm, or
approximately 2.26%.

The kinetic ablations support the physiology-guided component: PK-SSM improved over
first-order kinetics by 0.320670 bpm and over the unconstrained residual SSM by
0.267173 bpm.

## Wearable-exercise v4 unseen-user validation, outer fold 0

| Model | Validation MAE (bpm) |
|---|---:|
| PK-SSM without history | 9.371543 |
| Persistence | 11.191219 |
| Ridge context summary | 11.460440 |

On this participant-disjoint validation boundary, PK-SSM improved over persistence
by 1.819676 bpm, or approximately 16.26%, and over Ridge by 2.088896 bpm, or
approximately 18.23%. This fold contains only four validation participants, so the
result is a model-selection signal rather than final inferential evidence. It must be
expanded across the frozen outer folds before test access.

## Locked interpretation

1. PK-SSM must not be described as universally most accurate.
2. Ridge is the strongest same-user temporal comparator currently observed.
3. PK-SSM is the strongest tested model on the initial unseen-user validation fold.
4. The central confirmatory hypothesis becomes deployment-boundary robustness:
   physiology-guided dynamics may trade a small amount of same-user temporal accuracy
   for substantially better transfer to unseen users and activities.
5. Historical personalization remains a negative ablation and cannot be presented as
   a performance contribution.
6. Final claims require all frozen folds, uncertainty intervals, and sealed test
   evaluation; the current table is validation-only.

## Title implication

A framing that matches all current evidence is:

`Physiology-Guided Heart-Rate Transition Forecasting under User and Activity Shifts:
Accuracy, Robustness, and the Limits of Historical Personalization`

This title retains personalization as an evaluated boundary, not as an unsupported
positive claim.
