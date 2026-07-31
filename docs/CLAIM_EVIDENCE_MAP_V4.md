# Claim-evidence map v4

## One-sentence argument

In short-horizon exercise heart-rate transition forecasting, PK-SSM provides a bounded fast-slow representation whose effects on accuracy, dynamic fidelity, calibration, and transportability are tested under leakage-controlled user and activity boundaries, but locked evidence does not support accuracy superiority, longitudinal-history benefit, or physiological phenotyping.

## Major claims

| Claim | Decisive evidence | Status | Allowed wording |
|---|---|---|---|
| Participant-level leakage is controlled in the v4 user-shift experiment. | Five immutable participant folds; disjoint train, validation, calibration, and test roles; six passed leakage checks. | Supported | "Participant-disjoint" and "leakage-controlled" within the audited split scope. |
| PK-SSM outperforms TCN for held-out users. | Tagged-transition paired difference 0.0452 beats/min, 95% CI -0.2233 to 0.3095, p = 0.7646. | Contradicted | State that no supported difference was detected. |
| TCN has a short-horizon and schedule-wide advantage. | 30-s difference 0.2246, 95% CI 0.0457 to 0.3998; schedule-wide difference 0.2979, 95% CI 0.0808 to 0.5013. | Supported within stated boundaries | "TCN had lower error at 30 s and on schedule-wide origins." |
| Prior-session history improves personalization. | No Wearable history budget passed validation; GoldenCheetah budgets 1, 3, 5, and 10 were worse than no history. | Contradicted | Historical pathway failed validation and was disabled. |
| PK-SSM better reproduces rapid heart-rate transitions. | Wearable total-variation ratio 0.3634 and rapid-change amplitude ratio 0.1026; TCN ratios 3.0181 and 0.1490. | Contradicted for fidelity | PK-SSM is bounded and smoother, but markedly over-smoothed. |
| Conformal intervals are calibrated and operationally informative. | Near-nominal marginal coverage; Wearable 120-s widths about 88 beats/min and participant-block widths above 115 beats/min. | Calibration partly supported; informativeness contradicted | Report coverage and width together; call intervals broad. |
| PK-SSM is reliable at high heart rates. | Fold-specific high-HR MAE 20.1091 and fixed-160 MAE 20.1256 beats/min. | Contradicted | Do not make high-intensity or clinical-grade claims. |
| Latent kinetic variables are physiological traits. | Low participant ICC for most variables, near-zero seed rank agreement for time constants, and high within-origin seed variability. | Contradicted | Treat them as latent regularization variables only. |
| PK-SSM generalizes better than TCN on GoldenCheetah. | PK-SSM minus TCN -0.0943, 95% CI -0.3451 to 0.1563; Ridge was strongest. | Not supported | State that rankings were dataset-dependent and PK-SSM/TCN did not differ reliably. |
| GoldenCheetah tests unseen sports. | All test sports appeared during fitting. | Contradicted | Report bike/run as within-dataset strata only. |
| The internal index measures true physiological training load. | No laboratory oxygen-uptake, lactate, or criterion-load measurement is available. | Contradicted | Use "training-load proxy" only. |

## Submission-critical missing evidence

- Accepted Amendment 026 activity-shift cache with source-session membership enforced.
- Five-seed PK-SSM and TCN results for both shift directions and all three deployment boundaries.
- Source-only conformal coverage and interval width after activity transfer.
- Direction-specific dynamic-fidelity analysis.
- Confirmed city, postal code, and department for the second affiliation.

| Strict activity-shift trajectory superiority | Primary tagged-events joint-boundary paired inference | Not supported in either direction | Do not claim general superiority |
| Direction-dependent deployment contrast | Joint minus source-user paired contrasts | Supported descriptively in both models | Describe as deployment contrast, not a causal shift effect |
| Secondary held-out ANAEROBIC benefit | Evaluation-stride paired inference | Supported but policy-dependent | Label as secondary sensitivity result |
| Portable 95% interval coverage after activity transfer | Source-only participant-block conformal transport | Contradicted for held-out ANAEROBIC | Do not present as a safety guarantee |
| Activity identity unavailable to models | Masked-input protocol and export metadata | Supported | State explicitly |

