# Terminology ledger v4

This ledger is the canonical terminology source for the manuscript, supplement, figures, tables, code documentation, and submission files.

| Canonical term | First-use definition | Prohibited or non-canonical variants | Decision |
|---|---|---|---|
| physiology-guided kinetic state-space model (PK-SSM) | Full model name followed by PK-SSM | PhyShift-HR; physiological model; mechanistic physiology model | Use PK-SSM after first expansion. "Physiology-guided" refers only to structural constraints. |
| Wearable Device Dataset from Induced Stress and Structured Exercise Sessions | Primary structured exercise dataset | FitRec; Endomondo; PAMAP2; Wearable dataset as a formal title | Use the full repository title at first mention and "Wearable dataset" thereafter. |
| GoldenCheetah OpenData | Independent dataset-specific refitting benchmark | external validation; zero-shot transfer; unseen-sport benchmark | Call it an independent architectural replication after refitting. |
| tagged transitions | Protocol-defined aerobic boundaries, sprint onset/offset, and recovery events | event windows; transition samples | Use for the primary origin policy. |
| schedule-wide origins | Origins generated with a fixed 120-s evaluation stride | evaluation stride; random origins | Use for the secondary deployment policy. |
| held-out user | Participant absent from model fitting, validation, and calibration in the relevant fold | new subject; unseen patient | Prefer "held-out user" or "unseen user"; these are not patients. |
| source-user boundary | Source-protocol session from a user absent from source-model fitting | in-domain user shift | Use only in Amendment 026 activity analysis. |
| seen-user activity shift | Held-out protocol from a user represented by a source-protocol training session | activity transfer | Use only when the source-session membership condition is satisfied. |
| joint user-activity shift | Held-out protocol from a participant absent from fitting | unseen sport; zero-shot sport transfer | This is the primary activity-shift estimand. |
| current-context conditioning | Forecast conditioned on the individual's preceding 300-s signal | longitudinal personalization; historical personalization | This is the manuscript's bounded meaning of "personalized". |
| prior-session history pathway | Optional encoder for strictly earlier same-user sessions | personal profile; physiological memory | State explicitly that it failed validation and was disabled. |
| latent kinetic variables | Model-internal baseline, reserve, gains, and time constants | measured physiology; biomarkers; phenotypes | Never interpret as physiological measurements or stable traits. |
| trajectory MAE | Participant-macro mean absolute error over the complete 120-s forecast | point MAE; sample-wise MAE | Primary accuracy metric, reported in beats per minute. |
| paired difference | Participant-level PK-SSM minus TCN metric | improvement | Positive error differences favor TCN; always report the confidence interval. |
| 95% prediction interval | Conformalized lower and upper forecast bounds targeting marginal coverage | confidence interval | Report coverage together with full interval width. |
| training-load proxy | Exploratory internal index based on duration, heart-rate strata, and a bounded model response summary | training load; physiological load; TRIMP | Never equate with laboratory load, fatigue, recovery, strain, or clinical risk. |

## Statistical wording lock

- Use "showed no supported difference" when the paired confidence interval crosses zero and no equivalence test was conducted.
- Do not use "statistically equivalent" or "equivalent" as a formal inference.
- Use "had lower error" for a horizon-specific observed difference; do not generalize it to overall superiority.
- Use "descriptive" for recorded-sex, protocol, event-type, and sport strata unless a pre-specified inferential analysis supports a broader statement.

## Activity-shift terminology additions

| Canonical term | Definition | Usage boundary |
|---|---|---|
| activity-shift deployment contrast | Joint-user-and-activity MAE minus source-user MAE | Not a causal effect because protocol groups can differ in intrinsic difficulty |
| tagged-events policy | Pre-registered primary origin policy | Use as the primary activity-shift analysis |
| evaluation-stride policy | Pre-registered secondary origin policy | Use only for sensitivity analysis |
| masked activity identity | Activity labels are excluded from model inputs | State whenever interpreting strict activity transfer |
| source-only participant-block conformal calibration | Interval scores estimated only from source calibration participants | Does not guarantee target-activity coverage |

