# Activity-shift results lock for manuscript v4

## Locked analysis contract

- Protocol amendment: 027.
- Activity identity in model inputs: masked.
- History regime: none, budget 0.
- Models: PK-SSM and TCN.
- Resampling: five outer folds and five locked seeds per model.
- Primary origin policy: tagged events.
- Secondary origin policy: evaluation stride.
- Primary deployment boundary: joint user and activity shift.
- Inference: participant-paired mean differences, 10,000 participant bootstrap replicates for 95% confidence intervals, and paired Wilcoxon tests.
- Calibration: source-only participant-block conformal calibration; no target labels were used to set interval widths.
- Test opening: one-time locked opening after 100/100 source-validation fits and 200/200 source-calibration exports completed without failures.

The paired quantity called an activity-shift deployment contrast is defined as joint-user-and-activity MAE minus source-user MAE. It is a deployment contrast, not a causal estimate of activity shift, because source and target protocol groups can differ in intrinsic forecasting difficulty.

## Primary tagged-events results

### Participant-macro trajectory MAE

| Held-out protocol | Source protocol | Boundary | PK-SSM (bpm) | TCN (bpm) | Participants |
|---|---|---:|---:|---:|---:|
| AEROBIC | ANAEROBIC | Source user | 14.247 | 14.712 | 31 |
| AEROBIC | ANAEROBIC | Seen user, new activity | 9.469 | 9.607 | 29 |
| AEROBIC | ANAEROBIC | Joint user and activity | 10.049 | 10.168 | 30 |
| ANAEROBIC | AEROBIC | Source user | 7.550 | 7.274 | 30 |
| ANAEROBIC | AEROBIC | Seen user, new activity | 14.777 | 15.415 | 29 |
| ANAEROBIC | AEROBIC | Joint user and activity | 15.017 | 15.486 | 31 |

### Paired PK-SSM minus TCN differences at the joint boundary

Negative values favor PK-SSM. Horizon-specific endpoints are exploratory and were not multiplicity-adjusted.

| Held-out protocol | Metric | Difference (bpm) | Bootstrap 95% CI | Wilcoxon p | PK-SSM better | n |
|---|---|---:|---:|---:|---:|---:|
| AEROBIC | Trajectory MAE | -0.119 | [-0.638, 0.374] | 0.952 | 50.0% | 30 |
| AEROBIC | 30-s MAE | 0.856 | [0.352, 1.416] | 0.0047 | 26.7% | 30 |
| AEROBIC | 60-s MAE | -0.148 | [-0.821, 0.511] | 0.984 | 36.7% | 30 |
| AEROBIC | 120-s MAE | -1.117 | [-2.004, -0.279] | 0.0155 | 73.3% | 30 |
| ANAEROBIC | Trajectory MAE | -0.469 | [-1.187, 0.174] | 0.152 | 58.1% | 31 |
| ANAEROBIC | 30-s MAE | 0.053 | [-0.377, 0.439] | 0.568 | 48.4% | 31 |
| ANAEROBIC | 60-s MAE | -0.415 | [-1.185, 0.324] | 0.327 | 61.3% | 31 |
| ANAEROBIC | 120-s MAE | -1.073 | [-2.434, 0.103] | 0.195 | 64.5% | 31 |

The opposite signs at 30 s and 120 s for held-out AEROBIC, together with the unsupported trajectory-level contrast, preclude a claim of general superiority.

## Activity-shift deployment contrasts

The contrast is joint-user-and-activity MAE minus source-user MAE. Negative values mean the joint target happened to be easier than the source-user boundary; positive values mean it was harder.

### Primary tagged-events policy

| Held-out protocol | Model | Contrast (bpm) | Bootstrap 95% CI | Wilcoxon p | n |
|---|---|---:|---:|---:|---:|
| AEROBIC | PK-SSM | -4.216 | [-6.529, -1.809] | 0.0013 | 30 |
| AEROBIC | TCN | -4.526 | [-7.102, -1.933] | 0.0019 | 30 |
| ANAEROBIC | PK-SSM | 7.502 | [5.364, 9.658] | <0.000001 | 30 |
| ANAEROBIC | TCN | 8.215 | [5.668, 10.848] | <0.000001 | 30 |

The PK-SSM minus TCN difference in deployment contrasts was 0.310 bpm for held-out AEROBIC (95% CI -0.445 to 1.070; p = 0.465) and -0.714 bpm for held-out ANAEROBIC (95% CI -1.509 to 0.037; p = 0.064). Thus, the primary policy did not support a model-specific reduction in the deployment contrast.

### Secondary evaluation-stride policy

| Held-out protocol | Model | Contrast (bpm) | Bootstrap 95% CI | Wilcoxon p | n |
|---|---|---:|---:|---:|---:|
| AEROBIC | PK-SSM | -1.538 | [-3.419, 0.235] | 0.245 | 30 |
| AEROBIC | TCN | -1.459 | [-3.401, 0.415] | 0.229 | 30 |
| ANAEROBIC | PK-SSM | 4.391 | [2.788, 6.271] | 0.000001 | 30 |
| ANAEROBIC | TCN | 5.347 | [3.541, 7.465] | 0.000002 | 30 |

The model difference in deployment contrasts was -0.079 bpm for held-out AEROBIC (95% CI -0.756 to 0.639; p = 0.641) and -0.956 bpm for held-out ANAEROBIC (95% CI -1.498 to -0.456; p = 0.0020). This secondary result is direction-specific and does not replace the null primary-policy contrast.

## Sensitivity analysis of joint-boundary model differences

Under the evaluation-stride policy, the paired trajectory-MAE difference was -0.347 bpm for held-out AEROBIC (95% CI -0.880 to 0.145; p = 0.477; n = 30) and -0.493 bpm for held-out ANAEROBIC (95% CI -0.764 to -0.225; p = 0.0019; n = 31). The held-out ANAEROBIC result suggests a modest secondary-policy benefit, but the absence of a primary tagged-events trajectory effect means that the benefit is not robust to the origin policy.

## Source-calibrated interval transport at the primary joint boundary

Values are means across five outer folds. Widths were determined without target-label recalibration.

| Held-out protocol | Model | 120-s point coverage | 120-s mean width (bpm) | Simultaneous curve coverage | Simultaneous mean width (bpm) | Raw point coverage | High-HR raw coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| AEROBIC | PK-SSM | 0.978 | 164.0 | 0.995 | 143.0 | 0.878 | 0.728 |
| AEROBIC | TCN | 0.956 | 103.4 | 0.968 | 114.9 | 0.969 | 0.859 |
| ANAEROBIC | PK-SSM | 0.924 | 125.7 | 0.903 | 129.9 | 0.811 | 0.751 |
| ANAEROBIC | TCN | 0.859 | 113.2 | 0.830 | 110.7 | 0.677 | 0.660 |

Source-calibrated intervals were not uniformly transportable. They approached or exceeded nominal pointwise coverage for held-out AEROBIC only by becoming very wide. Both models under-covered held-out ANAEROBIC, particularly TCN, and simultaneous curve coverage fell to 0.830 for TCN. These intervals should therefore be presented as deployment diagnostics rather than reliable safety guarantees.

## Ready-to-insert Results text

### Strict activity-shift evaluation

We next evaluated bidirectional activity transfer with activity identity masked from the inputs and no history pathway. Under the primary tagged-events policy, transferring from ANAEROBIC to held-out AEROBIC yielded joint-user-and-activity trajectory MAEs of 10.049 bpm for PK-SSM and 10.168 bpm for TCN. Their participant-paired difference was -0.119 bpm (95% bootstrap CI, -0.638 to 0.374; p = 0.952; n = 30). In the reverse direction, joint-boundary MAE increased to 15.017 bpm for PK-SSM and 15.486 bpm for TCN, with a paired difference of -0.469 bpm (95% CI, -1.187 to 0.174; p = 0.152; n = 31). Thus, the primary analysis did not support a trajectory-level advantage for PK-SSM in either transfer direction. Exploratory horizon-specific comparisons were mixed: for held-out AEROBIC, PK-SSM was worse at 30 s but better at 120 s, whereas no horizon-specific confidence interval excluded zero for held-out ANAEROBIC.

The magnitude and sign of the deployment contrast depended more strongly on transfer direction than on model class. Relative to the source-user boundary, joint-boundary trajectory MAE decreased by 4.216 bpm for PK-SSM and 4.526 bpm for TCN when AEROBIC was held out, but increased by 7.502 and 8.215 bpm, respectively, when ANAEROBIC was held out. The corresponding PK-SSM minus TCN contrast-of-contrasts was unsupported in both primary comparisons. Because this estimand compares different protocol groups, it quantifies a deployment contrast rather than a causal penalty attributable only to distribution shift. Under the secondary evaluation-stride policy, PK-SSM showed a 0.493-bpm trajectory advantage for held-out ANAEROBIC (95% CI, -0.764 to -0.225; p = 0.0019), but not for held-out AEROBIC. This origin-policy dependence further argues against a general activity-transfer benefit.

### Interval transport under activity shift

Participant-block conformal scores fitted only on source calibration data did not transfer uniformly across activity directions. At the primary joint boundary, mean 120-s pointwise coverage was 0.978 for PK-SSM and 0.956 for TCN when AEROBIC was held out, with mean widths of 164.0 and 103.4 bpm. For held-out ANAEROBIC, coverage decreased to 0.924 and 0.859 despite mean widths of 125.7 and 113.2 bpm. Simultaneous 120-s curve coverage showed the same direction asymmetry and fell to 0.830 for TCN in held-out ANAEROBIC. Source-only calibration therefore provided a useful stress test but not a portable uncertainty guarantee.

## Ready-to-insert Discussion text

The strict activity-shift analysis changes the interpretation of PK-SSM from a uniformly superior forecaster to a structured model whose relative performance depends on the deployment boundary and evaluation policy. In the primary tagged-events analysis, neither transfer direction supported a trajectory-level advantage over TCN. A modest advantage appeared for held-out ANAEROBIC only under the secondary evaluation-stride policy. More importantly, the joint-minus-source contrast changed sign across directions and was several times larger than the between-model difference. This asymmetry can reflect both distribution shift and intrinsic differences between the source and target protocol groups, so it should not be interpreted as a causal measure of transfer difficulty. The uncertainty results reinforce the same boundary: source-calibrated intervals were extremely wide for held-out AEROBIC and still under-covered held-out ANAEROBIC. These findings support reporting activity shift as a deployment stress test, not as evidence that kinetic structure guarantees cross-activity robustness.

## Figure 6 legend draft

**Figure 6. Strict bidirectional activity-shift evaluation.** **a**, Participant-macro trajectory MAE under the primary tagged-events policy for the source-user, seen-user/new-activity, and joint-user-and-activity boundaries. Models were trained on the protocol group named in parentheses, and activity identity was masked from all model inputs. **b**, Participant-paired PK-SSM minus TCN trajectory-MAE differences at the joint boundary under the primary tagged-events and secondary evaluation-stride policies. Negative values favor PK-SSM. **c**, Paired activity-shift deployment contrasts, defined as joint-user-and-activity MAE minus source-user MAE. Negative values indicate that the joint target was easier than the source-user boundary; positive values indicate that it was harder. This contrast is not a causal estimate of activity shift because source and target protocol groups may differ in intrinsic difficulty. **d**, Fold-macro 120-s participant-block conformal coverage versus mean interval width at the primary joint boundary. The dashed line marks nominal 0.95 coverage. Forest-plot points show participant-paired mean differences and bars show 95% confidence intervals from 10,000 participant-bootstrap replicates. Results pool five locked seeds within each of five outer folds; sample sizes were 29-31 participants depending on boundary. Source data are provided with the figure.

## Claim-evidence boundary

| Candidate claim | Evidence status | Permitted wording |
|---|---|---|
| PK-SSM is generally better under activity shift | Contradicted by primary trajectory inference | Do not claim |
| PK-SSM improves held-out ANAEROBIC under evaluation stride | Supported only as a secondary-policy result | Report as modest, direction-specific, and policy-dependent |
| Activity shift always increases forecasting error | Contradicted; held-out AEROBIC contrasts were negative | Do not claim |
| Transfer direction dominates model differences | Supported descriptively by deployment contrasts | State as an observed asymmetry, not a causal mechanism |
| Source calibration guarantees 95% coverage after activity shift | Contradicted for held-out ANAEROBIC | Do not claim |
| Physiological structure guarantees cross-activity robustness | Not supported | Do not claim |
| Activity identity was unavailable to the models | Supported by the masked-input contract | State explicitly |

