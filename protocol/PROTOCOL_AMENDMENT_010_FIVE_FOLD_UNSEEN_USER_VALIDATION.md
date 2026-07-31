# Protocol Amendment 010: Five-Fold Unseen-User Validation Lock

Date: 2026-07-31

## Purpose

This amendment records the pre-test, participant-disjoint validation evidence used to assess unseen-user robustness in the wearable v4 benchmark. It also records a prespecified sensitivity analysis for the unstable extrapolation of the linear Ridge baseline.

No calibration or test partition was accessed for this analysis.

## Frozen comparison

- Data cache: `data/processed/wearable_exercise_transition_v2`
- Split design: five participant-disjoint outer folds
- Selection partition: validation only
- Seed: `20260730`
- Primary metric: participant-macro transition MAE in bpm
- Models: PK-SSM without historical personalization, last-value persistence, and Ridge summary baseline
- Ridge sensitivity analysis: predictions clipped to the broad physiological output range `[30, 220]` bpm

The clipping analysis was added because the original Ridge model exhibited extreme out-of-range extrapolation in fold 3. Both the raw and clipped results are retained. The clipped result is a robustness sensitivity analysis, not a replacement for the original baseline.

## Validation results

| Outer fold | PK-SSM, no history | Persistence | Ridge, raw | Ridge, clipped |
|---:|---:|---:|---:|---:|
| 0 | 9.3715 | 11.1912 | 11.4604 | 11.4604 |
| 1 | 9.9294 | 11.2845 | 11.0180 | 11.0180 |
| 2 | 9.2187 | 10.7958 | 10.4891 | 10.4891 |
| 3 | 7.8798 | 8.3997 | 255.5735 | 14.8777 |
| 4 | 8.1963 | 9.2178 | 8.6149 | 8.6149 |
| Mean | 8.9191 | 10.1778 | 59.4312 | 11.2920 |

All values are validation participant-macro transition MAE in bpm. Lower is better.

## Locked interpretation

1. PK-SSM outperformed persistence in all five folds. Its mean improvement was 1.2587 bpm, or 12.37% relative to persistence.
2. PK-SSM outperformed the clipped Ridge sensitivity baseline in all five folds. Its mean improvement was 2.3729 bpm, or 21.01% relative to clipped Ridge.
3. Raw Ridge was numerically unstable under participant shift in fold 3. Broad physiological clipping reduced, but did not remove, the performance failure.
4. These results support a robustness claim under unseen-user validation. They do not establish universal superiority, clinical utility, or a benefit from historical personalization.
5. The historical personalization pathway remains excluded from the primary model because both wearable and GoldenCheetah validation gates failed under the frozen protocol.

## Model-selection consequence

The confirmatory primary candidate remains PK-SSM without historical personalization. The evidence-supported framing is:

> Physiology-guided state dynamics may trade a small amount of same-user temporal accuracy for more stable performance under unseen-user shift, while historical personalization provides no reliable validation benefit in the evaluated setting.

This statement remains provisional until the sealed test evaluations are opened. Test-set language and effect estimates must be written only after all model and analysis choices are frozen.

## Reporting requirements

- Report all five fold values, not only the mean.
- Report both raw and clipped Ridge results and identify clipping as a sensitivity analysis.
- Keep same-user temporal and unseen-user conclusions separate.
- Do not describe the historical pathway as beneficial.
- Do not use calibration or test results to revise the selected architecture, clipping rule, or primary interpretation.
