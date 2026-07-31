# Protocol Amendment 013: PK-SSM and TCN Seed-Stability Lock

Date: 2026-07-31

## Purpose

This amendment records the prespecified five-seed stability comparison between the retained PK-SSM architecture and the strongest fixed-configuration validation baseline, TCN. All fits used only train and validation roles from Wearable v4. Calibration and test roles remained sealed.

## Fixed comparison

- PK-SSM: `pkssm_64x4_r6`, historical personalization disabled
- TCN: fixed 64-unit configuration used in Amendment 011
- Seeds: `20260730` through `20260734`
- Outer folds: 0 through 4
- Primary quantity per seed: mean participant-macro transition trajectory MAE across five validation folds
- Paired difference: PK-SSM MAE minus TCN MAE; positive values favor TCN

## Seed-level results

| Seed | PK-SSM mean MAE | TCN mean MAE | Paired difference |
|---:|---:|---:|---:|
| 20260730 | 8.9191 | 8.6409 | +0.2782 |
| 20260731 | 8.9038 | 8.9473 | -0.0435 |
| 20260732 | 9.0073 | 8.7380 | +0.2693 |
| 20260733 | 9.1297 | 8.6552 | +0.4745 |
| 20260734 | 9.1757 | 8.6629 | +0.5128 |
| Across-seed mean | 9.0271 | 8.7288 | +0.2983 |
| Across-seed sample SD | 0.1223 | 0.1278 | 0.2209 |

All errors and differences are in bpm. Lower MAE is better.

## Paired stability summary

- TCN had the lower five-fold mean in four of five seeds.
- TCN had the lower MAE in 22 of the 25 matched fold-by-seed comparisons.
- PK-SSM had the lower MAE in three matched comparisons, all within seed `20260731`.
- The mean paired seed-level advantage of TCN was 0.2983 bpm.

## Locked interpretation

1. The single-seed TCN advantage reported in Amendment 011 is not an isolated initialization result.
2. TCN is the strongest validation accuracy model under the evaluated fixed task and capacity settings.
3. PK-SSM must not be described as the most accurate model overall.
4. PK-SSM remains superior to the matched unconstrained residual SSM and is competitive with other deep baselines, so physiology-guided dynamics may still support mechanistic interpretability, bounded trajectories, uncertainty modeling, or boundary-specific robustness.
5. Historical personalization remains unsupported and cannot be restored by selecting a favorable seed.
6. The working title does not establish efficacy. The term `Personalized` must be defined as context-conditioned individual kinetic inference, while the historical-record pathway is reported as a negative validation result.

## Final-model consequence

No single best seed may be selected for test evaluation. Before test opening, the protocol must freeze one of the following without reference to test data:

- a five-seed ensemble for both PK-SSM and TCN;
- paired reporting of all five seed-specific test fits; or
- one canonical seed for both models, with the multi-seed validation result retained as the primary stability evidence.

The same rule must be applied symmetrically to PK-SSM and TCN.

## Remaining pre-test questions

- Whether PK-SSM provides better calibrated uncertainty than TCN after calibration-only conformalization.
- Whether PK-SSM is more robust under activity shift, high-HR transitions, or sparse-signal conditions.
- Whether learned kinetic parameters show stable participant-specific structure without claiming clinical validity.
- Whether an explicitly frozen validation-only ensemble is scientifically justified; no ensemble may be invented after test access.
