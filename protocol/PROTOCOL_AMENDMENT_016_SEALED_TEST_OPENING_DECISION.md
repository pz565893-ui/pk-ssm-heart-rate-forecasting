# Protocol Amendment 016: Sealed Test Opening Decision

Date: 2026-07-31

## Decision

The Wearable v4 sealed test roles are authorized for one locked evaluation after completion of all pre-test gates. This amendment is written before generation of any test prediction or metric.

## Completed gates

- Participant-disjoint v4 split generation and audit completed.
- Historical-personalization validation gate completed and failed.
- Full fixed-configuration baseline validation completed.
- Prespecified PK-SSM architecture screen completed.
- Five-seed PK-SSM and TCN stability analysis completed.
- Calibration access opened only after architecture freeze.
- Forecast-export and conformal-scoring scripts dry-run on validation and calibration roles.
- Evaluation scripts hashed in Amendment 015.
- All 100 primary calibration bundles exported with frozen code.

## Locked test scope

### Primary paired comparison

- Models: PK-SSM `pkssm_64x4_r6` without historical personalization and fixed-capacity TCN
- Seeds: `20260730` through `20260734`
- Outer folds: 0 through 4
- Origin policies: `tagged_events` and `evaluation_stride`
- Outputs: every seed-specific result plus a symmetric five-seed arithmetic-mean ensemble

### Secondary fixed-seed comparison

- Seed: `20260730`
- Models: GRU, LSTM, Transformer, first-order kinetics, unconstrained residual SSM, persistence, and damped trend
- Outer folds: 0 through 4
- Origin policies: `tagged_events` and `evaluation_stride`

Ridge requires its frozen summary-feature evaluator and will be added without modifying neural predictions. Raw and `[30, 220]` bpm clipped Ridge results must both be retained.

## Frozen output roots

- Forecast bundles: `outputs/locked_evaluation_v1`
- Calibrated reports: `outputs/locked_scoring_v1`
- Cross-fold summaries: `outputs/locked_summary_v1`

## Access control

The frozen exporter requires the exact token:

`BSPC_V4_TEST_OPEN_20260731`

Supplying this token authorizes only the locked commands described above. It does not authorize model selection, hyperparameter tuning, subgroup redefinition, or deletion of unfavorable outputs.

## Checkpoint-location exception

For PK-SSM seed `20260730`, outer fold 0 is stored under `outputs/pretest_model_selection_v4_seedfix`, while outer folds 1 through 4 are stored under `outputs/wearable_v4_pretest_seedfix`. This storage difference was identified during calibration export and does not reflect a model or metric difference. Seeds `20260731` through `20260734` use the former root for all folds.

## Post-opening rules

1. Every generated test bundle is immutable and carries input and output hashes.
2. No best seed may be selected.
3. TCN must remain the primary accuracy reference even if a subgroup favors PK-SSM.
4. Historical personalization remains a negative result.
5. Test outcomes cannot change the high-HR threshold, conformal rule, subgroup eligibility threshold, origin policy, or title definition.
6. Any execution failure must preserve completed artifacts and be documented before a rerun.
7. Any genuine implementation bug requires a new amendment and preservation of the original outputs.

## Prespecified interpretation hierarchy

1. Primary transition accuracy and paired PK-SSM versus TCN difference.
2. Multi-seed stability of that difference.
3. Conformal coverage and interval width.
4. Dynamic fidelity, including total-variation ratio.
5. Protocol, sex, event-type, and high-HR subgroup results.
6. Schedule-wide secondary performance.
7. Efficiency and learned-parameter interpretability.

The manuscript will report unfavorable findings at each level and will not substitute a lower-ranked positive result for an unfavorable primary result.
