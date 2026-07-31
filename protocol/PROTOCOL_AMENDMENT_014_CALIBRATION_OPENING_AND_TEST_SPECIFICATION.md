# Protocol Amendment 014: Calibration Opening and Final Test Specification

Date: 2026-07-31

## Purpose

This amendment authorizes calibration-role access after completion of architecture selection and seed-stability analysis. Test roles remain sealed. It freezes the uncertainty and multi-seed evaluation rules before any test outcome is generated.

## Model boundary

### Primary paired models

- PK-SSM: `pkssm_64x4_r6`, historical personalization disabled
- TCN: fixed 64-unit configuration
- Seeds for both models: `20260730` through `20260734`

### Secondary fixed-seed comparators

- GRU
- LSTM
- Transformer
- First-order kinetics
- Unconstrained residual SSM
- Persistence
- Damped trend
- Ridge, raw and clipped sensitivity

Secondary comparators use seed `20260730` because their purpose is benchmark context rather than seed-level inference.

## Multi-seed rule

For PK-SSM and TCN, the final report will include:

1. Every seed-specific estimate.
2. The arithmetic mean and sample standard deviation across the five paired seeds.
3. Paired PK-SSM minus TCN differences within each seed and outer fold.
4. A prespecified arithmetic-mean seed ensemble as a secondary analysis.

No best seed may be selected. The same ensemble construction must be used for both models.

## Evaluation origin sets

- Primary transition analysis: `tagged_events`
- Secondary schedule-wide analysis: `evaluation_stride`

The primary manuscript claim must use tagged transition origins. Schedule-wide results must be labeled secondary and cannot replace an unfavorable transition result.

## Point prediction metrics

- Participant-macro trajectory MAE over all valid seconds in the 120-second horizon
- Participant-macro MAE at 30, 60, and 120 seconds
- Participant-macro RMSE over the full trajectory
- Mean signed error
- Total-variation ratio between predicted and observed trajectories
- Rapid-change attenuation and lag summaries where the event definition is available

Metrics will also be stratified by protocol, sex, high-HR status, and tagged event type when sample sizes permit. Subgroups with fewer than five participants will be labeled descriptive and will not support inferential claims.

## High-HR definition

The primary high-HR threshold will be derived only from pooled training and calibration targets within each outer fold as the 90th percentile of valid HR. A fixed threshold of 160 bpm will be reported as a sensitivity analysis. Test targets cannot define or revise the threshold.

## Raw probabilistic intervals

Both PK-SSM and TCN output heteroscedastic Student-t distributions. Raw central 95% intervals will use each model's predicted location, scale, and degrees of freedom.

## Split-conformal calibration

All conformal thresholds are estimated separately for each outer fold, model, seed or ensemble, origin set, and interval target.

### Pointwise horizons

For 30, 60, and 120 seconds, the nonconformity score is:

`abs(observed - predicted_mean) / max(predicted_scale, 1e-6)`

The finite-sample split-conformal quantile uses the ceiling-adjusted empirical rank for nominal 95% coverage.

### Simultaneous 120-second curve

For each origin, the score is the maximum normalized absolute residual across all valid forecast seconds. The resulting threshold expands the predicted scale at every horizon second and targets simultaneous curve coverage.

### Participant-block sensitivity

As a conservative dependence-aware sensitivity analysis, one score per calibration participant is formed by taking the maximum applicable origin score for that participant. A ceiling-adjusted 95% quantile across participant scores is then used. This interval is expected to be wider and will not replace the origin-level primary analysis.

## Coverage reporting

- Marginal coverage at 30, 60, and 120 seconds
- Simultaneous 120-second curve coverage
- Mean and median interval width
- Participant-macro coverage
- Calibration error relative to 95%
- Coverage in the high-HR subgroup

Coverage confidence intervals will use participant-cluster bootstrap resampling. Calibration outcomes may determine conformal thresholds but cannot change point predictions or model architecture.

## Efficiency reporting

- Trainable parameter count
- CPU inference time per origin after warm-up
- Batch throughput
- Peak resident memory where measurable

All models will be timed with the same batch sizes and CPU thread setting. Training wall-clock values from differently scheduled runs will not be compared as if they were controlled benchmarks.

## Test-opening gate

Test access is permitted only after a test-only evaluation script has been created and successfully dry-run on validation and calibration roles. The dry run may check schema, tensor shape, and metric completeness, but it must not inspect any test target or prediction.

After the script is frozen and hashed, test evaluation will be run once. Any post-test bug fix must be documented, must not change the scientific decision rule, and must preserve the original output for audit.
