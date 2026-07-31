# Protocol Amendment 026: Wearable v4 Bidirectional Activity-Shift Lock

Date: 2026-07-31

## Purpose

This amendment resolves the activity-shift evidence required by the working title without reusing PAMAP2 or FitRec as a central benchmark. It is frozen before any model is fitted to a source-only activity cache or any activity-shift test target is inspected.

## Data and split inheritance

The experiment inherits the immutable five-fold participant assignments from Wearable v4. No participant assignment, session signal, tagged event, or original origin definition is changed. Derived caches hard-link the immutable one-second session arrays and alter only legal origin roles and source-only normalization statistics.

The two directions are:

- train, select, and calibrate on `AEROBIC`; hold out `ANAEROBIC`;
- train, select, and calibrate on `ANAEROBIC`; hold out `AEROBIC`.

The held-out protocol is absent from model fitting, early stopping, normalization, and conformal calibration.

## Deployment boundaries

Three boundaries are reported separately.

1. `source_user`: source-protocol sessions from outer-fold test participants. This isolates user shift while preserving the source activity.
2. `seen_user_activity`: held-out-protocol sessions from participants whose source-protocol sessions are in model fitting. This isolates activity shift with seen users.
3. `joint_user_activity`: held-out-protocol sessions from outer-fold test participants. This is the primary activity-shift estimand because both user and protocol are unseen.

No raw errors from protocols of different difficulty will be interpreted as a causal shift penalty without the corresponding source boundary.

## Information boundary

- Context: 300 seconds at one-second resolution.
- Forecast: the next 120 seconds, with 30-, 60-, and 120-second endpoints.
- Primary origins: tagged physiological transitions.
- Secondary origins: the frozen 120-second evaluation stride.
- Future heart rate and acceleration are unavailable.
- Historical-session personalization is disabled because it failed the validation gates recorded in Amendments 004 through 009.
- Activity identity is masked in source and target data. This prevents an untrained held-out activity embedding from becoming an arbitrary target-domain identifier.
- Schedule labels remain unavailable to the signal-only primary analysis.

## Model and seed lock

The primary paired models are the already selected `pkssm_64x4_r6` PK-SSM and the fixed 64-unit TCN. The paired seeds are `20260730` through `20260734`. Architecture, optimizer, stopping rule, context, forecast horizon, and physiological bounds are inherited unchanged from the v4 model-selection contract.

A fixed-seed secondary benchmark may include persistence, damped trend, GRU, LSTM, Transformer, first-order kinetics, and unconstrained residual SSM. Secondary models provide ranking context and cannot replace the five-seed PK-SSM versus TCN contrast.

## Selection and calibration

Each direction and outer fold is fitted independently. Early stopping uses tagged transitions from source-protocol validation participants only. Split-conformal thresholds use source-protocol calibration participants only and are applied unchanged to each test boundary. Coverage under the target protocol is empirical transport performance, not a distribution-free guarantee.

## Statistical analysis

The primary metric is participant-macro complete-trajectory MAE on `joint_user_activity`. Secondary metrics are MAE at 30, 60, and 120 seconds, RMSE, signed error, total-variation ratio, high-HR error, empirical 95% coverage, and interval width.

For each direction and boundary, PK-SSM minus TCN differences are first aggregated per participant. Participants, not origins, are resampled in 10,000 paired bootstrap replicates. A Wilcoxon signed-rank test is a sensitivity analysis. Seen-user activity results average repeated outer-fold estimates within participant before inference.

## Leakage acceptance rules

Every derived cache must satisfy all of the following before fitting:

- training, validation, and calibration contain the source protocol only;
- target test rows contain exactly the declared protocol;
- training and test session identifiers are disjoint;
- `joint_user_activity` and `source_user` participants are disjoint from training participants;
- `seen_user_activity` participants are a subset of source-training participants;
- normalization is fitted only on source-protocol base-training sessions;
- no historical-session input or activity identity reaches the model.

## Test-opening rule

Activity-shift test export requires the token `BSPC_V4_ACTIVITY_TEST_OPEN_20260731`. Before the token is used, the cache builder and activity-shift exporter must be frozen and hashed, fixed-seed source-validation fitting must complete across both directions and all five folds, and a dry run must establish output-schema compatibility without reading test targets.

## Claim boundary

Supported only if the locked results provide the corresponding evidence:

- performance under a strictly held-out structured exercise protocol;
- separate seen-user activity and joint user-activity behavior;
- source-calibrated interval transport under activity shift;
- direction-dependent failure or robustness.

Not supported regardless of outcome:

- universal activity invariance;
- transfer to arbitrary sports or sensor systems;
- stable physiological identification from latent kinetic parameters;
- benefit from historical personalization;
- clinical-grade high-intensity forecasting;
- equivalence of the training-load proxy to laboratory physiological load.
