# GoldenCheetah Lightweight Longitudinal Protocol v1

Date: 2026-07-31

## Purpose

GoldenCheetah OpenData is considered as a separate longitudinal evidence set for
historical personalization. It does not replace the wearable-exercise v4 cohort,
which remains the controlled user- and activity-shift benchmark. It also avoids
reusing FitRec as the central evidence source.

## Resource constraint

The local extraction contains 150 user archives and more than 51,000 CSV files. The
full collection will not be copied or processed. A manifest-only subset will refer
to source files in place.

## Proposed frozen subset

- 30 users selected by a seeded SHA-256 ordering.
- 20 chronologically ordered eligible sessions per user.
- At least 30 metadata-eligible sessions before selection.
- Session duration of at least 600 seconds.
- Metadata average HR between 30 and 220 bpm.
- Preference for users with at least 10 bike and 5 run sessions when enough users
  satisfy this condition.
- Raw CSV eligibility must subsequently confirm at least 90% finite HR coverage and
  at least 600 common one-second samples.
- No raw files are duplicated.

The resulting maximum is 600 sessions, which is small enough for the available CPU
environment while providing substantially deeper history than the wearable-exercise
cohort.

## Temporal roles

Within each selected user, sessions are ordered by recorded time before role
assignment. Earlier sessions may support later sessions, never the reverse. The
planned per-user sequence is:

- Sessions 1-10: historical support and model-training period.
- Sessions 11-14: validation period.
- Sessions 15-17: calibration period.
- Sessions 18-20: sealed temporal test period.

For strict unseen-user evaluation, users rather than sessions are assigned to outer
roles before any origin generation. For few-shot evaluation, only the first `k`
chronological support sessions of a held-out user may be accessed, with
`k in {0, 1, 3, 5, 10}`.

## Signals

The native one-second CSV fields are `secs`, `km`, `power`, `hr`, `cad`, and `alt`.
Derived causal inputs may include speed from backward distance differences, backward
grade, cadence, power, elapsed time, activity class, observed-value masks, and the
past HR context. Future HR, speed, power, cadence, and altitude are unavailable to
the primary forecast input.

## Decision boundary

GoldenCheetah may support the word `Personalized` only if historical support produces
a practically meaningful and cross-user-consistent improvement under the frozen
temporal protocol. The negative sparse-history result in the wearable-exercise
cohort remains reportable and cannot be removed if GoldenCheetah is favorable.
