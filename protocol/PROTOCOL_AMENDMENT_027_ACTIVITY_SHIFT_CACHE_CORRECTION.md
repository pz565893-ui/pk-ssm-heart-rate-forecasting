# Protocol Amendment 027: Non-destructive correction of the Wearable activity-shift cache

Date: 2026-07-31

## Trigger

The first cache-construction attempt under Amendment 026 stopped at its own leakage-acceptance gate. For `seen_user_activity`, some participants with held-out-protocol sessions in the base training role had no source-protocol session represented in model fitting. Consequently, the candidate test-participant set was not a subset of source-training participants.

No activity-shift model fitting, calibration export, test export, target inspection, model comparison, or manuscript result insertion occurred from the failed partial cache. The failed `wearable_activity_shift_v1` directory is retained unchanged as an audit artifact.

## Correction

The v2 builder retains held-out-protocol `seen_user_activity` rows only for participants in the intersection of:

- participants with a source-protocol session in the immutable base training role; and
- participants with a held-out-protocol session in the immutable base training role.

The correction does not alter participant folds, session arrays, signal values, event tags, forecast origins, targets, source-user rows, joint-user-activity rows, normalization rules, model identities, seeds, calibration rules, or the test-opening token.

## Versioned artifacts

- Preserved failed cache: `data/processed/wearable_activity_shift_v1`
- Corrected cache: `data/processed/wearable_activity_shift_v2`
- Corrected builder: `scripts/build_wearable_activity_shift_cache_v2.py`
- Contract version: `2.0.0`

Every fold, direction, and deployment boundary must pass the Amendment 026 acceptance rules plus an explicit pre-write seen-user filter check. The corrected builder and preserved v1 builder are both hashed in the v2 root summary.

## Information and claim boundary

Amendment 026 otherwise remains in force. The test-opening token may be used only after the corrected cache is accepted, source-only validation fitting is complete, and output-schema compatibility is established without reading test targets. No activity-shift result or claim may be added to the manuscript before those steps are recorded.
