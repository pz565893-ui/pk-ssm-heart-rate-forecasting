# Frozen wearable-exercise v4 splits

V4 preserves v3 and corrects sex-segregated outer test folds through exact sex quotas. It adds session-level chronology, legal history pairs, activity-shift roles, temporal feasibility, fold-balance evidence, and a machine-readable leakage audit.

Strict unseen-user evaluation uses `train_prior`, which gives held-out participants zero same-user history. Few-shot personalization uses `role_prior` and must be reported separately.
