# Protocol Amendment 018: Ridge Test Evaluator Lock

Date: 2026-07-31

## Purpose

This amendment freezes the Ridge summary-feature test evaluator before its first execution. Ridge coefficients, intercepts, and feature-scaling parameters were fitted and frozen during validation-only model development. No Ridge test prediction has yet been generated.

## Frozen evaluator

| File | SHA256 |
|---|---|
| `scripts/evaluate_locked_ridge_test.py` | `B96CC5C19EF01BDB16A5CDF4F99FBA986C8FCA468BD3F126750B21D94AE9E65E` |

## Fixed model

- Summary features: 73
- Forecast outputs: 120 one-second horizons
- Ridge alpha: 1
- Fitted coefficients plus intercepts: 8,880
- Seed label: `20260730`
- Feature scaler: frozen training-only mean and scale

## Required paired outputs

The same frozen Ridge prediction must be reported in two forms:

1. Raw, without output clipping.
2. Clipped to `[30, 220]` bpm as a physiological-range sensitivity analysis.

Clipping cannot replace or hide the raw result.

## Test scope

- Outer folds: 0 through 4
- Origin policies: `tagged_events` and `evaluation_stride`
- Metrics: participant-macro trajectory MAE, RMSE, signed error, 30/60/120 second MAE, total-variation ratio, and high-HR errors
- Output root: `outputs/locked_ridge_test_v1`

The previously frozen test-opening token is required. Results will be appended to the baseline table without changing any neural or state-space prediction.
