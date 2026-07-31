# Protocol Amendment 017: Test Summary Code Lock

Date: 2026-07-31

## Purpose

This amendment freezes the cross-fold test summarizer before its first execution and before inspection of any aggregate test result. Test forecast bundles and calibrated fold reports already exist, but their aggregate values have not been read or used to modify the analysis below.

## Frozen summarizer

| File | SHA256 |
|---|---|
| `scripts/summarize_locked_v4_results.py` | `9A9C31BAEF0AFDAAC36D7247C9F7CC71A827B5BB3E1D63666698A4D3756129F9` |

## Prespecified outputs

- Five-seed PK-SSM and TCN participant-macro summaries
- Symmetric five-seed ensemble summaries
- Paired participant bootstrap confidence intervals with 10,000 replicates
- Wilcoxon signed-rank sensitivity tests
- 30, 60, and 120 second horizon errors
- Full-trajectory MAE, RMSE, and signed error
- Total-variation ratio and rapid-change amplitude ratio
- High-HR errors using fold-specific train-plus-calibration thresholds
- Fixed 160 bpm high-HR sensitivity results
- Origin-level and participant-block conformal coverage and widths
- Protocol, sex, and event-type subgroup summaries
- Fixed-seed secondary baseline ranking
- Parameter-count and CPU inference-throughput summaries

## Direction of paired effects

All paired differences are defined as:

`PK-SSM metric minus TCN metric`

For error metrics, positive values favor TCN and negative values favor PK-SSM. Confidence intervals and win fractions must retain this direction.

## Statistical unit

The primary resampling unit is the held-out participant. Each participant is keyed by outer fold and participant identifier. Origin-level observations are not treated as independent replicates for the primary confidence interval.

## Immutable output root

`outputs/locked_summary_v1`

The first successful execution owns this directory. Any later alternative analysis must use a new output root and a new protocol amendment; it cannot overwrite these outputs.
