# Protocol Amendment 021: GoldenCheetah Summary Code Lock

Date: 2026-07-31

## Purpose

This amendment freezes the GoldenCheetah cross-model and participant-level test summarizer before its first execution and before inspection of any aggregate GoldenCheetah test result.

## Frozen summarizer

| File | SHA256 |
|---|---|
| `scripts/summarize_locked_goldencheetah_results.py` | `6E6699C78310C8FC57EA1471047DAF255663F04C12301B26FA01621D65FF6A77` |

## Prespecified outputs

- Nine-model test ranking on tagged transitions and schedule-wide origins
- Participant-macro MAE, RMSE, signed error, 30/60/120 second MAE, and total-variation ratio
- PK-SSM minus TCN participant-paired differences
- 10,000-replicate participant bootstrap intervals
- Wilcoxon signed-rank sensitivity tests
- Activity-label and sex summaries with sample-size labels
- Raw Student-t and origin-level conformal coverage and interval widths
- Parameter count and measured CPU throughput

## Statistical direction

Paired differences are PK-SSM minus TCN. Positive error differences favor TCN; negative differences favor PK-SSM.

## Immutable output root

`outputs/goldencheetah_locked_summary_v1`

No aggregate test result has been inspected at the time of this code lock.
