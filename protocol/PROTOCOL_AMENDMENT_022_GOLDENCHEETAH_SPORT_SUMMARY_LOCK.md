# Protocol Amendment 022: GoldenCheetah Sport-Summary Code Lock

Date: 2026-07-31

## Purpose

This amendment freezes the GoldenCheetah sport-label summarizer before its first execution. Existing aggregate results used the intensity-protocol field and did not yet inspect the frozen `protocol_version` sport labels.

## Frozen summarizer

| File | SHA256 |
|---|---|
| `scripts/summarize_goldencheetah_sport_groups.py` | `58BEECA7BA29FA0A60B06F45C564952A09AA42F8DA3454721F78575F6E6DBE8A` |

## Fixed analysis

- Sport label source: `protocol_version` in the frozen origin manifest
- Models: all nine frozen GoldenCheetah models
- Primary paired comparison: PK-SSM minus TCN
- Resampling unit: participant within sport label
- Bootstrap replicates: 10,000
- Eligibility threshold: at least five participants
- Output root: `outputs/goldencheetah_sport_summary_v1`

## Interpretation boundary

The analysis will explicitly list training and test sport sets. A sport-stratified result is not an unseen-sport result unless its test label is absent from all training origins. Walking cannot be claimed unless a walking label exists in the frozen benchmark.
