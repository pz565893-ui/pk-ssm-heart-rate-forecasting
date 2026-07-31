# Protocol Amendment 024: Kinetic-Parameter Audit Lock

Date: 2026-07-31

## Purpose

This amendment freezes the PK-SSM kinetic-parameter audit before aggregate parameter results are inspected.

## Frozen audit code

| File | SHA256 |
|---|---|
| `scripts/audit_locked_kinetic_parameters.py` | `A65EF1F305A297C905313EDFC0D7A83EC1AC92FAA70FA88BE7A4DE5493B1F667` |

## Prespecified analyses

- Five-seed ensemble parameter distribution
- Within-origin seed variability
- Pairwise seed-rank correlation
- One-way participant ICC
- Between-participant and within-participant variability
- Proximity to fixed model bounds
- Exploratory Spearman correlations with current HR, future mean HR, and future end-minus-current HR
- Participant-level parameter summaries

## Interpretation boundary

The parameters are model-internal, context-conditioned latent states. They are not direct measurements of resting HR, maximum HR, recovery constants, oxygen uptake, cardiovascular fitness, or any laboratory physiological endpoint.

High ICC alone does not establish physiological validity. Strong correlation with current or future HR may indicate that the parameter head re-encodes observable signal level rather than identifying a stable participant trait.

## Immutable output root

`outputs/locked_kinetic_parameter_audit_v1`
