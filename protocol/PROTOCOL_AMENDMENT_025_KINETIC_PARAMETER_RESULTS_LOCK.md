# Protocol Amendment 025: Kinetic-Parameter Results and Interpretation Lock

Date: 2026-07-31

## Purpose

This amendment records the locked five-seed audit of PK-SSM context-conditioned kinetic parameters on the Wearable v4 held-out participants.

## Main finding

The learned parameters do not behave as reliably identified, stable participant-level physiological traits. Most time constants had weak or near-zero rank agreement across random seeds, and several nominally physiological quantities primarily re-encoded current heart-rate level.

## Signal-level re-encoding

On tagged transitions:

| Parameter | Spearman with current HR | Spearman with future mean HR | Participant ICC | Mean pairwise seed Spearman |
|---|---:|---:|---:|---:|
| `rest_hr` | 0.9711 | 0.8230 | 0.1935 | 0.7674 |
| `hr_reserve` | -0.8078 | -0.6736 | 0.2156 | 0.3549 |
| `feasible_max_hr` | 0.1084 | 0.0853 | 0.2694 | 0.2096 |

The model's `rest_hr` output must not be interpreted as measured resting heart rate. Its near-deterministic association with current exercise HR indicates context-level signal encoding.

## Time-constant stability

| Parameter | Participant ICC | Mean pairwise seed Spearman | Median within-origin seed SD |
|---|---:|---:|---:|
| `tau_fast_rise` | 0.3697 | 0.0131 | 18.24 s |
| `tau_fast_recovery` | 0.3738 | -0.0132 | 8.45 s |
| `tau_slow_rise` | 0.2142 | -0.0097 | 73.70 s |
| `tau_slow_recovery` | 0.2104 | -0.0225 | 68.48 s |

The near-zero seed-rank correlations show that individual time-constant ordering is not reproducible across initializations.

## Gain parameters

`gain_fast` had a high ensemble participant ICC of 0.9365, but its mean pairwise seed Spearman was only 0.0362 and its median within-origin seed SD was 0.1299, compared with an ensemble cross-origin SD of 0.0287. The apparent participant separation is therefore not stable across independently initialized models.

`gain_slow` had participant ICC 0.0301 and mean pairwise seed Spearman 0.0826.

## Bound saturation

No audited parameter showed material concentration within 1% of its configured finite lower or upper bound. The identifiability problem is therefore not explained by simple hard-bound saturation.

## Locked interpretation

### Supported

- The kinetic parameterization provides a constrained latent decomposition of forecast dynamics.
- Some ensemble-averaged parameters vary between participants.
- The constraints prevent grossly infeasible parameter values and output ranges.

### Not supported

- The model recovers stable individual resting HR, maximum HR, or recovery constants.
- Parameter values are reproducible across random seeds at the individual-origin level.
- The parameter head provides validated physiological phenotyping.
- High participant ICC in one ensemble parameter establishes physiological meaning.

## Manuscript consequence

The term `Personalized` can only refer to context-conditioned, participant-specific prediction, not stable trait recovery. Parameter figures must show seed variability and must label all quantities as model-internal latent kinetic parameters. Clinical or laboratory terminology must be avoided.
