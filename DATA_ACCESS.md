# Data access and redistribution boundary

## Primary dataset

Wearable Device Dataset from Induced Stress and Structured Exercise Sessions,
PhysioNet version 1.0.1: https://doi.org/10.13026/he0v-tf17

## External dataset

GoldenCheetah OpenData, OSF: https://doi.org/10.17605/OSF.IO/6HFPZ

Download both datasets directly from their providers and follow their current
terms of use. This repository does not redistribute raw records or reconstructed
individual trajectories.

The public release contains split-generation code, the frozen split policy,
aggregate leakage audits, and de-identified figure source data. Participant- and
session-level role manifests remain private because pseudonymous identifiers are
still individual-level metadata.

Store downloaded records outside version control. The public `.gitignore`
excludes common raw-data, cache, checkpoint, and prediction directories.
