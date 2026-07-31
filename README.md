# PK-SSM heart-rate transition forecasting

This repository accompanies the manuscript **Deployment-Aware Evaluation of Physiology-Guided Heart-Rate Transition Forecasting under User and Activity Shifts**, prepared for *Biomedical Signal Processing and Control*.

## Scope

The repository contains the PK-SSM implementation, comparator models, the locked
candidate configuration and protocol records, split-generation and leakage-audit code,
aggregate evaluation outputs, and de-identified source data for Figures 1-6.

The release intentionally does **not** contain raw wearable records,
participant- or session-level manifests, cached tensors, model checkpoints,
per-origin predictions, or local machine paths. Raw data must be obtained from
the original providers under their respective terms.

## Data sources

1. Wearable Device Dataset from Induced Stress and Structured Exercise Sessions,
   PhysioNet, version 1.0.1, DOI: https://doi.org/10.13026/he0v-tf17
2. GoldenCheetah OpenData, OSF, DOI: https://doi.org/10.17605/OSF.IO/6HFPZ

See `DATA_ACCESS.md` for the data boundary and expected local preparation.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the virtual environment using the command appropriate for the local
operating system before running the pipeline.

## Repository map

- `transition_forecasting/`: PK-SSM, baselines, datasets, training, conformal
  calibration, and transition metrics.
- `scripts/`: split, cache, model-selection, evaluation, summary, and figure code.
- `configs/`: locked PK-SSM candidate grid.
- `protocol/`: frozen experiment policy and chronological amendments.
- `outputs/`: aggregate, non-identifying locked results only.
- `splits/wearable_exercise_v4/`: public split policy and aggregate audits only.
- `figures_v4/`: publication figures and de-identified source data.

## Reproduction entry points

Each command exposes its accepted arguments through `--help`. Run commands from
the repository root.

```bash
python scripts/freeze_wearable_exercise_splits_v4.py --help
python scripts/freeze_goldencheetah_light_subset_v1.py --help
python scripts/build_primary_transition_cache_v4.py --help
python scripts/build_goldencheetah_transition_cache_v1.py --help
python scripts/run_pretest_model_selection.py --help
python scripts/run_activity_shift_model_selection_v2.py --help
python scripts/summarize_locked_v4_results.py --help
python scripts/summarize_locked_goldencheetah_results.py --help
python scripts/build_figures_v4.py --help
python scripts/build_figure6_v4.py --help
```

The complete order, input boundaries, and output mapping are documented in
`REPRODUCIBILITY.md`. Full model refitting requires locally downloaded source
data and substantially more compute than reusing the included aggregate outputs.

## Privacy and provenance

`PACKAGE_AUDIT.json`, `MANIFEST.csv`, and `SHA256SUMS.txt` document the public
allowlist, privacy checks, file sizes, and cryptographic hashes used for this
release. The release builder is included at
`tools/build_public_release_package_v4.py`.

## Citation and license

Use `CITATION.cff` for author and manuscript metadata. Add the final GitHub URL
and Zenodo DOI after archiving the release. No software license has been selected
yet; the corresponding author must choose one before public distribution. See
`LICENSE_SELECTION_REQUIRED.md`.
