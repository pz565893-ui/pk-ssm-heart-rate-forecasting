# Reproducibility workflow

Run the workflow from the repository root after obtaining the source datasets.
Use each script's `--help` output for exact local paths and resource options.

1. Audit the downloaded datasets with `audit_wearable_exercise.py` and
   `audit_goldencheetah_longitudinal_eligibility.py`.
2. Generate the primary leakage-controlled roles with
   `freeze_wearable_exercise_splits_v4.py`.
3. Freeze the lightweight longitudinal external subset with
   `freeze_goldencheetah_light_subset_v1.py`.
4. Build transition caches with `build_primary_transition_cache_v4.py`,
   `build_goldencheetah_transition_cache_v1.py`, and the activity-shift cache
   scripts.
5. Run source-only model selection with `run_pretest_model_selection.py` and
   `run_activity_shift_model_selection_v2.py`.
6. Export locked forecasts and apply source-only conformal calibration with the
   export and scoring scripts.
7. Produce aggregate tables with the three `summarize_*` workflows.
8. Generate Figures 1-5 with `build_figures_v4.py` and Figure 6 with
   `build_figure6_v4.py`.

The chronological protocol and amendment files in `protocol/` define the locked
decision order. Public aggregate outputs can be checked against `MANIFEST.csv`
and `SHA256SUMS.txt`. Raw records, individual role manifests, cached tensors,
checkpoints, and per-origin forecasts are deliberately excluded.
