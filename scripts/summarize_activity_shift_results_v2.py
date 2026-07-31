#!/usr/bin/env python3
"""Summarize Amendment 027 activity-shift forecasts with paired shift penalties."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.score_calibrated_forecasts import (  # noqa: E402
    calibrated_intervals,
    load_ensemble,
    point_metrics,
)
from scripts.summarize_activity_shift_results_v1 import (  # noqa: E402
    bundle_path,
    collapse_repeated,
    paired_inference,
    participant_metric_maps,
    sha256_file,
    source_high_hr_threshold,
)


PROTOCOLS = ("AEROBIC", "ANAEROBIC")
BOUNDARIES = ("source_user", "seen_user_activity", "joint_user_activity")
POLICIES = ("tagged_events", "evaluation_stride")
HORIZONS = (30, 60, 120)
METRICS = (
    "trajectory_mae_bpm",
    "mae_30s_bpm",
    "mae_60s_bpm",
    "mae_120s_bpm",
)


def paired_values_inference(
    differences: np.ndarray,
    label: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size < 2:
        raise ValueError(f"Too few paired participants for {label}")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, differences.size, size=(bootstrap_replicates, differences.size))
    bootstrap = np.mean(differences[draws], axis=1)
    nonzero = differences[np.abs(differences) > 1.0e-12]
    return {
        "estimand": label,
        "participants": int(differences.size),
        "mean_difference_bpm": float(np.mean(differences)),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if nonzero.size else 1.0,
        "negative_fraction": float(np.mean(differences < 0.0)),
        "positive_fraction": float(np.mean(differences > 0.0)),
        "ties_fraction": float(np.mean(differences == 0.0)),
    }


def shift_penalty(
    source: dict[str, dict[str, float]],
    joint: dict[str, dict[str, float]],
    metric: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    participants = sorted(set(source) & set(joint))
    differences = np.asarray(
        [joint[p][metric] - source[p][metric] for p in participants], dtype=np.float64
    )
    return paired_values_inference(
        differences,
        "joint_user_activity minus source_user MAE",
        bootstrap_replicates,
        seed,
    )


def penalty_contrast(
    pk_source: dict[str, dict[str, float]],
    pk_joint: dict[str, dict[str, float]],
    tcn_source: dict[str, dict[str, float]],
    tcn_joint: dict[str, dict[str, float]],
    metric: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    participants = sorted(
        set(pk_source) & set(pk_joint) & set(tcn_source) & set(tcn_joint)
    )
    differences = np.asarray(
        [
            (pk_joint[p][metric] - pk_source[p][metric])
            - (tcn_joint[p][metric] - tcn_source[p][metric])
            for p in participants
        ],
        dtype=np.float64,
    )
    return paired_values_inference(
        differences,
        "PK-SSM shift penalty minus TCN shift penalty",
        bootstrap_replicates,
        seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecast-root",
        type=Path,
        default=Path("outputs/activity_shift_locked_evaluation_v2"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/processed/wearable_activity_shift_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/activity_shift_locked_summary_v2"),
    )
    parser.add_argument("--models", nargs="+", default=["pk_ssm", "tcn"])
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260730, 20260731, 20260732, 20260733, 20260734],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_reports: list[dict[str, Any]] = []
    participant_records: dict[
        tuple[str, str, str, str], dict[str, list[dict[str, float]]]
    ] = defaultdict(lambda: defaultdict(list))

    for policy in POLICIES:
        for heldout in PROTOCOLS:
            source = next(value for value in PROTOCOLS if value != heldout)
            for boundary in BOUNDARIES:
                for fold in range(5):
                    cache_dir = args.cache_root / heldout / boundary / f"outer_fold_{fold}"
                    threshold = source_high_hr_threshold(cache_dir, source)
                    for model in args.models:
                        calibration_paths = [
                            bundle_path(
                                args.forecast_root,
                                heldout,
                                "seen_user_activity",
                                "calibration",
                                policy,
                                fold,
                                model,
                                seed,
                            )
                            for seed in args.seeds
                        ]
                        evaluation_paths = [
                            bundle_path(
                                args.forecast_root,
                                heldout,
                                boundary,
                                "test",
                                policy,
                                fold,
                                model,
                                seed,
                            )
                            for seed in args.seeds
                        ]
                        missing = [
                            path
                            for path in calibration_paths + evaluation_paths
                            if not path.is_file()
                        ]
                        if missing:
                            raise FileNotFoundError(missing[0])
                        calibration, calibration_metadata = load_ensemble(calibration_paths)
                        evaluation, evaluation_metadata = load_ensemble(evaluation_paths)
                        fold_reports.append(
                            {
                                "origin_policy": policy,
                                "heldout_protocol": heldout,
                                "source_protocol": source,
                                "boundary": boundary,
                                "outer_fold": fold,
                                "model": model,
                                "seeds": args.seeds,
                                "point_metrics": point_metrics(
                                    evaluation, evaluation_metadata, threshold
                                ),
                                "calibration": calibrated_intervals(
                                    calibration,
                                    calibration_metadata,
                                    evaluation,
                                    evaluation_metadata,
                                    0.05,
                                    threshold,
                                ),
                                "calibration_bundle_sha256": {
                                    str(path): sha256_file(path)
                                    for path in calibration_paths
                                },
                                "evaluation_bundle_sha256": {
                                    str(path): sha256_file(path)
                                    for path in evaluation_paths
                                },
                            }
                        )
                        maps = participant_metric_maps(evaluation, evaluation_metadata)
                        target = participant_records[(policy, heldout, boundary, model)]
                        for participant, metrics in maps.items():
                            target[participant].append(metrics)

    aggregate_models: list[dict[str, Any]] = []
    paired_models: dict[str, Any] = {}
    paired_shifts: dict[str, Any] = {}
    paired_penalty_contrasts: dict[str, Any] = {}
    collapsed_all: dict[
        tuple[str, str, str, str], dict[str, dict[str, float]]
    ] = {}

    for key, records in participant_records.items():
        collapsed_all[key] = collapse_repeated(records)

    for policy in POLICIES:
        paired_models[policy] = {}
        paired_shifts[policy] = {}
        paired_penalty_contrasts[policy] = {}
        for heldout in PROTOCOLS:
            source_protocol = next(value for value in PROTOCOLS if value != heldout)
            paired_models[policy][heldout] = {}
            paired_shifts[policy][heldout] = {}
            for boundary in BOUNDARIES:
                collapsed = {
                    model: collapsed_all[(policy, heldout, boundary, model)]
                    for model in args.models
                }
                for model, model_map in collapsed.items():
                    aggregate_models.append(
                        {
                            "origin_policy": policy,
                            "heldout_protocol": heldout,
                            "source_protocol": source_protocol,
                            "boundary": boundary,
                            "model": model,
                            "participants": len(model_map),
                            **{
                                metric: float(
                                    np.mean([values[metric] for values in model_map.values()])
                                )
                                for metric in METRICS
                            },
                        }
                    )
                paired_models[policy][heldout][boundary] = {
                    metric: paired_inference(
                        collapsed["pk_ssm"],
                        collapsed["tcn"],
                        metric,
                        args.bootstrap_replicates,
                        args.bootstrap_seed,
                    )
                    for metric in METRICS
                }

            for model in args.models:
                source_map = collapsed_all[
                    (policy, heldout, "source_user", model)
                ]
                joint_map = collapsed_all[
                    (policy, heldout, "joint_user_activity", model)
                ]
                paired_shifts[policy][heldout][model] = {
                    metric: shift_penalty(
                        source_map,
                        joint_map,
                        metric,
                        args.bootstrap_replicates,
                        args.bootstrap_seed,
                    )
                    for metric in METRICS
                }
            paired_penalty_contrasts[policy][heldout] = {
                metric: penalty_contrast(
                    collapsed_all[(policy, heldout, "source_user", "pk_ssm")],
                    collapsed_all[(policy, heldout, "joint_user_activity", "pk_ssm")],
                    collapsed_all[(policy, heldout, "source_user", "tcn")],
                    collapsed_all[(policy, heldout, "joint_user_activity", "tcn")],
                    metric,
                    args.bootstrap_replicates,
                    args.bootstrap_seed,
                )
                for metric in METRICS
            }

    report = {
        "artifact_status": "locked_wearable_v4_activity_shift_summary",
        "protocol_amendment": "027",
        "models": args.models,
        "seeds": args.seeds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "primary_boundary": "joint_user_activity",
        "primary_origin_policy": "tagged_events",
        "secondary_origin_policy": "evaluation_stride",
        "activity_identity": "masked",
        "history_regime": "none",
        "aggregate_model_metrics": aggregate_models,
        "paired_model_inference": paired_models,
        "paired_shift_penalty": paired_shifts,
        "paired_shift_penalty_model_contrast": paired_penalty_contrasts,
        "fold_reports": fold_reports,
    }
    report_path = args.output_dir / "activity_shift_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SHA256SUMS.txt").write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n", encoding="ascii"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
