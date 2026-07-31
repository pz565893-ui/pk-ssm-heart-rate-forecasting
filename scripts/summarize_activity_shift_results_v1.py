#!/usr/bin/env python3
"""Summarize locked bidirectional Wearable v4 activity-shift forecasts."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


PROTOCOLS = ("AEROBIC", "ANAEROBIC")
BOUNDARIES = ("source_user", "seen_user_activity", "joint_user_activity")
HORIZONS = (30, 60, 120)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_path(
    root: Path,
    heldout: str,
    boundary: str,
    role: str,
    policy: str,
    fold: int,
    model: str,
    seed: int,
) -> Path:
    return (
        root
        / heldout
        / boundary
        / role
        / policy
        / f"outer_fold_{fold}"
        / model
        / f"seed_{seed}"
        / "forecast_bundle.npz"
    )


def source_high_hr_threshold(cache_dir: Path, source_protocol: str) -> float:
    values: list[np.ndarray] = []
    for path in sorted((cache_dir / "sessions").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["protocol"].item()) != source_protocol:
                continue
            if str(archive["role"].item()) not in {"train", "calibration"}:
                continue
            valid = archive["hr_range_valid"].astype(bool) & archive[
                "hr_offline_qc_valid"
            ].astype(bool)
            if np.any(valid):
                values.append(archive["hr_bpm"][valid].astype(np.float64))
    if not values:
        raise ValueError(f"No source HR values in {cache_dir}")
    return float(np.quantile(np.concatenate(values), 0.90))


def participant_metric_maps(
    bundle: dict[str, np.ndarray], metadata: list[dict[str, str]]
) -> dict[str, dict[str, float]]:
    participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)
    mean = bundle["mean"]
    target = bundle["target"]
    valid = bundle["target_valid_mask"].astype(bool)
    absolute = np.abs(mean - target)
    output: dict[str, dict[str, float]] = {}
    for participant in np.unique(participants):
        rows = participants == participant
        selected_valid = valid[rows]
        if not np.any(selected_valid):
            continue
        values = {
            "trajectory_mae_bpm": float(np.mean(absolute[rows][selected_valid]))
        }
        for horizon in HORIZONS:
            index = horizon - 1
            horizon_valid = valid[rows, index]
            values[f"mae_{horizon}s_bpm"] = float(
                np.mean(absolute[rows, index][horizon_valid])
            )
        output[str(participant)] = values
    return output


def collapse_repeated(
    records: dict[str, list[dict[str, float]]]
) -> dict[str, dict[str, float]]:
    return {
        participant: {
            metric: float(np.mean([record[metric] for record in participant_records]))
            for metric in participant_records[0]
        }
        for participant, participant_records in records.items()
    }


def paired_inference(
    first: dict[str, dict[str, float]],
    second: dict[str, dict[str, float]],
    metric: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    participants = sorted(set(first) & set(second))
    differences = np.asarray(
        [first[p][metric] - second[p][metric] for p in participants], dtype=np.float64
    )
    if differences.size < 2:
        raise ValueError(f"Too few paired participants for {metric}")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, differences.size, size=(bootstrap_replicates, differences.size))
    bootstrap = np.mean(differences[draws], axis=1)
    nonzero = differences[np.abs(differences) > 1.0e-12]
    p_value = float(wilcoxon(nonzero).pvalue) if nonzero.size else 1.0
    return {
        "participants": len(participants),
        "mean_pk_ssm_minus_tcn": float(np.mean(differences)),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wilcoxon_p": p_value,
        "pk_ssm_better_fraction": float(np.mean(differences < 0.0)),
        "tcn_better_fraction": float(np.mean(differences > 0.0)),
        "ties_fraction": float(np.mean(differences == 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecast-root",
        type=Path,
        default=Path("outputs/activity_shift_locked_evaluation_v1"),
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/processed/wearable_activity_shift_v1")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/activity_shift_locked_summary_v1")
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
        tuple[str, str, str], dict[str, list[dict[str, float]]]
    ] = defaultdict(lambda: defaultdict(list))
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
                            "tagged_events",
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
                            "tagged_events",
                            fold,
                            model,
                            seed,
                        )
                        for seed in args.seeds
                    ]
                    missing = [path for path in calibration_paths + evaluation_paths if not path.is_file()]
                    if missing:
                        raise FileNotFoundError(missing[0])
                    calibration, calibration_metadata = load_ensemble(calibration_paths)
                    evaluation, evaluation_metadata = load_ensemble(evaluation_paths)
                    fold_reports.append(
                        {
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
                                str(path): sha256_file(path) for path in calibration_paths
                            },
                            "evaluation_bundle_sha256": {
                                str(path): sha256_file(path) for path in evaluation_paths
                            },
                        }
                    )
                    maps = participant_metric_maps(evaluation, evaluation_metadata)
                    target = participant_records[(heldout, boundary, model)]
                    for participant, metrics in maps.items():
                        target[participant].append(metrics)

    aggregate_models: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    for heldout in PROTOCOLS:
        paired[heldout] = {}
        for boundary in BOUNDARIES:
            collapsed: dict[str, dict[str, dict[str, float]]] = {}
            for model in args.models:
                collapsed[model] = collapse_repeated(
                    participant_records[(heldout, boundary, model)]
                )
                aggregate_models.append(
                    {
                        "heldout_protocol": heldout,
                        "source_protocol": next(
                            value for value in PROTOCOLS if value != heldout
                        ),
                        "boundary": boundary,
                        "model": model,
                        "participants": len(collapsed[model]),
                        **{
                            metric: float(
                                np.mean(
                                    [values[metric] for values in collapsed[model].values()]
                                )
                            )
                            for metric in next(iter(collapsed[model].values()))
                        },
                    }
                )
            if {"pk_ssm", "tcn"} <= set(collapsed):
                paired[heldout][boundary] = {
                    metric: paired_inference(
                        collapsed["pk_ssm"],
                        collapsed["tcn"],
                        metric,
                        args.bootstrap_replicates,
                        args.bootstrap_seed,
                    )
                    for metric in (
                        "trajectory_mae_bpm",
                        "mae_30s_bpm",
                        "mae_60s_bpm",
                        "mae_120s_bpm",
                    )
                }

    report = {
        "artifact_status": "locked_wearable_v4_activity_shift_summary",
        "models": args.models,
        "seeds": args.seeds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "primary_boundary": "joint_user_activity",
        "primary_origin_policy": "tagged_events",
        "activity_identity": "masked",
        "history_regime": "none",
        "aggregate_model_metrics": aggregate_models,
        "paired_inference": paired,
        "fold_reports": fold_reports,
    }
    report_path = args.output_dir / "activity_shift_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "SHA256SUMS.txt").write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n", encoding="ascii"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
