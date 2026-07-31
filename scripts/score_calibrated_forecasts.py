"""Apply frozen split-conformal rules to immutable forecast bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t


HORIZON_SECONDS = (30, 60, 120)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--evaluation-bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--fold-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def load_metadata(bundle_path: Path) -> list[dict[str, str]]:
    path = bundle_path.parent / "origin_metadata.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_ensemble(paths: list[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    bundles: list[dict[str, np.ndarray]] = []
    metadata_reference: list[dict[str, str]] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            bundle = {name: archive[name] for name in archive.files}
        metadata = load_metadata(path)
        if metadata_reference is None:
            metadata_reference = metadata
        elif metadata != metadata_reference:
            raise ValueError("Bundle metadata differs across ensemble members.")
        if bundles:
            for key in ("target", "target_valid_mask", "current_hr"):
                if not np.array_equal(bundle[key], bundles[0][key]):
                    raise ValueError(f"Bundle field differs across ensemble members: {key}")
        bundles.append(bundle)
    if not bundles or metadata_reference is None:
        raise ValueError("At least one bundle is required.")

    member_means = np.stack([bundle["mean"] for bundle in bundles], axis=0).astype(np.float64)
    member_scales = np.stack([bundle["scale"] for bundle in bundles], axis=0).astype(np.float64)
    mean = np.mean(member_means, axis=0)
    scale = np.sqrt(
        np.maximum(
            np.mean(np.square(member_scales) + np.square(member_means - mean[None, ...]), axis=0),
            1e-12,
        )
    )
    degrees_of_freedom = np.min(
        np.stack([bundle["degrees_of_freedom"] for bundle in bundles], axis=0), axis=0
    ).astype(np.float64)
    combined = {
        "mean": mean,
        "scale": scale,
        "degrees_of_freedom": degrees_of_freedom,
        "target": bundles[0]["target"].astype(np.float64),
        "target_valid_mask": bundles[0]["target_valid_mask"].astype(bool),
        "current_hr": bundles[0]["current_hr"].astype(np.float64),
    }
    return combined, metadata_reference


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite conformal scores are available.")
    rank = int(np.ceil((finite.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), finite.size)
    return float(np.partition(finite, rank - 1)[rank - 1])


def participant_macro(values: np.ndarray, valid: np.ndarray, participants: np.ndarray) -> float:
    summaries: list[float] = []
    for participant in np.unique(participants):
        rows = participants == participant
        selected = values[rows]
        selected_valid = valid[rows]
        if np.any(selected_valid):
            summaries.append(float(np.mean(selected[selected_valid])))
    return float(np.mean(summaries)) if summaries else float("nan")


def high_hr_threshold(fold_cache_dir: Path) -> float:
    values: list[np.ndarray] = []
    for session_path in sorted((fold_cache_dir / "sessions").glob("*.npz")):
        with np.load(session_path, allow_pickle=False) as archive:
            role = str(archive["role"].item())
            if role not in {"train", "calibration"}:
                continue
            valid = archive["hr_range_valid"].astype(bool) & archive["hr_offline_qc_valid"].astype(bool)
            if np.any(valid):
                values.append(archive["hr_bpm"][valid].astype(np.float64))
    if not values:
        raise ValueError("No valid training or calibration HR values were found.")
    return float(np.quantile(np.concatenate(values), 0.90))


def point_metrics(
    bundle: dict[str, np.ndarray], metadata: list[dict[str, str]], high_threshold: float
) -> dict[str, Any]:
    mean = bundle["mean"]
    target = bundle["target"]
    valid = bundle["target_valid_mask"]
    participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)
    error = mean - target
    absolute_error = np.abs(error)
    squared_error = np.square(error)
    result: dict[str, Any] = {
        "participant_macro_trajectory_mae_bpm": participant_macro(
            absolute_error, valid, participants
        ),
        "participant_macro_signed_error_bpm": participant_macro(error, valid, participants),
    }
    participant_rmse: list[float] = []
    for participant in np.unique(participants):
        rows = participants == participant
        selected_valid = valid[rows]
        if np.any(selected_valid):
            participant_rmse.append(float(np.sqrt(np.mean(squared_error[rows][selected_valid]))))
    result["participant_macro_trajectory_rmse_bpm"] = float(np.mean(participant_rmse))
    for horizon in HORIZON_SECONDS:
        index = horizon - 1
        result[f"participant_macro_mae_{horizon}s_bpm"] = participant_macro(
            absolute_error[:, index], valid[:, index], participants
        )

    high_mask = valid & (target >= high_threshold)
    fixed_high_mask = valid & (target >= 160.0)
    result["high_hr_threshold_bpm"] = high_threshold
    result["high_hr_valid_points"] = int(np.sum(high_mask))
    result["participant_macro_high_hr_mae_bpm"] = participant_macro(
        absolute_error, high_mask, participants
    )
    result["fixed_160_high_hr_valid_points"] = int(np.sum(fixed_high_mask))
    result["participant_macro_fixed_160_high_hr_mae_bpm"] = participant_macro(
        absolute_error, fixed_high_mask, participants
    )

    pair_valid = valid[:, 1:] & valid[:, :-1]
    observed_tv = np.sum(np.abs(np.diff(target, axis=1)) * pair_valid, axis=1)
    predicted_tv = np.sum(np.abs(np.diff(mean, axis=1)) * pair_valid, axis=1)
    ratios = np.divide(
        predicted_tv,
        observed_tv,
        out=np.full_like(predicted_tv, np.nan),
        where=observed_tv > 1e-8,
    )
    result["participant_macro_total_variation_ratio"] = participant_macro(
        ratios, np.isfinite(ratios), participants
    )
    return result


def group_mae(
    bundle: dict[str, np.ndarray], metadata: list[dict[str, str]], field: str
) -> list[dict[str, Any]]:
    participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)
    labels = np.asarray([row[field] for row in metadata], dtype=str)
    absolute_error = np.abs(bundle["mean"] - bundle["target"])
    valid = bundle["target_valid_mask"]
    rows: list[dict[str, Any]] = []
    for label in sorted(np.unique(labels)):
        selected = labels == label
        group_participants = np.unique(participants[selected])
        rows.append(
            {
                "field": field,
                "label": str(label),
                "origins": int(np.sum(selected)),
                "participants": int(group_participants.size),
                "participant_macro_mae_bpm": participant_macro(
                    absolute_error[selected], valid[selected], participants[selected]
                ),
                "inferential_status": "eligible" if group_participants.size >= 5 else "descriptive_only",
            }
        )
    return rows


def calibrated_intervals(
    calibration: dict[str, np.ndarray],
    calibration_metadata: list[dict[str, str]],
    evaluation: dict[str, np.ndarray],
    evaluation_metadata: list[dict[str, str]],
    alpha: float,
    high_threshold: float,
) -> dict[str, Any]:
    cal_mean = calibration["mean"]
    cal_scale = np.maximum(calibration["scale"], 1e-6)
    cal_target = calibration["target"]
    cal_valid = calibration["target_valid_mask"]
    cal_participants = np.asarray(
        [row["participant_id"] for row in calibration_metadata], dtype=str
    )

    eval_mean = evaluation["mean"]
    eval_scale = np.maximum(evaluation["scale"], 1e-6)
    eval_target = evaluation["target"]
    eval_valid = evaluation["target_valid_mask"]
    eval_participants = np.asarray(
        [row["participant_id"] for row in evaluation_metadata], dtype=str
    )

    result: dict[str, Any] = {"nominal_coverage": 1.0 - alpha, "pointwise": {}}
    normalized = np.abs(cal_target - cal_mean) / cal_scale
    for horizon in HORIZON_SECONDS:
        index = horizon - 1
        valid_scores = normalized[:, index][cal_valid[:, index]]
        origin_q = conformal_quantile(valid_scores, alpha)
        participant_scores: list[float] = []
        for participant in np.unique(cal_participants):
            rows = cal_participants == participant
            selected_valid = cal_valid[rows, index]
            if np.any(selected_valid):
                participant_scores.append(
                    float(np.max(normalized[rows, index][selected_valid]))
                )
        participant_q = conformal_quantile(np.asarray(participant_scores), alpha)

        eval_half_width = origin_q * eval_scale[:, index]
        eval_covered = np.abs(eval_target[:, index] - eval_mean[:, index]) <= eval_half_width
        eval_horizon_valid = eval_valid[:, index]
        participant_half_width = participant_q * eval_scale[:, index]
        participant_covered = (
            np.abs(eval_target[:, index] - eval_mean[:, index]) <= participant_half_width
        )
        result["pointwise"][str(horizon)] = {
            "origin_level_q": origin_q,
            "participant_block_q": participant_q,
            "calibration_origin_scores": int(valid_scores.size),
            "calibration_participants": int(len(participant_scores)),
            "origin_level_participant_macro_coverage": participant_macro(
                eval_covered.astype(np.float64), eval_horizon_valid, eval_participants
            ),
            "origin_level_mean_width_bpm": float(
                np.mean((2.0 * eval_half_width)[eval_horizon_valid])
            ),
            "participant_block_participant_macro_coverage": participant_macro(
                participant_covered.astype(np.float64), eval_horizon_valid, eval_participants
            ),
            "participant_block_mean_width_bpm": float(
                np.mean((2.0 * participant_half_width)[eval_horizon_valid])
            ),
        }

    per_origin_scores = np.full(cal_mean.shape[0], np.nan, dtype=np.float64)
    for row in range(cal_mean.shape[0]):
        if np.any(cal_valid[row]):
            per_origin_scores[row] = float(np.max(normalized[row][cal_valid[row]]))
    simultaneous_q = conformal_quantile(per_origin_scores, alpha)
    participant_maxima: list[float] = []
    for participant in np.unique(cal_participants):
        selected = per_origin_scores[cal_participants == participant]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            participant_maxima.append(float(np.max(selected)))
    participant_simultaneous_q = conformal_quantile(np.asarray(participant_maxima), alpha)

    def simultaneous_summary(q_value: float) -> dict[str, Any]:
        covered_origins = np.zeros(eval_mean.shape[0], dtype=np.float64)
        widths: list[float] = []
        for row in range(eval_mean.shape[0]):
            row_valid = eval_valid[row]
            if not np.any(row_valid):
                covered_origins[row] = np.nan
                continue
            half_width = q_value * eval_scale[row]
            covered_origins[row] = float(
                np.all(np.abs(eval_target[row][row_valid] - eval_mean[row][row_valid]) <= half_width[row_valid])
            )
            widths.extend((2.0 * half_width[row_valid]).tolist())
        origin_valid = np.isfinite(covered_origins)
        return {
            "participant_macro_curve_coverage": participant_macro(
                covered_origins, origin_valid, eval_participants
            ),
            "mean_pointwise_width_bpm": float(np.mean(widths)),
        }

    result["simultaneous_120s"] = {
        "origin_level_q": simultaneous_q,
        "participant_block_q": participant_simultaneous_q,
        "calibration_origins": int(np.sum(np.isfinite(per_origin_scores))),
        "calibration_participants": int(len(participant_maxima)),
        "origin_level": simultaneous_summary(simultaneous_q),
        "participant_block": simultaneous_summary(participant_simultaneous_q),
    }

    df = np.maximum(evaluation["degrees_of_freedom"], 2.100001)
    raw_half_width = student_t.ppf(1.0 - alpha / 2.0, df) * eval_scale
    raw_covered = np.abs(eval_target - eval_mean) <= raw_half_width
    high_mask = eval_valid & (eval_target >= high_threshold)
    result["raw_student_t"] = {
        "participant_macro_point_coverage": participant_macro(
            raw_covered.astype(np.float64), eval_valid, eval_participants
        ),
        "mean_width_bpm": float(np.mean((2.0 * raw_half_width)[eval_valid])),
        "participant_macro_high_hr_coverage": participant_macro(
            raw_covered.astype(np.float64), high_mask, eval_participants
        ),
        "high_hr_valid_points": int(np.sum(high_mask)),
    }
    return result


def main() -> None:
    args = parse_args()
    if len(args.calibration_bundles) != len(args.evaluation_bundles):
        raise ValueError("Calibration and evaluation ensemble sizes must match.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between zero and one.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Immutable output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration, calibration_metadata = load_ensemble(args.calibration_bundles)
    evaluation, evaluation_metadata = load_ensemble(args.evaluation_bundles)
    threshold = high_hr_threshold(args.fold_cache_dir)

    report = {
        "artifact_status": "locked_calibrated_evaluation",
        "label": args.label,
        "alpha": args.alpha,
        "ensemble_members": len(args.calibration_bundles),
        "calibration_bundle_sha256": {
            str(path): sha256_file(path) for path in args.calibration_bundles
        },
        "evaluation_bundle_sha256": {
            str(path): sha256_file(path) for path in args.evaluation_bundles
        },
        "point_metrics": point_metrics(evaluation, evaluation_metadata, threshold),
        "calibration": calibrated_intervals(
            calibration,
            calibration_metadata,
            evaluation,
            evaluation_metadata,
            args.alpha,
            threshold,
        ),
        "groups": (
            group_mae(evaluation, evaluation_metadata, "protocol")
            + group_mae(evaluation, evaluation_metadata, "sex")
            + group_mae(evaluation, evaluation_metadata, "event_types")
        ),
    }
    report_path = args.output_dir / "calibrated_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    checksum_path = args.output_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n", encoding="ascii"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
