"""Summarize locked GoldenCheetah test bundles before result inspection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


MODELS = (
    "pk_ssm",
    "tcn",
    "gru",
    "lstm",
    "transformer",
    "first_order_kinetics",
    "residual_ssm",
    "persistence",
    "damped_trend",
)
POLICIES = ("tagged_events", "evaluation_stride")
HORIZONS = (30, 60, 120)
SEED = 20260730
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260731


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-root", type=Path, default=Path("outputs/goldencheetah_locked_evaluation_v1")
    )
    parser.add_argument(
        "--score-root", type=Path, default=Path("outputs/goldencheetah_locked_scoring_v1")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/goldencheetah_locked_summary_v1")
    )
    return parser.parse_args()


def bundle_path(root: Path, policy: str, model: str) -> Path:
    return (
        root
        / "test"
        / policy
        / "outer_fold_0"
        / model
        / f"seed_{SEED}"
        / "forecast_bundle.npz"
    )


def load_bundle(path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    with np.load(path, allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    bundle["target_valid_mask"] = bundle["target_valid_mask"].astype(bool)
    with (path.parent / "origin_metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    return bundle, metadata


def participant_rows(
    bundle: dict[str, np.ndarray], metadata: list[dict[str, str]], model: str, policy: str
) -> list[dict[str, Any]]:
    mean = bundle["mean"].astype(np.float64)
    target = bundle["target"].astype(np.float64)
    valid = bundle["target_valid_mask"]
    participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)
    sexes = np.asarray([row["sex"] for row in metadata], dtype=str)
    absolute_error = np.abs(mean - target)
    signed_error = mean - target
    squared_error = np.square(signed_error)
    pair_valid = valid[:, 1:] & valid[:, :-1]
    observed_tv = np.sum(np.abs(np.diff(target, axis=1)) * pair_valid, axis=1)
    predicted_tv = np.sum(np.abs(np.diff(mean, axis=1)) * pair_valid, axis=1)
    tv_ratio = np.divide(
        predicted_tv,
        observed_tv,
        out=np.full_like(predicted_tv, np.nan),
        where=observed_tv > 1e-8,
    )
    rows: list[dict[str, Any]] = []
    for participant in np.unique(participants):
        selected = participants == participant
        selected_valid = valid[selected]
        row: dict[str, Any] = {
            "participant_id": participant,
            "model": model,
            "policy": policy,
            "sex": str(np.unique(sexes[selected])[0]),
            "trajectory_mae_bpm": float(np.mean(absolute_error[selected][selected_valid])),
            "trajectory_rmse_bpm": float(np.sqrt(np.mean(squared_error[selected][selected_valid]))),
            "signed_error_bpm": float(np.mean(signed_error[selected][selected_valid])),
            "total_variation_ratio": float(np.nanmean(tv_ratio[selected])),
        }
        for horizon in HORIZONS:
            index = horizon - 1
            horizon_valid = selected_valid[:, index]
            row[f"mae_{horizon}s_bpm"] = float(
                np.mean(absolute_error[selected, index][horizon_valid])
            )
        rows.append(row)
    return rows


def mean_field(rows: list[dict[str, Any]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return float(np.mean(values[np.isfinite(values)]))


def paired(first: list[dict[str, Any]], second: list[dict[str, Any]], field: str) -> dict[str, Any]:
    first_map = {row["participant_id"]: float(row[field]) for row in first}
    second_map = {row["participant_id"]: float(row[field]) for row in second}
    participants = sorted(set(first_map) & set(second_map))
    differences = np.asarray(
        [first_map[participant] - second_map[participant] for participant in participants],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(
        differences, size=(BOOTSTRAP_REPLICATES, differences.size), replace=True
    )
    bootstrap_means = np.mean(draws, axis=1)
    p_value = 1.0 if np.allclose(differences, 0.0) else float(wilcoxon(differences).pvalue)
    return {
        "participants": len(participants),
        "mean_pk_ssm_minus_tcn": float(np.mean(differences)),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "wilcoxon_p": p_value,
        "pk_ssm_better_fraction": float(np.mean(differences < 0.0)),
        "tcn_better_fraction": float(np.mean(differences > 0.0)),
    }


def group_summary(
    bundle: dict[str, np.ndarray], metadata: list[dict[str, str]], model: str, policy: str
) -> list[dict[str, Any]]:
    mean = bundle["mean"].astype(np.float64)
    target = bundle["target"].astype(np.float64)
    valid = bundle["target_valid_mask"]
    absolute_error = np.abs(mean - target)
    participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)
    result: list[dict[str, Any]] = []
    for field in ("protocol", "sex"):
        labels = np.asarray([row[field] for row in metadata], dtype=str)
        for label in sorted(np.unique(labels)):
            selected = labels == label
            participant_values: list[float] = []
            for participant in np.unique(participants[selected]):
                participant_selected = selected & (participants == participant)
                participant_valid = valid[participant_selected]
                participant_values.append(
                    float(np.mean(absolute_error[participant_selected][participant_valid]))
                )
            result.append(
                {
                    "policy": policy,
                    "model": model,
                    "field": field,
                    "label": label,
                    "origins": int(np.sum(selected)),
                    "participants": len(participant_values),
                    "participant_macro_mae_bpm": float(np.mean(participant_values)),
                    "inferential_status": (
                        "eligible" if len(participant_values) >= 5 else "descriptive_only"
                    ),
                }
            )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Immutable output directory is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    model_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    participant_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    uncertainty_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for model in MODELS:
            path = bundle_path(args.bundle_root, policy, model)
            bundle, metadata = load_bundle(path)
            rows = participant_rows(bundle, metadata, model, policy)
            participant_cache[(policy, model)] = rows
            export_report = json.loads((path.parent / "export_report.json").read_text(encoding="utf-8"))
            summary: dict[str, Any] = {
                "policy": policy,
                "model": model,
                "participants": len(rows),
                "trajectory_mae_bpm": mean_field(rows, "trajectory_mae_bpm"),
                "trajectory_rmse_bpm": mean_field(rows, "trajectory_rmse_bpm"),
                "signed_error_bpm": mean_field(rows, "signed_error_bpm"),
                "total_variation_ratio": mean_field(rows, "total_variation_ratio"),
                "parameters": int(export_report["trainable_parameters"]),
                "origins_per_second": float(export_report["runtime"]["origins_per_second"]),
            }
            for horizon in HORIZONS:
                summary[f"mae_{horizon}s_bpm"] = mean_field(rows, f"mae_{horizon}s_bpm")
            model_rows.append(summary)
            subgroup_rows.extend(group_summary(bundle, metadata, model, policy))

        for model in ("pk_ssm", "tcn"):
            report_path = (
                args.score_root
                / "test"
                / policy
                / "outer_fold_0"
                / model
                / f"seed_{SEED}"
                / "calibrated_evaluation_report.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            row: dict[str, Any] = {
                "policy": policy,
                "model": model,
                "raw_point_coverage": report["calibration"]["raw_student_t"][
                    "participant_macro_point_coverage"
                ],
                "raw_mean_width_bpm": report["calibration"]["raw_student_t"]["mean_width_bpm"],
                "origin_level_curve_coverage": report["calibration"]["simultaneous_120s"][
                    "origin_level"
                ]["participant_macro_curve_coverage"],
            }
            for horizon in HORIZONS:
                point = report["calibration"]["pointwise"][str(horizon)]
                row[f"conformal_coverage_{horizon}s"] = point[
                    "origin_level_participant_macro_coverage"
                ]
                row[f"conformal_width_{horizon}s_bpm"] = point["origin_level_mean_width_bpm"]
            uncertainty_rows.append(row)

    paired_results: dict[str, Any] = {}
    for policy in POLICIES:
        policy_results: dict[str, Any] = {}
        for field in ("trajectory_mae_bpm", "mae_30s_bpm", "mae_60s_bpm", "mae_120s_bpm"):
            policy_results[field] = paired(
                participant_cache[(policy, "pk_ssm")], participant_cache[(policy, "tcn")], field
            )
        paired_results[policy] = policy_results

    write_csv(args.output_root / "model_summary.csv", model_rows)
    write_csv(args.output_root / "subgroup_summary.csv", subgroup_rows)
    write_csv(args.output_root / "uncertainty_summary.csv", uncertainty_rows)
    paired_path = args.output_root / "paired_inference.json"
    paired_path.write_text(json.dumps(paired_results, indent=2, sort_keys=True), encoding="utf-8")
    overview = {
        "artifact_status": "locked_goldencheetah_test_summary",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "model_summary": model_rows,
        "paired_inference": paired_results,
        "uncertainty_summary": uncertainty_rows,
        "ridge_status": "pending_schema-compatible_locked_evaluation",
    }
    overview_path = args.output_root / "summary.json"
    overview_path.write_text(json.dumps(overview, indent=2, sort_keys=True), encoding="utf-8")
    files = sorted(path for path in args.output_root.iterdir() if path.is_file())
    (args.output_root / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii"
    )
    print(json.dumps(overview, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
