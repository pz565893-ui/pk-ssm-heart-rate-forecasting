"""Create prespecified participant-level summaries from locked v4 test bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import wilcoxon


SEEDS = tuple(range(20260730, 20260735))
PRIMARY_MODELS = ("pk_ssm", "tcn")
SECONDARY_MODELS = (
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
ALPHA = 0.05
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
    parser.add_argument("--bundle-root", type=Path, default=Path("outputs/locked_evaluation_v1"))
    parser.add_argument("--score-root", type=Path, default=Path("outputs/locked_scoring_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/locked_summary_v1"))
    return parser.parse_args()


def bundle_path(root: Path, policy: str, fold: int, model: str, seed: int) -> Path:
    return (
        root
        / "test"
        / policy
        / f"outer_fold_{fold}"
        / model
        / f"seed_{seed}"
        / "forecast_bundle.npz"
    )


def score_path(root: Path, policy: str, fold: int, model: str, member: str) -> Path:
    return (
        root
        / "test"
        / policy
        / f"outer_fold_{fold}"
        / model
        / member
        / "calibrated_evaluation_report.json"
    )


def load_metadata(path: Path) -> list[dict[str, str]]:
    with (path.parent / "origin_metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_bundle(path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    with np.load(path, allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    bundle["target_valid_mask"] = bundle["target_valid_mask"].astype(bool)
    return bundle, load_metadata(path)


def ensemble_bundle(paths: Iterable[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    loaded = [load_bundle(path) for path in paths]
    bundles = [item[0] for item in loaded]
    metadata = loaded[0][1]
    for bundle, member_metadata in loaded[1:]:
        if member_metadata != metadata:
            raise ValueError("Ensemble metadata mismatch.")
        for field in ("target", "target_valid_mask", "current_hr"):
            if not np.array_equal(bundle[field], bundles[0][field]):
                raise ValueError(f"Ensemble target mismatch: {field}")
    means = np.stack([bundle["mean"].astype(np.float64) for bundle in bundles], axis=0)
    scales = np.stack([bundle["scale"].astype(np.float64) for bundle in bundles], axis=0)
    mean = np.mean(means, axis=0)
    scale = np.sqrt(
        np.maximum(
            np.mean(np.square(scales) + np.square(means - mean[None, ...]), axis=0),
            1e-12,
        )
    )
    return (
        {
            "mean": mean,
            "scale": scale,
            "target": bundles[0]["target"].astype(np.float64),
            "target_valid_mask": bundles[0]["target_valid_mask"],
            "current_hr": bundles[0]["current_hr"].astype(np.float64),
        },
        metadata,
    )


def participant_rows(
    bundle: dict[str, np.ndarray],
    metadata: list[dict[str, str]],
    fold: int,
    model: str,
    member: str,
    policy: str,
    high_hr_threshold: float,
) -> list[dict[str, Any]]:
    mean = bundle["mean"].astype(np.float64)
    target = bundle["target"].astype(np.float64)
    valid = bundle["target_valid_mask"].astype(bool)
    participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)
    sexes = np.asarray([row["sex"] for row in metadata], dtype=str)
    absolute_error = np.abs(mean - target)
    signed_error = mean - target
    squared_error = np.square(signed_error)
    pair_valid = valid[:, 1:] & valid[:, :-1]
    observed_tv = np.sum(np.abs(np.diff(target, axis=1)) * pair_valid, axis=1)
    predicted_tv = np.sum(np.abs(np.diff(mean, axis=1)) * pair_valid, axis=1)
    origin_tv_ratio = np.divide(
        predicted_tv,
        observed_tv,
        out=np.full_like(predicted_tv, np.nan),
        where=observed_tv > 1e-8,
    )

    rapid_pair_valid = valid[:, 10:] & valid[:, :-10]
    observed_delta_10s = np.abs(target[:, 10:] - target[:, :-10])
    predicted_delta_10s = np.abs(mean[:, 10:] - mean[:, :-10])
    rapid_mask = rapid_pair_valid & (observed_delta_10s >= 5.0)

    rows: list[dict[str, Any]] = []
    for participant in np.unique(participants):
        selected = participants == participant
        selected_valid = valid[selected]
        high_mask = selected_valid & (target[selected] >= high_hr_threshold)
        fixed_high_mask = selected_valid & (target[selected] >= 160.0)
        selected_rapid = rapid_mask[selected]
        observed_rapid = observed_delta_10s[selected][selected_rapid]
        predicted_rapid = predicted_delta_10s[selected][selected_rapid]
        rapid_ratio = (
            float(np.mean(predicted_rapid) / np.mean(observed_rapid))
            if observed_rapid.size and np.mean(observed_rapid) > 1e-8
            else float("nan")
        )
        row: dict[str, Any] = {
            "participant_key": f"fold{fold}:{participant}",
            "participant_id": participant,
            "fold": fold,
            "model": model,
            "member": member,
            "policy": policy,
            "sex": str(np.unique(sexes[selected])[0]),
            "trajectory_mae_bpm": float(np.mean(absolute_error[selected][selected_valid])),
            "trajectory_rmse_bpm": float(np.sqrt(np.mean(squared_error[selected][selected_valid]))),
            "signed_error_bpm": float(np.mean(signed_error[selected][selected_valid])),
            "total_variation_ratio": float(np.nanmean(origin_tv_ratio[selected])),
            "rapid_change_amplitude_ratio": rapid_ratio,
            "rapid_change_pairs": int(np.sum(selected_rapid)),
            "high_hr_mae_bpm": (
                float(np.mean(absolute_error[selected][high_mask])) if np.any(high_mask) else float("nan")
            ),
            "high_hr_points": int(np.sum(high_mask)),
            "fixed_160_high_hr_mae_bpm": (
                float(np.mean(absolute_error[selected][fixed_high_mask]))
                if np.any(fixed_high_mask)
                else float("nan")
            ),
            "fixed_160_high_hr_points": int(np.sum(fixed_high_mask)),
        }
        for horizon in HORIZONS:
            index = horizon - 1
            horizon_valid = selected_valid[:, index]
            row[f"mae_{horizon}s_bpm"] = (
                float(np.mean(absolute_error[selected, index][horizon_valid]))
                if np.any(horizon_valid)
                else float("nan")
            )
        rows.append(row)
    return rows


def macro(rows: list[dict[str, Any]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def summarize_participants(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "participants": len(rows),
        "trajectory_mae_bpm": macro(rows, "trajectory_mae_bpm"),
        "trajectory_rmse_bpm": macro(rows, "trajectory_rmse_bpm"),
        "signed_error_bpm": macro(rows, "signed_error_bpm"),
        "total_variation_ratio": macro(rows, "total_variation_ratio"),
        "rapid_change_amplitude_ratio": macro(rows, "rapid_change_amplitude_ratio"),
        "high_hr_mae_bpm": macro(rows, "high_hr_mae_bpm"),
        "fixed_160_high_hr_mae_bpm": macro(rows, "fixed_160_high_hr_mae_bpm"),
    }
    for horizon in HORIZONS:
        result[f"mae_{horizon}s_bpm"] = macro(rows, f"mae_{horizon}s_bpm")
    return result


def paired_bootstrap(
    first_rows: list[dict[str, Any]], second_rows: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    first = {row["participant_key"]: float(row[field]) for row in first_rows}
    second = {row["participant_key"]: float(row[field]) for row in second_rows}
    keys = sorted(set(first) & set(second))
    differences = np.asarray([first[key] - second[key] for key in keys], dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(differences, size=(BOOTSTRAP_REPLICATES, differences.size), replace=True)
    bootstrap_means = np.mean(sampled, axis=1)
    if np.allclose(differences, 0.0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
    return {
        "participants": int(differences.size),
        "mean_paired_difference": float(np.mean(differences)),
        "median_paired_difference": float(np.median(differences)),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "wilcoxon_signed_rank_p": p_value,
        "first_better_fraction": float(np.mean(differences < 0.0)),
        "second_better_fraction": float(np.mean(differences > 0.0)),
        "ties_fraction": float(np.mean(differences == 0.0)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(
    bundle: dict[str, np.ndarray], metadata: list[dict[str, str]], fold: int, model: str, policy: str
) -> list[dict[str, Any]]:
    mean = bundle["mean"].astype(np.float64)
    target = bundle["target"].astype(np.float64)
    valid = bundle["target_valid_mask"].astype(bool)
    absolute_error = np.abs(mean - target)
    participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)
    result: list[dict[str, Any]] = []
    label_sets: dict[str, list[list[str]]] = {
        "sex": [[row["sex"]] for row in metadata],
        "protocol": [[row["protocol"]] for row in metadata],
        "event_type": [
            [item for item in row["event_types"].split(";") if item] or ["none"] for row in metadata
        ],
    }
    for field, row_labels in label_sets.items():
        labels = sorted({label for labels_for_row in row_labels for label in labels_for_row})
        for label in labels:
            selected = np.asarray([label in labels_for_row for labels_for_row in row_labels], dtype=bool)
            participant_values: list[float] = []
            for participant in np.unique(participants[selected]):
                participant_selected = selected & (participants == participant)
                participant_valid = valid[participant_selected]
                if np.any(participant_valid):
                    participant_values.append(
                        float(np.mean(absolute_error[participant_selected][participant_valid]))
                    )
            result.append(
                {
                    "policy": policy,
                    "fold": fold,
                    "model": model,
                    "field": field,
                    "label": label,
                    "origins": int(np.sum(selected)),
                    "participants": len(participant_values),
                    "participant_macro_mae_bpm": (
                        float(np.mean(participant_values)) if participant_values else float("nan")
                    ),
                    "inferential_status": "eligible" if len(participant_values) >= 5 else "descriptive_only",
                }
            )
    return result


def uncertainty_rows(
    bundle_root: Path, score_root: Path, policy: str, model: str
) -> list[dict[str, Any]]:
    participant_values: defaultdict[str, list[float]] = defaultdict(list)
    point_width_values: defaultdict[str, list[float]] = defaultdict(list)
    for fold in range(5):
        paths = [bundle_path(bundle_root, policy, fold, model, seed) for seed in SEEDS]
        bundle, metadata = ensemble_bundle(paths)
        report = json.loads(
            score_path(score_root, policy, fold, model, "ensemble_5seed").read_text(encoding="utf-8")
        )
        participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)
        mean = bundle["mean"]
        target = bundle["target"]
        valid = bundle["target_valid_mask"]
        scale = np.maximum(bundle["scale"], 1e-6)
        high_threshold = float(report["point_metrics"]["high_hr_threshold_bpm"])

        for interval_type, q_field in (
            ("origin_level", "origin_level_q"),
            ("participant_block", "participant_block_q"),
        ):
            for horizon in HORIZONS:
                index = horizon - 1
                q_value = float(report["calibration"]["pointwise"][str(horizon)][q_field])
                half_width = q_value * scale[:, index]
                covered = np.abs(target[:, index] - mean[:, index]) <= half_width
                for participant in np.unique(participants):
                    selected = (participants == participant) & valid[:, index]
                    if np.any(selected):
                        participant_values[f"{interval_type}_coverage_{horizon}s"].append(
                            float(np.mean(covered[selected]))
                        )
                        point_width_values[f"{interval_type}_width_{horizon}s"].append(
                            float(np.mean(2.0 * half_width[selected]))
                        )

            simultaneous_key = (
                "origin_level_q" if interval_type == "origin_level" else "participant_block_q"
            )
            q_curve = float(report["calibration"]["simultaneous_120s"][simultaneous_key])
            point_covered = np.abs(target - mean) <= q_curve * scale
            curve_covered = np.zeros(mean.shape[0], dtype=bool)
            for row in range(mean.shape[0]):
                curve_covered[row] = bool(np.all(point_covered[row][valid[row]])) if np.any(valid[row]) else False
            high_mask = valid & (target >= high_threshold)
            for participant in np.unique(participants):
                selected_rows = participants == participant
                if np.any(selected_rows):
                    participant_values[f"{interval_type}_curve_coverage"].append(
                        float(np.mean(curve_covered[selected_rows]))
                    )
                selected_high = high_mask[selected_rows]
                if np.any(selected_high):
                    participant_values[f"{interval_type}_high_hr_point_coverage"].append(
                        float(np.mean(point_covered[selected_rows][selected_high]))
                    )
    rows: list[dict[str, Any]] = []
    for interval_type in ("origin_level", "participant_block"):
        row: dict[str, Any] = {
            "policy": policy,
            "model": model,
            "interval_type": interval_type,
            "nominal_coverage": 1.0 - ALPHA,
            "simultaneous_curve_coverage": float(
                np.mean(participant_values[f"{interval_type}_curve_coverage"])
            ),
            "high_hr_point_coverage": float(
                np.mean(participant_values[f"{interval_type}_high_hr_point_coverage"])
            )
            if participant_values[f"{interval_type}_high_hr_point_coverage"]
            else float("nan"),
        }
        for horizon in HORIZONS:
            coverage = float(np.mean(participant_values[f"{interval_type}_coverage_{horizon}s"]))
            row[f"coverage_{horizon}s"] = coverage
            row[f"absolute_calibration_error_{horizon}s"] = abs(coverage - (1.0 - ALPHA))
            row[f"mean_width_{horizon}s_bpm"] = float(
                np.mean(point_width_values[f"{interval_type}_width_{horizon}s"])
            )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Immutable output directory is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    primary_seed_rows: list[dict[str, Any]] = []
    primary_ensemble_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    uncertainty_summary: list[dict[str, Any]] = []
    participant_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for policy in POLICIES:
        for model in PRIMARY_MODELS:
            for seed in SEEDS:
                all_rows: list[dict[str, Any]] = []
                for fold in range(5):
                    path = bundle_path(args.bundle_root, policy, fold, model, seed)
                    bundle, metadata = load_bundle(path)
                    score_report = json.loads(
                        score_path(args.score_root, policy, fold, model, f"seed_{seed}").read_text(
                            encoding="utf-8"
                        )
                    )
                    threshold = float(score_report["point_metrics"]["high_hr_threshold_bpm"])
                    all_rows.extend(
                        participant_rows(bundle, metadata, fold, model, f"seed_{seed}", policy, threshold)
                    )
                summary = summarize_participants(all_rows)
                primary_seed_rows.append(
                    {"policy": policy, "model": model, "seed": seed, **summary}
                )

            ensemble_rows_for_model: list[dict[str, Any]] = []
            runtime_values: list[float] = []
            parameters: list[int] = []
            for fold in range(5):
                paths = [bundle_path(args.bundle_root, policy, fold, model, seed) for seed in SEEDS]
                bundle, metadata = ensemble_bundle(paths)
                score_report = json.loads(
                    score_path(args.score_root, policy, fold, model, "ensemble_5seed").read_text(
                        encoding="utf-8"
                    )
                )
                threshold = float(score_report["point_metrics"]["high_hr_threshold_bpm"])
                ensemble_rows_for_model.extend(
                    participant_rows(bundle, metadata, fold, model, "ensemble_5seed", policy, threshold)
                )
                subgroup_rows.extend(group_rows(bundle, metadata, fold, model, policy))
                export_report = json.loads(
                    (paths[0].parent / "export_report.json").read_text(encoding="utf-8")
                )
                runtime_values.append(float(export_report["runtime"]["origins_per_second"]))
                parameters.append(int(export_report["trainable_parameters"]))
            participant_cache[(policy, model, "ensemble_5seed")] = ensemble_rows_for_model
            primary_ensemble_rows.append(
                {
                    "policy": policy,
                    "model": model,
                    **summarize_participants(ensemble_rows_for_model),
                    "parameters_per_member": int(np.median(parameters)),
                    "ensemble_parameters_total": int(np.median(parameters) * len(SEEDS)),
                    "median_member_origins_per_second": float(np.median(runtime_values)),
                }
            )
            uncertainty_summary.extend(
                uncertainty_rows(args.bundle_root, args.score_root, policy, model)
            )

        for model in SECONDARY_MODELS:
            rows_for_model: list[dict[str, Any]] = []
            runtime_values: list[float] = []
            parameters: list[int] = []
            for fold in range(5):
                path = bundle_path(args.bundle_root, policy, fold, model, SEEDS[0])
                bundle, metadata = load_bundle(path)
                if model in PRIMARY_MODELS:
                    score_report = json.loads(
                        score_path(args.score_root, policy, fold, model, f"seed_{SEEDS[0]}").read_text(
                            encoding="utf-8"
                        )
                    )
                    threshold = float(score_report["point_metrics"]["high_hr_threshold_bpm"])
                else:
                    primary_report = json.loads(
                        score_path(args.score_root, policy, fold, "tcn", "ensemble_5seed").read_text(
                            encoding="utf-8"
                        )
                    )
                    threshold = float(primary_report["point_metrics"]["high_hr_threshold_bpm"])
                rows_for_model.extend(
                    participant_rows(bundle, metadata, fold, model, f"seed_{SEEDS[0]}", policy, threshold)
                )
                export_report = json.loads((path.parent / "export_report.json").read_text(encoding="utf-8"))
                runtime_values.append(float(export_report["runtime"]["origins_per_second"]))
                parameters.append(int(export_report["trainable_parameters"]))
            secondary_rows.append(
                {
                    "policy": policy,
                    "model": model,
                    "seed": SEEDS[0],
                    **summarize_participants(rows_for_model),
                    "parameters": int(np.median(parameters)),
                    "median_origins_per_second": float(np.median(runtime_values)),
                }
            )

    paired_inference: dict[str, Any] = {}
    for policy in POLICIES:
        pk_rows = participant_cache[(policy, "pk_ssm", "ensemble_5seed")]
        tcn_rows = participant_cache[(policy, "tcn", "ensemble_5seed")]
        policy_results: dict[str, Any] = {}
        for field in ("trajectory_mae_bpm", "mae_30s_bpm", "mae_60s_bpm", "mae_120s_bpm"):
            policy_results[field] = paired_bootstrap(pk_rows, tcn_rows, field)
        paired_inference[policy] = policy_results

    seed_difference_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for seed in SEEDS:
            pk = next(
                row for row in primary_seed_rows if row["policy"] == policy and row["model"] == "pk_ssm" and row["seed"] == seed
            )
            tcn = next(
                row for row in primary_seed_rows if row["policy"] == policy and row["model"] == "tcn" and row["seed"] == seed
            )
            seed_difference_rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "pk_ssm_mae_bpm": pk["trajectory_mae_bpm"],
                    "tcn_mae_bpm": tcn["trajectory_mae_bpm"],
                    "pk_ssm_minus_tcn_bpm": pk["trajectory_mae_bpm"] - tcn["trajectory_mae_bpm"],
                }
            )

    write_csv(args.output_root / "primary_seed_summary.csv", primary_seed_rows)
    write_csv(args.output_root / "primary_seed_differences.csv", seed_difference_rows)
    write_csv(args.output_root / "primary_ensemble_summary.csv", primary_ensemble_rows)
    write_csv(args.output_root / "secondary_fixed_seed_summary.csv", secondary_rows)
    write_csv(args.output_root / "uncertainty_summary.csv", uncertainty_summary)
    write_csv(args.output_root / "subgroup_fold_summary.csv", subgroup_rows)

    paired_path = args.output_root / "paired_participant_inference.json"
    paired_path.write_text(json.dumps(paired_inference, indent=2, sort_keys=True), encoding="utf-8")
    overview = {
        "artifact_status": "locked_v4_test_summary",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "policies": POLICIES,
        "seeds": SEEDS,
        "primary_ensemble_summary": primary_ensemble_rows,
        "paired_inference": paired_inference,
        "ridge_status": "pending_frozen_summary_feature_test_evaluator",
    }
    overview_path = args.output_root / "summary.json"
    overview_path.write_text(json.dumps(overview, indent=2, sort_keys=True), encoding="utf-8")

    output_files = sorted(path for path in args.output_root.iterdir() if path.is_file())
    checksum_path = args.output_root / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in output_files),
        encoding="ascii",
    )
    print(json.dumps(overview, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
