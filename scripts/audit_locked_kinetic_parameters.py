"""Audit identifiability and stability of locked PK-SSM kinetic parameters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


SEEDS = tuple(range(20260730, 20260735))
POLICIES = ("tagged_events", "evaluation_stride")
PARAMETER_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "rest_hr": (35.0, 120.0),
    "feasible_max_hr": (120.0, 220.0),
    "hr_reserve": (0.0, 185.0),
    "tau_fast_rise": (1.0, 120.0),
    "tau_slow_rise": (None, 600.0),
    "tau_fast_recovery": (1.0, 120.0),
    "tau_slow_recovery": (None, 600.0),
    "gain_fast": (0.0, None),
    "gain_slow": (0.0, None),
    "initial_fast_fraction": (0.0, 1.0),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=Path("outputs/locked_evaluation_v1"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/locked_kinetic_parameter_audit_v1")
    )
    return parser.parse_args()


def member_dir(root: Path, policy: str, fold: int, seed: int) -> Path:
    return (
        root
        / "test"
        / policy
        / f"outer_fold_{fold}"
        / "pk_ssm"
        / f"seed_{seed}"
    )


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def effective_group_size(group_sizes: np.ndarray) -> float:
    total = float(np.sum(group_sizes))
    groups = int(group_sizes.size)
    if groups <= 1 or total <= 0.0:
        return float("nan")
    return float((total - np.sum(np.square(group_sizes)) / total) / (groups - 1))


def participant_icc(values: np.ndarray, participants: np.ndarray) -> dict[str, float]:
    unique = np.unique(participants)
    group_values = [values[participants == participant] for participant in unique]
    group_sizes = np.asarray([group.size for group in group_values], dtype=np.float64)
    total = int(np.sum(group_sizes))
    groups = len(group_values)
    grand_mean = float(np.mean(values))
    ss_between = float(
        sum(group.size * np.square(float(np.mean(group)) - grand_mean) for group in group_values)
    )
    ss_within = float(
        sum(np.sum(np.square(group - float(np.mean(group)))) for group in group_values)
    )
    ms_between = ss_between / (groups - 1) if groups > 1 else float("nan")
    ms_within = ss_within / (total - groups) if total > groups else float("nan")
    effective_n = effective_group_size(group_sizes)
    denominator = ms_between + (effective_n - 1.0) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator > 0.0 else float("nan")
    participant_means = np.asarray([np.mean(group) for group in group_values], dtype=np.float64)
    return {
        "participants": float(groups),
        "origins": float(total),
        "icc_one_way_random": float(icc),
        "between_participant_sd_of_means": float(np.std(participant_means, ddof=1)),
        "pooled_within_participant_sd": float(np.sqrt(max(ms_within, 0.0))),
        "effective_group_size": float(effective_n),
    }


def boundary_fraction(values: np.ndarray, bounds: tuple[float | None, float | None]) -> dict[str, float]:
    lower, upper = bounds
    if lower is not None and upper is not None:
        tolerance = 0.01 * (upper - lower)
    else:
        tolerance = 0.01 * max(float(np.ptp(values)), 1.0)
    return {
        "near_lower_fraction": (
            float(np.mean(values <= lower + tolerance)) if lower is not None else float("nan")
        ),
        "near_upper_fraction": (
            float(np.mean(values >= upper - tolerance)) if upper is not None else float("nan")
        ),
    }


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    if np.sum(finite) < 3 or np.ptp(first[finite]) == 0.0 or np.ptp(second[finite]) == 0.0:
        return float("nan")
    return float(spearmanr(first[finite], second[finite]).statistic)


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

    summary_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        parameter_members: dict[str, list[np.ndarray]] = {}
        metadata_reference: list[dict[str, str]] = []
        current_hr_parts: list[np.ndarray] = []
        target_mean_parts: list[np.ndarray] = []
        target_delta_parts: list[np.ndarray] = []
        fold_parts: list[np.ndarray] = []

        seed_parameter_parts: dict[int, dict[str, list[np.ndarray]]] = {
            seed: {} for seed in SEEDS
        }
        for fold in range(5):
            fold_metadata_reference: list[dict[str, str]] | None = None
            for seed in SEEDS:
                directory = member_dir(args.bundle_root, policy, fold, seed)
                metadata = read_metadata(directory / "origin_metadata.csv")
                if fold_metadata_reference is None:
                    fold_metadata_reference = metadata
                elif metadata != fold_metadata_reference:
                    raise ValueError("Origin metadata differs across seeds.")
                with np.load(directory / "kinetic_parameters.npz", allow_pickle=False) as archive:
                    for name in archive.files:
                        seed_parameter_parts[seed].setdefault(name, []).append(
                            archive[name].astype(np.float64)
                        )
                if seed == SEEDS[0]:
                    with np.load(directory / "forecast_bundle.npz", allow_pickle=False) as bundle:
                        current_hr_parts.append(bundle["current_hr"].astype(np.float64))
                        target = bundle["target"].astype(np.float64)
                        valid = bundle["target_valid_mask"].astype(bool)
                        target_mean = np.asarray(
                            [np.mean(target[row][valid[row]]) for row in range(target.shape[0])],
                            dtype=np.float64,
                        )
                        target_end = np.asarray(
                            [target[row][np.flatnonzero(valid[row])[-1]] for row in range(target.shape[0])],
                            dtype=np.float64,
                        )
                        target_mean_parts.append(target_mean)
                        target_delta_parts.append(target_end - bundle["current_hr"].astype(np.float64))
                    metadata_reference.extend(metadata)
                    fold_parts.append(np.full(len(metadata), fold, dtype=np.int16))

        for seed in SEEDS:
            for name, parts in seed_parameter_parts[seed].items():
                parameter_members.setdefault(name, []).append(np.concatenate(parts))
        participants = np.asarray(
            [f"fold{fold}:{row['participant_id']}" for fold, row in zip(np.concatenate(fold_parts), metadata_reference)],
            dtype=str,
        )
        current_hr = np.concatenate(current_hr_parts)
        target_mean = np.concatenate(target_mean_parts)
        target_delta = np.concatenate(target_delta_parts)

        for name, member_arrays in sorted(parameter_members.items()):
            members = np.stack(member_arrays, axis=0)
            ensemble = np.mean(members, axis=0)
            icc = participant_icc(ensemble, participants)
            pairwise_seed_correlations = [
                safe_spearman(members[first], members[second])
                for first, second in combinations(range(len(SEEDS)), 2)
            ]
            bounds = PARAMETER_BOUNDS.get(name, (None, None))
            saturation = boundary_fraction(ensemble, bounds)
            summary_rows.append(
                {
                    "policy": policy,
                    "parameter": name,
                    "mean": float(np.mean(ensemble)),
                    "sd": float(np.std(ensemble, ddof=1)),
                    "p05": float(np.quantile(ensemble, 0.05)),
                    "p50": float(np.quantile(ensemble, 0.50)),
                    "p95": float(np.quantile(ensemble, 0.95)),
                    "minimum": float(np.min(ensemble)),
                    "maximum": float(np.max(ensemble)),
                    "median_within_origin_seed_sd": float(
                        np.median(np.std(members, axis=0, ddof=1))
                    ),
                    "mean_pairwise_seed_spearman": float(
                        np.nanmean(pairwise_seed_correlations)
                    ),
                    **icc,
                    **saturation,
                }
            )
            for covariate, values in (
                ("current_hr", current_hr),
                ("future_mean_hr", target_mean),
                ("future_end_minus_current_hr", target_delta),
            ):
                correlation_rows.append(
                    {
                        "policy": policy,
                        "parameter": name,
                        "covariate": covariate,
                        "spearman_rho": safe_spearman(ensemble, values),
                        "interpretation": "exploratory_noncausal",
                    }
                )
            for participant in np.unique(participants):
                selected = participants == participant
                participant_rows.append(
                    {
                        "policy": policy,
                        "parameter": name,
                        "participant_key": participant,
                        "origins": int(np.sum(selected)),
                        "mean": float(np.mean(ensemble[selected])),
                        "sd": float(np.std(ensemble[selected], ddof=1))
                        if np.sum(selected) > 1
                        else 0.0,
                    }
                )

    summary_path = args.output_root / "parameter_summary.csv"
    correlation_path = args.output_root / "parameter_correlations.csv"
    participant_path = args.output_root / "participant_parameter_summary.csv"
    write_csv(summary_path, summary_rows)
    write_csv(correlation_path, correlation_rows)
    write_csv(participant_path, participant_rows)
    overview = {
        "artifact_status": "locked_kinetic_parameter_audit",
        "interpretation": (
            "Parameters are model-internal context-conditioned latent states and are not "
            "laboratory-validated physiological measurements."
        ),
        "policies": POLICIES,
        "seeds": SEEDS,
        "parameters": sorted({row["parameter"] for row in summary_rows}),
    }
    overview_path = args.output_root / "summary.json"
    overview_path.write_text(json.dumps(overview, indent=2, sort_keys=True), encoding="utf-8")
    files = (summary_path, correlation_path, participant_path, overview_path)
    (args.output_root / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii"
    )
    print(json.dumps(overview, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
