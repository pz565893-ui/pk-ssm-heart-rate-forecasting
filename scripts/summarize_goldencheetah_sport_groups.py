"""Create locked GoldenCheetah sport-label summaries from frozen test bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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
        "--fold-cache-dir",
        type=Path,
        default=Path("data/processed/goldencheetah_transition_v1/outer_fold_0"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/goldencheetah_sport_summary_v1")
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


def load_origin_sports(path: Path) -> tuple[dict[str, str], set[str], set[str]]:
    mapping: dict[str, str] = {}
    train_sports: set[str] = set()
    test_sports: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sport = row["protocol_version"]
            if row["role"] == "train":
                train_sports.add(sport)
            elif row["role"] == "test":
                test_sports.add(sport)
                mapping[row["origin_id"]] = sport
    return mapping, train_sports, test_sports


def load_bundle(path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    with np.load(path, allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    bundle["target_valid_mask"] = bundle["target_valid_mask"].astype(bool)
    with (path.parent / "origin_metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    return bundle, metadata


def participant_sport_values(
    bundle: dict[str, np.ndarray],
    metadata: list[dict[str, str]],
    sport_mapping: dict[str, str],
) -> dict[str, dict[str, float]]:
    prediction = bundle["mean"].astype(np.float64)
    target = bundle["target"].astype(np.float64)
    valid = bundle["target_valid_mask"]
    absolute_error = np.abs(prediction - target)
    grouped_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for index, row in enumerate(metadata):
        origin_id = row["origin_id"]
        if origin_id not in sport_mapping:
            raise KeyError(f"No frozen sport label for origin: {origin_id}")
        sport = sport_mapping[origin_id]
        participant = row["participant_id"]
        grouped_values[(sport, participant)].extend(absolute_error[index][valid[index]].tolist())
    result: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for (sport, participant), values in grouped_values.items():
        result[sport][participant] = float(np.mean(values))
    return dict(result)


def paired_summary(
    pk_values: dict[str, float], tcn_values: dict[str, float]
) -> dict[str, Any]:
    participants = sorted(set(pk_values) & set(tcn_values))
    differences = np.asarray(
        [pk_values[participant] - tcn_values[participant] for participant in participants],
        dtype=np.float64,
    )
    if not differences.size:
        return {"participants": 0, "status": "not_estimable"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(differences, size=(BOOTSTRAP_REPLICATES, differences.size), replace=True)
    bootstrap_means = np.mean(draws, axis=1)
    return {
        "participants": int(differences.size),
        "mean_pk_ssm_minus_tcn_bpm": float(np.mean(differences)),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "pk_ssm_better_fraction": float(np.mean(differences < 0.0)),
        "tcn_better_fraction": float(np.mean(differences > 0.0)),
        "inferential_status": "eligible" if differences.size >= 5 else "descriptive_only",
    }


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

    sport_mapping, train_sports, test_sports = load_origin_sports(
        args.fold_cache_dir / "origin_manifest.csv"
    )
    summary_rows: list[dict[str, Any]] = []
    values_cache: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for policy in POLICIES:
        for model in MODELS:
            bundle, metadata = load_bundle(bundle_path(args.bundle_root, policy, model))
            values = participant_sport_values(bundle, metadata, sport_mapping)
            values_cache[(policy, model)] = values
            sport_counts: defaultdict[str, int] = defaultdict(int)
            for row in metadata:
                sport_counts[sport_mapping[row["origin_id"]]] += 1
            for sport in sorted(values):
                participant_values = np.asarray(list(values[sport].values()), dtype=np.float64)
                summary_rows.append(
                    {
                        "policy": policy,
                        "model": model,
                        "sport": sport,
                        "seen_in_training": sport in train_sports,
                        "origins": sport_counts[sport],
                        "participants": int(participant_values.size),
                        "participant_macro_mae_bpm": float(np.mean(participant_values)),
                        "inferential_status": (
                            "eligible" if participant_values.size >= 5 else "descriptive_only"
                        ),
                    }
                )

    paired: dict[str, Any] = {}
    for policy in POLICIES:
        policy_result: dict[str, Any] = {}
        pk = values_cache[(policy, "pk_ssm")]
        tcn = values_cache[(policy, "tcn")]
        for sport in sorted(set(pk) | set(tcn)):
            policy_result[sport] = paired_summary(pk.get(sport, {}), tcn.get(sport, {}))
        paired[policy] = policy_result

    summary_path = args.output_root / "sport_model_summary.csv"
    write_csv(summary_path, summary_rows)
    paired_path = args.output_root / "sport_paired_inference.json"
    paired_path.write_text(json.dumps(paired, indent=2, sort_keys=True), encoding="utf-8")
    overview = {
        "artifact_status": "locked_goldencheetah_sport_summary",
        "train_sports": sorted(train_sports),
        "test_sports": sorted(test_sports),
        "strictly_unseen_test_sports": sorted(test_sports - train_sports),
        "paired_inference": paired,
    }
    overview_path = args.output_root / "summary.json"
    overview_path.write_text(json.dumps(overview, indent=2, sort_keys=True), encoding="utf-8")
    files = (summary_path, paired_path, overview_path)
    (args.output_root / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii"
    )
    print(json.dumps(overview, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
