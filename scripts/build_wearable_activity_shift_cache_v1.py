#!/usr/bin/env python3
"""Derive source-only Wearable v4 caches for bidirectional activity shifts.

The script never edits the frozen v4 cache.  It hard-links session arrays,
refits normalization on source-protocol training sessions only, and rewrites
origin roles for three deployment boundaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOLS = ("AEROBIC", "ANAEROBIC")
BOUNDARIES = ("source_user", "seen_user_activity", "joint_user_activity")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader), list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty manifest: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Source-only normalization received no finite values")
    return {
        "mean": float(np.mean(finite)),
        "std": float(max(np.std(finite), 1.0e-6)),
        "median": float(np.median(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q99": float(np.quantile(finite, 0.99)),
    }


def source_normalization(session_dir: Path, source_protocol: str) -> dict[str, Any]:
    heart_rate: list[np.ndarray] = []
    acceleration: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    source_sessions = 0
    for path in sorted(session_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            protocol = str(archive["protocol"].item()).upper()
            role = str(archive["role"].item()).lower()
            if protocol != source_protocol or role != "train":
                continue
            names = tuple(str(value) for value in archive["acc_feature_names"].tolist())
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError("Acceleration feature schema differs across sessions")
            hr_valid = archive["hr_range_valid"].astype(bool)
            acc_valid = archive["acc_valid"].astype(bool)
            if np.any(hr_valid):
                heart_rate.append(archive["hr_bpm"][hr_valid].astype(np.float64))
            if np.any(acc_valid):
                acceleration.append(archive["acc_features"][acc_valid].astype(np.float64))
            source_sessions += 1
    if not heart_rate or not acceleration or feature_names is None:
        raise ValueError(
            f"No source training signals for protocol {source_protocol} in {session_dir}"
        )
    hr = np.concatenate(heart_rate)
    acc = np.concatenate(acceleration, axis=0)
    return {
        "fit_scope": "source-protocol training participants only",
        "source_protocol": source_protocol,
        "source_training_sessions": source_sessions,
        "hr_bpm": distribution(hr),
        "acc_features": {
            name: distribution(acc[:, index])
            for index, name in enumerate(feature_names)
        },
    }


def mapped_role(protocol: str, base_role: str, heldout: str, boundary: str) -> str | None:
    source = next(value for value in PROTOCOLS if value != heldout)
    if protocol == source and base_role in {"train", "validation", "calibration"}:
        return base_role
    if boundary == "source_user" and protocol == source and base_role == "test":
        return "test"
    if boundary == "seen_user_activity" and protocol == heldout and base_role == "train":
        return "test"
    if boundary == "joint_user_activity" and protocol == heldout and base_role == "test":
        return "test"
    return None


def role_participants(rows: list[dict[str, str]], role: str) -> set[str]:
    return {row["participant_id"] for row in rows if row["role"] == role}


def derive_rows(
    rows: list[dict[str, str]], heldout: str, boundary: str
) -> list[dict[str, str]]:
    source = next(value for value in PROTOCOLS if value != heldout)
    output: list[dict[str, str]] = []
    for row in rows:
        base_role = row["role"].lower()
        protocol = row["protocol"].upper()
        role = mapped_role(protocol, base_role, heldout, boundary)
        if role is None:
            continue
        derived = dict(row)
        derived["role"] = role
        derived["protocol_version"] = (
            f"activity_shift_v1:{source}_to_{heldout}:{boundary}"
        )
        derived["base_role"] = base_role
        derived["source_protocol"] = source
        derived["heldout_protocol"] = heldout
        derived["activity_shift_boundary"] = boundary
        output.append(derived)
    return output


def acceptance_checks(rows: list[dict[str, str]], heldout: str, boundary: str) -> dict[str, bool]:
    source = next(value for value in PROTOCOLS if value != heldout)
    train_rows = [row for row in rows if row["role"] == "train"]
    selection_rows = [row for row in rows if row["role"] in {"validation", "calibration"}]
    test_rows = [row for row in rows if row["role"] == "test"]
    train_participants = role_participants(rows, "train")
    test_participants = role_participants(rows, "test")
    expected_test_protocol = source if boundary == "source_user" else heldout
    checks = {
        "training_contains_source_only": bool(train_rows)
        and {row["protocol"] for row in train_rows} == {source},
        "selection_contains_source_only": bool(selection_rows)
        and {row["protocol"] for row in selection_rows} == {source},
        "test_contains_expected_protocol_only": bool(test_rows)
        and {row["protocol"] for row in test_rows} == {expected_test_protocol},
        "train_test_sessions_disjoint": not (
            {row["session_id"] for row in train_rows}
            & {row["session_id"] for row in test_rows}
        ),
        "joint_users_disjoint": (
            not (train_participants & test_participants)
            if boundary in {"joint_user_activity", "source_user"}
            else True
        ),
        "seen_user_test_is_subset_of_training_users": (
            test_participants <= train_participants
            if boundary == "seen_user_activity"
            else True
        ),
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-cache-root",
        type=Path,
        default=Path("data/processed/wearable_exercise_transition_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/wearable_activity_shift_v1"),
    )
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    root_summary: dict[str, Any] = {
        "artifact_status": "frozen_wearable_v4_activity_shift_cache",
        "contract_version": "1.0.0",
        "history_regime": "none",
        "activity_identity": "masked for source and target",
        "directions": {},
    }
    for heldout in PROTOCOLS:
        source = next(value for value in PROTOCOLS if value != heldout)
        direction = f"{source}_to_{heldout}"
        direction_summary: dict[str, Any] = {}
        for fold in range(5):
            base_dir = args.base_cache_root / f"outer_fold_{fold}"
            base_policy_path = base_dir / "cache_policy.json"
            base_manifest_path = base_dir / "origin_manifest.csv"
            if not base_policy_path.is_file() or not base_manifest_path.is_file():
                raise FileNotFoundError(f"Incomplete base cache: {base_dir}")
            base_policy = json.loads(base_policy_path.read_text(encoding="utf-8"))
            base_rows, base_fields = read_csv(base_manifest_path)
            normalization = source_normalization(base_dir / "sessions", source)
            fold_summary: dict[str, Any] = {}
            for boundary in BOUNDARIES:
                destination = (
                    args.output_root / heldout / boundary / f"outer_fold_{fold}"
                )
                destination.mkdir(parents=True, exist_ok=False)
                session_destination = destination / "sessions"
                session_destination.mkdir()
                for source_path in sorted((base_dir / "sessions").glob("*.npz")):
                    os.link(source_path, session_destination / source_path.name)

                rows = derive_rows(base_rows, heldout, boundary)
                checks = acceptance_checks(rows, heldout, boundary)
                if not all(checks.values()):
                    raise RuntimeError(
                        f"Activity-shift acceptance failure for {direction}, fold {fold}, "
                        f"{boundary}: {checks}"
                    )
                fields = base_fields + [
                    "base_role",
                    "source_protocol",
                    "heldout_protocol",
                    "activity_shift_boundary",
                ]
                manifest_path = destination / "origin_manifest.csv"
                write_csv(manifest_path, rows, fields)
                normalization_path = destination / "training_normalization.json"
                normalization_path.write_text(
                    json.dumps(normalization, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                role_counts = Counter(row["role"] for row in rows)
                policy = {
                    "artifact_status": "derived_wearable_v4_activity_shift_cache",
                    "contract_version": "1.0.0",
                    "outer_fold": fold,
                    "source_protocol": source,
                    "heldout_protocol": heldout,
                    "activity_shift_boundary": boundary,
                    "split_manifest": (
                        "splits/wearable_exercise_v4/outer_fold_roles.csv::"
                        f"activity_shift_v1::{direction}::{boundary}"
                    ),
                    "preprocessing_config": base_policy["preprocessing_config"],
                    "history_regime": "none",
                    "activity_identity": "masked",
                    "normalization_scope": "source protocol and base training role only",
                    "base_cache_policy_sha256": sha256_file(base_policy_path),
                    "base_origin_manifest_sha256": sha256_file(base_manifest_path),
                    "origin_rows": len(rows),
                    "role_origin_rows": dict(sorted(role_counts.items())),
                    "role_participants": {
                        role: len(role_participants(rows, role))
                        for role in ("train", "validation", "calibration", "test")
                    },
                    "acceptance_checks": checks,
                    "session_storage": "hard links to immutable v4 session arrays",
                }
                policy_path = destination / "cache_policy.json"
                policy_path.write_text(
                    json.dumps(policy, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                checksum_path = destination / "SHA256SUMS.txt"
                checksum_path.write_text(
                    "".join(
                        f"{sha256_file(path)}  {path.name}\n"
                        for path in (manifest_path, normalization_path, policy_path)
                    ),
                    encoding="ascii",
                )
                fold_summary[boundary] = policy
            direction_summary[str(fold)] = fold_summary
        root_summary["directions"][direction] = direction_summary
    summary_path = args.output_root / "activity_shift_cache_summary.json"
    summary_path.write_text(
        json.dumps(root_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "SHA256SUMS.txt").write_text(
        f"{sha256_file(summary_path)}  {summary_path.name}\n", encoding="ascii"
    )
    print(json.dumps(root_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
