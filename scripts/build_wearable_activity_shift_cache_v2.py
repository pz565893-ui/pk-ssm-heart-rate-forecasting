#!/usr/bin/env python3
"""Build corrected, non-destructive Wearable v4 activity-shift caches.

Version 2 preserves the frozen v1 builder and addresses its pre-fit acceptance
failure.  Seen-user target rows are retained only when that participant has a
source-protocol session in the base training role.  No signal, event, target,
participant fold, or test-origin definition is otherwise changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_wearable_activity_shift_cache_v1 import (
    BOUNDARIES,
    PROTOCOLS,
    acceptance_checks,
    mapped_role,
    read_csv,
    role_participants,
    sha256_file,
    source_normalization,
    write_csv,
)


CONTRACT_VERSION = "2.0.0"
PROTOCOL_LABEL = "activity_shift_v2"


def derive_rows_v2(
    rows: list[dict[str, str]], heldout: str, boundary: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    source = next(value for value in PROTOCOLS if value != heldout)
    source_training_participants = {
        row["participant_id"]
        for row in rows
        if row["protocol"].upper() == source and row["role"].lower() == "train"
    }
    candidate_seen_user_participants = {
        row["participant_id"]
        for row in rows
        if row["protocol"].upper() == heldout and row["role"].lower() == "train"
    }
    eligible_seen_user_participants = (
        source_training_participants & candidate_seen_user_participants
    )

    output: list[dict[str, str]] = []
    excluded_seen_user_rows = 0
    for row in rows:
        base_role = row["role"].lower()
        protocol = row["protocol"].upper()
        role = mapped_role(protocol, base_role, heldout, boundary)
        if role is None:
            continue
        if (
            boundary == "seen_user_activity"
            and role == "test"
            and row["participant_id"] not in eligible_seen_user_participants
        ):
            excluded_seen_user_rows += 1
            continue
        derived = dict(row)
        derived["role"] = role
        derived["protocol_version"] = (
            f"{PROTOCOL_LABEL}:{source}_to_{heldout}:{boundary}"
        )
        derived["base_role"] = base_role
        derived["source_protocol"] = source
        derived["heldout_protocol"] = heldout
        derived["activity_shift_boundary"] = boundary
        output.append(derived)

    audit = {
        "source_training_participants": len(source_training_participants),
        "candidate_heldout_base_train_participants": len(
            candidate_seen_user_participants
        ),
        "eligible_seen_user_participants": len(eligible_seen_user_participants),
        "excluded_seen_user_participants_without_source_training": len(
            candidate_seen_user_participants - source_training_participants
        ),
        "excluded_seen_user_origin_rows": excluded_seen_user_rows,
    }
    return output, audit


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
        default=Path("data/processed/wearable_activity_shift_v2"),
    )
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    v1_builder_path = script_path.with_name("build_wearable_activity_shift_cache_v1.py")
    root_summary: dict[str, Any] = {
        "artifact_status": "accepted_derived_wearable_v4_activity_shift_cache",
        "contract_version": CONTRACT_VERSION,
        "protocol_amendment": "027",
        "supersedes_failed_partial_cache": "wearable_activity_shift_v1",
        "correction": (
            "seen_user_activity target participants restricted to the intersection "
            "of held-out base-train users and source-protocol training users"
        ),
        "test_targets_opened_during_cache_build": False,
        "history_regime": "none",
        "activity_identity": "masked for source and target",
        "builder_sha256": sha256_file(script_path),
        "preserved_v1_builder_sha256": sha256_file(v1_builder_path),
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

                rows, filter_audit = derive_rows_v2(base_rows, heldout, boundary)
                checks = acceptance_checks(rows, heldout, boundary)
                checks["seen_user_filter_applied_before_manifest_write"] = (
                    boundary != "seen_user_activity"
                    or filter_audit["eligible_seen_user_participants"]
                    == len(role_participants(rows, "test"))
                )
                if not all(checks.values()):
                    raise RuntimeError(
                        f"Activity-shift v2 acceptance failure for {direction}, "
                        f"fold {fold}, {boundary}: {checks}"
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
                    "artifact_status": "accepted_derived_wearable_v4_activity_shift_cache",
                    "contract_version": CONTRACT_VERSION,
                    "protocol_amendment": "027",
                    "outer_fold": fold,
                    "source_protocol": source,
                    "heldout_protocol": heldout,
                    "activity_shift_boundary": boundary,
                    "split_manifest": (
                        "splits/wearable_exercise_v4/outer_fold_roles.csv::"
                        f"{PROTOCOL_LABEL}::{direction}::{boundary}"
                    ),
                    "preprocessing_config": base_policy["preprocessing_config"],
                    "history_regime": "none",
                    "activity_identity": "masked",
                    "normalization_scope": "source protocol and base training role only",
                    "base_cache_policy_sha256": sha256_file(base_policy_path),
                    "base_origin_manifest_sha256": sha256_file(base_manifest_path),
                    "builder_sha256": root_summary["builder_sha256"],
                    "origin_rows": len(rows),
                    "role_origin_rows": dict(sorted(role_counts.items())),
                    "role_participants": {
                        role: len(role_participants(rows, role))
                        for role in ("train", "validation", "calibration", "test")
                    },
                    "seen_user_filter_audit": filter_audit,
                    "acceptance_checks": checks,
                    "test_targets_opened_during_cache_build": False,
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
        json.dumps(root_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "SHA256SUMS.txt").write_text(
        f"{sha256_file(summary_path)}  {summary_path.name}\n",
        encoding="ascii",
    )
    print(json.dumps(root_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
