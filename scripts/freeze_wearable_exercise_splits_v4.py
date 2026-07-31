#!/usr/bin/env python3
"""Freeze sex-constrained v4 participant and session split artifacts.

V4 is written alongside v3. It enforces mixed-sex test folds, balances protocol
support, records chronological history eligibility, and defines strict activity
and joint user-activity shift roles without changing any frozen v3 artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from freeze_wearable_exercise_splits import (
    ROLE_ORDER,
    as_csv_value,
    attach_sex_metadata,
    balanced_select,
    derive_participants,
    feature_names,
    normalize_participant,
    sha256_file,
    stable_hash,
)


DEFAULT_SEED = 20260731
FREEZE_DATE = "2026-07-31"


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
) -> None:
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def parse_timestamp(value: str) -> float:
    token = value.strip()
    if not token:
        raise ValueError("Session timestamp is required for v4 chronology")
    try:
        numeric = float(token)
    except ValueError:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return numeric / 1000.0 if numeric > 1.0e12 else numeric


def largest_remainder_quotas(
    total: int,
    capacities: list[int],
    population: int,
    seed: int,
    label: str,
) -> list[int]:
    raw = [total * capacity / population for capacity in capacities]
    quotas = [int(math.floor(value)) for value in raw]
    remainder = total - sum(quotas)
    order = sorted(
        range(len(capacities)),
        key=lambda index: (
            -(raw[index] - quotas[index]),
            stable_hash(seed, label, index),
        ),
    )
    for index in order[:remainder]:
        quotas[index] += 1
    return quotas


def assignment_score(
    members: list[dict[str, Any]],
    candidate: dict[str, Any],
    quota: int,
    population: list[dict[str, Any]],
) -> float:
    after = members + [candidate]
    score = 4.0 * ((len(after) - quota) / max(quota, 1)) ** 2
    for field in (
        "aerobic_eligible_origins",
        "anaerobic_eligible_origins",
        "aerobic_sessions",
        "anaerobic_sessions",
    ):
        total = sum(float(record[field]) for record in population)
        target = total * quota / max(len(population), 1)
        observed = sum(float(record[field]) for record in after)
        score += ((observed - target) / max(target, 1.0)) ** 2
    cross_total = sum(bool(record["cross_protocol_eligible"]) for record in population)
    cross_target = cross_total * quota / max(len(population), 1)
    cross_observed = sum(bool(record["cross_protocol_eligible"]) for record in after)
    score += ((cross_observed - cross_target) / max(cross_target, 1.0)) ** 2
    return score


def assign_test_folds_v4(
    participants: list[dict[str, Any]],
    folds: int,
    seed: int,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    base, remainder = divmod(len(participants), folds)
    capacities = [base + int(index < remainder) for index in range(folds)]
    by_sex: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in participants:
        by_sex[record["sex"]].append(record)
    if set(by_sex) != {"female", "male"}:
        raise ValueError(f"V4 requires resolved female/male metadata, received {sorted(by_sex)}")

    female_total = len(by_sex["female"])
    minimum_female = folds
    maximum_female = sum(capacity - 1 for capacity in capacities)
    if not minimum_female <= female_total <= maximum_female:
        raise RuntimeError(
            "The participant sex counts cannot place both sexes in every test fold"
        )

    female_quotas = [1 for _ in range(folds)]
    for allocation_step in range(female_total - minimum_female):
        choices = [
            fold
            for fold, capacity in enumerate(capacities)
            if female_quotas[fold] < capacity - 1
        ]
        selected = min(
            choices,
            key=lambda fold: (
                (female_quotas[fold] + 1) / capacities[fold],
                female_quotas[fold],
                stable_hash(seed, "v4-sex-quota", allocation_step, fold),
            ),
        )
        female_quotas[selected] += 1

    quotas: dict[str, list[int]] = {
        "female": female_quotas,
        "male": [
            capacity - female_quotas[fold]
            for fold, capacity in enumerate(capacities)
        ],
    }
    for fold, capacity in enumerate(capacities):
        if sum(quotas[sex][fold] for sex in quotas) != capacity:
            raise RuntimeError("Sex quotas do not match the fold capacity")
        if any(quotas[sex][fold] == 0 for sex in ("female", "male")):
            raise RuntimeError("Every v4 test fold must contain both sexes")

    assignment: dict[str, int] = {}
    for sex, population in sorted(by_sex.items()):
        members: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
        ordered = sorted(
            population,
            key=lambda record: (
                -(record["aerobic_eligible_origins"] + record["anaerobic_eligible_origins"]),
                not bool(record["cross_protocol_eligible"]),
                stable_hash(seed, "v4-order", sex, record["participant_id"]),
            ),
        )
        for record in ordered:
            choices = [
                fold
                for fold in range(folds)
                if len(members[fold]) < quotas[sex][fold]
            ]
            selected = min(
                choices,
                key=lambda fold: (
                    assignment_score(
                        members[fold], record, quotas[sex][fold], population
                    ),
                    stable_hash(seed, "v4-fold", record["participant_id"], fold),
                ),
            )
            members[selected].append(record)
            assignment[record["participant_id"]] = selected
    return assignment, quotas


def role_sex_quota(
    population: list[dict[str, Any]], count: int, seed: int, label: str
) -> dict[str, int]:
    counts = Counter(record["sex"] for record in population)
    female = int(round(count * counts["female"] / len(population)))
    female = min(max(female, 1), count - 1)
    return {"female": female, "male": count - female}


def select_role(
    population: list[dict[str, Any]],
    count: int,
    seed: int,
    label: str,
    names: list[str],
) -> list[dict[str, Any]]:
    quotas = role_sex_quota(population, count, seed, label)
    selected: list[dict[str, Any]] = []
    for sex in ("female", "male"):
        candidates = [record for record in population if record["sex"] == sex]
        selected.extend(
            balanced_select(
                candidates,
                quotas[sex],
                seed,
                f"{label}-{sex}",
                names,
            )
        )
    return selected


def role_summary(
    rows: list[dict[str, Any]], fold: int, role: str
) -> dict[str, int]:
    subset = [row for row in rows if row["outer_fold"] == fold and row["role"] == role]
    return {
        "participants": len(subset),
        "female": sum(row["sex"] == "female" for row in subset),
        "male": sum(row["sex"] == "male" for row in subset),
        "aerobic_sessions": sum(int(row["aerobic_sessions"]) for row in subset),
        "anaerobic_sessions": sum(int(row["anaerobic_sessions"]) for row in subset),
        "aerobic_eligible_origins": sum(int(row["aerobic_eligible_origins"]) for row in subset),
        "anaerobic_eligible_origins": sum(int(row["anaerobic_eligible_origins"]) for row in subset),
        "cross_protocol_eligible": sum(bool(row["cross_protocol_eligible"]) for row in subset),
    }


def read_session_audit(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            protocol = source["protocol"].strip().upper()
            participant = normalize_participant(source["user_id"])
            session_token = source["session_id"].strip()
            start = parse_timestamp(source["common_support_start"] or source["hr_start"])
            end = parse_timestamp(source["common_support_end"])
            rows.append(
                {
                    "session_key": f"{protocol}:{session_token}",
                    "session_id": session_token,
                    "participant_id": participant,
                    "protocol": protocol,
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "eligible_origins": int(float(source["eligible_stride_120s_origins"] or 0)),
                    "is_split_fragment": int(source["is_split_fragment"].strip().lower() == "true"),
                }
            )
    return rows


def session_and_history_rows(
    sessions: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    folds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roles = {
        (int(row["outer_fold"]), row["participant_id"]): row["role"]
        for row in split_rows
    }
    session_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        materialized = [
            {
                **session,
                "outer_fold": fold,
                "role": roles[(fold, session["participant_id"])],
            }
            for session in sessions
        ]
        session_rows.extend(materialized)
        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in materialized:
            if row["eligible_origins"] > 0:
                by_user[row["participant_id"]].append(row)
        for user_rows in by_user.values():
            user_rows.sort(key=lambda row: (row["start_timestamp"], row["session_key"]))
            for target in user_rows:
                for history in user_rows:
                    if history["end_timestamp"] >= target["start_timestamp"]:
                        continue
                    history_rows.append(
                        {
                            "outer_fold": fold,
                            "participant_id": target["participant_id"],
                            "target_session_key": target["session_key"],
                            "target_role": target["role"],
                            "history_session_key": history["session_key"],
                            "history_role": history["role"],
                            "history_ends_before_target": 1,
                            "train_prior_eligible": int(history["role"] == "train"),
                            "role_prior_eligible": int(history["role"] == target["role"]),
                            "all_prior_eligible": 1,
                        }
                    )
    return session_rows, history_rows


def activity_shift_rows(session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for heldout in ("AEROBIC", "ANAEROBIC"):
        for row in session_rows:
            base_role = row["role"]
            if row["protocol"] != heldout:
                activity_role = base_role if base_role != "test" else "reserved_user_test"
            elif base_role == "train":
                activity_role = "activity_test_seen_user"
            elif base_role == "test":
                activity_role = "joint_user_activity_test"
            else:
                activity_role = "excluded_target_activity"
            output.append(
                {
                    "outer_fold": row["outer_fold"],
                    "heldout_protocol": heldout,
                    "session_key": row["session_key"],
                    "participant_id": row["participant_id"],
                    "protocol": row["protocol"],
                    "base_user_role": base_role,
                    "activity_shift_role": activity_role,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-audit",
        type=Path,
        default=Path("outputs/wearable_exercise_audit/session_summary.csv"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/wearable_exercise/s3_subset/Wearable_Dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits/wearable_exercise_v4"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--outer-folds", type=int, default=5)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite versioned v4 artifacts: {args.output_dir}")
    if not args.session_audit.is_file():
        raise FileNotFoundError(args.session_audit)
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    participants_by_id, source_fields = derive_participants(args.session_audit)
    metadata_files = attach_sex_metadata(participants_by_id, args.data_dir)
    participants = sorted(participants_by_id.values(), key=lambda row: row["participant_id"])
    names = feature_names(participants)
    outer_assignment, sex_quotas = assign_test_folds_v4(
        participants, args.outer_folds, args.seed
    )

    split_rows: list[dict[str, Any]] = []
    for fold in range(args.outer_folds):
        test_ids = {pid for pid, assigned in outer_assignment.items() if assigned == fold}
        remaining = [row for row in participants if row["participant_id"] not in test_ids]
        calibration_count = max(4, math.ceil(0.18 * len(remaining)))
        validation_count = max(4, math.ceil(0.15 * len(remaining)))
        calibration = select_role(
            remaining, calibration_count, args.seed, f"v4-calibration-{fold}", names
        )
        calibration_ids = {row["participant_id"] for row in calibration}
        validation_pool = [
            row for row in remaining if row["participant_id"] not in calibration_ids
        ]
        validation = select_role(
            validation_pool, validation_count, args.seed, f"v4-validation-{fold}", names
        )
        validation_ids = {row["participant_id"] for row in validation}
        for record in participants:
            participant_id = record["participant_id"]
            role = (
                "test"
                if participant_id in test_ids
                else "calibration"
                if participant_id in calibration_ids
                else "validation"
                if participant_id in validation_ids
                else "train"
            )
            split_rows.append(
                {
                    "outer_fold": fold,
                    "participant_id": participant_id,
                    "role": role,
                    "permanent_test_fold": outer_assignment[participant_id],
                    **record,
                }
            )

    participant_fields = [
        "participant_id", "cohort", "sex", "aerobic_sessions", "anaerobic_sessions",
        "aerobic_eligible_origins", "anaerobic_eligible_origins", "aerobic_eligible5",
        "anaerobic_eligible5", "cross_protocol_eligible", "permanent_test_fold",
    ]
    participant_rows = [
        {
            **{
                key: as_csv_value(record[key])
                for key in participant_fields
                if key != "permanent_test_fold"
            },
            "permanent_test_fold": outer_assignment[record["participant_id"]],
        }
        for record in participants
    ]
    split_fields = ["outer_fold", "participant_id", "role", "permanent_test_fold"] + [
        field for field in participant_fields if field not in {"participant_id", "permanent_test_fold"}
    ]
    normalized_split_rows = [
        {field: as_csv_value(row[field]) for field in split_fields}
        for row in split_rows
    ]
    normalized_split_rows.sort(
        key=lambda row: (int(row["outer_fold"]), ROLE_ORDER[row["role"]], row["participant_id"])
    )

    sessions = read_session_audit(args.session_audit)
    session_rows, history_rows = session_and_history_rows(
        sessions, normalized_split_rows, args.outer_folds
    )
    activity_rows = activity_shift_rows(session_rows)
    session_counts = Counter(row["participant_id"] for row in sessions)
    temporal_rows = [
        {
            "participant_id": participant_id,
            "session_count": count,
            "four_way_temporal_split_feasible": int(count >= 4),
            "history_curve_0_1_3_5_feasible": int(count >= 6),
        }
        for participant_id, count in sorted(session_counts.items())
    ]

    fold_summary = {
        str(fold): {
            role: role_summary(split_rows, fold, role)
            for role in ("train", "validation", "calibration", "test")
        }
        for fold in range(args.outer_folds)
    }
    test_summaries = [fold_summary[str(fold)]["test"] for fold in range(args.outer_folds)]
    total_origins = [
        row["aerobic_eligible_origins"] + row["anaerobic_eligible_origins"]
        for row in test_summaries
    ]
    origin_ratio = max(total_origins) / max(min(total_origins), 1)
    leakage_checks = {
        "each_participant_has_one_role_per_fold": all(
            len({row["participant_id"] for row in split_rows if row["outer_fold"] == fold})
            == len(participants)
            for fold in range(args.outer_folds)
        ),
        "each_participant_is_test_once": all(
            sum(row["participant_id"] == participant and row["role"] == "test" for row in split_rows) == 1
            for participant in participants_by_id
        ),
        "every_test_fold_contains_both_sexes": all(
            summary["female"] > 0 and summary["male"] > 0
            for summary in test_summaries
        ),
        "test_sex_counts_differ_by_at_most_one": (
            max(summary["female"] for summary in test_summaries)
            - min(summary["female"] for summary in test_summaries)
            <= 1
            and max(summary["male"] for summary in test_summaries)
            - min(summary["male"] for summary in test_summaries)
            <= 1
        ),
        "history_is_strictly_chronological": all(
            row["history_ends_before_target"] == 1 for row in history_rows
        ),
        "test_origin_max_min_ratio_le_1_5": origin_ratio <= 1.5,
    }
    if not all(leakage_checks.values()):
        raise RuntimeError(f"V4 acceptance checks failed: {leakage_checks}")

    balance_rows = []
    for fold in range(args.outer_folds):
        for role in ("train", "validation", "calibration", "test"):
            balance_rows.append(
                {"outer_fold": fold, "role": role, **fold_summary[str(fold)][role]}
            )

    paths = {
        "participant_manifest.csv": participant_rows,
        "outer_fold_roles.csv": normalized_split_rows,
        "session_manifest.csv": session_rows,
        "history_manifest.csv": history_rows,
        "activity_shift_roles.csv": activity_rows,
        "temporal_feasibility.csv": temporal_rows,
        "fold_balance_audit.csv": balance_rows,
    }
    fields = {
        "participant_manifest.csv": participant_fields,
        "outer_fold_roles.csv": split_fields,
        "session_manifest.csv": [
            "session_key", "session_id", "participant_id", "protocol", "start_timestamp",
            "end_timestamp", "eligible_origins", "is_split_fragment", "outer_fold", "role",
        ],
        "history_manifest.csv": [
            "outer_fold", "participant_id", "target_session_key", "target_role",
            "history_session_key", "history_role", "history_ends_before_target",
            "train_prior_eligible", "role_prior_eligible", "all_prior_eligible",
        ],
        "activity_shift_roles.csv": [
            "outer_fold", "heldout_protocol", "session_key", "participant_id", "protocol",
            "base_user_role", "activity_shift_role",
        ],
        "temporal_feasibility.csv": [
            "participant_id", "session_count", "four_way_temporal_split_feasible",
            "history_curve_0_1_3_5_feasible",
        ],
        "fold_balance_audit.csv": [
            "outer_fold", "role", "participants", "female", "male", "aerobic_sessions",
            "anaerobic_sessions", "aerobic_eligible_origins", "anaerobic_eligible_origins",
            "cross_protocol_eligible",
        ],
    }
    for name, rows in paths.items():
        write_csv(args.output_dir / name, rows, fields[name])

    leakage_path = args.output_dir / "leakage_audit.json"
    leakage_payload = {
        "contract_version": "0.4.0",
        "checks": leakage_checks,
        "test_origin_max_min_ratio": origin_ratio,
        "test_sex_quotas": sex_quotas,
        "history_pairs": len(history_rows),
    }
    leakage_path.write_text(
        json.dumps(leakage_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact_hashes = {
        name: sha256_file(args.output_dir / name) for name in paths
    }
    policy = {
        "contract_version": "0.4.0",
        "manuscript_title": "Deployment-Aware Evaluation of Physiology-Guided Heart-Rate Transition Forecasting under User and Activity Shifts",
        "dataset": "Wearable Device Dataset from Induced Stress and Structured Exercise Sessions v1.0.1",
        "freeze_date": FREEZE_DATE,
        "seed": args.seed,
        "outer_folds": args.outer_folds,
        "participants": len(participants),
        "highest_partition_unit": "base participant identifier",
        "assignment_algorithm": "exact per-fold sex quotas plus deterministic greedy protocol-origin balancing",
        "history_rule": "same participant and history end strictly before target start; regime legality recorded separately",
        "activity_shift_rule": "held-out protocol absent from model-fitting roles; seen-user activity and joint user-activity tests are distinct",
        "temporal_split_status": "feasibility recorded; no invalid four-way per-user split is fabricated when sessions are insufficient",
        "source_columns": source_fields,
        "source_sha256": {
            args.session_audit.as_posix(): sha256_file(args.session_audit),
            **{path.as_posix(): sha256_file(path) for path in metadata_files},
        },
        "fold_summary": fold_summary,
        "test_origin_max_min_ratio": origin_ratio,
        "artifact_sha256": artifact_hashes,
        "v3_status": "preserved and not overwritten",
        "revision_rule": "Never overwrite v4 after fitting; create v5 for any later approved revision.",
    }
    policy_path = args.output_dir / "split_policy.json"
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme_path = args.output_dir / "README.md"
    readme_path.write_text(
        "# Frozen wearable-exercise v4 splits\n\n"
        "V4 preserves v3 and corrects sex-segregated outer test folds through exact sex quotas. "
        "It adds session-level chronology, legal history pairs, activity-shift roles, temporal "
        "feasibility, fold-balance evidence, and a machine-readable leakage audit.\n\n"
        "Strict unseen-user evaluation uses `train_prior`, which gives held-out participants zero "
        "same-user history. Few-shot personalization uses `role_prior` and must be reported separately.\n",
        encoding="utf-8",
    )
    checksum_targets = [
        *(args.output_dir / name for name in paths),
        leakage_path,
        policy_path,
        readme_path,
    ]
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_targets),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir.as_posix(),
                "participants": len(participants),
                "outer_folds": args.outer_folds,
                "test_origin_max_min_ratio": origin_ratio,
                "checks": leakage_checks,
                "fold_summary": fold_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
