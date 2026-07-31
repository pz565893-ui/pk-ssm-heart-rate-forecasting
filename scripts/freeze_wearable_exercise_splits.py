#!/usr/bin/env python3
"""Freeze deterministic participant-disjoint splits for the primary dataset.

The script consumes the corrected session-level audit rather than raw forecast
windows. This makes the base participant identifier the highest partitioning
unit and prevents split recording fragments or protocols from crossing roles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_SEED = 20260730
ROLE_ORDER = {"train": 0, "validation": 1, "calibration": 2, "test": 3}


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_participant(value: str) -> str:
    value = value.strip()
    value = re.sub(r"_(?:a|b)$", "", value, flags=re.IGNORECASE)
    match = re.search(r"(?i)(?:^|[^a-z0-9])([sf]\d{1,3})(?:[^a-z0-9]|$)", value)
    if match:
        prefix = match.group(1)[0]
        number = int(match.group(1)[1:])
        return f"{prefix}{number:02d}"
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(seed: int, *parts: object) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def find_column(
    fieldnames: Iterable[str],
    exact: Iterable[str],
    predicate: Callable[[str], bool],
) -> str | None:
    normalized = {normalized_name(field): field for field in fieldnames}
    for candidate in exact:
        if normalized_name(candidate) in normalized:
            return normalized[normalized_name(candidate)]
    for field in fieldnames:
        if predicate(normalized_name(field)):
            return field
    return None


def parse_number(value: str | None) -> float:
    if value is None or not value.strip():
        return 0.0
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric audit value, received {value!r}") from exc


def derive_participants(session_csv: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows, fields = read_csv(session_csv)
    protocol_col = find_column(
        fields,
        ["protocol"],
        lambda name: name in {"activity", "exercise_protocol", "protocol_name"},
    )
    participant_col = find_column(
        fields,
        ["base_participant_id", "base_subject_id", "base_user_id", "base_user", "participant_id", "subject_id", "user_id"],
        lambda name: "base" in name and any(token in name for token in ("participant", "subject", "user")),
    )
    if participant_col is None:
        participant_col = find_column(
            fields,
            ["session_id", "session", "session_name"],
            lambda name: "session" in name and ("id" in name or "name" in name),
        )
    origin_col = find_column(
        fields,
        ["eligible_stride_origins", "eligible_origins_stride", "eligible_origins", "strict_stride_origins"],
        lambda name: "eligible" in name and "origin" in name and ("stride" in name or "120" in name),
    )
    if origin_col is None:
        origin_col = find_column(
            fields,
            [],
            lambda name: "eligible" in name and "origin" in name,
        )
    missing = [
        label
        for label, column in (
            ("protocol", protocol_col),
            ("participant", participant_col),
            ("eligible-origin count", origin_col),
        )
        if column is None
    ]
    if missing:
        raise ValueError(
            f"Could not resolve {', '.join(missing)} columns in {session_csv}. "
            f"Available columns: {fields}"
        )

    participants: dict[str, dict[str, Any]] = {}
    for row in rows:
        participant = normalize_participant(row[participant_col])
        if not participant:
            raise ValueError("Encountered an empty participant identifier")
        protocol = row[protocol_col].strip().upper()
        if protocol not in {"AEROBIC", "ANAEROBIC"}:
            continue
        record = participants.setdefault(
            participant,
            {
                "participant_id": participant,
                "cohort": "V1" if participant.startswith("S") else "V2" if participant.startswith("f") else "unknown",
                "sex": "unknown",
                "aerobic_sessions": 0,
                "anaerobic_sessions": 0,
                "aerobic_eligible_origins": 0,
                "anaerobic_eligible_origins": 0,
            },
        )
        key = protocol.lower()
        record[f"{key}_sessions"] += 1
        record[f"{key}_eligible_origins"] += int(parse_number(row[origin_col]))

    if not participants:
        raise ValueError("The corrected audit produced no exercise participants")

    for record in participants.values():
        record["aerobic_eligible5"] = record["aerobic_eligible_origins"] >= 5
        record["anaerobic_eligible5"] = record["anaerobic_eligible_origins"] >= 5
        record["cross_protocol_eligible"] = (
            record["aerobic_eligible5"] and record["anaerobic_eligible5"]
        )
    return participants, fields


def normalize_sex(value: str) -> str:
    token = normalized_name(value)
    if token in {"f", "female", "woman", "women", "femenino", "feminine"}:
        return "female"
    if token in {"m", "male", "man", "men", "masculino", "masculine"}:
        return "male"
    return "unknown"


def attach_sex_metadata(participants: dict[str, dict[str, Any]], data_dir: Path) -> list[Path]:
    metadata_files = sorted(
        path
        for path in data_dir.rglob("*.csv")
        if "subject" in normalized_name(path.name) and "info" in normalized_name(path.name)
    )
    observed: defaultdict[str, set[str]] = defaultdict(set)
    for path in metadata_files:
        rows, fields = read_csv(path)
        identifier_col = find_column(
            fields,
            ["Info", "subject_id", "participant_id", "user_id", "subject", "participant", "id"],
            lambda name: any(token in name for token in ("subject", "participant", "user")) and "id" in name,
        )
        sex_col = find_column(
            fields,
            ["sex", "gender"],
            lambda name: "sex" in name or "gender" in name,
        )
        if identifier_col is None or sex_col is None:
            continue
        for row in rows:
            participant = normalize_participant(row.get(identifier_col, ""))
            sex = normalize_sex(row.get(sex_col, ""))
            if participant in participants and sex != "unknown":
                observed[participant].add(sex)

    conflicts = {participant: values for participant, values in observed.items() if len(values) > 1}
    if conflicts:
        raise ValueError(f"Conflicting sex metadata: {conflicts}")
    for participant, values in observed.items():
        participants[participant]["sex"] = next(iter(values))
    return metadata_files


def feature_names(participants: list[dict[str, Any]]) -> list[str]:
    names = [
        "cohort_V1",
        "cohort_V2",
        "sex_female",
        "sex_male",
        "sex_unknown",
        "aerobic_eligible5",
        "anaerobic_eligible5",
        "cross_protocol_eligible",
        "aerobic_only_eligible",
        "anaerobic_only_eligible",
    ]
    return names


def feature_vector(record: dict[str, Any], names: list[str]) -> dict[str, int]:
    values = {
        "cohort_V1": record["cohort"] == "V1",
        "cohort_V2": record["cohort"] == "V2",
        "sex_female": record["sex"] == "female",
        "sex_male": record["sex"] == "male",
        "sex_unknown": record["sex"] == "unknown",
        "aerobic_eligible5": bool(record["aerobic_eligible5"]),
        "anaerobic_eligible5": bool(record["anaerobic_eligible5"]),
        "cross_protocol_eligible": bool(record["cross_protocol_eligible"]),
        "aerobic_only_eligible": bool(record["aerobic_eligible5"] and not record["anaerobic_eligible5"]),
        "anaerobic_only_eligible": bool(record["anaerobic_eligible5"] and not record["aerobic_eligible5"]),
    }
    return {name: int(values[name]) for name in names}


def squared_balance_score(
    members: list[dict[str, Any]],
    candidate: dict[str, Any],
    target_size: int,
    population_totals: Counter[str],
    population_size: int,
    names: list[str],
) -> float:
    after = members + [candidate]
    score = 5.0 * ((len(after) - target_size) / max(target_size, 1)) ** 2
    counts: Counter[str] = Counter()
    for record in after:
        counts.update(feature_vector(record, names))
    for name in names:
        target = population_totals[name] * target_size / population_size
        scale = max(target, 1.0)
        score += ((counts[name] - target) / scale) ** 2
    return score


def assign_test_folds(
    participants: list[dict[str, Any]], folds: int, seed: int, names: list[str]
) -> dict[str, int]:
    population_totals: Counter[str] = Counter()
    for record in participants:
        population_totals.update(feature_vector(record, names))
    base, remainder = divmod(len(participants), folds)
    capacities = [base + int(index < remainder) for index in range(folds)]
    rarity = {
        record["participant_id"]: sum(
            value / max(population_totals[name], 1)
            for name, value in feature_vector(record, names).items()
        )
        for record in participants
    }
    ordered = sorted(
        participants,
        key=lambda record: (
            -rarity[record["participant_id"]],
            stable_hash(seed, "outer-order", record["participant_id"]),
        ),
    )
    members: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
    for record in ordered:
        choices = [index for index in range(folds) if len(members[index]) < capacities[index]]
        selected = min(
            choices,
            key=lambda index: (
                squared_balance_score(
                    members[index],
                    record,
                    capacities[index],
                    population_totals,
                    len(participants),
                    names,
                ),
                stable_hash(seed, "outer-tie", record["participant_id"], index),
            ),
        )
        members[selected].append(record)
    return {
        record["participant_id"]: fold
        for fold, records in enumerate(members)
        for record in records
    }


def balanced_select(
    population: list[dict[str, Any]],
    count: int,
    seed: int,
    label: str,
    names: list[str],
) -> list[dict[str, Any]]:
    population_totals: Counter[str] = Counter()
    for record in population:
        population_totals.update(feature_vector(record, names))
    selected: list[dict[str, Any]] = []
    available = list(population)
    while len(selected) < count:
        record = min(
            available,
            key=lambda candidate: (
                squared_balance_score(
                    selected,
                    candidate,
                    count,
                    population_totals,
                    len(population),
                    names,
                ),
                stable_hash(seed, label, candidate["participant_id"]),
            ),
        )
        selected.append(record)
        available.remove(record)
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def role_summary(rows: list[dict[str, Any]], fold: int, role: str) -> dict[str, int]:
    subset = [row for row in rows if row["outer_fold"] == fold and row["role"] == role]
    return {
        "participants": len(subset),
        "female": sum(row["sex"] == "female" for row in subset),
        "male": sum(row["sex"] == "male" for row in subset),
        "unknown_sex": sum(row["sex"] == "unknown" for row in subset),
        "aerobic_eligible5": sum(bool(row["aerobic_eligible5"]) for row in subset),
        "anaerobic_eligible5": sum(bool(row["anaerobic_eligible5"]) for row in subset),
        "cross_protocol_eligible": sum(bool(row["cross_protocol_eligible"]) for row in subset),
    }


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
        default=Path("splits/wearable_exercise_v3"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.outer_folds < 3:
        raise ValueError("At least three outer folds are required")
    if not args.session_audit.is_file():
        raise FileNotFoundError(args.session_audit)
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite frozen artifacts in {args.output_dir}; use --force only for an approved protocol revision"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    participants_by_id, source_fields = derive_participants(args.session_audit)
    metadata_files = attach_sex_metadata(participants_by_id, args.data_dir)
    participants = sorted(participants_by_id.values(), key=lambda item: item["participant_id"])
    missing_sex = [
        record["participant_id"]
        for record in participants
        if record["sex"] == "unknown"
    ]
    if missing_sex:
        raise ValueError(
            "Final split generation requires resolved sex metadata for every "
            f"participant; unresolved identifiers: {missing_sex}"
        )
    names = feature_names(participants)
    outer_assignment = assign_test_folds(participants, args.outer_folds, args.seed, names)

    split_rows: list[dict[str, Any]] = []
    for outer_fold in range(args.outer_folds):
        test_ids = {
            participant_id
            for participant_id, fold in outer_assignment.items()
            if fold == outer_fold
        }
        remaining = [record for record in participants if record["participant_id"] not in test_ids]
        calibration_count = max(4, math.ceil(0.18 * len(remaining)))
        validation_count = max(4, math.ceil(0.15 * len(remaining)))
        if calibration_count + validation_count >= len(remaining):
            raise ValueError("Insufficient participants for disjoint train, validation, and calibration roles")
        calibration = balanced_select(
            remaining, calibration_count, args.seed, f"calibration-{outer_fold}", names
        )
        calibration_ids = {record["participant_id"] for record in calibration}
        validation_pool = [
            record for record in remaining if record["participant_id"] not in calibration_ids
        ]
        validation = balanced_select(
            validation_pool, validation_count, args.seed, f"validation-{outer_fold}", names
        )
        validation_ids = {record["participant_id"] for record in validation}

        for record in participants:
            participant_id = record["participant_id"]
            if participant_id in test_ids:
                role = "test"
            elif participant_id in calibration_ids:
                role = "calibration"
            elif participant_id in validation_ids:
                role = "validation"
            else:
                role = "train"
            split_rows.append(
                {
                    "outer_fold": outer_fold,
                    "participant_id": participant_id,
                    "role": role,
                    "permanent_test_fold": outer_assignment[participant_id],
                    **record,
                }
            )

    participant_fields = [
        "participant_id",
        "cohort",
        "sex",
        "aerobic_sessions",
        "anaerobic_sessions",
        "aerobic_eligible_origins",
        "anaerobic_eligible_origins",
        "aerobic_eligible5",
        "anaerobic_eligible5",
        "cross_protocol_eligible",
        "permanent_test_fold",
    ]
    participant_rows = [
        {
            **{key: as_csv_value(record[key]) for key in participant_fields if key != "permanent_test_fold"},
            "permanent_test_fold": outer_assignment[record["participant_id"]],
        }
        for record in participants
    ]
    split_fields = ["outer_fold", "participant_id", "role", "permanent_test_fold"] + [
        field for field in participant_fields if field not in {"participant_id", "permanent_test_fold"}
    ]
    normalized_split_rows = [
        {field: as_csv_value(row[field]) for field in split_fields} for row in split_rows
    ]
    normalized_split_rows.sort(
        key=lambda row: (int(row["outer_fold"]), ROLE_ORDER[str(row["role"])], str(row["participant_id"]))
    )

    participant_path = args.output_dir / "participant_manifest.csv"
    split_path = args.output_dir / "outer_fold_roles.csv"
    policy_path = args.output_dir / "split_policy.json"
    readme_path = args.output_dir / "README.md"
    checksums_path = args.output_dir / "SHA256SUMS.txt"
    write_csv(participant_path, participant_rows, participant_fields)
    write_csv(split_path, normalized_split_rows, split_fields)

    fold_summary = {
        str(fold): {
            role: role_summary(split_rows, fold, role)
            for role in ("train", "validation", "calibration", "test")
        }
        for fold in range(args.outer_folds)
    }
    sex_counts = Counter(record["sex"] for record in participants)
    policy = {
        "contract_version": "0.3.1",
        "manuscript_title": "Deployment-Aware Evaluation of Physiology-Guided Heart-Rate Transition Forecasting under User and Activity Shifts",
        "dataset": "Wearable Device Dataset from Induced Stress and Structured Exercise Sessions v1.0.1",
        "freeze_date": "2026-07-30",
        "seed": args.seed,
        "outer_folds": args.outer_folds,
        "highest_partition_unit": "base participant identifier",
        "fragment_rule": "remove _a/_b suffix; all fragments and protocols for a participant share one role",
        "assignment_algorithm": "deterministic greedy balancing of cohort, sex, and protocol eligibility with SHA-256 tie breaking",
        "role_policy": {
            "test": "one permanent subject-disjoint outer fold",
            "calibration": "max(4, ceil(18% of non-test participants)); disjoint from all other roles",
            "validation": "max(4, ceil(15% of non-test participants)); selected after calibration",
            "train": "all remaining participants",
        },
        "participants": len(participants),
        "sex_counts": dict(sorted(sex_counts.items())),
        "audit_source_columns": source_fields,
        "metadata_files": [path.as_posix() for path in metadata_files],
        "source_sha256": {
            args.session_audit.as_posix(): sha256_file(args.session_audit),
            **{path.as_posix(): sha256_file(path) for path in metadata_files},
        },
        "fold_summary": fold_summary,
        "artifact_sha256": {
            participant_path.name: sha256_file(participant_path),
            split_path.name: sha256_file(split_path),
        },
        "revision_rule": "Never overwrite these files after model fitting; create a versioned directory and disclose the revision instead.",
    }
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_path.write_text(
        "# Frozen primary-dataset participant splits\n\n"
        "These manifests were generated before model fitting from the corrected joint HR-ACC audit. "
        "Every base participant is assigned to exactly one role within each outer fold. Split fragments "
        "and aerobic/anaerobic records from the same participant cannot cross roles.\n\n"
        "`outer_fold_roles.csv` is the authoritative model-fitting manifest. `participant_manifest.csv` "
        "records the audit-derived support used for balancing. `split_policy.json` records the algorithm, "
        "seed, source hashes, role counts, and artifact hashes.\n\n"
        "Do not overwrite this directory after training begins. Any approved change must create a new "
        "versioned directory and be disclosed as a protocol amendment.\n",
        encoding="utf-8",
    )
    checksum_targets = [participant_path, split_path, policy_path, readme_path]
    checksums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_targets),
        encoding="utf-8",
    )

    output = {
        "participants": len(participants),
        "sex_counts": dict(sorted(sex_counts.items())),
        "outer_folds": args.outer_folds,
        "fold_summary": fold_summary,
        "split_manifest_sha256": sha256_file(split_path),
        "output_dir": args.output_dir.as_posix(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
