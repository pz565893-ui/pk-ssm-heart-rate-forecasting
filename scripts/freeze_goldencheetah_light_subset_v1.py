#!/usr/bin/env python3
"""Freeze a small, longitudinal GoldenCheetah manifest without copying raw data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_goldencheetah_longitudinal_eligibility import (
    canonical_sport,
    eligible_ride,
    scalar_float,
    stable_hash,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ride_time(value: object) -> datetime:
    return datetime.strptime(str(value), "%Y/%m/%d %H:%M:%S UTC")


def scan_csv(path: Path) -> dict[str, Any]:
    rows = 0
    valid_hr_rows = 0
    first_valid_second: float | None = None
    last_valid_second: float | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"secs", "hr"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            return {"eligible": False, "reason": "missing_secs_or_hr_columns"}
        for row in reader:
            try:
                second = float(row["secs"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(second):
                continue
            rows += 1
            try:
                hr = float(row["hr"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(hr) or not 30.0 <= hr <= 220.0:
                continue
            valid_hr_rows += 1
            first_valid_second = second if first_valid_second is None else min(
                first_valid_second, second
            )
            last_valid_second = second if last_valid_second is None else max(
                last_valid_second, second
            )
    coverage = valid_hr_rows / rows if rows else 0.0
    valid_span = (
        last_valid_second - first_valid_second
        if first_valid_second is not None and last_valid_second is not None
        else 0.0
    )
    eligible = bool(rows >= 600 and coverage >= 0.90 and valid_span >= 600.0)
    return {
        "eligible": eligible,
        "reason": "eligible" if eligible else "insufficient_hr_coverage_or_duration",
        "rows": rows,
        "valid_hr_rows": valid_hr_rows,
        "hr_coverage": coverage,
        "valid_hr_span_seconds": valid_span,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("outputs/goldencheetah_longitudinal_audit_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits/goldencheetah_light_v1"),
    )
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--sessions-per-user", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    if not args.audit.is_file():
        raise FileNotFoundError(args.audit)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen subset: {args.output_dir}")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    mixed_ids = {
        str(user["user_id"]) for user in audit["mixed_activity_candidates"]
    }
    candidates = list(audit["longitudinal_candidates"])
    candidates.sort(
        key=lambda user: (
            0 if str(user.get("gender", "")).lower() in {"f", "female"} else 1,
            0 if str(user["user_id"]) in mixed_ids else 1,
            stable_hash(args.seed, user["user_id"]),
        )
    )

    participants: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    skipped_users: list[dict[str, Any]] = []
    for candidate in candidates:
        if len(participants) >= args.users:
            break
        user_id = str(candidate["user_id"])
        user_dir = args.root / user_id
        json_files = sorted(user_dir.glob("*.json"))
        csv_files = sorted(user_dir.glob("*.csv"))
        try:
            payload = json.loads(json_files[0].read_text(encoding="utf-8-sig"))
        except Exception as exc:
            skipped_users.append({"user_id": user_id, "reason": str(exc)})
            continue
        rides = sorted(payload.get("RIDES") or [], key=lambda row: parse_ride_time(row["date"]))
        if len(rides) != len(csv_files):
            skipped_users.append({"user_id": user_id, "reason": "ride_csv_count_mismatch"})
            continue

        metadata_pairs: list[tuple[dict[str, Any], Path]] = []
        for ride, csv_path in zip(rides, csv_files):
            valid, _ = eligible_ride(ride)
            if valid:
                metadata_pairs.append((ride, csv_path))
        metadata_pairs.sort(
            key=lambda pair: stable_hash(
                args.seed, user_id, pair[0]["date"], pair[1].name
            )
        )

        accepted: list[dict[str, Any]] = []
        rejection_counts: dict[str, int] = {}
        for ride, csv_path in metadata_pairs:
            raw_audit = scan_csv(csv_path)
            if not raw_audit["eligible"]:
                reason = str(raw_audit["reason"])
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            metrics = ride.get("METRICS") or {}
            accepted.append(
                {
                    "user_id": user_id,
                    "gender": str((payload.get("ATHLETE") or {}).get("gender") or "unknown").lower(),
                    "recorded_time_utc": str(ride["date"]),
                    "sport_raw": str(ride.get("sport") or ""),
                    "sport_canonical": canonical_sport(ride.get("sport")),
                    "metadata_duration_seconds": scalar_float(metrics.get("workout_time")),
                    "metadata_average_hr_bpm": scalar_float(metrics.get("average_hr")),
                    "source_csv_relative": csv_path.relative_to(args.root).as_posix(),
                    "source_csv_bytes": csv_path.stat().st_size,
                    "source_csv_sha256": sha256_file(csv_path),
                    "csv_rows": raw_audit["rows"],
                    "valid_hr_rows": raw_audit["valid_hr_rows"],
                    "hr_coverage": raw_audit["hr_coverage"],
                    "valid_hr_span_seconds": raw_audit["valid_hr_span_seconds"],
                }
            )
            if len(accepted) >= args.sessions_per_user:
                break
        if len(accepted) < args.sessions_per_user:
            skipped_users.append(
                {
                    "user_id": user_id,
                    "reason": "fewer_than_required_raw_eligible_sessions",
                    "accepted_sessions": len(accepted),
                    "rejection_counts": rejection_counts,
                }
            )
            continue

        accepted.sort(key=lambda row: parse_ride_time(row["recorded_time_utc"]))
        sport_counts: dict[str, int] = {}
        for order, row in enumerate(accepted, start=1):
            if order <= 10:
                temporal_role = "support_train"
            elif order <= 14:
                temporal_role = "validation"
            elif order <= 17:
                temporal_role = "calibration"
            else:
                temporal_role = "temporal_test"
            row["session_order"] = order
            row["temporal_role"] = temporal_role
            row["session_id"] = f"{user_id}__{order:02d}"
            sessions.append(row)
            sport = str(row["sport_canonical"])
            sport_counts[sport] = sport_counts.get(sport, 0) + 1
        participants.append(
            {
                "user_id": user_id,
                "gender": accepted[0]["gender"],
                "selected_sessions": len(accepted),
                "metadata_eligible_sessions": candidate["eligible_sessions"],
                "selected_bike_sessions": sport_counts.get("bike", 0),
                "selected_run_sessions": sport_counts.get("run", 0),
                "mixed_activity_candidate": user_id in mixed_ids,
                "selection_hash": stable_hash(args.seed, user_id),
            }
        )

    if len(participants) != args.users:
        raise RuntimeError(
            f"Only {len(participants)} users passed raw validation; required {args.users}"
        )

    checks = {
        "participant_count_exact": len(participants) == args.users,
        "session_count_exact": len(sessions) == args.users * args.sessions_per_user,
        "each_user_has_exact_session_budget": all(
            sum(row["user_id"] == participant["user_id"] for row in sessions)
            == args.sessions_per_user
            for participant in participants
        ),
        "history_is_strictly_chronological": all(
            rows == sorted(rows)
            for rows in (
                [
                    parse_ride_time(row["recorded_time_utc"])
                    for row in sessions
                    if row["user_id"] == participant["user_id"]
                ]
                for participant in participants
            )
        ),
        "all_raw_hr_coverage_ge_0_90": all(
            float(row["hr_coverage"]) >= 0.90 for row in sessions
        ),
        "all_valid_hr_spans_ge_600s": all(
            float(row["valid_hr_span_seconds"]) >= 600.0 for row in sessions
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen subset checks failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    participant_fields = list(participants[0].keys())
    session_fields = list(sessions[0].keys())
    write_csv(args.output_dir / "participant_manifest.csv", participants, participant_fields)
    write_csv(args.output_dir / "session_manifest.csv", sessions, session_fields)
    policy = {
        "artifact_status": "frozen_goldencheetah_lightweight_longitudinal_subset_v1",
        "source_root": str(args.root.resolve()),
        "source_audit": str(args.audit),
        "seed": args.seed,
        "participants": len(participants),
        "sessions": len(sessions),
        "sessions_per_participant": args.sessions_per_user,
        "raw_files_copied": False,
        "temporal_roles": {
            "support_train": [1, 10],
            "validation": [11, 14],
            "calibration": [15, 17],
            "temporal_test": [18, 20],
        },
        "checks": checks,
        "selected_gender_counts": {
            gender: sum(row["gender"] == gender for row in participants)
            for gender in sorted({str(row["gender"]) for row in participants})
        },
        "selected_mixed_activity_candidates": sum(
            bool(row["mixed_activity_candidate"]) for row in participants
        ),
        "skipped_users_before_completion": skipped_users,
    }
    (args.output_dir / "split_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generated = (
        args.output_dir / "participant_manifest.csv",
        args.output_dir / "session_manifest.csv",
        args.output_dir / "split_policy.json",
    )
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in generated),
        encoding="utf-8",
    )
    print(json.dumps(policy, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
