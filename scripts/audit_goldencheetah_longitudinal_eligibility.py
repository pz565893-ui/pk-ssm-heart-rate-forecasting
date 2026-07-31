#!/usr/bin/env python3
"""Audit GoldenCheetah metadata for a lightweight longitudinal HR subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def stable_hash(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scalar_float(value: Any) -> float | None:
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_sport(raw: object) -> str:
    value = re.sub(r"\s+", " ", str(raw or "").strip().lower())
    if any(token in value for token in ("run", "carrera", "corsa")):
        return "run"
    if any(token in value for token in ("walk", "hike", "camin")):
        return "walk_hike"
    if any(token in value for token in ("swim", "nataci")):
        return "swim"
    if any(token in value for token in ("ski", "sci fondo")):
        return "ski"
    if any(
        token in value
        for token in (
            "bike",
            "ride",
            "cycling",
            "ciclismo",
            "giro",
            "pedalada",
            "zwift",
            "trainer",
            "kickr",
            "mtb",
            "road",
            "rolle",
            "rullo",
            "bici",
        )
    ):
        return "bike"
    return "other"


def eligible_ride(ride: dict[str, Any]) -> tuple[bool, str]:
    metrics = ride.get("METRICS") or {}
    average_hr = scalar_float(metrics.get("average_hr"))
    duration = scalar_float(metrics.get("workout_time"))
    valid = bool(
        average_hr is not None
        and 30.0 <= average_hr <= 220.0
        and duration is not None
        and duration >= 600.0
    )
    return valid, canonical_sport(ride.get("sport"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/goldencheetah_longitudinal_audit_v1.json"),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    if not args.root.is_dir():
        raise FileNotFoundError(args.root)

    users: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for user_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        json_files = sorted(user_dir.glob("*.json"))
        csv_count = sum(1 for _ in user_dir.glob("*.csv"))
        if len(json_files) != 1:
            parse_errors.append(
                {
                    "user_id": user_dir.name,
                    "error": f"expected one JSON file, found {len(json_files)}",
                }
            )
            continue
        try:
            payload = json.loads(json_files[0].read_text(encoding="utf-8-sig"))
        except Exception as exc:
            parse_errors.append({"user_id": user_dir.name, "error": str(exc)})
            continue

        sport_counts: Counter[str] = Counter()
        eligible_count = 0
        rides = payload.get("RIDES") or []
        for ride in rides:
            valid, sport = eligible_ride(ride)
            if valid:
                eligible_count += 1
                sport_counts[sport] += 1
        athlete = payload.get("ATHLETE") or {}
        gender = str(athlete.get("gender") or "unknown").strip().lower()
        users.append(
            {
                "user_id": user_dir.name,
                "gender": gender,
                "rides": len(rides),
                "csv_files": csv_count,
                "ride_csv_count_match": len(rides) == csv_count,
                "eligible_sessions": eligible_count,
                "sport_counts": dict(sorted(sport_counts.items())),
                "selection_hash": stable_hash(args.seed, user_dir.name),
            }
        )

    eligible_counts = sorted(
        (int(user["eligible_sessions"]) for user in users), reverse=True
    )
    gender_counts = Counter(str(user["gender"]) for user in users)
    sport_totals: Counter[str] = Counter()
    for user in users:
        sport_totals.update(user["sport_counts"])

    longitudinal_candidates = [
        user
        for user in users
        if user["ride_csv_count_match"] and user["eligible_sessions"] >= 30
    ]
    mixed_activity_candidates = [
        user
        for user in longitudinal_candidates
        if user["sport_counts"].get("bike", 0) >= 10
        and user["sport_counts"].get("run", 0) >= 5
    ]
    report = {
        "artifact_status": "metadata_only_longitudinal_eligibility_audit",
        "source_root": str(args.root.resolve()),
        "seed": args.seed,
        "selection_policy_proposal": {
            "users": 30,
            "sessions_per_user": 20,
            "minimum_metadata_eligible_sessions": 30,
            "minimum_duration_seconds": 600,
            "valid_average_hr_bpm": [30, 220],
            "preferred_mixed_activity_minimums": {"bike": 10, "run": 5},
            "copy_raw_files": False,
        },
        "summary": {
            "users_parsed": len(users),
            "json_parse_errors": len(parse_errors),
            "ride_csv_count_matches": sum(
                bool(user["ride_csv_count_match"]) for user in users
            ),
            "eligible_sessions": sum(eligible_counts),
            "users_ge_5": sum(count >= 5 for count in eligible_counts),
            "users_ge_10": sum(count >= 10 for count in eligible_counts),
            "users_ge_20": sum(count >= 20 for count in eligible_counts),
            "users_ge_30": sum(count >= 30 for count in eligible_counts),
            "median_eligible_sessions": (
                eligible_counts[len(eligible_counts) // 2] if eligible_counts else 0
            ),
            "longitudinal_candidates": len(longitudinal_candidates),
            "mixed_activity_candidates": len(mixed_activity_candidates),
            "gender_counts": dict(sorted(gender_counts.items())),
            "canonical_sport_totals": dict(sorted(sport_totals.items())),
        },
        "longitudinal_candidates": sorted(
            longitudinal_candidates,
            key=lambda user: (user["selection_hash"], user["user_id"]),
        ),
        "mixed_activity_candidates": sorted(
            mixed_activity_candidates,
            key=lambda user: (user["selection_hash"], user["user_id"]),
        ),
        "parse_errors": parse_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
