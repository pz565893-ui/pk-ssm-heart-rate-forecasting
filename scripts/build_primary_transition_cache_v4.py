#!/usr/bin/env python3
"""Build history-ready fold caches from the frozen wearable-exercise v4 split."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transition_forecasting.data_pipeline import (
    PreprocessingConfig,
    build_origin_rows,
    build_session_signals,
    discover_session_directories,
    load_fold_roles,
    save_session_cache,
    sha256_file,
    training_normalization,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/wearable_exercise/s3_subset/Wearable_Dataset"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("splits/wearable_exercise_v4/outer_fold_roles.csv"),
    )
    parser.add_argument("--outer-fold", type=int, required=True, choices=range(5))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/wearable_exercise_transition_v2"),
    )
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="Resume a fold directory that lacks the complete final manifest set.",
    )
    args = parser.parse_args()

    if "wearable_exercise_v4" not in args.split_manifest.as_posix():
        raise ValueError("The v4 cache builder accepts only the frozen v4 split")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    if not args.split_manifest.is_file():
        raise FileNotFoundError(args.split_manifest)
    output_dir = args.output_root / f"outer_fold_{args.outer_fold}"
    final_artifacts = (
        "cache_policy.json",
        "event_manifest.csv",
        "origin_manifest.csv",
        "training_normalization.json",
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        is_complete = all((output_dir / name).is_file() for name in final_artifacts)
        if is_complete or not args.resume_partial:
            raise FileExistsError(f"Refusing to overwrite versioned v4 cache: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PreprocessingConfig()
    roles = load_fold_roles(args.split_manifest, args.outer_fold)
    session_dirs = discover_session_directories(args.data_dir)
    if not session_dirs:
        raise ValueError("No exercise sessions with HR.csv and ACC.csv were discovered")

    sessions = []
    origin_rows = []
    event_rows = []
    source_hashes = {}
    for session_dir in session_dirs:
        session = build_session_signals(session_dir, args.data_dir, roles, config)
        sessions.append(session)
        session_origins, session_events = build_origin_rows(session, config)
        origin_rows.extend(session_origins)
        event_rows.extend(session_events)
        save_session_cache(
            session,
            output_dir / "sessions" / f"{session.session_id}.npz",
        )
        for source in (session.source_hr_path, session.source_acc_path, session.source_tags_path):
            if source:
                source_path = Path(source)
                source_hashes[source_path.as_posix()] = sha256_file(source_path)

    origin_rows.sort(
        key=lambda row: (
            row["participant_id"], row["session_id"], int(row["origin_unix_second"])
        )
    )
    event_rows.sort(
        key=lambda row: (
            row["participant_id"], row["session_id"], int(row["origin_unix_second"]), row["event_type"]
        )
    )
    write_csv(output_dir / "origin_manifest.csv", origin_rows)
    write_csv(output_dir / "event_manifest.csv", event_rows)
    write_json(output_dir / "training_normalization.json", training_normalization(sessions))
    summary = {
        "artifact_status": "v4_fold_specific_history_ready_cache",
        "outer_fold": args.outer_fold,
        "split_manifest": args.split_manifest.as_posix(),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "preprocessing_config": asdict(config),
        "sessions": len(sessions),
        "participants": len({session.participant_id for session in sessions}),
        "roles": {
            role: len({session.participant_id for session in sessions if session.role == role})
            for role in ("train", "validation", "calibration", "test")
        },
        "origins": len(origin_rows),
        "evaluation_stride_origins": sum(int(row["evaluation_stride_origin"]) for row in origin_rows),
        "tagged_event_origins": sum(int(row["tagged_event_origin"]) for row in origin_rows),
        "source_sha256": dict(sorted(source_hashes.items())),
        "history_contract": (
            "TransitionWindowDataset assembles same-participant sessions whose cached end timestamp "
            "is strictly before the target session start and applies the declared history regime"
        ),
        "rules": [
            "v4 participant roles loaded before origin generation",
            "normalization fitted on training participants only",
            "history is assembled after split assignment",
            "future observed HR and acceleration are absent from model inputs",
        ],
    }
    write_json(output_dir / "cache_policy.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
