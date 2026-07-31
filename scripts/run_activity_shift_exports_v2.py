#!/usr/bin/env python3
"""Two-phase, resumable activity-shift forecast export for Amendment 027."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELDOUT_PROTOCOLS = ("AEROBIC", "ANAEROBIC")
BOUNDARIES = ("source_user", "seen_user_activity", "joint_user_activity")
MODELS = ("pk_ssm", "tcn")
POLICIES = ("tagged_events", "evaluation_stride")
SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)
TEST_OPENING_TOKEN = "BSPC_V4_ACTIVITY_TEST_OPEN_20260731"
REQUIRED_BUNDLE_KEYS = {
    "mean",
    "scale",
    "degrees_of_freedom",
    "target",
    "target_valid_mask",
    "current_hr",
}
REQUIRED_METADATA_FIELDS = {
    "row_index",
    "participant_id",
    "session_id",
    "origin_id",
    "protocol",
    "sex",
    "event_types",
    "source_protocol",
    "heldout_protocol",
    "activity_shift_boundary",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def destination(
    output_root: Path,
    heldout: str,
    boundary: str,
    role: str,
    policy: str,
    fold: int,
    model: str,
    seed: int,
) -> Path:
    return (
        output_root
        / heldout
        / boundary
        / role
        / policy
        / f"outer_fold_{fold}"
        / model
        / f"seed_{seed}"
    )


def complete(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "forecast_bundle.npz",
            "origin_metadata.csv",
            "export_report.json",
            "SHA256SUMS.txt",
        )
    )


def require_model_selection_complete(root: Path) -> Path:
    summary_path = root / "model_selection_summary_v2.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("artifact_status") != "source_validation_selection_complete"
        or summary.get("planned_runs") != 100
        or summary.get("complete_runs") != 100
        or summary.get("failed_runs")
    ):
        raise RuntimeError("Source-only model selection is not complete for 100/100 runs")
    return summary_path


def run_export(
    cache_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    log_root: Path,
    heldout: str,
    boundary: str,
    role: str,
    policy: str,
    fold: int,
    model: str,
    seed: int,
    token: str,
) -> dict[str, Any]:
    target = destination(
        output_root, heldout, boundary, role, policy, fold, model, seed
    )
    key = f"{heldout}_{boundary}_{role}_{policy}_fold{fold}_{model}_seed{seed}"
    if complete(target):
        return {"key": key, "status": "already_complete", "seconds": 0.0}

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_activity_shift_forecasts_v1.py"),
        "--outer-fold",
        str(fold),
        "--heldout-protocol",
        heldout,
        "--boundary",
        boundary,
        "--model",
        model,
        "--seed",
        str(seed),
        "--role",
        role,
        "--origin-policy",
        policy,
        "--cache-root",
        str(cache_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--output-root",
        str(output_root),
        "--batch-size",
        "256",
        "--cpu-threads",
        "2",
        "--device",
        "cpu",
    ]
    if role == "test":
        command.extend(["--test-opening-token", token])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    environment["OPENBLAS_NUM_THREADS"] = "2"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{key}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.perf_counter() - started
    if result.returncode != 0 or not complete(target):
        return {
            "key": key,
            "status": "failed",
            "returncode": result.returncode,
            "seconds": elapsed,
            "log": str(log_path),
        }
    return {
        "key": key,
        "status": "completed",
        "seconds": elapsed,
        "log": str(log_path),
    }


def validate_calibration_schema(
    tasks: list[tuple[str, str, str, str, int, str, int]], output_root: Path
) -> dict[str, Any]:
    origins = 0
    participants: set[str] = set()
    bundle_hashes: dict[str, str] = {}
    for heldout, boundary, role, policy, fold, model, seed in tasks:
        folder = destination(
            output_root, heldout, boundary, role, policy, fold, model, seed
        )
        bundle_path = folder / "forecast_bundle.npz"
        metadata_path = folder / "origin_metadata.csv"
        report_path = folder / "export_report.json"
        with np.load(bundle_path, allow_pickle=False) as bundle:
            if set(bundle.files) != REQUIRED_BUNDLE_KEYS:
                raise ValueError(f"Unexpected bundle schema: {bundle_path}")
            n_rows = int(bundle["mean"].shape[0])
            if bundle["mean"].shape != bundle["target"].shape:
                raise ValueError(f"Mean/target shape mismatch: {bundle_path}")
            if bundle["target_valid_mask"].shape != bundle["target"].shape:
                raise ValueError(f"Mask/target shape mismatch: {bundle_path}")
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) != REQUIRED_METADATA_FIELDS:
                raise ValueError(f"Unexpected metadata schema: {metadata_path}")
            rows = list(reader)
        if len(rows) != n_rows:
            raise ValueError(f"Metadata/bundle row mismatch: {folder}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("role") != "calibration":
            raise ValueError(f"Non-calibration artifact in dry run: {report_path}")
        origins += n_rows
        participants.update(row["participant_id"] for row in rows)
        bundle_hashes[str(bundle_path)] = sha256_file(bundle_path)
    return {
        "artifact_status": "pretest_calibration_export_schema_accepted",
        "protocol_amendment": "027",
        "test_targets_accessed": False,
        "validated_exports": len(tasks),
        "origins_across_repeated_exports": origins,
        "distinct_source_calibration_participants": len(participants),
        "required_bundle_keys": sorted(REQUIRED_BUNDLE_KEYS),
        "required_metadata_fields": sorted(REQUIRED_METADATA_FIELDS),
        "bundle_sha256": bundle_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "test"), required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/processed/wearable_activity_shift_v2"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/activity_shift_model_selection_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/activity_shift_locked_evaluation_v2"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--test-opening-token", default="")
    args = parser.parse_args()

    selection_summary_path = require_model_selection_complete(args.checkpoint_root)
    exporter_path = PROJECT_ROOT / "scripts" / "export_activity_shift_forecasts_v1.py"
    builder_path = PROJECT_ROOT / "scripts" / "build_wearable_activity_shift_cache_v2.py"
    amendment_path = PROJECT_ROOT / "PROTOCOL_AMENDMENT_027_ACTIVITY_SHIFT_CACHE_CORRECTION.md"
    schema_path = args.output_root / "pretest_export_schema_v2.json"

    if args.phase == "calibration":
        tasks = [
            (heldout, "seen_user_activity", "calibration", policy, fold, model, seed)
            for heldout in HELDOUT_PROTOCOLS
            for policy in POLICIES
            for fold in range(5)
            for model in MODELS
            for seed in SEEDS
        ]
        token = ""
    else:
        if args.test_opening_token != TEST_OPENING_TOKEN:
            raise PermissionError("Exact activity-shift test-opening token is required")
        if not schema_path.is_file():
            raise FileNotFoundError(schema_path)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if schema.get("artifact_status") != "pretest_calibration_export_schema_accepted":
            raise RuntimeError("Pretest export schema was not accepted")
        current_hashes = {
            "exporter_sha256": sha256_file(exporter_path),
            "builder_sha256": sha256_file(builder_path),
            "amendment_sha256": sha256_file(amendment_path),
            "selection_summary_sha256": sha256_file(selection_summary_path),
        }
        if schema.get("frozen_code_sha256") != current_hashes:
            raise RuntimeError("Frozen exporter or prerequisite hash changed after dry run")
        tasks = [
            (heldout, boundary, "test", policy, fold, model, seed)
            for heldout in HELDOUT_PROTOCOLS
            for boundary in BOUNDARIES
            for policy in POLICIES
            for fold in range(5)
            for model in MODELS
            for seed in SEEDS
        ]
        token = args.test_opening_token

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "_logs" / args.phase
    progress_path = args.output_root / f"{args.phase}_export_progress_v2.jsonl"
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_export,
                args.cache_root,
                args.checkpoint_root,
                args.output_root,
                log_root,
                *task,
                token,
            ): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"[{index}/{len(tasks)}] {record['key']} {record['status']} "
                f"{record['seconds']:.2f}s",
                flush=True,
            )

    failed = [record for record in records if record["status"] == "failed"]
    if args.phase == "calibration" and not failed:
        schema = validate_calibration_schema(tasks, args.output_root)
        schema["frozen_code_sha256"] = {
            "exporter_sha256": sha256_file(exporter_path),
            "builder_sha256": sha256_file(builder_path),
            "amendment_sha256": sha256_file(amendment_path),
            "selection_summary_sha256": sha256_file(selection_summary_path),
        }
        schema_path.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    summary = {
        "artifact_status": (
            f"activity_shift_{args.phase}_exports_complete" if not failed else "incomplete"
        ),
        "phase": args.phase,
        "planned_exports": len(tasks),
        "complete_exports": len(tasks) - len(failed),
        "failed_exports": failed,
        "elapsed_seconds": time.perf_counter() - started,
        "test_opened": args.phase == "test" and not failed,
    }
    summary_path = args.output_root / f"{args.phase}_export_summary_v2.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
