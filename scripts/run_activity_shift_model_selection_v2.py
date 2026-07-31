#!/usr/bin/env python3
"""Run resumable source-only activity-shift model selection for Amendment 027."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELDOUT_PROTOCOLS = ("AEROBIC", "ANAEROBIC")
MODELS = ("pk_ssm", "tcn")
SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)
CANDIDATE_ID = "pkssm_64x4_r6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_directory(
    output_root: Path, heldout: str, fold: int, model: str, seed: int
) -> Path:
    return (
        output_root
        / heldout
        / f"outer_fold_{fold}"
        / model
        / CANDIDATE_ID
        / "signal_only"
        / "masked_activity"
        / "history_none_budget_0"
        / f"seed_{seed}"
    )


def is_complete(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "validation_selected_checkpoint.pt",
            "selection_report.json",
            "SHA256SUMS.txt",
        )
    )


def run_one(
    cache_root: Path,
    output_root: Path,
    log_root: Path,
    heldout: str,
    fold: int,
    model: str,
    seed: int,
    cpu_threads: int,
) -> dict[str, Any]:
    destination = output_directory(output_root, heldout, fold, model, seed)
    key = f"{heldout}_fold{fold}_{model}_seed{seed}"
    if is_complete(destination):
        return {"key": key, "status": "already_complete", "seconds": 0.0}

    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{key}.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pretest_model_selection.py"),
        "--outer-fold",
        str(fold),
        "--model",
        model,
        "--candidate-id",
        CANDIDATE_ID,
        "--seed",
        str(seed),
        "--batch-size",
        "64",
        "--mask-activity-identity",
        "--history-regime",
        "none",
        "--history-session-budget",
        "0",
        "--cache-root",
        str(cache_root / heldout / "seen_user_activity"),
        "--output-root",
        str(output_root / heldout),
        "--device",
        "cpu",
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(cpu_threads)
    environment["MKL_NUM_THREADS"] = str(cpu_threads)
    environment["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            "key": key,
            "status": "failed",
            "returncode": completed.returncode,
            "seconds": elapsed,
            "log": str(log_path),
        }
    if not is_complete(destination):
        return {
            "key": key,
            "status": "failed_missing_artifacts",
            "seconds": elapsed,
            "log": str(log_path),
        }
    return {
        "key": key,
        "status": "completed",
        "seconds": elapsed,
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/processed/wearable_activity_shift_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/activity_shift_model_selection_v2"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads-per-run", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.cpu_threads_per_run < 1:
        raise ValueError("Worker and thread counts must be positive")

    tasks = [
        (heldout, fold, model, seed)
        for heldout in HELDOUT_PROTOCOLS
        for fold in range(5)
        for model in MODELS
        for seed in SEEDS
    ]
    if args.limit > 0:
        tasks = tasks[: args.limit]
    log_root = args.output_root / "_logs"
    progress_path = args.output_root / "model_selection_progress_v2.jsonl"
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                args.cache_root,
                args.output_root,
                log_root,
                heldout,
                fold,
                model,
                seed,
                args.cpu_threads_per_run,
            ): (heldout, fold, model, seed)
            for heldout, fold, model, seed in tasks
        }
        for completed_count, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"[{completed_count}/{len(tasks)}] {record['key']} "
                f"{record['status']} {record['seconds']:.1f}s",
                flush=True,
            )

    failed = [record for record in records if record["status"].startswith("failed")]
    completed_directories = [
        output_directory(args.output_root, heldout, fold, model, seed)
        for heldout, fold, model, seed in tasks
    ]
    reports = [path / "selection_report.json" for path in completed_directories]
    summary = {
        "artifact_status": (
            "source_validation_selection_complete" if not failed else "incomplete"
        ),
        "protocol_amendment": "027",
        "cache_root": str(args.cache_root),
        "cache_summary_sha256": sha256_file(
            args.cache_root / "activity_shift_cache_summary.json"
        ),
        "models": list(MODELS),
        "heldout_protocols": list(HELDOUT_PROTOCOLS),
        "folds": list(range(5)),
        "seeds": list(SEEDS),
        "history_regime": "none",
        "activity_identity": "masked",
        "planned_runs": len(tasks),
        "complete_runs": sum(is_complete(path) for path in completed_directories),
        "failed_runs": failed,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_report_sha256": {
            str(path): sha256_file(path) for path in reports if path.is_file()
        },
        "test_roles_accessed": [],
    }
    summary_path = args.output_root / "model_selection_summary_v2.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
