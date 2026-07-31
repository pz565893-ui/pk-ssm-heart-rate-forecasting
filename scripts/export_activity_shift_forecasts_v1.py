#!/usr/bin/env python3
"""Export immutable predictions for the frozen Wearable v4 activity shift."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_locked_forecasts import (  # noqa: E402
    TRAINABLE_MODELS,
    metric_summary,
    sha256_file,
)
from scripts.run_pretest_model_selection import load_candidate  # noqa: E402
from transition_forecasting.dataset import (  # noqa: E402
    TransitionWindowDataset,
    move_batch_to_device,
)
from transition_forecasting.training import (  # noqa: E402
    build_model,
    forward_model,
    parameter_count,
    set_reproducible_seed,
)


PROTOCOLS = ("AEROBIC", "ANAEROBIC")
BOUNDARIES = ("source_user", "seen_user_activity", "joint_user_activity")
TEST_OPENING_TOKEN = "BSPC_V4_ACTIVITY_TEST_OPEN_20260731"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--heldout-protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--boundary", choices=BOUNDARIES, required=True)
    parser.add_argument(
        "--model",
        choices=sorted(TRAINABLE_MODELS | {"persistence", "damped_trend"}),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", choices=("validation", "calibration", "test"), required=True)
    parser.add_argument(
        "--origin-policy", choices=("tagged_events", "evaluation_stride"), required=True
    )
    parser.add_argument("--candidate-id", default="pkssm_64x4_r6")
    parser.add_argument(
        "--candidate-grid", type=Path, default=Path("configs/pk_ssm_candidate_grid_v1.json")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/processed/wearable_activity_shift_v1")
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("outputs/activity_shift_model_selection_v1")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/activity_shift_locked_evaluation_v1")
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--test-opening-token", default="")
    return parser.parse_args()


def checkpoint_directory(args: argparse.Namespace) -> Path:
    return (
        args.checkpoint_root
        / args.heldout_protocol
        / f"outer_fold_{args.outer_fold}"
        / args.model
        / args.candidate_id
        / "signal_only"
        / "masked_activity"
        / "history_none_budget_0"
        / f"seed_{args.seed}"
    )


def output_directory(args: argparse.Namespace) -> Path:
    return (
        args.output_root
        / args.heldout_protocol
        / args.boundary
        / args.role
        / args.origin_policy
        / f"outer_fold_{args.outer_fold}"
        / args.model
        / f"seed_{args.seed}"
    )


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.role == "test" and args.test_opening_token != TEST_OPENING_TOKEN:
        raise PermissionError("Activity-shift test is sealed; opening token missing")
    if args.role != "test" and args.boundary != "seen_user_activity":
        raise ValueError("Validation and calibration are exported once from seen_user_activity")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_protocol = next(value for value in PROTOCOLS if value != args.heldout_protocol)
    destination = output_directory(args)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Immutable destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.cpu_threads)
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    fold_cache = (
        args.cache_root
        / args.heldout_protocol
        / args.boundary
        / f"outer_fold_{args.outer_fold}"
    )
    dataset = TransitionWindowDataset(
        fold_cache_dir=fold_cache,
        role=args.role,
        origin_policy=args.origin_policy,
        schedule_aware=False,
        mask_activity_identity=True,
        history_regime="none",
        history_session_budget=0,
    )
    _, candidate = load_candidate(args.candidate_grid, args.candidate_id)
    checkpoint_dir = checkpoint_directory(args)
    checkpoint_path = checkpoint_dir / "validation_selected_checkpoint.pt"
    selection_report_path = checkpoint_dir / "selection_report.json"
    normalization = json.loads(
        (fold_cache / "training_normalization.json").read_text(encoding="utf-8")
    )
    hr_scale = float(normalization["hr_bpm"]["std"])
    if selection_report_path.is_file():
        selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
        hr_scale = float(selection_report["hr_feature_scale_bpm"])
    model = build_model(
        args.model,
        input_dim=dataset.input_dim,
        candidate=candidate,
        hr_feature_scale_bpm=hr_scale,
    ).to(device)
    if args.model in TRAINABLE_MODELS:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True
        )
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    arrays: dict[str, list[np.ndarray]] = {
        "mean": [],
        "scale": [],
        "degrees_of_freedom": [],
        "target": [],
        "target_valid_mask": [],
        "current_hr": [],
    }
    metadata: list[dict[str, str]] = []
    row_index = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            output = forward_model(args.model, model, batch)
            arrays["mean"].append(output.mean.detach().cpu().numpy().astype(np.float32))
            arrays["scale"].append(output.scale.detach().cpu().numpy().astype(np.float32))
            arrays["degrees_of_freedom"].append(
                output.degrees_of_freedom.detach().cpu().numpy().astype(np.float32)
            )
            arrays["target"].append(raw_batch["target"].numpy().astype(np.float32))
            arrays["target_valid_mask"].append(
                raw_batch["target_valid_mask"].numpy().astype(np.uint8)
            )
            arrays["current_hr"].append(raw_batch["current_hr"].numpy().astype(np.float32))
            for index in range(len(raw_batch["participant_id"])):
                metadata.append(
                    {
                        "row_index": str(row_index),
                        "participant_id": str(raw_batch["participant_id"][index]),
                        "session_id": str(raw_batch["session_id"][index]),
                        "origin_id": str(raw_batch["origin_id"][index]),
                        "protocol": str(raw_batch["protocol"][index]),
                        "sex": str(raw_batch["sex"][index]),
                        "event_types": str(raw_batch["event_types"][index]),
                        "source_protocol": source_protocol,
                        "heldout_protocol": args.heldout_protocol,
                        "activity_shift_boundary": args.boundary,
                    }
                )
                row_index += 1
    elapsed = time.perf_counter() - start
    bundle = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    bundle_path = destination / "forecast_bundle.npz"
    np.savez_compressed(bundle_path, **bundle)
    metadata_path = destination / "origin_metadata.csv"
    write_metadata(metadata_path, metadata)
    participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)
    report = {
        "artifact_status": "locked_wearable_v4_activity_shift_forecast",
        "outer_fold": args.outer_fold,
        "source_protocol": source_protocol,
        "heldout_protocol": args.heldout_protocol,
        "activity_shift_boundary": args.boundary,
        "model": args.model,
        "seed": args.seed,
        "role": args.role,
        "origin_policy": args.origin_policy,
        "history_regime": "none",
        "activity_identity": "masked",
        "origins": int(bundle["mean"].shape[0]),
        "participants": int(np.unique(participants).size),
        "trainable_parameters": int(parameter_count(model)),
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
        ),
        "cache_policy_sha256": sha256_file(fold_cache / "cache_policy.json"),
        "metrics": metric_summary(
            bundle["mean"],
            bundle["scale"],
            bundle["degrees_of_freedom"],
            bundle["target"],
            bundle["target_valid_mask"].astype(bool),
            participants,
        ),
        "runtime": {
            "device": str(device),
            "cpu_threads": args.cpu_threads,
            "batch_size": args.batch_size,
            "end_to_end_seconds": elapsed,
            "origins_per_second": float(bundle["mean"].shape[0] / elapsed),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    report_path = destination / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (bundle_path, metadata_path, report_path)
        ),
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
