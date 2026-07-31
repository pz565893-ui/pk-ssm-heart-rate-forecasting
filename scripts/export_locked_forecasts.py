"""Export immutable forecast bundles from frozen validation-selected checkpoints.

This script intentionally separates model inference from conformal calibration and
test scoring. Test-role access requires an explicit opening token that is frozen in
Protocol Amendment 014.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import t as student_t
from torch.utils.data import DataLoader

from scripts.run_pretest_model_selection import load_candidate
from transition_forecasting.dataset import TransitionWindowDataset, move_batch_to_device
from transition_forecasting.training import (
    build_model,
    forward_model,
    parameter_count,
    set_reproducible_seed,
)


TEST_OPENING_TOKEN = "BSPC_V4_TEST_OPEN_20260731"
HORIZON_SECONDS = (30, 60, 120)
TRAINABLE_MODELS = {
    "pk_ssm",
    "tcn",
    "gru",
    "lstm",
    "transformer",
    "residual_ssm",
    "first_order_kinetics",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--model",
        choices=sorted(TRAINABLE_MODELS | {"persistence", "damped_trend"}),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", choices=("validation", "calibration", "test"), required=True)
    parser.add_argument(
        "--origin-policy",
        choices=("tagged_events", "evaluation_stride"),
        required=True,
    )
    parser.add_argument("--candidate-id", default="pkssm_64x4_r6")
    parser.add_argument("--candidate-grid", type=Path, default=Path("configs/pk_ssm_candidate_grid_v1.json"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/processed/wearable_exercise_transition_v2"))
    parser.add_argument(
        "--pkssm-checkpoint-root",
        type=Path,
        default=Path("outputs/pretest_model_selection_v4_seedfix"),
    )
    parser.add_argument(
        "--baseline-checkpoint-root",
        type=Path,
        default=Path("outputs/pretest_baselines_v4_seedfix"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--test-opening-token", default="")
    return parser.parse_args()


def checkpoint_directory(args: argparse.Namespace) -> Path:
    root = args.pkssm_checkpoint_root if args.model == "pk_ssm" else args.baseline_checkpoint_root
    return (
        root
        / f"outer_fold_{args.outer_fold}"
        / args.model
        / args.candidate_id
        / "signal_only"
        / "known_activity"
        / "history_none_budget_0"
        / f"seed_{args.seed}"
    )


def output_directory(args: argparse.Namespace) -> Path:
    return (
        args.output_root
        / args.role
        / args.origin_policy
        / f"outer_fold_{args.outer_fold}"
        / args.model
        / f"seed_{args.seed}"
    )


def participant_macro(values: np.ndarray, valid: np.ndarray, participants: np.ndarray) -> float:
    participant_values: list[float] = []
    for participant in np.unique(participants):
        rows = participants == participant
        selected = values[rows]
        selected_valid = valid[rows]
        if np.any(selected_valid):
            participant_values.append(float(np.mean(selected[selected_valid])))
    return float(np.mean(participant_values)) if participant_values else float("nan")


def metric_summary(
    mean: np.ndarray,
    scale: np.ndarray,
    degrees_of_freedom: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    participants: np.ndarray,
) -> dict[str, Any]:
    error = mean - target
    absolute_error = np.abs(error)
    squared_error = np.square(error)
    summary: dict[str, Any] = {
        "participant_macro_trajectory_mae_bpm": participant_macro(
            absolute_error, valid, participants
        ),
        "participant_macro_signed_error_bpm": participant_macro(error, valid, participants),
    }

    participant_rmse: list[float] = []
    for participant in np.unique(participants):
        rows = participants == participant
        selected_valid = valid[rows]
        if np.any(selected_valid):
            participant_rmse.append(
                float(np.sqrt(np.mean(squared_error[rows][selected_valid])))
            )
    summary["participant_macro_trajectory_rmse_bpm"] = (
        float(np.mean(participant_rmse)) if participant_rmse else float("nan")
    )

    for horizon in HORIZON_SECONDS:
        index = horizon - 1
        horizon_valid = valid[:, index]
        summary[f"participant_macro_mae_{horizon}s_bpm"] = participant_macro(
            absolute_error[:, index], horizon_valid, participants
        )

    pair_valid = valid[:, 1:] & valid[:, :-1]
    observed_tv = np.sum(np.abs(np.diff(target, axis=1)) * pair_valid, axis=1)
    predicted_tv = np.sum(np.abs(np.diff(mean, axis=1)) * pair_valid, axis=1)
    tv_ratio = np.divide(
        predicted_tv,
        observed_tv,
        out=np.full_like(predicted_tv, np.nan, dtype=np.float64),
        where=observed_tv > 1e-8,
    )
    origin_valid = np.isfinite(tv_ratio)
    summary["participant_macro_total_variation_ratio"] = participant_macro(
        tv_ratio, origin_valid, participants
    )

    quantile = student_t.ppf(0.975, np.maximum(degrees_of_freedom, 2.100001))
    half_width = quantile * np.maximum(scale, 1e-6)
    covered = (target >= mean - half_width) & (target <= mean + half_width)
    summary["raw_student_t_participant_macro_point_coverage"] = participant_macro(
        covered.astype(np.float64), valid, participants
    )
    summary["raw_student_t_mean_interval_width_bpm"] = float(
        np.mean((2.0 * half_width)[valid])
    )
    for horizon in HORIZON_SECONDS:
        index = horizon - 1
        horizon_valid = valid[:, index]
        summary[f"raw_student_t_participant_macro_coverage_{horizon}s"] = participant_macro(
            covered[:, index].astype(np.float64), horizon_valid, participants
        )
    return summary


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "row_index",
        "participant_id",
        "session_id",
        "origin_id",
        "protocol",
        "sex",
        "event_types",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.role == "test" and args.test_opening_token != TEST_OPENING_TOKEN:
        raise PermissionError("Test role is sealed; the frozen opening token was not supplied.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    destination = output_directory(args)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Immutable output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.cpu_threads)
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    fold_cache = args.cache_root / f"outer_fold_{args.outer_fold}"
    dataset = TransitionWindowDataset(
        fold_cache_dir=fold_cache,
        role=args.role,
        origin_policy=args.origin_policy,
        schedule_aware=False,
        mask_activity_identity=False,
        history_regime="none",
        history_session_budget=0,
    )
    if len(dataset) == 0:
        raise RuntimeError("The requested role and origin policy produced no valid origins.")

    _, candidate = load_candidate(args.candidate_grid, args.candidate_id)
    checkpoint_dir = checkpoint_directory(args)
    checkpoint_path = checkpoint_dir / "validation_selected_checkpoint.pt"
    selection_report_path = checkpoint_dir / "selection_report.json"
    if selection_report_path.exists():
        selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
        hr_feature_scale_bpm = float(selection_report["hr_feature_scale_bpm"])
    else:
        normalization = json.loads(
            (fold_cache / "training_normalization.json").read_text(encoding="utf-8")
        )
        hr_feature_scale_bpm = float(normalization["hr"]["scale"])

    input_dim = int(dataset[0]["context"].shape[-1])
    model = build_model(
        args.model,
        input_dim=input_dim,
        candidate=candidate,
        hr_feature_scale_bpm=hr_feature_scale_bpm,
    ).to(device)
    if args.model in TRAINABLE_MODELS:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Frozen checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    dfs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    current_hr: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    kinetic_parameters: dict[str, list[np.ndarray]] = {}

    start = time.perf_counter()
    row_index = 0
    with torch.inference_mode():
        for batch in loader:
            device_batch = move_batch_to_device(batch, device)
            output = forward_model(args.model, model, device_batch)
            mean_tensor = output.mean
            scale_tensor = output.scale
            df_tensor = output.degrees_of_freedom
            means.append(mean_tensor.detach().cpu().numpy().astype(np.float32))
            scales.append(scale_tensor.detach().cpu().numpy().astype(np.float32))
            dfs.append(df_tensor.detach().cpu().numpy().astype(np.float32))
            targets.append(batch["target"].numpy().astype(np.float32))
            masks.append(batch["target_valid_mask"].numpy().astype(bool))
            current_hr.append(batch["current_hr"].numpy().astype(np.float32))

            parameter_mapping = getattr(output, "parameters", None)
            if isinstance(parameter_mapping, dict):
                for name, value in parameter_mapping.items():
                    if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == mean_tensor.shape[0]:
                        kinetic_parameters.setdefault(name, []).append(
                            value.detach().cpu().numpy().astype(np.float32)
                        )

            batch_size = int(mean_tensor.shape[0])
            for index in range(batch_size):
                metadata.append(
                    {
                        "row_index": str(row_index),
                        "participant_id": str(batch["participant_id"][index]),
                        "session_id": str(batch["session_id"][index]),
                        "origin_id": str(batch["origin_id"][index]),
                        "protocol": str(batch["protocol"][index]),
                        "sex": str(batch["sex"][index]),
                        "event_types": str(batch["event_types"][index]),
                    }
                )
                row_index += 1
    elapsed_seconds = time.perf_counter() - start

    mean_array = np.concatenate(means, axis=0)
    scale_array = np.concatenate(scales, axis=0)
    df_array = np.concatenate(dfs, axis=0)
    target_array = np.concatenate(targets, axis=0)
    mask_array = np.concatenate(masks, axis=0)
    current_hr_array = np.concatenate(current_hr, axis=0)
    participants = np.asarray([row["participant_id"] for row in metadata], dtype=str)

    bundle_path = destination / "forecast_bundle.npz"
    np.savez_compressed(
        bundle_path,
        mean=mean_array,
        scale=scale_array,
        degrees_of_freedom=df_array,
        target=target_array,
        target_valid_mask=mask_array.astype(np.uint8),
        current_hr=current_hr_array,
    )
    metadata_path = destination / "origin_metadata.csv"
    write_metadata(metadata_path, metadata)

    parameter_path: Path | None = None
    if kinetic_parameters:
        parameter_path = destination / "kinetic_parameters.npz"
        np.savez_compressed(
            parameter_path,
            **{name: np.concatenate(parts, axis=0) for name, parts in kinetic_parameters.items()},
        )

    report = {
        "artifact_status": "locked_forecast_export",
        "outer_fold": args.outer_fold,
        "model": args.model,
        "candidate_id": args.candidate_id,
        "seed": args.seed,
        "role": args.role,
        "origin_policy": args.origin_policy,
        "test_opening_token_required": args.role == "test",
        "history_regime": "none",
        "origins": int(mean_array.shape[0]),
        "horizon_seconds": int(mean_array.shape[1]),
        "participants": int(len(np.unique(participants))),
        "trainable_parameters": int(parameter_count(model)),
        "checkpoint": str(checkpoint_path) if args.model in TRAINABLE_MODELS else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
        "candidate_grid_sha256": sha256_file(args.candidate_grid),
        "cache_policy_sha256": sha256_file(fold_cache / "cache_policy.json"),
        "metrics": metric_summary(
            mean_array,
            scale_array,
            df_array,
            target_array,
            mask_array,
            participants,
        ),
        "runtime": {
            "device": str(device),
            "cpu_threads": args.cpu_threads,
            "batch_size": args.batch_size,
            "end_to_end_seconds": elapsed_seconds,
            "origins_per_second": float(mean_array.shape[0] / elapsed_seconds),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    report_path = destination / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    checksums = {
        bundle_path.name: sha256_file(bundle_path),
        metadata_path.name: sha256_file(metadata_path),
        report_path.name: sha256_file(report_path),
    }
    if parameter_path is not None:
        checksums[parameter_path.name] = sha256_file(parameter_path)
    checksum_path = destination / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
