"""Evaluate frozen Ridge summary-feature parameters on locked v4 roles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from scripts.run_ridge_summary_baseline import collect
from transition_forecasting.dataset import TransitionWindowDataset


TEST_OPENING_TOKEN = "BSPC_V4_TEST_OPEN_20260731"
HORIZONS = (30, 60, 120)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--role", choices=("validation", "test"), required=True)
    parser.add_argument("--origin-policy", choices=("tagged_events", "evaluation_stride"), required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("data/processed/wearable_exercise_transition_v2"))
    parser.add_argument("--parameter-root", type=Path, default=Path("outputs/wearable_v4_pretest_seedfix"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-opening-token", default="")
    return parser.parse_args()


def participant_macro(values: np.ndarray, valid: np.ndarray, participants: np.ndarray) -> float:
    summaries: list[float] = []
    for participant in np.unique(participants):
        selected = participants == participant
        selected_values = values[selected]
        selected_valid = valid[selected]
        if np.any(selected_valid):
            summaries.append(float(np.mean(selected_values[selected_valid])))
    return float(np.mean(summaries)) if summaries else float("nan")


def summarize(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    participants: np.ndarray,
    high_threshold: float,
) -> dict[str, Any]:
    error = prediction - target
    absolute_error = np.abs(error)
    squared_error = np.square(error)
    participant_rmse: list[float] = []
    for participant in np.unique(participants):
        selected = participants == participant
        selected_valid = valid[selected]
        if np.any(selected_valid):
            participant_rmse.append(float(np.sqrt(np.mean(squared_error[selected][selected_valid]))))
    pair_valid = valid[:, 1:] & valid[:, :-1]
    observed_tv = np.sum(np.abs(np.diff(target, axis=1)) * pair_valid, axis=1)
    predicted_tv = np.sum(np.abs(np.diff(prediction, axis=1)) * pair_valid, axis=1)
    tv_ratio = np.divide(
        predicted_tv,
        observed_tv,
        out=np.full_like(predicted_tv, np.nan),
        where=observed_tv > 1e-8,
    )
    result: dict[str, Any] = {
        "participant_macro_trajectory_mae_bpm": participant_macro(absolute_error, valid, participants),
        "participant_macro_trajectory_rmse_bpm": float(np.mean(participant_rmse)),
        "participant_macro_signed_error_bpm": participant_macro(error, valid, participants),
        "participant_macro_total_variation_ratio": participant_macro(
            tv_ratio, np.isfinite(tv_ratio), participants
        ),
    }
    for horizon in HORIZONS:
        index = horizon - 1
        result[f"participant_macro_mae_{horizon}s_bpm"] = participant_macro(
            absolute_error[:, index], valid[:, index], participants
        )
    high_mask = valid & (target >= high_threshold)
    fixed_high_mask = valid & (target >= 160.0)
    result["participant_macro_high_hr_mae_bpm"] = participant_macro(
        absolute_error, high_mask, participants
    )
    result["participant_macro_fixed_160_high_hr_mae_bpm"] = participant_macro(
        absolute_error, fixed_high_mask, participants
    )
    result["high_hr_valid_points"] = int(np.sum(high_mask))
    result["fixed_160_high_hr_valid_points"] = int(np.sum(fixed_high_mask))
    return result


def high_hr_threshold(fold_cache: Path) -> float:
    values: list[np.ndarray] = []
    for path in sorted((fold_cache / "sessions").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["role"].item()) not in {"train", "calibration"}:
                continue
            valid = archive["hr_range_valid"].astype(bool) & archive["hr_offline_qc_valid"].astype(bool)
            if np.any(valid):
                values.append(archive["hr_bpm"][valid].astype(np.float64))
    if not values:
        raise ValueError("No valid training or calibration HR values were found.")
    return float(np.quantile(np.concatenate(values), 0.90))


def main() -> None:
    args = parse_args()
    if args.role == "test" and args.test_opening_token != TEST_OPENING_TOKEN:
        raise PermissionError("Test role is sealed; the frozen opening token was not supplied.")
    destination = (
        args.output_root
        / args.role
        / args.origin_policy
        / f"outer_fold_{args.outer_fold}"
        / "ridge_summary"
    )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Immutable output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    features, target, valid, participant_ids = collect(loader)
    participants = np.asarray(participant_ids, dtype=str)

    parameter_path = (
        args.parameter_root
        / f"outer_fold_{args.outer_fold}"
        / "ridge_summary"
        / "alpha_1"
        / "clip_30_220"
        / "seed_20260730"
        / "ridge_parameters.npz"
    )
    with np.load(parameter_path, allow_pickle=False) as archive:
        coefficient = archive["coefficient"].astype(np.float64)
        intercept = archive["intercept"].astype(np.float64)
        scaler_mean = archive["scaler_mean"].astype(np.float64)
        scaler_scale = archive["scaler_scale"].astype(np.float64)
    scaled_features = (features - scaler_mean) / np.where(scaler_scale > 0.0, scaler_scale, 1.0)
    raw_prediction = scaled_features @ coefficient.T + intercept
    clipped_prediction = np.clip(raw_prediction, 30.0, 220.0)
    threshold = high_hr_threshold(fold_cache)

    bundle_path = destination / "ridge_forecast_bundle.npz"
    np.savez_compressed(
        bundle_path,
        raw_mean=raw_prediction.astype(np.float32),
        clipped_mean=clipped_prediction.astype(np.float32),
        target=target.astype(np.float32),
        target_valid_mask=valid.astype(np.uint8),
    )
    metadata_path = destination / "origin_metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_index", "participant_id"))
        writer.writerows((index, participant) for index, participant in enumerate(participants))

    report = {
        "artifact_status": "locked_ridge_summary_evaluation",
        "outer_fold": args.outer_fold,
        "role": args.role,
        "origin_policy": args.origin_policy,
        "origins": int(target.shape[0]),
        "participants": int(len(np.unique(participants))),
        "features": int(features.shape[1]),
        "parameters": int(coefficient.size + intercept.size),
        "high_hr_threshold_bpm": threshold,
        "parameter_path": str(parameter_path),
        "parameter_sha256": sha256_file(parameter_path),
        "raw": summarize(raw_prediction, target, valid, participants, threshold),
        "clipped_30_220": summarize(clipped_prediction, target, valid, participants, threshold),
    }
    report_path = destination / "ridge_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    checksum_path = destination / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (bundle_path, metadata_path, report_path)
        ),
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
