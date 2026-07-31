#!/usr/bin/env python3
"""Fit a fixed Ridge context-summary baseline on training and validation roles only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from transition_forecasting.dataset import TransitionWindowDataset
from transition_forecasting.training import set_reproducible_seed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary_features(context: np.ndarray, current_hr: np.ndarray) -> np.ndarray:
    recent = context[:, -30:, :]
    early = context[:, :30, :]
    return np.concatenate(
        (
            context[:, -1, :],
            context.mean(axis=1),
            context.std(axis=1),
            recent.mean(axis=1) - early.mean(axis=1),
            current_hr[:, None],
        ),
        axis=1,
    )


def collect(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    participants: list[str] = []
    for batch in loader:
        context = batch["context"].numpy()
        current_hr = batch["current_hr"].numpy()
        features.append(summary_features(context, current_hr))
        targets.append(batch["target"].numpy())
        masks.append(batch["target_valid_mask"].numpy().astype(bool))
        participants.extend(str(value) for value in batch["participant_id"])
    return (
        np.concatenate(features),
        np.concatenate(targets),
        np.concatenate(masks),
        participants,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--clip-min-bpm", type=float)
    parser.add_argument("--clip-max-bpm", type=float)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/processed/goldencheetah_transition_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/goldencheetah_pretest_v1"),
    )
    args = parser.parse_args()
    set_reproducible_seed(args.seed)

    fold_cache = args.cache_root / f"outer_fold_{args.outer_fold}"
    policy_path = fold_cache / "cache_policy.json"
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    training_dataset = TransitionWindowDataset(
        fold_cache,
        role="train",
        origin_policy="training_stride",
        schedule_aware=False,
        history_regime="none",
        history_session_budget=0,
    )
    validation_dataset = TransitionWindowDataset(
        fold_cache,
        role="validation",
        origin_policy="tagged_events",
        schedule_aware=False,
        history_regime="none",
        history_session_budget=0,
    )
    training_loader = DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    train_x, train_y, train_mask, _ = collect(training_loader)
    validation_x, validation_y, validation_mask, participants = collect(
        validation_loader
    )

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    validation_x = scaler.transform(validation_x)
    predictions = np.zeros_like(validation_y, dtype=float)
    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    fit_start = time.perf_counter()
    for horizon in range(train_y.shape[1]):
        valid = train_mask[:, horizon] & np.isfinite(train_y[:, horizon])
        if valid.sum() < train_x.shape[1] + 1:
            raise RuntimeError(f"Insufficient valid labels at horizon {horizon + 1}")
        model = Ridge(alpha=args.alpha, fit_intercept=True)
        model.fit(train_x[valid], train_y[valid, horizon])
        predictions[:, horizon] = model.predict(validation_x)
        coefficients.append(np.asarray(model.coef_, dtype=float))
        intercepts.append(float(model.intercept_))
    fit_seconds = time.perf_counter() - fit_start

    raw_predictions = predictions.copy()
    if (args.clip_min_bpm is None) != (args.clip_max_bpm is None):
        raise ValueError("Both clipping bounds must be supplied together")
    if args.clip_min_bpm is not None:
        if args.clip_min_bpm >= args.clip_max_bpm:
            raise ValueError("The lower clipping bound must be below the upper bound")
        predictions = np.clip(predictions, args.clip_min_bpm, args.clip_max_bpm)
    absolute = np.abs(predictions - validation_y)
    raw_absolute = np.abs(raw_predictions - validation_y)
    participant_errors: dict[str, list[float]] = {}
    raw_participant_errors: dict[str, list[float]] = {}
    inference_start = time.perf_counter()
    for index, participant in enumerate(participants):
        valid = validation_mask[index] & np.isfinite(validation_y[index])
        error = float(absolute[index, valid].mean())
        raw_error = float(raw_absolute[index, valid].mean())
        participant_errors.setdefault(participant, []).append(error)
        raw_participant_errors.setdefault(participant, []).append(raw_error)
    inference_seconds = time.perf_counter() - inference_start
    participant_mae = float(
        np.mean([np.mean(values) for values in participant_errors.values()])
    )
    raw_participant_mae = float(
        np.mean([np.mean(values) for values in raw_participant_errors.values()])
    )

    clipping_mode = (
        f"clip_{args.clip_min_bpm:g}_{args.clip_max_bpm:g}"
        if args.clip_min_bpm is not None
        else "unclipped"
    )
    output_dir = (
        args.output_root
        / f"outer_fold_{args.outer_fold}"
        / "ridge_summary"
        / f"alpha_{args.alpha:g}"
        / clipping_mode
        / f"seed_{args.seed}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Ridge baseline: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "ridge_parameters.npz",
        coefficient=np.stack(coefficients),
        intercept=np.asarray(intercepts),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    report = {
        "artifact_status": "pretest_validation_selection_only",
        "model": "ridge_context_summary",
        "alpha": args.alpha,
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "selection_role": "validation",
        "forbidden_roles_accessed": [],
        "calibration_or_test_metrics_present": False,
        "initialization_seed_set_before_model_build": True,
        "training_origins": len(training_dataset),
        "validation_transition_origins": len(validation_dataset),
        "validation_participants": len(participant_errors),
        "summary_feature_count": train_x.shape[1],
        "forecast_horizon_seconds": train_y.shape[1],
        "trainable_parameters": int(
            train_y.shape[1] * (train_x.shape[1] + 1)
        ),
        "best_validation_participant_transition_mae_bpm": participant_mae,
        "raw_unclipped_validation_participant_transition_mae_bpm": raw_participant_mae,
        "prediction_clip_bpm": (
            [args.clip_min_bpm, args.clip_max_bpm]
            if args.clip_min_bpm is not None
            else None
        ),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "cache_policy_sha256": sha256_file(policy_path),
    }
    report_path = output_dir / "selection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = {
        "ridge_parameters.npz": sha256_file(output_dir / "ridge_parameters.npz"),
        report_path.name: sha256_file(report_path),
    }
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
