#!/usr/bin/env python3
"""Fit one candidate using training users and validation events only."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transition_forecasting.dataset import TransitionWindowDataset
from transition_forecasting.training import (
    OptimizationConfig,
    build_model,
    fit_model,
    parameter_count,
    set_reproducible_seed,
)


ALLOWED_MODELS = (
    "persistence",
    "damped_trend",
    "gru",
    "lstm",
    "tcn",
    "transformer",
    "residual_ssm",
    "first_order_kinetics",
    "pk_ssm",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(path: Path, candidate_id: str) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in payload["model_candidates"]
    }
    if candidate_id not in candidates:
        raise ValueError(f"Unknown candidate_id {candidate_id}; available: {sorted(candidates)}")
    return payload, candidates[candidate_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True, choices=range(5))
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    parser.add_argument("--candidate-id", default="pkssm_64x4_r6")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--schedule-aware", action="store_true")
    parser.add_argument("--mask-activity-identity", action="store_true")
    parser.add_argument(
        "--history-regime",
        choices=("none", "train_prior", "role_prior", "all_prior"),
        default="train_prior",
    )
    parser.add_argument(
        "--history-session-budget",
        type=int,
        default=-1,
        help="-1 uses every legal prior session; 0 disables history; positive values use the latest N",
    )
    parser.add_argument("--history-seconds-per-session", type=int, default=300)
    parser.add_argument("--maximum-history-sessions", type=int, default=8)
    parser.add_argument(
        "--zero-gated-history",
        action="store_true",
        help="Use the single-parameter near-identity history reliability gate.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/processed/wearable_exercise_transition_v1"),
    )
    parser.add_argument(
        "--candidate-grid",
        type=Path,
        default=Path("configs/pk_ssm_candidate_grid_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/pretest_model_selection_v1"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    fold_cache = args.cache_root / f"outer_fold_{args.outer_fold}"
    policy_path = fold_cache / "cache_policy.json"
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not any(
        version in policy["split_manifest"]
        for version in (
            "wearable_exercise_v3",
            "wearable_exercise_v4",
            "goldencheetah_light_v1",
        )
    ):
        raise ValueError("Pretest selection refuses caches derived from provisional splits")
    grid, candidate = load_candidate(args.candidate_grid, args.candidate_id)
    candidate = {
        **candidate,
        "horizon_seconds": int(grid["fixed_task"]["horizon_seconds"]),
    }
    if args.zero_gated_history:
        candidate["zero_gated_history"] = True
    allowed_seeds = set(grid["optimization_candidates"]["training_seeds"])
    if args.seed not in allowed_seeds:
        raise ValueError(f"Seed {args.seed} is outside the frozen seed list")
    set_reproducible_seed(args.seed)

    training_dataset = TransitionWindowDataset(
        fold_cache,
        role="train",
        origin_policy="training_stride",
        schedule_aware=args.schedule_aware,
        mask_activity_identity=args.mask_activity_identity,
        history_regime=args.history_regime,
        history_session_budget=args.history_session_budget,
        history_seconds_per_session=args.history_seconds_per_session,
        maximum_history_sessions=args.maximum_history_sessions,
    )
    validation_dataset = TransitionWindowDataset(
        fold_cache,
        role="validation",
        origin_policy="tagged_events",
        schedule_aware=args.schedule_aware,
        mask_activity_identity=args.mask_activity_identity,
        history_regime=args.history_regime,
        history_session_budget=args.history_session_budget,
        history_seconds_per_session=args.history_seconds_per_session,
        maximum_history_sessions=args.maximum_history_sessions,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    hr_feature_scale_bpm = float(
        training_dataset.normalization["hr_bpm"]["std"]
    )
    model = build_model(
        args.model,
        training_dataset.input_dim,
        candidate,
        hr_feature_scale_bpm=hr_feature_scale_bpm,
    )
    optimization_grid = grid["optimization_candidates"]
    optimization = OptimizationConfig(
        learning_rate=float(optimization_grid["learning_rate"][0]),
        weight_decay=float(optimization_grid["weight_decay"][0]),
        maximum_epochs=int(optimization_grid["maximum_epochs"]),
        early_stopping_patience=int(optimization_grid["early_stopping_patience"]),
        gradient_norm_clip=float(optimization_grid["gradient_norm_clip"]),
    )
    mode = "schedule_aware" if args.schedule_aware else "signal_only"
    activity = "masked_activity" if args.mask_activity_identity else "known_activity"
    history_mode = (
        f"history_{args.history_regime}_budget_{args.history_session_budget}"
    )
    if args.zero_gated_history:
        history_mode += "_zero_gate"
    output_dir = (
        args.output_root
        / f"outer_fold_{args.outer_fold}"
        / args.model
        / args.candidate_id
        / mode
        / activity
        / history_mode
        / f"seed_{args.seed}"
    )
    final_artifacts = (
        "validation_selected_checkpoint.pt",
        "selection_report.json",
        "SHA256SUMS.txt",
    )
    if output_dir.exists() and all(
        (output_dir / name).is_file() for name in final_artifacts
    ):
        raise FileExistsError(f"Refusing to overwrite completed selection run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_state_path = output_dir / "training_state.pt"
    training_state_temp_path = output_dir / "training_state.tmp"
    allowed_partial_files = {training_state_path.name, training_state_temp_path.name}
    unexpected_partial_files = [
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in allowed_partial_files
    ]
    if unexpected_partial_files:
        raise FileExistsError(
            f"Refusing ambiguous partial selection run in {output_dir}: "
            f"{sorted(unexpected_partial_files)}"
        )
    resume_state = None
    if training_state_path.is_file():
        resume_state = torch.load(
            training_state_path,
            map_location=torch.device(args.device),
            weights_only=False,
        )

    def persist_training_state(state: dict[str, Any]) -> None:
        torch.save(state, training_state_temp_path)
        training_state_temp_path.replace(training_state_path)

    result = fit_model(
        args.model,
        model,
        training_loader,
        validation_loader,
        optimization,
        torch.device(args.device),
        args.seed,
        resume_state=resume_state,
        progress_callback=persist_training_state,
    )

    checkpoint_path = output_dir / "validation_selected_checkpoint.pt"
    torch.save(result.state_dict, checkpoint_path)
    report = {
        "artifact_status": "pretest_validation_selection_only",
        "outer_fold": args.outer_fold,
        "model": args.model,
        "candidate_id": args.candidate_id,
        "seed": args.seed,
        "initialization_seed_set_before_model_build": True,
        "mode": mode,
        "activity_identity": activity,
        "history": {
            "training": training_dataset.history_summary,
            "validation": validation_dataset.history_summary,
        },
        "history_adapter": (
            "single_parameter_near_identity_reliability_gate"
            if args.zero_gated_history
            else "original"
        ),
        "learned_history_gate": (
            model.history_gate_value() if args.zero_gated_history else None
        ),
        "training_role": "train",
        "selection_role": "validation",
        "forbidden_roles_accessed": [],
        "calibration_or_test_metrics_present": False,
        "hr_feature_scale_bpm": hr_feature_scale_bpm,
        "input_feature_names": training_dataset.input_feature_names,
        "trainable_parameters": parameter_count(model),
        "optimization": asdict(optimization),
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "resumed_from_epoch": int(resume_state["epoch"]) if resume_state else 0,
        "epoch_state_checkpointing": True,
        "best_validation_participant_transition_mae_bpm": result.best_validation_participant_mae_bpm,
        "training_origins": len(training_dataset),
        "validation_transition_origins": len(validation_dataset),
        "cache_policy_sha256": sha256_file(policy_path),
        "candidate_grid_sha256": sha256_file(args.candidate_grid),
        "checkpoint_sha256": "recorded_after serialization in SHA256SUMS.txt",
        "training_history": result.training_history,
    }
    report_path = output_dir / "selection_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = {
        checkpoint_path.name: sha256_file(checkpoint_path),
        report_path.name: sha256_file(report_path),
    }
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    training_state_path.unlink(missing_ok=True)
    training_state_temp_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
