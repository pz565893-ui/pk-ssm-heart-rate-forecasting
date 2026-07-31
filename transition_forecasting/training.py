"""Validation-only model fitting utilities for the pre-test development phase."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .baselines import (
    DampedTrendBaseline,
    DirectRecurrentForecaster,
    FirstOrderKineticForecaster,
    PersistenceBaseline,
    TCNForecaster,
    TransformerForecaster,
    UnconstrainedResidualSSM,
)
from .dataset import move_batch_to_device
from .pk_ssm import PKSSM, PKSSMConfig


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    maximum_epochs: int = 150
    early_stopping_patience: int = 15
    gradient_norm_clip: float = 1.0


@dataclass
class FitResult:
    best_epoch: int
    best_validation_participant_mae_bpm: float
    epochs_completed: int
    training_history: list[dict[str, float]]
    state_dict: dict[str, Tensor]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReliabilityGatedPKSSM(nn.Module):
    """Near-identity wrapper that lets evidence activate prior-session history."""

    def __init__(self, config: PKSSMConfig) -> None:
        super().__init__()
        self.core = PKSSM(config)
        self.history_reliability_logit = nn.Parameter(torch.tensor(-6.0))

    def forward(
        self,
        context: Tensor,
        current_hr: Tensor,
        context_valid_mask: Tensor,
        history: Tensor,
        history_valid_mask: Tensor,
        history_minutes: Tensor,
        activity_index: Tensor,
        future_event_index: Tensor,
    ) -> Any:
        reliability_gate = torch.sigmoid(self.history_reliability_logit)
        return self.core(
            context=context,
            current_hr=current_hr,
            context_valid_mask=context_valid_mask,
            history=history,
            history_valid_mask=history_valid_mask,
            history_minutes=history_minutes * reliability_gate,
            activity_index=activity_index,
            future_event_index=future_event_index,
        )

    def history_gate_value(self) -> float:
        return float(
            torch.sigmoid(self.history_reliability_logit).detach().cpu()
        )


def build_model(
    model_name: str,
    input_dim: int,
    candidate: dict[str, Any],
    hr_feature_scale_bpm: float | None = None,
) -> nn.Module:
    hidden_dim = int(candidate.get("hidden_dim", 64))
    horizon = int(candidate.get("horizon_seconds", 120))
    dropout = float(candidate.get("dropout", 0.10))
    dilations = tuple(candidate.get("convolution_dilations", [1, 2, 4, 8]))
    config = PKSSMConfig(
        input_dim=input_dim,
        horizon_seconds=horizon,
        hidden_dim=hidden_dim,
        convolution_dilations=dilations,
        dropout=dropout,
        residual_bound_bpm=float(candidate.get("residual_bound_bpm", 6.0)),
    )
    normalized = model_name.lower()
    if normalized == "pk_ssm":
        if bool(candidate.get("zero_gated_history", False)):
            return ReliabilityGatedPKSSM(config)
        return PKSSM(config)
    if normalized == "persistence":
        return PersistenceBaseline(horizon)
    if normalized == "damped_trend":
        if hr_feature_scale_bpm is None or hr_feature_scale_bpm <= 0.0:
            raise ValueError(
                "damped_trend requires the training-set HR standard deviation in bpm"
            )
        return DampedTrendBaseline(
            horizon_seconds=horizon,
            hr_feature_scale_bpm=hr_feature_scale_bpm,
        )
    if normalized == "gru":
        return DirectRecurrentForecaster(
            input_dim, horizon, hidden_dim, layers=2, dropout=dropout, cell="gru"
        )
    if normalized == "lstm":
        return DirectRecurrentForecaster(
            input_dim, horizon, hidden_dim, layers=2, dropout=dropout, cell="lstm"
        )
    if normalized == "tcn":
        return TCNForecaster(config)
    if normalized == "transformer":
        return TransformerForecaster(
            input_dim,
            horizon_seconds=horizon,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if normalized == "residual_ssm":
        return UnconstrainedResidualSSM(config)
    if normalized == "first_order_kinetics":
        return FirstOrderKineticForecaster(config)
    raise ValueError(f"Unknown model: {model_name}")


def forward_model(
    model_name: str,
    model: nn.Module,
    batch: dict[str, Any],
) -> Any:
    normalized = model_name.lower()
    if normalized == "persistence":
        return model(batch["current_hr"])
    if normalized in {"damped_trend", "gru", "lstm"}:
        return model(batch["context"], batch["current_hr"])
    if normalized in {"tcn", "transformer", "residual_ssm"}:
        return model(
            batch["context"],
            batch["current_hr"],
            batch["context_valid_mask"],
        )
    if normalized == "first_order_kinetics":
        return model(
            context=batch["context"],
            current_hr=batch["current_hr"],
            context_valid_mask=batch["context_valid_mask"],
        )
    if normalized == "pk_ssm":
        return model(
            context=batch["context"],
            current_hr=batch["current_hr"],
            context_valid_mask=batch["context_valid_mask"],
            history=batch["history"],
            history_valid_mask=batch["history_valid_mask"],
            history_minutes=batch["history_minutes"],
            activity_index=batch["activity_index"],
            future_event_index=batch["future_event_index"],
        )
    raise ValueError(f"Unknown model: {model_name}")


def masked_student_t_nll(
    output: Any,
    target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    distribution = torch.distributions.StudentT(
        df=output.degrees_of_freedom,
        loc=output.mean,
        scale=output.scale,
    )
    loss = -distribution.log_prob(target)
    weights = valid_mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def participant_aggregated_mae(
    model_name: str,
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
) -> float:
    model.eval()
    participant_errors: dict[str, list[float]] = {}
    with torch.no_grad():
        for raw_batch in loader:
            participant_ids = list(raw_batch["participant_id"])
            batch = move_batch_to_device(raw_batch, device)
            output = forward_model(model_name, model, batch)
            absolute = torch.abs(output.mean - batch["target"])
            weights = batch["target_valid_mask"].to(absolute.dtype)
            origin_mae = (absolute * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            for participant, error in zip(participant_ids, origin_mae.detach().cpu().tolist()):
                participant_errors.setdefault(str(participant), []).append(float(error))
    if not participant_errors:
        raise ValueError("Validation loader produced no participant errors")
    participant_means = [np.mean(values) for values in participant_errors.values()]
    return float(np.mean(participant_means))


def fit_model(
    model_name: str,
    model: nn.Module,
    training_loader: Iterable[dict[str, Any]],
    validation_event_loader: Iterable[dict[str, Any]],
    optimization: OptimizationConfig,
    device: torch.device,
    seed: int,
    resume_state: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> FitResult:
    set_reproducible_seed(seed)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        validation_mae = participant_aggregated_mae(
            model_name, model, validation_event_loader, device
        )
        return FitResult(
            best_epoch=0,
            best_validation_participant_mae_bpm=validation_mae,
            epochs_completed=0,
            training_history=[],
            state_dict=copy.deepcopy(model.state_dict()),
        )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )
    best_mae = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    patience = 0
    history: list[dict[str, float]] = []
    start_epoch = 1
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        best_mae = float(resume_state["best_mae"])
        best_epoch = int(resume_state["best_epoch"])
        best_state = resume_state["best_state"]
        patience = int(resume_state["patience"])
        history = list(resume_state["history"])
        start_epoch = int(resume_state["epoch"]) + 1
        random.setstate(resume_state["python_random_state"])
        np.random.set_state(resume_state["numpy_random_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        cuda_state = resume_state.get("cuda_rng_state_all")
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    for epoch in range(start_epoch, optimization.maximum_epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for raw_batch in training_loader:
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = forward_model(model_name, model, batch)
            loss = masked_student_t_nll(
                output,
                batch["target"],
                batch["target_valid_mask"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss for {model_name} at epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, optimization.gradient_norm_clip)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        if batches == 0:
            raise ValueError("Training loader produced no batches")
        validation_mae = participant_aggregated_mae(
            model_name,
            model,
            validation_event_loader,
            device,
        )
        history.append(
            {
                "epoch": float(epoch),
                "training_nll": running_loss / batches,
                "validation_participant_transition_mae_bpm": validation_mae,
            }
        )
        should_stop = False
        if validation_mae < best_mae - 1.0e-6:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            should_stop = patience >= optimization.early_stopping_patience
        if progress_callback is not None:
            progress_callback(
                {
                    "contract_version": 1,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_mae": best_mae,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "patience": patience,
                    "history": history,
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    ),
                }
            )
        if should_stop:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was selected")
    model.load_state_dict(best_state)
    return FitResult(
        best_epoch=best_epoch,
        best_validation_participant_mae_bpm=best_mae,
        epochs_completed=len(history),
        training_history=history,
        state_dict=best_state,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
