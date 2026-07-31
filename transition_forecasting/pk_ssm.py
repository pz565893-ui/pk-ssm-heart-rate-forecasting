"""Personalized two-phase kinetic state-space model.

This module contains no dataset-specific split logic. Inputs are causal context
signals and, optionally, user history that ends before the evaluation embargo.
Future event labels may be supplied only in the schedule-aware deployment mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PKSSMConfig:
    input_dim: int
    horizon_seconds: int = 120
    hidden_dim: int = 64
    convolution_kernel: int = 5
    convolution_dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.10
    num_event_types: int = 8
    event_embedding_dim: int = 8
    num_activity_types: int = 3
    activity_embedding_dim: int = 4
    residual_bound_bpm: float = 6.0
    residual_ramp_seconds: float = 5.0
    shrinkage_half_life_minutes: float = 10.0
    minimum_hr_bpm: float = 30.0
    maximum_hr_bpm: float = 220.0
    minimum_rest_hr_bpm: float = 35.0
    maximum_rest_hr_bpm: float = 120.0
    minimum_feasible_max_hr_bpm: float = 120.0
    minimum_tau_seconds: float = 1.0
    maximum_fast_tau_seconds: float = 120.0
    maximum_slow_tau_seconds: float = 600.0
    minimum_scale_bpm: float = 0.25
    minimum_student_df: float = 2.1

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")
        if not self.convolution_dilations:
            raise ValueError("At least one causal convolution dilation is required")
        if self.minimum_tau_seconds >= self.maximum_fast_tau_seconds:
            raise ValueError("The fast time-constant bounds are invalid")
        if self.maximum_fast_tau_seconds >= self.maximum_slow_tau_seconds:
            raise ValueError("The slow time-constant upper bound must be larger")


@dataclass
class PKSSMOutput:
    mean: Tensor
    scale: Tensor
    degrees_of_freedom: Tensor
    workload: Tensor
    fast_state: Tensor
    slow_state: Tensor
    residual: Tensor
    parameters: dict[str, Tensor]


class CausalConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.normalization = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = F.pad(values, (self.left_padding, 0))
        values = self.convolution(values)
        values = self.normalization(values)
        values = F.silu(values)
        values = self.dropout(values)
        return values + residual


class CausalTemporalEncoder(nn.Module):
    def __init__(self, config: PKSSMConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            CausalConvBlock(
                config.hidden_dim,
                config.convolution_kernel,
                dilation,
                config.dropout,
            )
            for dilation in config.convolution_dilations
        )
        self.output_projection = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, values: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if values.ndim != 3:
            raise ValueError("Temporal input must have shape [batch, time, features]")
        if valid_mask is not None and valid_mask.shape != values.shape[:2]:
            raise ValueError("valid_mask must have shape [batch, time]")
        sequence = self.input_projection(values).transpose(1, 2)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = sequence.transpose(1, 2)
        last = sequence[:, -1]
        if valid_mask is None:
            pooled = sequence.mean(dim=1)
        else:
            weights = valid_mask.to(sequence.dtype).unsqueeze(-1)
            pooled = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output_projection(torch.cat([last, pooled], dim=-1))


class PKSSM(nn.Module):
    """Two-phase constrained HR dynamics with hierarchical personalization."""

    parameter_names = (
        "rest_fraction",
        "maximum_fraction",
        "fast_rise",
        "slow_rise_gap",
        "fast_recovery",
        "slow_recovery_gap",
        "total_gain",
        "fast_gain_fraction",
        "initial_fast_fraction",
    )

    def __init__(self, config: PKSSMConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = CausalTemporalEncoder(config)
        self.history_encoder = CausalTemporalEncoder(config)
        parameter_count = len(self.parameter_names)
        self.population_raw_parameters = nn.Parameter(torch.zeros(parameter_count))
        self.personalization_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, parameter_count),
        )
        self.event_embedding = nn.Embedding(
            config.num_event_types,
            config.event_embedding_dim,
            padding_idx=0,
        )
        self.activity_embedding = nn.Embedding(
            config.num_activity_types,
            config.activity_embedding_dim,
            padding_idx=0,
        )
        future_input_dim = (
            config.hidden_dim
            + config.event_embedding_dim
            + config.activity_embedding_dim
            + 3
        )
        self.current_workload_head = nn.Linear(config.hidden_dim, 1)
        self.workload_head = nn.Sequential(
            nn.Linear(future_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(future_input_dim + 1, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.distribution_head = nn.Sequential(
            nn.Linear(future_input_dim + 1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2),
        )
        self.activity_parameter_head = nn.Linear(
            config.activity_embedding_dim,
            parameter_count,
            bias=False,
        )
        self._initialize_padding_embeddings()

    def _initialize_padding_embeddings(self) -> None:
        with torch.no_grad():
            self.event_embedding.weight[0].zero_()
            self.activity_embedding.weight[0].zero_()

    def _future_design(
        self,
        context_representation: Tensor,
        activity_index: Tensor | None,
        future_event_index: Tensor | None,
    ) -> Tensor:
        batch = context_representation.shape[0]
        horizon = self.config.horizon_seconds
        device = context_representation.device
        dtype = context_representation.dtype
        steps = torch.arange(1, horizon + 1, device=device, dtype=dtype)
        phase = steps / float(horizon)
        time_features = torch.stack(
            [phase, torch.sin(torch.pi * phase), torch.cos(torch.pi * phase)],
            dim=-1,
        ).unsqueeze(0).expand(batch, -1, -1)
        context = context_representation.unsqueeze(1).expand(-1, horizon, -1)
        if activity_index is None:
            activity_index = torch.zeros(batch, dtype=torch.long, device=device)
        if activity_index.shape != (batch,):
            raise ValueError("activity_index must have shape [batch]")
        activity = self.activity_embedding(activity_index)
        activity = activity.unsqueeze(1).expand(-1, horizon, -1)
        if future_event_index is None:
            future_event_index = torch.zeros(
                (batch, horizon), dtype=torch.long, device=device
            )
        if future_event_index.shape != (batch, horizon):
            raise ValueError("future_event_index must have shape [batch, horizon]")
        event = self.event_embedding(future_event_index)
        return torch.cat([context, activity, event, time_features], dim=-1)

    def _raw_parameters(
        self,
        batch: int,
        history: Tensor | None,
        history_valid_mask: Tensor | None,
        history_minutes: Tensor | None,
        activity_index: Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        raw = self.population_raw_parameters.to(dtype=dtype).unsqueeze(0).expand(batch, -1)
        if history is not None:
            if history.shape[0] != batch or history.shape[2] != self.config.input_dim:
                raise ValueError("history must have shape [batch, time, input_dim]")
            history_representation = self.history_encoder(history, history_valid_mask)
            user_offset = self.personalization_head(history_representation)
            if history_minutes is None:
                valid = (
                    history_valid_mask.to(dtype).sum(dim=1)
                    if history_valid_mask is not None
                    else torch.full(
                        (batch,), history.shape[1], dtype=dtype, device=device
                    )
                )
                history_minutes = valid / 60.0
            if history_minutes.shape != (batch,):
                raise ValueError("history_minutes must have shape [batch]")
            reliability = history_minutes.to(dtype).clamp_min(0.0)
            reliability = reliability / (
                reliability + self.config.shrinkage_half_life_minutes
            )
            raw = raw + reliability.unsqueeze(-1) * user_offset
        if activity_index is not None:
            activity_offset = self.activity_parameter_head(
                self.activity_embedding(activity_index)
            )
            raw = raw + activity_offset
        return raw

    def _transform_parameters(self, raw: Tensor, current_hr: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        rest_upper = current_hr.clamp(
            min=cfg.minimum_rest_hr_bpm + 1.0,
            max=cfg.maximum_rest_hr_bpm,
        )
        rest_hr = cfg.minimum_rest_hr_bpm + (
            rest_upper - cfg.minimum_rest_hr_bpm
        ) * torch.sigmoid(raw[:, 0])
        feasible_max = cfg.minimum_feasible_max_hr_bpm + (
            cfg.maximum_hr_bpm - cfg.minimum_feasible_max_hr_bpm
        ) * torch.sigmoid(raw[:, 1])
        feasible_max = torch.maximum(feasible_max, current_hr + 1.0).clamp_max(
            cfg.maximum_hr_bpm
        )
        reserve = (feasible_max - rest_hr).clamp_min(1.0)

        tau_fast_rise = cfg.minimum_tau_seconds + (
            cfg.maximum_fast_tau_seconds - cfg.minimum_tau_seconds
        ) * torch.sigmoid(raw[:, 2])
        tau_slow_rise = tau_fast_rise + (
            cfg.maximum_slow_tau_seconds - tau_fast_rise
        ) * torch.sigmoid(raw[:, 3])
        tau_fast_recovery = cfg.minimum_tau_seconds + (
            cfg.maximum_fast_tau_seconds - cfg.minimum_tau_seconds
        ) * torch.sigmoid(raw[:, 4])
        tau_slow_recovery = tau_fast_recovery + (
            cfg.maximum_slow_tau_seconds - tau_fast_recovery
        ) * torch.sigmoid(raw[:, 5])

        total_gain = torch.sigmoid(raw[:, 6])
        fast_gain_fraction = torch.sigmoid(raw[:, 7])
        gain_fast = total_gain * fast_gain_fraction
        gain_slow = total_gain * (1.0 - fast_gain_fraction)
        initial_fast_fraction = torch.sigmoid(raw[:, 8])
        return {
            "rest_hr": rest_hr,
            "feasible_max_hr": feasible_max,
            "hr_reserve": reserve,
            "tau_fast_rise": tau_fast_rise,
            "tau_slow_rise": tau_slow_rise,
            "tau_fast_recovery": tau_fast_recovery,
            "tau_slow_recovery": tau_slow_recovery,
            "gain_fast": gain_fast,
            "gain_slow": gain_slow,
            "initial_fast_fraction": initial_fast_fraction,
        }

    def forward(
        self,
        context: Tensor,
        current_hr: Tensor,
        context_valid_mask: Tensor | None = None,
        history: Tensor | None = None,
        history_valid_mask: Tensor | None = None,
        history_minutes: Tensor | None = None,
        activity_index: Tensor | None = None,
        future_event_index: Tensor | None = None,
    ) -> PKSSMOutput:
        if context.ndim != 3 or context.shape[2] != self.config.input_dim:
            raise ValueError("context must have shape [batch, time, input_dim]")
        batch = context.shape[0]
        if current_hr.shape != (batch,):
            raise ValueError("current_hr must have shape [batch]")
        context_representation = self.context_encoder(context, context_valid_mask)
        raw_parameters = self._raw_parameters(
            batch,
            history,
            history_valid_mask,
            history_minutes,
            activity_index,
            context.device,
            context.dtype,
        )
        parameters = self._transform_parameters(raw_parameters, current_hr)
        future_design = self._future_design(
            context_representation, activity_index, future_event_index
        )
        workload_target = torch.sigmoid(self.workload_head(future_design).squeeze(-1))
        current_workload = torch.sigmoid(
            self.current_workload_head(context_representation)
        )
        steps = torch.arange(
            1,
            self.config.horizon_seconds + 1,
            device=context.device,
            dtype=context.dtype,
        )
        workload_ramp = 1.0 - torch.exp(-steps / self.config.residual_ramp_seconds)
        workload = (
            (1.0 - workload_ramp.unsqueeze(0)) * current_workload
            + workload_ramp.unsqueeze(0) * workload_target
        )

        elevation = (current_hr - parameters["rest_hr"]).clamp_min(0.0)
        fast_state = elevation * parameters["initial_fast_fraction"]
        slow_state = elevation - fast_state
        fast_states: list[Tensor] = []
        slow_states: list[Tensor] = []
        means: list[Tensor] = []
        residuals: list[Tensor] = []

        distribution_input = torch.cat(
            [future_design, workload.unsqueeze(-1)], dim=-1
        )
        residual_raw = self.residual_head(distribution_input).squeeze(-1)
        residual_ramp = 1.0 - torch.exp(
            -steps / self.config.residual_ramp_seconds
        )
        residual_trajectory = (
            self.config.residual_bound_bpm
            * residual_ramp.unsqueeze(0)
            * torch.tanh(residual_raw)
        )

        for index in range(self.config.horizon_seconds):
            equilibrium_fast = (
                parameters["hr_reserve"]
                * parameters["gain_fast"]
                * workload[:, index]
            )
            equilibrium_slow = (
                parameters["hr_reserve"]
                * parameters["gain_slow"]
                * workload[:, index]
            )
            tau_fast = torch.where(
                equilibrium_fast >= fast_state,
                parameters["tau_fast_rise"],
                parameters["tau_fast_recovery"],
            )
            tau_slow = torch.where(
                equilibrium_slow >= slow_state,
                parameters["tau_slow_rise"],
                parameters["tau_slow_recovery"],
            )
            alpha_fast = torch.exp(-1.0 / tau_fast.clamp_min(1.0e-6))
            alpha_slow = torch.exp(-1.0 / tau_slow.clamp_min(1.0e-6))
            fast_state = alpha_fast * fast_state + (1.0 - alpha_fast) * equilibrium_fast
            slow_state = alpha_slow * slow_state + (1.0 - alpha_slow) * equilibrium_slow
            physical_mean = (
                parameters["rest_hr"]
                + fast_state
                + slow_state
                + residual_trajectory[:, index]
            )
            bounded_mean = torch.minimum(
                torch.maximum(
                    physical_mean,
                    torch.full_like(physical_mean, self.config.minimum_hr_bpm),
                ),
                parameters["feasible_max_hr"],
            )
            fast_states.append(fast_state)
            slow_states.append(slow_state)
            residuals.append(residual_trajectory[:, index])
            means.append(bounded_mean)

        distribution_raw = self.distribution_head(distribution_input)
        scale = F.softplus(distribution_raw[..., 0]) + self.config.minimum_scale_bpm
        degrees_of_freedom = (
            F.softplus(distribution_raw[..., 1]) + self.config.minimum_student_df
        )
        return PKSSMOutput(
            mean=torch.stack(means, dim=1),
            scale=scale,
            degrees_of_freedom=degrees_of_freedom,
            workload=workload,
            fast_state=torch.stack(fast_states, dim=1),
            slow_state=torch.stack(slow_states, dim=1),
            residual=torch.stack(residuals, dim=1),
            parameters=parameters,
        )


def student_t_nll(
    output: PKSSMOutput,
    target: Tensor,
    valid_mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Negative Student-t log likelihood over a future HR trajectory."""
    if target.shape != output.mean.shape:
        raise ValueError("target and model output must have identical [batch, horizon] shape")
    distribution = torch.distributions.StudentT(
        df=output.degrees_of_freedom,
        loc=output.mean,
        scale=output.scale,
    )
    loss = -distribution.log_prob(target)
    if valid_mask is not None:
        if valid_mask.shape != target.shape:
            raise ValueError("valid_mask must match target shape")
        loss = loss * valid_mask.to(loss.dtype)
        denominator = valid_mask.to(loss.dtype).sum().clamp_min(1.0)
    else:
        denominator = torch.tensor(loss.numel(), dtype=loss.dtype, device=loss.device)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / denominator
    raise ValueError(f"Unsupported reduction: {reduction}")


def parameter_snapshot(output: PKSSMOutput) -> dict[str, Any]:
    """Detach interpretable parameters for audit logging, not model fitting."""
    return {
        name: values.detach().cpu().numpy()
        for name, values in output.parameters.items()
    }
