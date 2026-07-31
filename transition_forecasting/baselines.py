"""Neural and analytic baselines with a common trajectory output contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .pk_ssm import CausalTemporalEncoder, PKSSMConfig


@dataclass
class DistributionalForecast:
    mean: Tensor
    scale: Tensor
    degrees_of_freedom: Tensor


def _bounded_distribution(
    raw: Tensor,
    current_hr: Tensor,
    minimum_hr_bpm: float = 30.0,
    maximum_hr_bpm: float = 220.0,
    minimum_scale_bpm: float = 0.25,
    minimum_student_df: float = 2.1,
) -> DistributionalForecast:
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("raw forecast must have shape [batch, horizon, 3]")
    mean = current_hr.unsqueeze(-1) + 120.0 * torch.tanh(raw[..., 0])
    mean = mean.clamp(minimum_hr_bpm, maximum_hr_bpm)
    scale = F.softplus(raw[..., 1]) + minimum_scale_bpm
    degrees_of_freedom = F.softplus(raw[..., 2]) + minimum_student_df
    return DistributionalForecast(mean, scale, degrees_of_freedom)


class PersistenceBaseline(nn.Module):
    """Last observed HR repeated over the complete horizon."""

    def __init__(self, horizon_seconds: int = 120) -> None:
        super().__init__()
        self.horizon_seconds = horizon_seconds
        self.raw_scale = nn.Parameter(torch.tensor(1.0))
        self.raw_df = nn.Parameter(torch.tensor(2.0))

    def forward(self, current_hr: Tensor) -> DistributionalForecast:
        mean = current_hr.unsqueeze(-1).expand(-1, self.horizon_seconds)
        scale = (F.softplus(self.raw_scale) + 0.25).expand_as(mean)
        degrees_of_freedom = (F.softplus(self.raw_df) + 2.1).expand_as(mean)
        return DistributionalForecast(mean, scale, degrees_of_freedom)


class DampedTrendBaseline(nn.Module):
    """Causal local linear trend whose derivative decays exponentially."""

    def __init__(
        self,
        horizon_seconds: int = 120,
        hr_feature_index: int = 0,
        trend_window_seconds: int = 30,
        damping_time_seconds: float = 60.0,
        hr_feature_scale_bpm: float = 1.0,
    ) -> None:
        super().__init__()
        if hr_feature_scale_bpm <= 0.0:
            raise ValueError("hr_feature_scale_bpm must be positive")
        self.horizon_seconds = horizon_seconds
        self.hr_feature_index = hr_feature_index
        self.trend_window_seconds = trend_window_seconds
        self.damping_time_seconds = damping_time_seconds
        self.register_buffer(
            "hr_feature_scale_bpm",
            torch.tensor(float(hr_feature_scale_bpm)),
        )
        self.raw_scale = nn.Parameter(torch.tensor(1.0))
        self.raw_df = nn.Parameter(torch.tensor(2.0))

    def forward(self, context: Tensor, current_hr: Tensor) -> DistributionalForecast:
        if context.ndim != 3:
            raise ValueError("context must have shape [batch, time, features]")
        window = min(self.trend_window_seconds, context.shape[1])
        hr = context[:, -window:, self.hr_feature_index]
        time = torch.arange(window, device=context.device, dtype=context.dtype)
        centered = time - time.mean()
        denominator = torch.sum(centered**2).clamp_min(1.0)
        standardized_slope = (
            torch.sum((hr - hr.mean(dim=1, keepdim=True)) * centered, dim=1)
            / denominator
        )
        slope_bpm_per_second = standardized_slope * self.hr_feature_scale_bpm.to(
            dtype=context.dtype
        )
        steps = torch.arange(
            1, self.horizon_seconds + 1, device=context.device, dtype=context.dtype
        )
        integrated_damping = self.damping_time_seconds * (
            1.0 - torch.exp(-steps / self.damping_time_seconds)
        )
        mean = (
            current_hr.unsqueeze(-1)
            + slope_bpm_per_second.unsqueeze(-1) * integrated_damping
        )
        mean = mean.clamp(30.0, 220.0)
        scale = (F.softplus(self.raw_scale) + 0.25).expand_as(mean)
        degrees_of_freedom = (F.softplus(self.raw_df) + 2.1).expand_as(mean)
        return DistributionalForecast(mean, scale, degrees_of_freedom)


class DirectRecurrentForecaster(nn.Module):
    """Direct GRU or LSTM trajectory forecaster."""

    def __init__(
        self,
        input_dim: int,
        horizon_seconds: int = 120,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.10,
        cell: str = "gru",
    ) -> None:
        super().__init__()
        normalized = cell.lower()
        if normalized not in {"gru", "lstm"}:
            raise ValueError("cell must be 'gru' or 'lstm'")
        recurrent = nn.GRU if normalized == "gru" else nn.LSTM
        self.horizon_seconds = horizon_seconds
        self.recurrent = recurrent(
            input_dim,
            hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_seconds * 3),
        )

    def forward(self, context: Tensor, current_hr: Tensor) -> DistributionalForecast:
        _, hidden = self.recurrent(context)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        representation = hidden[-1]
        raw = self.head(representation).reshape(-1, self.horizon_seconds, 3)
        return _bounded_distribution(raw, current_hr)


class TCNForecaster(nn.Module):
    def __init__(self, config: PKSSMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CausalTemporalEncoder(config)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.horizon_seconds * 3),
        )

    def forward(
        self,
        context: Tensor,
        current_hr: Tensor,
        context_valid_mask: Tensor | None = None,
    ) -> DistributionalForecast:
        representation = self.encoder(context, context_valid_mask)
        raw = self.head(representation).reshape(-1, self.config.horizon_seconds, 3)
        return _bounded_distribution(raw, current_hr)


class TransformerForecaster(nn.Module):
    """Transformer over the entirely historical context.

    Bidirectional attention inside the past context is permissible because no
    token occurs after the forecast origin.
    """

    def __init__(
        self,
        input_dim: int,
        context_seconds: int = 300,
        horizon_seconds: int = 120,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.context_seconds = context_seconds
        self.horizon_seconds = horizon_seconds
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, context_seconds, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_seconds * 3),
        )

    def forward(
        self,
        context: Tensor,
        current_hr: Tensor,
        context_valid_mask: Tensor | None = None,
    ) -> DistributionalForecast:
        if context.shape[1] > self.context_seconds:
            raise ValueError("context exceeds the configured positional length")
        values = self.input_projection(context) + self.position[:, : context.shape[1]]
        padding_mask = None if context_valid_mask is None else ~context_valid_mask.bool()
        encoded = self.encoder(values, src_key_padding_mask=padding_mask)
        last = encoded[:, -1]
        if context_valid_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            weights = context_valid_mask.to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        raw = self.head(torch.cat([last, pooled], dim=-1)).reshape(
            -1, self.horizon_seconds, 3
        )
        return _bounded_distribution(raw, current_hr)


class UnconstrainedResidualSSM(nn.Module):
    """Parameter-matched latent rollout without physiological kinetics."""

    def __init__(
        self,
        config: PKSSMConfig,
        maximum_step_change_bpm: float = 12.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.maximum_step_change_bpm = maximum_step_change_bpm
        self.encoder = CausalTemporalEncoder(config)
        self.state_cell = nn.GRUCell(4, config.hidden_dim)
        self.step_head = nn.Linear(config.hidden_dim, 1)
        self.distribution_head = nn.Linear(config.hidden_dim, 2)

    def forward(
        self,
        context: Tensor,
        current_hr: Tensor,
        context_valid_mask: Tensor | None = None,
    ) -> DistributionalForecast:
        state = self.encoder(context, context_valid_mask)
        previous = current_hr
        means: list[Tensor] = []
        scales: list[Tensor] = []
        dfs: list[Tensor] = []
        for step in range(1, self.config.horizon_seconds + 1):
            phase = torch.full_like(current_hr, step / self.config.horizon_seconds)
            state_input = torch.stack(
                [
                    previous / 220.0,
                    phase,
                    torch.sin(torch.pi * phase),
                    torch.cos(torch.pi * phase),
                ],
                dim=-1,
            )
            state = self.state_cell(state_input, state)
            change = self.maximum_step_change_bpm * torch.tanh(
                self.step_head(state).squeeze(-1)
            )
            previous = (previous + change).clamp(30.0, 220.0)
            distribution = self.distribution_head(state)
            means.append(previous)
            scales.append(F.softplus(distribution[:, 0]) + 0.25)
            dfs.append(F.softplus(distribution[:, 1]) + 2.1)
        return DistributionalForecast(
            torch.stack(means, dim=1),
            torch.stack(scales, dim=1),
            torch.stack(dfs, dim=1),
        )


class FirstOrderKineticForecaster(nn.Module):
    """Personalized one-state kinetic comparator with rise/recovery asymmetry."""

    def __init__(self, config: PKSSMConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = CausalTemporalEncoder(config)
        self.history_encoder = CausalTemporalEncoder(config)
        self.population = nn.Parameter(torch.zeros(5))
        self.personalization_head = nn.Linear(config.hidden_dim, 5)
        self.workload_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 3, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.residual_scale_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 3, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(
        self,
        context: Tensor,
        current_hr: Tensor,
        context_valid_mask: Tensor | None = None,
        history: Tensor | None = None,
        history_valid_mask: Tensor | None = None,
        history_minutes: Tensor | None = None,
    ) -> DistributionalForecast:
        representation = self.context_encoder(context, context_valid_mask)
        batch = context.shape[0]
        raw = self.population.unsqueeze(0).expand(batch, -1)
        if history is not None:
            history_representation = self.history_encoder(history, history_valid_mask)
            offset = self.personalization_head(history_representation)
            if history_minutes is None:
                history_minutes = torch.full(
                    (batch,), history.shape[1] / 60.0, device=context.device, dtype=context.dtype
                )
            reliability = history_minutes / (
                history_minutes + self.config.shrinkage_half_life_minutes
            )
            raw = raw + reliability.unsqueeze(-1) * offset
        rest_upper = current_hr.clamp(36.0, 120.0)
        rest = 35.0 + (rest_upper - 35.0) * torch.sigmoid(raw[:, 0])
        feasible_max = 120.0 + 100.0 * torch.sigmoid(raw[:, 1])
        feasible_max = torch.maximum(feasible_max, current_hr + 1.0).clamp_max(220.0)
        reserve = (feasible_max - rest).clamp_min(1.0)
        tau_rise = 1.0 + 599.0 * torch.sigmoid(raw[:, 2])
        tau_recovery = 1.0 + 599.0 * torch.sigmoid(raw[:, 3])
        gain = torch.sigmoid(raw[:, 4])

        steps = torch.arange(
            1, self.config.horizon_seconds + 1, device=context.device, dtype=context.dtype
        )
        phase = steps / self.config.horizon_seconds
        time_features = torch.stack(
            [phase, torch.sin(torch.pi * phase), torch.cos(torch.pi * phase)], dim=-1
        ).unsqueeze(0).expand(batch, -1, -1)
        repeated = representation.unsqueeze(1).expand(-1, self.config.horizon_seconds, -1)
        design = torch.cat([repeated, time_features], dim=-1)
        workload = torch.sigmoid(self.workload_head(design).squeeze(-1))
        distribution_raw = self.residual_scale_head(design)
        state = (current_hr - rest).clamp_min(0.0)
        means: list[Tensor] = []
        for index in range(self.config.horizon_seconds):
            equilibrium = reserve * gain * workload[:, index]
            tau = torch.where(equilibrium >= state, tau_rise, tau_recovery)
            alpha = torch.exp(-1.0 / tau)
            state = alpha * state + (1.0 - alpha) * equilibrium
            means.append((rest + state).clamp(30.0, 220.0))
        return DistributionalForecast(
            mean=torch.stack(means, dim=1),
            scale=F.softplus(distribution_raw[..., 0]) + 0.25,
            degrees_of_freedom=F.softplus(distribution_raw[..., 1]) + 2.1,
        )
