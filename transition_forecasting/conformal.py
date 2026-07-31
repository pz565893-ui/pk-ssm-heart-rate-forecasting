"""User-disjoint conformal calibration for trajectory forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def assert_disjoint_user_sets(**named_user_sets: Iterable[str]) -> None:
    normalized = {
        name: {str(user) for user in users}
        for name, users in named_user_sets.items()
    }
    names = list(normalized)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = normalized[first] & normalized[second]
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(
                    f"User leakage between {first} and {second}: {examples}"
                )


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if len(scores) == 0:
        raise ValueError("No finite calibration scores are available")
    rank = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    try:
        return float(np.quantile(scores, rank, method="higher"))
    except TypeError:
        return float(np.quantile(scores, rank, interpolation="higher"))


@dataclass(frozen=True)
class CalibrationSummary:
    alpha: float
    pointwise_horizons_seconds: tuple[int, ...]
    pointwise_multipliers: dict[int, float]
    simultaneous_multiplier: float
    calibration_trajectories: int
    calibration_users: int


class ConformalTrajectoryCalibrator:
    def __init__(
        self,
        alpha: float = 0.05,
        pointwise_horizons_seconds: tuple[int, ...] = (30, 60, 120),
        epsilon: float = 1.0e-6,
    ) -> None:
        self.alpha = alpha
        self.pointwise_horizons_seconds = pointwise_horizons_seconds
        self.epsilon = epsilon
        self.summary: CalibrationSummary | None = None

    def fit(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        scale: np.ndarray,
        calibration_user_ids: Iterable[str],
        valid_mask: np.ndarray | None = None,
    ) -> CalibrationSummary:
        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        scale = np.asarray(scale, dtype=float)
        if observed.ndim != 2 or observed.shape != predicted.shape or observed.shape != scale.shape:
            raise ValueError("observed, predicted, and scale must share [trajectory, horizon] shape")
        user_ids = np.asarray(list(calibration_user_ids), dtype=str)
        if len(user_ids) != observed.shape[0]:
            raise ValueError("One calibration user ID is required per trajectory")
        valid = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(scale)
        valid &= scale > 0.0
        if valid_mask is not None:
            supplied = np.asarray(valid_mask, dtype=bool)
            if supplied.shape != observed.shape:
                raise ValueError("valid_mask must match trajectory shape")
            valid &= supplied
        normalized = np.full(observed.shape, np.nan, dtype=float)
        normalized[valid] = np.abs(observed[valid] - predicted[valid]) / (
            scale[valid] + self.epsilon
        )
        pointwise: dict[int, float] = {}
        for horizon in self.pointwise_horizons_seconds:
            index = horizon - 1
            if index < 0 or index >= observed.shape[1]:
                raise ValueError(f"Horizon {horizon} is outside the trajectory")
            pointwise[horizon] = finite_sample_quantile(
                normalized[:, index], self.alpha
            )
        with np.errstate(all="ignore"):
            simultaneous_scores = np.nanmax(normalized, axis=1)
        simultaneous = finite_sample_quantile(simultaneous_scores, self.alpha)
        self.summary = CalibrationSummary(
            alpha=self.alpha,
            pointwise_horizons_seconds=self.pointwise_horizons_seconds,
            pointwise_multipliers=pointwise,
            simultaneous_multiplier=simultaneous,
            calibration_trajectories=observed.shape[0],
            calibration_users=len(set(user_ids.tolist())),
        )
        return self.summary

    def simultaneous_interval(
        self, predicted: np.ndarray, scale: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.summary is None:
            raise RuntimeError("The calibrator must be fitted before interval construction")
        predicted = np.asarray(predicted, dtype=float)
        scale = np.asarray(scale, dtype=float)
        if predicted.shape != scale.shape:
            raise ValueError("predicted and scale must have equal shape")
        radius = self.summary.simultaneous_multiplier * scale
        return predicted - radius, predicted + radius

    def pointwise_interval(
        self,
        predicted: np.ndarray,
        scale: np.ndarray,
        horizon_seconds: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.summary is None:
            raise RuntimeError("The calibrator must be fitted before interval construction")
        if horizon_seconds not in self.summary.pointwise_multipliers:
            raise ValueError(f"Horizon {horizon_seconds} was not calibrated")
        predicted = np.asarray(predicted, dtype=float)
        scale = np.asarray(scale, dtype=float)
        radius = self.summary.pointwise_multipliers[horizon_seconds] * scale
        return predicted - radius, predicted + radius
