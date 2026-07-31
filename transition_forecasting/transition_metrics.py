"""Event-level metrics that expose timing error and over-smoothing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class KineticFit:
    converged: bool
    baseline_bpm: float
    asymptote_bpm: float
    fast_tau_seconds: float
    slow_tau_seconds: float
    fast_fraction: float
    rmse_bpm: float
    amplitude_bpm: float


@dataclass(frozen=True)
class TransitionMetrics:
    trajectory_mae_bpm: float
    signed_lag_seconds: float
    absolute_lag_seconds: float
    slope_attenuation_ratio: float
    total_variation_ratio: float
    recovered_peak_change_fraction: float
    observed_peak_slope_bpm_per_second: float
    predicted_peak_slope_bpm_per_second: float
    observed_total_variation_bpm: float
    predicted_total_variation_bpm: float
    observed_peak_change_bpm: float
    predicted_peak_change_bpm: float
    observed_fast_tau_seconds: float
    predicted_fast_tau_seconds: float
    fast_tau_absolute_error_seconds: float
    observed_slow_tau_seconds: float
    predicted_slow_tau_seconds: float
    slow_tau_absolute_error_seconds: float
    valid_seconds: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _paired_valid(
    observed: np.ndarray,
    predicted: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted trajectories must have equal shape")
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if supplied.shape != observed.shape:
            raise ValueError("valid_mask must match the trajectory shape")
        valid &= supplied
    return observed[valid], predicted[valid]


def _peak_window_slope(values: np.ndarray, window_seconds: int, sample_period: float) -> float:
    if len(values) < max(3, window_seconds):
        return float("nan")
    window = max(3, int(round(window_seconds / sample_period)))
    time = np.arange(window, dtype=float) * sample_period
    centered_time = time - time.mean()
    denominator = np.sum(centered_time**2)
    slopes = []
    for start in range(0, len(values) - window + 1):
        segment = values[start : start + window]
        slopes.append(float(np.sum(centered_time * (segment - segment.mean())) / denominator))
    return float(np.max(np.abs(slopes)))


def _signed_lag(
    observed: np.ndarray,
    predicted: np.ndarray,
    max_lag_seconds: int,
    sample_period: float,
) -> float:
    maximum_lag = int(round(max_lag_seconds / sample_period))
    candidates: list[tuple[float, int]] = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag > 0:
            reference = observed[:-lag]
            estimate = predicted[lag:]
        elif lag < 0:
            reference = observed[-lag:]
            estimate = predicted[:lag]
        else:
            reference = observed
            estimate = predicted
        if len(reference) < 5 or np.std(reference) < 1.0e-8 or np.std(estimate) < 1.0e-8:
            correlation = -np.inf
        else:
            correlation = float(np.corrcoef(reference, estimate)[0, 1])
        candidates.append((correlation, lag))
    best_correlation = max(value for value, _ in candidates)
    if not np.isfinite(best_correlation):
        return float("nan")
    tied = [lag for value, lag in candidates if np.isclose(value, best_correlation)]
    best_lag = min(tied, key=lambda lag: (abs(lag), lag))
    return float(best_lag * sample_period)


def fit_two_phase_response(
    values: np.ndarray,
    sample_period: float = 1.0,
    minimum_amplitude_bpm: float = 5.0,
) -> KineticFit:
    """Fit a descriptive two-exponential transition response.

    The fitted observed constants are trajectory-derived references, not direct
    physiological measurements. Fits below the amplitude gate return NaNs.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return KineticFit(False, *(float("nan") for _ in range(7)))
    baseline = float(np.median(values[: min(5, len(values))]))
    endpoint = float(np.median(values[-min(10, len(values)) :]))
    amplitude = abs(endpoint - baseline)
    if amplitude < minimum_amplitude_bpm:
        return KineticFit(
            False,
            baseline,
            endpoint,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            amplitude,
        )
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("scipy is required for kinetic reference fitting") from exc

    time = np.arange(len(values), dtype=float) * sample_period

    def sigmoid(value: float) -> float:
        return 1.0 / (1.0 + np.exp(-value))

    def unpack(parameters: np.ndarray) -> tuple[float, float, float, float]:
        asymptote = float(parameters[0])
        fast_tau = float(np.exp(parameters[1]))
        slow_tau = fast_tau + float(np.exp(parameters[2]))
        fraction = sigmoid(float(parameters[3]))
        return asymptote, fast_tau, slow_tau, fraction

    def model(parameters: np.ndarray) -> np.ndarray:
        asymptote, fast_tau, slow_tau, fraction = unpack(parameters)
        decay = fraction * np.exp(-time / fast_tau) + (1.0 - fraction) * np.exp(
            -time / slow_tau
        )
        return asymptote + (baseline - asymptote) * decay

    initial = np.array([endpoint, np.log(10.0), np.log(50.0), 0.0], dtype=float)
    lower = np.array([30.0, np.log(1.0), np.log(1.0), -6.0])
    upper = np.array([220.0, np.log(120.0), np.log(600.0), 6.0])
    result = least_squares(
        lambda parameters: model(parameters) - values,
        x0=initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=2000,
    )
    asymptote, fast_tau, slow_tau, fraction = unpack(result.x)
    rmse = float(np.sqrt(np.mean((model(result.x) - values) ** 2)))
    return KineticFit(
        bool(result.success),
        baseline,
        asymptote,
        fast_tau,
        slow_tau,
        fraction,
        rmse,
        amplitude,
    )


def transition_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    origin_hr_bpm: float,
    valid_mask: np.ndarray | None = None,
    sample_period: float = 1.0,
    maximum_lag_seconds: int = 30,
    slope_window_seconds: int = 5,
    ratio_epsilon: float = 1.0e-6,
    fit_kinetics: bool = True,
) -> TransitionMetrics:
    """Compute one event-level transition profile before user aggregation."""
    observed, predicted = _paired_valid(observed, predicted, valid_mask)
    if len(observed) < 20:
        raise ValueError("At least 20 paired valid seconds are required")
    trajectory_mae = float(np.mean(np.abs(observed - predicted)))
    signed_lag = _signed_lag(
        observed, predicted, maximum_lag_seconds, sample_period
    )
    observed_slope = _peak_window_slope(
        observed, slope_window_seconds, sample_period
    )
    predicted_slope = _peak_window_slope(
        predicted, slope_window_seconds, sample_period
    )
    slope_ratio = predicted_slope / max(observed_slope, ratio_epsilon)
    observed_tv = float(np.sum(np.abs(np.diff(observed))))
    predicted_tv = float(np.sum(np.abs(np.diff(predicted))))
    tv_ratio = predicted_tv / max(observed_tv, ratio_epsilon)
    observed_peak_change = float(np.max(np.abs(observed - origin_hr_bpm)))
    predicted_peak_change = float(np.max(np.abs(predicted - origin_hr_bpm)))
    peak_fraction = predicted_peak_change / max(observed_peak_change, ratio_epsilon)

    if fit_kinetics:
        observed_fit = fit_two_phase_response(observed, sample_period)
        predicted_fit = fit_two_phase_response(predicted, sample_period)
    else:
        nan_fit = KineticFit(False, *(float("nan") for _ in range(7)))
        observed_fit = nan_fit
        predicted_fit = nan_fit
    fast_error = (
        abs(predicted_fit.fast_tau_seconds - observed_fit.fast_tau_seconds)
        if observed_fit.converged and predicted_fit.converged
        else float("nan")
    )
    slow_error = (
        abs(predicted_fit.slow_tau_seconds - observed_fit.slow_tau_seconds)
        if observed_fit.converged and predicted_fit.converged
        else float("nan")
    )
    return TransitionMetrics(
        trajectory_mae_bpm=trajectory_mae,
        signed_lag_seconds=signed_lag,
        absolute_lag_seconds=abs(signed_lag),
        slope_attenuation_ratio=float(slope_ratio),
        total_variation_ratio=float(tv_ratio),
        recovered_peak_change_fraction=float(peak_fraction),
        observed_peak_slope_bpm_per_second=observed_slope,
        predicted_peak_slope_bpm_per_second=predicted_slope,
        observed_total_variation_bpm=observed_tv,
        predicted_total_variation_bpm=predicted_tv,
        observed_peak_change_bpm=observed_peak_change,
        predicted_peak_change_bpm=predicted_peak_change,
        observed_fast_tau_seconds=observed_fit.fast_tau_seconds,
        predicted_fast_tau_seconds=predicted_fit.fast_tau_seconds,
        fast_tau_absolute_error_seconds=float(fast_error),
        observed_slow_tau_seconds=observed_fit.slow_tau_seconds,
        predicted_slow_tau_seconds=predicted_fit.slow_tau_seconds,
        slow_tau_absolute_error_seconds=float(slow_error),
        valid_seconds=len(observed),
    )
