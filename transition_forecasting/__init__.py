"""Transition-focused heart-rate forecasting components."""

from .conformal import ConformalTrajectoryCalibrator, assert_disjoint_user_sets
from .baselines import (
    DampedTrendBaseline,
    DirectRecurrentForecaster,
    DistributionalForecast,
    FirstOrderKineticForecaster,
    PersistenceBaseline,
    TCNForecaster,
    TransformerForecaster,
    UnconstrainedResidualSSM,
)
from .classical import RidgeTrajectoryForecaster, XGBoostTrajectoryForecaster
from .pk_ssm import PKSSM, PKSSMConfig, PKSSMOutput, student_t_nll
from .transition_metrics import (
    KineticFit,
    TransitionMetrics,
    fit_two_phase_response,
    transition_metrics,
)

__all__ = [
    "ConformalTrajectoryCalibrator",
    "DampedTrendBaseline",
    "DirectRecurrentForecaster",
    "DistributionalForecast",
    "FirstOrderKineticForecaster",
    "KineticFit",
    "PKSSM",
    "PKSSMConfig",
    "PKSSMOutput",
    "PersistenceBaseline",
    "RidgeTrajectoryForecaster",
    "TCNForecaster",
    "TransformerForecaster",
    "TransitionMetrics",
    "UnconstrainedResidualSSM",
    "XGBoostTrajectoryForecaster",
    "assert_disjoint_user_sets",
    "fit_two_phase_response",
    "student_t_nll",
    "transition_metrics",
]
