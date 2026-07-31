"""Classical trajectory baselines fitted only on designated training users."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def extract_context_features(
    context: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert causal contexts into fixed features without future samples."""
    values = np.asarray(context, dtype=float)
    if values.ndim != 3:
        raise ValueError("context must have shape [samples, time, features]")
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape[:2]:
            raise ValueError("valid_mask must have shape [samples, time]")
        values = np.where(mask[..., None], values, np.nan)
    sample_count, time_count, feature_count = values.shape
    time = np.arange(time_count, dtype=float)
    centered_time = time - time.mean()
    denominator = np.sum(centered_time**2)
    feature_blocks = [
        values[:, -1, :],
        np.nanmean(values, axis=1),
        np.nanstd(values, axis=1),
        np.nanmin(values, axis=1),
        np.nanmax(values, axis=1),
    ]
    centered_values = values - np.nanmean(values, axis=1, keepdims=True)
    slopes = np.nansum(centered_values * centered_time[None, :, None], axis=1) / max(
        denominator, 1.0
    )
    feature_blocks.append(slopes)
    for recent_seconds in (30, 60):
        width = min(recent_seconds, time_count)
        feature_blocks.append(np.nanmean(values[:, -width:, :], axis=1))
        feature_blocks.append(np.nanstd(values[:, -width:, :], axis=1))
    missing_fraction = np.mean(~np.isfinite(values), axis=1)
    feature_blocks.append(missing_fraction)
    features = np.concatenate(feature_blocks, axis=1)
    if features.shape[0] != sample_count or features.shape[1] < feature_count:
        raise RuntimeError("Unexpected classical feature shape")
    column_medians = np.nanmedian(features, axis=0)
    column_medians = np.where(np.isfinite(column_medians), column_medians, 0.0)
    missing = ~np.isfinite(features)
    features[missing] = np.take(column_medians, np.where(missing)[1])
    return features


@dataclass
class ClassicalForecast:
    mean: np.ndarray
    scale: np.ndarray


class RidgeTrajectoryForecaster:
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.pipeline: Any = None
        self.residual_scale: np.ndarray | None = None

    def fit(
        self,
        context: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> "RidgeTrajectoryForecaster":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        target = np.asarray(target, dtype=float)
        if target.ndim != 2 or not np.all(np.isfinite(target)):
            raise ValueError("Ridge training targets must be finite [samples, horizon]")
        features = extract_context_features(context, valid_mask)
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.pipeline.fit(features, target)
        residual = target - self.pipeline.predict(features)
        self.residual_scale = np.maximum(
            np.nanmedian(np.abs(residual), axis=0) / 0.67448975,
            0.25,
        )
        return self

    def predict(
        self, context: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> ClassicalForecast:
        if self.pipeline is None or self.residual_scale is None:
            raise RuntimeError("RidgeTrajectoryForecaster is not fitted")
        mean = np.asarray(
            self.pipeline.predict(extract_context_features(context, valid_mask)),
            dtype=float,
        )
        scale = np.broadcast_to(self.residual_scale, mean.shape).copy()
        return ClassicalForecast(np.clip(mean, 30.0, 220.0), scale)


class XGBoostTrajectoryForecaster:
    """CPU-bounded XGBoost multi-output baseline.

    The wrapper deliberately uses one thread per estimator so outer-fold and
    random-seed orchestration controls total CPU use on a modest workstation.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 20260730,
    ) -> None:
        self.parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
        }
        self.model: Any = None
        self.residual_scale: np.ndarray | None = None

    def fit(
        self,
        context: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> "XGBoostTrajectoryForecaster":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is required for the prespecified tree baseline") from exc
        from sklearn.multioutput import MultiOutputRegressor

        target = np.asarray(target, dtype=float)
        if target.ndim != 2 or not np.all(np.isfinite(target)):
            raise ValueError("XGBoost training targets must be finite [samples, horizon]")
        features = extract_context_features(context, valid_mask)
        estimator = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
            **self.parameters,
        )
        self.model = MultiOutputRegressor(estimator, n_jobs=1)
        self.model.fit(features, target)
        residual = target - self.model.predict(features)
        self.residual_scale = np.maximum(
            np.nanmedian(np.abs(residual), axis=0) / 0.67448975,
            0.25,
        )
        return self

    def predict(
        self, context: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> ClassicalForecast:
        if self.model is None or self.residual_scale is None:
            raise RuntimeError("XGBoostTrajectoryForecaster is not fitted")
        mean = np.asarray(
            self.model.predict(extract_context_features(context, valid_mask)),
            dtype=float,
        )
        scale = np.broadcast_to(self.residual_scale, mean.shape).copy()
        return ClassicalForecast(np.clip(mean, 30.0, 220.0), scale)
