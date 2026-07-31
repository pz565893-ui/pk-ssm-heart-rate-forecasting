#!/usr/bin/env python3
"""Build a causal PK-SSM cache from the frozen GoldenCheetah light subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "speed_mps",
    "speed_change_5s_mps",
    "power_w",
    "power_change_5s_w",
    "cadence_rpm",
    "cadence_change_5s_rpm",
    "altitude_m",
    "vertical_speed_5s_mps",
    "grade_percent_10s",
    "distance_km",
    "power_observed",
    "cadence_observed",
)


@dataclass(frozen=True)
class CacheConfig:
    context_seconds: int = 300
    horizon_seconds: int = 120
    minimum_hr_coverage: float = 0.90
    minimum_context_hr_coverage: float = 0.90
    maximum_training_origins_per_session: int = 12
    maximum_evaluation_origins_per_session: int = 6
    maximum_events_per_session: int = 8
    candidate_training_stride_seconds: int = 60
    evaluation_stride_seconds: int = 120
    minimum_event_separation_seconds: int = 60
    minimum_hr_bpm: float = 30.0
    maximum_hr_bpm: float = 220.0
    local_median_window_seconds: int = 11
    maximum_local_hr_deviation_bpm: float = 25.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y/%m/%d %H:%M:%S UTC").replace(
        tzinfo=timezone.utc
    )


def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def causal_fill(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    series = pd.Series(values, dtype=float).ffill().fillna(default)
    return series.to_numpy(dtype=float)


def backward_change(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.zeros_like(values, dtype=float)
    if len(values) > lag:
        output[lag:] = values[lag:] - values[:-lag]
    return output


def resample_session(path: Path, start_unix: int, config: CacheConfig) -> dict[str, Any]:
    frame = pd.read_csv(path, low_memory=False)
    seconds = numeric_column(frame, "secs")
    valid_second = seconds.notna() & np.isfinite(seconds)
    frame = frame.loc[valid_second].copy()
    frame["grid_second"] = np.rint(seconds.loc[valid_second]).astype(int)
    frame = frame.groupby("grid_second", sort=True).last()
    first_second = int(frame.index.min())
    last_second = int(frame.index.max())
    grid = np.arange(first_second, last_second + 1, dtype=int)
    frame = frame.reindex(grid)

    hr_raw = numeric_column(frame, "hr").to_numpy(dtype=float)
    hr_range_valid = (
        np.isfinite(hr_raw)
        & (hr_raw >= config.minimum_hr_bpm)
        & (hr_raw <= config.maximum_hr_bpm)
    )
    hr_for_median = pd.Series(hr_raw).where(hr_range_valid)
    local_median = hr_for_median.rolling(
        config.local_median_window_seconds,
        center=True,
        min_periods=3,
    ).median()
    local_ok = (
        np.abs(hr_for_median.to_numpy(dtype=float) - local_median.to_numpy(dtype=float))
        <= config.maximum_local_hr_deviation_bpm
    )
    local_missing = ~np.isfinite(local_median.to_numpy(dtype=float))
    hr_offline_valid = hr_range_valid & (local_ok | local_missing)
    hr_bpm = np.where(hr_range_valid, hr_raw, np.nan)

    distance_raw = numeric_column(frame, "km").to_numpy(dtype=float)
    power_raw = numeric_column(frame, "power").to_numpy(dtype=float)
    cadence_raw = numeric_column(frame, "cad").to_numpy(dtype=float)
    altitude_raw = numeric_column(frame, "alt").to_numpy(dtype=float)
    distance_observed = np.isfinite(distance_raw)
    power_observed = np.isfinite(power_raw)
    cadence_observed = np.isfinite(cadence_raw)
    altitude_observed = np.isfinite(altitude_raw)

    distance = causal_fill(distance_raw, 0.0)
    power = np.clip(causal_fill(power_raw, 0.0), 0.0, 2500.0)
    cadence = np.clip(causal_fill(cadence_raw, 0.0), 0.0, 250.0)
    altitude = causal_fill(altitude_raw, 0.0)
    distance_delta_m = np.maximum(backward_change(distance * 1000.0, 1), 0.0)
    speed = np.clip(distance_delta_m, 0.0, 30.0)
    speed = pd.Series(speed).rolling(3, min_periods=1).median().to_numpy(dtype=float)
    speed_change = backward_change(speed, 5)
    power_change = backward_change(power, 5)
    cadence_change = backward_change(cadence, 5)
    vertical_speed = backward_change(altitude, 5) / 5.0
    distance_10s_m = backward_change(distance * 1000.0, 10)
    altitude_10s_m = backward_change(altitude, 10)
    grade = np.divide(
        100.0 * altitude_10s_m,
        distance_10s_m,
        out=np.zeros_like(altitude_10s_m),
        where=distance_10s_m >= 5.0,
    )
    grade = np.clip(grade, -30.0, 30.0)
    features = np.column_stack(
        (
            speed,
            speed_change,
            power,
            power_change,
            cadence,
            cadence_change,
            altitude,
            vertical_speed,
            grade,
            distance,
            power_observed.astype(float),
            cadence_observed.astype(float),
        )
    )
    feature_valid = (
        distance_observed | power_observed | cadence_observed | altitude_observed
    )
    return {
        "unix_seconds": start_unix + grid.astype(np.int64),
        "hr_bpm": hr_bpm,
        "hr_range_valid": hr_range_valid,
        "hr_offline_valid": hr_offline_valid,
        "features": features,
        "feature_valid": feature_valid,
        "power_observed_fraction": float(power_observed.mean()),
        "cadence_observed_fraction": float(cadence_observed.mean()),
    }


def evenly_limit(indices: list[int], maximum: int) -> list[int]:
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum)
    return [indices[int(round(position))] for position in positions]


def detect_events(signal: dict[str, Any], valid_origins: set[int], config: CacheConfig) -> dict[int, str]:
    features = signal["features"]
    power_fraction = float(signal["power_observed_fraction"])
    cadence_fraction = float(signal["cadence_observed_fraction"])
    raw_events: list[tuple[int, float, str]] = []
    for index in sorted(valid_origins):
        if index < 40:
            continue
        speed_before = float(np.median(features[index - 40 : index - 10, 0]))
        speed_now = float(np.median(features[index - 9 : index + 1, 0]))
        speed_delta = speed_now - speed_before
        power_delta = 0.0
        cadence_delta = 0.0
        if power_fraction >= 0.20:
            power_delta = float(
                np.median(features[index - 9 : index + 1, 2])
                - np.median(features[index - 40 : index - 10, 2])
            )
        if cadence_fraction >= 0.20:
            cadence_delta = float(
                np.median(features[index - 9 : index + 1, 4])
                - np.median(features[index - 40 : index - 10, 4])
            )
        score = max(
            abs(speed_delta) / 0.60,
            abs(power_delta) / 35.0,
            abs(cadence_delta) / 12.0,
        )
        if score < 1.0:
            continue
        signed_driver = max(
            (speed_delta / 0.60, speed_delta),
            (power_delta / 35.0, power_delta),
            (cadence_delta / 12.0, cadence_delta),
            key=lambda pair: abs(pair[0]),
        )[1]
        event_type = "effort_increase" if signed_driver >= 0.0 else "effort_decrease"
        raw_events.append((index, score, event_type))

    selected: list[tuple[int, float, str]] = []
    for event in sorted(raw_events, key=lambda row: (-row[1], row[0])):
        if all(
            abs(event[0] - existing[0]) >= config.minimum_event_separation_seconds
            for existing in selected
        ):
            selected.append(event)
        if len(selected) >= config.maximum_events_per_session:
            break
    return {index: event_type for index, _, event_type in selected}


def valid_origins(signal: dict[str, Any], config: CacheConfig) -> tuple[list[int], dict[int, tuple[float, float]]]:
    hr_valid = signal["hr_offline_valid"].astype(float)
    feature_valid = signal["feature_valid"].astype(float)
    candidates: list[int] = []
    coverages: dict[int, tuple[float, float]] = {}
    final_origin = len(hr_valid) - config.horizon_seconds - 1
    for origin in range(config.context_seconds - 1, final_origin + 1):
        context_start = origin - config.context_seconds + 1
        future = slice(origin + 1, origin + config.horizon_seconds + 1)
        context_hr_coverage = float(hr_valid[context_start : origin + 1].mean())
        future_hr_coverage = float(hr_valid[future].mean())
        context_feature_coverage = float(feature_valid[context_start : origin + 1].mean())
        if (
            not bool(signal["hr_offline_valid"][origin])
            or context_hr_coverage < config.minimum_context_hr_coverage
            or future_hr_coverage < config.minimum_hr_coverage
            or context_feature_coverage < 0.50
        ):
            continue
        candidates.append(origin)
        coverages[origin] = (future_hr_coverage, context_feature_coverage)
    return candidates, coverages


def stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot fit normalization from an empty feature")
    standard_deviation = float(np.std(finite))
    return {
        "mean": float(np.mean(finite)),
        "std": standard_deviation if standard_deviation >= 1.0e-8 else 1.0,
        "median": float(np.median(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q99": float(np.quantile(finite, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("splits/goldencheetah_light_v1/session_manifest.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/goldencheetah_transition_v1"),
    )
    args = parser.parse_args()
    config = CacheConfig()

    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    output_dir = args.output_root / "outer_fold_0"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite GoldenCheetah cache: {output_dir}")

    rows = read_manifest(args.manifest)
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=False)
    origin_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    train_hr: list[np.ndarray] = []
    train_features: list[np.ndarray] = []
    role_counts: dict[str, int] = {}
    protocol_counts: dict[str, int] = {}

    role_map = {
        "support_train": "train",
        "validation": "validation",
        "calibration": "calibration",
        "temporal_test": "test",
    }
    for manifest_row in rows:
        session_id = manifest_row["session_id"]
        participant_id = manifest_row["user_id"]
        role = role_map[manifest_row["temporal_role"]]
        source_path = args.root / manifest_row["source_csv_relative"]
        start_unix = int(parse_utc(manifest_row["recorded_time_utc"]).timestamp())
        signal = resample_session(source_path, start_unix, config)
        canonical = manifest_row["sport_canonical"]
        if canonical == "bike":
            protocol = "AEROBIC"
            protocol_version = "GC_BIKE"
        elif canonical == "run":
            protocol = "ANAEROBIC"
            protocol_version = "GC_RUN"
        elif signal["power_observed_fraction"] >= 0.20 or signal["cadence_observed_fraction"] >= 0.20:
            protocol = "AEROBIC"
            protocol_version = f"GC_INFERRED_BIKE_{canonical.upper()}"
        else:
            protocol = "ANAEROBIC"
            protocol_version = f"GC_OTHER_{canonical.upper()}"
        sex = {"f": "female", "m": "male"}.get(manifest_row["gender"], "unknown")

        candidates, coverages = valid_origins(signal, config)
        candidate_set = set(candidates)
        events = detect_events(signal, candidate_set, config)
        training_candidates = [
            index
            for index in candidates
            if (index - (config.context_seconds - 1))
            % config.candidate_training_stride_seconds
            == 0
        ]
        training_indices = set(
            evenly_limit(training_candidates, config.maximum_training_origins_per_session)
            if role == "train"
            else []
        )
        evaluation_candidates = [
            index
            for index in candidates
            if (index - (config.context_seconds - 1)) % config.evaluation_stride_seconds
            == 0
        ]
        evaluation_indices = set(
            evenly_limit(
                evaluation_candidates, config.maximum_evaluation_origins_per_session
            )
        )
        retained = sorted(training_indices | evaluation_indices | set(events))
        event_unix_seconds = np.array(
            [signal["unix_seconds"][index] for index in sorted(events)], dtype=np.int64
        )

        np.savez_compressed(
            session_dir / f"{session_id}.npz",
            unix_seconds=signal["unix_seconds"],
            hr_bpm=signal["hr_bpm"],
            hr_range_valid=signal["hr_range_valid"].astype(np.uint8),
            hr_offline_qc_valid=signal["hr_offline_valid"].astype(np.uint8),
            acc_features=signal["features"],
            acc_valid=signal["feature_valid"].astype(np.uint8),
            tag_unix_seconds=event_unix_seconds,
            acc_feature_names=np.asarray(FEATURE_NAMES),
            participant_id=np.asarray(participant_id),
            protocol=np.asarray(protocol),
            role=np.asarray(role),
            sex=np.asarray(sex),
            protocol_version=np.asarray(protocol_version),
        )

        for index in retained:
            future_hr_coverage, feature_coverage = coverages[index]
            event_type = events.get(index, "")
            unix_second = int(signal["unix_seconds"][index])
            origin_rows.append(
                {
                    "origin_id": f"{session_id}:{unix_second}",
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "protocol": protocol,
                    "role": role,
                    "sex": sex,
                    "protocol_version": protocol_version,
                    "origin_index": index,
                    "origin_unix_second": unix_second,
                    "training_stride_origin": int(index in training_indices),
                    "evaluation_stride_origin": int(index in evaluation_indices),
                    "tagged_event_origin": int(index in events),
                    "event_types": event_type,
                    "interval_hr_qc_coverage": future_hr_coverage,
                    "interval_acc_coverage": feature_coverage,
                }
            )
            if index in events:
                event_rows.append(
                    {
                        "event_id": f"{session_id}:{event_type}:{unix_second}",
                        "origin_unix_second": unix_second,
                        "event_type": event_type,
                        "interval_seconds": "",
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "protocol": protocol,
                        "role": role,
                        "origin_index": index,
                        "sex": sex,
                        "protocol_version": protocol_version,
                    }
                )
        if role == "train":
            train_hr.append(signal["hr_bpm"][signal["hr_offline_valid"]])
            train_features.append(signal["features"][signal["feature_valid"]])
        role_counts[role] = role_counts.get(role, 0) + 1
        protocol_counts[protocol_version] = protocol_counts.get(protocol_version, 0) + 1

    if not origin_rows or not event_rows or not train_hr or not train_features:
        raise RuntimeError("GoldenCheetah cache generation produced an empty required artifact")
    normalization = {
        "fit_scope": "support_train sessions only",
        "hr_bpm": stats(np.concatenate(train_hr)),
        "acc_features": {
            name: stats(np.concatenate([array[:, column] for array in train_features]))
            for column, name in enumerate(FEATURE_NAMES)
        },
    }
    origin_fields = [
        "origin_id",
        "participant_id",
        "session_id",
        "protocol",
        "role",
        "sex",
        "protocol_version",
        "origin_index",
        "origin_unix_second",
        "training_stride_origin",
        "evaluation_stride_origin",
        "tagged_event_origin",
        "event_types",
        "interval_hr_qc_coverage",
        "interval_acc_coverage",
    ]
    event_fields = [
        "event_id",
        "origin_unix_second",
        "event_type",
        "interval_seconds",
        "participant_id",
        "session_id",
        "protocol",
        "role",
        "origin_index",
        "sex",
        "protocol_version",
    ]
    write_csv(output_dir / "origin_manifest.csv", origin_rows, origin_fields)
    write_csv(output_dir / "event_manifest.csv", event_rows, event_fields)
    (output_dir / "training_normalization.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    policy = {
        "artifact_status": "goldencheetah_light_v1_causal_transition_cache",
        "split_manifest": str(args.manifest),
        "split_manifest_sha256": sha256_file(args.manifest),
        "source_root": str(args.root.resolve()),
        "sessions": len(rows),
        "participants": len({row["user_id"] for row in rows}),
        "origins": len(origin_rows),
        "tagged_event_origins": len(event_rows),
        "roles": role_counts,
        "protocol_versions": protocol_counts,
        "activity_index_alias": {
            "AEROBIC": "bike or power/cadence-supported activity",
            "ANAEROBIC": "run or other activity without cycling channels",
        },
        "history_contract": (
            "Only same-participant sessions ending strictly before the target session "
            "start may enter the history tensor"
        ),
        "future_input_policy": (
            "No future HR, distance, speed, power, cadence, or altitude enters the model input"
        ),
        "event_detection_policy": (
            "Effort transitions use only the 40 seconds ending at the forecast origin"
        ),
        "preprocessing_config": asdict(config),
    }
    policy_path = output_dir / "cache_policy.json"
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generated = (
        output_dir / "origin_manifest.csv",
        output_dir / "event_manifest.csv",
        output_dir / "training_normalization.json",
        policy_path,
    )
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in generated),
        encoding="utf-8",
    )
    print(json.dumps(policy, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
