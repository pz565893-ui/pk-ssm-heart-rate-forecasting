"""Leakage-controlled preprocessing for the structured exercise dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACC_FEATURE_NAMES = (
    "acc_x_mean_g",
    "acc_y_mean_g",
    "acc_z_mean_g",
    "acc_x_std_g",
    "acc_y_std_g",
    "acc_z_std_g",
    "acc_magnitude_mean_g",
    "acc_magnitude_std_g",
    "acc_dynamic_magnitude_mean_g",
    "acc_dynamic_magnitude_std_g",
    "acc_energy_g2",
    "acc_saturation_fraction",
)


@dataclass(frozen=True)
class PreprocessingConfig:
    context_seconds: int = 300
    horizon_seconds: int = 120
    training_stride_seconds: int = 10
    evaluation_stride_seconds: int = 120
    minimum_hr_bpm: float = 30.0
    maximum_hr_bpm: float = 220.0
    local_median_window_seconds: int = 11
    maximum_local_deviation_bpm: float = 25.0
    minimum_hr_coverage: float = 0.90
    expected_acc_rate_hz: int = 32
    minimum_acc_fraction_per_second: float = 0.90
    gravity_time_constant_seconds: float = 2.0

    @property
    def minimum_acc_samples_per_second(self) -> int:
        return math.ceil(
            self.expected_acc_rate_hz * self.minimum_acc_fraction_per_second
        )


@dataclass
class SessionSignals:
    session_id: str
    participant_id: str
    protocol: str
    role: str
    sex: str
    protocol_version: str
    unix_seconds: np.ndarray
    hr_bpm: np.ndarray
    hr_range_valid: np.ndarray
    hr_offline_qc_valid: np.ndarray
    acc_features: np.ndarray
    acc_valid: np.ndarray
    tag_unix_seconds: np.ndarray
    source_hr_path: str
    source_acc_path: str
    source_tags_path: str

    @property
    def joint_valid(self) -> np.ndarray:
        return self.hr_offline_qc_valid & self.acc_valid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_participant(value: str) -> str:
    value = re.sub(r"_(?:a|b)$", "", value.strip(), flags=re.IGNORECASE)
    match = re.fullmatch(r"([sSfF])(\d{1,3})", value)
    if not match:
        return value
    prefix = "S" if match.group(1).lower() == "s" else "f"
    return f"{prefix}{int(match.group(2)):02d}"


def parse_timestamp(value: str) -> float:
    token = value.strip()
    if not token:
        raise ValueError("Empty timestamp")
    try:
        numeric = float(token)
    except ValueError:
        normalized = token.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    if numeric > 1.0e12:
        numeric /= 1000.0
    return numeric


def _read_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]


def _first_nonempty(row: Iterable[str]) -> str:
    for value in row:
        if value.strip():
            return value
    raise ValueError("Row has no nonempty value")


def read_empatica_hr(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    rows = _read_rows(path)
    if len(rows) < 3:
        raise ValueError(f"HR file is too short: {path}")
    start = parse_timestamp(_first_nonempty(rows[0]))
    sample_rate = float(_first_nonempty(rows[1]))
    if sample_rate <= 0.0:
        raise ValueError(f"Invalid HR sample rate in {path}")
    values = np.asarray([float(_first_nonempty(row)) for row in rows[2:]], dtype=float)
    sample_seconds = np.floor(start + np.arange(len(values)) / sample_rate).astype(np.int64)
    unique_seconds, inverse = np.unique(sample_seconds, return_inverse=True)
    aggregated = np.full(len(unique_seconds), np.nan, dtype=float)
    for index in range(len(unique_seconds)):
        second_values = values[inverse == index]
        finite = second_values[np.isfinite(second_values)]
        if len(finite):
            aggregated[index] = float(np.median(finite))
    return unique_seconds, aggregated, sample_rate


def read_empatica_acc(
    path: Path,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows = _read_rows(path)
    if len(rows) < 3:
        raise ValueError(f"ACC file is too short: {path}")
    start = parse_timestamp(_first_nonempty(rows[0]))
    sample_rate = float(_first_nonempty(rows[1]))
    if sample_rate <= 0.0:
        raise ValueError(f"Invalid ACC sample rate in {path}")
    raw = np.asarray(
        [[float(row[column]) for column in range(3)] for row in rows[2:] if len(row) >= 3],
        dtype=float,
    )
    sample_seconds = np.floor(start + np.arange(len(raw)) / sample_rate).astype(np.int64)
    unique_seconds = np.unique(sample_seconds)
    features = np.full((len(unique_seconds), len(ACC_FEATURE_NAMES)), np.nan, dtype=float)
    valid = np.zeros(len(unique_seconds), dtype=bool)
    gravity: np.ndarray | None = None
    gravity_alpha = math.exp(-1.0 / config.gravity_time_constant_seconds)
    minimum_samples = max(
        config.minimum_acc_samples_per_second,
        math.ceil(sample_rate * config.minimum_acc_fraction_per_second),
    )
    for index, second in enumerate(unique_seconds):
        second_raw = raw[sample_seconds == second]
        finite_rows = second_raw[np.all(np.isfinite(second_raw), axis=1)]
        if len(second_raw) < minimum_samples or len(finite_rows) < minimum_samples:
            continue
        samples_g = finite_rows / 64.0
        axis_mean = np.mean(samples_g, axis=0)
        axis_std = np.std(samples_g, axis=0)
        if gravity is None:
            gravity = axis_mean.copy()
        else:
            gravity = gravity_alpha * gravity + (1.0 - gravity_alpha) * axis_mean
        magnitude = np.linalg.norm(samples_g, axis=1)
        dynamic_magnitude = np.linalg.norm(samples_g - gravity[None, :], axis=1)
        energy = float(np.mean(np.sum(samples_g**2, axis=1)))
        saturation = float(np.mean(np.any(np.abs(finite_rows) >= 127.0, axis=1)))
        features[index] = np.concatenate(
            [
                axis_mean,
                axis_std,
                np.asarray(
                    [
                        np.mean(magnitude),
                        np.std(magnitude),
                        np.mean(dynamic_magnitude),
                        np.std(dynamic_magnitude),
                        energy,
                        saturation,
                    ]
                ),
            ]
        )
        valid[index] = True
    return unique_seconds, features, valid, sample_rate


def read_tags(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.empty(0, dtype=np.int64)
    tags: list[int] = []
    for row in _read_rows(path):
        try:
            tags.append(int(math.floor(parse_timestamp(_first_nonempty(row)))))
        except (ValueError, OverflowError):
            continue
    return np.asarray(sorted(set(tags)), dtype=np.int64)


def _offline_hr_qc(hr: np.ndarray, config: PreprocessingConfig) -> tuple[np.ndarray, np.ndarray]:
    range_valid = (
        np.isfinite(hr)
        & (hr >= config.minimum_hr_bpm)
        & (hr <= config.maximum_hr_bpm)
    )
    outlier = np.zeros(len(hr), dtype=bool)
    half = config.local_median_window_seconds // 2
    for index in np.flatnonzero(range_valid):
        start = max(0, index - half)
        end = min(len(hr), index + half + 1)
        local = hr[start:end][range_valid[start:end]]
        if len(local) >= 3 and abs(hr[index] - np.median(local)) > config.maximum_local_deviation_bpm:
            outlier[index] = True
    return range_valid, range_valid & ~outlier


def _protocol_from_path(path: Path) -> str:
    for part in path.parts:
        upper = part.upper()
        if upper in {"AEROBIC", "ANAEROBIC"}:
            return upper
    raise ValueError(f"Could not infer exercise protocol from {path}")


def discover_session_directories(data_dir: Path) -> list[Path]:
    sessions = []
    for hr_path in data_dir.rglob("HR.csv"):
        try:
            _protocol_from_path(hr_path)
        except ValueError:
            continue
        if (hr_path.parent / "ACC.csv").is_file():
            sessions.append(hr_path.parent)
    return sorted(set(sessions), key=lambda path: path.as_posix())


def load_fold_roles(split_manifest: Path, outer_fold: int) -> dict[str, dict[str, str]]:
    with split_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"outer_fold", "participant_id", "role", "sex", "cohort"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Split manifest is missing {sorted(required - set(reader.fieldnames or []))}"
            )
        roles: dict[str, dict[str, str]] = {}
        for row in reader:
            if int(row["outer_fold"]) != outer_fold:
                continue
            participant = normalize_participant(row["participant_id"])
            if participant in roles:
                raise ValueError(f"Duplicate participant in outer fold {outer_fold}: {participant}")
            role = row["role"].strip().lower()
            if role not in {"train", "validation", "calibration", "test"}:
                raise ValueError(f"Invalid split role: {role}")
            roles[participant] = {
                "role": role,
                "sex": row["sex"].strip().lower(),
                "cohort": row["cohort"].strip(),
            }
    if not roles:
        raise ValueError(f"No participants found for outer fold {outer_fold}")
    if all(record["sex"] == "unknown" for record in roles.values()):
        raise ValueError(
            "The split manifest has no resolved sex metadata and is provisional; refuse to build model windows"
        )
    return roles


def build_session_signals(
    session_dir: Path,
    data_dir: Path,
    roles: dict[str, dict[str, str]],
    config: PreprocessingConfig,
) -> SessionSignals:
    protocol = _protocol_from_path(session_dir)
    fragment_id = session_dir.name
    participant = normalize_participant(fragment_id)
    if participant not in roles:
        raise ValueError(f"Participant {participant} is absent from the split manifest")
    hr_path = session_dir / "HR.csv"
    acc_path = session_dir / "ACC.csv"
    tags_path = session_dir / "tags.csv"
    hr_seconds, hr_values, _ = read_empatica_hr(hr_path)
    acc_seconds, acc_features, acc_valid, _ = read_empatica_acc(acc_path, config)
    support_start = max(int(hr_seconds.min()), int(acc_seconds.min()))
    support_end = min(int(hr_seconds.max()), int(acc_seconds.max()))
    if support_end <= support_start:
        raise ValueError(f"No HR-ACC temporal intersection in {session_dir}")
    timeline = np.arange(support_start, support_end + 1, dtype=np.int64)
    hr = np.full(len(timeline), np.nan, dtype=float)
    hr_positions = hr_seconds - support_start
    hr_in = (hr_positions >= 0) & (hr_positions < len(timeline))
    hr[hr_positions[hr_in]] = hr_values[hr_in]
    aligned_acc = np.full((len(timeline), len(ACC_FEATURE_NAMES)), np.nan, dtype=float)
    aligned_acc_valid = np.zeros(len(timeline), dtype=bool)
    acc_positions = acc_seconds - support_start
    acc_in = (acc_positions >= 0) & (acc_positions < len(timeline))
    aligned_acc[acc_positions[acc_in]] = acc_features[acc_in]
    aligned_acc_valid[acc_positions[acc_in]] = acc_valid[acc_in]
    hr_range_valid, hr_offline_qc_valid = _offline_hr_qc(hr, config)
    tags = read_tags(tags_path)
    tags = tags[(tags >= support_start) & (tags <= support_end)]
    relative = session_dir.relative_to(data_dir).as_posix()
    session_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", relative)
    role_record = roles[participant]
    return SessionSignals(
        session_id=session_id,
        participant_id=participant,
        protocol=protocol,
        role=role_record["role"],
        sex=role_record["sex"],
        protocol_version=role_record["cohort"],
        unix_seconds=timeline,
        hr_bpm=hr,
        hr_range_valid=hr_range_valid,
        hr_offline_qc_valid=hr_offline_qc_valid,
        acc_features=aligned_acc,
        acc_valid=aligned_acc_valid,
        tag_unix_seconds=tags,
        source_hr_path=hr_path.as_posix(),
        source_acc_path=acc_path.as_posix(),
        source_tags_path=tags_path.as_posix() if tags_path.is_file() else "",
    )


def eligible_origin_mask(
    session: SessionSignals,
    config: PreprocessingConfig,
) -> np.ndarray:
    eligible = np.zeros(len(session.unix_seconds), dtype=bool)
    first = config.context_seconds - 1
    last = len(session.unix_seconds) - config.horizon_seconds - 1
    for origin in range(first, last + 1):
        start = origin - config.context_seconds + 1
        stop = origin + config.horizon_seconds + 1
        if not np.all(session.acc_valid[start:stop]):
            continue
        hr_coverage = np.mean(session.hr_offline_qc_valid[start:stop])
        if hr_coverage >= config.minimum_hr_coverage:
            eligible[origin] = True
    return eligible


def derive_tagged_events(session: SessionSignals) -> list[dict[str, Any]]:
    tags = session.tag_unix_seconds
    events: list[dict[str, Any]] = []
    if session.protocol == "AEROBIC":
        for index, tag in enumerate(tags):
            events.append(
                {
                    "event_id": f"{session.session_id}:aerobic_stage:{index}",
                    "origin_unix_second": int(tag),
                    "event_type": "aerobic_stage_boundary",
                    "interval_seconds": "",
                }
            )
        return events
    for index in range(len(tags) - 1):
        start = int(tags[index])
        end = int(tags[index + 1])
        duration = end - start
        if 20 <= duration <= 75:
            events.append(
                {
                    "event_id": f"{session.session_id}:sprint_onset:{index}",
                    "origin_unix_second": start,
                    "event_type": "sprint_onset",
                    "interval_seconds": duration,
                }
            )
            events.append(
                {
                    "event_id": f"{session.session_id}:sprint_offset:{index}",
                    "origin_unix_second": end,
                    "event_type": "sprint_offset",
                    "interval_seconds": duration,
                }
            )
        if 150 <= duration <= 300:
            events.append(
                {
                    "event_id": f"{session.session_id}:recovery_onset:{index}",
                    "origin_unix_second": start,
                    "event_type": "recovery_onset",
                    "interval_seconds": duration,
                }
            )
            events.append(
                {
                    "event_id": f"{session.session_id}:recovery_end:{index}",
                    "origin_unix_second": end,
                    "event_type": "recovery_end",
                    "interval_seconds": duration,
                }
            )
    return events


def build_origin_rows(
    session: SessionSignals,
    config: PreprocessingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = eligible_origin_mask(session, config)
    eligible_indices = np.flatnonzero(eligible)
    if not len(eligible_indices):
        return [], []
    anchor = int(eligible_indices[0])
    training_indices = {
        index
        for index in range(anchor, len(eligible), config.training_stride_seconds)
        if eligible[index]
    }
    evaluation_indices = {
        index
        for index in range(anchor, len(eligible), config.evaluation_stride_seconds)
        if eligible[index]
    }
    time_to_index = {int(second): index for index, second in enumerate(session.unix_seconds)}
    event_rows: list[dict[str, Any]] = []
    event_by_index: dict[int, list[str]] = {}
    for event in derive_tagged_events(session):
        index = time_to_index.get(int(event["origin_unix_second"]))
        if index is None or not eligible[index]:
            continue
        event_by_index.setdefault(index, []).append(str(event["event_type"]))
        event_rows.append(
            {
                **event,
                "participant_id": session.participant_id,
                "session_id": session.session_id,
                "protocol": session.protocol,
                "role": session.role,
                "origin_index": index,
                "sex": session.sex,
                "protocol_version": session.protocol_version,
            }
        )
    all_indices = sorted(training_indices | evaluation_indices | set(event_by_index))
    origin_rows = []
    for index in all_indices:
        start = index - config.context_seconds + 1
        stop = index + config.horizon_seconds + 1
        origin_rows.append(
            {
                "origin_id": f"{session.session_id}:{int(session.unix_seconds[index])}",
                "participant_id": session.participant_id,
                "session_id": session.session_id,
                "protocol": session.protocol,
                "role": session.role,
                "sex": session.sex,
                "protocol_version": session.protocol_version,
                "origin_index": index,
                "origin_unix_second": int(session.unix_seconds[index]),
                "training_stride_origin": int(index in training_indices),
                "evaluation_stride_origin": int(index in evaluation_indices),
                "tagged_event_origin": int(index in event_by_index),
                "event_types": ";".join(sorted(set(event_by_index.get(index, [])))),
                "interval_hr_qc_coverage": float(
                    np.mean(session.hr_offline_qc_valid[start:stop])
                ),
                "interval_acc_coverage": float(np.mean(session.acc_valid[start:stop])),
            }
        )
    return origin_rows, event_rows


def save_session_cache(session: SessionSignals, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        unix_seconds=session.unix_seconds,
        hr_bpm=session.hr_bpm,
        hr_range_valid=session.hr_range_valid.astype(np.uint8),
        hr_offline_qc_valid=session.hr_offline_qc_valid.astype(np.uint8),
        acc_features=session.acc_features,
        acc_valid=session.acc_valid.astype(np.uint8),
        tag_unix_seconds=session.tag_unix_seconds,
        acc_feature_names=np.asarray(ACC_FEATURE_NAMES),
        participant_id=np.asarray(session.participant_id),
        protocol=np.asarray(session.protocol),
        role=np.asarray(session.role),
        sex=np.asarray(session.sex),
        protocol_version=np.asarray(session.protocol_version),
    )


def training_normalization(sessions: list[SessionSignals]) -> dict[str, Any]:
    training = [session for session in sessions if session.role == "train"]
    if not training:
        raise ValueError("No training sessions are available for normalization")
    hr_values = np.concatenate(
        [session.hr_bpm[session.hr_range_valid] for session in training]
    )
    acc_values = np.concatenate(
        [session.acc_features[session.acc_valid] for session in training], axis=0
    )
    if not len(hr_values) or not len(acc_values):
        raise ValueError("Training signals contain no valid HR or ACC values")
    return {
        "fit_scope": "training participants only",
        "hr_bpm": {
            "mean": float(np.mean(hr_values)),
            "std": float(max(np.std(hr_values), 1.0e-6)),
            "median": float(np.median(hr_values)),
            "q01": float(np.quantile(hr_values, 0.01)),
            "q99": float(np.quantile(hr_values, 0.99)),
        },
        "acc_features": {
            name: {
                "mean": float(np.nanmean(acc_values[:, index])),
                "std": float(max(np.nanstd(acc_values[:, index]), 1.0e-6)),
                "median": float(np.nanmedian(acc_values[:, index])),
                "q01": float(np.nanquantile(acc_values[:, index], 0.01)),
                "q99": float(np.nanquantile(acc_values[:, index], 0.99)),
            }
            for index, name in enumerate(ACC_FEATURE_NAMES)
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
