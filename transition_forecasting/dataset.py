"""PyTorch datasets backed by fold-specific transition caches."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


EVENT_INDEX = {
    "": 0,
    "aerobic_stage_boundary": 1,
    "sprint_onset": 2,
    "sprint_offset": 3,
    "recovery_onset": 4,
    "recovery_end": 5,
}
ACTIVITY_INDEX = {"AEROBIC": 1, "ANAEROBIC": 2}


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader)


def _interpolate_short_context_gaps(
    values: np.ndarray,
    valid: np.ndarray,
    maximum_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    filled = np.asarray(values, dtype=float).copy()
    observed = np.asarray(valid, dtype=bool).copy()
    index = 0
    while index < len(filled):
        if observed[index]:
            index += 1
            continue
        start = index
        while index < len(filled) and not observed[index]:
            index += 1
        end = index
        gap = end - start
        if gap <= maximum_gap and start > 0 and end < len(filled):
            if observed[start - 1] and observed[end]:
                filled[start:end] = np.linspace(
                    filled[start - 1], filled[end], gap + 2
                )[1:-1]
    return filled, observed


class TransitionWindowDataset(Dataset[dict[str, Any]]):
    """Causal 300 s context and 120 s target windows for one split role."""

    def __init__(
        self,
        fold_cache_dir: Path,
        role: str,
        origin_policy: str,
        schedule_aware: bool,
        mask_activity_identity: bool = False,
        maximum_interpolated_gap_seconds: int = 5,
        history_regime: str = "none",
        history_session_budget: int = 0,
        history_seconds_per_session: int = 300,
        maximum_history_sessions: int = 8,
    ) -> None:
        role = role.lower()
        if role not in {"train", "validation", "calibration", "test"}:
            raise ValueError(f"Unsupported role: {role}")
        if origin_policy not in {"training_stride", "evaluation_stride", "tagged_events"}:
            raise ValueError(f"Unsupported origin policy: {origin_policy}")
        if history_regime not in {"none", "train_prior", "role_prior", "all_prior"}:
            raise ValueError(f"Unsupported history regime: {history_regime}")
        if history_session_budget < -1:
            raise ValueError("history_session_budget must be -1, 0, or positive")
        if history_seconds_per_session <= 0:
            raise ValueError("history_seconds_per_session must be positive")
        if maximum_history_sessions <= 0:
            raise ValueError("maximum_history_sessions must be positive")
        if history_session_budget > maximum_history_sessions:
            raise ValueError(
                "history_session_budget exceeds maximum_history_sessions; "
                "increase the explicit safety cap"
            )
        self.fold_cache_dir = Path(fold_cache_dir)
        self.role = role
        self.origin_policy = origin_policy
        self.schedule_aware = schedule_aware
        self.mask_activity_identity = mask_activity_identity
        self.maximum_interpolated_gap_seconds = maximum_interpolated_gap_seconds
        self.history_regime = history_regime
        self.history_session_budget = history_session_budget
        self.history_seconds_per_session = history_seconds_per_session
        self.maximum_history_sessions = maximum_history_sessions
        policy_path = self.fold_cache_dir / "cache_policy.json"
        manifest_path = self.fold_cache_dir / "origin_manifest.csv"
        normalization_path = self.fold_cache_dir / "training_normalization.json"
        for required in (policy_path, manifest_path, normalization_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        self.policy = json.loads(policy_path.read_text(encoding="utf-8"))
        self.preprocessing = self.policy["preprocessing_config"]
        self.context_seconds = int(self.preprocessing["context_seconds"])
        self.horizon_seconds = int(self.preprocessing["horizon_seconds"])
        self.normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        all_rows = _read_manifest(manifest_path)
        flag = {
            "training_stride": "training_stride_origin",
            "evaluation_stride": "evaluation_stride_origin",
            "tagged_events": "tagged_event_origin",
        }[origin_policy]
        self.rows = [
            row
            for row in all_rows
            if row["role"].lower() == role and int(row[flag]) == 1
        ]
        if not self.rows:
            raise ValueError(
                f"No {origin_policy} origins are available for role {role} in {fold_cache_dir}"
            )
        self._sessions: dict[str, dict[str, np.ndarray]] = {}
        self.acc_feature_names = tuple(self.normalization["acc_features"].keys())
        self.input_feature_names = (
            "hr_standardized",
            *self.acc_feature_names,
            "hr_observed_mask",
            "acc_valid_mask",
            "elapsed_hours",
            "protocol_aerobic",
            "protocol_anaerobic",
        )
        self._session_metadata = self._index_sessions()
        self._history_sessions_by_target = self._build_history_index()
        maximum_available = max(
            (len(values) for values in self._history_sessions_by_target.values()),
            default=0,
        )
        if self.history_session_budget == -1 and maximum_available > maximum_history_sessions:
            raise ValueError(
                f"All-history mode requires {maximum_available} sessions, exceeding "
                f"maximum_history_sessions={maximum_history_sessions}"
            )
        if history_regime == "none" or history_session_budget == 0:
            self.history_capacity = 1
        elif history_session_budget > 0:
            self.history_capacity = history_session_budget
        else:
            self.history_capacity = max(1, maximum_available)

    @property
    def input_dim(self) -> int:
        return len(self.input_feature_names)

    def __len__(self) -> int:
        return len(self.rows)

    def _index_sessions(self) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        session_dir = self.fold_cache_dir / "sessions"
        for path in sorted(session_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                timeline = archive["unix_seconds"]
                if not len(timeline):
                    continue
                session_id = path.stem
                metadata[session_id] = {
                    "session_id": session_id,
                    "participant_id": str(archive["participant_id"].item()),
                    "protocol": str(archive["protocol"].item()),
                    "role": str(archive["role"].item()).lower(),
                    "start": int(timeline[0]),
                    "end": int(timeline[-1]),
                }
        missing = sorted({row["session_id"] for row in self.rows} - set(metadata))
        if missing:
            raise FileNotFoundError(
                f"Origin manifest references sessions missing from the cache: {missing[:5]}"
            )
        return metadata

    def _history_is_allowed(
        self,
        history: dict[str, Any],
        target: dict[str, Any],
    ) -> bool:
        if self.history_regime == "none" or self.history_session_budget == 0:
            return False
        if history["participant_id"] != target["participant_id"]:
            return False
        if history["session_id"] == target["session_id"]:
            return False
        if history["end"] >= target["start"]:
            return False
        if self.history_regime == "train_prior":
            return history["role"] == "train"
        if self.history_regime == "role_prior":
            return history["role"] == target["role"]
        return True

    def _build_history_index(self) -> dict[str, list[str]]:
        by_target: dict[str, list[str]] = {}
        target_ids = sorted({row["session_id"] for row in self.rows})
        sessions = list(self._session_metadata.values())
        for target_id in target_ids:
            target = self._session_metadata[target_id]
            eligible = [
                history
                for history in sessions
                if self._history_is_allowed(history, target)
            ]
            eligible.sort(key=lambda item: (item["end"], item["session_id"]))
            if self.history_session_budget > 0:
                eligible = eligible[-self.history_session_budget :]
            by_target[target_id] = [item["session_id"] for item in eligible]
        return by_target

    @property
    def history_summary(self) -> dict[str, Any]:
        counts = [len(values) for values in self._history_sessions_by_target.values()]
        return {
            "regime": self.history_regime,
            "session_budget": self.history_session_budget,
            "seconds_per_session": self.history_seconds_per_session,
            "tensor_capacity_sessions": self.history_capacity,
            "target_sessions": len(counts),
            "target_sessions_with_history": sum(count > 0 for count in counts),
            "minimum_history_sessions": min(counts, default=0),
            "maximum_history_sessions": max(counts, default=0),
        }

    def _load_session(self, session_id: str) -> dict[str, np.ndarray]:
        if session_id not in self._sessions:
            path = self.fold_cache_dir / "sessions" / f"{session_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                self._sessions[session_id] = {
                    key: archive[key].copy() for key in archive.files
                }
        return self._sessions[session_id]

    def _context_features(
        self,
        session: dict[str, np.ndarray],
        origin: int,
        protocol: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        start = origin - self.context_seconds + 1
        stop = origin + 1
        hr = session["hr_bpm"][start:stop].astype(float)
        hr_valid = session["hr_range_valid"][start:stop].astype(bool)
        hr, observed = _interpolate_short_context_gaps(
            hr,
            hr_valid,
            self.maximum_interpolated_gap_seconds,
        )
        hr_stats = self.normalization["hr_bpm"]
        hr_fill = float(hr_stats["median"])
        hr[~np.isfinite(hr)] = hr_fill
        hr_standardized = (hr - float(hr_stats["mean"])) / float(hr_stats["std"])

        acc = session["acc_features"][start:stop].astype(float)
        acc_valid = session["acc_valid"][start:stop].astype(bool)
        normalized_acc = np.empty_like(acc, dtype=float)
        for column, feature_name in enumerate(self.acc_feature_names):
            stats = self.normalization["acc_features"][feature_name]
            values = acc[:, column]
            values[~np.isfinite(values)] = float(stats["median"])
            normalized_acc[:, column] = (
                values - float(stats["mean"])
            ) / float(stats["std"])

        elapsed = (
            session["unix_seconds"][start:stop].astype(float)
            - float(session["unix_seconds"][0])
        ) / 3600.0
        aerobic = np.full(self.context_seconds, float(protocol == "AEROBIC"))
        anaerobic = np.full(self.context_seconds, float(protocol == "ANAEROBIC"))
        if self.mask_activity_identity:
            aerobic.fill(0.0)
            anaerobic.fill(0.0)
        features = np.column_stack(
            [
                hr_standardized,
                normalized_acc,
                observed.astype(float),
                acc_valid.astype(float),
                elapsed,
                aerobic,
                anaerobic,
            ]
        ).astype(np.float32)
        current_candidates = np.flatnonzero(observed & np.isfinite(hr))
        if not len(current_candidates):
            raise ValueError("Context contains no valid current HR")
        current_hr = float(hr[current_candidates[-1]])
        context_valid = observed & acc_valid
        return features, context_valid, current_hr

    def _sample_history_features(
        self,
        session: dict[str, np.ndarray],
        protocol: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        sample_count = self.history_seconds_per_session
        indices = np.linspace(
            0,
            len(session["hr_bpm"]) - 1,
            sample_count,
            dtype=np.int64,
        )
        hr = session["hr_bpm"][indices].astype(float)
        observed = session["hr_range_valid"][indices].astype(bool)
        hr_stats = self.normalization["hr_bpm"]
        hr[~np.isfinite(hr)] = float(hr_stats["median"])
        hr_standardized = (hr - float(hr_stats["mean"])) / float(hr_stats["std"])

        acc = session["acc_features"][indices].astype(float)
        acc_valid = session["acc_valid"][indices].astype(bool)
        normalized_acc = np.empty_like(acc, dtype=float)
        for column, feature_name in enumerate(self.acc_feature_names):
            stats = self.normalization["acc_features"][feature_name]
            values = acc[:, column]
            values[~np.isfinite(values)] = float(stats["median"])
            normalized_acc[:, column] = (
                values - float(stats["mean"])
            ) / float(stats["std"])

        elapsed = (
            session["unix_seconds"][indices].astype(float)
            - float(session["unix_seconds"][0])
        ) / 3600.0
        aerobic = np.full(sample_count, float(protocol == "AEROBIC"))
        anaerobic = np.full(sample_count, float(protocol == "ANAEROBIC"))
        if self.mask_activity_identity:
            aerobic.fill(0.0)
            anaerobic.fill(0.0)
        features = np.column_stack(
            [
                hr_standardized,
                normalized_acc,
                observed.astype(float),
                acc_valid.astype(float),
                elapsed,
                aerobic,
                anaerobic,
            ]
        ).astype(np.float32)
        valid = observed & acc_valid
        full_valid_seconds = np.sum(
            session["hr_range_valid"].astype(bool)
            & session["acc_valid"].astype(bool)
        )
        return features, valid, float(full_valid_seconds) / 60.0

    def _history_features(
        self,
        target_session_id: str,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        total_steps = self.history_capacity * self.history_seconds_per_session
        history = np.zeros((total_steps, self.input_dim), dtype=np.float32)
        valid = np.zeros(total_steps, dtype=bool)
        history_ids = self._history_sessions_by_target.get(target_session_id, [])
        if not history_ids:
            return history, valid, 0.0, 0
        segments = []
        masks = []
        history_minutes = 0.0
        for session_id in history_ids[-self.history_capacity :]:
            metadata = self._session_metadata[session_id]
            segment, segment_valid, minutes = self._sample_history_features(
                self._load_session(session_id),
                metadata["protocol"],
            )
            segments.append(segment)
            masks.append(segment_valid)
            history_minutes += minutes
        concatenated = np.concatenate(segments, axis=0)
        concatenated_mask = np.concatenate(masks, axis=0)
        start = total_steps - len(concatenated)
        history[start:] = concatenated
        valid[start:] = concatenated_mask
        return history, valid, history_minutes, len(segments)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        session = self._load_session(row["session_id"])
        origin = int(row["origin_index"])
        context, context_valid, current_hr = self._context_features(
            session, origin, row["protocol"]
        )
        target_start = origin + 1
        target_stop = origin + self.horizon_seconds + 1
        target = session["hr_bpm"][target_start:target_stop].astype(np.float32)
        target_valid = session["hr_offline_qc_valid"][target_start:target_stop].astype(bool)
        target = np.where(target_valid, target, 0.0).astype(np.float32)
        event_types = [
            token for token in row["event_types"].split(";") if token
        ]
        event_index = EVENT_INDEX.get(event_types[0], 0) if event_types else 0
        if not self.schedule_aware:
            event_index = 0
        future_event = np.full(self.horizon_seconds, event_index, dtype=np.int64)
        activity = 0 if self.mask_activity_identity else ACTIVITY_INDEX[row["protocol"]]
        history, history_valid, history_minutes, history_count = self._history_features(
            row["session_id"]
        )
        return {
            "context": torch.from_numpy(context),
            "context_valid_mask": torch.from_numpy(context_valid),
            "current_hr": torch.tensor(current_hr, dtype=torch.float32),
            "activity_index": torch.tensor(activity, dtype=torch.long),
            "future_event_index": torch.from_numpy(future_event),
            "history": torch.from_numpy(history),
            "history_valid_mask": torch.from_numpy(history_valid),
            "history_minutes": torch.tensor(history_minutes, dtype=torch.float32),
            "history_session_count": torch.tensor(history_count, dtype=torch.long),
            "target": torch.from_numpy(target),
            "target_valid_mask": torch.from_numpy(target_valid),
            "participant_id": row["participant_id"],
            "session_id": row["session_id"],
            "origin_id": row["origin_id"],
            "protocol": row["protocol"],
            "event_types": row["event_types"],
            "sex": row["sex"],
            "protocol_version": row["protocol_version"],
        }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }
