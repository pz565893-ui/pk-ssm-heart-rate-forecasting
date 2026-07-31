#!/usr/bin/env python3
"""Feasibility audit for the PhysioNet structured wearable exercise dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOLS = ("AEROBIC", "ANAEROBIC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-seconds", type=int, default=300)
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--stride-seconds", type=int, default=120)
    parser.add_argument("--minimum-window-coverage", type=float, default=0.90)
    return parser.parse_args()


def base_user_id(session_id: str) -> str:
    return re.sub(r"_[ab]$", "", session_id)


def protocol_version(session_id: str) -> str:
    return "V1" if session_id.startswith("S") else "V2"


def read_empatica_series(path: Path) -> tuple[pd.Timestamp, float, np.ndarray]:
    rows = pd.read_csv(path, header=None)
    if len(rows) < 3:
        raise ValueError(f"Insufficient rows in {path}")
    start = pd.to_datetime(str(rows.iloc[0, 0]), errors="raise")
    sample_rate = float(rows.iloc[1, 0])
    values = pd.to_numeric(rows.iloc[2:, 0], errors="coerce").to_numpy(dtype=np.float64)
    return start, sample_rate, values


def read_accelerometer(
    path: Path,
) -> tuple[dict[str, float | int | str | bool], np.ndarray]:
    rows = pd.read_csv(path, header=None)
    if len(rows) < 3:
        raise ValueError(f"Insufficient rows in {path}")
    start = pd.to_datetime(str(rows.iloc[0, 0]), errors="raise")
    sample_rate = float(rows.iloc[1, 0])
    values = rows.iloc[2:, :3].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    row_valid = finite.all(axis=1)
    duration = len(values) / sample_rate if sample_rate > 0 else math.nan
    saturated = np.abs(values) >= 127.0
    if sample_rate <= 0:
        raise ValueError(f"Non-positive ACC sampling rate {sample_rate} in {path}")
    if len(values):
        bin_index = np.floor(np.arange(len(values), dtype=np.float64) / sample_rate).astype(
            np.int64
        )
        number_of_bins = int(bin_index[-1]) + 1
        total_by_bin = np.bincount(bin_index, minlength=number_of_bins)
        valid_by_bin = np.bincount(
            bin_index, weights=row_valid.astype(np.int32), minlength=number_of_bins
        )
        required_samples = int(math.ceil(0.90 * sample_rate))
        valid_second_bins = (total_by_bin >= required_samples) & (
            valid_by_bin >= required_samples
        )
    else:
        valid_second_bins = np.zeros(0, dtype=bool)
    summary = {
        "acc_start": start.isoformat(),
        "acc_start_epoch": float(start.timestamp()),
        "acc_sample_rate_hz": sample_rate,
        "acc_samples": int(len(values)),
        "acc_duration_seconds": float(duration),
        "acc_valid_row_percent": float(100.0 * row_valid.mean()) if len(values) else math.nan,
        "acc_saturated_value_percent": float(100.0 * saturated.mean()) if values.size else math.nan,
        "acc_second_bins": int(len(valid_second_bins)),
        "acc_valid_second_bin_percent": (
            float(100.0 * valid_second_bins.mean()) if len(valid_second_bins) else math.nan
        ),
    }
    return summary, valid_second_bins


def read_ibi(path: Path) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    if path.stat().st_size == 0:
        return None, pd.DataFrame(columns=["offset", "ibi"])
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1", newline="") as handle:
            rows = list(csv.reader(handle))
    if not rows:
        return None, pd.DataFrame(columns=["offset", "ibi"])
    start = pd.to_datetime(rows[0][0], errors="coerce")
    numeric_rows = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        try:
            numeric_rows.append((float(row[0]), float(row[1])))
        except ValueError:
            continue
    return start if not pd.isna(start) else None, pd.DataFrame(numeric_rows, columns=["offset", "ibi"])


def read_tags(path: Path) -> list[pd.Timestamp]:
    if path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path, header=None)
    tags = pd.to_datetime(frame.iloc[:, 0], errors="coerce").dropna().sort_values()
    return list(tags)


def robust_hr(values: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    series = pd.Series(values)
    plausible = series.between(30.0, 220.0)
    local_median = series.where(plausible).rolling(11, center=True, min_periods=3).median()
    local_consistent = (series - local_median).abs().le(25.0) | local_median.isna()
    valid = plausible & local_consistent
    qc = series.where(valid).to_numpy(dtype=np.float64)
    differences = series.diff().abs()
    return qc, {
        "hr_raw_valid_percent": float(100.0 * plausible.mean()) if len(series) else math.nan,
        "hr_qc_valid_percent": float(100.0 * valid.mean()) if len(series) else math.nan,
        "hr_local_outlier_seconds": int((plausible & ~local_consistent).sum()),
        "hr_jump_gt20_seconds": int((differences > 20.0).sum()),
    }


def ibi_consistency(
    hr_start: pd.Timestamp,
    hr_values: np.ndarray,
    ibi_start: pd.Timestamp | None,
    ibi: pd.DataFrame,
) -> dict[str, float | int]:
    if ibi_start is None or ibi.empty:
        return {
            "ibi_rows": 0,
            "ibi_valid_percent": math.nan,
            "ibi_matched_seconds": 0,
            "ibi_hr_median_absolute_difference_bpm": math.nan,
            "ibi_hr_within_10_bpm_percent": math.nan,
        }
    valid = ibi["ibi"].between(0.25, 2.0)
    subset = ibi[valid].copy()
    subset["epoch_second"] = np.rint(
        float(ibi_start.timestamp()) + subset["offset"].to_numpy(dtype=np.float64)
    ).astype(np.int64)
    subset["ibi_hr"] = 60.0 / subset["ibi"]
    by_second = subset.groupby("epoch_second", as_index=False)["ibi_hr"].median()
    hr_seconds = np.rint(float(hr_start.timestamp()) + np.arange(len(hr_values))).astype(np.int64)
    wearable = pd.DataFrame({"epoch_second": hr_seconds, "hr": hr_values})
    wearable = wearable[wearable["hr"].between(30.0, 220.0)]
    matched = wearable.merge(by_second, on="epoch_second", how="inner")
    difference = (matched["hr"] - matched["ibi_hr"]).abs()
    return {
        "ibi_rows": int(len(ibi)),
        "ibi_valid_percent": float(100.0 * valid.mean()) if len(ibi) else math.nan,
        "ibi_matched_seconds": int(len(matched)),
        "ibi_hr_median_absolute_difference_bpm": (
            float(difference.median()) if len(difference) else math.nan
        ),
        "ibi_hr_within_10_bpm_percent": (
            float(100.0 * (difference <= 10.0).mean()) if len(difference) else math.nan
        ),
    }


def eligible_origins(
    qc_hr: np.ndarray,
    context: int,
    horizon: int,
    stride: int,
    minimum_coverage: float,
) -> tuple[int, int]:
    window_length = context + horizon + 1
    number = len(qc_hr) - window_length + 1
    if number <= 0:
        return 0, 0
    valid = np.isfinite(qc_hr).astype(np.int32)
    prefix = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    starts = np.arange(number, dtype=np.int64)
    counts = prefix[starts + window_length] - prefix[starts]
    eligible = counts >= math.ceil(minimum_coverage * window_length)
    origins = starts + context
    grid = (origins - context) % stride == 0
    return int(eligible.sum()), int((eligible & grid).sum())


def event_metrics(
    protocol: str,
    session_id: str,
    event_type: str,
    tag_index: int,
    event_time: pd.Timestamp,
    hr_start: pd.Timestamp,
    qc_hr: np.ndarray,
    context: int,
    horizon: int,
    minimum_coverage: float,
) -> dict[str, object]:
    event_offset = int(round((event_time - hr_start).total_seconds()))
    start = event_offset - context
    stop = event_offset + horizon
    in_bounds = start >= 0 and stop < len(qc_hr)
    if in_bounds:
        window = qc_hr[start : stop + 1]
        coverage = float(np.isfinite(window).mean())
        baseline_values = qc_hr[event_offset - 60 : event_offset]
        at_30_values = qc_hr[event_offset + 25 : event_offset + 36]
        post_values = qc_hr[event_offset + 90 : event_offset + 121]
        baseline = float(np.nanmedian(baseline_values)) if np.isfinite(baseline_values).any() else math.nan
        at_30 = float(np.nanmedian(at_30_values)) if np.isfinite(at_30_values).any() else math.nan
        post = float(np.nanmedian(post_values)) if np.isfinite(post_values).any() else math.nan
    else:
        coverage = 0.0
        baseline = math.nan
        at_30 = math.nan
        post = math.nan
    eligible = bool(
        in_bounds
        and coverage >= minimum_coverage
        and math.isfinite(baseline)
        and math.isfinite(post)
    )
    change_30 = at_30 - baseline if math.isfinite(at_30) and math.isfinite(baseline) else math.nan
    return {
        "protocol": protocol,
        "session_id": session_id,
        "user_id": base_user_id(session_id),
        "version": protocol_version(session_id),
        "tag_index": tag_index,
        "event_type": event_type,
        "event_time": event_time.isoformat(),
        "event_offset_seconds": event_offset,
        "coverage": coverage,
        "eligible": eligible,
        "baseline_hr": baseline,
        "hr_around_30s": at_30,
        "hr_90_120s": post,
        "change_around_30s_bpm": change_30,
        "change_90_120s_bpm": post - baseline if eligible else math.nan,
        "rapid_change_ge_10_bpm": bool(math.isfinite(change_30) and abs(change_30) >= 10.0),
    }


def classify_events(protocol: str, tags: list[pd.Timestamp]) -> list[tuple[int, str, pd.Timestamp]]:
    events: list[tuple[int, str, pd.Timestamp]] = []
    if protocol == "AEROBIC":
        for index, tag in enumerate(tags):
            events.append((index, "aerobic_stage_boundary", tag))
        return events

    for index in range(len(tags) - 1):
        duration = (tags[index + 1] - tags[index]).total_seconds()
        if 20.0 <= duration <= 75.0:
            events.append((index, "sprint_onset", tags[index]))
            events.append((index + 1, "sprint_offset", tags[index + 1]))
        elif 150.0 <= duration <= 300.0:
            events.append((index, "recovery_interval_start", tags[index]))
    deduplicated = {(index, kind, tag): None for index, kind, tag in events}
    return list(deduplicated)


def audit_session(
    protocol: str,
    session_dir: Path,
    context: int,
    horizon: int,
    stride: int,
    minimum_coverage: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    session_id = session_dir.name
    hr_start, hr_rate, hr_values = read_empatica_series(session_dir / "HR.csv")
    if not math.isclose(hr_rate, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Unexpected HR sampling rate {hr_rate} in {session_dir}")
    qc_hr, hr_summary = robust_hr(hr_values)
    ibi_start, ibi = read_ibi(session_dir / "IBI.csv")
    ibi_summary = ibi_consistency(hr_start, hr_values, ibi_start, ibi)
    acc_summary, acc_valid_second_bins = read_accelerometer(session_dir / "ACC.csv")
    tags = read_tags(session_dir / "tags.csv")
    hr_start_epoch = int(round(float(hr_start.timestamp())))
    acc_start_epoch = int(round(float(acc_summary["acc_start_epoch"])))
    hr_epoch_seconds = hr_start_epoch + np.arange(len(qc_hr), dtype=np.int64)
    acc_bin_positions = hr_epoch_seconds - acc_start_epoch
    within_acc = (acc_bin_positions >= 0) & (
        acc_bin_positions < len(acc_valid_second_bins)
    )
    joint_support = np.zeros(len(qc_hr), dtype=bool)
    joint_support[within_acc] = acc_valid_second_bins[acc_bin_positions[within_acc]]
    joint_qc_hr = np.where(joint_support, qc_hr, np.nan)
    dense, stride_origins = eligible_origins(
        joint_qc_hr, context, horizon, stride, minimum_coverage
    )

    events = [
        event_metrics(
            protocol,
            session_id,
            event_type,
            index,
            tag,
            hr_start,
            joint_qc_hr,
            context,
            horizon,
            minimum_coverage,
        )
        for index, event_type, tag in classify_events(protocol, tags)
    ]
    acc_start_epoch_float = float(acc_summary["acc_start_epoch"])
    acc_rate = float(acc_summary["acc_sample_rate_hz"])
    acc_samples = int(acc_summary["acc_samples"])
    acc_end_epoch = (
        acc_start_epoch_float + (acc_samples - 1) / acc_rate
        if acc_samples
        else acc_start_epoch_float
    )
    hr_start_epoch_float = float(hr_start.timestamp())
    hr_end_epoch = hr_start_epoch_float + max(0, len(hr_values) - 1)
    supported_indices = np.flatnonzero(joint_support)
    if len(supported_indices):
        common_start = hr_start + pd.Timedelta(seconds=int(supported_indices[0]))
        common_end = hr_start + pd.Timedelta(seconds=int(supported_indices[-1]))
    else:
        common_start = pd.NaT
        common_end = pd.NaT

    summary: dict[str, object] = {
        "protocol": protocol,
        "session_id": session_id,
        "user_id": base_user_id(session_id),
        "version": protocol_version(session_id),
        "is_split_fragment": bool(re.search(r"_[ab]$", session_id)),
        "hr_start": hr_start.isoformat(),
        "hr_seconds": int(len(hr_values)),
        "hr_duration_minutes": float(len(hr_values) / 60.0),
        "tag_count": int(len(tags)),
        "tags_within_hr_record": int(
            sum(hr_start <= tag < hr_start + pd.Timedelta(seconds=len(hr_values)) for tag in tags)
        ),
        "eligible_dense_origins": dense,
        f"eligible_stride_{stride}s_origins": stride_origins,
        "joint_hr_acc_supported_seconds": int(joint_support.sum()),
        "joint_hr_acc_support_percent": (
            float(100.0 * joint_support.mean()) if len(joint_support) else math.nan
        ),
        "joint_qc_hr_valid_percent": (
            float(100.0 * np.isfinite(joint_qc_hr).mean()) if len(joint_qc_hr) else math.nan
        ),
        "common_support_start": common_start.isoformat() if not pd.isna(common_start) else "",
        "common_support_end": common_end.isoformat() if not pd.isna(common_end) else "",
        "acc_covers_hr": bool(
            acc_start_epoch_float <= hr_start_epoch_float and acc_end_epoch >= hr_end_epoch
        ),
    }
    summary.update(hr_summary)
    summary.update(ibi_summary)
    summary.update(acc_summary)
    return summary, events


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."

    def render(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.2f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def make_report(
    sessions: pd.DataFrame,
    events: pd.DataFrame,
    user_protocol: pd.DataFrame,
    summary: dict[str, object],
) -> str:
    protocol_summary = (
        sessions.groupby("protocol", as_index=False)
        .agg(
            users=("user_id", "nunique"),
            sessions=("session_id", "nunique"),
            total_hr_hours=("hr_seconds", lambda x: x.sum() / 3600.0),
            median_session_minutes=("hr_duration_minutes", "median"),
            eligible_origins=("eligible_stride_120s_origins", "sum"),
            median_qc_hr_percent=("hr_qc_valid_percent", "median"),
            median_joint_support_percent=("joint_hr_acc_support_percent", "median"),
            median_joint_qc_hr_percent=("joint_qc_hr_valid_percent", "median"),
            median_ibi_hr_mad=("ibi_hr_median_absolute_difference_bpm", "median"),
        )
        .sort_values("protocol")
    )
    eligible_events = events[events["eligible"] == True]  # noqa: E712
    event_summary = (
        eligible_events.groupby(["protocol", "event_type"], as_index=False)
        .agg(
            users=("user_id", "nunique"),
            events=("user_id", "size"),
            rapid_events=("rapid_change_ge_10_bpm", "sum"),
            median_change_30s=("change_around_30s_bpm", "median"),
            median_change_120s=("change_90_120s_bpm", "median"),
        )
        .sort_values(["protocol", "event_type"])
    )
    decision = summary["decision"]
    return "\n".join(
        [
            "# Structured wearable exercise dataset audit",
            "",
            "## Frozen audit contract",
            "",
            "- Context: `300 s`",
            "- Forecast horizon: `120 s`",
            "- Primary origin stride: `120 s`",
            "- Required full-window valid-HR coverage: `90%`",
            "- Plausible HR range: `30-220 bpm`",
            "- Sprint interval definition: tag-to-tag duration `20-75 s`",
            "- Recovery interval definition: tag-to-tag duration `150-300 s`",
            "",
            "## Protocol support",
            "",
            frame_to_markdown(protocol_summary),
            "",
            "## Cross-protocol user support",
            "",
            f"- Users with any aerobic data: **{summary['aerobic_users']}**",
            f"- Users with any anaerobic data: **{summary['anaerobic_users']}**",
            f"- Users with eligible windows in both protocols: **{summary['cross_protocol_eligible_users']}**",
            f"- Total eligible 120 s-stride origins: **{summary['total_eligible_stride_origins']}**",
            "",
            "## Eligible tagged events",
            "",
            frame_to_markdown(event_summary),
            "",
            "## Pre-specified gates",
            "",
            f"- Main aerobic window gate: **{'PASS' if decision['aerobic_window_gate'] else 'FAIL'}**",
            f"- Main anaerobic window gate: **{'PASS' if decision['anaerobic_window_gate'] else 'FAIL'}**",
            f"- Cross-activity personalization gate: **{'PASS' if decision['cross_activity_gate'] else 'FAIL'}**",
            f"- Anaerobic transition-event gate: **{'PASS' if decision['transition_event_gate'] else 'FAIL'}**",
            "",
            "## Decision",
            "",
            str(decision["recommendation"]),
            "",
            "## Constraints",
            "",
            "- HR and IBI are derived from wrist PPG rather than reference ECG.",
            "- Protocol versions V1 and V2 differ and must be modeled or stratified explicitly.",
            "- Split files are fragments of one user-session and must never cross data partitions.",
            "- Session/window counts are not independent sample sizes; users are the highest inferential unit.",
            "- Aerobic and anaerobic sessions differ in both activity type and acquisition protocol, so activity-shift claims must retain this compound interpretation.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    session_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        protocol_dir = args.data_dir / protocol
        session_dirs = sorted(path for path in protocol_dir.iterdir() if path.is_dir())
        for session_dir in session_dirs:
            session, events = audit_session(
                protocol,
                session_dir,
                args.context_seconds,
                args.horizon_seconds,
                args.stride_seconds,
                args.minimum_window_coverage,
            )
            session_rows.append(session)
            event_rows.extend(events)

    sessions = pd.DataFrame(session_rows).sort_values(["protocol", "user_id", "session_id"])
    events = pd.DataFrame(event_rows).sort_values(
        ["protocol", "user_id", "event_time", "event_type"]
    )
    stride_column = f"eligible_stride_{args.stride_seconds}s_origins"
    user_protocol = (
        sessions.groupby(["user_id", "protocol"], as_index=False)
        .agg(
            sessions=("session_id", "nunique"),
            eligible_origins=(stride_column, "sum"),
            hr_seconds=("hr_seconds", "sum"),
            median_qc_hr_percent=("hr_qc_valid_percent", "median"),
        )
    )
    pivot = user_protocol.pivot(index="user_id", columns="protocol", values="eligible_origins").fillna(0)
    for protocol in PROTOCOLS:
        if protocol not in pivot.columns:
            pivot[protocol] = 0
    cross_eligible = int(((pivot["AEROBIC"] >= 5) & (pivot["ANAEROBIC"] >= 5)).sum())
    protocol_user_counts = {
        protocol: int(user_protocol[user_protocol["protocol"] == protocol]["user_id"].nunique())
        for protocol in PROTOCOLS
    }
    protocol_eligible_users = {
        protocol: int(
            (
                user_protocol[user_protocol["protocol"] == protocol]["eligible_origins"]
                >= 5
            ).sum()
        )
        for protocol in PROTOCOLS
    }
    eligible_events = events[events["eligible"] == True]  # noqa: E712
    sprint_events = eligible_events[
        eligible_events["event_type"].isin(["sprint_onset", "sprint_offset"])
    ]
    sprint_users_four = int((sprint_events.groupby("user_id").size() >= 4).sum())

    aerobic_window_gate = protocol_eligible_users["AEROBIC"] >= 25
    anaerobic_window_gate = protocol_eligible_users["ANAEROBIC"] >= 25
    cross_activity_gate = cross_eligible >= 24
    transition_event_gate = sprint_users_four >= 20
    all_pass = all(
        [
            aerobic_window_gate,
            anaerobic_window_gate,
            cross_activity_gate,
            transition_event_gate,
        ]
    )
    if all_pass:
        recommendation = (
            "The dataset passes all pre-specified feasibility gates. It can replace MMASH as the "
            "primary source for cross-user, cross-activity, and transition-conditioned forecasting. "
            "Freeze a user/session split manifest before fitting any model."
        )
    elif aerobic_window_gate and anaerobic_window_gate:
        recommendation = (
            "The dataset supports protocol-level forecasting but not every cross-activity or "
            "transition claim. Retain only analyses whose gate passed and add an independent source "
            "before making broader physiology-guided claims."
        )
    else:
        recommendation = (
            "The dataset does not support the primary 300 s to 120 s protocol. Stop this route and "
            "select a different primary source."
        )

    summary: dict[str, object] = {
        "dataset": "Wearable Device Dataset from Induced Stress and Structured Exercise Sessions v1.0.1",
        "sessions": int(len(sessions)),
        "aerobic_users": protocol_user_counts["AEROBIC"],
        "anaerobic_users": protocol_user_counts["ANAEROBIC"],
        "aerobic_users_with_5_origins": protocol_eligible_users["AEROBIC"],
        "anaerobic_users_with_5_origins": protocol_eligible_users["ANAEROBIC"],
        "cross_protocol_eligible_users": cross_eligible,
        "total_eligible_stride_origins": int(sessions[stride_column].sum()),
        "eligible_tagged_events": int(len(eligible_events)),
        "eligible_sprint_events": int(len(sprint_events)),
        "users_with_4_sprint_events": sprint_users_four,
        "median_hr_qc_valid_percent": float(sessions["hr_qc_valid_percent"].median()),
        "median_joint_hr_acc_support_percent": float(
            sessions["joint_hr_acc_support_percent"].median()
        ),
        "median_joint_qc_hr_valid_percent": float(
            sessions["joint_qc_hr_valid_percent"].median()
        ),
        "median_ibi_hr_mad_bpm": float(sessions["ibi_hr_median_absolute_difference_bpm"].median()),
        "sessions_with_acc_covering_hr": int(sessions["acc_covers_hr"].sum()),
        "decision": {
            "aerobic_window_gate": bool(aerobic_window_gate),
            "anaerobic_window_gate": bool(anaerobic_window_gate),
            "cross_activity_gate": bool(cross_activity_gate),
            "transition_event_gate": bool(transition_event_gate),
            "all_primary_gates_pass": bool(all_pass),
            "recommendation": recommendation,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(args.output_dir / "session_summary.csv", index=False)
    events.to_csv(args.output_dir / "tagged_events.csv", index=False)
    user_protocol.to_csv(args.output_dir / "user_protocol_summary.csv", index=False)
    (args.output_dir / "wearable_exercise_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output_dir / "WEARABLE_EXERCISE_DATA_AUDIT.md").write_text(
        make_report(sessions, events, user_protocol, summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"REPORT={args.output_dir / 'WEARABLE_EXERCISE_DATA_AUDIT.md'}")


if __name__ == "__main__":
    main()
