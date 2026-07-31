from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "activity_shift_locked_summary_v2_final"
    / "activity_shift_summary.json"
)
FIGURE_DIR = ROOT / "figures_v4"
SOURCE_DIR = FIGURE_DIR / "source_data"
OUTPUT_BASE = FIGURE_DIR / "Figure_6_activity_shift_evaluation"

PK_COLOR = "#0F4D92"
TCN_COLOR = "#D88032"
NEUTRAL = "#5F6368"
LIGHT_BLUE = "#E8F1F8"
LIGHT_ORANGE = "#F8EEE3"
METHOD_COLORS = {"pk_ssm": PK_COLOR, "tcn": TCN_COLOR}
METHOD_LABELS = {"pk_ssm": "PK-SSM", "tcn": "TCN"}
PROTOCOL_LABELS = {
    "AEROBIC": "Held-out aerobic",
    "ANAEROBIC": "Held-out anaerobic",
}
BOUNDARY_LABELS = {
    "source_user": "Source\nuser",
    "seen_user_activity": "Seen\nactivity",
    "joint_user_activity": "Joint\nshift",
}


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


def load_locked_summary() -> tuple[dict, str]:
    raw = SUMMARY_PATH.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    expected = {
        "artifact_status": "locked_wearable_v4_activity_shift_summary",
        "activity_identity": "masked",
        "history_regime": "none",
        "primary_boundary": "joint_user_activity",
        "primary_origin_policy": "tagged_events",
        "secondary_origin_policy": "evaluation_stride",
        "protocol_amendment": "027",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"Locked summary mismatch for {key}: {data.get(key)!r}")
    if data.get("bootstrap_replicates") != 10_000:
        raise ValueError("Expected 10,000 bootstrap replicates")
    if len(data.get("seeds", [])) != 5:
        raise ValueError("Expected five locked random seeds")
    return data, hashlib.sha256(raw).hexdigest()


def flatten_model_inference(data: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for policy, by_protocol in data["paired_model_inference"].items():
        for heldout, by_boundary in by_protocol.items():
            for boundary, by_metric in by_boundary.items():
                for metric, result in by_metric.items():
                    rows.append(
                        {
                            "origin_policy": policy,
                            "heldout_protocol": heldout,
                            "boundary": boundary,
                            "metric": metric,
                            "pk_ssm_minus_tcn_bpm": result[
                                "mean_pk_ssm_minus_tcn"
                            ],
                            "ci_low_bpm": result["bootstrap_95_ci"][0],
                            "ci_high_bpm": result["bootstrap_95_ci"][1],
                            "wilcoxon_p": result["wilcoxon_p"],
                            "participants": result["participants"],
                            "pk_ssm_better_fraction": result[
                                "pk_ssm_better_fraction"
                            ],
                            "tcn_better_fraction": result["tcn_better_fraction"],
                            "ties_fraction": result["ties_fraction"],
                        }
                    )
    return pd.DataFrame(rows)


def flatten_deployment_contrasts(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict] = []
    for policy, by_protocol in data["paired_shift_penalty"].items():
        for heldout, by_model in by_protocol.items():
            for model, by_metric in by_model.items():
                for metric, result in by_metric.items():
                    model_rows.append(
                        {
                            "origin_policy": policy,
                            "heldout_protocol": heldout,
                            "model": model,
                            "metric": metric,
                            "joint_minus_source_bpm": result[
                                "mean_difference_bpm"
                            ],
                            "ci_low_bpm": result["bootstrap_95_ci"][0],
                            "ci_high_bpm": result["bootstrap_95_ci"][1],
                            "wilcoxon_p": result["wilcoxon_p"],
                            "participants": result["participants"],
                            "positive_fraction": result["positive_fraction"],
                            "negative_fraction": result["negative_fraction"],
                            "ties_fraction": result["ties_fraction"],
                        }
                    )

    contrast_rows: list[dict] = []
    for policy, by_protocol in data["paired_shift_penalty_model_contrast"].items():
        for heldout, by_metric in by_protocol.items():
            for metric, result in by_metric.items():
                contrast_rows.append(
                    {
                        "origin_policy": policy,
                        "heldout_protocol": heldout,
                        "metric": metric,
                        "pk_ssm_minus_tcn_deployment_contrast_bpm": result[
                            "mean_difference_bpm"
                        ],
                        "ci_low_bpm": result["bootstrap_95_ci"][0],
                        "ci_high_bpm": result["bootstrap_95_ci"][1],
                        "wilcoxon_p": result["wilcoxon_p"],
                        "participants": result["participants"],
                    }
                )
    return pd.DataFrame(model_rows), pd.DataFrame(contrast_rows)


def aggregate_interval_transport(data: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for report in data["fold_reports"]:
        calibration = report["calibration"]
        row = {
            "origin_policy": report["origin_policy"],
            "heldout_protocol": report["heldout_protocol"],
            "boundary": report["boundary"],
            "model": report["model"],
            "outer_fold": report["outer_fold"],
            "participants": report["point_metrics"].get(
                "participants", np.nan
            ),
            "raw_point_coverage": calibration["raw_student_t"][
                "participant_macro_point_coverage"
            ],
            "raw_high_hr_coverage": calibration["raw_student_t"][
                "participant_macro_high_hr_coverage"
            ],
            "raw_mean_width_bpm": calibration["raw_student_t"][
                "mean_width_bpm"
            ],
            "simultaneous_120s_coverage": calibration["simultaneous_120s"][
                "participant_block"
            ]["participant_macro_curve_coverage"],
            "simultaneous_120s_mean_width_bpm": calibration[
                "simultaneous_120s"
            ]["participant_block"]["mean_pointwise_width_bpm"],
        }
        for horizon in (30, 60, 120):
            point = calibration["pointwise"][str(horizon)]
            row[f"pointwise_{horizon}s_coverage"] = point[
                "participant_block_participant_macro_coverage"
            ]
            row[f"pointwise_{horizon}s_mean_width_bpm"] = point[
                "participant_block_mean_width_bpm"
            ]
        rows.append(row)

    fold_df = pd.DataFrame(rows)
    group_cols = ["origin_policy", "heldout_protocol", "boundary", "model"]
    numeric_cols = [
        col
        for col in fold_df.columns
        if col not in group_cols + ["outer_fold", "participants"]
    ]
    aggregate = (
        fold_df.groupby(group_cols, as_index=False)[numeric_cols]
        .mean()
        .merge(
            fold_df.groupby(group_cols, as_index=False)["outer_fold"]
            .nunique()
            .rename(columns={"outer_fold": "outer_folds"}),
            on=group_cols,
            how="left",
        )
    )
    return fold_df, aggregate


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def forest_row(
    ax: plt.Axes,
    y: float,
    estimate: float,
    low: float,
    high: float,
    color: str,
    marker: str,
    filled: bool,
) -> None:
    ax.plot([low, high], [y, y], color=color, lw=1.35, solid_capstyle="round")
    ax.plot(
        estimate,
        y,
        marker=marker,
        ms=5.0,
        mfc=color if filled else "white",
        mec=color,
        mew=1.0,
        linestyle="none",
        zorder=4,
    )


def panel_absolute_mae(ax: plt.Axes, aggregate: pd.DataFrame) -> None:
    primary = aggregate[aggregate["origin_policy"] == "tagged_events"].copy()
    boundaries = ["source_user", "seen_user_activity", "joint_user_activity"]
    x_by_protocol = {
        "AEROBIC": np.array([0.0, 1.0, 2.0]),
        "ANAEROBIC": np.array([4.0, 5.0, 6.0]),
    }
    offsets = {"pk_ssm": -0.08, "tcn": 0.08}

    ax.axvspan(-0.45, 2.45, color=LIGHT_BLUE, alpha=0.65, zorder=0)
    ax.axvspan(3.55, 6.45, color=LIGHT_ORANGE, alpha=0.65, zorder=0)
    for heldout, xs in x_by_protocol.items():
        for model in ("pk_ssm", "tcn"):
            values = []
            for boundary in boundaries:
                row = primary[
                    (primary["heldout_protocol"] == heldout)
                    & (primary["boundary"] == boundary)
                    & (primary["model"] == model)
                ]
                if len(row) != 1:
                    raise ValueError(
                        f"Expected one aggregate row for {heldout}/{boundary}/{model}"
                    )
                values.append(float(row.iloc[0]["trajectory_mae_bpm"]))
            shifted = xs + offsets[model]
            ax.plot(
                shifted,
                values,
                color=METHOD_COLORS[model],
                marker="o" if model == "pk_ssm" else "s",
                ms=4.4,
                lw=1.15,
                label=METHOD_LABELS[model] if heldout == "AEROBIC" else None,
            )

    labels = [BOUNDARY_LABELS[b] for b in boundaries] * 2
    ax.set_xticks([0, 1, 2, 4, 5, 6])
    ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylabel("Trajectory MAE (bpm)")
    ax.set_ylim(5.5, 16.5)
    ax.grid(axis="y", color="#D7D7D7", lw=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.text(
        1.0,
        16.15,
        "Held-out aerobic\n(train anaerobic)",
        ha="center",
        va="top",
        fontsize=6.3,
        fontweight="bold",
    )
    ax.text(
        5.0,
        16.15,
        "Held-out anaerobic\n(train aerobic)",
        ha="center",
        va="top",
        fontsize=6.3,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.02,
        "Participant-macro; n = 29-31",
        transform=ax.transAxes,
        fontsize=5.5,
        color=NEUTRAL,
    )
    ax.set_title("Absolute trajectory error", loc="left", fontsize=7.3, pad=5)
    add_panel_label(ax, "a")


def panel_model_difference(ax: plt.Axes, inference: pd.DataFrame) -> None:
    subset = inference[
        (inference["boundary"] == "joint_user_activity")
        & (inference["metric"] == "trajectory_mae_bpm")
    ].copy()
    order = [
        ("tagged_events", "AEROBIC"),
        ("tagged_events", "ANAEROBIC"),
        ("evaluation_stride", "AEROBIC"),
        ("evaluation_stride", "ANAEROBIC"),
    ]
    y = np.arange(len(order))[::-1]
    labels = []
    for yi, (policy, heldout) in zip(y, order):
        row = subset[
            (subset["origin_policy"] == policy)
            & (subset["heldout_protocol"] == heldout)
        ]
        if len(row) != 1:
            raise ValueError(f"Missing paired result for {policy}/{heldout}")
        row = row.iloc[0]
        forest_row(
            ax,
            yi,
            row["pk_ssm_minus_tcn_bpm"],
            row["ci_low_bpm"],
            row["ci_high_bpm"],
            PK_COLOR if policy == "tagged_events" else "#6083A5",
            "o" if policy == "tagged_events" else "D",
            policy == "tagged_events",
        )
        prefix = "Tagged" if policy == "tagged_events" else "Stride"
        labels.append(f"{prefix} | {heldout.title()}")

    ax.axvline(0, color=NEUTRAL, lw=0.8, ls="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.1)
    ax.set_xlabel("PK-SSM - TCN trajectory MAE (bpm)")
    ax.set_xlim(-2.7, 1.3)
    ax.grid(axis="x", color="#D7D7D7", lw=0.5, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_title("Paired model difference at joint boundary", loc="left", fontsize=7.3, pad=5)
    add_panel_label(ax, "b")


def panel_deployment_contrast(ax: plt.Axes, deployment: pd.DataFrame) -> None:
    subset = deployment[deployment["metric"] == "trajectory_mae_bpm"].copy()
    order = []
    for policy in ("tagged_events", "evaluation_stride"):
        for heldout in ("AEROBIC", "ANAEROBIC"):
            for model in ("pk_ssm", "tcn"):
                order.append((policy, heldout, model))

    y = np.arange(len(order))[::-1]
    labels = []
    for yi, (policy, heldout, model) in zip(y, order):
        row = subset[
            (subset["origin_policy"] == policy)
            & (subset["heldout_protocol"] == heldout)
            & (subset["model"] == model)
        ]
        if len(row) != 1:
            raise ValueError(
                f"Missing deployment result for {policy}/{heldout}/{model}"
            )
        row = row.iloc[0]
        forest_row(
            ax,
            yi,
            row["joint_minus_source_bpm"],
            row["ci_low_bpm"],
            row["ci_high_bpm"],
            METHOD_COLORS[model],
            "o" if model == "pk_ssm" else "s",
            policy == "tagged_events",
        )
        policy_label = "Tagged" if policy == "tagged_events" else "Stride"
        labels.append(
            f"{policy_label} | {heldout.title()} | {METHOD_LABELS[model]}"
        )

    ax.axhspan(3.5, 7.5, color=LIGHT_BLUE, alpha=0.45, zorder=0)
    ax.axhspan(-0.5, 3.5, color="#F3F3F3", alpha=0.65, zorder=0)
    ax.axvline(0, color=NEUTRAL, lw=0.8, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.7)
    ax.set_xlim(-8.0, 12.0)
    ax.set_xlabel("Joint - source-user trajectory MAE (bpm)")
    ax.grid(axis="x", color="#D7D7D7", lw=0.5, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_title("Activity-shift deployment contrast", loc="left", fontsize=7.3, pad=5)
    add_panel_label(ax, "c")


def panel_interval_transport(ax: plt.Axes, intervals: pd.DataFrame) -> None:
    subset = intervals[
        (intervals["origin_policy"] == "tagged_events")
        & (intervals["boundary"] == "joint_user_activity")
    ].copy()
    if len(subset) != 4:
        raise ValueError("Expected four primary joint-boundary interval rows")

    marker_by_protocol = {"AEROBIC": "o", "ANAEROBIC": "s"}
    offsets = {
        ("AEROBIC", "pk_ssm"): (4, -9),
        ("AEROBIC", "tcn"): (4, 5),
        ("ANAEROBIC", "pk_ssm"): (4, 5),
        ("ANAEROBIC", "tcn"): (4, -10),
    }
    for _, row in subset.iterrows():
        heldout = row["heldout_protocol"]
        model = row["model"]
        x = row["pointwise_120s_mean_width_bpm"]
        y = row["pointwise_120s_coverage"]
        ax.scatter(
            x,
            y,
            s=34,
            marker=marker_by_protocol[heldout],
            facecolor=METHOD_COLORS[model],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        dx, dy = offsets[(heldout, model)]
        short_protocol = "Aerobic" if heldout == "AEROBIC" else "Anaerobic"
        ax.annotate(
            f"{short_protocol}, {METHOD_LABELS[model]}",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.4,
            color=METHOD_COLORS[model],
        )

    ax.axhline(0.95, color=NEUTRAL, lw=0.85, ls="--")
    ax.text(
        0.99,
        0.955,
        "Nominal 0.95",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=NEUTRAL,
    )
    ax.set_xlim(94, 174)
    ax.set_ylim(0.82, 1.005)
    ax.set_xlabel("Mean interval width (bpm)")
    ax.set_ylabel("Participant-macro coverage")
    ax.grid(color="#D7D7D7", lw=0.5, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_title("Source-calibrated 120-s intervals", loc="left", fontsize=7.3, pad=5)
    add_panel_label(ax, "d")


def save_outputs(fig: plt.Figure) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        OUTPUT_BASE.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> int:
    data, summary_sha256 = load_locked_summary()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    aggregate = pd.DataFrame(data["aggregate_model_metrics"])
    inference = flatten_model_inference(data)
    deployment, deployment_model_contrast = flatten_deployment_contrasts(data)
    interval_folds, intervals = aggregate_interval_transport(data)

    if len(aggregate) != 24:
        raise ValueError(f"Expected 24 aggregate model rows, found {len(aggregate)}")
    if aggregate.isna().any().any():
        raise ValueError("Aggregate model table contains missing values")

    aggregate.to_csv(
        SOURCE_DIR / "figure6_absolute_trajectory_mae.csv", index=False
    )
    inference.to_csv(
        SOURCE_DIR / "figure6_paired_model_inference.csv", index=False
    )
    deployment.to_csv(
        SOURCE_DIR / "figure6_activity_shift_deployment_contrast.csv", index=False
    )
    deployment_model_contrast.to_csv(
        SOURCE_DIR / "figure6_deployment_contrast_model_difference.csv", index=False
    )
    interval_folds.to_csv(
        SOURCE_DIR / "figure6_interval_transport_by_fold.csv", index=False
    )
    intervals.to_csv(
        SOURCE_DIR / "figure6_interval_transport_fold_macro.csv", index=False
    )

    fig = plt.figure(figsize=(7.2, 6.15), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.18, 1.0],
        height_ratios=[0.92, 1.12],
        left=0.10,
        right=0.985,
        bottom=0.10,
        top=0.91,
        wspace=0.47,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    panel_absolute_mae(ax_a, aggregate)
    panel_model_difference(ax_b, inference)
    panel_deployment_contrast(ax_c, deployment)
    panel_interval_transport(ax_d, intervals)

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color=PK_COLOR,
            marker="o",
            ms=4.5,
            lw=1.2,
            label="PK-SSM",
        ),
        mlines.Line2D(
            [],
            [],
            color=TCN_COLOR,
            marker="s",
            ms=4.5,
            lw=1.2,
            label="TCN",
        ),
        mlines.Line2D(
            [],
            [],
            color=NEUTRAL,
            marker="o",
            mfc=NEUTRAL,
            ms=4.5,
            lw=0,
            label="Tagged events (primary)",
        ),
        mlines.Line2D(
            [],
            [],
            color=NEUTRAL,
            marker="D",
            mfc="white",
            ms=4.5,
            lw=0,
            label="Evaluation stride (secondary)",
        ),
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=4,
        fontsize=6.2,
        columnspacing=1.4,
        handletextpad=0.45,
    )
    fig.text(
        0.10,
        0.025,
        "Five outer folds x five seeds; MAE is participant-macro. "
        "Forest-plot bars show participant-bootstrap 95% confidence intervals. "
        "Activities were masked from model inputs.",
        fontsize=5.7,
        color=NEUTRAL,
        ha="left",
    )

    save_outputs(fig)
    plt.close(fig)

    qa_text = "\n".join(
        [
            "Figure 6 source and export audit",
            f"Locked summary SHA-256: {summary_sha256}",
            "Core conclusion: Activity-shift effects are direction-dependent, "
            "primary-policy trajectory differences between PK-SSM and TCN are "
            "not supported at the joint boundary, and transported intervals can "
            "be very wide while still under-covering held-out anaerobic activity.",
            "Archetype: quantitative grid",
            "Backend: Python/matplotlib only",
            "Final nominal size: 7.2 x 6.15 inches (double-column)",
            "Statistics: five outer folds, five locked seeds, 10,000 participant "
            "bootstrap replicates, paired Wilcoxon tests in source tables",
            "Privacy: no participant or session identifiers are exported",
            "Exports: editable SVG, editable-font PDF, 300 dpi PNG, 600 dpi LZW TIFF",
        ]
    )
    (SOURCE_DIR / "figure6_qa_notes.txt").write_text(qa_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
