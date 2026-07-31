#!/usr/bin/env python3
"""Build BSPC v4 Figures 1-5 from locked aggregate and forecast artifacts.

Python is the exclusive graphics backend. The script exports editable SVG,
PDF, 600-dpi TIFF, PNG previews, and de-identified source-data tables. It does
not generate Figure 6 because the activity-shift test remains sealed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures_v4"
SRC = OUT / "source_data"
WEAR = ROOT / "outputs" / "locked_summary_v1"
WEAR_EVAL = ROOT / "outputs" / "locked_evaluation_v1"
GOLD_DATA = ROOT / "outputs" / "goldencheetah_locked_summary_v1"
KIN = ROOT / "outputs" / "locked_kinetic_parameter_audit_v1"

MM = 1 / 25.4
PK = "#0F4D92"
TCN = "#D88032"
INK = "#272727"
MID = "#767676"
LIGHT = "#D8D8D8"
PALE_BLUE = "#DCEAF7"
PALE_ORANGE = "#F8E4CF"
TEAL = "#42949E"
RED = "#B64342"
GREEN = "#3D8B5B"
GOLD = "#C99A2E"

MODEL_LABELS = {
    "pk_ssm": "PK-SSM",
    "tcn": "TCN",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "first_order_kinetics": "First-order kinetics",
    "residual_ssm": "Residual SSM",
    "persistence": "Persistence",
    "damped_trend": "Damped trend",
    "ridge": "Ridge",
    "ridge_raw": "Ridge, raw",
    "ridge_clipped": "Ridge, clipped",
    "ridge_clipped_30_220": "Ridge, clipped",
}

PARAM_LABELS = {
    "rest_hr": "Rest-HR latent",
    "hr_reserve": "HR-reserve latent",
    "gain_fast": "Fast gain",
    "gain_slow": "Slow gain",
    "tau_fast_rise": "Fast rise tau",
    "tau_fast_recovery": "Fast recovery tau",
    "tau_slow_rise": "Slow rise tau",
    "tau_slow_recovery": "Slow recovery tau",
    "feasible_max_hr": "Feasible-max latent",
    "initial_fast_fraction": "Initial fast fraction",
}


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["xtick.labelsize"] = 6.5
    plt.rcParams["ytick.labelsize"] = 6.5
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fontsize"] = 6.5
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")


def clean_axis(ax) -> None:
    ax.tick_params(width=0.7, length=3)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def write_source(df: pd.DataFrame, name: str) -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    df.to_csv(SRC / name, index=False)


def model_label(value: str) -> str:
    return MODEL_LABELS.get(value, value.replace("_", " ").title())


def model_color(value: str) -> str:
    if value == "pk_ssm":
        return PK
    if value == "tcn":
        return TCN
    return MID


def add_box(ax, xy: Tuple[float, float], width: float, height: float, text: str,
            face: str, edge: str = "none", fontsize: float = 6.5) -> None:
    box = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=face, edgecolor=edge, linewidth=0.8,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=INK)


def arrow(ax, start: Tuple[float, float], end: Tuple[float, float], color: str = MID) -> None:
    ax.add_patch(patches.FancyArrowPatch(start, end, arrowstyle="-|>",
                                         mutation_scale=8, lw=0.9, color=color))


def figure1() -> None:
    fig = plt.figure(figsize=(183 * MM, 142 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.72, 1.55], hspace=0.2, wspace=0.24)

    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(-320, 140)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(patches.FancyBboxPatch((-300, 0.28), 300, 0.34,
                                        boxstyle="round,pad=0.02", facecolor=PALE_BLUE,
                                        edgecolor=PK, linewidth=1))
    ax.add_patch(patches.FancyBboxPatch((0, 0.28), 120, 0.34,
                                        boxstyle="round,pad=0.02", facecolor=PALE_ORANGE,
                                        edgecolor=TCN, linewidth=1))
    ax.axvline(0, ymin=0.25, ymax=0.76, color=INK, lw=1.2)
    ax.text(-150, 0.45, "Causal context\nHR + acceleration summaries", ha="center", va="center", fontsize=7)
    ax.text(60, 0.45, "Forecast every second\nfor 120 s", ha="center", va="center", fontsize=7)
    ax.text(-300, 0.18, "-300 s", ha="center")
    ax.text(0, 0.16, "transition origin", ha="center", va="center", fontweight="bold")
    ax.text(120, 0.18, "+120 s", ha="center")
    ax.annotate(
        "No future signal crosses the origin",
        xy=(0, 0.77), xytext=(-18, 0.84),
        ha="right", va="center", color=RED, fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.8},
    )
    panel_label(ax, "a", x=-0.01, y=0.93)

    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    role_specs = [
        ("Train", "15/16 users", "#B8CBE2"),
        ("Validate", "4 users", "#D7E2EF"),
        ("Calibrate", "5 users", "#F5D9BC"),
        ("Test", "6/7 users", "#E8B5AE"),
    ]
    y = 0.74
    for name, count, color in role_specs:
        add_box(ax, (0.12, y), 0.76, 0.12, f"{name}  |  {count}", color)
        y -= 0.16
    ax.text(0.5, 0.06, "Five rotations: each participant is tested once\nNo participant crosses roles within a fold",
            ha="center", va="center", fontsize=6.2, color=INK)
    ax.set_title("Participant-disjoint v4 roles", pad=5, fontweight="bold")
    panel_label(ax, "b", x=-0.03, y=1.01)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_box(ax, (0.03, 0.72), 0.24, 0.13, "300-s\ncontext", PALE_BLUE)
    add_box(ax, (0.34, 0.72), 0.28, 0.13, "Causal\nencoder", "#B8CBE2")
    add_box(ax, (0.69, 0.72), 0.27, 0.13, "Future-workload\npath", "#DDE9E5")
    arrow(ax, (0.27, 0.785), (0.34, 0.785))
    arrow(ax, (0.62, 0.785), (0.69, 0.785))
    add_box(ax, (0.10, 0.43), 0.31, 0.13, "Fast state\nrise / recovery", "#CDE7E9")
    add_box(ax, (0.59, 0.43), 0.31, 0.13, "Slow state\nrise / recovery", "#F2E4BC")
    arrow(ax, (0.80, 0.72), (0.32, 0.56), TEAL)
    arrow(ax, (0.84, 0.72), (0.72, 0.56), GOLD)
    add_box(ax, (0.05, 0.17), 0.23, 0.12, "Bounded\nresidual", "#E5E5E5")
    add_box(ax, (0.34, 0.17), 0.61, 0.12, "Output mean + Student-t scale\nSource-only conformal calibration", "#E7EEF7")
    arrow(ax, (0.26, 0.43), (0.51, 0.29), TEAL)
    arrow(ax, (0.74, 0.43), (0.66, 0.29), GOLD)
    arrow(ax, (0.28, 0.23), (0.34, 0.23))
    ax.set_title("PK-SSM representation", pad=5, fontweight="bold")
    panel_label(ax, "c", x=-0.03, y=1.01)

    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boundaries = [
        ("Held-out user", "LOCKED", PALE_BLUE, PK),
        ("Current-context conditioning", "LOCKED; history gate OFF", "#E8EEF5", MID),
        ("Bidirectional activity shift", "PROTOCOL FROZEN; RESULTS PENDING", "#F7E4DF", RED),
        ("GoldenCheetah", "INDEPENDENT REFIT LOCKED", PALE_ORANGE, TCN),
    ]
    y = 0.76
    for title, status, fill, edge in boundaries:
        box = patches.FancyBboxPatch((0.06, y), 0.88, 0.15,
                                     boxstyle="round,pad=0.015,rounding_size=0.02",
                                     facecolor=fill, edgecolor=edge, lw=0.9)
        ax.add_patch(box)
        ax.text(0.10, y + 0.10, title, ha="left", va="center", fontsize=6.7, fontweight="bold")
        ax.text(0.10, y + 0.045, status, ha="left", va="center", fontsize=5.7,
                color=edge, fontweight="bold")
        y -= 0.205
    ax.set_title("Deployment boundaries", pad=5, fontweight="bold")
    panel_label(ax, "d", x=-0.03, y=1.01)

    write_source(pd.DataFrame([
        {"element": "context_length_s", "value": 300},
        {"element": "forecast_horizon_s", "value": 120},
        {"element": "input_features", "value": 18},
        {"element": "hidden_channels", "value": 64},
        {"element": "residual_bound_bpm", "value": 6},
        {"element": "outer_folds", "value": 5},
    ]), "figure1_design_constants.csv")
    save_bundle(fig, "Figure_1_study_design_and_PKSSM")


def paired_effect_rows(data: Dict, mappings: Sequence[Tuple[str, str, str]], golden: bool = False) -> pd.DataFrame:
    rows = []
    for policy, metric, label in mappings:
        rec = data[policy][metric]
        estimate_key = "mean_pk_ssm_minus_tcn" if golden else "mean_paired_difference"
        p_key = "wilcoxon_p" if golden else "wilcoxon_signed_rank_p"
        rows.append({
            "policy": policy,
            "endpoint": metric,
            "label": label,
            "estimate_bpm": rec[estimate_key],
            "ci_low_bpm": rec["bootstrap_95_ci"][0],
            "ci_high_bpm": rec["bootstrap_95_ci"][1],
            "participants": rec["participants"],
            "wilcoxon_p": rec[p_key],
        })
    return pd.DataFrame(rows)


def draw_forest(ax, df: pd.DataFrame, xlabel: str) -> None:
    y = np.arange(len(df))[::-1]
    for yi, row in zip(y, df.itertuples()):
        color = TCN if row.estimate_bpm > 0 else PK
        ax.plot([row.ci_low_bpm, row.ci_high_bpm], [yi, yi], color=color, lw=1.5)
        ax.plot(row.estimate_bpm, yi, "o", color=color, ms=4.5)
    ax.axvline(0, color=MID, ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel(xlabel)
    lim = max(abs(df["ci_low_bpm"].min()), abs(df["ci_high_bpm"].max())) * 1.15
    ax.set_xlim(-lim, lim)
    clean_axis(ax)


def figure2() -> None:
    primary = pd.read_csv(WEAR / "primary_ensemble_summary.csv")
    fixed = pd.read_csv(WEAR / "secondary_fixed_seed_summary.csv")
    paired = json.loads((WEAR / "paired_participant_inference.json").read_text(encoding="utf-8"))
    tagged = primary[primary["policy"] == "tagged_events"].copy()
    horizon_rows = []
    for row in tagged.itertuples():
        for h in (30, 60, 120):
            horizon_rows.append({"model": row.model, "horizon_s": h,
                                 "mae_bpm": getattr(row, f"mae_{h}s_bpm"),
                                 "participants": row.participants,
                                 "seed_policy": "five-seed arithmetic ensemble"})
    horizon = pd.DataFrame(horizon_rows)
    effects = paired_effect_rows(paired, [
        ("tagged_events", "trajectory_mae_bpm", "Tagged trajectory"),
        ("tagged_events", "mae_30s_bpm", "Tagged 30 s"),
        ("tagged_events", "mae_60s_bpm", "Tagged 60 s"),
        ("tagged_events", "mae_120s_bpm", "Tagged 120 s"),
        ("evaluation_stride", "trajectory_mae_bpm", "Schedule-wide trajectory"),
    ])
    deploy = primary[["policy", "model", "participants", "trajectory_mae_bpm"]].copy()
    baseline = fixed[fixed["policy"] == "tagged_events"].copy()
    write_source(horizon, "figure2a_horizon_accuracy.csv")
    write_source(effects, "figure2b_paired_effects.csv")
    write_source(deploy, "figure2c_deployment_accuracy.csv")
    write_source(baseline, "figure2d_fixed_seed_baselines.csv")

    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 145 * MM), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.15], "height_ratios": [1.0, 1.15]})
    ax = axes[0, 0]
    for model, color, marker in (("pk_ssm", PK, "o"), ("tcn", TCN, "s")):
        d = horizon[horizon["model"] == model].sort_values("horizon_s")
        ax.plot(d["horizon_s"], d["mae_bpm"], color=color, marker=marker, lw=1.8, ms=4.5,
                label=model_label(model))
        for x, y in zip(d["horizon_s"], d["mae_bpm"]):
            ax.text(x, y + 0.35, f"{y:.2f}", color=color, ha="center", fontsize=5.8)
    ax.set_xticks([30, 60, 120])
    ax.set_xlabel("Forecast horizon (s)")
    ax.set_ylabel("MAE (beats/min)")
    ax.set_ylim(4.8, 17.5)
    ax.legend(loc="upper left")
    ax.set_title("Five-seed tagged-transition ensembles", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "a")

    ax = axes[0, 1]
    draw_forest(ax, effects, "PK-SSM - TCN MAE (beats/min)")
    ax.text(0.01, 0.02, "Left favors PK-SSM; right favors TCN", transform=ax.transAxes,
            fontsize=5.8, color=MID)
    ax.set_title("Participant-bootstrap 95% intervals", fontweight="bold")
    panel_label(ax, "b")

    ax = axes[1, 0]
    policies = ["tagged_events", "evaluation_stride"]
    names = ["Tagged\ntransitions", "Schedule-wide"]
    x = np.arange(2)
    width = 0.34
    for offset, model, color in ((-width / 2, "pk_ssm", PK), (width / 2, "tcn", TCN)):
        vals = [deploy[(deploy.policy == p) & (deploy.model == model)].trajectory_mae_bpm.iloc[0] for p in policies]
        bars = ax.bar(x + offset, vals, width=width, color=color, label=model_label(model), edgecolor="white")
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.07, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=5.8, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Trajectory MAE (beats/min)")
    ax.set_ylim(8.0, 10.5)
    ax.legend(loc="upper right")
    ax.set_title("Accuracy depends on origin policy", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    base = baseline.sort_values("trajectory_mae_bpm", ascending=False).copy()
    normal = base[base["trajectory_mae_bpm"] < 20]
    extreme = base[base["trajectory_mae_bpm"] >= 20]
    y = np.arange(len(normal))
    colors = [model_color(m) for m in normal["model"]]
    ax.hlines(y, normal["trajectory_mae_bpm"].min() - 0.15, normal["trajectory_mae_bpm"], color=LIGHT, lw=1)
    ax.scatter(normal["trajectory_mae_bpm"], y, c=colors, s=25, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([model_label(m) for m in normal["model"]])
    xmin = max(9.7, normal["trajectory_mae_bpm"].min() - 0.2)
    xmax = normal["trajectory_mae_bpm"].max() + 0.45
    ax.set_xlim(xmin, xmax)
    for yi, value in zip(y, normal["trajectory_mae_bpm"]):
        ax.text(value + 0.04, yi, f"{value:.2f}", va="center", fontsize=5.5)
    if not extreme.empty:
        label = "; ".join(f"{model_label(r.model)} {r.trajectory_mae_bpm:.1f}" for r in extreme.itertuples())
        ax.text(0.99, 0.02, label + " (outside axis)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.5, color=RED)
    ax.set_xlabel("Fixed-seed trajectory MAE (beats/min)")
    ax.set_title("No fixed-seed accuracy leadership", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "d")
    save_bundle(fig, "Figure_2_heldout_user_accuracy")


def load_wearable_ensemble_cases() -> Tuple[pd.DataFrame, pd.DataFrame]:
    seeds = [20260730, 20260731, 20260732, 20260733, 20260734]
    records: List[Dict] = []
    for fold in range(5):
        base = WEAR_EVAL / "test" / "tagged_events" / f"outer_fold_{fold}"
        meta = pd.read_csv(base / "pk_ssm" / f"seed_{seeds[0]}" / "origin_metadata.csv")
        model_means: Dict[str, np.ndarray] = {}
        target = mask = current = None
        for model in ("pk_ssm", "tcn"):
            seed_arrays = []
            for seed in seeds:
                with np.load(base / model / f"seed_{seed}" / "forecast_bundle.npz", allow_pickle=False) as z:
                    seed_arrays.append(z["mean"].astype(float))
                    if target is None:
                        target = z["target"].astype(float)
                        mask = z["target_valid_mask"].astype(bool)
                        current = z["current_hr"].astype(float)
            model_means[model] = np.mean(seed_arrays, axis=0)
        assert target is not None and mask is not None and current is not None
        valid_n = np.maximum(mask.sum(axis=1), 1)
        pk_err = (np.abs(model_means["pk_ssm"] - target) * mask).sum(axis=1) / valid_n
        tcn_err = (np.abs(model_means["tcn"] - target) * mask).sum(axis=1) / valid_n
        for idx in range(len(meta)):
            records.append({
                "fold": fold,
                "row": idx,
                "combined_mae": float((pk_err[idx] + tcn_err[idx]) / 2),
                "current": float(current[idx]),
                "target": target[idx],
                "mask": mask[idx],
                "pk_ssm": model_means["pk_ssm"][idx],
                "tcn": model_means["tcn"][idx],
            })
    errors = np.array([r["combined_mae"] for r in records])
    quantiles = [0.25, 0.50, 0.75]
    selected: List[int] = []
    for q in quantiles:
        target_q = float(np.quantile(errors, q))
        for idx in np.argsort(np.abs(errors - target_q)):
            if int(idx) not in selected:
                selected.append(int(idx))
                break
    source_rows = []
    case_meta = []
    for q, idx in zip(quantiles, selected):
        rec = records[idx]
        case = f"q{int(q * 100):02d}"
        case_meta.append({"case": case, "selection_quantile": q, "combined_origin_mae_bpm": rec["combined_mae"]})
        source_rows.append({"case": case, "time_s": 0, "observed_hr_bpm": rec["current"],
                            "pk_ssm_hr_bpm": rec["current"], "tcn_hr_bpm": rec["current"]})
        for t in range(120):
            if rec["mask"][t]:
                source_rows.append({"case": case, "time_s": t + 1,
                                    "observed_hr_bpm": rec["target"][t],
                                    "pk_ssm_hr_bpm": rec["pk_ssm"][t],
                                    "tcn_hr_bpm": rec["tcn"][t]})
    return pd.DataFrame(source_rows), pd.DataFrame(case_meta)


def figure3() -> None:
    primary = pd.read_csv(WEAR / "primary_ensemble_summary.csv")
    uncertainty = pd.read_csv(WEAR / "uncertainty_summary.csv")
    traces, trace_meta = load_wearable_ensemble_cases()
    tagged = primary[primary.policy == "tagged_events"].copy()
    unc = uncertainty[uncertainty.policy == "tagged_events"].copy()

    dynamics = tagged[["model", "participants", "total_variation_ratio", "rapid_change_amplitude_ratio"]].copy()
    high = tagged[["model", "participants", "high_hr_mae_bpm", "fixed_160_high_hr_mae_bpm"]].copy()
    write_source(traces, "figure3a_representative_trajectories.csv")
    write_source(trace_meta, "figure3a_selection_quantiles.csv")
    write_source(dynamics, "figure3b_dynamic_fidelity.csv")
    write_source(unc, "figure3c_d_uncertainty.csv")
    write_source(high, "figure3e_high_hr_error.csv")

    fig = plt.figure(figsize=(183 * MM, 178 * MM), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.08], hspace=0.34)
    top = outer[0].subgridspec(1, 4, wspace=0.34)
    bottom = outer[1].subgridspec(
        1, 4, width_ratios=[1.65, 1.0, 1.65, 1.05], wspace=0.48
    )
    legend_ax = fig.add_subplot(top[0, 3])
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([0], [0], color=INK, lw=1.5, label="Observed"),
        Line2D([0], [0], color=PK, lw=1.6, label="PK-SSM"),
        Line2D([0], [0], color=TCN, lw=1.6, label="TCN"),
    ], loc="center", fontsize=7)
    legend_ax.text(0.5, 0.18, "Cases fixed by q25, q50 and q75\nof combined per-origin MAE",
                   ha="center", va="center", fontsize=6, color=MID)
    for i, case in enumerate(["q25", "q50", "q75"]):
        ax = fig.add_subplot(top[0, i])
        d = traces[traces.case == case]
        ax.plot(d.time_s, d.observed_hr_bpm, color=INK, lw=1.35)
        ax.plot(d.time_s, d.pk_ssm_hr_bpm, color=PK, lw=1.45)
        ax.plot(d.time_s, d.tcn_hr_bpm, color=TCN, lw=1.35)
        ax.axvline(0, color=MID, lw=0.7, ls="--")
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 30, 60, 120])
        ax.set_xlabel("Forecast time (s)")
        if i == 0:
            ax.set_ylabel("Heart rate (beats/min)")
            panel_label(ax, "a", y=1.11)
        else:
            ax.set_yticklabels([])
        err = trace_meta.loc[trace_meta.case == case, "combined_origin_mae_bpm"].iloc[0]
        ax.set_title(f"{case.upper()} | MAE {err:.1f} bpm", fontweight="bold")
        clean_axis(ax)

    ax = fig.add_subplot(bottom[0, 0])
    labels = ["Total variation", "Rapid-change amplitude"]
    y = np.arange(2)
    for model, color, offset in (("pk_ssm", PK, 0.10), ("tcn", TCN, -0.10)):
        row = dynamics[dynamics.model == model].iloc[0]
        vals = [row.total_variation_ratio, row.rapid_change_amplitude_ratio]
        ax.scatter(vals, y + offset, color=color, s=27, label=model_label(model), zorder=3)
        for value, yy in zip(vals, y + offset):
            ax.text(value * 1.08, yy, f"{value:.2f}", va="center", fontsize=5.5, color=color)
    ax.axvline(1, color=MID, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 12.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted / observed ratio (log scale)")
    ax.set_title("b    Dynamic under-response", loc="center", y=1.22, fontweight="bold")
    clean_axis(ax)

    ax = fig.add_subplot(bottom[0, 1])
    horizons = [30, 60, 120]
    for model, color, marker in (("pk_ssm", PK, "o"), ("tcn", TCN, "s")):
        row = unc[(unc.model == model) & (unc.interval_type == "origin_level")].iloc[0]
        vals = [row[f"coverage_{h}s"] for h in horizons]
        ax.plot(horizons, vals, color=color, marker=marker, lw=1.6, ms=4, label=model_label(model))
    ax.axhline(0.95, color=MID, ls="--", lw=0.9, label="Nominal 0.95")
    ax.set_xticks(horizons)
    ax.set_ylim(0.91, 0.96)
    ax.set_xlabel("Forecast horizon (s)")
    ax.set_ylabel("Marginal coverage")
    ax.set_title("c    Coverage", loc="center", y=1.22, fontweight="bold")
    clean_axis(ax)

    ax = fig.add_subplot(bottom[0, 2])
    for model, color in (("pk_ssm", PK), ("tcn", TCN)):
        for interval, ls, marker in (("origin_level", "-", "o"), ("participant_block", ":", "^")):
            row = unc[(unc.model == model) & (unc.interval_type == interval)].iloc[0]
            vals = [row[f"mean_width_{h}s_bpm"] for h in horizons]
            ax.plot(horizons, vals, color=color, ls=ls, marker=marker, lw=1.3, ms=3.5)
    ax.set_xticks(horizons)
    ax.set_xlabel("Forecast horizon (s)")
    ax.set_ylabel("Full interval width (beats/min)")
    ax.set_title("d    Interval width", loc="center", y=1.22, fontweight="bold")
    interval_handles = [
        Line2D([0], [0], color=PK, lw=1.4, label="PK-SSM"),
        Line2D([0], [0], color=TCN, lw=1.4, label="TCN"),
        Line2D([0], [0], color=INK, lw=1.2, ls="-", marker="o", ms=3,
               label="Origin-level"),
        Line2D([0], [0], color=INK, lw=1.2, ls=":", marker="^", ms=3,
               label="Participant block"),
    ]
    ax.legend(
        handles=interval_handles, fontsize=4.8, loc="lower center",
        bbox_to_anchor=(0.5, 1.01), ncol=2, borderaxespad=0,
        columnspacing=0.8, handletextpad=0.4,
    )
    clean_axis(ax)

    ax = fig.add_subplot(bottom[0, 3])
    x = np.arange(2)
    width = 0.34
    for offset, model, color in ((-width / 2, "pk_ssm", PK), (width / 2, "tcn", TCN)):
        row = high[high.model == model].iloc[0]
        vals = [row.high_hr_mae_bpm, row.fixed_160_high_hr_mae_bpm]
        bars = ax.bar(x + offset, vals, width=width, color=color, label=model_label(model))
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.1f}",
                    ha="center", fontsize=5.5, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(["Fold p90", "Fixed 160"], rotation=28, ha="right", fontsize=4.8)
    ax.set_ylim(18.6, 20.7)
    ax.set_ylabel("MAE (beats/min)")
    ax.set_title("e    High-HR MAE", loc="center", y=1.22, fontweight="bold")
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
        fontsize=5.0, borderaxespad=0, columnspacing=0.7, handletextpad=0.4,
    )
    clean_axis(ax)
    save_bundle(fig, "Figure_3_dynamics_and_uncertainty")


def figure4() -> None:
    models = pd.read_csv(GOLD_DATA / "model_summary.csv")
    paired = json.loads((GOLD_DATA / "paired_inference.json").read_text(encoding="utf-8"))
    subgroup = pd.read_csv(GOLD_DATA / "subgroup_summary.csv")
    golden_unc = pd.read_csv(GOLD_DATA / "uncertainty_summary.csv")
    wear_primary = pd.read_csv(WEAR / "primary_ensemble_summary.csv")
    wear_unc = pd.read_csv(WEAR / "uncertainty_summary.csv")
    tagged = models[models.policy == "tagged_events"].copy()
    effects = paired_effect_rows(paired, [
        ("tagged_events", "trajectory_mae_bpm", "Tagged trajectory"),
        ("tagged_events", "mae_30s_bpm", "Tagged 30 s"),
        ("tagged_events", "mae_60s_bpm", "Tagged 60 s"),
        ("tagged_events", "mae_120s_bpm", "Tagged 120 s"),
        ("evaluation_stride", "trajectory_mae_bpm", "Schedule-wide trajectory"),
    ], golden=True)
    sport = subgroup[(subgroup.model.isin(["pk_ssm", "tcn"])) &
                     (subgroup.field.str.lower().isin(["sport", "activity", "activity_type"]))].copy()
    if sport.empty:
        sport = subgroup[(subgroup.model.isin(["pk_ssm", "tcn"])) & (subgroup.field == "protocol")].copy()

    cross_rows = []
    for dataset, point_df, unc_df in (("Wearable", wear_primary, wear_unc), ("GoldenCheetah", models, golden_unc)):
        for model in ("pk_ssm", "tcn"):
            point = point_df[(point_df.policy == "tagged_events") & (point_df.model == model)].iloc[0]
            if dataset == "Wearable":
                u = unc_df[(unc_df.policy == "tagged_events") & (unc_df.model == model) &
                           (unc_df.interval_type == "origin_level")].iloc[0]
                width120 = u.mean_width_120s_bpm
                coverage120 = u.coverage_120s
            else:
                u = unc_df[(unc_df.policy == "tagged_events") & (unc_df.model == model)].iloc[0]
                width120 = u.conformal_width_120s_bpm
                coverage120 = u.conformal_coverage_120s
            cross_rows.append({"dataset": dataset, "model": model,
                               "total_variation_ratio": point.total_variation_ratio,
                               "coverage_120s": coverage120, "width_120s_bpm": width120})
    cross = pd.DataFrame(cross_rows)
    write_source(tagged, "figure4a_golden_model_ranking.csv")
    write_source(effects, "figure4b_golden_paired_effects.csv")
    write_source(sport, "figure4c_within_dataset_strata.csv")
    write_source(cross, "figure4d_cross_dataset_dynamics_calibration.csv")

    fig = plt.figure(figsize=(183 * MM, 143 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1], hspace=0.28, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    rank = tagged.sort_values("trajectory_mae_bpm", ascending=False)
    y = np.arange(len(rank))
    ax.hlines(y, rank.trajectory_mae_bpm.min() - 0.2, rank.trajectory_mae_bpm, color=LIGHT, lw=1)
    ax.scatter(rank.trajectory_mae_bpm, y, c=[model_color(m) for m in rank.model], s=25)
    ax.set_yticks(y)
    ax.set_yticklabels([model_label(m) for m in rank.model])
    ax.set_xlabel("Tagged-transition MAE (beats/min)")
    ax.set_title("Ridge leads after dataset-specific refitting", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    draw_forest(ax, effects, "PK-SSM - TCN MAE (beats/min)")
    ax.set_title("No supported PK-SSM/TCN difference", fontweight="bold")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    labels = list(dict.fromkeys(sport.label.astype(str)))
    x = np.arange(len(labels))
    for policy, policy_label, linestyle in (("tagged_events", "tagged", "-"),
                                             ("evaluation_stride", "schedule-wide", "--")):
        for model, color, marker, offset in (("pk_ssm", PK, "o", -0.06), ("tcn", TCN, "s", 0.06)):
            d = sport[(sport.model == model) & (sport.policy == policy)].set_index("label")
            vals = [d.loc[label, "participant_macro_mae_bpm"] for label in labels]
            ax.plot(x + offset, vals, color=color, marker=marker, linestyle=linestyle,
                    lw=1.2, ms=4, label=f"{model_label(model)}, {policy_label}")
    tick_labels = []
    for label in labels:
        n = int(sport[sport.label == label].participants.max())
        tick_labels.append(f"{label}\n(n={n})")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Participant-macro MAE (beats/min)")
    ax.set_title("All displayed strata were seen during fitting", fontweight="bold")
    ax.legend(loc="center right", fontsize=4.8)
    clean_axis(ax)
    panel_label(ax, "c")

    sub = gs[1, 1].subgridspec(1, 2, wspace=0.45)
    ax1 = fig.add_subplot(sub[0, 0])
    ax2 = fig.add_subplot(sub[0, 1])
    datasets = ["Wearable", "GoldenCheetah"]
    x = np.arange(2)
    width = 0.34
    for offset, model, color in ((-width / 2, "pk_ssm", PK), (width / 2, "tcn", TCN)):
        vals = [cross[(cross.dataset == ds) & (cross.model == model)].total_variation_ratio.iloc[0] for ds in datasets]
        ax1.bar(x + offset, vals, width=width, color=color, label=model_label(model))
        vals_w = [cross[(cross.dataset == ds) & (cross.model == model)].width_120s_bpm.iloc[0] for ds in datasets]
        ax2.bar(x + offset, vals_w, width=width, color=color, label=model_label(model))
    ax1.axhline(1, color=MID, ls="--", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Wearable", "Golden\nCheetah"], rotation=20, ha="right")
    ax1.set_ylabel("Total-variation ratio")
    ax1.set_title("Dynamics", fontweight="bold")
    ax1.legend(fontsize=5.2)
    clean_axis(ax1)
    panel_label(ax1, "d")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Wearable", "Golden\nCheetah"], rotation=20, ha="right")
    ax2.set_ylabel("120-s interval width\n(beats/min)")
    ax2.set_title("Calibration width", fontweight="bold")
    clean_axis(ax2)
    save_bundle(fig, "Figure_4_goldencheetah_replication")


def figure5() -> None:
    corr = pd.read_csv(KIN / "parameter_correlations.csv")
    params = pd.read_csv(KIN / "parameter_summary.csv")
    corr = corr[corr.policy == "tagged_events"].copy()
    params = params[params.policy == "tagged_events"].copy()
    selected_corr = corr[(corr.parameter.isin(["rest_hr", "hr_reserve"])) &
                         (corr.covariate.isin(["current_hr", "future_mean_hr"]))].copy()
    selected_params = params[params.parameter.isin(PARAM_LABELS)].copy()
    taus = params[params.parameter.str.startswith("tau_")].copy()
    bounds = selected_params[["parameter", "near_lower_fraction", "near_upper_fraction"]].copy()
    write_source(selected_corr, "figure5a_latent_hr_correlations.csv")
    write_source(selected_params, "figure5b_parameter_stability.csv")
    write_source(taus, "figure5c_time_constant_seed_variability.csv")
    write_source(bounds, "figure5d_bound_saturation.csv")

    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 140 * MM), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0, 0]
    parameters = ["rest_hr", "hr_reserve"]
    y = np.arange(2)
    for cov, color, offset, label in (("current_hr", PK, -0.11, "Current HR"),
                                      ("future_mean_hr", TCN, 0.11, "Future mean HR")):
        vals = [selected_corr[(selected_corr.parameter == p) &
                              (selected_corr.covariate == cov)].spearman_rho.iloc[0] for p in parameters]
        ax.barh(y + offset, vals, height=0.20, color=color)
        for yy, value in zip(y + offset, vals):
            ax.text(value + (0.03 if value >= 0 else -0.03), yy, f"{value:.2f}",
                    va="center", ha="left" if value >= 0 else "right", fontsize=5.7, color=color)
            ax.text(0.04 if value >= 0 else -0.04, yy, label,
                    va="center", ha="left" if value >= 0 else "right",
                    fontsize=4.8, fontweight="bold",
                    color="white" if color == PK else INK)
    ax.axvline(0, color=MID, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([PARAM_LABELS[p] for p in parameters])
    ax.set_xlim(-1.12, 1.12)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Latents re-encode heart-rate level", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "a")

    ax = axes[0, 1]
    stable = selected_params.dropna(subset=["icc_one_way_random", "mean_pairwise_seed_spearman"])
    ax.axvspan(0.5, 1.0, ymin=0.5, ymax=1.0, color="#E9F3E9", zorder=0)
    label_offsets = {"gain_fast": (-4, 5)}
    for row in stable.itertuples():
        color = PK if row.parameter in ("rest_hr", "hr_reserve", "gain_fast") else MID
        ax.scatter(row.icc_one_way_random, row.mean_pairwise_seed_spearman, color=color, s=27, zorder=3)
        if row.parameter.startswith("tau_"):
            continue
        dx, dy = label_offsets.get(row.parameter, (4, 4))
        ax.annotate(PARAM_LABELS.get(row.parameter, row.parameter),
                    (row.icc_one_way_random, row.mean_pairwise_seed_spearman),
                    xytext=(dx, dy), textcoords="offset points", fontsize=4.7,
                    ha="right" if row.parameter == "gain_fast" else "left")
    tau_points = stable[stable.parameter.str.startswith("tau_")]
    tau_x = float(tau_points.icc_one_way_random.mean())
    tau_y = float(tau_points.mean_pairwise_seed_spearman.mean())
    ax.annotate("Rise/recovery tau\n(4 parameters)", xy=(tau_x, tau_y), xytext=(0.58, 0.25),
                textcoords="data", ha="center", va="center", fontsize=4.8, color=MID,
                arrowprops={"arrowstyle": "->", "color": MID, "lw": 0.7})
    ax.axvline(0.5, color=LIGHT, ls="--", lw=0.8)
    ax.axhline(0.5, color=LIGHT, ls="--", lw=0.8)
    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-0.08, 0.84)
    ax.set_xlabel("Participant ICC")
    ax.set_ylabel("Mean seed rank agreement")
    ax.text(0.98, 0.96, "Trait-like region", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.7, color=GREEN)
    ax.set_title("Participant separation does not imply seed stability", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    taus = taus.sort_values("median_within_origin_seed_sd")
    y = np.arange(len(taus))
    colors = [TEAL if "fast" in p else GOLD for p in taus.parameter]
    bars = ax.barh(y, taus.median_within_origin_seed_sd, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([PARAM_LABELS[p] for p in taus.parameter])
    ax.set_xlabel("Median within-origin seed SD (s)")
    for bar, value in zip(bars, taus.median_within_origin_seed_sd):
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=5.7)
    ax.set_title("Time constants vary across initializations", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    bound_plot = bounds.copy()
    bound_plot["max_near_bound_fraction"] = bound_plot[["near_lower_fraction", "near_upper_fraction"]].max(axis=1, skipna=True).fillna(0)
    bound_plot = bound_plot.sort_values("max_near_bound_fraction")
    y = np.arange(len(bound_plot))
    ax.scatter(bound_plot.max_near_bound_fraction * 100, y,
               c=[PK if p in ("rest_hr", "hr_reserve", "gain_fast") else MID for p in bound_plot.parameter],
               s=24)
    ax.set_yticks(y)
    ax.set_yticklabels([PARAM_LABELS[p] for p in bound_plot.parameter])
    xmax = max(1.0, float(bound_plot.max_near_bound_fraction.max() * 100 * 1.2))
    ax.set_xlim(-0.1, xmax)
    ax.set_xlabel("Origins near either finite bound (%)")
    ax.set_title("Instability is not caused by clipping", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "d")
    save_bundle(fig, "Figure_5_kinetic_parameter_identifiability")


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    generated = sorted(p.name for p in OUT.iterdir() if p.is_file())
    print(f"FIGURE_FILES={len(generated)}")
    print(f"SOURCE_TABLES={len(list(SRC.glob('*.csv')))}")
    for name in generated:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
