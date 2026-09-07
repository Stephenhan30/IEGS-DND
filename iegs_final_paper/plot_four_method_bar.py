#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-style figures for the four-method comparison.

The plotting style is tuned for an Applied Energy mixed layout:
- only the first two figures are switched to double-column horizontal layouts;
- serif / Times-like typography;
- readable 7.2--8.6 pt text;
- light dashed grids and thin black axes;
- pastel method colors with compact legends;
- wide multi-panel layouts are stacked vertically.

The script keeps the original data-reading logic and output file names so that
other experiment files / LaTeX references do not need to be changed.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ============================================================================
# >>> [OUTPUT-NAME-UPDATE] 2026-09-05
# Requested figure file names:
#   load_shedding_comparison.png
#   linepack_trajectories.png
#   algorithm_performance.png
#   exchanged_data_volume.png
# Radar figure name remains unchanged.
# The corresponding CSV files use the same stems.
# <<< [OUTPUT-NAME-UPDATE]
# ============================================================================

# -----------------------------------------------------------------------------
# Global paper style
# -----------------------------------------------------------------------------

# ============================================================================
# >>> [MARKED-MIXED-LAYOUT] 2026-09-05
# 说明：本版本采用“混合版式”
#   1) 只有 performance 三子图改为双栏横向 1x3
#   2) 只有 linepack/shedding 四子图改为双栏横向 2x2
#   3) 收敛图、通信图、雷达图继续保持单栏
#   4) 不需要修改 simulation.py
# 搜索关键字：MARKED-MIXED-LAYOUT / DOUBLECOL-1 / DOUBLECOL-2 / SINGLECOL-KEEP
# <<< [MARKED-MIXED-LAYOUT]
# ============================================================================

LOG_FLOOR = 1e-12
# >>> [MARKED-MIXED-LAYOUT-WIDTH]
COL_W = 7.10                 # 仅供前两个双栏图使用
SINGLE_W = 3.45              # 其余图继续使用单栏宽度
# <<< [MARKED-MIXED-LAYOUT-WIDTH]
LINE_W = 1.15
AXIS_W = 0.70
GRID_W = 0.55
MARKER_SIZE = 4.2
BAR_EDGE_W = 0.65

# >>> [COMPACT-BAR-GAP] 2026-09-05
# 保持细柱，同时压缩柱中心间距；重点修正通信量图“柱子细了但空隙过大”的问题。
# <<< [COMPACT-BAR-GAP]

# >>> [THIN-BAR-STYLE] 参考 Applied Energy 实验图的紧凑柱宽
PERFORMANCE_BAR_WIDTH = 0.30   # [THINNER-BAR-1] 原 0.40：双栏性能柱进一步收窄
COMMUNICATION_BAR_WIDTH = 0.20 # [THINNER-BAR-2] 原 0.38：通信量柱明显收窄，更接近期刊风格
# <<< [THIN-BAR-STYLE]

# Pastel colors close to the visual language used by the reference paper.
BLUE = "#5B9BD5"
ORANGE = "#F4A261"
GREEN = "#70AD47"
PURPLE = "#8E73B4"
RED = "#D95F5F"
YELLOW = "#E8C66A"
BLACK = "#222222"
GRID = "#B9B9B9"
PANEL_BG = "#F7FBF6"

METHODS = OrderedDict([
    ("Centralized ND", "centralized"),
    ("Distributed ND", "distributed"),
    ("ADMM", "admm"),
    ("ALADIN", "aladin"),
])
DISTRIBUTED_METHODS = ["Distributed ND", "ADMM", "ALADIN"]
SHORT = {
    "Centralized ND": "CND",
    "Distributed ND": "DND",
    "ADMM": "ADMM",
    "ALADIN": "ALADIN",
}
METHOD_COLOR = {
    "Centralized ND": BLUE,
    "Distributed ND": ORANGE,
    "ADMM": GREEN,
    "ALADIN": PURPLE,
}
METHOD_STYLE = {
    "Distributed ND": "-",
    "ADMM": "--",
    "ALADIN": "-.",
}
METHOD_MARKER = {
    "Distributed ND": "o",
    "ADMM": "s",
    "ALADIN": "^",
}


def setup_style() -> None:
    """Set compact, print-friendly single-column plotting defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.4,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.0,
        "axes.linewidth": AXIS_W,
        "lines.linewidth": LINE_W,
        "xtick.major.width": AXIS_W,
        "ytick.major.width": AXIS_W,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def paper_axes(ax: plt.Axes, *, grid_axis: str = "both", background: bool = False) -> None:
    """Apply the common axis appearance used throughout the paper figures."""
    if background:
        ax.set_facecolor(PANEL_BG)
    ax.grid(
        True,
        axis=grid_axis,
        linestyle="--",
        dashes=(2.5, 2.5),
        linewidth=GRID_W,
        color=GRID,
        alpha=0.55,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True, width=AXIS_W)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_W)
        spine.set_color(BLACK)


def save(fig: plt.Figure, stem: Path, dpi: int) -> None:
    """Save PNG only, matching the original workflow."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.035,
    )


def write_rows(stem: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(
        stem.with_suffix(".csv"), index=False, encoding="utf-8-sig"
    )


def pick(df: pd.DataFrame, *names: str, required: bool = True) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"Missing {names}; available={list(df.columns)}")
    return None


def num(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(float)


def newest(root: Path, pattern: str) -> Path:
    files = [p for p in root.glob(pattern) if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No file found: {root / pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def result_dir(root: Path, slug: str, system: str) -> Path:
    direct = root / f"results_{slug}" / system / "mip-nd"
    if direct.is_dir():
        return direct
    return newest(root, f"results_{slug}/**/run_summary.json").parent


def read_summary(directory: Path) -> dict[str, float]:
    data = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    best = data["best"]["coordinated"]
    thresholds = data.get("linepack_thresholds", {})

    def value(*names: str) -> float:
        for name in names:
            if name in best:
                try:
                    return float(best[name])
                except (TypeError, ValueError):
                    pass
        return math.nan

    def threshold(name: str, default: float) -> float:
        x = float(thresholds.get(name, default))
        return 100.0 * x if x <= 1.0 else x

    return {
        "shed": value("final_total_shed", "total_shed_mwh"),
        "min_lp": value("final_min_lp_pct", "min_lp_pct"),
        "dos_lp": value("final_dos_lp_at_trigger_pct", "dos_lp_at_trigger_pct"),
        "t_fdia": value("T_fdia", "t_fdia"),
        "t_dos": value("T_dos", "t_dos"),
        "replenish": threshold("replenish", 0.60),
        "trip": threshold("trip", 0.30),
    }


def load_results(root: Path, system: str):
    dirs, summaries = OrderedDict(), OrderedDict()
    for method, slug in METHODS.items():
        dirs[method] = result_dir(root, slug, system)
        summaries[method] = read_summary(dirs[method])
    return dirs, summaries


# -----------------------------------------------------------------------------
# Figure 1: vertically stacked performance comparison (single column)
# -----------------------------------------------------------------------------
def plot_performance_3x1(summaries, out: Path, system: str, dpi: int) -> Path:
    # >>> [PERFORMANCE-2X1-SINGLECOL] 2026-09-06
    # 修改：删除原来的 (c) Relative attack effect 子图。
    # 剩余 (a) Cumulative load shedding 和 (b) Linepack drop
    # 改为纵向 2x1，并恢复为单栏宽度 SINGLE_W。
    # 输出文件名仍保持 load_shedding_comparison.png / .csv。
    # <<< [PERFORMANCE-2X1-SINGLECOL]
    methods = list(summaries)
    labels = [SHORT[m] for m in methods]
    colors = [METHOD_COLOR[m] for m in methods]

    # 单栏图里保持“细柱 + 紧凑间距”。
    PERFORMANCE_X_SPACING = 0.82
    x = np.arange(len(methods), dtype=float) * PERFORMANCE_X_SPACING

    shed = np.array([summaries[m]["shed"] for m in methods], dtype=float)
    dos_lp = np.array([summaries[m]["dos_lp"] for m in methods], dtype=float)
    min_lp = np.array([summaries[m]["min_lp"] for m in methods], dtype=float)

    # >>> [PERFORMANCE-2X1-LAYOUT]
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_W, 4.75))
    # <<< [PERFORMANCE-2X1-LAYOUT]
    rows: list[dict] = []

    # ------------------------------------------------------------------
    # (a) Cumulative load shedding
    # ------------------------------------------------------------------
    ax = axes[0]
    bars = ax.bar(
        x, shed,
        width=PERFORMANCE_BAR_WIDTH,
        color=colors,
        edgecolor=BLACK,
        linewidth=BAR_EDGE_W,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e2, max(1.5e5, float(np.nanmax(shed)) * 1.85))
    ax.set_xticks(x, labels)
    ax.set_xlim(x[0] - 0.34, x[-1] + 0.34)
    ax.set_ylabel("Cumulative load shedding (MWh)")
    ax.text(
        0.00, 1.025, "(a)", transform=ax.transAxes,
        ha="left", va="bottom", fontweight="bold", clip_on=False,
    )
    paper_axes(ax, grid_axis="y", background=True)

    for bar, value in zip(bars, shed):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:.1f}",
            ha="center", va="bottom", fontsize=6.8,
        )

    # ------------------------------------------------------------------
    # (b) Linepack drop
    # ------------------------------------------------------------------
    ax = axes[1]
    rep = summaries[methods[0]]["replenish"]
    trip = summaries[methods[0]]["trip"]
    top_y = max(100.0, max(float(np.nanmax(dos_lp)), rep) + 8.0)
    low_y = max(0.0, min(float(np.nanmin(min_lp)), trip) - 8.0)

    for i, method in enumerate(methods):
        c = METHOD_COLOR[method]
        xi = x[i]

        ax.plot(
            [xi, xi], [min_lp[i], dos_lp[i]],
            color=c, linewidth=1.55, zorder=2,
        )
        ax.plot(
            xi, dos_lp[i], marker="o", ms=5.0, mfc="white", mec=c,
            mew=1.15, linestyle="None", zorder=4,
        )
        ax.plot(
            xi, min_lp[i], marker="o", ms=5.0, mfc=c, mec="white",
            mew=0.7, linestyle="None", zorder=4,
        )

        if dos_lp[i] >= rep - 5.0:
            dos_y, dos_va = dos_lp[i] - 2.0, "top"
        else:
            dos_y, dos_va = dos_lp[i] + 2.0, "bottom"

        ax.text(
            xi + 0.05, dos_y, f"{dos_lp[i]:.2f}",
            fontsize=6.4, ha="left", va=dos_va,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.12),
        )
        ax.text(
            xi + 0.05, min_lp[i] - 2.2, f"{min_lp[i]:.2f}",
            fontsize=6.4, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.12),
        )

    ax.axhline(rep, color=GREEN, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(trip, color=RED, linestyle="-.", linewidth=1.0, zorder=1)
    ax.set_xticks(x, labels)
    ax.set_xlim(x[0] - 0.34, x[-1] + 0.34)
    ax.set_ylim(low_y, top_y)
    ax.set_ylabel("Network linepack (%)")
    ax.text(
        0.00, 1.025, "(b)", transform=ax.transAxes,
        ha="left", va="bottom", fontweight="bold", clip_on=False,
    )
    paper_axes(ax, grid_axis="y", background=True)

    ax.text(
        0.50, 0.965, "○ DoS trigger   ● Minimum",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
    )
    ax.text(
        x[-1] + 0.28, rep + 0.9, "60%",
        ha="right", va="bottom", fontsize=6.1, color=GREEN,
    )
    ax.text(
        x[-1] + 0.28, trip + 0.9, "30%",
        ha="right", va="bottom", fontsize=6.1, color=RED,
    )

    # CSV 只保留当前两幅图实际使用的指标。
    for method, s, dlp, mlp in zip(methods, shed, dos_lp, min_lp):
        rows.append({
            "method": method,
            "cumulative_load_shedding_mwh": s,
            "dos_trigger_linepack_pct": dlp,
            "minimum_linepack_pct": mlp,
            "linepack_drop_pct_points": dlp - mlp,
        })

    fig.subplots_adjust(
        left=0.19, right=0.985, top=0.975, bottom=0.075, hspace=0.40,
    )

    stem = out / "load_shedding_comparison"
    save(fig, stem, dpi)
    plt.close(fig)
    write_rows(stem, rows)
    return stem


# -----------------------------------------------------------------------------
# Figure 2: four method trajectories stacked vertically
# -----------------------------------------------------------------------------
def trajectory_data(directory: Path, summary: dict):
    path = directory / "trajectories.csv"
    if not path.is_file():
        path = newest(directory, "**/trajectories.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "scenario" not in df.columns:
        raise KeyError(f"No scenario column in {path}")

    rows = df[df["scenario"].astype(str).str.contains(
        "coordinated", case=False, na=False)].copy()
    if rows.empty:
        raise ValueError(f"No coordinated scenario in {path}")
    first = str(rows["scenario"].iloc[0])
    rows = rows[rows["scenario"].astype(str) == first].copy()

    tcol = pick(rows, "time", "hour", "period", "t")
    lpcol = pick(rows, "lp_pct_pre", "lp_pct", "linepack_pct_pre",
                 "linepack_pct", "linepack_percentage")
    shedcol = pick(rows, "step_shed", "hourly_shed", "load_shedding",
                   "load_shedding_mw", "shed_mw", "power_load_shedding",
                   required=False)
    if shedcol is None:
        raise KeyError(
            f"No hourly load-shedding column in {path}; available={list(rows.columns)}"
        )

    data = pd.DataFrame({
        "hour": num(rows, tcol),
        "lp": num(rows, lpcol),
        "shed": num(rows, shedcol),
    })
    data = data.replace([np.inf, -np.inf], np.nan).dropna().sort_values("hour")
    data = data.drop_duplicates("hour", keep="last")
    return {
        "path": path,
        "hour": data["hour"].to_numpy(float),
        "lp": data["lp"].to_numpy(float),
        "shed": np.maximum(data["shed"].to_numpy(float), 0.0),
        "summary": summary,
    }


def plot_linepack_and_shedding(dirs, summaries, out: Path,
                               system: str, dpi: int) -> Path:
    # >>> [DOUBLECOL-2] 修改点：仅这个“四子图”改成双栏横向 2x2
    # 原来：4x1 纵向单栏；现在：2x2 双栏。
    # (a) CND / (b) DND / (c) ADMM / (d) ALADIN 均放到子图框外。
    # <<< [DOUBLECOL-2]
    """Only this figure is switched to a double-column horizontal 2x2 layout."""
    data = OrderedDict((m, trajectory_data(dirs[m], summaries[m])) for m in METHODS)
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 5.15), sharex=True)
    axes = axes.ravel()
    rows: list[dict] = []

    rep = summaries["Centralized ND"]["replenish"]
    trip = summaries["Centralized ND"]["trip"]
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]
    panel_headers = [f"{panel_letters[i]} {SHORT[m]}" for i, m in enumerate(data.keys())]

    all_lp = np.concatenate([item["lp"] for item in data.values()])
    ymin = max(0.0, min(float(np.nanmin(all_lp)), trip) - 5.0)
    ymax = max(float(np.nanmax(all_lp)), rep) + 5.0

    for idx, (ax, (method, item)) in enumerate(zip(axes, data.items())):
        hours, lp, shed = item["hour"], item["lp"], item["shed"]
        summary = item["summary"]
        color = BLUE

        t_fdia = float(np.ceil(summary["t_fdia"])) if summary["t_fdia"] < 90 else math.nan
        t_dos = float(np.ceil(summary["t_dos"])) if summary["t_dos"] < 90 else math.nan
        xmin, xmax = float(np.min(hours)), float(np.max(hours))

        if np.isfinite(t_fdia):
            ax.axvspan(xmin, t_fdia, color="#EAF3E6", alpha=0.55, zorder=0)
        if np.isfinite(t_fdia) and np.isfinite(t_dos):
            ax.axvspan(t_fdia, t_dos, color="#FFF3D6", alpha=0.55, zorder=0)
        if np.isfinite(t_dos):
            ax.axvspan(t_dos, xmax, color="#F9E7E7", alpha=0.42, zorder=0)

        ax.plot(hours, lp, color=color, linewidth=1.35, marker="o",
                markersize=2.2, markevery=max(1, len(hours)//10),
                label="Linepack state", zorder=4)
        ax.axhline(rep, color=GREEN, linestyle="--", linewidth=1.0, zorder=2)
        ax.axhline(trip, color=RED, linestyle="-.", linewidth=1.0, zorder=2)

        if np.isfinite(t_fdia):
            ax.axvline(t_fdia, color=BLUE, linestyle="--", linewidth=0.88, alpha=0.8)
        if np.isfinite(t_dos):
            ax.axvline(t_dos, color=RED, linestyle=":", linewidth=0.92, alpha=0.9)

        ax2 = ax.twinx()
        ax2.plot(hours, shed, color=ORANGE, linestyle="--", linewidth=1.10,
                 marker="s", markersize=2.0, markevery=max(1, len(hours)//9),
                 label="Hourly load shedding", zorder=5)
        ymax_shed = float(np.nanmax(shed)) if np.any(np.isfinite(shed)) else 0.0
        ax2.set_ylim(0.0, max(1.0, ymax_shed * 1.16))
        ax2.tick_params(direction="in", labelsize=6.7, width=AXIS_W, colors=BLACK)
        ax2.spines["right"].set_linewidth(AXIS_W)
        ax2.spines["right"].set_color(BLACK)

        # show right-axis labels only on the right column
        if idx % 2 == 0:
            ax2.set_yticklabels([])

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        paper_axes(ax, grid_axis="both", background=False)

        if np.isfinite(t_dos) and np.isfinite(summary["dos_lp"]):
            ax.plot(t_dos, summary["dos_lp"], marker="o", ms=4.6,
                    mfc="white", mec=BLACK, mew=0.8, linestyle="None", zorder=12)
            ax.annotate(
                f"DoS {summary['dos_lp']:.1f}%",
                (t_dos, summary["dos_lp"]), xytext=(4, -9),
                textcoords="offset points", ha="left", va="top", fontsize=6.1,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.22),
                zorder=15,
            )

        min_idx = int(np.nanargmin(lp))
        ax.plot(hours[min_idx], lp[min_idx], marker="o", ms=4.6,
                mfc=color, mec="white", mew=0.7, linestyle="None", zorder=12)
        ax.annotate(
            f"min {lp[min_idx]:.1f}%",
            (hours[min_idx], lp[min_idx]), xytext=(4, 4),
            textcoords="offset points", ha="left", va="bottom", fontsize=6.1,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.22),
            zorder=15,
        )

        status = "Trip" if float(np.nanmin(lp)) <= trip else "No trip"
        ax.text(0.985, 0.08, status, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5,
                color=GREEN if status == "Trip" else RED,
                fontweight="bold")

        for h, l, s in zip(hours, lp, shed):
            rows.append({
                "method": method,
                "hour": h,
                "linepack_at_period_start_pct": l,
                "hourly_load_shedding_mw": s,
                "source_csv": str(item["path"]),
            })

    axes[2].set_xlabel("Scheduling period (hour)")
    axes[3].set_xlabel("Scheduling period (hour)")
    fig.text(0.014, 0.50, "Network linepack at period start (%)",
             rotation=90, va="center", ha="left", fontsize=8.2)
    fig.text(0.986, 0.50, "Hourly load shedding (MW)",
             rotation=-90, va="center", ha="right", fontsize=8.0)

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=1.35, marker="o", ms=3.1,
               label="Linepack state"),
        Line2D([0], [0], color=ORANGE, lw=1.10, ls="--", marker="s", ms=3.0,
               label="Hourly load shedding"),
        Line2D([0], [0], color=GREEN, lw=1.0, ls="--", label="60% threshold"),
        Line2D([0], [0], color=RED, lw=1.0, ls="-.", label="30% threshold"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=4, frameon=False, columnspacing=0.9,
        handlelength=1.8, handletextpad=0.42, fontsize=6.7,
    )

    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.11, hspace=0.24, wspace=0.24)

    # >>> [DOUBLECOL-2-TITLE-OUTSIDE] 修改点：标题放到每个子图框外
    for ax, header in zip(axes, panel_headers):
        pos = ax.get_position()
        fig.text(pos.x0, pos.y1 + 0.006, header, ha="left", va="bottom",
                 fontsize=8.0, fontweight="bold")
    # <<< [DOUBLECOL-2-TITLE-OUTSIDE]

    stem = out / "linepack_trajectories"  # [RENAME-2] requested output name
    save(fig, stem, dpi)
    plt.close(fig)
    write_rows(stem, rows)
    return stem


# -----------------------------------------------------------------------------
# [SITUATION-2 / NEW-MECHANISM] Attack-mechanism validation figure
# -----------------------------------------------------------------------------
def _read_trajectory_table(directory: Path) -> tuple[pd.DataFrame, Path]:
    path = directory / "trajectories.csv"
    if not path.is_file():
        path = newest(directory, "**/trajectories.csv")
    return pd.read_csv(path, encoding="utf-8-sig"), path


def _select_scenario_rows(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    if "scenario" not in df.columns:
        raise KeyError("Trajectory CSV has no 'scenario' column.")
    rows = df[df["scenario"].astype(str).str.contains(keyword, case=False, na=False)].copy()
    if rows.empty:
        raise ValueError(f"No scenario containing '{keyword}' was found.")
    scenario_name = str(rows["scenario"].iloc[0])
    rows = rows[rows["scenario"].astype(str) == scenario_name].copy()
    tcol = pick(rows, "time", "hour", "period", "t")
    rows[tcol] = pd.to_numeric(rows[tcol], errors="coerce")
    return rows.dropna(subset=[tcol]).sort_values(tcol).drop_duplicates(tcol, keep="last")


def _require_export_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            "The DND trajectory file does not contain the mechanism/spatial export fields: "
            f"{missing}. These fields are produced by the Situation-2 simulation.py. "
            "Rerun python run_distributed_main.py with that simulation.py and then rerun "
            f"this plotting script. Source: {source}"
        )


def _parse_vector_text(value) -> np.ndarray:
    if pd.isna(value):
        return np.array([], dtype=float)
    value = str(value).strip()
    if not value:
        return np.array([], dtype=float)
    return np.fromstring(value, sep=";", dtype=float)


def _stack_vector_column(rows: pd.DataFrame, column: str) -> np.ndarray:
    vectors = [_parse_vector_text(v) for v in rows[column].tolist()]
    if not vectors or any(v.size == 0 for v in vectors):
        raise ValueError(f"Column '{column}' contains empty vector data.")
    width = vectors[0].size
    if any(v.size != width for v in vectors):
        raise ValueError(f"Column '{column}' has inconsistent vector lengths.")
    return np.vstack(vectors)


def plot_attack_mechanism_3x1(dirs, summaries, out: Path,
                              system: str, dpi: int) -> Path:
    """[SITUATION-2-A] Validate the physical FDIA--DoS coordination mechanism.

    [MECH-DOUBLECOL-1] Double-column horizontal 1x3 figure using only DND
    Baseline and Coordinated trajectories:
      (a) aggregate GFU output;
      (b) aggregate gas-source injection;
      (c) total linepack.
    """
    directory = dirs["Distributed ND"]
    df, source = _read_trajectory_table(directory)
    required = [
        "gfpp_output_mw",
        "gas_source_total",
        "lp_pct_pre",
        "s_t",
        "fdia_active",
    ]
    _require_export_columns(df, required, source)

    baseline = _select_scenario_rows(df, "baseline")
    attack = _select_scenario_rows(df, "coordinated")
    tcol_b = pick(baseline, "time", "hour", "period", "t")
    tcol_a = pick(attack, "time", "hour", "period", "t")

    hb = num(baseline, tcol_b)
    ha = num(attack, tcol_a)
    summary = summaries["Distributed ND"]
    t_fdia = float(np.ceil(summary["t_fdia"]))
    t_dos = float(np.ceil(summary["t_dos"]))
    rep = summary["replenish"]
    trip = summary["trip"]

    # >>> [MECH-DOUBLECOL-1] 机理图改为横向双栏 1x3 布局
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 2.78), sharex=True)
    panel_headers = [
        "(a) GFU output",
        "(b) Gas-source injection",
        "(c) Total linepack",
    ]
    # <<< [MECH-DOUBLECOL-1]

    baseline_style = dict(
        color="#777777", linestyle="--", linewidth=1.15,
        marker="o", markersize=2.3, markevery=3, zorder=3,
    )
    attack_style = dict(
        color=BLUE, linestyle="-", linewidth=1.45,
        marker="o", markersize=2.5, markevery=3, zorder=4,
    )

    def add_attack_events(ax):
        xmin = float(min(np.min(hb), np.min(ha)))
        xmax = float(max(np.max(hb), np.max(ha)))
        ax.axvspan(t_fdia, t_dos, color="#FFF3D6", alpha=0.42, zorder=0)
        ax.axvspan(t_dos, xmax, color="#F9E7E7", alpha=0.34, zorder=0)
        ax.axvline(t_fdia, color=BLUE, linestyle="--", linewidth=0.95, alpha=0.85, zorder=2)
        ax.axvline(t_dos, color=RED, linestyle=":", linewidth=1.0, alpha=0.90, zorder=2)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks([1, 5, 10, 15, 20, 24])
        paper_axes(ax, grid_axis="both", background=False)

    # (a) FDIA-induced GFPP output / gas-use pathway.
    ax = axes[0]
    h_base = ax.plot(hb, num(baseline, "gfpp_output_mw"), label="Baseline", **baseline_style)[0]
    h_att = ax.plot(ha, num(attack, "gfpp_output_mw"), label="Coordinated attack", **attack_style)[0]
    # >>> [MECH-YLABEL-ONELINE] 纵坐标说明改为单行，避免横向 1x3 时相互重叠
    ax.set_ylabel("GFU output (MW)")
    ax.set_xlabel("Scheduling period (hour)")
    add_attack_events(ax)
    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    ax.text(t_fdia + 0.15, ylim[1] - 0.08 * yrange, r"$T_{\mathrm{FDIA}}$",
            color=BLUE, fontsize=6.0, ha="left", va="top")
    ax.text(t_dos + 0.15, ylim[1] - 0.19 * yrange, r"$T_{\mathrm{DoS}}$",
            color=RED, fontsize=6.0, ha="left", va="top")

    # (b) After DoS, source command is locked by the actual simulation logic.
    ax = axes[1]
    ax.plot(hb, num(baseline, "gas_source_total"), **baseline_style)
    ax.plot(ha, num(attack, "gas_source_total"), **attack_style)
    ax.set_ylabel("Gas-source injection")
    ax.set_xlabel("Scheduling period (hour)")
    add_attack_events(ax)
    y_attack = num(attack, "gas_source_total")
    if y_attack.size:
        idx = int(np.argmin(np.abs(ha - t_dos)))
        ax.annotate(
            "DoS: source command locked",
            (ha[idx], y_attack[idx]),
            xytext=(8, -20), textcoords="offset points",
            ha="left", va="top", fontsize=5.8,
            arrowprops=dict(arrowstyle="->", linewidth=0.65, color=BLACK),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.22),
        )

    # (c) Resulting cross-period linepack evolution.
    ax = axes[2]
    ax.plot(hb, num(baseline, "lp_pct_pre"), **baseline_style)
    ax.plot(ha, num(attack, "lp_pct_pre"), **attack_style)
    ax.axhline(rep, color=GREEN, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(trip, color=RED, linestyle="-.", linewidth=1.0, zorder=1)
    ax.set_ylabel("Total linepack (%)")
    # <<< [MECH-YLABEL-ONELINE]
    ax.set_xlabel("Scheduling period (hour)")
    ax.set_ylim(0, 105)
    add_attack_events(ax)
    ax.text(23.8, rep + 1.0, "60%", ha="right", va="bottom", fontsize=6.0, color=GREEN,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.10))
    ax.text(23.8, trip + 1.0, "30%", ha="right", va="bottom", fontsize=6.0, color=RED,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.10))

    # >>> [MECH-DOUBLECOL-2] 使用整图总图例，并将 (a)(b)(c) 放到各子图框外上方
    fig.legend(
        [h_base, h_att], ["Baseline", "Coordinated attack"],
        loc="upper center", bbox_to_anchor=(0.5, 0.995),
        frameon=False, ncol=2, fontsize=6.6,
        handlelength=1.8, columnspacing=0.9, handletextpad=0.45,
    )
    # >>> [MECH-WIDER-PANELS] 缩小子图间隙，让每个子图更宽
    fig.subplots_adjust(left=0.050, right=0.995, top=0.82, bottom=0.22, wspace=0.25)
    # <<< [MECH-WIDER-PANELS]
    for ax, header in zip(axes, panel_headers):
        pos = ax.get_position()
        fig.text(pos.x0, min(0.985, pos.y1 + 0.012), header,
                 ha="left", va="bottom", fontsize=8.0, fontweight="bold")
    # <<< [MECH-DOUBLECOL-2]

    rows = []
    for scenario_name, rows_df, tcol in [
        ("Baseline", baseline, tcol_b),
        ("Coordinated attack", attack, tcol_a),
    ]:
        for _, r in rows_df.iterrows():
            rows.append({
                "scenario": scenario_name,
                "hour": float(r[tcol]),
                "gfpp_output_mw": float(r["gfpp_output_mw"]),
                "gas_source_total_model_unit": float(r["gas_source_total"]),
                "total_linepack_pct_at_period_start": float(r["lp_pct_pre"]),
                "fdia_active": int(r["fdia_active"]),
                "s_t": int(r["s_t"]),
                "source_csv": str(source),
            })

    stem = out / "attack_mechanism"
    save(fig, stem, dpi)
    plt.close(fig)
    write_rows(stem, rows)
    return stem


# -----------------------------------------------------------------------------
# [SITUATION-2 / NEW-SPATIAL] Spatial distribution of electric/gas damage
# -----------------------------------------------------------------------------
def plot_spatial_damage_2x1(dirs, out: Path, system: str, dpi: int,
                            top_k: int = 10) -> Path:
    """[SITUATION-2-B] Single-column 2x1 spatial damage distribution.

    (a) Top-k buses by cumulative load shedding.
    (b) Top-k gas pipelines by maximum attack-induced linepack depletion
        relative to Baseline at the same hour.
    """
    directory = dirs["Distributed ND"]
    df, source = _read_trajectory_table(directory)
    required = ["bus_shedding_vector_mw", "pipeline_linepack_pct_vector"]
    _require_export_columns(df, required, source)

    baseline = _select_scenario_rows(df, "baseline")
    attack = _select_scenario_rows(df, "coordinated")
    tcol_b = pick(baseline, "time", "hour", "period", "t")
    tcol_a = pick(attack, "time", "hour", "period", "t")

    # One-hour scheduling periods: sum of bus MW values across periods = MWh.
    bus_shed = _stack_vector_column(attack, "bus_shedding_vector_mw")
    cumulative_bus_shed = np.sum(np.maximum(bus_shed, 0.0), axis=0)
    k_bus = min(int(top_k), cumulative_bus_shed.size)
    top_bus_idx = np.argsort(cumulative_bus_shed)[::-1][:k_bus]
    top_bus_val = cumulative_bus_shed[top_bus_idx]

    # Baseline subtraction isolates attack-induced gas-side spatial damage.
    baseline_by_hour = {
        float(r[tcol_b]): _parse_vector_text(r["pipeline_linepack_pct_vector"])
        for _, r in baseline.iterrows()
    }
    attack_by_hour = {
        float(r[tcol_a]): _parse_vector_text(r["pipeline_linepack_pct_vector"])
        for _, r in attack.iterrows()
    }
    common_hours = sorted(set(baseline_by_hour).intersection(attack_by_hour))
    if not common_hours:
        raise ValueError("Baseline and coordinated trajectories have no common hours.")

    depletion_rows = []
    for hour in common_hours:
        b = baseline_by_hour[hour]
        a = attack_by_hour[hour]
        if b.size != a.size:
            raise ValueError("Baseline/attack pipeline vector lengths do not match.")
        depletion_rows.append(np.maximum(b - a, 0.0))
    pipeline_depletion = np.max(np.vstack(depletion_rows), axis=0)
    k_pipe = min(int(top_k), pipeline_depletion.size)
    top_pipe_idx = np.argsort(pipeline_depletion)[::-1][:k_pipe]
    top_pipe_val = pipeline_depletion[top_pipe_idx]

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_W, 5.05))
    panel_headers = [
        f"(a) Top-{k_bus} buses by cumulative load shedding",
        f"(b) Top-{k_pipe} pipelines by attack-induced linepack depletion",
    ]

    # Use narrow bars with compact center spacing to match the rest of the paper.
    x_bus = np.arange(k_bus, dtype=float) * 0.72
    bar_width = 0.42
    ax = axes[0]
    bars = ax.bar(x_bus, top_bus_val, width=bar_width, color=BLUE,
                  edgecolor=BLACK, linewidth=BAR_EDGE_W, zorder=3)
    ax.set_xticks(x_bus, [f"B{i+1}" for i in top_bus_idx], rotation=35, ha="right")
    ax.set_ylabel("Cumulative load\nshedding (MWh)")
    paper_axes(ax, grid_axis="y", background=True)
    ymax = max(float(np.max(top_bus_val)) if top_bus_val.size else 0.0, 1.0)
    ax.set_ylim(0.0, ymax * 1.19)
    ax.set_xlim(x_bus[0] - 0.42, x_bus[-1] + 0.42)
    for bar, value in zip(bars, top_bus_val):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.018*ymax,
                f"{value:.0f}", ha="center", va="bottom", fontsize=5.9, rotation=90)

    x_pipe = np.arange(k_pipe, dtype=float) * 0.72
    ax = axes[1]
    bars = ax.bar(x_pipe, top_pipe_val, width=bar_width, color=ORANGE,
                  edgecolor=BLACK, linewidth=BAR_EDGE_W, zorder=3)
    ax.set_xticks(x_pipe, [f"P{i+1}" for i in top_pipe_idx], rotation=35, ha="right")
    ax.set_ylabel("Max. attack-induced\nlinepack depletion (p.p.)")
    paper_axes(ax, grid_axis="y", background=True)
    ymax = max(float(np.max(top_pipe_val)) if top_pipe_val.size else 0.0, 1.0)
    ax.set_ylim(0.0, ymax * 1.19)
    ax.set_xlim(x_pipe[0] - 0.42, x_pipe[-1] + 0.42)
    for bar, value in zip(bars, top_pipe_val):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.018*ymax,
                f"{value:.1f}", ha="center", va="bottom", fontsize=5.9, rotation=90)

    fig.subplots_adjust(left=0.22, right=0.985, top=0.965, bottom=0.10, hspace=0.48)
    for ax, header in zip(axes, panel_headers):
        pos = ax.get_position()
        fig.text(pos.x0, pos.y1 + 0.006, header,
                 ha="left", va="bottom", fontsize=7.7, fontweight="bold")

    rows = []
    for rank, (idx, value) in enumerate(zip(top_bus_idx, top_bus_val), start=1):
        rows.append({
            "panel": "bus_cumulative_load_shedding",
            "rank": rank,
            "component_index_1based": int(idx + 1),
            "component_label": f"Bus {idx+1}",
            "value": float(value),
            "unit": "MWh",
            "source_csv": str(source),
        })
    for rank, (idx, value) in enumerate(zip(top_pipe_idx, top_pipe_val), start=1):
        rows.append({
            "panel": "pipeline_attack_induced_linepack_depletion",
            "rank": rank,
            "component_index_1based": int(idx + 1),
            "component_label": f"Pipeline {idx+1}",
            "value": float(value),
            "unit": "percentage points",
            "source_csv": str(source),
        })

    stem = out / "spatial_damage_distribution"
    save(fig, stem, dpi)
    plt.close(fig)
    write_rows(stem, rows)
    return stem


# -----------------------------------------------------------------------------
# Figure 3: distributed-method convergence
# -----------------------------------------------------------------------------
def upper_curve(directory: Path, system: str):
    path = directory / f"upper_search_curve_{system}.csv"
    if not path.is_file():
        path = newest(directory, "upper_search_curve_*.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    xcol = pick(df, "evaluated", "evaluated_candidates", "candidate_count",
                "num_evaluated", "cumulative_evaluated", "evaluation_index")
    ycol = pick(df, "incumbent_best_feasible_load_shedding",
                "best_feasible_load_shedding", "incumbent_best_load_shedding",
                "best_load_shedding", "best_damage", "best_objective")
    data = pd.DataFrame({"x": num(df, xcol), "y": num(df, ycol)})
    data = data.replace([np.inf, -np.inf], np.nan).dropna().sort_values("x")
    data = data.drop_duplicates("x", keep="last")
    x = data["x"].to_numpy(float)
    y = np.maximum.accumulate(np.maximum(data["y"].to_numpy(float), 0.0))
    return x, y, path


def calibrate(y: np.ndarray, final: float) -> np.ndarray:
    y = np.maximum.accumulate(y.copy())
    scale = max(1.0, float(np.max(np.abs(y))))
    updates = np.flatnonzero(np.r_[False, np.abs(np.diff(y)) > 1e-10 * scale])
    last = int(updates[-1]) if updates.size else len(y) - 1
    y[:last] = np.minimum(y[:last], final)
    y[last:] = final
    return np.maximum.accumulate(y)


def selected_hour(df: pd.DataFrame, requested: int | None):
    if "physical_hour" not in df.columns:
        return df.copy(), None
    available = sorted({int(round(x)) for x in
                        pd.to_numeric(df["physical_hour"], errors="coerce").dropna()})
    if not available:
        return df.copy(), None
    hour = requested if requested in available else available[-1]
    mask = np.isclose(pd.to_numeric(df["physical_hour"], errors="coerce"), hour)
    return df.loc[mask].copy(), hour


def last_tol(df: pd.DataFrame, names: tuple[str, ...], default=1e-3) -> float:
    for name in names:
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce")
            values = values[np.isfinite(values) & (values > 0)]
            if not values.empty:
                return float(values.iloc[-1])
    return default


def add_ratio(parts: list[np.ndarray], df: pd.DataFrame,
              residuals: tuple[str, ...], tolerances: tuple[str, ...]) -> None:
    for name in residuals:
        if name in df.columns:
            parts.append(num(df, name) / max(last_tol(df, tolerances), LOG_FLOOR))
            return


def stopping_trace(method: str, directory: Path, system: str, hour: int | None):
    if method == "Distributed ND":
        path = newest(directory, f"lower_nd_convergence_{system}.csv")
    elif method == "ADMM":
        path = newest(directory, f"admm_convergence_{system}.csv")
    else:
        path = newest(directory, f"aladin_convergence_{system}.csv")

    df, hour = selected_hour(pd.read_csv(path, encoding="utf-8-sig"), hour)
    parts: list[np.ndarray] = []

    if method == "Distributed ND":
        native = num(df, pick(df, "effective_step", "raw_step"))
        add_ratio(parts, df, ("state_relative_residual", "projected_gradient_residual"),
                  ("state_relative_tolerance", "projected_gradient_tolerance"))
        add_ratio(parts, df, ("gt_proposal_relative_consensus_residual",
                              "gt_relative_consensus_residual"),
                  ("gt_relative_consensus_tolerance", "gt_consensus_tolerance"))
        rounds = np.arange(1, len(df) + 1, dtype=float)
    elif method == "ADMM":
        native = num(df, pick(df, "admm_iteration", "iteration"))
        add_ratio(parts, df, ("admm_relative_primal_residual",),
                  ("admm_primal_rel_tolerance", "admm_consensus_rel_tolerance"))
        add_ratio(parts, df, ("admm_relative_dual_residual",),
                  ("admm_dual_rel_tolerance",))
        add_ratio(parts, df, ("admm_relative_consensus_residual",),
                  ("admm_consensus_rel_tolerance",))
        add_ratio(parts, df, ("admm_full_state_relative_residual", "state_relative_residual"),
                  ("admm_state_rel_tolerance", "state_relative_tolerance"))
        rounds = native.copy()
    else:
        native = num(df, pick(df, "aladin_iteration", "iteration"))
        add_ratio(parts, df, ("aladin_relative_consensus_residual",),
                  ("aladin_consensus_rel_tolerance",))
        add_ratio(parts, df, ("aladin_relative_step_residual",),
                  ("aladin_step_rel_tolerance",))
        add_ratio(parts, df, ("aladin_relative_kkt_residual",),
                  ("aladin_kkt_rel_tolerance",))
        add_ratio(parts, df, ("aladin_full_state_relative_residual", "state_relative_residual"),
                  ("aladin_state_rel_tolerance", "state_relative_tolerance"))
        rounds = native.copy()

    if not parts:
        raise ValueError(f"No stopping residual components were found in {path}")

    n = min(len(native), *(len(v) for v in parts))
    matrix = np.vstack([v[:n] for v in parts])
    matrix[~np.isfinite(matrix)] = np.nan
    residual = np.nanmax(matrix, axis=0)
    valid = np.isfinite(native[:n]) & np.isfinite(residual)
    native = native[:n][valid]
    residual = residual[valid]
    rounds = rounds[:n][valid]
    progress = np.linspace(0.0, 100.0, len(residual)) if len(residual) > 1 else np.array([100.0])

    return {
        "path": path,
        "hour": hour,
        "native": native,
        "progress": progress,
        "residual": residual,
        "K": int(math.ceil(rounds[-1])),
    }


def load_traces(dirs, system: str, hour: int | None):
    return OrderedDict((m, stopping_trace(m, dirs[m], system, hour))
                       for m in DISTRIBUTED_METHODS)


# -----------------------------------------------------------------------------
# Extra figure: ablation linepack trajectories
# -----------------------------------------------------------------------------
def ablation_dir(root: Path, system: str) -> Path:
    direct = root / "results_ablation" / system / "mip-nd"
    if not direct.is_dir():
        raise FileNotFoundError(f"Ablation directory not found: {direct}")
    return direct


def plot_attack_ablation_linepack(root: Path, out: Path, system: str, dpi: int) -> Path:
    # >>> [ABLATION-LINEPACK-1] 新增：基于 results_ablation 的消融实验管存变化图
    # 场景：Baseline / FDIA-only / DoS-only / Coordinated
    # 数据来源：attack_ablation_trajectories.csv + attack_ablation_summary.json
    # <<< [ABLATION-LINEPACK-1]
    directory = ablation_dir(root, system)
    traj = pd.read_csv(directory / "attack_ablation_trajectories.csv")
    meta = json.loads((directory / "attack_ablation_summary.json").read_text(encoding="utf-8"))

    lp_col = pick(traj, "lp_pct_pre", "lp_pct", "lp_pct_post")
    time_col = pick(traj, "time", "hour")
    scenarios = ["Baseline", "FDIA-only", "DoS-only", "Coordinated"]
    present = [s for s in scenarios if s in traj["scenario"].unique()]
    if not present:
        raise ValueError("No expected ablation scenarios found in attack_ablation_trajectories.csv")

    final_rows = meta.get("final_replay", [])
    replay = {row.get("scenario"): row for row in final_rows if isinstance(row, dict)}
    t_fdia = replay.get("Coordinated", {}).get("T_FDIA_h")
    t_dos = replay.get("Coordinated", {}).get("T_DoS_h")
    if t_fdia is None:
        t_fdia = meta.get("coordinated_optimization", {}).get("T_FDIA_h")
    if t_dos is None:
        t_dos = meta.get("coordinated_optimization", {}).get("T_DoS_h")

    color_map = {
        "Baseline": "#6E6E6E",  # [ABLATION-LINEPACK-2] baseline 改为灰色，避免与坐标轴颜色过近
        "FDIA-only": BLUE,
        "DoS-only": GREEN,
        "Coordinated": ORANGE,
    }
    style_map = {
        "Baseline": "-",
        "FDIA-only": "--",
        "DoS-only": "-.",
        "Coordinated": "-",
    }
    marker_map = {
        "Baseline": "o",
        "FDIA-only": "s",
        "DoS-only": "^",
        "Coordinated": "D",
    }

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.85))
    rows: list[dict] = []

    ax.axhspan(30.0, 60.0, color="#FFF3D6", alpha=0.45, zorder=0)
    ax.axhline(60.0, color=GREEN, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(30.0, color=RED, linestyle="-.", linewidth=1.0, zorder=1)

    handles = []
    labels = []
    for scenario in present:
        sub = traj[traj["scenario"] == scenario].copy()
        sub = sub.sort_values(time_col)
        x = num(sub, time_col)
        y = num(sub, lp_col)
        lw = 1.45 if scenario == "Coordinated" else 1.25
        z = 5 if scenario == "Coordinated" else 4
        line = ax.plot(
            x, y,
            color=color_map[scenario],
            linestyle=style_map[scenario],
            linewidth=lw,
            marker=marker_map[scenario],
            markersize=2.6,
            markevery=max(1, len(x) // 8),
            label=scenario,
            zorder=z,
        )[0]
        handles.append(line)
        labels.append(scenario)

        if scenario != "Baseline" and np.isfinite(y).any():
            idx = int(np.nanargmin(y))
            ax.plot(x[idx], y[idx], marker="o", ms=4.5, mfc="white", mec=color_map[scenario],
                    mew=0.9, linestyle="None", zorder=7)
            ax.annotate(
                f"min {y[idx]:.1f}%",
                (x[idx], y[idx]), xytext=(4, -9 if scenario == "DoS-only" else 5),
                textcoords="offset points", ha="left",
                va="top" if scenario == "DoS-only" else "bottom",
                fontsize=6.1,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.20),
                zorder=8,
            )

        for _, row in sub.iterrows():
            rows.append({
                "scenario": scenario,
                "hour": float(row[time_col]),
                "linepack_at_period_start_pct": float(row[lp_col]),
                "step_shed_mw": float(row.get("step_shed", 0.0)),
                "fdia_active": int(row.get("fdia_active", 0)),
                "dos_vulnerable": int(row.get("dos_vulnerable", 0)),
                "source_csv": str(directory / "attack_ablation_trajectories.csv"),
            })

    if t_fdia is not None:
        ax.axvline(float(t_fdia), color=BLUE, linestyle=":", linewidth=1.0, alpha=0.95, zorder=2)
        ax.text(float(t_fdia) + 0.15, 97.5, r"$T_{\mathrm{FDIA}}$", color=BLUE, fontsize=6.4,  # [ABLATION-LINEPACK-4] T_FDIA 下移
                ha="left", va="bottom")
    if t_dos is not None:
        ax.axvline(float(t_dos), color=RED, linestyle=":", linewidth=1.0, alpha=0.95, zorder=2)
        ax.text(float(t_dos) + 0.15, 95.0, r"$T_{\mathrm{DoS}}$", color=RED, fontsize=6.4,
                ha="left", va="bottom")

    paper_axes(ax, grid_axis="both", background=True)
    ax.set_xlim(1.0, 24.0)
    ax.set_ylim(10.0, 104.0)
    ax.set_xticks([1, 5, 10, 15, 20, 24])
    ax.set_xlabel("Scheduling period (hour)")
    ax.set_ylabel("Network linepack at period start (%)")
    # >>> [ABLATION-LINEPACK-3] 去掉标题；将 60%/30% 阈值文字移到右侧
    ax.text(23.85, 60.8, "60%", color=GREEN, fontsize=6.2, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.10))
    ax.text(23.85, 30.8, "30%", color=RED, fontsize=6.2, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.10))
    # <<< [ABLATION-LINEPACK-3]

    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.19),  # [ABLATION-LINEPACK-5] 图例上移
              ncol=2, frameon=False, columnspacing=0.9, handlelength=1.8,
              handletextpad=0.4, fontsize=6.5)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.88, bottom=0.17)

    stem = out / "attack_ablation_linepack"
    save(fig, stem, dpi)
    plt.close(fig)
    write_rows(stem, rows)
    return stem


def plot_combined_convergence(dirs, summaries, traces, out: Path,
                              system: str, dpi: int) -> Path:
    # >>> [SINGLECOL-KEEP-1] 保持单栏：收敛图仍使用 SINGLE_W
    # <<< [SINGLECOL-KEEP-1]
    """Single-column convergence figure: broken upper search + residual panel."""
    fig = plt.figure(figsize=(SINGLE_W, 5.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.33)
    upper_grid = outer[0].subgridspec(2, 1, height_ratios=[0.72, 1.0], hspace=0.045)
    ax_hi = fig.add_subplot(upper_grid[0])
    ax_lo = fig.add_subplot(upper_grid[1], sharex=ax_hi)
    ax_res = fig.add_subplot(outer[1])
    rows: list[dict] = []

    plot_order = ["ALADIN", "ADMM", "Distributed ND"]
    zorder_map = {"ALADIN": 2, "ADMM": 3, "Distributed ND": 4}
    legend_handles: dict[str, object] = {}

    for method in plot_order:
        x, raw, path = upper_curve(dirs[method], system)
        y = calibrate(raw, summaries[method]["shed"])
        for target in (ax_hi, ax_lo):
            h = target.step(
                x, y, where="post", color=METHOD_COLOR[method],
                linestyle=METHOD_STYLE[method], linewidth=1.35,
                marker=METHOD_MARKER[method], markersize=2.3,
                markevery=max(1, len(x)//7), label=SHORT[method],
                zorder=zorder_map[method],
            )[0]
            legend_handles.setdefault(SHORT[method], h)
        rows.extend({
            "figure_panel": "upper_search",
            "method": method,
            "horizontal_value": xi,
            "value": yi,
            "source_csv": str(path),
        } for xi, yi in zip(x, y))

    for target in (ax_hi, ax_lo):
        target.set_xscale("log")
        paper_axes(target, grid_axis="both", background=True)
        target.margins(x=0.04)

    ax_hi.set_ylim(10000.0, 52000.0)
    ax_hi.set_yticks([10000, 30000, 50000])
    ax_lo.set_ylim(0.0, 1100.0)
    ax_lo.set_yticks([0, 500, 1000])

    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_lo.tick_params(axis="x", which="both", top=False)

    d = 0.009
    kw = dict(color=BLACK, clip_on=False, linewidth=AXIS_W)
    ax_hi.plot((-d, +d), (-d, +d), transform=ax_hi.transAxes, **kw)
    ax_hi.plot((1-d, 1+d), (-d, +d), transform=ax_hi.transAxes, **kw)
    ax_lo.plot((-d, +d), (1-d, 1+d), transform=ax_lo.transAxes, **kw)
    ax_lo.plot((1-d, 1+d), (1-d, 1+d), transform=ax_lo.transAxes, **kw)

    # >>> [FIX-YLABEL-ALIGN] 统一上下两个纵坐标标题的位置
    ax_lo.set_xlabel("Evaluated attack strategies")
    ylabel_x = 0.078
    fig.text(
        ylabel_x,
        0.765,
        "Best-so-far load shedding (MWh)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8.2,
    )
    # <<< [FIX-YLABEL-ALIGN]

    desired = ["DND", "ADMM", "ALADIN"]
    ax_hi.legend(
        [legend_handles[n] for n in desired], desired,
        loc="upper left", bbox_to_anchor=(0.14, 1.02),
        frameon=False, ncol=3, columnspacing=0.8,
        handlelength=1.7, handletextpad=0.35, fontsize=6.5,
    )

    # Lower stopping-residual panel.
    for method in plot_order:
        trace = traces[method]
        residual = np.maximum(trace["residual"], LOG_FLOOR)
        ax_res.semilogy(
            trace["progress"], residual,
            color=METHOD_COLOR[method], linestyle=METHOD_STYLE[method],
            linewidth=1.35, marker=METHOD_MARKER[method], markersize=2.5,
            markevery=max(1, len(residual)//8), label=SHORT[method],
            zorder=zorder_map[method],
        )
        rows.extend({
            "figure_panel": "lower_stopping",
            "method": method,
            "horizontal_value": p,
            "value": r,
            "physical_hour": trace["hour"],
            "source_csv": str(trace["path"]),
        } for p, r in zip(trace["progress"], residual))

    ax_res.axhline(1.0, color=BLACK, linestyle=":", linewidth=1.0,
                   label=r"Threshold $R_m^k=1$")
    ax_res.set_xlim(0.0, 100.0)

    all_residuals = np.concatenate([
        np.maximum(traces[m]["residual"], LOG_FLOOR) for m in DISTRIBUTED_METHODS
    ])
    positive = all_residuals[np.isfinite(all_residuals) & (all_residuals > 0)]
    if positive.size:
        ymin = 10.0 ** math.floor(math.log10(float(np.min(positive))))
        ymax = 10.0 ** math.ceil(math.log10(float(np.max(positive))))
        ax_res.set_ylim(min(ymin, 1e-3), max(ymax, 1e3))

    ax_res.set_xlabel("Normalized solver progress (%)")

    # >>> [FIX-YLABEL-ALIGN] 与上半部分使用相同的 ylabel_x，保证上下完全对齐
    fig.text(
        ylabel_x,
        0.285,
        r"Normalized residual $R_m^k$",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8.2,
    )
    # <<< [FIX-YLABEL-ALIGN]

    paper_axes(ax_res, grid_axis="both", background=True)

    # >>> [ALG-PANEL-OUTSIDE-3]
    # 先完成整体布局，再读取子图位置；否则 get_position() 得到的是调整前坐标，
    # (a)/(b) 会与纵坐标标题或刻度发生重合。
    fig.subplots_adjust(left=0.19, right=0.985, top=0.955, bottom=0.075)
    pos_hi = ax_hi.get_position()
    pos_res = ax_res.get_position()
    fig.text(pos_hi.x0 + 0.004, pos_hi.y1 + 0.010, "(a)",
             ha="left", va="bottom", fontsize=8.4, fontweight="bold")
    fig.text(pos_res.x0 + 0.004, pos_res.y1 + 0.010, "(b)",
             ha="left", va="bottom", fontsize=8.4, fontweight="bold")
    # <<< [ALG-PANEL-OUTSIDE-3]

    handles, labels = ax_res.get_legend_handles_labels()
    order_names = ["DND", "ADMM", "ALADIN", r"Threshold $R_m^k=1$"]
    order = [labels.index(n) for n in order_names]
    ax_res.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2,
        frameon=False, columnspacing=0.8, handlelength=1.8,
        handletextpad=0.4, fontsize=6.3,
    )

    stem = out / "algorithm_performance"  # [RENAME-3] requested output name
    save(fig, stem, dpi)
    plt.close(fig)
    write_rows(stem, rows)
    return stem


# -----------------------------------------------------------------------------
# Figure 4: communication overhead
# -----------------------------------------------------------------------------
def plot_communication(traces, out: Path, system: str, dpi: int, gfpp: int) -> Path:
    # >>> [SINGLECOL-KEEP-2] 保持单栏：通信图仍使用 SINGLE_W
    # <<< [SINGLECOL-KEEP-2]
    methods = DISTRIBUTED_METHODS
    dmap = {
        "Distributed ND": 4 * gfpp,
        "ADMM": 4 * gfpp,
        "ALADIN": 10 * gfpp + 2,
    }
    K = np.array([traces[m]["K"] for m in methods], dtype=int)
    d = np.array([dmap[m] for m in methods], dtype=int)
    D = K * d
    efficiency = np.min(D) / np.maximum(D.astype(float), LOG_FLOOR)

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.78))
    ax2 = ax.twinx()
    # >>> [COMPACT-BAR-GAP-2] 通信图：保留 0.20 的细柱宽，但缩短柱中心间距
    # 原中心间距=1.00；现在=0.68，视觉上更接近期刊窄柱风格。
    COMMUNICATION_X_SPACING = 0.68
    x = np.arange(len(methods), dtype=float) * COMMUNICATION_X_SPACING
    colors = [METHOD_COLOR[m] for m in methods]
    # <<< [COMPACT-BAR-GAP-2]

    # >>> [THIN-BAR-3] 通信量柱宽缩窄
    bars = ax.bar(
        # [THINNER-BAR-2] 通信量柱使用更窄宽度，避免单栏图显得过重
        x, D, width=COMMUNICATION_BAR_WIDTH, color=colors, edgecolor=BLACK,
        linewidth=BAR_EDGE_W, zorder=3,
    )
    # <<< [THIN-BAR-3]
    ax.set_xticks(x, [SHORT[m] for m in methods])
    ax.set_ylabel(r"Exchanged data volume $D_m$ (scalars)")
    ax.set_ylim(0.0, float(np.max(D)) * 1.34)
    # [COMPACT-BAR-GAP-3] 收紧左右空白，而不是把柱子加粗。
    ax.set_xlim(x[0] - 0.28, x[-1] + 0.28)
    paper_axes(ax, grid_axis="y", background=True)

    ymax = float(np.max(D))
    for bar, total, k_value, d_value in zip(bars, D, K, d):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            total + 0.035 * ymax,
            f"{int(total):,}\n$K_m$={int(k_value)}",
            ha="center", va="bottom", fontsize=6.5,
        )

    line, = ax2.plot(
        x, efficiency, color=BLUE, linestyle="-", linewidth=1.25,
        marker="o", markersize=4.7, markerfacecolor="white",
        markeredgecolor=BLUE, markeredgewidth=1.0,
        label="Relative communication efficiency", zorder=6,
    )
    ax2.set_ylim(0.0, 1.12)
    ax2.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax2.set_ylabel("Relative communication efficiency")
    ax2.tick_params(direction="in", top=True, right=True, width=AXIS_W, colors=BLACK)
    ax2.spines["right"].set_linewidth(AXIS_W)
    ax2.spines["right"].set_color(BLACK)

    for xi, eta in zip(x, efficiency):
        ax2.annotate(f"{eta:.2f}", xy=(xi, eta), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.5)

    # The right y-axis already identifies the line, so no legend is needed.

    fig.subplots_adjust(left=0.20, right=0.82, top=0.97, bottom=0.16)
    stem = out / "exchanged_data_volume"  # [RENAME-4] requested output name
    save(fig, stem, dpi)
    plt.close(fig)

    pd.DataFrame({
        "method": methods,
        "coordination_iterations_Km": K,
        "scalars_per_iteration_dm": d,
        "cumulative_exchanged_scalars_Dm": D,
        "relative_communication_efficiency_eta": efficiency,
    }).to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    return stem


# -----------------------------------------------------------------------------
# Figure 5: radar chart
# -----------------------------------------------------------------------------
def plot_comprehensive_radar(dirs, summaries, traces, out: Path,
                             system: str, dpi: int, gfpp: int) -> Path:
    # >>> [SINGLECOL-KEEP-3] 保持单栏：雷达图仍使用 SINGLE_W
    # <<< [SINGLECOL-KEEP-3]
    methods = DISTRIBUTED_METHODS
    RADAR_FLOOR = 0.20

    def metric_scale(values: np.ndarray, *, higher_is_better: bool,
                     logarithmic: bool, floor: float = RADAR_FLOOR) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Radar metric contains non-finite values: {values}")
        transformed = values.copy()
        if logarithmic:
            transformed = np.log10(np.maximum(transformed, LOG_FLOOR))
        lower, upper = float(np.min(transformed)), float(np.max(transformed))
        if abs(upper - lower) <= 1e-12:
            return np.full(transformed.shape, 0.5 * (1.0 + floor), dtype=float)
        normalized = (transformed - lower) / (upper - lower)
        if not higher_is_better:
            normalized = 1.0 - normalized
        return np.clip(floor + (1.0 - floor) * normalized, floor, 1.0)

    attack_raw = np.array([summaries[m]["shed"] for m in methods], dtype=float)
    attack_score = metric_scale(attack_raw, higher_is_better=True, logarithmic=True)

    linepack_drop_raw = np.array([
        summaries[m]["dos_lp"] - summaries[m]["min_lp"] for m in methods
    ], dtype=float)
    linepack_drop_raw = np.maximum(linepack_drop_raw, 0.0)
    linepack_score = metric_scale(linepack_drop_raw, higher_is_better=True, logarithmic=False)

    dmap = {"Distributed ND": 4 * gfpp, "ADMM": 4 * gfpp, "ALADIN": 10 * gfpp + 2}
    rounds = np.array([traces[m]["K"] for m in methods], dtype=float)
    data_per_round = np.array([dmap[m] for m in methods], dtype=float)
    communication_volume_raw = rounds * data_per_round
    communication_efficiency_raw = (
        np.min(communication_volume_raw)
        / np.maximum(communication_volume_raw, LOG_FLOOR)
    )
    communication_score = metric_scale(
        communication_efficiency_raw, higher_is_better=True, logarithmic=False
    )

    search_candidates_raw = []
    for method in methods:
        x, raw_y, _ = upper_curve(dirs[method], system)
        displayed_y = calibrate(raw_y, summaries[method]["shed"])
        scale = max(1.0, float(np.max(np.abs(displayed_y))))
        change_mask = np.r_[True, np.abs(np.diff(displayed_y)) > 1e-10 * scale]
        changed = np.flatnonzero(change_mask)
        last_index = int(changed[-1]) if changed.size else len(x) - 1
        search_candidates_raw.append(max(float(x[last_index]), 1.0))
    search_candidates_raw = np.asarray(search_candidates_raw, dtype=float)
    search_score = metric_scale(search_candidates_raw, higher_is_better=False, logarithmic=True)

    final_residual_raw = np.array([
        max(float(traces[m]["residual"][-1]), LOG_FLOOR) for m in methods
    ], dtype=float)
    convergence_score = metric_scale(
        final_residual_raw, higher_is_better=False, logarithmic=True
    )

    metric_labels = [
        "Attack\neffect",
        "Linepack\ndepletion",
        "Communication\nefficiency",
        "Search\nefficiency",
        "Convergence\nquality",
    ]
    score_matrix = np.column_stack([
        attack_score, linepack_score, communication_score,
        search_score, convergence_score,
    ])

    n_metrics = len(metric_labels)
    angles = np.linspace(0.0, 2.0 * np.pi, n_metrics, endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    fig, ax = plt.subplots(figsize=(SINGLE_W, 3.50), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_facecolor(PANEL_BG)

    plot_order = ["ALADIN", "ADMM", "Distributed ND"]
    method_index = {m: i for i, m in enumerate(methods)}
    handles = {}

    for draw_idx, method in enumerate(plot_order):
        vals = score_matrix[method_index[method]]
        closed_vals = np.r_[vals, vals[0]]
        line, = ax.plot(
            closed_angles, closed_vals,
            color=METHOD_COLOR[method], linestyle=METHOD_STYLE[method],
            linewidth=1.25, marker=METHOD_MARKER[method], markersize=3.7,
            markerfacecolor="white", markeredgecolor=METHOD_COLOR[method],
            markeredgewidth=0.9, label=SHORT[method], zorder=5 + draw_idx,
        )
        ax.fill(closed_angles, closed_vals, color=METHOD_COLOR[method], alpha=0.045, zorder=1)
        handles[method] = line

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6.3)
    ax.set_rlabel_position(60)
    ax.tick_params(axis="y", pad=0.5, colors="#555555")
    for lab in ax.get_yticklabels():
        lab.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.05})

    ax.set_xticks([])
    label_radius = [1.035, 1.09, 1.105, 1.09, 1.09]
    for idx, (angle, label, radius) in enumerate(zip(angles, metric_labels, label_radius)):
        deg = np.degrees(angle) % 360.0
        if idx == 0:
            ha, va = "center", "bottom"
        elif 0.0 < deg < 180.0:
            ha, va = "left", "center"
        else:
            ha, va = "right", "center"
        ax.text(angle, radius, label, ha=ha, va=va, fontsize=6.9,
                color=BLACK, clip_on=False, zorder=30)

    ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=GRID_W, alpha=0.65)
    ax.xaxis.grid(True, color="#C9C9C9", linestyle="-", linewidth=GRID_W, alpha=0.48)
    ax.spines["polar"].set_color(BLACK)
    ax.spines["polar"].set_linewidth(AXIS_W)

    fig.legend(
        handles=[handles["Distributed ND"], handles["ADMM"], handles["ALADIN"]],
        labels=["DND", "ADMM", "ALADIN"],
        loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3,
        frameon=False, handlelength=1.8, columnspacing=1.1, fontsize=6.8,
    )
    fig.subplots_adjust(left=0.105, right=0.895, top=0.92, bottom=0.15)

    stem = out / f"three_distributed_methods_radar_{system}"  # [RADAR-NAME-UNCHANGED]
    save(fig, stem, dpi)
    plt.close(fig)

    csv_rows = []
    for i, method in enumerate(methods):
        csv_rows.append({
            "method": method,
            "attack_effect_raw_mwh": attack_raw[i],
            "attack_effect_radar_score": attack_score[i],
            "linepack_drop_raw_percentage_points": linepack_drop_raw[i],
            "linepack_depletion_radar_score": linepack_score[i],
            "communication_volume_raw_scalars": communication_volume_raw[i],
            "communication_efficiency_raw": communication_efficiency_raw[i],
            "communication_efficiency_radar_score": communication_score[i],
            "last_best_update_candidate": search_candidates_raw[i],
            "search_efficiency_radar_score": search_score[i],
            "final_normalized_stopping_residual": final_residual_raw[i],
            "convergence_quality_radar_score": convergence_score[i],
            "average_radar_score": float(np.mean(score_matrix[i])),
        })
    pd.DataFrame(csv_rows).to_csv(
        stem.with_suffix(".csv"), index=False, encoding="utf-8-sig"
    )
    return stem


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--system", default="118-135")
    parser.add_argument("--out-dir", default="results_method_compare/118-135")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--physical-hour", type=int, default=None)
    parser.add_argument("--gfpp-count", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out_dir)
    out = out if out.is_absolute() else root / out
    out.mkdir(parents=True, exist_ok=True)

    setup_style()
    dirs, summaries = load_results(root, args.system)
    traces = load_traces(dirs, args.system, args.physical_hour)

        # >>> [MARKED-MIXED-LAYOUT-MAIN]
    # 当前版式：performance 为单栏2x1；linepack trajectories 为双栏2x2；
    # 其余比较图保持单栏；保留 mechanism，取消 spatial_damage_distribution。
    figures = [
        plot_performance_3x1(summaries, out, args.system, args.dpi),
        plot_linepack_and_shedding(dirs, summaries, out, args.system, args.dpi),
        plot_combined_convergence(dirs, summaries, traces, out, args.system, args.dpi),
        plot_communication(traces, out, args.system, args.dpi, args.gfpp_count),
        plot_comprehensive_radar(
            dirs, summaries, traces, out, args.system, args.dpi, args.gfpp_count
        ),
        plot_attack_ablation_linepack(root, out, args.system, args.dpi),
        # >>> [SITUATION-2] Newly enabled after rerunning DND with the extra exports.
        plot_attack_mechanism_3x1(dirs, summaries, out, args.system, args.dpi),
        # [REMOVE-SPATIAL-1] 用户要求去掉 spatial_damage_distribution 图
        # <<< [SITUATION-2]
    ]
    # <<< [MARKED-MIXED-LAYOUT-MAIN]

    print("Completed. Generated:")
    for stem in figures:
        print(f"  {stem}.png and {stem}.csv")


if __name__ == "__main__":
    main()
