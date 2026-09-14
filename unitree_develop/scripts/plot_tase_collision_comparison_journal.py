#!/usr/bin/env python3
"""Clean journal-style plots for the current paired SIFR/DIFR logs.

This file deliberately leaves ``plot_tase_collision_comparison.py`` unchanged.
It reuses that script's event alignment and metric definitions, while replacing
the presentation layer and accepting both legacy and current extended logs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "plot_tase_collision_comparison.py"
SPEC = importlib.util.spec_from_file_location("_tase_collision_base", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load plotting helpers from {SOURCE}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


# Colour-blind-safe Okabe-Ito palette, with restrained Nature-style backgrounds.
DIFR = "#0072B2"
SIFR = "#D55E00"
REFERENCE = "#3B3B3B"
SAFE = "#009E73"
DROP = "#7A3E9D"       # deliberately distinct from both method curves
CONNECTOR = "#B9BEC3"
GRID = "#E2E6E9"
PRE_FILL = "#F2F5F7"
IMPACT_FILL = "#F7E8CF"
POST_FILL = "#EEF5EC"


def configure_journal_style(_allow_fallback: bool = True) -> None:
    """Use a portable sans-serif face and compact double-column styling."""
    candidates = ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
    family = "DejaVu Sans"
    for candidate in candidates:
        try:
            fm.findfont(fm.FontProperties(family=candidate),
                        fallback_to_default=False)
            family = candidate
            break
        except ValueError:
            pass
    plt.rcParams.update({
        "font.family": family,
        "mathtext.fontset": "dejavusans",
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.8,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#333333",
        "lines.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def load_extended_log(path: str | Path) -> dict:
    """Load the stable first 31 columns from any current log (>=37 columns)."""
    a = np.loadtxt(path, comments="#", ndmin=2, encoding="utf-8")
    if a.shape[1] < 37:
        raise ValueError(f"{path}: expected at least 37 columns; got {a.shape[1]}")
    d = {
        "path": str(path), "time": a[:, 0], "phase": a[:, 1].astype(int),
        "internal": a[:, 23], "external_corrected": a[:, 24],
        "external_raw": a[:, 25], "desired": a[:, 26],
        "slip": a[:, 27], "action": a[:, 28], "fc_logged": a[:, 29],
        "active": a[:, 30],
        "actual_closure": a[:, 37] if a.shape[1] >= 42
        else np.full(len(a), np.nan),
    }
    mask = d["phase"] == 2
    if np.count_nonzero(mask) < 5:
        raise ValueError(f"{path}: TEST phase contains too few samples")
    for key, value in tuple(d.items()):
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            d[key] = value[mask]
    d["time"] = d["time"] - d["time"][0]
    return d


def clean_axis(ax, before: float, after: float, xlabel: bool = False) -> None:
    ax.set_xlim(-before, after)
    ax.margins(x=0)
    ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel("Time relative to impact peak [s]")


def phase_background(ax, start: float, end: float, before: float,
                     after: float) -> None:
    ax.axvspan(-before, start, color=PRE_FILL, zorder=-4)
    ax.axvspan(start, end, color=IMPACT_FILL, zorder=-4)
    ax.axvspan(end, after, color=POST_FILL, zorder=-4)
    ax.axvline(0.0, color="#73777A", linestyle=":", linewidth=0.8, zorder=1)


def horizontal_drop(ax, when: float, text: str = "Object dropped") -> None:
    """Draw a high-contrast drop marker with horizontal text."""
    ax.axvline(when, color=DROP, linestyle=(0, (4, 2)), linewidth=1.05, zorder=5)
    lo, hi = ax.get_xlim()
    on_right = when > lo + 0.68 * (hi - lo)
    ax.annotate(text, xy=(when, 0.94), xycoords=ax.get_xaxis_transform(),
                xytext=(-4 if on_right else 4, 0), textcoords="offset points",
                ha="right" if on_right else "left", va="top", rotation=0,
                color=DROP, fontsize=7.0, fontweight="semibold",
                bbox={"boxstyle": "round,pad=0.14", "fc": "white",
                      "ec": "none", "alpha": 0.82}, zorder=6)


def time_history_figure(pairs, labels, out, mu, nominal, before, after,
                        slip_plot_limit_mm):
    """Figure 1: only disturbance, internal force, and slip (three rows)."""
    n = len(pairs)
    width = max(3.45, 7.16 * n / 3.0)
    fig, axes = plt.subplots(3, n, figsize=(width, 4.75), sharex="col",
                             squeeze=False)
    for col, ((sifr, difr), label) in enumerate(zip(pairs, labels)):
        start, end = base.collision_span(sifr, difr)
        mismatch = abs(sifr["peak_external"] - difr["peak_external"]) / max(
            0.5 * (sifr["peak_external"] + difr["peak_external"]), 1e-6)
        for ax in axes[:, col]:
            phase_background(ax, start, end, before, after)
            if sifr.get("dropped", False):
                ax.axvline(sifr["drop_time"], color=DROP,
                           linestyle=(0, (4, 2)), linewidth=1.05, zorder=5)
        if sifr.get("dropped", False):
            horizontal_drop(axes[0, col], sifr["drop_time"])

        ax = axes[0, col]
        for d, name, colour, style in ((difr, "DIFR", DIFR, "-"),
                                        (sifr, "SIFR", SIFR, "--")):
            t, y = base.crop(d, "external", before, after)
            ax.plot(t, y, color=colour, linestyle=style, label=name)
        suffix = " (unmatched)" if mismatch > 0.10 else ""
        ax.set_title(f"({chr(97 + col)}) {label}{suffix}\n"
                     rf"$F_{{E,pk}}$ S/D: {sifr['peak_external']:.2f}/"
                     f"{difr['peak_external']:.2f} N")
        if col == 0:
            ax.set_ylabel(r"$|F_E|$ [N]")

        ax = axes[1, col]
        for d, name, colour, style in ((difr, "DIFR measured", DIFR, "-"),
                                        (sifr, "SIFR measured", SIFR, "--")):
            t, y = base.crop(d, "internal", before, after)
            ax.plot(t, y, color=colour, linestyle=style, label=name)
        t, y = base.crop(difr, "desired", before, after)
        ax.plot(t, y, color=REFERENCE, linestyle=(0, (1.5, 2.0)),
                label=r"DIFR $F_I^{des}$")
        ax.axhline(nominal, color="#8B9297", linestyle="-.", linewidth=0.9,
                   label=rf"Nominal {nominal:g} N")
        if col == 0:
            ax.set_ylabel(r"$F_I$ [N]")

        ax = axes[2, col]
        for d, name, colour, style in ((difr, "DIFR", DIFR, "-"),
                                        (sifr, "SIFR", SIFR, "--")):
            t, y = base.crop(d, "slip", before, after)
            ax.plot(t, np.clip(1000 * y, -slip_plot_limit_mm,
                               slip_plot_limit_mm),
                    color=colour, linestyle=style, label=name)
        if col == 0:
            ax.set_ylabel(r"$\delta$ [mm]")

        for row, ax in enumerate(axes[:, col]):
            clean_axis(ax, before, after, xlabel=(row == 2))

    handles = [
        Line2D([0], [0], color=DIFR, label="DIFR"),
        Line2D([0], [0], color=SIFR, linestyle="--", label="SIFR"),
        Line2D([0], [0], color=REFERENCE, linestyle=":", label="DIFR desired"),
        Line2D([0], [0], color="#8B9297", linestyle="-.", label="Nominal force"),
        Patch(facecolor=PRE_FILL, edgecolor="none", label="Pre-impact"),
        Patch(facecolor=IMPACT_FILL, edgecolor="none", label="Impact"),
        Patch(facecolor=POST_FILL, edgecolor="none", label="Post-impact"),
    ]
    if any(s.get("dropped", False) for s, _ in pairs):
        handles.append(Line2D([0], [0], color=DROP, linestyle=(0, (4, 2)),
                              label="Object dropped"))
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center",
               bbox_to_anchor=(0.54, 0.995), columnspacing=1.05,
               handlelength=2.5)
    fig.subplots_adjust(left=.09, right=.995, bottom=.10, top=.84,
                        wspace=.22, hspace=.18)
    base.save_figure(fig, out, "fig1_paired_time_histories_journal")


def grouped_metric_figure(rows, labels, out, sifr_dropped,
                          slip_plot_limit_mm):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.25))
    metrics = (("peak_external_N", "Disturbance peak [N]"),
               ("peak_internal_N", "Peak internal force [N]"),
               ("peak_friction_utilization", "Peak friction utilization"),
               ("peak_estimated_slip_mm", "Peak estimated slip [mm]"))
    x = np.arange(len(labels))
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        sv = np.array([next(r[key] for r in rows if r["condition"] == lab
                            and r["method"] == "SIFR") for lab in labels])
        dv = np.array([next(r[key] for r in rows if r["condition"] == lab
                            and r["method"] == "DIFR") for lab in labels])
        raw_s, raw_d = sv.copy(), dv.copy()
        if key == "peak_estimated_slip_mm":
            sv = np.minimum(sv, slip_plot_limit_mm)
            dv = np.minimum(dv, slip_plot_limit_mm)
        for xi, s, d in zip(x, sv, dv):
            ax.plot([xi - .10, xi + .10], [s, d], color=CONNECTOR,
                    linewidth=1.15, zorder=1)
        ax.scatter(x - .10, sv, s=34, facecolor="white", edgecolor=SIFR,
                   linewidth=1.3, marker="o", label="SIFR", zorder=3)
        ax.scatter(x + .10, dv, s=34, facecolor=DIFR, edgecolor=DIFR,
                   linewidth=.8, marker="D", label="DIFR", zorder=3)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=2, loc="best")
        if key == "peak_friction_utilization":
            ax.axhline(1, color=REFERENCE, linestyle=":", linewidth=1)
        if key == "peak_estimated_slip_mm":
            for xi, lab, value, raw in zip(x, labels, sv, raw_s):
                if lab in sifr_dropped:
                    ax.annotate("Dropped", (xi - .10, value), xytext=(0, 7),
                                textcoords="offset points", ha="center",
                                rotation=0, color=DROP, fontsize=7.1,
                                fontweight="semibold")
                elif raw > slip_plot_limit_mm:
                    ax.annotate("overflow", (xi - .10, value), xytext=(0, 7),
                                textcoords="offset points", ha="center",
                                color=DROP, fontsize=6.8)
            for xi, value, raw in zip(x, dv, raw_d):
                if raw > slip_plot_limit_mm:
                    ax.annotate(rf"$>{slip_plot_limit_mm:g}$", (xi + .10, value),
                                xytext=(0, 7), textcoords="offset points",
                                ha="center", color=DROP, fontsize=6.8)
            ax.set_ylim(0, 1.18 * slip_plot_limit_mm)
    for ax, title in zip(axes.flat,
                         ("(a) Matched disturbance", "(b) Internal-force response",
                          "(c) Friction-cone margin", "(d) Slip-state response")):
        ax.set_title(title)
    fig.tight_layout(pad=.4, w_pad=.8, h_pad=.8)
    base.save_figure(fig, out, "fig2_cross_condition_metrics_journal")


def friction_margin_figure(pairs, labels, out, mu, before, after):
    n = len(pairs)
    width = max(3.45, 7.16 * n / 3.0)
    fig, axes = plt.subplots(1, n, figsize=(width, 2.25), sharex=True,
                             sharey=True, squeeze=False)
    all_values = []
    for col, ((sifr, difr), label) in enumerate(zip(pairs, labels)):
        ax = axes[0, col]
        for d, name, colour, style in ((difr, "DIFR", DIFR, "-"),
                                        (sifr, "SIFR", SIFR, "--")):
            margin = mu * np.abs(d["internal"]) - d["external"]
            d["friction_margin"] = margin
            all_values.append(margin)
            t, y = base.crop(d, "friction_margin", before, after)
            if d.get("dropped", False):
                y = y.copy()
                y[t >= d["drop_time"]] = np.nan
            ax.plot(t, y, color=colour, linestyle=style, label=name)
        ax.axhspan(0, 100, color=POST_FILL, zorder=-4)
        ax.axhspan(-100, 0, color="#FAEEEE", zorder=-4)
        ax.axhline(0, color=REFERENCE, linestyle=":", linewidth=1)
        ax.axvline(0, color="#73777A", linestyle=":", linewidth=.8)
        if sifr.get("dropped", False):
            horizontal_drop(ax, sifr["drop_time"], "Contact lost")
        ax.set_title(f"({chr(97 + col)}) {label}")
        clean_axis(ax, before, after, xlabel=True)
        if col == 0:
            ax.set_ylabel(r"Margin $\mu F_I-|F_E|$ [N]")
            ax.legend(frameon=False, loc="best")
    bound = max(.25, 1.1 * max(float(np.nanmax(np.abs(v))) for v in all_values))
    for ax in axes.flat:
        ax.set_ylim(-bound, bound)
    fig.tight_layout(pad=.4, w_pad=.55)
    base.save_figure(fig, out, "fig3_friction_margin_time_history_journal")


def install_overrides() -> None:
    base.PROPOSED = DIFR
    base.BASELINE = SIFR
    base.DESIRED = REFERENCE
    base.SAFE = SAFE
    base.COLLISION_FILL = IMPACT_FILL
    base.GRID = GRID
    base.require_times_new_roman = configure_journal_style
    base.load_log = load_extended_log
    base.style_axis = clean_axis
    base.time_history_figure = time_history_figure
    base.grouped_metric_figure = grouped_metric_figure
    base.friction_margin_figure = friction_margin_figure


def add_default_argument(flag: str, value: str | None = None) -> None:
    if flag not in sys.argv:
        sys.argv.append(flag)
        if value is not None:
            sys.argv.append(value)


if __name__ == "__main__":
    install_overrides()
    # Defaults reflect the current 0.45 kg, mu=0.4 experiment.  Zero plotting
    # smoothing keeps the measured collision peak intact; opt in explicitly
    # with --force-smooth-ms if a smoothed presentation is desired.
    add_default_argument("--mass", "0.45")
    add_default_argument("--mu", "0.4")
    add_default_argument("--nominal-force", "5.6")
    add_default_argument("--force-smooth-ms", "0")
    add_default_argument("--output-dir", "tase_collision_figures_journal")
    add_default_argument("--allow-font-fallback")
    base.main()
