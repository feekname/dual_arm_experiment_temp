#!/usr/bin/env python3
"""T-ASE-ready paired SIFR/DIFR collision figures for 3+ impact levels."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


# Match the visual language already used in Figs. 7-10 of the manuscript.
PROPOSED = "#1f77b4"       # blue solid: DIFR / proposed
BASELINE = "#d62728"       # red dashed: SIFR / baseline
DESIRED = "#555555"        # dark gray dashed: desired/reference
SAFE = "#2ca02c"           # green: safe/static-friction region
COLLISION_FILL = "#f2d9e6" # pale pink: collision interval
GRID = "#d9d9d9"


def require_times_new_roman(allow_fallback=False):
    try:
        fm.findfont(fm.FontProperties(family="Times New Roman"),
                    fallback_to_default=False)
        family = "Times New Roman"
    except ValueError:
        if not allow_fallback:
            raise RuntimeError(
                "Times New Roman is not installed. Install msttcorefonts, rebuild "
                "the Matplotlib font cache, or pass --allow-font-fallback for debugging.")
        family = "Liberation Serif"
        print("[warning] Times New Roman unavailable; using Liberation Serif.")
    plt.rcParams.update({
        "font.family": family,
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def load_log(path):
    a = np.loadtxt(path, comments="#", ndmin=2, encoding="utf-8")
    if a.shape[1] not in (37, 41, 42):
        raise ValueError(f"{path}: expected 37, 41, or 42 columns; got {a.shape[1]}")
    d = {
        "path": str(path), "time": a[:, 0], "phase": a[:, 1].astype(int),
        "internal": a[:, 23], "external_corrected": a[:, 24],
        "external_raw": a[:, 25], "desired": a[:, 26],
        "slip": a[:, 27], "action": a[:, 28], "fc_logged": a[:, 29],
        "active": a[:, 30],
    }
    d["actual_closure"] = a[:, 37] if a.shape[1] >= 41 else np.full(len(a), np.nan)
    mask = d["phase"] == 2
    if np.count_nonzero(mask) < 5:
        raise ValueError(f"{path}: TEST phase contains too few samples")
    for key in tuple(d):
        if isinstance(d[key], np.ndarray) and len(d[key]) == len(mask):
            d[key] = d[key][mask]
    d["time"] = d["time"] - d["time"][0]
    return d


def moving_average(x, samples):
    samples = max(1, int(samples))
    if samples == 1:
        return x.copy()
    kernel = np.ones(samples) / samples
    left = samples // 2
    right = samples - 1 - left
    padded = np.pad(x, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def prepare_event(d, baseline_seconds, smooth_ms, impact_fraction):
    t = d["time"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    # New logs already store a GRASP-tared signal in column 24. Subtracting the
    # initial TEST median is harmless and also repairs older logs.
    early = t <= min(baseline_seconds, max(0.2, 0.2*t[-1]))
    baseline = float(np.median(d["external_corrected"][early]))
    signed = d["external_corrected"] - baseline
    samples = max(1, round((smooth_ms/1000.0) / max(dt, 1e-6)))
    filtered = moving_average(signed, samples)
    magnitude = np.abs(filtered)
    peak_index = int(np.nanargmax(magnitude))
    peak = float(magnitude[peak_index])
    rel = t - t[peak_index]

    threshold = max(0.15, impact_fraction * peak)
    above = magnitude >= threshold
    # Bridge brief gaps between peaks from the same physical push. The former
    # strictly-contiguous rule could reduce an impulsive trial to one sample.
    max_gap = max(1, round(0.15/max(dt, 1e-6)))
    indices = np.flatnonzero(above)
    left_candidates = indices[indices <= peak_index]
    right_candidates = indices[indices >= peak_index]
    lo = peak_index
    last = peak_index
    for idx in left_candidates[::-1]:
        if last - idx > max_gap:
            break
        lo = int(idx); last = int(idx)
    hi = peak_index
    last = peak_index
    for idx in right_candidates:
        if idx - last > max_gap:
            break
        hi = int(idx); last = int(idx)
    d.update({
        "external": magnitude, "external_signed": filtered,
        "relative_time": rel, "peak_external": peak,
        "impact_start": float(rel[lo]), "impact_end": float(rel[hi]),
        "impact_mask": np.arange(len(t)) >= lo,
        "collision_mask": (np.arange(len(t)) >= lo) & (np.arange(len(t)) <= hi),
        "post_mask": np.arange(len(t)) > hi,
        "dt": dt, "external_baseline": baseline,
    })
    return d


def crop(d, key, before, after):
    mask = (d["relative_time"] >= -before) & (d["relative_time"] <= after)
    return d["relative_time"][mask], d[key][mask]


def collision_span(a, b):
    return min(a["impact_start"], b["impact_start"]), max(a["impact_end"], b["impact_end"])


def mark_declared_drops(pairs, labels, dropped_labels, nominal,
                        force_fraction=0.30, hold_seconds=0.25):
    """Attach a drop time to declared SIFR trials.

    The declaration is treated as ground truth.  The force signal is used only
    to place the marker: first sustained post-impact loss of normal force.
    """
    dropped_labels = set(dropped_labels)
    for (sifr, difr), label in zip(pairs, labels):
        for d in (sifr, difr):
            d["dropped"] = False
            d["drop_time"] = np.nan
        if label not in dropped_labels:
            continue
        sifr["dropped"] = True
        rel = sifr["relative_time"]
        low = (rel >= 0.0) & (np.abs(sifr["internal"]) < force_fraction*nominal)
        hold_n = max(1, int(round(hold_seconds/max(sifr["dt"], 1e-6))))
        run = 0
        for i, flag in enumerate(low):
            run = run + 1 if flag else 0
            if run >= hold_n:
                sifr["drop_time"] = float(rel[i-hold_n+1])
                break
        if not np.isfinite(sifr["drop_time"]):
            # A declared drop is still shown even if normal force does not give
            # an unambiguous transition (e.g. sensor unloads slowly).
            sifr["drop_time"] = max(0.0, float(sifr["impact_end"]))


def style_axis(ax, before, after, xlabel=False):
    ax.set_xlim(-before, after)
    ax.margins(x=0)
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.55)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel("Time relative to impact peak [s]")


def save_figure(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def time_history_figure(pairs, labels, out, mu, nominal, before, after,
                        slip_plot_limit_mm):
    n = len(pairs)
    fig, axes = plt.subplots(4, n, figsize=(7.16, 6.4), sharex="col", squeeze=False)
    for col, ((sifr, difr), label) in enumerate(zip(pairs, labels)):
        span = collision_span(sifr, difr)
        mismatch = abs(sifr["peak_external"]-difr["peak_external"]) / max(
            .5*(sifr["peak_external"]+difr["peak_external"]), 1e-6)
        for row in range(4):
            axes[row, col].axvspan(span[0], span[1], color=COLLISION_FILL,
                                  zorder=0)
            axes[row, col].axvline(0.0, color="#777777", linestyle=":", linewidth=0.8)
            if sifr.get("dropped", False):
                drop_t = sifr["drop_time"]
                axes[row, col].axvspan(drop_t, after, facecolor="#eeeeee",
                                      edgecolor="#aaaaaa", hatch="////",
                                      alpha=.55, linewidth=0, zorder=.2)
        if sifr.get("dropped", False):
            axes[0, col].text(sifr["drop_time"]+.04, .96, "Object dropped",
                              transform=axes[0, col].get_xaxis_transform(),
                              va="top", ha="left", rotation=90,
                              color=BASELINE, fontsize=7.2)

        ax = axes[0, col]
        for d, name, color, ls in ((difr, "DIFR", PROPOSED, "-"),
                                    (sifr, "SIFR", BASELINE, "--")):
            t, y = crop(d, "external", before, after)
            ax.plot(t, y, color=color, linestyle=ls, label=name)
        warning = " - unmatched" if mismatch > .10 else ""
        ax.set_title(f"({chr(97+col)}) {label}{warning}\n"
                     f"$F_{{E,pk}}$ S/D: {sifr['peak_external']:.2f}/"
                     f"{difr['peak_external']:.2f} N")
        ax.set_ylabel(r"$|F_E|$ [N]" if col == 0 else "")

        ax = axes[1, col]
        for d, name, color, ls in ((difr, "DIFR measured", PROPOSED, "-"),
                                    (sifr, "SIFR measured", BASELINE, "--")):
            t, y = crop(d, "internal", before, after)
            ax.plot(t, y, color=color, linestyle=ls, label=name)
        t, y = crop(difr, "desired", before, after)
        ax.plot(t, y, color=DESIRED, linestyle=":", label=r"DIFR $F_I^{des}$")
        ax.axhline(nominal, color="#999999", linestyle="-.", linewidth=0.9,
                   label=rf"nominal {nominal:g} N")
        ax.set_ylabel(r"$F_I$ [N]" if col == 0 else "")

        ax = axes[2, col]
        for d, name, color, ls in ((difr, "DIFR", PROPOSED, "-"),
                                    (sifr, "SIFR", BASELINE, "--")):
            t, y = crop(d, "slip", before, after)
            y_mm = 1000*y
            ax.plot(t, np.clip(y_mm, -slip_plot_limit_mm, slip_plot_limit_mm),
                    color=color, linestyle=ls, label=name)
        ax.set_ylabel(r"$\delta$ [mm]" if col == 0 else "")

        ax = axes[3, col]
        for d, name, color, ls in ((difr, "DIFR", PROPOSED, "-"),
                                    (sifr, "SIFR", BASELINE, "--")):
            measured_fc = d["external"] / np.maximum(mu*np.abs(d["internal"]), 1e-6)
            d["fc_measured"] = measured_fc
            t, y = crop(d, "fc_measured", before, after)
            ax.plot(t, y, color=color, linestyle=ls, label=name)
        ax.axhline(1.0, color=DESIRED, linestyle=":", label="friction boundary")
        ax.set_ylabel(r"$|F_E|/(\mu F_I)$" if col == 0 else "")

        for row in range(4):
            style_axis(axes[row, col], before, after, xlabel=(row == 3))

    legend_handles = [
        Line2D([0], [0], color=PROPOSED, linestyle="-", label="DIFR"),
        Line2D([0], [0], color=BASELINE, linestyle="--", label="SIFR"),
        Line2D([0], [0], color=DESIRED, linestyle=":", label="DIFR desired / boundary"),
        Line2D([0], [0], color="#999999", linestyle="-.", label="Nominal force"),
        Patch(facecolor=COLLISION_FILL, edgecolor="none", label="Impact interval"),
    ]
    if any(s.get("dropped", False) for s, _ in pairs):
        legend_handles.append(Patch(facecolor="#eeeeee", edgecolor="#aaaaaa",
                                    hatch="////", label="Contact lost / dropped"))
    fig.legend(handles=legend_handles, ncol=5, frameon=False,
               loc="upper center", bbox_to_anchor=(.54, .995),
               columnspacing=1.1, handlelength=2.4)
    fig.subplots_adjust(left=.085, right=.995, bottom=.075, top=.88,
                        wspace=.20, hspace=.18)
    save_figure(fig, out, "fig1_paired_time_histories")


def metric_row(d, method, condition, mu):
    c = d["collision_mask"]
    fc = d["external"] / np.maximum(mu*np.abs(d["internal"]), 1e-6)
    return {
        "condition": condition, "method": method,
        "peak_external_N": d["peak_external"],
        "impact_impulse_Ns": float(np.trapz(d["external"][c], d["time"][c])),
        "peak_internal_N": float(np.max(d["internal"][c])),
        "peak_desired_internal_N": float(np.max(d["desired"][c])),
        "peak_friction_utilization": float(np.max(fc[c])),
        "friction_violation_ms": float(1000*np.count_nonzero(fc[c] > 1.0)*d["dt"]),
        "peak_estimated_slip_mm": float(1000*np.max(np.abs(d["slip"][c]))),
        "object_dropped": int(bool(d.get("dropped", False))),
        "baseline_removed_N": d["external_baseline"],
    }


def grouped_metric_figure(rows, labels, out, sifr_dropped, slip_plot_limit_mm):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.4))
    metrics = (("peak_external_N", "Measured disturbance peak [N]"),
               ("peak_internal_N", "Peak internal force [N]"),
               ("peak_friction_utilization", "Peak friction utilization"),
               ("peak_estimated_slip_mm", "Peak estimated slip [mm]"))
    x = np.arange(len(labels))
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        s = [next(r[key] for r in rows if r["condition"] == lab and r["method"] == "SIFR")
             for lab in labels]
        d = [next(r[key] for r in rows if r["condition"] == lab and r["method"] == "DIFR")
             for lab in labels]
        raw_s, raw_d = list(s), list(d)
        if key == "peak_estimated_slip_mm":
            s = np.minimum(s, slip_plot_limit_mm)
            d = np.minimum(d, slip_plot_limit_mm)
        # Paired dumbbell plot is visually lighter than bars and makes unfair
        # disturbance matching immediately visible.
        for xi, sv, dv in zip(x, s, d):
            ax.plot([xi-.10, xi+.10], [sv, dv], color="#bdbdbd",
                    linewidth=1.2, zorder=1)
        ax.scatter(x-.10, s, s=34, facecolor="white", edgecolor=BASELINE,
                   linewidth=1.3, marker="o", label="SIFR", zorder=3)
        ax.scatter(x+.10, d, s=34, facecolor=PROPOSED, edgecolor=PROPOSED,
                   linewidth=.8, marker="D", label="DIFR", zorder=3)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.margins(x=0.03)
        ax.grid(axis="y", color=GRID, linestyle="--", linewidth=.55)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=2, loc="best")
        if key == "peak_friction_utilization":
            ax.axhline(1.0, color=DESIRED, linestyle=":", linewidth=1.0)
        if key == "peak_estimated_slip_mm":
            for xi, lab, value, raw in zip(x, labels, s, raw_s):
                if lab in sifr_dropped:
                    ax.annotate("Dropped", (xi-.10, value), xytext=(0, 7),
                                textcoords="offset points", ha="center",
                                color=BASELINE, fontsize=7.5, fontstyle="italic")
                elif raw > slip_plot_limit_mm:
                    ax.annotate("estimator overflow", (xi-.10, value), xytext=(0, 7),
                                textcoords="offset points", ha="center",
                                color=BASELINE, fontsize=6.8, fontstyle="italic")
            for xi, value, raw in zip(x, d, raw_d):
                if raw > slip_plot_limit_mm:
                    ax.annotate(rf"$>{slip_plot_limit_mm:g}$", (xi+.10, value),
                                xytext=(0, 7), textcoords="offset points",
                                ha="center", color=PROPOSED, fontsize=7)
            ax.set_ylim(bottom=0, top=1.18*slip_plot_limit_mm)
    titles = ("(a) Matched disturbance", "(b) Internal-force response",
              "(c) Friction-cone margin", "(d) Slip-state response")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
    fig.tight_layout(pad=.35, w_pad=.8, h_pad=.7)
    save_figure(fig, out, "fig2_cross_condition_metrics")


def friction_margin_figure(pairs, labels, out, mu, before, after):
    """Directly show remaining friction capacity; positive is safe."""
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(7.16, 2.25), sharex=True,
                             sharey=True, squeeze=False)
    for col, ((sifr, difr), label) in enumerate(zip(pairs, labels)):
        ax = axes[0, col]
        ax.axhspan(0, 100, color="#edf6ed", zorder=0)
        ax.axhspan(-100, 0, color="#f9e5e5", zorder=0)
        for d, name, color, ls in ((difr, "DIFR", PROPOSED, "-"),
                                    (sifr, "SIFR", BASELINE, "--")):
            margin = mu*np.abs(d["internal"]) - d["external"]
            d["friction_margin"] = margin
            t, y = crop(d, "friction_margin", before, after)
            if d.get("dropped", False):
                y = y.copy()
                y[t >= d["drop_time"]] = np.nan
            ax.plot(t, y, color=color, linestyle=ls, label=name)
        if sifr.get("dropped", False):
            ax.axvspan(sifr["drop_time"], after, facecolor="#eeeeee",
                       edgecolor="#aaaaaa", hatch="////", alpha=.65,
                       linewidth=0, zorder=.2)
            ax.text(sifr["drop_time"]+.04, .96, "contact lost",
                    transform=ax.get_xaxis_transform(), va="top", ha="left",
                    rotation=90, color=BASELINE, fontsize=7)
        ax.axhline(0, color=DESIRED, linestyle=":", linewidth=1.0)
        ax.axvline(0, color="#777777", linestyle=":", linewidth=.8)
        ax.set_title(f"({chr(97+col)}) {label}")
        style_axis(ax, before, after, xlabel=True)
        if col == 0:
            ax.set_ylabel(r"Margin $\mu F_I-|F_E|$ [N]")
            ax.legend(frameon=False, loc="best")
    # Apply finite shared bounds instead of the temporary +/-100 fill limits.
    values = [mu*np.abs(d["internal"]) - d["external"] for pair in pairs for d in pair]
    bound = max(.25, 1.10*max(float(np.nanmax(np.abs(v))) for v in values))
    for ax in axes.flat:
        ax.set_ylim(-bound, bound)
    fig.tight_layout(pad=.35, w_pad=.55)
    save_figure(fig, out, "fig3_friction_margin_time_history")


def force_space_figure(pairs, labels, out, mu):
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(7.16, 3.8), squeeze=False,
                             sharex=True, sharey=True)
    all_internal = np.concatenate([np.abs(d["internal"]) for pair in pairs for d in pair])
    xmax = max(1.0, float(np.nanmax(all_internal))*1.08)
    xline = np.linspace(0, xmax, 200)
    for col, ((sifr, difr), label) in enumerate(zip(pairs, labels)):
        for row, (d, name) in enumerate(((sifr, "SIFR"), (difr, "DIFR"))):
            ax = axes[row, col]
            fi = np.abs(d["internal"]); fe = d["external"]
            masks = ((~d["impact_mask"], "Pre-impact", "#9e9e9e"),
                     (d["collision_mask"], "Impact", BASELINE),
                     (d["post_mask"], "Post-impact", SAFE))
            step = max(1, len(fi)//1000)
            for mask, phase, color in masks:
                if d.get("dropped", False):
                    mask = mask & (d["relative_time"] < d["drop_time"])
                idx = np.flatnonzero(mask)[::step]
                ax.scatter(fi[idx], fe[idx], s=7, color=color, alpha=.58,
                           linewidths=0, label=phase if col == 0 else None)
            if d.get("dropped", False):
                k = int(np.nanargmin(np.abs(d["relative_time"]-d["drop_time"])))
                ax.scatter(fi[k], fe[k], s=30, marker="x", color="black",
                           linewidths=1.0, zorder=5,
                           label="Contact lost" if col == 0 else None)
            ax.plot(xline, mu*xline, color=DESIRED, linestyle="--", linewidth=1.1,
                    label=rf"$|F_E|=\mu F_I$" if col == 0 else None)
            ax.fill_between(xline, 0, mu*xline, color="#e6f2e6", zorder=-1)
            ax.set_xlim(0, xmax); ax.set_ylim(bottom=0); ax.margins(x=0)
            ax.grid(True, color=GRID, linestyle="--", linewidth=.5)
            ax.set_title(f"({chr(97 + row*n + col)}) {name}, {label}")
            if col == 0:
                ax.set_ylabel(r"Tangential force $|F_E|$ [N]")
                ax.legend(frameon=False, loc="upper left")
            if row == 1:
                ax.set_xlabel(r"Internal force $F_I$ [N]")
    fig.tight_layout(pad=.35, w_pad=.5, h_pad=.55)
    save_figure(fig, out, "fig4_force_space_friction_cone")


def main():
    ap = argparse.ArgumentParser(
        description="Publication figures for paired SIFR/DIFR impact levels")
    ap.add_argument("--sifr", nargs="+", required=True,
                    help="SIFR logs ordered from low to high impact")
    ap.add_argument("--difr", nargs="+", required=True,
                    help="paired DIFR logs in the same order")
    ap.add_argument("--labels", nargs="+", help="condition labels, e.g. Low Medium High")
    ap.add_argument("--sifr-dropped", nargs="*", default=[], metavar="LABEL",
                    help="condition labels where the SIFR trial dropped the object")
    ap.add_argument("--drop-force-fraction", type=float, default=0.30,
                    help="normal-force fraction used to place declared drop marker")
    ap.add_argument("--drop-hold", type=float, default=0.25,
                    help="required sustained force loss [s] for drop marker")
    ap.add_argument("--slip-plot-limit-mm", type=float, default=50.0,
                    help="display limit for the slip estimator; raw values stay in CSV")
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--mass", type=float, default=0.2)
    ap.add_argument("--nominal-force", type=float, default=3.0)
    ap.add_argument("--before", type=float, default=1.5)
    ap.add_argument("--after", type=float, default=4.0)
    ap.add_argument("--baseline-seconds", type=float, default=1.0)
    ap.add_argument("--force-smooth-ms", type=float, default=50.0)
    ap.add_argument("--impact-fraction", type=float, default=0.2,
                    help="fraction of peak used to delimit the impact interval")
    ap.add_argument("--output-dir", default="tase_collision_figures")
    ap.add_argument("--allow-font-fallback", action="store_true")
    args = ap.parse_args()

    if len(args.sifr) != len(args.difr):
        ap.error("--sifr and --difr must contain the same number of paired logs")
    if len(args.sifr) < 3:
        ap.error("at least three paired impact conditions are required")
    labels = args.labels or [f"T{i+1}" for i in range(len(args.sifr))]
    if len(labels) != len(args.sifr):
        ap.error("--labels count must match the number of pairs")
    if args.mu <= 0 or args.mass <= 0 or args.nominal_force <= 0:
        ap.error("--mu, --mass, and --nominal-force must be positive")

    require_times_new_roman(args.allow_font_fallback)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for s_path, d_path in zip(args.sifr, args.difr):
        s = prepare_event(load_log(s_path), args.baseline_seconds,
                          args.force_smooth_ms, args.impact_fraction)
        d = prepare_event(load_log(d_path), args.baseline_seconds,
                          args.force_smooth_ms, args.impact_fraction)
        pairs.append((s, d))

    unknown_drops = set(args.sifr_dropped) - set(labels)
    if unknown_drops:
        ap.error("--sifr-dropped contains unknown labels: " + ", ".join(sorted(unknown_drops)))
    mark_declared_drops(pairs, labels, args.sifr_dropped, args.nominal_force,
                        args.drop_force_fraction, args.drop_hold)

    time_history_figure(pairs, labels, out, args.mu, args.nominal_force,
                        args.before, args.after, args.slip_plot_limit_mm)
    rows = []
    for (s, d), label in zip(pairs, labels):
        rows.extend((metric_row(s, "SIFR", label, args.mu),
                     metric_row(d, "DIFR", label, args.mu)))
    grouped_metric_figure(rows, labels, out, set(args.sifr_dropped),
                          args.slip_plot_limit_mm)
    friction_margin_figure(pairs, labels, out, args.mu, args.before, args.after)
    force_space_figure(pairs, labels, out, args.mu)

    with open(out / "collision_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    theoretical = args.mass * 9.81 / (2.0 * args.mu)
    print(f"Theoretical per-contact minimum mg/(2mu) = {theoretical:.3f} N")
    print(f"Configured nominal internal force = {args.nominal_force:.3f} N")
    logged_nominals = [float(np.median(d["desired"][d["relative_time"] < -0.2]))
                       for pair in pairs for d in pair
                       if np.any(d["relative_time"] < -0.2)]
    if logged_nominals and abs(float(np.median(logged_nominals))-args.nominal_force) > .15:
        print("[warning] --nominal-force differs from the logged pre-impact desired force: "
              f"{np.median(logged_nominals):.3f} N")
    for (s, d), label in zip(pairs, labels):
        mismatch = abs(s["peak_external"]-d["peak_external"]) / max(
            0.5*(s["peak_external"]+d["peak_external"]), 1e-6)
        flag = "  [repeat recommended: >10% mismatch]" if mismatch > 0.10 else ""
        print(f"{label}: impact peaks SIFR/DIFR = "
              f"{s['peak_external']:.3f}/{d['peak_external']:.3f} N{flag}")
    print(f"Saved publication figures and metrics to: {out.resolve()}")


if __name__ == "__main__":
    main()
