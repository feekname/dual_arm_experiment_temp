#!/usr/bin/env python3
"""Generate T-ASE-ready DIFR/SIFR figures and a metrics CSV from 37-column logs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"


def configure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.7, "lines.linewidth": 1.1,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    })


def load_log(path):
    a = np.loadtxt(path, comments="#", ndmin=2, encoding="utf-8")
    if a.shape[1] != 37:
        raise ValueError(f"{path}: expected 37 columns, got {a.shape[1]}")
    d = {
        "path": str(path), "time": a[:, 0], "phase": a[:, 1].astype(int),
        "left_q": a[:, 2:9], "right_q": a[:, 9:16],
        "left_gmo": a[:, 16:19], "right_gmo": a[:, 19:22],
        "normal_signed": a[:, 22], "internal": a[:, 23],
        "external": a[:, 24], "external_sensor": a[:, 25],
        "desired": a[:, 26], "slip": a[:, 27], "roll": a[:, 28],
        "fc": a[:, 29], "active": a[:, 30], "wrench": a[:, 31:37],
    }
    mask = d["phase"] == 2
    if not np.any(mask):
        raise ValueError(f"{path}: no TEST samples (phase=2)")
    d["test_mask"] = mask
    d["test_time"] = d["time"][mask] - d["time"][mask][0]
    return d


def test(d, key):
    return d[key][d["test_mask"]]


def save_both(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def representative_figure(difr, sifr, out):
    fig, ax = plt.subplots(2, 2, figsize=(7.16, 4.8), sharex=True)
    td, ts = difr["test_time"], sifr["test_time"]

    ax[0, 0].plot(ts, np.abs(test(sifr, "external_sensor")), color=GRAY, label="SIFR")
    ax[0, 0].plot(td, np.abs(test(difr, "external_sensor")), color=BLUE, label="DIFR")
    ax[0, 0].set_ylabel(r"Tangential force $|F_E|$ [N]")

    ax[0, 1].plot(ts, test(sifr, "internal"), color=GRAY, label=r"SIFR $F_I$")
    ax[0, 1].plot(td, test(difr, "internal"), color=BLUE, label=r"DIFR $F_I$")
    ax[0, 1].plot(td, test(difr, "desired"), color=ORANGE, linestyle="--",
                  label=r"DIFR $F_I^{des}$")
    ax[0, 1].set_ylabel("Internal force [N]")

    ax[1, 0].plot(ts, test(sifr, "fc"), color=GRAY, label="SIFR")
    ax[1, 0].plot(td, test(difr, "fc"), color=BLUE, label="DIFR")
    ax[1, 0].axhline(1.0, color=ORANGE, linestyle="--", label="Friction limit")
    active = test(difr, "active") > 0.5
    ax[1, 0].fill_between(td, 0, 1, where=active, transform=ax[1, 0].get_xaxis_transform(),
                          color=BLUE, alpha=.10, label="DIFR active")
    ax[1, 0].set_ylabel(r"Friction utilization $f_c$")
    ax[1, 0].set_xlabel("TEST time [s]")

    ax[1, 1].plot(ts, 1000 * test(sifr, "slip"), color=GRAY, label="SIFR")
    ax[1, 1].plot(td, 1000 * test(difr, "slip"), color=BLUE, label="DIFR")
    ax[1, 1].set_ylabel(r"Estimated slip state $\delta$ [mm]")
    ax[1, 1].set_xlabel("TEST time [s]")

    titles = ("(a) Applied disturbance", "(b) Dynamic internal-force regulation",
              "(c) Friction-cone utilization", "(d) Estimated slip dynamics")
    for a, title in zip(ax.flat, titles):
        a.set_title(title); a.grid(True, linestyle=":", alpha=.45); a.legend(frameon=False)
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    save_both(fig, out, "fig1_time_history")


def friction_cone_figure(difr, sifr, out, mu):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), sharex=True, sharey=True)
    all_n = np.concatenate([test(sifr, "internal"), test(difr, "internal")])
    xmax = max(1.0, float(np.nanmax(all_n)) * 1.08)
    xline = np.linspace(0, xmax, 200)
    for a, d, name, color in ((axes[0], sifr, "SIFR", GRAY),
                               (axes[1], difr, "DIFR", BLUE)):
        n = test(d, "internal"); tang = np.abs(test(d, "external_sensor"))
        step = max(1, len(n) // 1200)
        sc = a.scatter(n[::step], tang[::step], c=d["test_time"][::step],
                       cmap="viridis", s=5, alpha=.65, linewidths=0)
        a.plot(xline, mu * xline, color=ORANGE, linestyle="--",
               label=rf"$|F_E|=\mu F_I$, $\mu={mu:g}$")
        a.fill_between(xline, 0, mu*xline, color=GREEN, alpha=.07)
        a.set_title(f"({chr(97 + (name == 'DIFR'))}) {name}")
        a.set_xlabel(r"Measured internal force $F_I$ [N]")
        a.grid(True, linestyle=":", alpha=.4); a.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel(r"Tangential force $|F_E|$ [N]")
    fig.colorbar(sc, ax=axes, label="TEST time [s]", fraction=.035, pad=.03)
    fig.subplots_adjust(left=.09, right=.90, bottom=.18, top=.90, wspace=.12)
    save_both(fig, out, "fig2_friction_cone")


def metrics(d, method, trial, mu):
    fi = test(d, "internal"); ft = np.abs(test(d, "external_sensor"))
    fc_measured = ft / np.maximum(mu * fi, 1e-6)
    return {
        "method": method, "trial": trial,
        "peak_tangential_N": float(np.max(ft)),
        "mean_internal_N": float(np.mean(fi)),
        "peak_internal_N": float(np.max(fi)),
        "peak_desired_internal_N": float(np.max(test(d, "desired"))),
        "peak_estimated_slip_mm": float(1000*np.max(np.abs(test(d, "slip")))),
        "friction_violation_percent": float(100*np.mean(fc_measured > 1.0)),
        "active_percent": float(100*np.mean(test(d, "active") > .5)),
    }


def aggregate_figure(rows, out):
    methods = ("SIFR", "DIFR")
    keys = (("peak_estimated_slip_mm", "Peak estimated slip [mm]"),
            ("friction_violation_percent", "Friction-limit violation [%]"),
            ("peak_internal_N", "Peak measured internal force [N]"))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    for a, (key, ylabel) in zip(axes, keys):
        values = [[r[key] for r in rows if r["method"] == m] for m in methods]
        means = [np.mean(v) if v else np.nan for v in values]
        stds = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in values]
        a.bar(methods, means, yerr=stds, color=[GRAY, BLUE], width=.62,
              capsize=3, edgecolor="black", linewidth=.5)
        for i, vals in enumerate(values):
            a.scatter(np.full(len(vals), i), vals, color="black", s=8, zorder=3)
        a.set_ylabel(ylabel); a.grid(axis="y", linestyle=":", alpha=.4)
    fig.tight_layout(w_pad=1.2)
    save_both(fig, out, "fig3_aggregate_metrics")


def main():
    p = argparse.ArgumentParser(description="T-ASE DIFR/SIFR publication figures")
    p.add_argument("--difr", nargs="+", required=True, help="one or more DIFR logs")
    p.add_argument("--sifr", nargs="+", required=True, help="one or more SIFR logs")
    p.add_argument("--mu", type=float, required=True)
    p.add_argument("--output-dir", default="tase_figures")
    args = p.parse_args()
    configure_style()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    difr = [load_log(x) for x in args.difr]
    sifr = [load_log(x) for x in args.sifr]
    representative_figure(difr[0], sifr[0], out)
    friction_cone_figure(difr[0], sifr[0], out, args.mu)
    rows = []
    rows += [metrics(d, "DIFR", Path(d["path"]).stem, args.mu) for d in difr]
    rows += [metrics(d, "SIFR", Path(d["path"]).stem, args.mu) for d in sifr]
    aggregate_figure(rows, out)
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Publication figures and metrics saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
