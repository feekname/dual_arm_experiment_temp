#!/usr/bin/env python3
"""Plot the 37-column single-left-FT-sensor SIFR/DIFR experiment logs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def load_log(path):
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != 37:
        raise ValueError(
            f"{path}: 需要新版单传感器日志37列，实际为{data.shape[1]}列"
        )
    return {
        "time": data[:, 0], "phase": data[:, 1].astype(int),
        "left_q": data[:, 2:9], "right_q": data[:, 9:16],
        "left_gmo": data[:, 16:19], "right_gmo": data[:, 19:22],
        "normal_signed": data[:, 22], "internal": data[:, 23],
        "external": data[:, 24], "external_sensor": data[:, 25],
        "desired": data[:, 26], "slip": data[:, 27],
        "roll": data[:, 28], "fc": data[:, 29], "active": data[:, 30],
        "wrench": data[:, 31:37],
    }


def shade_test(ax, d):
    mask = d["phase"] == 2
    if np.any(mask):
        t = d["time"][mask]
        ax.axvspan(t[0], t[-1], color="#ffe7e7", alpha=0.45, label="TEST")


def plot_one(d, label, output):
    t = d["time"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    ax = axes[0, 0]
    for i, name in enumerate(AXES[:3]):
        ax.plot(t, d["wrench"][:, i], label=name, linewidth=1)
    shade_test(ax, d); ax.set_ylabel("Force [N]"); ax.legend(ncol=4); ax.grid(alpha=.25)
    ax.set_title("Raw left-hand force: use this panel to choose axes/signs")

    ax = axes[0, 1]
    for i, name in enumerate(AXES[3:], 3):
        ax.plot(t, d["wrench"][:, i], label=name, linewidth=1)
    shade_test(ax, d); ax.set_ylabel("Moment [N m]"); ax.legend(ncol=3); ax.grid(alpha=.25)
    ax.set_title("Raw left-hand moment")

    ax = axes[1, 0]
    ax.plot(t, d["normal_signed"], label="signed normal", alpha=.65)
    ax.plot(t, d["internal"], label="measured $F_I$")
    ax.plot(t, d["desired"], label="desired $F_I$", linewidth=2)
    shade_test(ax, d); ax.set_ylabel("Force [N]"); ax.legend(); ax.grid(alpha=.25)
    ax.set_title("Internal-force tracking")

    ax = axes[1, 1]
    ax.plot(t, d["external_sensor"], label="signed tangential force")
    ax.plot(t, np.linalg.norm(d["left_gmo"], axis=1), label="left GMO norm", alpha=.7)
    shade_test(ax, d); ax.set_ylabel("Force [N]"); ax.legend(); ax.grid(alpha=.25)
    ax.set_title("Disturbance and collision trigger")

    ax = axes[2, 0]
    ax.plot(t, 1000*d["slip"], label=r"estimated $\delta$")
    ax2 = ax.twinx(); ax2.plot(t, d["fc"], color="tab:red", label="$f_c$")
    ax2.axhline(1, color="tab:red", linestyle="--", alpha=.5)
    shade_test(ax, d); ax.set_ylabel("Slip [mm]"); ax2.set_ylabel("Friction ratio")
    ax.grid(alpha=.25); ax.set_title("Eq. (39)-(41) states")

    ax = axes[2, 1]
    ax.plot(t, np.degrees(d["roll"]), label="roll correction")
    ax.fill_between(t, 0, d["active"], step="mid", alpha=.25, label="DIFR active")
    shade_test(ax, d); ax.set_ylabel("Correction [deg]"); ax.legend(); ax.grid(alpha=.25)
    ax.set_title("Controller action")

    for ax in axes[-1, :]: ax.set_xlabel("Time [s]")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_comparison(difr, sifr, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    items = (("desired", "Desired internal force [N]"),
             ("internal", "Measured internal force [N]"),
             ("slip", "Estimated slip [m]"), ("fc", "Friction ratio"))
    for ax, (key, ylabel) in zip(axes.flat, items):
        ax.plot(difr["time"], difr[key], label="DIFR")
        ax.plot(sifr["time"], sifr[key], label="SIFR", alpha=.8)
        ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend()
    for ax in axes[-1, :]: ax.set_xlabel("Time [s]")
    fig.tight_layout(); fig.savefig(output, dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="单左手六维力传感器实验调试绘图")
    parser.add_argument("--difr")
    parser.add_argument("--sifr")
    parser.add_argument("--output-dir", default="force_debug_plots")
    args = parser.parse_args()
    if not args.difr and not args.sifr:
        parser.error("至少提供 --difr 或 --sifr")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    difr = load_log(args.difr) if args.difr else None
    sifr = load_log(args.sifr) if args.sifr else None
    if difr is not None: plot_one(difr, "DIFR single-sensor debug", out/"difr_debug.png")
    if sifr is not None: plot_one(sifr, "SIFR single-sensor debug", out/"sifr_debug.png")
    if difr is not None and sifr is not None:
        plot_comparison(difr, sifr, out/"difr_vs_sifr.png")
    print(f"图像已保存到: {out.resolve()}")


if __name__ == "__main__":
    main()
