#!/usr/bin/env python3
"""Plot logs produced by difr_force_experiment.py/sifr_force_experiment.py.

The script only reads experiment logs. It never modifies or interpolates the
recorded values. By default it plots the TEST phase with time reset to 0 s.

Examples
--------
Plot one DIFR experiment (default TEST phase, first 15 s)::

    python3 plot_force_experiment_results_v2.py \
        --difr difr_debug_low.txt --output-dir figures_low

Plot a paired DIFR/SIFR experiment and generate a comparison figure::

    python3 plot_force_experiment_results_v2.py \
        --difr difr_high.txt --sifr sifr_high.txt \
        --output-dir figures_high

Plot the whole three-stage experiment::

    python3 plot_force_experiment_results_v2.py \
        --difr difr_high.txt --phase all --duration 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COL = {
    "time": 0,
    "phase": 1,
    "normal_force": 22,
    "internal": 23,
    "external": 24,
    "external_raw": 25,
    "desired": 26,
    "slip": 27,
    "action": 28,
    "friction_ratio": 29,
    "active": 30,
    "actual_closure": 37,
    "closure_limit": 38,
    "left_joint_rmse": 39,
    "right_joint_rmse": 40,
    "sent_closure": 41,
}

PHASE_NUMBER = {"move": 0, "grasp": 1, "test": 2}
PHASE_STYLE = {
    0: ("MOVE", "#DCEAF7"),
    1: ("GRASP", "#FFF0CE"),
    2: ("TEST", "#FBE1E4"),
}

COLORS = {
    "external": "#E07A5F",
    "external_raw": "#9C9C9C",
    "qp": "#7A5195",
    "desired": "#F28E2B",
    "measured": "#277DA1",
    "slip": "#43AA8B",
    "slip_rate": "#D1495B",
    "difr": "#277DA1",
    "sifr": "#E07A5F",
}


def read_metadata(path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if ":" in text:
                key, value = text.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
    return metadata


def infer_method(path: Path, metadata: Dict[str, str]) -> str:
    description = " ".join(metadata.values()).upper()
    filename = path.name.upper()
    if "DIFR" in description or "DIFR" in filename:
        return "DIFR"
    if "SIFR" in description or "SIFR" in filename:
        return "SIFR"
    return "UNKNOWN"


def load_log(path_like: str) -> Dict[str, np.ndarray]:
    path = Path(path_like).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment log not found: {path}")

    metadata = read_metadata(path)
    data = np.loadtxt(path, comments="#", ndmin=2, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"No numeric samples in: {path}")
    if data.shape[1] < 37:
        raise ValueError(
            f"{path.name}: expected at least 37 columns, got {data.shape[1]}"
        )

    method = infer_method(path, metadata)
    sample_count = data.shape[0]
    nan = np.full(sample_count, np.nan)

    result: Dict[str, np.ndarray] = {
        key: data[:, index] if index < data.shape[1] else nan.copy()
        for key, index in COL.items()
    }
    result["phase"] = np.rint(result["phase"]).astype(int)

    # New logs append columns without moving any of the original columns:
    #   DIFR (44 columns): col 42 = raw QP force, col 43 = slip rate
    #   SIFR (43 columns): col 42 = slip rate
    if method == "DIFR" and data.shape[1] >= 44:
        result["qp_raw"] = data[:, 42]
        result["slip_rate"] = data[:, 43]
    elif method == "SIFR" and data.shape[1] >= 43:
        result["qp_raw"] = nan.copy()
        result["slip_rate"] = data[:, 42]
    else:
        # Old logs remain plottable, but missing channels stay absent rather
        # than being reconstructed or fabricated.
        result["qp_raw"] = nan.copy()
        result["slip_rate"] = nan.copy()

    result["path"] = path  # type: ignore[assignment]
    result["method"] = method  # type: ignore[assignment]
    result["metadata"] = metadata  # type: ignore[assignment]
    return result


def select_interval(
    log: Dict[str, np.ndarray], phase_name: str, duration: float
) -> Dict[str, np.ndarray]:
    if phase_name == "all":
        mask = np.ones(log["time"].shape, dtype=bool)
        time_origin = float(log["time"][0])
    else:
        phase_number = PHASE_NUMBER[phase_name]
        mask = log["phase"] == phase_number
        if not np.any(mask):
            raise ValueError(
                f"{log['path']}: no samples for phase '{phase_name}'"
            )
        time_origin = float(log["time"][np.flatnonzero(mask)[0]])

    relative_time = log["time"] - time_origin
    if duration > 0.0:
        mask &= relative_time <= duration + 1e-9

    selected: Dict[str, np.ndarray] = {}
    for key, value in log.items():
        if isinstance(value, np.ndarray) and value.shape == log["time"].shape:
            selected[key] = value[mask]
        else:
            selected[key] = value
    selected["time"] = relative_time[mask]
    return selected


def moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    """Centered display-only moving average; samples=1 plots raw data."""
    if samples <= 1 or values.size < samples:
        return values
    kernel = np.ones(samples, dtype=float) / samples
    padded = np.pad(values, (samples // 2, samples - 1 - samples // 2),
                    mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def finite(values: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(values)))


def shade_phases(ax: plt.Axes, log: Dict[str, np.ndarray]) -> None:
    time = log["time"]
    phase = log["phase"]
    for number, (name, color) in PHASE_STYLE.items():
        indices = np.flatnonzero(phase == number)
        if not indices.size:
            continue
        ax.axvspan(time[indices[0]], time[indices[-1]], color=color,
                   alpha=0.34, linewidth=0, label=f"{name} phase")


def shade_impact(
    ax: plt.Axes, log: Dict[str, np.ndarray], threshold: float
) -> None:
    """Shade pre-impact, impact and post-impact using recorded external force."""
    time = log["time"]
    if not time.size:
        return
    indices = np.flatnonzero(np.abs(log["external"]) >= threshold)
    if not indices.size:
        return
    start = float(time[indices[0]])
    end = float(time[indices[-1]])
    left = float(time[0])
    right = float(time[-1])
    ax.axvspan(left, start, color="#E8F1F2", alpha=0.42, linewidth=0,
               label="Pre-impact")
    ax.axvspan(start, end, color="#F6D6AD", alpha=0.45, linewidth=0,
               label="Impact")
    ax.axvspan(end, right, color="#E2F0D9", alpha=0.42, linewidth=0,
               label="Post-impact")


def add_background(
    ax: plt.Axes, log: Dict[str, np.ndarray], phase_name: str,
    impact_threshold: float
) -> None:
    if phase_name == "all":
        shade_phases(ax, log)
    elif phase_name == "test":
        shade_impact(ax, log, impact_threshold)


def unique_legend(ax: plt.Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(unique.values(), unique.keys(), **kwargs)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single(
    log: Dict[str, np.ndarray], output: Path, phase_name: str,
    impact_threshold: float, force_max: float, smooth: int, dpi: int
) -> None:
    time = log["time"]
    method = str(log["method"])
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)

    external = moving_average(log["external"], smooth)
    external_raw = moving_average(log["external_raw"], smooth)
    axes[0].plot(time, external_raw, color=COLORS["external_raw"],
                 linewidth=1.0, alpha=0.7, label="Tangential sensor force")
    axes[0].plot(time, external, color=COLORS["external"], linewidth=2.0,
                 label=r"Corrected disturbance $F_E$")
    axes[0].axhline(impact_threshold, color="#777777", linewidth=1.0,
                    linestyle=":", label="Impact threshold")
    axes[0].axhline(-impact_threshold, color="#777777", linewidth=1.0,
                    linestyle=":")
    axes[0].set_ylabel("Force [N]")
    axes[0].set_title("External disturbance")

    if finite(log["qp_raw"]):
        axes[1].plot(time, moving_average(log["qp_raw"], smooth),
                     color=COLORS["qp"], linewidth=1.5, linestyle=":",
                     label=r"Raw QP force $F_{I,qp}$")
    axes[1].plot(time, moving_average(log["desired"], smooth),
                 color=COLORS["desired"], linewidth=2.0, linestyle="--",
                 label=r"Desired force $F_{I,des}$")
    axes[1].plot(time, moving_average(log["internal"], smooth),
                 color=COLORS["measured"], linewidth=2.0,
                 label=r"Measured force $F_{I,est}$")
    axes[1].axhline(force_max, color="#C44E52", linewidth=1.2,
                    linestyle="-.", label=f"Desired-force limit ({force_max:g} N)")
    axes[1].set_ylabel("Internal force [N]")
    axes[1].set_title("Internal-force command and response")

    slip_mm = 1000.0 * moving_average(log["slip"], smooth)
    axes[2].plot(time, slip_mm, color=COLORS["slip"], linewidth=2.2,
                 label=r"Estimated slip $\delta$")
    axes[2].set_ylabel("Estimated slip [mm]", color=COLORS["slip"])
    axes[2].tick_params(axis="y", labelcolor=COLORS["slip"])
    axes[2].set_title("Slip-state response")
    if finite(log["slip_rate"]):
        rate_axis = axes[2].twinx()
        rate_axis.plot(time, 1000.0 * moving_average(log["slip_rate"], smooth),
                       color=COLORS["slip_rate"], linewidth=1.6,
                       linestyle="--", label=r"Slip rate $\dot{\delta}$")
        rate_axis.set_ylabel("Slip rate [mm/s]", color=COLORS["slip_rate"])
        rate_axis.tick_params(axis="y", labelcolor=COLORS["slip_rate"])
        rate_axis.spines["top"].set_visible(False)
        handles1, labels1 = axes[2].get_legend_handles_labels()
        handles2, labels2 = rate_axis.get_legend_handles_labels()
        axes[2].legend(handles1 + handles2, labels1 + labels2,
                       loc="upper right", frameon=False)

    for axis in axes:
        add_background(axis, log, phase_name, impact_threshold)
        style_axis(axis)
        if axis is not axes[2] or not finite(log["slip_rate"]):
            unique_legend(axis, loc="best", frameon=False, ncol=2)

    axes[-1].set_xlabel(
        "Time in selected phase [s]" if phase_name != "all" else "Experiment time [s]"
    )
    if time.size:
        axes[-1].set_xlim(float(time[0]), float(time[-1]))
    fig.suptitle(f"{method} force experiment — {phase_name.upper()} phase",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    difr: Dict[str, np.ndarray], sifr: Dict[str, np.ndarray], output: Path,
    phase_name: str, smooth: int, dpi: int
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)
    for log, method, color in (
        (difr, "DIFR", COLORS["difr"]),
        (sifr, "SIFR", COLORS["sifr"]),
    ):
        time = log["time"]
        axes[0].plot(time, moving_average(log["external"], smooth),
                     color=color, linewidth=1.8, label=method)
        axes[1].plot(time, moving_average(log["internal"], smooth),
                     color=color, linewidth=2.0, label=f"{method} measured")
        axes[1].plot(time, moving_average(log["desired"], smooth),
                     color=color, linewidth=1.6, linestyle="--",
                     label=f"{method} desired")
        axes[2].plot(time, 1000.0 * moving_average(log["slip"], smooth),
                     color=color, linewidth=2.0, label=method)

    axes[0].set_ylabel("Disturbance [N]")
    axes[0].set_title("Matched external disturbances")
    axes[1].set_ylabel("Internal force [N]")
    axes[1].set_title("Internal-force response")
    axes[2].set_ylabel("Estimated slip [mm]")
    axes[2].set_title("Estimated slip response")
    axes[2].set_xlabel(
        "Time in selected phase [s]" if phase_name != "all" else "Experiment time [s]"
    )
    for axis in axes:
        style_axis(axis)
        unique_legend(axis, loc="best", frameon=False, ncol=2)
    common_end = min(float(difr["time"][-1]), float(sifr["time"][-1]))
    axes[-1].set_xlim(0.0, common_end)
    fig.suptitle(f"DIFR vs SIFR — {phase_name.upper()} phase",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_summary(log: Dict[str, np.ndarray]) -> None:
    method = str(log["method"])
    print(f"[{method}] {log['path']}")
    print(f"  samples: {log['time'].size}")
    print(f"  time: {log['time'][0]:.3f} .. {log['time'][-1]:.3f} s")
    print(f"  peak |F_E|: {np.nanmax(np.abs(log['external'])):.3f} N")
    if finite(log["qp_raw"]):
        print(f"  peak raw QP force: {np.nanmax(log['qp_raw']):.3f} N")
    print(f"  desired force range: {np.nanmin(log['desired']):.3f} .. "
          f"{np.nanmax(log['desired']):.3f} N")
    print(f"  measured force range: {np.nanmin(log['internal']):.3f} .. "
          f"{np.nanmax(log['internal']):.3f} N")
    print(f"  peak estimated slip: {1000*np.nanmax(log['slip']):.3f} mm")
    if finite(log["slip_rate"]):
        print(f"  peak slip rate: {1000*np.nanmax(log['slip_rate']):.3f} mm/s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the current DIFR/SIFR force-experiment text logs"
    )
    parser.add_argument("--difr", help="DIFR log file")
    parser.add_argument("--sifr", help="SIFR log file")
    parser.add_argument("--output-dir", default="force_experiment_figures",
                        help="directory for generated figures")
    parser.add_argument("--phase", choices=["move", "grasp", "test", "all"],
                        default="test", help="experiment interval to plot")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="seconds from selected phase; 0 keeps all samples")
    parser.add_argument("--impact-threshold", type=float, default=0.8,
                        help="|F_E| threshold used only for impact shading (N)")
    parser.add_argument("--force-max", type=float, default=12.0,
                        help="desired-force limit reference line (N)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="display-only moving-average samples; 1 means raw data")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    args = parser.parse_args()

    if not args.difr and not args.sifr:
        parser.error("provide at least one of --difr or --sifr")
    if args.duration < 0.0 or args.impact_threshold < 0.0:
        parser.error("--duration and --impact-threshold cannot be negative")
    if args.force_max <= 0.0 or args.smooth < 1 or args.dpi <= 0:
        parser.error("--force-max, --smooth and --dpi must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    difr: Optional[Dict[str, np.ndarray]] = None
    sifr: Optional[Dict[str, np.ndarray]] = None
    if args.difr:
        difr = select_interval(load_log(args.difr), args.phase, args.duration)
        print_summary(difr)
        output = output_dir / f"difr_{args.phase}.{args.format}"
        plot_single(difr, output, args.phase, args.impact_threshold,
                    args.force_max, args.smooth, args.dpi)
        print(f"  saved: {output}")
    if args.sifr:
        sifr = select_interval(load_log(args.sifr), args.phase, args.duration)
        print_summary(sifr)
        output = output_dir / f"sifr_{args.phase}.{args.format}"
        plot_single(sifr, output, args.phase, args.impact_threshold,
                    args.force_max, args.smooth, args.dpi)
        print(f"  saved: {output}")
    if difr is not None and sifr is not None:
        output = output_dir / f"difr_vs_sifr_{args.phase}.{args.format}"
        plot_comparison(difr, sifr, output, args.phase, args.smooth, args.dpi)
        print(f"[comparison] saved: {output}")


if __name__ == "__main__":
    main()
