"""Matplotlib figures: hero (path + error), per-axis error with ±3σ, NEES vs chi-square bounds."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dronefuse.metrics import NEES_DOF, chi2_bounds
from dronefuse.pipeline import RunResult
from dronefuse.simulate import EstimateTrajectory, Truth

Dropouts = Sequence[tuple[float, float]]

COLORS = {"truth": "#0b0b0b", "ekf": "#2a78d6", "kf": "#eb6834", "dr": "#1baf7a"}
LABELS = {
    "truth": "Ground truth",
    "dr": "Dead reckoning (IMU only)",
    "kf": "Linear KF (fixes only)",
    "ekf": "Error-state EKF (IMU + fixes)",
}
_MUTED = "#52514e"
_LINE_W = 2.0
_DPI = 150
_XY_PAD_M = 5.0


def _style(ax: plt.Axes) -> None:
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _shade_dropouts(ax: plt.Axes, dropouts: Dropouts) -> None:
    for i, (start, end) in enumerate(dropouts):
        label = "fix dropout" if i == 0 else None
        ax.axvspan(start, end, color=_MUTED, alpha=0.15, lw=0, label=label)


def plot_hero(
    truth: Truth, trajectories: Mapping[str, EstimateTrajectory], dropouts: Dropouts, path: Path
) -> None:
    """Left: top-down path, truth vs estimators. Right: position error vs time (log scale)."""
    fig, (ax_xy, ax_err) = plt.subplots(1, 2, figsize=(12, 5))
    ax_xy.plot(
        truth.pos[:, 0],
        truth.pos[:, 1],
        color=COLORS["truth"],
        lw=_LINE_W,
        ls="--",
        label=LABELS["truth"],
    )
    for name, traj in trajectories.items():
        ax_xy.plot(
            traj.pos[:, 0], traj.pos[:, 1], color=COLORS[name], lw=_LINE_W, label=LABELS[name]
        )
    ax_xy.set_xlim(truth.pos[:, 0].min() - _XY_PAD_M, truth.pos[:, 0].max() + _XY_PAD_M)
    ax_xy.set_ylim(truth.pos[:, 1].min() - _XY_PAD_M, truth.pos[:, 1].max() + _XY_PAD_M)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Top-down path")
    ax_xy.legend(loc="upper left", fontsize=8)
    _style(ax_xy)

    for name, traj in trajectories.items():
        err = np.linalg.norm(traj.pos - truth.pos, axis=1)
        ax_err.plot(truth.t, err, color=COLORS[name], lw=_LINE_W, label=LABELS[name])
    _shade_dropouts(ax_err, dropouts)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("position error [m]")
    ax_err.set_title("Position error")
    ax_err.legend(loc="upper left", fontsize=8)
    _style(ax_err)

    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_errors(truth: Truth, traj: EstimateTrajectory, dropouts: Dropouts, path: Path) -> None:
    """Per-axis position error with the filter's own ±3σ band."""
    if traj.cov_pv is None:
        raise ValueError(f"estimator {traj.name!r} has no covariance to plot")
    err = traj.pos - truth.pos
    sigma = np.sqrt(np.diagonal(traj.cov_pv[:, :3, :3], axis1=1, axis2=2))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for k, (ax, axis) in enumerate(zip(axes, "xyz", strict=True)):
        ax.fill_between(
            truth.t,
            -3 * sigma[:, k],
            3 * sigma[:, k],
            color=COLORS[traj.name],
            alpha=0.2,
            lw=0,
            label="±3σ",
        )
        ax.plot(truth.t, err[:, k], color=COLORS[traj.name], lw=1.5, label="error")
        _shade_dropouts(ax, dropouts)
        ax.set_ylabel(f"{axis} error [m]")
        _style(ax)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{LABELS[traj.name]}: position error and ±3σ")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_nees(
    t: np.ndarray, nees_by_name: Mapping[str, np.ndarray], dropouts: Dropouts, path: Path
) -> None:
    """NEES per timestep against the 95 % chi-square interval (dof = 6)."""
    lo, hi = chi2_bounds(NEES_DOF)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, values in nees_by_name.items():
        ax.plot(t, values, color=COLORS[name], lw=1.2, label=LABELS[name])
    ax.axhline(lo, color=_MUTED, ls="--", lw=1.0, label="95 % chi-square bounds")
    ax.axhline(hi, color=_MUTED, ls="--", lw=1.0)
    ax.axhline(NEES_DOF, color=_MUTED, ls=":", lw=1.0, label=f"ideal mean = {NEES_DOF}")
    _shade_dropouts(ax, dropouts)
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NEES (pos + vel)")
    ax.set_title("Filter consistency: NEES")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def write_plots(result: RunResult, out_dir: Path) -> list[Path]:
    """Write hero, per-estimator error plots (where a covariance exists) and NEES. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    hero = out_dir / f"hero_{result.scenario}.png"
    plot_hero(result.truth, result.trajectories, result.dropouts, hero)
    written.append(hero)
    for name, traj in result.trajectories.items():
        if traj.cov_pv is None:
            continue
        path = out_dir / f"errors_{name}.png"
        plot_errors(result.truth, traj, result.dropouts, path)
        written.append(path)
    nees_by_name = {n: m.nees for n, m in result.metrics.items() if m.nees is not None}
    if nees_by_name:
        path = out_dir / "nees.png"
        plot_nees(result.truth.t, nees_by_name, result.dropouts, path)
        written.append(path)
    return written
