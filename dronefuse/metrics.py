"""Evaluation metrics: RMSE, NEES consistency, dropout recovery."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from dronefuse.simulate import EstimateTrajectory, Truth

NEES_DOF = 6  # position + velocity
NEES_CONFIDENCE = 0.95


def rmse(err: np.ndarray) -> float:
    """Root mean square of the Euclidean norm of (N, k) error vectors."""
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def nees(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared, eᵀ P⁻¹ e, per timestep. err (N, k), cov (N, k, k)."""
    solved = np.linalg.solve(cov, err[:, :, None])[:, :, 0]
    return np.einsum("ni,ni->n", err, solved)


def chi2_bounds(dof: int, confidence: float = NEES_CONFIDENCE) -> tuple[float, float]:
    """Two-sided chi-square interval that a consistent filter's NEES should fall in."""
    tail = 0.5 * (1.0 - confidence)
    return float(chi2.ppf(tail, dof)), float(chi2.ppf(1.0 - tail, dof))


def fraction_in_bounds(values: np.ndarray, lo: float, hi: float) -> float:
    return float(np.mean((values >= lo) & (values <= hi)))


def recovery_time(
    t: np.ndarray, pos_err: np.ndarray, t_resume: float, threshold: float
) -> float | None:
    """Seconds after t_resume until the position error first drops below threshold."""
    idx = np.flatnonzero((t >= t_resume) & (pos_err < threshold))
    if idx.size == 0:
        return None
    return float(t[idx[0]] - t_resume)


@dataclass(frozen=True)
class Metrics:
    estimator: str
    pos_rmse: float  # m
    vel_rmse: float  # m/s
    max_pos_err: float  # m
    nees: np.ndarray | None  # (N,) or None when the estimator has no covariance
    nees_mean: float | None
    nees_in_bounds: float | None  # fraction of timesteps inside the 95 % chi-square interval


def evaluate(truth: Truth, traj: EstimateTrajectory) -> Metrics:
    err_pos = traj.pos - truth.pos
    err_vel = traj.vel - truth.vel
    nees_values = nees_mean = nees_frac = None
    if traj.cov_pv is not None:
        nees_values = nees(np.hstack([err_pos, err_vel]), traj.cov_pv)
        lo, hi = chi2_bounds(NEES_DOF)
        nees_mean = float(nees_values.mean())
        nees_frac = fraction_in_bounds(nees_values, lo, hi)
    return Metrics(
        estimator=traj.name,
        pos_rmse=rmse(err_pos),
        vel_rmse=rmse(err_vel),
        max_pos_err=float(np.linalg.norm(err_pos, axis=1).max()),
        nees=nees_values,
        nees_mean=nees_mean,
        nees_in_bounds=nees_frac,
    )


def format_metrics_table(metrics: Sequence[Metrics]) -> str:
    """Markdown table, one row per estimator."""
    lines = [
        "| Estimator | Pos RMSE [m] | Vel RMSE [m/s] | Max pos err [m] "
        "| Mean NEES (ideal 6) | NEES in 95 % bounds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        nees_mean = f"{m.nees_mean:.2f}" if m.nees_mean is not None else "n/a"
        frac = f"{100 * m.nees_in_bounds:.1f} %" if m.nees_in_bounds is not None else "n/a"
        lines.append(
            f"| {m.estimator.upper()} | {m.pos_rmse:.3f} | {m.vel_rmse:.3f} "
            f"| {m.max_pos_err:.2f} | {nees_mean} | {frac} |"
        )
    return "\n".join(lines)
