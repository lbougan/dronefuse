import numpy as np

from dronefuse.metrics import (
    Metrics,
    chi2_bounds,
    evaluate,
    format_metrics_table,
    fraction_in_bounds,
    nees,
    recovery_time,
    rmse,
)
from dronefuse.simulate import EstimateTrajectory, Truth


def test_rmse_of_vector_errors() -> None:
    err = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])  # norms 5 and 0
    assert np.isclose(rmse(err), np.sqrt((25.0 + 0.0) / 2))


def test_nees_with_identity_covariance_is_squared_norm() -> None:
    err = np.array([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0, 0.0, 0.0]])
    cov = np.stack([np.eye(6), np.eye(6)])
    np.testing.assert_allclose(nees(err, cov), [5.0, 9.0])


def test_nees_scales_with_inverse_covariance() -> None:
    err = np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    cov = np.stack([4.0 * np.eye(6)])
    np.testing.assert_allclose(nees(err, cov), [1.0])


def test_chi2_bounds_for_6_dof() -> None:
    lo, hi = chi2_bounds(dof=6, confidence=0.95)
    assert np.isclose(lo, 1.237, atol=1e-3)
    assert np.isclose(hi, 14.449, atol=1e-3)


def test_fraction_in_bounds() -> None:
    assert fraction_in_bounds(np.array([0.5, 2.0, 5.0, 20.0]), 1.0, 10.0) == 0.5


def test_recovery_time_is_first_time_below_threshold_after_resume() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    err = np.array([0.1, 5.0, 2.0, 0.4, 0.2])
    assert recovery_time(t, err, t_resume=1.0, threshold=0.5) == 2.0
    assert recovery_time(t, err, t_resume=1.0, threshold=0.05) is None


def _const_truth(n: int = 5) -> Truth:
    t = np.arange(n, dtype=float)
    return Truth(
        t=t,
        pos=np.zeros((n, 3)),
        vel=np.zeros((n, 3)),
        acc=np.zeros((n, 3)),
        quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        omega=np.zeros((n, 3)),
    )


def test_evaluate_without_covariance_skips_nees() -> None:
    truth = _const_truth()
    traj = EstimateTrajectory(
        name="dr", pos=np.full((5, 3), 1.0), vel=np.zeros((5, 3)), cov_pv=None
    )
    m = evaluate(truth, traj)
    assert m.estimator == "dr"
    assert np.isclose(m.pos_rmse, np.sqrt(3.0))
    assert np.isclose(m.max_pos_err, np.sqrt(3.0))
    assert m.nees is None and m.nees_mean is None and m.nees_in_bounds is None


def test_evaluate_with_covariance_reports_nees() -> None:
    truth = _const_truth()
    traj = EstimateTrajectory(
        name="ekf",
        pos=np.full((5, 3), 1.0),
        vel=np.zeros((5, 3)),
        cov_pv=np.stack([np.eye(6)] * 5),
    )
    m = evaluate(truth, traj)
    assert m.nees is not None
    np.testing.assert_allclose(m.nees, 3.0)
    assert m.nees_mean == 3.0
    assert m.nees_in_bounds == 1.0  # 3.0 is inside [1.237, 14.449]


def test_format_metrics_table_is_markdown() -> None:
    m = Metrics(
        estimator="ekf",
        pos_rmse=0.123,
        vel_rmse=0.045,
        max_pos_err=0.5,
        nees=np.array([6.0]),
        nees_mean=6.0,
        nees_in_bounds=0.95,
    )
    table = format_metrics_table([m])
    assert table.splitlines()[0].startswith("| Estimator |")
    assert "| EKF | 0.123 | 0.045 | 0.50 | 6.00 | 95.0 % |" in table
