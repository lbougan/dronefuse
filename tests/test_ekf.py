import numpy as np

from dronefuse import quaternion as quat
from dronefuse.estimators.base import InitConfig, InitialState
from dronefuse.estimators.ekf import (
    STATE_DIM,
    EkfConfig,
    ErrorStateEKF,
    NominalState,
    error_between,
    inject,
    propagate_nominal,
    transition_jacobian,
)
from dronefuse.metrics import rmse
from dronefuse.scenario import FigureEight, Hover
from dronefuse.sensors import ImuSample, PosSample, SensorConfig
from dronefuse.simulate import (
    perturbed_initial_state,
    run_estimator,
    sample_truth,
    simulate_sensors,
)

NOMINAL = NominalState(
    pos=np.array([1.0, 2.0, 3.0]),
    vel=np.array([0.5, -0.2, 0.1]),
    quat=quat.from_rotation_vector(np.array([0.3, -0.2, 0.5])),
    accel_bias=np.array([0.05, -0.02, 0.01]),
    gyro_bias=np.array([0.001, -0.002, 0.003]),
)
IMU = ImuSample(t=0.0, accel=np.array([0.3, -0.5, 9.9]), gyro=np.array([0.2, -0.1, 0.3]))
NOISE_FREE = SensorConfig(
    accel_noise_std=0.0,
    gyro_noise_std=0.0,
    accel_bias_std0=0.0,
    gyro_bias_std0=0.0,
    accel_bias_walk_std=0.0,
    gyro_bias_walk_std=0.0,
    pos_noise_std=0.0,
)


def test_inject_then_error_between_roundtrips() -> None:
    dx = 1e-3 * np.arange(1, STATE_DIM + 1)
    np.testing.assert_allclose(error_between(NOMINAL, inject(NOMINAL, dx)), dx, atol=1e-12)


def test_transition_jacobian_matches_finite_differences() -> None:
    dt, eps = 0.005, 1e-6
    F = transition_jacobian(NOMINAL, IMU, dt)
    base = propagate_nominal(NOMINAL, IMU, dt)
    F_num = np.empty_like(F)
    for j in range(STATE_DIM):
        dx = np.zeros(STATE_DIM)
        dx[j] = eps
        F_num[:, j] = error_between(base, propagate_nominal(inject(NOMINAL, dx), IMU, dt)) / eps
    np.testing.assert_allclose(F_num, F, atol=2e-4)


def test_update_with_diagonal_prior_matches_scalar_kalman_gain() -> None:
    init = InitialState(pos=np.zeros(3), vel=np.zeros(3), quat=quat.IDENTITY)
    cfg = EkfConfig(pos_meas_std=1.0, pos_std0=1.0)
    ekf = ErrorStateEKF(init, cfg)
    ekf.update(PosSample(t=0.0, pos=np.array([2.0, 0.0, 0.0])))
    np.testing.assert_allclose(ekf.estimate.pos, [1.0, 0.0, 0.0])  # K = 1/(1+1)
    np.testing.assert_allclose(ekf.covariance[:3, :3], 0.5 * np.eye(3))
    np.testing.assert_allclose(ekf.estimate.vel, 0.0)


def test_noise_free_ekf_tracks_truth_closely() -> None:
    truth = sample_truth(FigureEight(), 30.0, NOISE_FREE.imu_rate_hz)
    log = simulate_sensors(truth, NOISE_FREE, seed=0)
    init = InitialState(pos=truth.pos[0], vel=truth.vel[0], quat=truth.quat[0])
    # Perfect sensors, but the filter keeps its normal noise model: R = Q = 0 would make the
    # innovation covariance singular after the first update.
    ekf = ErrorStateEKF(init, EkfConfig.from_sensor(SensorConfig(), InitConfig()))
    traj = run_estimator(ekf, log)
    assert rmse(traj.pos - truth.pos) < 0.05


def test_ekf_converges_on_hover_and_estimates_z_accel_bias() -> None:
    sensor_cfg, init_cfg = SensorConfig(), InitConfig()
    truth = sample_truth(Hover(), 30.0, sensor_cfg.imu_rate_hz)
    log = simulate_sensors(truth, sensor_cfg, seed=0)
    init = perturbed_initial_state(truth, init_cfg, seed=1)
    ekf = ErrorStateEKF(init, EkfConfig.from_sensor(sensor_cfg, init_cfg))
    traj = run_estimator(ekf, log)
    last_10s = truth.t >= 20.0
    assert rmse(traj.pos[last_10s] - truth.pos[last_10s]) < 0.3
    assert rmse(traj.vel[last_10s] - truth.vel[last_10s]) < 0.1
    assert abs(ekf.nominal.accel_bias[2] - log.accel_bias[-1, 2]) < 0.02
    P = ekf.covariance
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(P) > 0.0)
