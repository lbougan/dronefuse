import numpy as np

from dronefuse import quaternion as quat
from dronefuse.estimators.base import InitialState
from dronefuse.estimators.kf import KfConfig, LinearKF
from dronefuse.estimators.linalg import joseph_update
from dronefuse.metrics import rmse
from dronefuse.scenario import FigureEight
from dronefuse.sensors import ImuSample, PosSample, SensorConfig
from dronefuse.simulate import run_estimator, sample_truth, simulate_sensors

DUMMY_IMU = ImuSample(t=0.0, accel=np.zeros(3), gyro=np.zeros(3))


def kf_1d(accel_noise_std: float = 0.0, pos_std0: float = 1.0, vel_std0: float = 1.0) -> LinearKF:
    init = InitialState(pos=np.array([0.0]), vel=np.array([0.0]), quat=quat.IDENTITY)
    cfg = KfConfig(
        accel_noise_std=accel_noise_std, pos_meas_std=1.0, pos_std0=pos_std0, vel_std0=vel_std0
    )
    return LinearKF(init, cfg)


def test_joseph_update_matches_standard_form() -> None:
    P = np.array([[2.0, 1.0], [1.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[1.0]])
    K, P_new = joseph_update(P, H, R)
    np.testing.assert_allclose(K, [[2 / 3], [1 / 3]])
    np.testing.assert_allclose(P_new, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
    np.testing.assert_allclose(P_new, P_new.T)


def test_predict_matches_hand_computation() -> None:
    kf = kf_1d()
    kf.predict(DUMMY_IMU, dt=1.0)
    np.testing.assert_allclose(kf.estimate.cov_pv, [[2.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(kf.estimate.pos, [0.0])


def test_update_matches_hand_computation() -> None:
    kf = kf_1d()
    kf.predict(DUMMY_IMU, dt=1.0)
    kf.update(PosSample(t=1.0, pos=np.array([1.0])))
    np.testing.assert_allclose(kf.estimate.pos, [2 / 3])
    np.testing.assert_allclose(kf.estimate.vel, [1 / 3])
    np.testing.assert_allclose(kf.estimate.cov_pv, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])


def test_process_noise_is_continuous_white_noise_acceleration_model() -> None:
    # Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]] with q = 4, dt = 0.5
    kf = kf_1d(accel_noise_std=2.0, pos_std0=0.0, vel_std0=0.0)
    kf.predict(DUMMY_IMU, dt=0.5)
    np.testing.assert_allclose(kf.estimate.cov_pv, [[1 / 6, 0.5], [0.5, 2.0]])


def test_kf_3d_beats_raw_position_fixes_on_figure8() -> None:
    sensor_cfg = SensorConfig()
    truth = sample_truth(FigureEight(), 60.0, sensor_cfg.imu_rate_hz)
    log = simulate_sensors(truth, sensor_cfg, seed=0)
    init = InitialState(pos=truth.pos[0], vel=truth.vel[0], quat=truth.quat[0])
    traj = run_estimator(LinearKF(init, KfConfig(pos_meas_std=sensor_cfg.pos_noise_std)), log)
    assert traj.cov_pv is not None and traj.cov_pv.shape == (len(truth), 6, 6)
    # rmse() is the 3D norm, so raw fixes score pos_noise_std * sqrt(3) ~= 0.52 m. A CV model
    # on a 0.9 m/s^2 figure-eight lands near 0.4 m: better than the fixes, worse than the EKF.
    raw_fix_rmse = sensor_cfg.pos_noise_std * np.sqrt(3.0)
    assert rmse(traj.pos - truth.pos) < raw_fix_rmse
    # Velocity is essentially differenced 0.3 m fixes at 5 Hz, so ~0.7 m/s is expected.
    assert rmse(traj.vel - truth.vel) < 1.0
