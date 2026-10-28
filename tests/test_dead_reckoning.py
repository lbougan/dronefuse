import numpy as np

from dronefuse.estimators.base import InitialState
from dronefuse.estimators.dead_reckoning import DeadReckoning
from dronefuse.metrics import rmse
from dronefuse.scenario import FigureEight
from dronefuse.sensors import PosSample, SensorConfig
from dronefuse.simulate import (
    EstimateTrajectory,
    Truth,
    run_estimator,
    sample_truth,
    simulate_sensors,
)

NOISE_FREE = SensorConfig(
    accel_noise_std=0.0,
    gyro_noise_std=0.0,
    accel_bias_std0=0.0,
    gyro_bias_std0=0.0,
    accel_bias_walk_std=0.0,
    gyro_bias_walk_std=0.0,
    pos_noise_std=0.0,
)


def run_dr(cfg: SensorConfig, duration: float = 60.0) -> tuple[Truth, EstimateTrajectory]:
    truth = sample_truth(FigureEight(), duration, cfg.imu_rate_hz)
    log = simulate_sensors(truth, cfg, seed=0)
    init = InitialState(pos=truth.pos[0], vel=truth.vel[0], quat=truth.quat[0])
    return truth, run_estimator(DeadReckoning(init), log)


def test_noise_free_dead_reckoning_tracks_truth() -> None:
    # First-order integration holds each gyro sample over its 5 ms step, which leaves a
    # ~1.7 mrad yaw bias on the figure-eight; that grows to ~0.3 m RMSE over 60 s. Anything
    # larger means the IMU model and the integrator disagree (frames, gravity sign, quaternion).
    truth, traj = run_dr(NOISE_FREE)
    assert rmse(traj.pos - truth.pos) < 0.5
    assert rmse(traj.vel - truth.vel) < 0.02


def test_noisy_dead_reckoning_drifts_by_metres() -> None:
    truth, traj = run_dr(SensorConfig())
    assert rmse(traj.pos - truth.pos) >= 5.0


def test_update_is_ignored() -> None:
    init = InitialState(pos=np.zeros(3), vel=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    dr = DeadReckoning(init)
    dr.update(PosSample(t=0.0, pos=np.array([5.0, 5.0, 5.0])))
    np.testing.assert_array_equal(dr.estimate.pos, np.zeros(3))
    assert dr.estimate.cov_pv is None
