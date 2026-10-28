import numpy as np

from dronefuse import quaternion as quat
from dronefuse.estimators.base import Estimate, InitConfig
from dronefuse.scenario import FigureEight
from dronefuse.sensors import ImuSample, PosSample, SensorConfig
from dronefuse.simulate import (
    perturbed_initial_state,
    run_estimator,
    sample_truth,
    simulate_sensors,
)


class CountingEstimator:
    name = "counting"

    def __init__(self) -> None:
        self.predicts = 0
        self.updates = 0

    def predict(self, imu: ImuSample, dt: float) -> None:
        self.predicts += 1

    def update(self, fix: PosSample) -> None:
        self.updates += 1

    @property
    def estimate(self) -> Estimate:
        return Estimate(pos=np.zeros(3), vel=np.zeros(3), quat=None, cov_pv=None)


def test_sample_truth_has_one_sample_per_imu_step_inclusive() -> None:
    truth = sample_truth(FigureEight(), duration=1.0, rate_hz=200.0)
    assert len(truth) == 201
    assert truth.pos.shape == (201, 3)
    assert truth.quat.shape == (201, 4)
    np.testing.assert_allclose(truth.t[-1], 1.0)


def test_truth_state_roundtrips_scenario_sample() -> None:
    sc = FigureEight()
    truth = sample_truth(sc, duration=1.0, rate_hz=200.0)
    s = truth.state(40)
    np.testing.assert_allclose(s.pos, sc.sample(0.2).pos)
    np.testing.assert_allclose(s.omega, sc.sample(0.2).omega)


def test_simulate_sensors_places_fixes_every_40th_sample() -> None:
    truth = sample_truth(FigureEight(), duration=1.0, rate_hz=200.0)
    log = simulate_sensors(truth, SensorConfig(), seed=0)
    has_fix = ~np.isnan(log.fix[:, 0])
    assert has_fix.sum() == 6
    assert list(np.flatnonzero(has_fix)) == [0, 40, 80, 120, 160, 200]
    assert log.fix_at(40) is not None
    assert log.fix_at(41) is None
    assert log.imu(3).t == truth.t[3]
    assert log.dt == 0.005


def test_simulate_sensors_respects_dropouts() -> None:
    truth = sample_truth(FigureEight(), duration=2.0, rate_hz=200.0)
    # Fix indices are multiples of 40; the interval is half-open [start, end).
    log = simulate_sensors(truth, SensorConfig(dropouts=((0.4, 1.4),)), seed=0)
    assert log.fix_at(40) is not None  # t = 0.2
    assert log.fix_at(80) is None  # t = 0.4, dropout starts
    assert log.fix_at(200) is None  # t = 1.0
    assert log.fix_at(280) is not None  # t = 1.4, dropout over


def test_simulate_sensors_records_true_biases() -> None:
    truth = sample_truth(FigureEight(), duration=1.0, rate_hz=200.0)
    log = simulate_sensors(truth, SensorConfig(), seed=0)
    assert log.accel_bias.shape == (201, 3)
    assert log.gyro_bias.shape == (201, 3)


def test_run_estimator_calls_predict_and_update_the_right_number_of_times() -> None:
    truth = sample_truth(FigureEight(), duration=1.0, rate_hz=200.0)
    log = simulate_sensors(truth, SensorConfig(), seed=0)
    est = CountingEstimator()
    traj = run_estimator(est, log)
    assert est.predicts == 200
    assert est.updates == 6
    assert traj.name == "counting"
    assert traj.pos.shape == (201, 3)
    assert traj.cov_pv is None


def test_perturbed_initial_state_is_seeded_and_off_truth() -> None:
    truth = sample_truth(FigureEight(), duration=1.0, rate_hz=200.0)
    a = perturbed_initial_state(truth, InitConfig(), seed=1)
    b = perturbed_initial_state(truth, InitConfig(), seed=1)
    np.testing.assert_array_equal(a.pos, b.pos)
    # Independent of every sensor stream: the draw must not equal seed 2's sensor bias draw.
    assert not np.allclose(
        a.pos - truth.pos[0], np.random.default_rng(2).normal(0.0, InitConfig().pos_std, 3)
    )
    assert np.linalg.norm(a.pos - truth.pos[0]) > 0.0
    assert np.isclose(np.linalg.norm(a.quat), 1.0)
    att_err = quat.to_rotation_vector(quat.multiply(quat.conjugate(truth.quat[0]), a.quat))
    assert np.linalg.norm(att_err) > 0.0
