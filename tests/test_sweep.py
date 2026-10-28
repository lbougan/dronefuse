import numpy as np
import pytest

from dronefuse.estimators import make_estimator
from dronefuse.estimators.base import InitConfig, InitialState
from dronefuse.estimators.ekf import ErrorStateEKF
from dronefuse.pipeline import run_pipeline
from dronefuse.sensors import SensorConfig
from dronefuse.sweep import Knob, format_sweep_table, sweep

INIT = InitialState(pos=np.zeros(3), vel=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))


def test_make_estimator_applies_overrides() -> None:
    ekf = make_estimator("ekf", INIT, SensorConfig(), InitConfig(), {"pos_meas_std": 9.0})
    assert isinstance(ekf, ErrorStateEKF)
    assert ekf._cfg.pos_meas_std == 9.0
    with pytest.raises(TypeError):
        make_estimator("kf", INIT, SensorConfig(), InitConfig(), {"no_such_field": 1.0})


def test_run_pipeline_overrides_change_only_that_filter() -> None:
    base = run_pipeline("hover", ["kf", "ekf"], seed=0, duration=1.0)
    tuned = run_pipeline(
        "hover", ["kf", "ekf"], seed=0, duration=1.0, overrides={"ekf": {"pos_meas_std": 3.0}}
    )
    np.testing.assert_array_equal(base.trajectories["kf"].pos, tuned.trajectories["kf"].pos)
    assert not np.array_equal(base.trajectories["ekf"].pos, tuned.trajectories["ekf"].pos)


def test_knob_parse_and_default() -> None:
    assert Knob.parse("ekf.accel_noise_std") == Knob("ekf", "accel_noise_std")
    assert Knob.parse("sensor.pos_noise_std").default(SensorConfig(), InitConfig()) == 0.3
    with pytest.raises(ValueError, match="target.field"):
        Knob.parse("accel_noise_std")
    with pytest.raises(ValueError, match="no field"):
        Knob.parse("kf.bogus").default(SensorConfig(), InitConfig())


def test_sweep_scaled_values_and_table() -> None:
    knob = Knob.parse("kf.accel_noise_std")
    rows = sweep(
        knob, [0.5, 2.0], ["dr", "kf"], scenario="hover", seeds=(0, 1), duration=1.0, scale=True
    )
    assert [r.value for r in rows] == [0.5, 0.5, 2.0, 2.0]
    assert [r.estimator for r in rows] == ["dr", "kf", "dr", "kf"]
    assert rows[0].nees_mean is None and rows[1].nees_mean is not None
    assert rows[1].nees_in_bounds + rows[1].nees_below + rows[1].nees_above == pytest.approx(1.0)
    table = format_sweep_table(knob, rows)
    assert table.splitlines()[0].startswith("| kf.accel_noise_std | Estimator |")
    assert "| 0.5 | KF |" in table and "| DR |" in table and "n/a" in table


def test_sweep_sensor_knob_changes_simulation() -> None:
    knob = Knob.parse("sensor.pos_noise_std")
    rows = sweep(knob, [0.1, 0.3], ["kf"], scenario="hover", duration=1.0)
    assert rows[0].pos_rmse < rows[1].pos_rmse
