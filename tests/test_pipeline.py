import numpy as np
import pytest

from dronefuse.estimators import ESTIMATOR_NAMES, make_estimator
from dronefuse.estimators.base import InitConfig, InitialState
from dronefuse.pipeline import run_pipeline
from dronefuse.sensors import SensorConfig


def test_estimator_names_and_factory() -> None:
    assert ESTIMATOR_NAMES == ("dr", "kf", "ekf")
    init = InitialState(pos=np.zeros(3), vel=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    for name in ESTIMATOR_NAMES:
        assert make_estimator(name, init, SensorConfig(), InitConfig()).name == name
    with pytest.raises(ValueError, match="unknown estimator"):
        make_estimator("ukf", init, SensorConfig(), InitConfig())


def test_run_pipeline_short_run_is_reproducible() -> None:
    a = run_pipeline("hover", ["ekf"], seed=0, duration=2.0)
    b = run_pipeline("hover", ["ekf"], seed=0, duration=2.0)
    np.testing.assert_array_equal(a.trajectories["ekf"].pos, b.trajectories["ekf"].pos)
    assert set(a.metrics) == {"ekf"}
    assert a.scenario == "hover"
    assert len(a.truth) == 401


def test_run_pipeline_passes_dropouts_to_sensors() -> None:
    r = run_pipeline("line", ["dr"], seed=0, duration=3.0, dropouts=[(1.0, 2.0)])
    assert r.dropouts == ((1.0, 2.0),)
    assert r.log.fix_at(200) is None  # t = 1.0
    assert r.log.fix_at(400) is not None  # t = 2.0
