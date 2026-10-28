"""Scenario -> sensors -> estimators -> metrics, in one call."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from dronefuse.estimators import Overrides, make_estimator
from dronefuse.estimators.base import InitConfig
from dronefuse.metrics import Metrics, evaluate
from dronefuse.scenario import make_scenario
from dronefuse.sensors import SensorConfig
from dronefuse.simulate import (
    EstimateTrajectory,
    SensorLog,
    Truth,
    perturbed_initial_state,
    run_estimator,
    sample_truth,
    simulate_sensors,
)


@dataclass(frozen=True)
class RunResult:
    scenario: str
    seed: int
    dropouts: tuple[tuple[float, float], ...]
    truth: Truth
    log: SensorLog
    trajectories: dict[str, EstimateTrajectory]
    metrics: dict[str, Metrics]


def run_pipeline(
    scenario_name: str,
    estimator_names: Sequence[str],
    seed: int = 0,
    duration: float = 60.0,
    dropouts: Sequence[tuple[float, float]] = (),
    sensor_cfg: SensorConfig | None = None,
    init_cfg: InitConfig | None = None,
    overrides: Mapping[str, Overrides] | None = None,
) -> RunResult:
    """Run every named estimator over the same seeded sensor log and evaluate each.

    `overrides` maps an estimator name to config-field overrides (see make_estimator); every
    key must name one of `estimator_names`. `dropouts`, when given, replace those of sensor_cfg.
    """
    sensor_cfg = sensor_cfg or SensorConfig()
    if dropouts:
        sensor_cfg = replace(sensor_cfg, dropouts=tuple(dropouts))
    init_cfg = init_cfg or InitConfig()
    overrides = overrides or {}
    unused = set(overrides) - set(estimator_names)
    if unused:
        raise ValueError(f"overrides for estimators that are not run: {sorted(unused)}")
    truth = sample_truth(make_scenario(scenario_name), duration, sensor_cfg.imu_rate_hz)
    log = simulate_sensors(truth, sensor_cfg, seed)
    init = perturbed_initial_state(truth, init_cfg, seed)
    trajectories = {
        name: run_estimator(
            make_estimator(name, init, sensor_cfg, init_cfg, overrides.get(name)), log
        )
        for name in estimator_names
    }
    metrics = {name: evaluate(truth, traj) for name, traj in trajectories.items()}
    return RunResult(
        scenario=scenario_name,
        seed=seed,
        dropouts=sensor_cfg.dropouts,
        truth=truth,
        log=log,
        trajectories=trajectories,
        metrics=metrics,
    )
