"""Estimators sharing the predict/update/estimate interface."""

from collections.abc import Mapping
from dataclasses import replace

from dronefuse.estimators.base import Estimator, InitConfig, InitialState
from dronefuse.estimators.dead_reckoning import DeadReckoning
from dronefuse.estimators.ekf import EkfConfig, ErrorStateEKF
from dronefuse.estimators.kf import KfConfig, LinearKF
from dronefuse.sensors import SensorConfig

ESTIMATOR_NAMES: tuple[str, ...] = (DeadReckoning.name, LinearKF.name, ErrorStateEKF.name)


Overrides = Mapping[str, float]


def default_config(
    name: str, sensor_cfg: SensorConfig, init_cfg: InitConfig
) -> KfConfig | EkfConfig | None:
    """Filter config derived from the simulator's noise model (None for dead reckoning)."""
    if name == DeadReckoning.name:
        return None
    if name == LinearKF.name:
        return KfConfig(
            pos_meas_std=sensor_cfg.pos_noise_std,
            pos_std0=init_cfg.pos_std,
            vel_std0=init_cfg.vel_std,
        )
    if name == ErrorStateEKF.name:
        return EkfConfig.from_sensor(sensor_cfg, init_cfg)
    raise ValueError(f"unknown estimator {name!r}; choose from {ESTIMATOR_NAMES}")


def make_estimator(
    name: str,
    init: InitialState,
    sensor_cfg: SensorConfig,
    init_cfg: InitConfig,
    overrides: Overrides | None = None,
) -> Estimator:
    """Build an estimator whose noise model matches the simulator, with optional field overrides.

    `overrides` maps config field names (e.g. "accel_noise_std") to values that replace the
    derived defaults; this is how `dronefuse sweep` tunes a filter without touching the sensors.
    """
    cfg = default_config(name, sensor_cfg, init_cfg)
    if cfg is not None and overrides:
        cfg = replace(cfg, **dict(overrides))
    if name == DeadReckoning.name:
        return DeadReckoning(init)
    if isinstance(cfg, KfConfig):
        return LinearKF(init, cfg)
    assert isinstance(cfg, EkfConfig)
    return ErrorStateEKF(init, cfg)
