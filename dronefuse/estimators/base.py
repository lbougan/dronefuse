"""Shared estimator interface and initial-state types."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dronefuse.sensors import ImuSample, PosSample


@dataclass(frozen=True)
class InitialState:
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)
    quat: np.ndarray  # (4,) body -> world


@dataclass(frozen=True)
class InitConfig:
    """Std of the initial estimate error; also used to build P0 of the filters."""

    pos_std: float = 0.5  # m
    vel_std: float = 0.1  # m/s
    att_std: float = 0.02  # rad


@dataclass(frozen=True)
class Estimate:
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)
    quat: np.ndarray | None  # (4,) or None if the estimator has no attitude
    cov_pv: np.ndarray | None  # (6, 6) covariance of [pos, vel], or None


class Estimator(Protocol):
    name: str

    def predict(self, imu: ImuSample, dt: float) -> None: ...

    def update(self, fix: PosSample) -> None: ...

    @property
    def estimate(self) -> Estimate: ...
