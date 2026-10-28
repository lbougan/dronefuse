"""Dead reckoning: integrate the IMU, ignore position fixes. The drift baseline."""

import numpy as np

from dronefuse.estimators.base import Estimate, InitialState
from dronefuse.estimators.kinematics import propagate
from dronefuse.sensors import GRAVITY_W, ImuSample, PosSample


class DeadReckoning:
    name = "dr"

    def __init__(self, init: InitialState, gravity: np.ndarray = GRAVITY_W) -> None:
        self._pos = init.pos.copy()
        self._vel = init.vel.copy()
        self._quat = init.quat.copy()
        self._gravity = gravity

    def predict(self, imu: ImuSample, dt: float) -> None:
        self._pos, self._vel, self._quat = propagate(
            self._pos, self._vel, self._quat, imu.accel, imu.gyro, dt, self._gravity
        )

    def update(self, fix: PosSample) -> None:
        """Dead reckoning never uses fixes; present to satisfy the Estimator interface."""
        return None

    @property
    def estimate(self) -> Estimate:
        return Estimate(pos=self._pos, vel=self._vel, quat=self._quat, cov_pv=None)
