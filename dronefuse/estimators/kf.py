"""Linear Kalman filter: constant-velocity model, position-fix updates, IMU ignored."""

from dataclasses import dataclass

import numpy as np

from dronefuse.estimators.base import Estimate, InitialState
from dronefuse.estimators.linalg import joseph_update
from dronefuse.sensors import ImuSample, PosSample


@dataclass(frozen=True)
class KfConfig:
    accel_noise_std: float = 1.0  # m/s^2/sqrt(s), continuous white-noise acceleration PSD
    pos_meas_std: float = 0.3  # m
    pos_std0: float = 0.5  # m
    vel_std0: float = 0.1  # m/s


class LinearKF:
    """State x = [pos, vel] of dimension 2*dim; dim is taken from init.pos."""

    name = "kf"

    def __init__(self, init: InitialState, cfg: KfConfig) -> None:
        self._dim = len(init.pos)
        self._cfg = cfg
        self._x = np.concatenate([init.pos, init.vel]).astype(float)
        self._P = np.diag([cfg.pos_std0**2] * self._dim + [cfg.vel_std0**2] * self._dim)

    def predict(self, imu: ImuSample, dt: float) -> None:
        """Constant-velocity prediction; the IMU sample is deliberately unused."""
        d = self._dim
        eye = np.eye(d)
        zero = np.zeros((d, d))
        F = np.block([[eye, dt * eye], [zero, eye]])
        # Continuous white-noise acceleration, integrated over dt (Bar-Shalom CWNA model).
        # Velocity variance grows as q*dt per step, so the tuning is independent of the
        # 200 Hz prediction rate.
        q = self._cfg.accel_noise_std**2
        Q = q * np.block([[dt**3 / 3 * eye, dt**2 / 2 * eye], [dt**2 / 2 * eye, dt * eye]])
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def update(self, fix: PosSample) -> None:
        d = self._dim
        H = np.hstack([np.eye(d), np.zeros((d, d))])
        R = self._cfg.pos_meas_std**2 * np.eye(d)
        K, P_new = joseph_update(self._P, H, R)
        self._x = self._x + K @ (fix.pos - H @ self._x)
        self._P = P_new

    @property
    def estimate(self) -> Estimate:
        d = self._dim
        return Estimate(pos=self._x[:d], vel=self._x[d:], quat=None, cov_pv=self._P.copy())
