"""Seeded IMU and position-fix simulator."""

from dataclasses import dataclass

import numpy as np

from dronefuse import quaternion as quat
from dronefuse.scenario import TrueState

GRAVITY_W = np.array([0.0, 0.0, -9.81])


@dataclass(frozen=True)
class SensorConfig:
    """Noise model. The EKF is configured from these same numbers (EkfConfig.from_sensor)."""

    imu_rate_hz: float = 200.0
    fix_rate_hz: float = 5.0
    accel_noise_std: float = 0.05  # m/s^2, white, per axis per sample
    gyro_noise_std: float = 0.002  # rad/s, white
    accel_bias_std0: float = 0.05  # m/s^2, std of the initial (constant part of the) bias
    gyro_bias_std0: float = 0.005  # rad/s
    accel_bias_walk_std: float = 0.001  # m/s^2 per sqrt(s), bias random walk
    gyro_bias_walk_std: float = 0.0001  # rad/s per sqrt(s)
    pos_noise_std: float = 0.3  # m, per axis
    dropouts: tuple[tuple[float, float], ...] = ()  # (start, end) s; no fixes while start<=t<end


@dataclass(frozen=True)
class ImuSample:
    t: float
    accel: np.ndarray  # (3,) specific force, body frame, m/s^2
    gyro: np.ndarray  # (3,) angular rate, body frame, rad/s


@dataclass(frozen=True)
class PosSample:
    t: float
    pos: np.ndarray  # (3,) world frame, m


class SensorSim:
    """Turns a TrueState into noisy sensor samples. One instance = one seeded sensor history."""

    def __init__(self, cfg: SensorConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / cfg.imu_rate_hz
        self.accel_bias = self.rng.normal(0.0, cfg.accel_bias_std0, 3)
        self.gyro_bias = self.rng.normal(0.0, cfg.gyro_bias_std0, 3)

    def imu(self, s: TrueState) -> ImuSample:
        """Specific force and angular rate in the body frame, then advance the bias walk."""
        cfg = self.cfg
        rot = quat.to_rotation_matrix(s.quat)
        specific_force = rot.T @ (s.acc - GRAVITY_W)
        accel = specific_force + self.accel_bias + self.rng.normal(0.0, cfg.accel_noise_std, 3)
        gyro = s.omega + self.gyro_bias + self.rng.normal(0.0, cfg.gyro_noise_std, 3)
        sqrt_dt = np.sqrt(self.dt)
        self.accel_bias = self.accel_bias + self.rng.normal(
            0.0, cfg.accel_bias_walk_std * sqrt_dt, 3
        )
        self.gyro_bias = self.gyro_bias + self.rng.normal(0.0, cfg.gyro_bias_walk_std * sqrt_dt, 3)
        return ImuSample(t=s.t, accel=accel, gyro=gyro)

    def position_fix(self, s: TrueState) -> PosSample | None:
        """Noisy position, or None inside a dropout interval."""
        if self.in_dropout(s.t):
            return None
        return PosSample(t=s.t, pos=s.pos + self.rng.normal(0.0, self.cfg.pos_noise_std, 3))

    def in_dropout(self, t: float) -> bool:
        return any(start <= t < end for start, end in self.cfg.dropouts)
