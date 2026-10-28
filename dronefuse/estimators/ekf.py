"""Error-state EKF, following Solà (2017), "Quaternion kinematics for the error-state KF".

Nominal state: position p, velocity v, attitude quaternion q (body -> world),
accelerometer bias a_b, gyroscope bias w_b. Error state (15):
    dx = [dp, dv, dtheta, da_b, dw_b]
with a local attitude perturbation, q_true = q ⊗ Exp(dtheta).
Predict with the IMU (eq. 260-272), correct with position fixes (eq. 274-286).
"""

from dataclasses import dataclass, replace

import numpy as np

from dronefuse import quaternion as quat
from dronefuse.estimators.base import Estimate, InitConfig, InitialState
from dronefuse.estimators.kinematics import propagate
from dronefuse.estimators.linalg import joseph_update
from dronefuse.sensors import GRAVITY_W, ImuSample, PosSample, SensorConfig

STATE_DIM = 15
POS = slice(0, 3)
VEL = slice(3, 6)
ATT = slice(6, 9)
ABIAS = slice(9, 12)
GBIAS = slice(12, 15)


@dataclass(frozen=True)
class EkfConfig:
    accel_noise_std: float = 0.05
    gyro_noise_std: float = 0.002
    accel_bias_walk_std: float = 0.001
    gyro_bias_walk_std: float = 0.0001
    pos_meas_std: float = 0.3
    pos_std0: float = 0.5
    vel_std0: float = 0.1
    att_std0: float = 0.02
    accel_bias_std0: float = 0.05
    gyro_bias_std0: float = 0.005

    @classmethod
    def from_sensor(cls, sensor: SensorConfig, init: InitConfig) -> "EkfConfig":
        """Filter model that matches the simulator, which is what makes NEES consistent."""
        return cls(
            accel_noise_std=sensor.accel_noise_std,
            gyro_noise_std=sensor.gyro_noise_std,
            accel_bias_walk_std=sensor.accel_bias_walk_std,
            gyro_bias_walk_std=sensor.gyro_bias_walk_std,
            pos_meas_std=sensor.pos_noise_std,
            pos_std0=init.pos_std,
            vel_std0=init.vel_std,
            att_std0=init.att_std,
            accel_bias_std0=sensor.accel_bias_std0,
            gyro_bias_std0=sensor.gyro_bias_std0,
        )


@dataclass(frozen=True)
class NominalState:
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray


def propagate_nominal(
    x: NominalState, imu: ImuSample, dt: float, gravity: np.ndarray = GRAVITY_W
) -> NominalState:
    """Nominal kinematics with bias-corrected IMU (Solà eq. 260). Biases stay constant."""
    pos, vel, q = propagate(
        x.pos, x.vel, x.quat, imu.accel - x.accel_bias, imu.gyro - x.gyro_bias, dt, gravity
    )
    return replace(x, pos=pos, vel=vel, quat=q)


def transition_jacobian(x: NominalState, imu: ImuSample, dt: float) -> np.ndarray:
    """Discrete error-state transition F_x (Solà eq. 270), first order in dt."""
    accel = imu.accel - x.accel_bias
    gyro = imu.gyro - x.gyro_bias
    rot = quat.to_rotation_matrix(x.quat)
    eye = np.eye(3)
    F = np.eye(STATE_DIM)
    F[POS, VEL] = eye * dt
    F[VEL, ATT] = -rot @ quat.skew(accel) * dt
    F[VEL, ABIAS] = -rot * dt
    F[ATT, ATT] = quat.to_rotation_matrix(quat.from_rotation_vector(gyro * dt)).T
    F[ATT, GBIAS] = -eye * dt
    return F


def process_noise(cfg: EkfConfig, dt: float) -> np.ndarray:
    """F_i Q_i F_iᵀ (Solà eq. 271-272): white IMU noise on dv/dtheta, random walk on biases."""
    eye = np.eye(3)
    Q = np.zeros((STATE_DIM, STATE_DIM))
    Q[VEL, VEL] = (cfg.accel_noise_std * dt) ** 2 * eye
    Q[ATT, ATT] = (cfg.gyro_noise_std * dt) ** 2 * eye
    Q[ABIAS, ABIAS] = cfg.accel_bias_walk_std**2 * dt * eye
    Q[GBIAS, GBIAS] = cfg.gyro_bias_walk_std**2 * dt * eye
    return Q


def inject(x: NominalState, dx: np.ndarray) -> NominalState:
    """x ⊕ dx (Solà eq. 282-283)."""
    return NominalState(
        pos=x.pos + dx[POS],
        vel=x.vel + dx[VEL],
        quat=quat.normalize(quat.multiply(x.quat, quat.from_rotation_vector(dx[ATT]))),
        accel_bias=x.accel_bias + dx[ABIAS],
        gyro_bias=x.gyro_bias + dx[GBIAS],
    )


def error_between(x: NominalState, y: NominalState) -> np.ndarray:
    """y ⊖ x: the dx with inject(x, dx) == y."""
    dx = np.empty(STATE_DIM)
    dx[POS] = y.pos - x.pos
    dx[VEL] = y.vel - x.vel
    dx[ATT] = quat.to_rotation_vector(quat.multiply(quat.conjugate(x.quat), y.quat))
    dx[ABIAS] = y.accel_bias - x.accel_bias
    dx[GBIAS] = y.gyro_bias - x.gyro_bias
    return dx


class ErrorStateEKF:
    name = "ekf"

    def __init__(self, init: InitialState, cfg: EkfConfig, gravity: np.ndarray = GRAVITY_W) -> None:
        self._cfg = cfg
        self._gravity = gravity
        self._x = NominalState(
            pos=init.pos.copy(),
            vel=init.vel.copy(),
            quat=quat.normalize(init.quat),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
        )
        stds = np.concatenate(
            [
                np.full(3, cfg.pos_std0),
                np.full(3, cfg.vel_std0),
                np.full(3, cfg.att_std0),
                np.full(3, cfg.accel_bias_std0),
                np.full(3, cfg.gyro_bias_std0),
            ]
        )
        self._P = np.diag(stds**2)

    @property
    def nominal(self) -> NominalState:
        return self._x

    @property
    def covariance(self) -> np.ndarray:
        """Full 15x15 error-state covariance."""
        return self._P

    def predict(self, imu: ImuSample, dt: float) -> None:
        F = transition_jacobian(self._x, imu, dt)
        self._P = F @ self._P @ F.T + process_noise(self._cfg, dt)
        self._x = propagate_nominal(self._x, imu, dt, self._gravity)

    def update(self, fix: PosSample) -> None:
        H = np.zeros((3, STATE_DIM))
        H[:, POS] = np.eye(3)
        R = self._cfg.pos_meas_std**2 * np.eye(3)
        K, P_new = joseph_update(self._P, H, R)
        dx = K @ (fix.pos - self._x.pos)
        self._x = inject(self._x, dx)
        # Reset of the error state (Solà eq. 285-286); only the attitude block is non-identity.
        G = np.eye(STATE_DIM)
        G[ATT, ATT] = np.eye(3) - quat.skew(0.5 * dx[ATT])
        self._P = G @ P_new @ G.T

    @property
    def estimate(self) -> Estimate:
        return Estimate(
            pos=self._x.pos, vel=self._x.vel, quat=self._x.quat, cov_pv=self._P[:6, :6].copy()
        )
