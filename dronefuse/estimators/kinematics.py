"""Strapdown integration step shared by dead reckoning and the EKF nominal state."""

import numpy as np

from dronefuse import quaternion as quat


def propagate(
    pos: np.ndarray,
    vel: np.ndarray,
    q: np.ndarray,
    accel_body: np.ndarray,
    gyro_body: np.ndarray,
    dt: float,
    gravity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate one IMU step (Solà eq. 260-262). Returns new (pos, vel, quat)."""
    acc_world = quat.to_rotation_matrix(q) @ accel_body + gravity
    new_pos = pos + vel * dt + 0.5 * acc_world * dt * dt
    new_vel = vel + acc_world * dt
    new_q = quat.normalize(quat.multiply(q, quat.from_rotation_vector(gyro_body * dt)))
    return new_pos, new_vel, new_q
