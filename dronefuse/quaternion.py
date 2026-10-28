"""Unit-quaternion utilities.

Convention (Solà 2017): Hamilton product, components (w, x, y, z), and q is the
rotation body -> world, i.e. v_world = to_rotation_matrix(q) @ v_body.
"""

import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
_SMALL_ANGLE = 1e-9


def normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm."""
    return q / np.linalg.norm(q)


def multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product q ⊗ r."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate, which is the inverse for a unit quaternion."""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R with v_world = R @ v_body."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def from_rotation_vector(theta: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector (axis * angle) -> unit quaternion (Solà eq. 101)."""
    angle = float(np.linalg.norm(theta))
    if angle < _SMALL_ANGLE:
        return normalize(np.array([1.0, *(0.5 * theta)]))
    axis = theta / angle
    return np.concatenate([[np.cos(0.5 * angle)], axis * np.sin(0.5 * angle)])


def to_rotation_vector(q: np.ndarray) -> np.ndarray:
    """Logarithmic map: unit quaternion -> rotation vector with angle in [0, pi]."""
    if q[0] < 0:
        q = -q
    vec = q[1:]
    norm = float(np.linalg.norm(vec))
    if norm < _SMALL_ANGLE:
        return 2.0 * vec
    angle = 2.0 * np.arctan2(norm, q[0])
    return angle * vec / norm


def from_yaw(yaw: float) -> np.ndarray:
    """Quaternion for a pure rotation about the world z axis."""
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]x such that [v]x @ u == cross(v, u)."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
