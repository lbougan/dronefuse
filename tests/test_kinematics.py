import numpy as np

from dronefuse import quaternion as quat
from dronefuse.estimators.kinematics import propagate
from dronefuse.sensors import GRAVITY_W


def test_free_fall_with_zero_specific_force() -> None:
    pos, vel, q = propagate(
        np.zeros(3), np.zeros(3), quat.IDENTITY, np.zeros(3), np.zeros(3), 0.1, GRAVITY_W
    )
    np.testing.assert_allclose(vel, [0.0, 0.0, -0.981])
    np.testing.assert_allclose(pos, [0.0, 0.0, -0.5 * 9.81 * 0.01])
    np.testing.assert_allclose(q, quat.IDENTITY)


def test_hover_specific_force_cancels_gravity() -> None:
    pos, vel, _ = propagate(
        np.zeros(3),
        np.zeros(3),
        quat.IDENTITY,
        np.array([0.0, 0.0, 9.81]),
        np.zeros(3),
        0.1,
        GRAVITY_W,
    )
    np.testing.assert_allclose(vel, 0.0, atol=1e-15)
    np.testing.assert_allclose(pos, 0.0, atol=1e-15)


def test_gyro_integrates_yaw() -> None:
    _, _, q = propagate(
        np.zeros(3),
        np.zeros(3),
        quat.IDENTITY,
        np.zeros(3),
        np.array([0.0, 0.0, 0.5]),
        0.2,
        GRAVITY_W,
    )
    np.testing.assert_allclose(q, quat.from_yaw(0.1), atol=1e-12)
