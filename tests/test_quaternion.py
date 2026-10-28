import numpy as np

from dronefuse import quaternion as quat

THETA = np.array([0.3, -0.2, 0.5])


def test_identity_is_neutral_element() -> None:
    q = quat.from_rotation_vector(THETA)
    np.testing.assert_allclose(quat.multiply(quat.IDENTITY, q), q)
    np.testing.assert_allclose(quat.multiply(q, quat.IDENTITY), q)


def test_yaw_90deg_rotates_body_x_to_world_y() -> None:
    rot = quat.to_rotation_matrix(quat.from_yaw(np.pi / 2))
    np.testing.assert_allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_product_matches_rotation_matrix_composition() -> None:
    q = quat.from_rotation_vector(THETA)
    r = quat.from_rotation_vector(np.array([-0.1, 0.4, 0.2]))
    np.testing.assert_allclose(
        quat.to_rotation_matrix(quat.multiply(q, r)),
        quat.to_rotation_matrix(q) @ quat.to_rotation_matrix(r),
        atol=1e-12,
    )


def test_exp_log_roundtrip() -> None:
    np.testing.assert_allclose(
        quat.to_rotation_vector(quat.from_rotation_vector(THETA)), THETA, atol=1e-12
    )


def test_log_handles_negative_scalar_part() -> None:
    q = -quat.from_rotation_vector(THETA)
    np.testing.assert_allclose(quat.to_rotation_vector(q), THETA, atol=1e-12)


def test_small_angle_exp_is_first_order() -> None:
    q = quat.from_rotation_vector(np.array([1e-12, 0.0, 0.0]))
    np.testing.assert_allclose(q, [1.0, 5e-13, 0.0, 0.0], atol=1e-15)


def test_conjugate_is_inverse() -> None:
    q = quat.from_rotation_vector(THETA)
    np.testing.assert_allclose(quat.multiply(q, quat.conjugate(q)), quat.IDENTITY, atol=1e-12)


def test_rotation_matrix_is_orthonormal() -> None:
    rot = quat.to_rotation_matrix(quat.from_rotation_vector(THETA))
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) > 0


def test_skew_matches_cross_product() -> None:
    v = np.array([1.0, 2.0, 3.0])
    u = np.array([-0.5, 0.25, 2.0])
    np.testing.assert_allclose(quat.skew(v) @ u, np.cross(v, u))
