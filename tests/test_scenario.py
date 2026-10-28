import numpy as np
import pytest

from dronefuse import quaternion as quat
from dronefuse.scenario import SCENARIOS, FigureEight, Hover, Line, make_scenario

FD_STEP = 1e-4


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("t", [0.0, 3.7, 17.2])
def test_velocity_and_acceleration_are_derivatives_of_position(name: str, t: float) -> None:
    sc = make_scenario(name)
    p_prev, v_prev, _ = sc.kinematics(t - FD_STEP)
    p_next, v_next, _ = sc.kinematics(t + FD_STEP)
    _, vel, acc = sc.kinematics(t)
    np.testing.assert_allclose((p_next - p_prev) / (2 * FD_STEP), vel, atol=1e-6)
    np.testing.assert_allclose((v_next - v_prev) / (2 * FD_STEP), acc, atol=1e-6)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_sample_returns_unit_quaternion_and_shapes(name: str) -> None:
    s = make_scenario(name).sample(2.5)
    assert s.pos.shape == s.vel.shape == s.acc.shape == s.omega.shape == (3,)
    assert s.quat.shape == (4,)
    assert np.isclose(np.linalg.norm(s.quat), 1.0)


def test_hover_is_static_and_level() -> None:
    s = Hover(altitude=2.0).sample(10.0)
    np.testing.assert_allclose(s.pos, [0.0, 0.0, 2.0])
    np.testing.assert_allclose(s.vel, 0.0)
    np.testing.assert_allclose(s.quat, quat.IDENTITY)
    np.testing.assert_allclose(s.omega, 0.0, atol=1e-12)


def test_line_moves_along_x_with_zero_angular_rate() -> None:
    s = Line(speed=2.0, altitude=2.0).sample(3.0)
    np.testing.assert_allclose(s.pos, [6.0, 0.0, 2.0])
    np.testing.assert_allclose(s.omega, 0.0, atol=1e-12)


def test_figure8_body_x_points_along_horizontal_velocity() -> None:
    s = FigureEight().sample(2.0)
    body_x_in_world = quat.to_rotation_matrix(s.quat) @ np.array([1.0, 0.0, 0.0])
    heading = s.vel[:2] / np.linalg.norm(s.vel[:2])
    np.testing.assert_allclose(body_x_in_world[:2], heading, atol=1e-9)
    assert abs(body_x_in_world[2]) < 1e-12


def test_figure8_angular_rate_integrates_orientation() -> None:
    sc = FigureEight()
    dt = 1e-3
    s0 = sc.sample(4.0)
    q_pred = quat.multiply(s0.quat, quat.from_rotation_vector(s0.omega * dt))
    np.testing.assert_allclose(q_pred, sc.sample(4.0 + dt).quat, atol=1e-6)


def test_figure8_passes_through_origin_at_t0() -> None:
    s = FigureEight(altitude=5.0).sample(0.0)
    np.testing.assert_allclose(s.pos, [0.0, 0.0, 5.0])


def test_unknown_scenario_raises() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        make_scenario("spiral")
