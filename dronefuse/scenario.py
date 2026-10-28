"""Analytic ground-truth trajectories, sampled at any time t. No physics engine."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from dronefuse import quaternion as quat

_SPEED_EPS = 1e-6
_FD_STEP = 1e-3  # s, central-difference step used for the angular rate


@dataclass(frozen=True)
class TrueState:
    """Ground truth at one instant. World frame is z-up; quat is body -> world."""

    t: float
    pos: np.ndarray  # (3,) m
    vel: np.ndarray  # (3,) m/s
    acc: np.ndarray  # (3,) m/s^2, world frame, gravity NOT included
    quat: np.ndarray  # (4,) (w, x, y, z)
    omega: np.ndarray  # (3,) rad/s, body frame


class Scenario(ABC):
    """A scripted path. Subclasses provide analytic position/velocity/acceleration."""

    name: str = "base"

    @abstractmethod
    def kinematics(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pos, vel, acc) at time t in the world frame."""

    def orientation(self, t: float) -> np.ndarray:
        """Body x axis follows the horizontal velocity (yaw only); level when static."""
        _, vel, _ = self.kinematics(t)
        if np.hypot(vel[0], vel[1]) < _SPEED_EPS:
            return quat.IDENTITY.copy()
        return quat.from_yaw(float(np.arctan2(vel[1], vel[0])))

    def angular_rate(self, t: float) -> np.ndarray:
        """Body-frame angular rate from a central difference of the orientation."""
        q_prev = self.orientation(t - _FD_STEP)
        q_next = self.orientation(t + _FD_STEP)
        delta = quat.multiply(quat.conjugate(q_prev), q_next)
        return quat.to_rotation_vector(delta) / (2.0 * _FD_STEP)

    def sample(self, t: float) -> TrueState:
        pos, vel, acc = self.kinematics(t)
        return TrueState(
            t=t, pos=pos, vel=vel, acc=acc, quat=self.orientation(t), omega=self.angular_rate(t)
        )


class Hover(Scenario):
    name = "hover"

    def __init__(self, altitude: float = 2.0) -> None:
        self.altitude = altitude

    def kinematics(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array([0.0, 0.0, self.altitude]), np.zeros(3), np.zeros(3)


class Line(Scenario):
    name = "line"

    def __init__(self, speed: float = 2.0, altitude: float = 2.0) -> None:
        self.speed = speed
        self.altitude = altitude

    def kinematics(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.array([self.speed * t, 0.0, self.altitude])
        vel = np.array([self.speed, 0.0, 0.0])
        return pos, vel, np.zeros(3)


class FigureEight(Scenario):
    """Lissajous figure-eight x = a sin(wt), y = b sin(2wt), with a gentle altitude bob."""

    name = "figure8"

    def __init__(
        self,
        a: float = 10.0,
        b: float = 5.0,
        period: float = 30.0,
        altitude: float = 5.0,
        bob: float = 0.5,
    ) -> None:
        self.a = a
        self.b = b
        self.altitude = altitude
        self.bob = bob
        self.w = 2.0 * np.pi / period

    def kinematics(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.w
        pos = np.array(
            [
                self.a * np.sin(w * t),
                self.b * np.sin(2 * w * t),
                self.altitude + self.bob * np.sin(w * t),
            ]
        )
        vel = np.array(
            [
                self.a * w * np.cos(w * t),
                2 * self.b * w * np.cos(2 * w * t),
                self.bob * w * np.cos(w * t),
            ]
        )
        acc = np.array(
            [
                -self.a * w * w * np.sin(w * t),
                -4 * self.b * w * w * np.sin(2 * w * t),
                -self.bob * w * w * np.sin(w * t),
            ]
        )
        return pos, vel, acc


SCENARIOS: dict[str, type[Scenario]] = {
    Hover.name: Hover,
    Line.name: Line,
    FigureEight.name: FigureEight,
}


def make_scenario(name: str) -> Scenario:
    """Instantiate a scenario by name with its default parameters."""
    try:
        return SCENARIOS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}") from exc
