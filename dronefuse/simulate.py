"""Simulation harness: sample truth, generate a sensor log, drive an estimator over it."""

from dataclasses import dataclass

import numpy as np

from dronefuse import quaternion as quat
from dronefuse.estimators.base import Estimator, InitConfig, InitialState
from dronefuse.scenario import Scenario, TrueState
from dronefuse.sensors import ImuSample, PosSample, SensorConfig, SensorSim


@dataclass(frozen=True)
class Truth:
    t: np.ndarray  # (N,)
    pos: np.ndarray  # (N, 3)
    vel: np.ndarray  # (N, 3)
    acc: np.ndarray  # (N, 3)
    quat: np.ndarray  # (N, 4)
    omega: np.ndarray  # (N, 3)

    def __len__(self) -> int:
        return len(self.t)

    def state(self, i: int) -> TrueState:
        return TrueState(
            t=float(self.t[i]),
            pos=self.pos[i],
            vel=self.vel[i],
            acc=self.acc[i],
            quat=self.quat[i],
            omega=self.omega[i],
        )


@dataclass(frozen=True)
class SensorLog:
    """All sensor samples for one run. fix rows are NaN where no fix arrived."""

    t: np.ndarray  # (N,)
    dt: float
    accel: np.ndarray  # (N, 3)
    gyro: np.ndarray  # (N, 3)
    fix: np.ndarray  # (N, 3), NaN rows = no fix
    accel_bias: np.ndarray  # (N, 3) true bias, for diagnostics only
    gyro_bias: np.ndarray  # (N, 3)

    def __len__(self) -> int:
        return len(self.t)

    def imu(self, i: int) -> ImuSample:
        return ImuSample(t=float(self.t[i]), accel=self.accel[i], gyro=self.gyro[i])

    def fix_at(self, i: int) -> PosSample | None:
        if np.isnan(self.fix[i, 0]):
            return None
        return PosSample(t=float(self.t[i]), pos=self.fix[i])


@dataclass(frozen=True)
class EstimateTrajectory:
    name: str
    pos: np.ndarray  # (N, 3)
    vel: np.ndarray  # (N, 3)
    cov_pv: np.ndarray | None  # (N, 6, 6) or None


def sample_truth(scenario: Scenario, duration: float, rate_hz: float) -> Truth:
    """Ground truth on the IMU grid t = 0, 1/rate, ..., duration (inclusive)."""
    t = np.arange(int(round(duration * rate_hz)) + 1) / rate_hz
    states = [scenario.sample(float(ti)) for ti in t]
    return Truth(
        t=t,
        pos=np.stack([s.pos for s in states]),
        vel=np.stack([s.vel for s in states]),
        acc=np.stack([s.acc for s in states]),
        quat=np.stack([s.quat for s in states]),
        omega=np.stack([s.omega for s in states]),
    )


def simulate_sensors(truth: Truth, cfg: SensorConfig, seed: int) -> SensorLog:
    """IMU at every truth sample, a position fix every imu_rate/fix_rate samples."""
    sim = SensorSim(cfg, seed)
    fix_every = int(round(cfg.imu_rate_hz / cfg.fix_rate_hz))
    n = len(truth)
    accel = np.empty((n, 3))
    gyro = np.empty((n, 3))
    fix = np.full((n, 3), np.nan)
    accel_bias = np.empty((n, 3))
    gyro_bias = np.empty((n, 3))
    for i in range(n):
        s = truth.state(i)
        accel_bias[i] = sim.accel_bias
        gyro_bias[i] = sim.gyro_bias
        imu = sim.imu(s)
        accel[i] = imu.accel
        gyro[i] = imu.gyro
        if i % fix_every == 0:
            sample = sim.position_fix(s)
            if sample is not None:
                fix[i] = sample.pos
    return SensorLog(
        t=truth.t,
        dt=1.0 / cfg.imu_rate_hz,
        accel=accel,
        gyro=gyro,
        fix=fix,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )


def perturbed_initial_state(truth: Truth, cfg: InitConfig, seed: int) -> InitialState:
    """Truth at t=0 plus a Gaussian error drawn from InitConfig, so P0 is honest."""
    rng = np.random.default_rng(seed)
    dtheta = rng.normal(0.0, cfg.att_std, 3)
    return InitialState(
        pos=truth.pos[0] + rng.normal(0.0, cfg.pos_std, 3),
        vel=truth.vel[0] + rng.normal(0.0, cfg.vel_std, 3),
        quat=quat.multiply(truth.quat[0], quat.from_rotation_vector(dtheta)),
    )


def run_estimator(estimator: Estimator, log: SensorLog) -> EstimateTrajectory:
    """Drive one estimator over the log. Estimate i is after predict (i>0) and after update."""
    n = len(log)
    pos = np.empty((n, 3))
    vel = np.empty((n, 3))
    cov = np.empty((n, 6, 6)) if estimator.estimate.cov_pv is not None else None
    for i in range(n):
        if i > 0:
            estimator.predict(log.imu(i - 1), log.dt)
        fix = log.fix_at(i)
        if fix is not None:
            estimator.update(fix)
        est = estimator.estimate
        pos[i] = est.pos
        vel[i] = est.vel
        if cov is not None:
            cov[i] = est.cov_pv
    return EstimateTrajectory(name=estimator.name, pos=pos, vel=vel, cov_pv=cov)
