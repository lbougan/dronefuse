import numpy as np

from dronefuse import quaternion as quat
from dronefuse.scenario import FigureEight, Hover
from dronefuse.sensors import GRAVITY_W, SensorConfig, SensorSim


def zero_noise(**overrides: object) -> SensorConfig:
    base = dict(
        accel_noise_std=0.0,
        gyro_noise_std=0.0,
        accel_bias_std0=0.0,
        gyro_bias_std0=0.0,
        accel_bias_walk_std=0.0,
        gyro_bias_walk_std=0.0,
        pos_noise_std=0.0,
    )
    return SensorConfig(**{**base, **overrides})  # type: ignore[arg-type]


def test_default_config_matches_spec_rates() -> None:
    cfg = SensorConfig()
    assert cfg.imu_rate_hz == 200.0
    assert cfg.fix_rate_hz == 5.0


def test_noise_free_imu_at_hover_measures_gravity_up() -> None:
    sim = SensorSim(zero_noise(), seed=0)
    imu = sim.imu(Hover().sample(0.0))
    np.testing.assert_allclose(imu.accel, [0.0, 0.0, 9.81])
    np.testing.assert_allclose(imu.gyro, 0.0, atol=1e-12)


def test_noise_free_imu_on_figure8_recovers_world_acceleration() -> None:
    sim = SensorSim(zero_noise(), seed=0)
    s = FigureEight().sample(4.0)
    imu = sim.imu(s)
    rot = quat.to_rotation_matrix(s.quat)
    np.testing.assert_allclose(rot @ imu.accel + GRAVITY_W, s.acc, atol=1e-12)
    np.testing.assert_allclose(imu.gyro, s.omega)


def test_noise_free_fix_equals_truth() -> None:
    sim = SensorSim(zero_noise(), seed=0)
    s = FigureEight().sample(4.0)
    fix = sim.position_fix(s)
    assert fix is not None
    np.testing.assert_allclose(fix.pos, s.pos)
    assert fix.t == 4.0


def test_fix_is_none_during_dropout_and_returns_after() -> None:
    sim = SensorSim(zero_noise(dropouts=((10.0, 15.0),)), seed=0)
    sc = FigureEight()
    assert sim.position_fix(sc.sample(9.9)) is not None
    assert sim.position_fix(sc.sample(10.0)) is None
    assert sim.position_fix(sc.sample(12.0)) is None
    assert sim.position_fix(sc.sample(15.0)) is not None


def test_same_seed_gives_same_samples() -> None:
    s = FigureEight().sample(1.0)
    a = SensorSim(SensorConfig(), seed=3).imu(s)
    b = SensorSim(SensorConfig(), seed=3).imu(s)
    np.testing.assert_array_equal(a.accel, b.accel)
    np.testing.assert_array_equal(a.gyro, b.gyro)


def test_accel_noise_has_configured_std() -> None:
    cfg = zero_noise(accel_noise_std=0.05)
    sim = SensorSim(cfg, seed=0)
    s = Hover().sample(0.0)
    samples = np.array([sim.imu(s).accel for _ in range(20_000)])
    np.testing.assert_allclose(samples.std(axis=0), 0.05, rtol=0.05)


def test_bias_random_walk_moves_bias() -> None:
    sim = SensorSim(zero_noise(accel_bias_walk_std=1.0), seed=0)
    s = Hover().sample(0.0)
    before = sim.accel_bias.copy()
    for _ in range(100):
        sim.imu(s)
    assert np.linalg.norm(sim.accel_bias - before) > 0.0
