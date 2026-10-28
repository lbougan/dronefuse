"""Parameter sweeps: rerun the pipeline while one knob varies, average over seeds.

A knob is written `target.field`, where target is `sensor` (SensorConfig), `init` (InitConfig),
`kf` (KfConfig) or `ekf` (EkfConfig). Sensor/init knobs change the simulation (and the filters
derived from it); kf/ekf knobs change only that filter, which is how to test robustness to a
mis-modelled noise level.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np

from dronefuse.estimators import default_config
from dronefuse.estimators.base import InitConfig
from dronefuse.metrics import NEES_DOF, Metrics, chi2_bounds
from dronefuse.pipeline import run_pipeline
from dronefuse.sensors import SensorConfig

SIM_TARGETS = ("sensor", "init")
FILTER_TARGETS = ("kf", "ekf")


@dataclass(frozen=True)
class Knob:
    target: str
    field: str

    @classmethod
    def parse(cls, text: str) -> "Knob":
        target, sep, field = text.partition(".")
        if not sep or target not in (*SIM_TARGETS, *FILTER_TARGETS):
            raise ValueError(
                f"knob must be 'target.field' with target in "
                f"{(*SIM_TARGETS, *FILTER_TARGETS)}, got {text!r}"
            )
        return cls(target, field)

    def default(self, sensor_cfg: SensorConfig, init_cfg: InitConfig) -> float:
        """Value of the field before any sweep, used by scaled sweeps."""
        cfg = {
            "sensor": sensor_cfg,
            "init": init_cfg,
            "kf": default_config("kf", sensor_cfg, init_cfg),
            "ekf": default_config("ekf", sensor_cfg, init_cfg),
        }[self.target]
        if self.field not in {f.name for f in fields(cfg)}:
            raise ValueError(f"{type(cfg).__name__} has no field {self.field!r}")
        value = getattr(cfg, self.field)
        if not isinstance(value, int | float):
            raise ValueError(f"knob {self.target}.{self.field} is not numeric ({value!r})")
        return float(value)


@dataclass(frozen=True)
class SweepRow:
    value: float
    estimator: str
    pos_rmse: float
    vel_rmse: float
    nees_mean: float | None
    nees_in_bounds: float | None
    nees_below: float | None  # fraction of timesteps under the lower chi-square bound
    nees_above: float | None  # fraction over the upper bound (over-confident filter)


def _summarise(value: float, name: str, runs: Sequence[Metrics]) -> SweepRow:
    lo, hi = chi2_bounds(NEES_DOF)
    has_nees = all(m.nees is not None for m in runs)
    nees_all = np.concatenate([m.nees for m in runs]) if has_nees else None  # type: ignore[misc]
    return SweepRow(
        value=value,
        estimator=name,
        pos_rmse=float(np.mean([m.pos_rmse for m in runs])),
        vel_rmse=float(np.mean([m.vel_rmse for m in runs])),
        nees_mean=float(nees_all.mean()) if nees_all is not None else None,
        nees_in_bounds=(
            float(np.mean((nees_all >= lo) & (nees_all <= hi))) if nees_all is not None else None
        ),
        nees_below=float(np.mean(nees_all < lo)) if nees_all is not None else None,
        nees_above=float(np.mean(nees_all > hi)) if nees_all is not None else None,
    )


def sweep(
    knob: Knob,
    values: Sequence[float],
    estimator_names: Sequence[str],
    scenario: str = "figure8",
    seeds: Sequence[int] = (0,),
    duration: float = 60.0,
    dropouts: Sequence[tuple[float, float]] = (),
    scale: bool = False,
) -> list[SweepRow]:
    """One row per (value, estimator), metrics averaged over seeds.

    With scale=True the values multiply the knob's default instead of replacing it.
    """
    if knob.target in FILTER_TARGETS and knob.target not in estimator_names:
        raise ValueError(
            f"knob {knob.target}.{knob.field} targets {knob.target!r}, "
            f"which is not among the estimators run {list(estimator_names)}"
        )
    base_sensor, base_init = SensorConfig(), InitConfig()
    default = knob.default(base_sensor, base_init)
    rows: list[SweepRow] = []
    for raw in values:
        value = raw * default if scale else raw
        sensor_cfg, init_cfg, overrides = base_sensor, base_init, {}
        if knob.target == "sensor":
            sensor_cfg = replace(base_sensor, **{knob.field: value})
        elif knob.target == "init":
            init_cfg = replace(base_init, **{knob.field: value})
        else:
            overrides = {knob.target: {knob.field: value}}
        results = [
            run_pipeline(
                scenario,
                estimator_names,
                seed=seed,
                duration=duration,
                dropouts=dropouts,
                sensor_cfg=sensor_cfg,
                init_cfg=init_cfg,
                overrides=overrides,
            )
            for seed in seeds
        ]
        for name in estimator_names:
            rows.append(_summarise(value, name, [r.metrics[name] for r in results]))
    return rows


def format_sweep_table(knob: Knob, rows: Sequence[SweepRow]) -> str:
    """Markdown table of a sweep."""
    lines = [
        f"| {knob.target}.{knob.field} | Estimator | Pos RMSE [m] | Vel RMSE [m/s] "
        "| Mean NEES | In bounds | Below | Above |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        pct = [
            f"{100 * v:.1f} %" if v is not None else "n/a"
            for v in (r.nees_in_bounds, r.nees_below, r.nees_above)
        ]
        mean = f"{r.nees_mean:.2f}" if r.nees_mean is not None else "n/a"
        lines.append(
            f"| {r.value:.4g} | {r.estimator.upper()} | {r.pos_rmse:.3f} | {r.vel_rmse:.3f} "
            f"| {mean} | {pct[0]} | {pct[1]} | {pct[2]} |"
        )
    return "\n".join(lines)
