"""Command-line entry point.

dronefuse run   --scenario figure8 --estimator all --seed 0 --dropout 30:35
dronefuse sweep --knob ekf.gyro_noise_std --values 1,2,4 --scale --seeds 0,1,2
"""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dronefuse.estimators import ESTIMATOR_NAMES
from dronefuse.metrics import format_metrics_table, recovery_time
from dronefuse.pipeline import RunResult, run_pipeline
from dronefuse.plots import write_plots
from dronefuse.scenario import SCENARIOS
from dronefuse.sweep import Knob, format_sweep_table, sweep

RECOVERY_THRESHOLD_M = 0.5


def parse_dropout(text: str) -> tuple[float, float]:
    """'30:35' -> (30.0, 35.0)."""
    try:
        start, end = (float(part) for part in text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected START:END, got {text!r}") from exc
    if end <= start:
        raise argparse.ArgumentTypeError("dropout end must be after start")
    return start, end


def parse_floats(text: str) -> list[float]:
    """'0.5, 1,2' -> [0.5, 1.0, 2.0]."""
    try:
        return [float(part) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from exc


def parse_ints(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from exc


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="figure8")
    parser.add_argument("--estimator", choices=[*ESTIMATOR_NAMES, "all"], default="all")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds")
    parser.add_argument(
        "--dropout",
        type=parse_dropout,
        action="append",
        default=[],
        metavar="START:END",
        help="position-fix dropout interval in seconds; repeatable",
    )
    parser.add_argument("--out", type=Path, default=Path("out"), help="output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dronefuse", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate, estimate, evaluate and plot")
    _add_common(run)
    run.add_argument("--seed", type=int, default=0)

    sw = sub.add_parser("sweep", help="rerun while one config knob varies, averaged over seeds")
    _add_common(sw)
    sw.add_argument(
        "--knob",
        type=Knob.parse,
        required=True,
        metavar="TARGET.FIELD",
        help="e.g. ekf.gyro_noise_std, kf.accel_noise_std, sensor.pos_noise_std, init.pos_std",
    )
    sw.add_argument("--values", type=parse_floats, required=True, metavar="V1,V2,...")
    sw.add_argument(
        "--scale", action="store_true", help="treat values as multipliers of the default"
    )
    sw.add_argument("--seeds", type=parse_ints, default=[0], metavar="S1,S2,...")
    return parser


def format_recovery_report(result: RunResult) -> str:
    """One line per estimator with a covariance: time to get back under the threshold."""
    lines = []
    for name, traj in result.trajectories.items():
        if traj.cov_pv is None:
            continue
        err = np.linalg.norm(traj.pos - result.truth.pos, axis=1)
        for _, end in result.dropouts:
            rec = recovery_time(result.truth.t, err, end, RECOVERY_THRESHOLD_M)
            rec_text = f"{rec:.2f} s" if rec is not None else "never"
            lines.append(
                f"{name.upper()}: recovery after dropout ending at {end:.1f} s "
                f"(error < {RECOVERY_THRESHOLD_M} m): {rec_text}"
            )
    return "\n".join(lines)


def _estimator_names(args: argparse.Namespace) -> list[str]:
    return list(ESTIMATOR_NAMES) if args.estimator == "all" else [args.estimator]


def cmd_run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    result = run_pipeline(
        args.scenario,
        _estimator_names(args),
        seed=args.seed,
        duration=args.duration,
        dropouts=args.dropout,
    )
    table = format_metrics_table(list(result.metrics.values()))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.md").write_text(table + "\n")
    written = write_plots(result, args.out)
    elapsed = time.perf_counter() - started
    print(f"scenario={args.scenario} seed={args.seed} duration={args.duration:g}s\n")
    print(table)
    if result.dropouts:
        print()
        print(format_recovery_report(result))
    print(f"\nwrote {len(written)} plots to {args.out}/ ({elapsed:.1f} s)")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    rows = sweep(
        args.knob,
        args.values,
        _estimator_names(args),
        scenario=args.scenario,
        seeds=args.seeds,
        duration=args.duration,
        dropouts=args.dropout,
        scale=args.scale,
    )
    table = format_sweep_table(args.knob, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"sweep_{args.knob.target}.{args.knob.field}.md"
    path.write_text(table + "\n")
    elapsed = time.perf_counter() - started
    print(f"scenario={args.scenario} seeds={args.seeds} duration={args.duration:g}s\n")
    print(table)
    print(f"\nwrote {path} ({elapsed:.1f} s)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sweep":
        return cmd_sweep(args)
    return cmd_run(args)
