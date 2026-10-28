import subprocess
import sys
from pathlib import Path

import pytest

from dronefuse.cli import build_parser, main, parse_dropout, parse_floats, parse_ints


def test_parse_dropout() -> None:
    assert parse_dropout("30:35") == (30.0, 35.0)
    with pytest.raises(Exception, match="START:END"):
        parse_dropout("30-35")
    with pytest.raises(Exception, match="end must be after start"):
        parse_dropout("35:30")


def test_parse_lists() -> None:
    assert parse_floats("0.5, 1,2") == [0.5, 1.0, 2.0]
    assert parse_ints("0,1,2") == [0, 1, 2]
    with pytest.raises(Exception, match="comma-separated"):
        parse_floats("a,b")


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["run"])
    assert args.scenario == "figure8"
    assert args.estimator == "all"
    assert args.seed == 0
    assert args.duration == 60.0
    assert args.dropout == []
    assert args.out == Path("out")


def test_main_writes_metrics_and_plots(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "run",
            "--scenario",
            "hover",
            "--estimator",
            "all",
            "--duration",
            "2",
            "--dropout",
            "0.5:1.0",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| Estimator |" in out
    assert "recovery after dropout" in out
    assert (tmp_path / "metrics.md").exists()
    assert (tmp_path / "hero_hover.png").exists()
    assert (tmp_path / "nees.png").exists()


def test_main_sweep_writes_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "sweep",
            "--knob",
            "kf.accel_noise_std",
            "--values",
            "0.5,2",
            "--scale",
            "--estimator",
            "kf",
            "--scenario",
            "hover",
            "--seeds",
            "0,1",
            "--duration",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| kf.accel_noise_std | Estimator |" in out
    assert (tmp_path / "sweep_kf.accel_noise_std.md").exists()


def test_cli_runs_as_module(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dronefuse",
            "run",
            "--estimator",
            "ekf",
            "--duration",
            "1",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "EKF" in proc.stdout
