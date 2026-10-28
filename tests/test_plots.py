from pathlib import Path

import pytest

from dronefuse.pipeline import RunResult, run_pipeline
from dronefuse.plots import plot_errors, plot_hero, plot_nees, write_plots


@pytest.fixture(scope="module")
def result() -> RunResult:
    return run_pipeline("figure8", ["dr", "kf", "ekf"], seed=0, duration=3.0, dropouts=[(1.0, 2.0)])


def test_plot_hero_writes_png(result: RunResult, tmp_path: Path) -> None:
    path = tmp_path / "hero.png"
    plot_hero(result.truth, result.trajectories, result.dropouts, path)
    assert path.stat().st_size > 10_000


def test_plot_errors_writes_png(result: RunResult, tmp_path: Path) -> None:
    path = tmp_path / "errors.png"
    plot_errors(result.truth, result.trajectories["ekf"], result.dropouts, path)
    assert path.exists()


def test_plot_errors_rejects_estimator_without_covariance(
    result: RunResult, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no covariance"):
        plot_errors(result.truth, result.trajectories["dr"], result.dropouts, tmp_path / "x.png")


def test_plot_nees_writes_png(result: RunResult, tmp_path: Path) -> None:
    path = tmp_path / "nees.png"
    nees_by_name = {n: m.nees for n, m in result.metrics.items() if m.nees is not None}
    plot_nees(result.truth.t, nees_by_name, result.dropouts, path)
    assert path.exists()


def test_write_plots_skips_estimators_without_covariance(result: RunResult, tmp_path: Path) -> None:
    written = write_plots(result, tmp_path)
    names = sorted(p.name for p in written)
    assert names == ["errors_ekf.png", "errors_kf.png", "hero_figure8.png", "nees.png"]
