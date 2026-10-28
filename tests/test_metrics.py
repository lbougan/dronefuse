import numpy as np

from dronefuse.metrics import rmse


def test_rmse_of_vector_errors() -> None:
    err = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])  # norms 5 and 0
    assert np.isclose(rmse(err), np.sqrt((25.0 + 0.0) / 2))
