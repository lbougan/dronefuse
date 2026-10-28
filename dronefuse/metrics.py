"""Evaluation metrics: RMSE, NEES consistency, dropout recovery."""

import numpy as np


def rmse(err: np.ndarray) -> float:
    """Root mean square of the Euclidean norm of (N, k) error vectors."""
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
