"""Kalman update in Joseph form, shared by the linear KF and the EKF."""

import numpy as np


def joseph_update(P: np.ndarray, H: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (K, P_posterior) for prior covariance P, measurement Jacobian H, noise R.

    Joseph form keeps P symmetric positive semi-definite under round-off.
    """
    S = H @ P @ H.T + R
    K = np.linalg.solve(S.T, H @ P).T  # = P Hᵀ S⁻¹ (P symmetric)
    i_kh = np.eye(P.shape[0]) - K @ H
    P_new = i_kh @ P @ i_kh.T + K @ R @ K.T
    return K, 0.5 * (P_new + P_new.T)
