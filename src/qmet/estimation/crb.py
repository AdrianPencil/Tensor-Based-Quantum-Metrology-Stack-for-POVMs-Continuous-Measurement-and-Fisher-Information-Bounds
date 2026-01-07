"""
Cramér–Rao bound (CRB) utilities.

Given Fisher information matrix I(theta), the covariance of any unbiased estimator
satisfies:
    Cov >= I^{-1}

This module provides numerically cautious inversions with a small diagonal ridge
when needed.
"""

import numpy as np
import numpy.typing as npt

__all__ = [
    "crb_covariance",
    "crb_variance_scalar",
]


def crb_covariance(
    fisher: npt.NDArray[np.float64],
    ridge: float = 0.0,
) -> npt.NDArray[np.float64]:
    """
    Compute CRB covariance matrix as inv(Fisher + ridge * I).

    Args:
        fisher: Fisher information matrix (p, p).
        ridge: nonnegative diagonal regularizer.

    Returns:
        covariance lower bound (p, p).
    """
    fim = np.asarray(fisher, dtype=np.float64)
    if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
        raise ValueError("fisher must be a square (p, p) matrix.")

    r = float(ridge)
    if r < 0.0:
        raise ValueError("ridge must be >= 0.")

    a = fim
    if r > 0.0:
        a = fim + r * np.eye(fim.shape[0], dtype=np.float64)

    try:
        cov = np.linalg.inv(a)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("Fisher matrix inversion failed; consider ridge > 0.") from exc

    return cov.astype(np.float64, copy=False)


def crb_variance_scalar(fisher_scalar: float) -> float:
    """
    Scalar CRB: Var(theta_hat) >= 1 / I(theta) for unbiased estimators.

    Args:
        fisher_scalar: Fisher information (must be > 0).

    Returns:
        variance lower bound.
    """
    i = float(fisher_scalar)
    if i <= 0.0:
        raise ValueError("fisher_scalar must be > 0.")
    return float(1.0 / i)
