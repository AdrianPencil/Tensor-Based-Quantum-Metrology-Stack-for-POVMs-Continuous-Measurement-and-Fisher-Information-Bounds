"""
Fisher information utilities.

Primary use:
- Gaussian IID readout where the mean signal mu(theta) is differentiable.

Key identity (independent samples, known sigma):
    I(theta) = (1/sigma^2) * J^T J
where J_ij = d mu_i / d theta_j.
"""

import numpy as np
import numpy.typing as npt

__all__ = [
    "fisher_gaussian_iid",
    "fisher_scalar_gaussian_iid",
]


def fisher_gaussian_iid(
    jac: npt.NDArray[np.float64],
    sigma: float,
) -> npt.NDArray[np.float64]:
    """
    Fisher information matrix for Gaussian IID with known sigma.

    Args:
        jac: Jacobian J = dmu/dtheta of shape (n, p).
        sigma: standard deviation of readout noise (> 0).

    Returns:
        Fisher information matrix I of shape (p, p).
    """
    j = np.asarray(jac, dtype=np.float64)
    if j.ndim != 2:
        raise ValueError("jac must be 2D with shape (n, p).")
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("sigma must be > 0.")

    fim = (1.0 / (s * s)) * (j.T @ j)
    return fim.astype(np.float64, copy=False)


def fisher_scalar_gaussian_iid(
    dmu_dtheta: npt.NDArray[np.float64],
    sigma: float,
) -> float:
    """
    Scalar Fisher information for a 1-parameter Gaussian IID model:

        I = (1/sigma^2) * sum_i (dmu_i/dtheta)^2

    Args:
        dmu_dtheta: derivative vector of shape (n,).
        sigma: standard deviation (> 0).

    Returns:
        scalar Fisher information.
    """
    g = np.asarray(dmu_dtheta, dtype=np.float64)
    if g.ndim != 1:
        raise ValueError("dmu_dtheta must be 1D.")
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("sigma must be > 0.")
    return float((1.0 / (s * s)) * np.sum(g * g))
