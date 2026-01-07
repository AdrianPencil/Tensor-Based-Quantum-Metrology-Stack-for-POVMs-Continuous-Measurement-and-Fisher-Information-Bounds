"""
Sensitivity metrics.

This module keeps "how sensitive is the readout to a parameter?" in a small set
of numerically stable utilities, primarily for Gaussian readout.

Two common forms:
1) slope/noise sensitivity:
      sigma_theta ≈ sigma_y / sqrt(sum (dmu/dtheta)^2)
2) Fisher-limited bound (CRB):
      Var(theta_hat) >= 1 / I(theta)
"""

import numpy as np
import numpy.typing as npt

from ..estimation.fisher import fisher_scalar_gaussian_iid
from ..estimation.crb import crb_variance_scalar

__all__ = [
    "sigma_theta_from_slope",
    "crb_sigma_theta_gaussian_iid",
]


def sigma_theta_from_slope(
    dmu_dtheta: npt.NDArray[np.float64],
    sigma_y: float,
) -> float:
    """
    Sensitivity estimate for a 1-parameter model with Gaussian IID noise.

    If y_i = mu_i(theta) + noise, noise ~ N(0, sigma_y^2), then a local bound is:
        sigma_theta ≈ sigma_y / sqrt(sum_i (dmu_i/dtheta)^2)

    Args:
        dmu_dtheta: derivative array (n,).
        sigma_y: readout standard deviation (> 0).

    Returns:
        sigma_theta (standard deviation estimate) as float.
    """
    g = np.asarray(dmu_dtheta, dtype=np.float64)
    if g.ndim != 1:
        raise ValueError("dmu_dtheta must be 1D.")
    s = float(sigma_y)
    if s <= 0.0:
        raise ValueError("sigma_y must be > 0.")
    denom = float(np.sqrt(np.sum(g * g)))
    if denom == 0.0:
        return float(np.inf)
    return float(s / denom)


def crb_sigma_theta_gaussian_iid(
    dmu_dtheta: npt.NDArray[np.float64],
    sigma_y: float,
) -> float:
    """
    Fisher/CRB-limited sigma_theta for a single parameter in Gaussian IID model.

    Args:
        dmu_dtheta: derivative array (n,).
        sigma_y: readout standard deviation (> 0).

    Returns:
        sqrt(CRB variance) as float.
    """
    i = fisher_scalar_gaussian_iid(dmu_dtheta=dmu_dtheta, sigma=sigma_y)
    v = crb_variance_scalar(i)
    return float(np.sqrt(v))
