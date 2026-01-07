"""
Workflow-level experiment runners.

This module provides small, end-to-end experiment functions that connect:
- protocol model -> mean signal + Jacobian
- likelihood/Fisher -> sensitivity metrics

It is intentionally conservative: one "canonical" Ramsey runner is enough
to make the whole stack usable early, and it becomes a reference point for tests.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..estimation.fisher import fisher_scalar_gaussian_iid
from ..estimation.crb import crb_variance_scalar
from ..protocols.ramsey import RamseyParams, ramsey_signal_and_jac_omega

__all__ = [
    "RamseyExperimentResult",
    "run_ramsey_frequency_experiment",
]


@dataclass(frozen=True, slots=True)
class RamseyExperimentResult:
    """
    Results of a 1-parameter Ramsey frequency experiment.

    Attributes:
        t_s: time samples (n,)
        mu: mean signal (n,)
        dmu_domega: derivative (n,)
        fisher_omega: scalar Fisher information for omega
        crb_var_omega: CRB variance lower bound for omega
    """

    t_s: npt.NDArray[np.float64]
    mu: npt.NDArray[np.float64]
    dmu_domega: npt.NDArray[np.float64]
    fisher_omega: float
    crb_var_omega: float


def run_ramsey_frequency_experiment(
    t_s: npt.NDArray[np.float64],
    omega: float,
    params: RamseyParams,
    sigma_y: float,
) -> RamseyExperimentResult:
    """
    Run a Ramsey frequency-sensing experiment (omega is the parameter).

    This returns a complete estimation-ready package:
    - mean signal mu(t)
    - derivative dmu/domega
    - Fisher information and CRB variance bound assuming Gaussian IID noise

    Args:
        t_s: time samples (n,).
        omega: angular frequency in rad/s.
        params: RamseyParams (dephasing, phase offset, visibility).
        sigma_y: readout noise std (> 0) for the Gaussian model.

    Returns:
        RamseyExperimentResult.
    """
    tt = np.asarray(t_s, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError("t_s must be 1D.")
    s = float(sigma_y)
    if s <= 0.0:
        raise ValueError("sigma_y must be > 0.")

    mu, dmu = ramsey_signal_and_jac_omega(t=tt, omega=float(omega), params=params)
    fisher = fisher_scalar_gaussian_iid(dmu_dtheta=dmu, sigma=s)
    crb_var = crb_variance_scalar(fisher)

    return RamseyExperimentResult(
        t_s=tt,
        mu=mu,
        dmu_domega=dmu,
        fisher_omega=float(fisher),
        crb_var_omega=float(crb_var),
    )
