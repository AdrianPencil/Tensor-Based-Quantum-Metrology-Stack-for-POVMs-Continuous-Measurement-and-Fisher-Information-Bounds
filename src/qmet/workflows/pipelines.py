"""
Pipelines compose protocol + noise + estimation into a repeatable computation unit.

This file keeps a small number of "ready to run" pipelines that can be called from:
- CLI entrypoints
- tests
- flagship experiments
- reports

Initial scope: a Ramsey frequency pipeline that can optionally simulate noisy data.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..protocols.ramsey import RamseyParams, ramsey_signal_and_jac_omega
from ..estimation.fisher import fisher_scalar_gaussian_iid
from ..estimation.crb import crb_variance_scalar

__all__ = [
    "RamseyFrequencyPipelineResult",
    "run_ramsey_frequency_pipeline",
]


@dataclass(frozen=True, slots=True)
class RamseyFrequencyPipelineResult:
    """
    Output of the Ramsey frequency pipeline for parameter omega.

    If y is None, only the deterministic mean model and information bounds are produced.
    """

    t_s: npt.NDArray[np.float64]
    mu: npt.NDArray[np.float64]
    dmu_domega: npt.NDArray[np.float64]
    sigma_y: float
    fisher_omega: float
    crb_var_omega: float
    y: Optional[npt.NDArray[np.float64]] = None


def run_ramsey_frequency_pipeline(
    t_s: npt.NDArray[np.float64],
    omega: float,
    params: RamseyParams,
    sigma_y: float,
    rng: Optional[np.random.Generator] = None,
) -> RamseyFrequencyPipelineResult:
    """
    Run a compact Ramsey pipeline:
      mu(t), dmu/domega, Fisher, CRB; optionally simulate noisy observations.

    Args:
        t_s: time samples (n,).
        omega: angular frequency (rad/s).
        params: RamseyParams.
        sigma_y: Gaussian readout std (> 0).
        rng: if provided, simulate y = mu + N(0, sigma_y^2).

    Returns:
        RamseyFrequencyPipelineResult.
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

    y = None
    if rng is not None:
        y = (mu + rng.normal(loc=0.0, scale=s, size=mu.shape)).astype(np.float64, copy=False)

    return RamseyFrequencyPipelineResult(
        t_s=tt,
        mu=mu,
        dmu_domega=dmu,
        sigma_y=s,
        fisher_omega=float(fisher),
        crb_var_omega=float(crb_var),
        y=y,
    )
