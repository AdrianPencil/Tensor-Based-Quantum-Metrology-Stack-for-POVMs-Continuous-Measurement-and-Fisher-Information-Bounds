"""
Case study: NV-like Ramsey sensitivity.

This is intentionally "NV-inspired" rather than a full NV Hamiltonian simulator.
The goal is to give a clean, realistic workflow:
- define a Ramsey dephasing envelope (gamma_phi) and visibility
- choose a time grid
- compute mu(t), dmu/domega
- compute Fisher/CRB for omega under Gaussian readout noise
- identify the time that maximizes Fisher contribution (a practical design knob)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..protocols.ramsey import RamseyParams, ramsey_signal_and_jac_omega
from ..estimation.fisher import fisher_scalar_gaussian_iid

__all__ = [
    "NVRamseySensitivityConfig",
    "NVRamseySensitivityResult",
    "run_nv_ramsey_sensitivity",
]


@dataclass(frozen=True, slots=True)
class NVRamseySensitivityConfig:
    """
    Configuration for an NV-inspired Ramsey sensitivity sweep.

    omega: nominal angular frequency (rad/s)
    gamma_phi: pure dephasing rate (1/s)
    visibility: contrast factor in [0, 1]
    sigma_y: Gaussian readout standard deviation (> 0)

    t_s: if provided, use these time samples directly
    otherwise use a linspace defined by t_min_s, t_max_s, n
    """

    omega: float
    gamma_phi: float
    visibility: float
    sigma_y: float
    phase0: float = 0.0
    t_s: Optional[npt.NDArray[np.float64]] = None
    t_min_s: float = 0.0
    t_max_s: float = 2e-3
    n: int = 400

    def __post_init__(self) -> None:
        if float(self.gamma_phi) < 0.0:
            raise ValueError("gamma_phi must be >= 0.")
        if not (0.0 <= float(self.visibility) <= 1.0):
            raise ValueError("visibility must be in [0, 1].")
        if float(self.sigma_y) <= 0.0:
            raise ValueError("sigma_y must be > 0.")
        if self.t_s is None:
            if float(self.t_max_s) <= float(self.t_min_s):
                raise ValueError("t_max_s must be > t_min_s.")
            if int(self.n) < 2:
                raise ValueError("n must be >= 2.")
        else:
            tt = np.asarray(self.t_s, dtype=np.float64)
            if tt.ndim != 1 or tt.size < 2:
                raise ValueError("t_s must be a 1D array with length >= 2.")


@dataclass(frozen=True, slots=True)
class NVRamseySensitivityResult:
    """
    Output of the NV-inspired Ramsey sensitivity sweep.

    per_sample_fisher is the additive contribution per time sample:
        I_i = (1/sigma^2) * (dmu_i/domega)^2
    """

    t_s: npt.NDArray[np.float64]
    mu: npt.NDArray[np.float64]
    dmu_domega: npt.NDArray[np.float64]
    per_sample_fisher: npt.NDArray[np.float64]
    fisher_total: float
    crb_std_omega: float
    best_time_s: float
    best_index: int


def run_nv_ramsey_sensitivity(cfg: NVRamseySensitivityConfig) -> NVRamseySensitivityResult:
    """
    Run the NV-inspired Ramsey sensitivity study.

    Args:
        cfg: NVRamseySensitivityConfig.

    Returns:
        NVRamseySensitivityResult.
    """
    if cfg.t_s is None:
        t_s = np.linspace(float(cfg.t_min_s), float(cfg.t_max_s), int(cfg.n), dtype=np.float64)
    else:
        t_s = np.asarray(cfg.t_s, dtype=np.float64)

    params = RamseyParams(
        gamma_phi=float(cfg.gamma_phi),
        phase0=float(cfg.phase0),
        visibility=float(cfg.visibility),
    )

    mu, dmu = ramsey_signal_and_jac_omega(t=t_s, omega=float(cfg.omega), params=params)

    s = float(cfg.sigma_y)
    inv_var = 1.0 / (s * s)
    per = (inv_var * (dmu * dmu)).astype(np.float64, copy=False)

    fisher_total = fisher_scalar_gaussian_iid(dmu_dtheta=dmu, sigma=s)
    crb_std = float(np.sqrt(1.0 / fisher_total)) if fisher_total > 0.0 else float("inf")

    best_index = int(np.argmax(per))
    best_time = float(t_s[best_index])

    return NVRamseySensitivityResult(
        t_s=t_s,
        mu=mu,
        dmu_domega=dmu,
        per_sample_fisher=per,
        fisher_total=float(fisher_total),
        crb_std_omega=float(crb_std),
        best_time_s=best_time,
        best_index=best_index,
    )
