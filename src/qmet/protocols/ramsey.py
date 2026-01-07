"""
Ramsey protocol primitives.

This file provides a minimal Ramsey signal model for a qubit evolving under
H = (omega/2) sigma_z with optional pure dephasing in the sigma_z basis.

Output is intentionally "estimation-ready":
- a mean readout signal mu(t) in [-1, 1] (typically an expectation)
- and its derivative w.r.t. omega for Fisher/CRB calculations
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "RamseyParams",
    "ramsey_signal",
    "ramsey_signal_and_jac_omega",
]


@dataclass(frozen=True, slots=True)
class RamseyParams:
    """
    Ramsey parameters for a simple closed-form model.

    gamma_phi: pure dephasing rate (1/s)
    phase0: additional phase offset (rad), e.g. due to control phase
    visibility: multiplicative contrast factor in [0, 1]
    """

    gamma_phi: float = 0.0
    phase0: float = 0.0
    visibility: float = 1.0

    def __post_init__(self) -> None:
        if float(self.gamma_phi) < 0.0:
            raise ValueError("gamma_phi must be >= 0.")
        if not (0.0 <= float(self.visibility) <= 1.0):
            raise ValueError("visibility must be in [0, 1].")


def ramsey_signal(
    t: npt.NDArray[np.float64],
    omega: float,
    params: RamseyParams,
) -> npt.NDArray[np.float64]:
    """
    Mean Ramsey signal model (dimensionless):

        mu(t) = V * exp(-gamma_phi t) * cos(omega t + phase0)

    Args:
        t: times in seconds (n,).
        omega: angular frequency in rad/s.
        params: RamseyParams.

    Returns:
        mu: mean signal (n,) in float64.
    """
    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError("t must be 1D.")
    w = float(omega)
    g = float(params.gamma_phi)
    v = float(params.visibility)
    p0 = float(params.phase0)

    env = np.exp(-g * tt) if g > 0.0 else np.ones_like(tt)
    mu = v * env * np.cos(w * tt + p0)
    return mu.astype(np.float64, copy=False)


def ramsey_signal_and_jac_omega(
    t: npt.NDArray[np.float64],
    omega: float,
    params: RamseyParams,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Ramsey mean signal and Jacobian for the single parameter omega.

    d/domega mu(t) = -V * exp(-gamma_phi t) * t * sin(omega t + phase0)

    Args:
        t: times in seconds (n,).
        omega: angular frequency in rad/s.
        params: RamseyParams.

    Returns:
        mu: (n,)
        dmu_domega: (n,)
    """
    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError("t must be 1D.")
    w = float(omega)
    g = float(params.gamma_phi)
    v = float(params.visibility)
    p0 = float(params.phase0)

    env = np.exp(-g * tt) if g > 0.0 else np.ones_like(tt)
    arg = w * tt + p0
    mu = v * env * np.cos(arg)
    dmu = -v * env * tt * np.sin(arg)
    return mu.astype(np.float64, copy=False), dmu.astype(np.float64, copy=False)
