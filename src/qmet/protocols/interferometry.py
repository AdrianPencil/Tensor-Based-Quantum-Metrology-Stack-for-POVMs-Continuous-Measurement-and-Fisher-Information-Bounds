"""
Interferometry protocol primitives.

This module provides a minimal cosine-fringe model used in many sensing contexts:
- Mach-Zehnder-like interferometry
- phase estimation with a sinusoidal readout

Model:
    mu(phi) = offset + visibility * cos(phi + phase0)

Optionally, a phase may be time-dependent phi(t) = omega * t + phi0.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "FringeParams",
    "fringe",
    "fringe_and_jac_phase",
    "fringe_time_series_and_jac_omega",
]


@dataclass(frozen=True, slots=True)
class FringeParams:
    """
    Cosine fringe parameters.

    visibility: amplitude of the cosine term (>= 0)
    phase0: phase offset added inside cosine (rad)
    offset: additive offset
    """

    visibility: float = 1.0
    phase0: float = 0.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if float(self.visibility) < 0.0:
            raise ValueError("visibility must be >= 0.")


def fringe(phi: npt.NDArray[np.float64], params: FringeParams) -> npt.NDArray[np.float64]:
    """
    Compute a cosine fringe mu(phi).

    Args:
        phi: phase array in radians (n,).
        params: FringeParams.

    Returns:
        mu: (n,) float64.
    """
    ph = np.asarray(phi, dtype=np.float64)
    if ph.ndim != 1:
        raise ValueError("phi must be 1D.")
    v = float(params.visibility)
    p0 = float(params.phase0)
    off = float(params.offset)
    mu = off + v * np.cos(ph + p0)
    return mu.astype(np.float64, copy=False)


def fringe_and_jac_phase(
    phi: npt.NDArray[np.float64],
    params: FringeParams,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Fringe and derivative w.r.t. phase.

    d/dphi mu(phi) = -visibility * sin(phi + phase0)

    Args:
        phi: (n,)
        params: FringeParams

    Returns:
        mu: (n,)
        dmu_dphi: (n,)
    """
    ph = np.asarray(phi, dtype=np.float64)
    if ph.ndim != 1:
        raise ValueError("phi must be 1D.")
    v = float(params.visibility)
    p0 = float(params.phase0)
    off = float(params.offset)
    arg = ph + p0
    mu = off + v * np.cos(arg)
    dmu = -v * np.sin(arg)
    return mu.astype(np.float64, copy=False), dmu.astype(np.float64, copy=False)


def fringe_time_series_and_jac_omega(
    t: npt.NDArray[np.float64],
    omega: float,
    phi0: float,
    params: FringeParams,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Time-series fringe mu(t) with phase phi(t) = omega * t + phi0, and Jacobian for omega.

    d/domega mu(t) = dmu/dphi * dphi/domega = (-v sin(phi+p0)) * t

    Args:
        t: times in seconds (n,).
        omega: angular frequency (rad/s).
        phi0: initial phase (rad).
        params: FringeParams.

    Returns:
        mu: (n,)
        dmu_domega: (n,)
    """
    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError("t must be 1D.")
    w = float(omega)
    ph = w * tt + float(phi0)
    mu, dmu_dphi = fringe_and_jac_phase(ph, params=params)
    dmu_domega = dmu_dphi * tt
    return mu.astype(np.float64, copy=False), dmu_domega.astype(np.float64, copy=False)
