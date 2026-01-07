"""
Spin-echo protocol primitives.

This module provides a minimal echo envelope model suitable for linking to
noise + estimation stacks. It intentionally does not try to encode every
microscopic detail; the goal is a stable, estimation-ready API.

Model (one common phenomenology):
    mu(t) = V * exp(-(t / T2)^p) * cos(omega t + phase0)

For "echo", the effective dephasing is reduced; you encode that by using a larger
T2 (or different p) compared to Ramsey.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "EchoParams",
    "echo_signal",
    "echo_signal_and_jac_omega",
]


@dataclass(frozen=True, slots=True)
class EchoParams:
    """
    Echo envelope parameters.

    t2: coherence time scale (s), must be > 0
    p: stretch exponent, typically in [1, 3]
    phase0: phase offset (rad)
    visibility: contrast factor in [0, 1]
    """

    t2: float
    p: float = 1.0
    phase0: float = 0.0
    visibility: float = 1.0

    def __post_init__(self) -> None:
        if float(self.t2) <= 0.0:
            raise ValueError("t2 must be > 0.")
        if float(self.p) <= 0.0:
            raise ValueError("p must be > 0.")
        if not (0.0 <= float(self.visibility) <= 1.0):
            raise ValueError("visibility must be in [0, 1].")


def echo_signal(
    t: npt.NDArray[np.float64],
    omega: float,
    params: EchoParams,
) -> npt.NDArray[np.float64]:
    """
    Mean echo signal:
        mu(t) = V * exp(-(t/T2)^p) * cos(omega t + phase0)

    Args:
        t: times in seconds (n,).
        omega: angular frequency in rad/s.
        params: EchoParams.

    Returns:
        mu: (n,) float64.
    """
    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError("t must be 1D.")

    w = float(omega)
    v = float(params.visibility)
    t2 = float(params.t2)
    p = float(params.p)
    p0 = float(params.phase0)

    env = np.exp(-np.power(tt / t2, p))
    mu = v * env * np.cos(w * tt + p0)
    return mu.astype(np.float64, copy=False)


def echo_signal_and_jac_omega(
    t: npt.NDArray[np.float64],
    omega: float,
    params: EchoParams,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Echo mean signal and Jacobian for omega.

    d/domega mu(t) = -V * exp(-(t/T2)^p) * t * sin(omega t + phase0)

    Args:
        t: times in seconds (n,).
        omega: angular frequency in rad/s.
        params: EchoParams.

    Returns:
        mu: (n,)
        dmu_domega: (n,)
    """
    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1:
        raise ValueError("t must be 1D.")

    w = float(omega)
    v = float(params.visibility)
    t2 = float(params.t2)
    p = float(params.p)
    p0 = float(params.phase0)

    env = np.exp(-np.power(tt / t2, p))
    arg = w * tt + p0
    mu = v * env * np.cos(arg)
    dmu = -v * env * tt * np.sin(arg)
    return mu.astype(np.float64, copy=False), dmu.astype(np.float64, copy=False)
