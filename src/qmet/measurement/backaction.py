"""
Measurement backaction (state update rules).

This module provides a single, well-defined SME update step for diffusive
continuous measurement of a Hermitian operator.

Stochastic master equation step (Ito form):
    dρ = κ D[O](ρ) dt + sqrt(η κ) H[O](ρ) dW

where:
    D[O](ρ) = O ρ O - 1/2 {O^2, ρ}
    H[O](ρ) = O ρ + ρ O - 2 Tr(O ρ) ρ

The normalization is consistent with measurement/continuous.py.
"""

import numpy as np

from ..core.linalg import anticomm, lindblad_dissipator, trace, to_density_matrix
from ..core.types import ComplexArray

__all__ = [
    "step_diffusive_hermitian",
]


def step_diffusive_hermitian(
    rho: ComplexArray,
    observable: ComplexArray,
    kappa: float,
    eta: float,
    dt: float,
    d_w: float,
) -> ComplexArray:
    """
    One stochastic update step for diffusive measurement of a Hermitian observable.

    Args:
        rho: (d, d) density matrix at current time.
        observable: (d, d) Hermitian operator O.
        kappa: measurement strength (> 0).
        eta: efficiency in [0, 1].
        dt: time step (> 0).
        d_w: Wiener increment ~ N(0, dt).

    Returns:
        rho_next: (d, d) density matrix after the SME step.
    """
    rho_m = to_density_matrix(np.asarray(rho, dtype=np.complex128))
    o = np.asarray(observable, dtype=np.complex128)

    if o.ndim != 2 or o.shape[0] != o.shape[1]:
        raise ValueError("observable must be a square (d, d) array.")
    if float(kappa) <= 0.0:
        raise ValueError("kappa must be > 0.")
    if not (0.0 <= float(eta) <= 1.0):
        raise ValueError("eta must be in [0, 1].")
    if float(dt) <= 0.0:
        raise ValueError("dt must be > 0.")

    k = float(kappa)
    e = float(eta)
    dt_f = float(dt)
    d_w_f = float(d_w)

    d_det = k * lindblad_dissipator(rho_m, o) * dt_f

    exp_o = trace(o @ rho_m)
    exp_o_r = float(np.real(exp_o))
    h = o @ rho_m + rho_m @ o - 2.0 * exp_o_r * rho_m
    d_stoch = np.sqrt(e * k) * h * d_w_f

    rho_next = rho_m + d_det + d_stoch
    return to_density_matrix(rho_next)
