"""
Linear algebra primitives for quantum models.

This file provides the minimal "quantum LA toolkit" used everywhere:
- dagger, commutator, anticommutator
- density-matrix normalization helpers
- Lindblad dissipator (building block for open-system models)

All functions are pure and vectorized, using NumPy/SciPy kernels.
"""

from typing import Final

import numpy as np
import numpy.typing as npt

from .types import ComplexArray, RealArray

__all__ = [
    "dagger",
    "comm",
    "anticomm",
    "is_hermitian",
    "trace",
    "to_density_matrix",
    "lindblad_dissipator",
]

_ATOL_DEFAULT: Final[float] = 1e-12
_RTOL_DEFAULT: Final[float] = 1e-12


def dagger(a: ComplexArray) -> ComplexArray:
    """Conjugate transpose."""
    return np.conjugate(a).T


def trace(a: ComplexArray) -> np.complex128:
    """Matrix trace."""
    return np.trace(a)


def comm(a: ComplexArray, b: ComplexArray) -> ComplexArray:
    """Commutator [A, B] = AB - BA."""
    return a @ b - b @ a


def anticomm(a: ComplexArray, b: ComplexArray) -> ComplexArray:
    """Anticommutator {A, B} = AB + BA."""
    return a @ b + b @ a


def is_hermitian(a: ComplexArray, *, atol: float = _ATOL_DEFAULT, rtol: float = _RTOL_DEFAULT) -> bool:
    """Check Hermiticity with numeric tolerance."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return False
    return bool(np.allclose(a, dagger(a), atol=atol, rtol=rtol))


def to_density_matrix(rho: ComplexArray, *, atol: float = _ATOL_DEFAULT) -> ComplexArray:
    """
    Project a nearly-valid density matrix onto a clean, consistent form.

    Steps:
    - symmetrize: rho <- (rho + rho†)/2
    - trace normalize: rho <- rho / Tr(rho)
    - clip tiny negative eigenvalues caused by roundoff
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square 2D array.")

    rho_h = 0.5 * (rho + dagger(rho))
    tr = trace(rho_h)
    if abs(tr) < atol:
        raise ValueError("rho has near-zero trace; cannot normalize.")

    rho_n = rho_h / tr
    w, v = np.linalg.eigh(rho_n)
    w_clip = np.maximum(w, 0.0)
    s = float(np.sum(w_clip))
    if s < atol:
        raise ValueError("rho is numerically non-positive with near-zero total weight.")

    w_clip /= s
    rho_psd = (v * w_clip) @ dagger(v)
    return 0.5 * (rho_psd + dagger(rho_psd))


def lindblad_dissipator(rho: ComplexArray, l_op: ComplexArray) -> ComplexArray:
    """
    Lindblad dissipator:
        D[L](rho) = L rho L† - 1/2 {L† L, rho}
    """
    ldag = dagger(l_op)
    jump = l_op @ rho @ ldag
    k = ldag @ l_op
    return jump - 0.5 * anticomm(k, rho)
