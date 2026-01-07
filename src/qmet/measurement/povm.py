"""
POVM measurement models (discrete outcomes).

A POVM is a set of positive semidefinite operators {E_k} such that:
    sum_k E_k = I

Given a density matrix rho:
    p_k = Tr(E_k rho)

This file keeps the abstraction small:
- validate the POVM
- compute probabilities
- sample outcomes
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.linalg import trace
from ..core.types import ComplexArray

__all__ = [
    "POVM",
]


@dataclass(frozen=True, slots=True)
class POVM:
    """
    Discrete POVM for a fixed Hilbert-space dimension.

    Elements are stored as an array of shape (K, d, d).
    """

    elements: ComplexArray

    def __post_init__(self) -> None:
        e = np.asarray(self.elements, dtype=np.complex128)
        if e.ndim != 3:
            raise ValueError("POVM.elements must have shape (K, d, d).")
        if e.shape[1] != e.shape[2]:
            raise ValueError("POVM elements must be square matrices.")
        object.__setattr__(self, "elements", e)

    @property
    def num_outcomes(self) -> int:
        return int(self.elements.shape[0])

    @property
    def dim(self) -> int:
        return int(self.elements.shape[1])

    def validate(self, *, atol: float = 1e-10) -> None:
        """
        Validate completeness and basic numeric sanity.

        This does not attempt a full PSD check (which is more expensive),
        but it does enforce:
        - sum_k E_k ≈ I
        - finite values
        """
        if not np.isfinite(self.elements).all():
            raise ValueError("POVM contains non-finite values.")

        s = np.sum(self.elements, axis=0)
        i = np.eye(self.dim, dtype=np.complex128)
        if not np.allclose(s, i, atol=atol, rtol=0.0):
            raise ValueError("POVM completeness failed: sum(E_k) != I (within tolerance).")

    def probabilities(self, rho: ComplexArray) -> np.ndarray:
        """
        Compute p_k = Tr(E_k rho) for all outcomes, returned as float64.

        Args:
            rho: (d, d) density matrix.

        Returns:
            p: (K,) probabilities, clipped into [0, 1] and renormalized.
        """
        rho_m = np.asarray(rho, dtype=np.complex128)
        if rho_m.shape != (self.dim, self.dim):
            raise ValueError("rho has wrong shape for this POVM.")

        tr_vals = np.einsum("kij,ji->k", self.elements, rho_m, optimize=True)
        p = np.real(tr_vals).astype(np.float64, copy=False)
        p = np.clip(p, 0.0, 1.0)
        s = float(np.sum(p))
        if s <= 0.0:
            raise ValueError("All POVM probabilities are zero after clipping.")
        return p / s

    def sample(self, rho: ComplexArray, rng: np.random.Generator) -> int:
        """
        Sample an outcome index according to p_k.

        Args:
            rho: (d, d) density matrix.
            rng: NumPy RNG.

        Returns:
            outcome index in [0, K).
        """
        p = self.probabilities(rho)
        return int(rng.choice(self.num_outcomes, p=p))
