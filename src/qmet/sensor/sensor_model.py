"""
Sensor models.

This module defines compact, reusable sensor dynamics models that output quantum states
(or state-dependent expectations) as a function of time and parameters.

Initial scope:
- A minimal qubit sensor model with closed-form free evolution under a Z Hamiltonian
  and optional pure dephasing, which is enough to support Ramsey-style pipelines.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from ..core.linalg import dagger, to_density_matrix
from ..core.types import ComplexArray

__all__ = [
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "QubitSensorModel",
]

pauli_x: Final[ComplexArray] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
pauli_y: Final[ComplexArray] = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
pauli_z: Final[ComplexArray] = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class QubitSensorModel:
    """
    A minimal qubit sensor suitable for metrology protocols.

    Model:
      H = (1/2) * omega * sigma_z, where omega may include an unknown parameter.
      Pure dephasing in the sigma_z basis: rho_01 -> rho_01 * exp(-gamma_phi * t).

    This is intentionally narrow: it is a clean backbone for Ramsey/echo and
    for linking to measurement + estimation without premature generalization.
    """

    gamma_phi: float = 0.0

    def free_evolution(self, rho0: ComplexArray, t: float, omega: float) -> ComplexArray:
        """
        Evolve a qubit density matrix under Z Hamiltonian + optional pure dephasing.

        Args:
            rho0: (2, 2) density matrix.
            t: evolution time in seconds.
            omega: angular frequency in rad/s.

        Returns:
            rho_t: (2, 2) density matrix at time t.
        """
        rho0 = to_density_matrix(rho0)
        phase = np.exp(-1.0j * 0.5 * float(omega) * float(t))
        u = np.array([[phase, 0.0], [0.0, np.conjugate(phase)]], dtype=np.complex128)

        rho_u = u @ rho0 @ dagger(u)

        g = float(self.gamma_phi)
        if g <= 0.0:
            return to_density_matrix(rho_u)

        decay = float(np.exp(-g * float(t)))
        rho_out = rho_u.copy()
        rho_out[0, 1] *= decay
        rho_out[1, 0] *= decay
        return to_density_matrix(rho_out)
