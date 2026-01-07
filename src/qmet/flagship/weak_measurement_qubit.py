"""
Flagship: continuous weak measurement of a qubit.

This flagship demonstrates the continuous measurement stack:
- measurement record increments dY
- stochastic backaction update for the density matrix

It is a reference implementation for:
- measurement/continuous.py (record model)
- measurement/backaction.py (SME step)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..core.types import ComplexArray
from ..core.linalg import to_density_matrix
from ..measurement.backaction import step_diffusive_hermitian
from ..measurement.continuous import ContinuousMeasurement
from ..sensor.sensor_model import pauli_z

__all__ = [
    "WeakMeasurementConfig",
    "WeakMeasurementResult",
    "run_flagship_weak_measurement_qubit",
]


@dataclass(frozen=True, slots=True)
class WeakMeasurementConfig:
    """
    Configuration for the weak measurement flagship.

    n: number of increments
    dt: time step (s)
    kappa: measurement strength (> 0)
    eta: efficiency in [0, 1]
    seed: RNG seed (None disables deterministic randomness)
    """

    n: int = 2000
    dt: float = 1e-6
    kappa: float = 5e5
    eta: float = 1.0
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if int(self.n) < 1:
            raise ValueError("n must be >= 1.")
        if float(self.dt) <= 0.0:
            raise ValueError("dt must be > 0.")
        if float(self.kappa) <= 0.0:
            raise ValueError("kappa must be > 0.")
        if not (0.0 <= float(self.eta) <= 1.0):
            raise ValueError("eta must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class WeakMeasurementResult:
    """
    Output of a weak measurement trajectory simulation.
    """

    t_s: npt.NDArray[np.float64]
    d_y: npt.NDArray[np.float64]
    rho_traj: ComplexArray


def run_flagship_weak_measurement_qubit(
    cfg: WeakMeasurementConfig,
    rho0: Optional[ComplexArray] = None,
    observable: Optional[ComplexArray] = None,
) -> WeakMeasurementResult:
    """
    Simulate a diffusive continuous measurement trajectory for a qubit.

    Args:
        cfg: WeakMeasurementConfig.
        rho0: optional initial density matrix (2, 2). Default is |+><+|.
        observable: optional Hermitian observable (2, 2). Default is sigma_z.

    Returns:
        WeakMeasurementResult with:
          - t_s shape (n,)
          - d_y shape (n,)
          - rho_traj shape (n+1, 2, 2)
    """
    obs = pauli_z if observable is None else np.asarray(observable, dtype=np.complex128)

    if rho0 is None:
        psi = np.array([1.0, 1.0], dtype=np.complex128)
        psi /= np.linalg.norm(psi)
        rho_init = np.outer(psi, np.conjugate(psi)).astype(np.complex128, copy=False)
    else:
        rho_init = np.asarray(rho0, dtype=np.complex128)

    rho_init = to_density_matrix(rho_init)

    rng = np.random.default_rng(int(cfg.seed)) if cfg.seed is not None else np.random.default_rng()

    n = int(cfg.n)
    dt = float(cfg.dt)
    t_s = (np.arange(n, dtype=np.float64) * dt).astype(np.float64, copy=False)

    meas = ContinuousMeasurement(observable=obs, kappa=float(cfg.kappa), eta=float(cfg.eta))

    rho_traj = np.empty((n + 1, 2, 2), dtype=np.complex128)
    d_y = np.empty(n, dtype=np.float64)
    rho_traj[0] = rho_init

    for k in range(n):
        rho_k = rho_traj[k]
        m = meas.mean_signal(rho_k)
        d_w = float(rng.normal(loc=0.0, scale=np.sqrt(dt)))
        d_y[k] = float(m * dt + d_w)
        rho_traj[k + 1] = step_diffusive_hermitian(
            rho=rho_k,
            observable=obs,
            kappa=float(cfg.kappa),
            eta=float(cfg.eta),
            dt=dt,
            d_w=d_w,
        )

    return WeakMeasurementResult(t_s=t_s, d_y=d_y, rho_traj=rho_traj)
