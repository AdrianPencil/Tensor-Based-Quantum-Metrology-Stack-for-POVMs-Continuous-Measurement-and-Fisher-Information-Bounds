"""
Continuous measurement readout model (diffusive channel).

This module generates or scores measurement records for a continuous monitor.

Core abstraction:
- a Hermitian observable O
- a measurement strength kappa and efficiency eta
- mean signal proportional to <O>_t

We keep this file small by focusing on the measurement record model; the state
update (backaction) is handled separately in measurement/backaction.py.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..core.linalg import trace
from ..core.types import ComplexArray

__all__ = [
    "ContinuousMeasurement",
]


@dataclass(frozen=True, slots=True)
class ContinuousMeasurement:
    """
    Diffusive continuous measurement of a Hermitian observable O.

    Measurement record model (increment form):
        dY = m(t) dt + dW

    with mean:
        m(t) = 2 * sqrt(eta * kappa) * <O>_t

    This is a standard normalization choice; it pairs naturally with the SME step
    implemented in measurement/backaction.py.
    """

    observable: ComplexArray
    kappa: float
    eta: float = 1.0

    def __post_init__(self) -> None:
        o = np.asarray(self.observable, dtype=np.complex128)
        if o.ndim != 2 or o.shape[0] != o.shape[1]:
            raise ValueError("observable must be a square (d, d) array.")
        if float(self.kappa) <= 0.0:
            raise ValueError("kappa must be > 0.")
        if not (0.0 <= float(self.eta) <= 1.0):
            raise ValueError("eta must be in [0, 1].")
        object.__setattr__(self, "observable", o)

    def mean_signal(self, rho: ComplexArray) -> float:
        """
        Compute m = 2 * sqrt(eta * kappa) * <O>.

        Args:
            rho: (d, d) density matrix.

        Returns:
            mean signal (float).
        """
        rho_m = np.asarray(rho, dtype=np.complex128)
        exp_o = trace(self.observable @ rho_m)
        return float(2.0 * np.sqrt(float(self.eta) * float(self.kappa)) * np.real(exp_o))

    def simulate_increments(
        self,
        exp_series: npt.NDArray[np.float64],
        dt: float,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """
        Simulate dY increments given a time series of <O> expectations.

        Args:
            exp_series: <O>_t as float64 array (n,).
            dt: time step in seconds.
            rng: NumPy RNG.

        Returns:
            dY: increments array (n,).
        """
        e = np.asarray(exp_series, dtype=np.float64)
        if e.ndim != 1:
            raise ValueError("exp_series must be 1D.")

        dt_f = float(dt)
        if dt_f <= 0.0:
            raise ValueError("dt must be > 0.")

        scale = float(2.0 * np.sqrt(float(self.eta) * float(self.kappa)))
        mean = scale * e * dt_f
        noise = rng.normal(loc=0.0, scale=np.sqrt(dt_f), size=e.shape).astype(np.float64, copy=False)
        return (mean + noise).astype(np.float64, copy=False)
