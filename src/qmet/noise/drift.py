"""
Drift models.

This module provides small, composable drift generators for "slow parameter wander"
used in stability and tracking case studies.

Initial scope:
- Random walk drift: x_{k+1} = x_k + sigma * sqrt(dt) * N(0,1)
- Linear drift: x(t) = x0 + slope * t
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = [
    "RandomWalkDrift",
    "LinearDrift",
]


@dataclass(frozen=True, slots=True)
class RandomWalkDrift:
    """Random walk drift with diffusion amplitude sigma."""
    sigma: float

    def __post_init__(self) -> None:
        if float(self.sigma) < 0.0:
            raise ValueError("sigma must be >= 0.")

    def simulate(
        self,
        n: int,
        dt: float,
        rng: np.random.Generator,
        x0: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """
        Simulate a random walk drift series.

        Args:
            n: number of samples (> 0).
            dt: time step in seconds (> 0).
            rng: NumPy RNG.
            x0: initial value.

        Returns:
            x: drift series (n,).
        """
        n_i = int(n)
        if n_i <= 0:
            raise ValueError("n must be > 0.")
        dt_f = float(dt)
        if dt_f <= 0.0:
            raise ValueError("dt must be > 0.")

        step_std = float(self.sigma) * float(np.sqrt(dt_f))
        steps = rng.normal(loc=0.0, scale=step_std, size=n_i - 1).astype(np.float64, copy=False)

        x = np.empty(n_i, dtype=np.float64)
        x[0] = float(x0)
        if n_i > 1:
            x[1:] = float(x0) + np.cumsum(steps, dtype=np.float64)
        return x


@dataclass(frozen=True, slots=True)
class LinearDrift:
    """Linear drift x(t) = x0 + slope * t."""
    slope: float

    def simulate(self, t: npt.NDArray[np.float64], x0: float = 0.0) -> npt.NDArray[np.float64]:
        """
        Evaluate linear drift at time samples.

        Args:
            t: time samples in seconds (n,).
            x0: initial value.

        Returns:
            x: drift series (n,).
        """
        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.ndim != 1:
            raise ValueError("t must be 1D.")
        return (float(x0) + float(self.slope) * t_arr).astype(np.float64, copy=False)
