"""
Readout noise models.

These models sit on top of an "ideal" mean signal mu(t) and define a likelihood
for the measured readout y(t).

Initial scope:
- Gaussian additive noise, vectorized log-likelihood
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "GaussianReadoutNoise",
]


@dataclass(frozen=True, slots=True)
class GaussianReadoutNoise:
    """
    Additive Gaussian readout noise:

        y = mu + eps,   eps ~ N(0, sigma^2)

    Provides a fast log-likelihood for arrays.
    """

    sigma: float

    def __post_init__(self) -> None:
        if float(self.sigma) <= 0.0:
            raise ValueError("sigma must be > 0.")

    def logpdf(
        self,
        y: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Vectorized log p(y | mu).

        Args:
            y: measured values.
            mu: mean values (same shape).

        Returns:
            logp: array of log densities (same shape).
        """
        y_arr = np.asarray(y, dtype=np.float64)
        mu_arr = np.asarray(mu, dtype=np.float64)
        if y_arr.shape != mu_arr.shape:
            raise ValueError("y and mu must have the same shape.")

        s = float(self.sigma)
        z = (y_arr - mu_arr) / s
        return (-0.5 * z * z - np.log(s) - 0.5 * np.log(2.0 * np.pi)).astype(np.float64, copy=False)

    def loglik(
        self,
        y: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
    ) -> float:
        """
        Total log-likelihood: sum over samples.

        Args:
            y: measured values.
            mu: mean values.

        Returns:
            total log-likelihood as float.
        """
        return float(np.sum(self.logpdf(y=y, mu=mu)))
