"""
Likelihood models.

This module defines small, reusable likelihood blocks used by estimators.
Initial focus: independent Gaussian readout with known standard deviation.

Conventions:
- y: observed data, shape (n,)
- mu: model mean, shape (n,)
- jac: Jacobian d mu / d theta, shape (n, p)
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "GaussianIIDLikelihood",
]


@dataclass(frozen=True, slots=True)
class GaussianIIDLikelihood:
    """
    Independent Gaussian likelihood with fixed sigma:

        y_i ~ Normal(mu_i, sigma^2)

    Provides log-likelihood and (when given a Jacobian) score and observed Fisher.
    """

    sigma: float

    def __post_init__(self) -> None:
        if float(self.sigma) <= 0.0:
            raise ValueError("sigma must be > 0.")

    def loglik(self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]) -> float:
        """
        Compute total log-likelihood log p(y | mu, sigma).

        Args:
            y: observations (n,).
            mu: predicted mean (n,).

        Returns:
            total log-likelihood.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        mu_arr = np.asarray(mu, dtype=np.float64)
        if y_arr.shape != mu_arr.shape:
            raise ValueError("y and mu must have the same shape.")

        s = float(self.sigma)
        z = (y_arr - mu_arr) / s
        ll = -0.5 * np.sum(z * z) - y_arr.size * (np.log(s) + 0.5 * np.log(2.0 * np.pi))
        return float(ll)

    def score(
        self,
        y: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        jac: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Score vector for parameters theta given mu(theta) and Jacobian J = dmu/dtheta:

            s(theta) = d/dtheta log p(y|theta)
                     = (1/sigma^2) * J^T (y - mu)

        Args:
            y: observations (n,).
            mu: predicted mean (n,).
            jac: Jacobian (n, p).

        Returns:
            score: (p,) float64.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        mu_arr = np.asarray(mu, dtype=np.float64)
        j = np.asarray(jac, dtype=np.float64)

        if y_arr.ndim != 1 or mu_arr.ndim != 1:
            raise ValueError("y and mu must be 1D.")
        if y_arr.shape != mu_arr.shape:
            raise ValueError("y and mu must have the same length.")
        if j.ndim != 2 or j.shape[0] != y_arr.shape[0]:
            raise ValueError("jac must have shape (n, p) matching y length.")

        r = y_arr - mu_arr
        inv_var = 1.0 / (float(self.sigma) ** 2)
        s = inv_var * (j.T @ r)
        return s.astype(np.float64, copy=False)

    def observed_fisher(
        self,
        jac: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Observed Fisher information for Gaussian IID (also equals expected Fisher):

            I = (1/sigma^2) * J^T J

        Args:
            jac: Jacobian (n, p).

        Returns:
            I: Fisher information matrix (p, p).
        """
        j = np.asarray(jac, dtype=np.float64)
        if j.ndim != 2:
            raise ValueError("jac must be 2D.")
        inv_var = 1.0 / (float(self.sigma) ** 2)
        fim = inv_var * (j.T @ j)
        return fim.astype(np.float64, copy=False)
