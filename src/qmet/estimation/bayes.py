"""
Bayesian estimation on a 1D parameter grid.

This is a pragmatic, grounded building block for case studies and validation:
- define a parameter grid theta
- maintain a posterior density over theta
- update with log-likelihood contributions

The representation is discrete:
    p(theta_i) over a grid, normalized after each update.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "GridPosterior1D",
]


@dataclass(frozen=True, slots=True)
class GridPosterior1D:
    """
    Discrete posterior on a fixed 1D grid.

    Attributes:
        theta: grid points (m,).
        logp: log posterior up to an additive constant (m,).
    """

    theta: npt.NDArray[np.float64]
    logp: npt.NDArray[np.float64]

    @classmethod
    def from_prior(
        cls,
        theta: npt.NDArray[np.float64],
        prior_pdf: npt.NDArray[np.float64],
    ) -> "GridPosterior1D":
        """
        Create posterior object from a prior defined on the grid.

        Args:
            theta: grid points (m,).
            prior_pdf: prior density values (m,), nonnegative.

        Returns:
            GridPosterior1D with normalized log posterior.
        """
        th = np.asarray(theta, dtype=np.float64)
        p = np.asarray(prior_pdf, dtype=np.float64)
        if th.ndim != 1 or p.ndim != 1 or th.shape != p.shape:
            raise ValueError("theta and prior_pdf must be 1D arrays of the same shape.")
        if np.any(p < 0.0):
            raise ValueError("prior_pdf must be nonnegative.")
        s = float(np.sum(p))
        if s <= 0.0:
            raise ValueError("prior_pdf must have positive total mass.")
        p_n = p / s
        logp = np.log(p_n)
        return cls(theta=th, logp=logp.astype(np.float64, copy=False))

    def normalized_pdf(self) -> npt.NDArray[np.float64]:
        """Return normalized posterior density on the grid."""
        a = self.logp - float(np.max(self.logp))
        w = np.exp(a)
        w /= float(np.sum(w))
        return w.astype(np.float64, copy=False)

    def update_with_loglik(self, loglik: npt.NDArray[np.float64]) -> "GridPosterior1D":
        """
        Update posterior using an array of log-likelihood values over theta.

        Args:
            loglik: log p(data | theta_i) values (m,).

        Returns:
            new GridPosterior1D instance with updated logp.
        """
        ll = np.asarray(loglik, dtype=np.float64)
        if ll.ndim != 1 or ll.shape != self.theta.shape:
            raise ValueError("loglik must be 1D with the same shape as theta.")
        return GridPosterior1D(theta=self.theta, logp=(self.logp + ll).astype(np.float64, copy=False))

    def mean(self) -> float:
        """Posterior mean on the grid."""
        w = self.normalized_pdf()
        return float(np.sum(w * self.theta))

    def var(self) -> float:
        """Posterior variance on the grid."""
        w = self.normalized_pdf()
        m = float(np.sum(w * self.theta))
        return float(np.sum(w * (self.theta - m) ** 2))

    def map_estimate(self) -> float:
        """Maximum a posteriori (grid argmax)."""
        i = int(np.argmax(self.logp))
        return float(self.theta[i])
