"""
Squeezing-related metrics (protocol-level).

This module keeps squeezing in a "metrology-facing" form rather than a full
many-body simulator. The main quantity is the Wineland spin-squeezing parameter:

    xi^2 = (N * Var(J_perp)) / <J>^2

where:
- N is the number of spins
- <J> is the mean collective spin length along the mean-spin direction
- Var(J_perp) is the variance in an orthogonal (best) quadrature

This provides a clean link to sensitivity improvements in case studies.
"""

from dataclasses import dataclass

__all__ = [
    "WinelandSqueezing",
]


@dataclass(frozen=True, slots=True)
class WinelandSqueezing:
    """
    A compact container for Wineland squeezing.

    Attributes:
        n_spins: number of spins N (>= 1)
        j_mean: mean collective spin length <J> (>= 0)
        var_perp: variance in the optimal perpendicular quadrature (>= 0)
    """

    n_spins: int
    j_mean: float
    var_perp: float

    def __post_init__(self) -> None:
        if int(self.n_spins) < 1:
            raise ValueError("n_spins must be >= 1.")
        if float(self.j_mean) < 0.0:
            raise ValueError("j_mean must be >= 0.")
        if float(self.var_perp) < 0.0:
            raise ValueError("var_perp must be >= 0.")

    def xi_squared(self) -> float:
        """
        Compute xi^2 = (N * Var(J_perp)) / <J>^2.

        Returns:
            xi^2 as float (lower is better, < 1 indicates squeezing).
        """
        jm = float(self.j_mean)
        if jm == 0.0:
            raise ValueError("Cannot compute xi^2 with j_mean = 0.")
        return float(int(self.n_spins) * float(self.var_perp) / (jm * jm))

    def xi_db(self) -> float:
        """
        Convert xi^2 to dB of squeezing:
            10 log10(xi^2)

        Returns:
            squeezing in dB (negative means improved sensitivity).
        """
        import numpy as np

        xi2 = self.xi_squared()
        return float(10.0 * np.log10(xi2))
