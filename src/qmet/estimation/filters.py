"""
Simple filters for online estimation.

Initial scope:
- A minimal 1D Kalman filter for a random-walk latent parameter.

Model:
    x_{k+1} = x_k + w_k,      w_k ~ N(0, q)
    y_k     = x_k + v_k,      v_k ~ N(0, r)
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "ScalarKalmanFilter",
]


@dataclass(frozen=True, slots=True)
class ScalarKalmanFilter:
    """
    1D Kalman filter with random-walk dynamics.

    State:
        mean: E[x]
        var: Var[x]
    """

    mean: float
    var: float
    q: float
    r: float

    def __post_init__(self) -> None:
        if float(self.var) < 0.0:
            raise ValueError("var must be >= 0.")
        if float(self.q) < 0.0:
            raise ValueError("q must be >= 0.")
        if float(self.r) <= 0.0:
            raise ValueError("r must be > 0.")

    def predict(self) -> "ScalarKalmanFilter":
        """
        Prediction step for random walk:
            mean <- mean
            var  <- var + q
        """
        return ScalarKalmanFilter(
            mean=float(self.mean),
            var=float(self.var) + float(self.q),
            q=float(self.q),
            r=float(self.r),
        )

    def update(self, y: float) -> "ScalarKalmanFilter":
        """
        Update step with measurement y:
            K = var / (var + r)
            mean <- mean + K (y - mean)
            var <- (1 - K) var
        """
        v = float(self.var)
        k = v / (v + float(self.r)) if v > 0.0 else 0.0
        m = float(self.mean) + k * (float(y) - float(self.mean))
        v_new = (1.0 - k) * v
        return ScalarKalmanFilter(mean=m, var=v_new, q=float(self.q), r=float(self.r))

    def step(self, y: float) -> "ScalarKalmanFilter":
        """Convenience: predict then update."""
        return self.predict().update(y)

    def batch(self, y: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Run the filter over a sequence of observations.

        Args:
            y: observations (n,).

        Returns:
            means: (n,) filtered means.
            vars: (n,) filtered variances.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1:
            raise ValueError("y must be 1D.")
        means = np.empty_like(y_arr, dtype=np.float64)
        vars_ = np.empty_like(y_arr, dtype=np.float64)
        kf: ScalarKalmanFilter = self
        for i in range(y_arr.size):
            kf = kf.step(float(y_arr[i]))
            means[i] = float(kf.mean)
            vars_[i] = float(kf.var)
        return means, vars_
