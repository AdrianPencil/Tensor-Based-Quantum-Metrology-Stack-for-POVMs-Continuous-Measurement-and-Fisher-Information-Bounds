"""
Case study: readout chain example (sensor -> device -> noise -> likelihood).

This shows a complete "measurement record" story without extra complexity:
- Start with a clean sensor-level mean signal x(t) in [-1, 1] (here: Ramsey mu(t))
- Map it through an affine device response y_mean(t) = gain * x(t) + offset
- Add Gaussian readout noise to generate y(t)
- Score the record using a Gaussian log-likelihood

This is designed to be used in docs and quick validation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..protocols.ramsey import RamseyParams, ramsey_signal
from ..sensor.device_response import AffineDeviceResponse
from ..measurement.readout_noise import GaussianReadoutNoise

__all__ = [
    "ReadoutChainConfig",
    "ReadoutChainResult",
    "run_readout_chain_example",
]


@dataclass(frozen=True, slots=True)
class ReadoutChainConfig:
    """
    Configuration for the readout chain example.

    omega: Ramsey angular frequency (rad/s)
    gamma_phi: Ramsey pure dephasing rate (1/s)
    visibility: Ramsey contrast in [0, 1]
    phase0: Ramsey phase offset (rad)

    device_gain/device_offset: affine readout map
    sigma_y: additive Gaussian readout std (> 0)

    t_s: if provided use directly; else linspace(t_min_s, t_max_s, n)
    seed: RNG seed (None for non-deterministic)
    """

    omega: float
    gamma_phi: float
    visibility: float
    sigma_y: float
    phase0: float = 0.0
    device_gain: float = 1.0
    device_offset: float = 0.0
    t_s: Optional[npt.NDArray[np.float64]] = None
    t_min_s: float = 0.0
    t_max_s: float = 1e-3
    n: int = 200
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if float(self.gamma_phi) < 0.0:
            raise ValueError("gamma_phi must be >= 0.")
        if not (0.0 <= float(self.visibility) <= 1.0):
            raise ValueError("visibility must be in [0, 1].")
        if float(self.sigma_y) <= 0.0:
            raise ValueError("sigma_y must be > 0.")
        if self.t_s is None:
            if float(self.t_max_s) <= float(self.t_min_s):
                raise ValueError("t_max_s must be > t_min_s.")
            if int(self.n) < 2:
                raise ValueError("n must be >= 2.")
        else:
            tt = np.asarray(self.t_s, dtype=np.float64)
            if tt.ndim != 1 or tt.size < 2:
                raise ValueError("t_s must be a 1D array with length >= 2.")


@dataclass(frozen=True, slots=True)
class ReadoutChainResult:
    """
    Output of the readout chain simulation + scoring.
    """

    t_s: npt.NDArray[np.float64]
    x_mu: npt.NDArray[np.float64]
    y_mu: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    loglik: float


def run_readout_chain_example(cfg: ReadoutChainConfig) -> ReadoutChainResult:
    """
    Run the readout chain example.

    Args:
        cfg: ReadoutChainConfig.

    Returns:
        ReadoutChainResult including a simulated record and its log-likelihood.
    """
    if cfg.t_s is None:
        t_s = np.linspace(float(cfg.t_min_s), float(cfg.t_max_s), int(cfg.n), dtype=np.float64)
    else:
        t_s = np.asarray(cfg.t_s, dtype=np.float64)

    params = RamseyParams(
        gamma_phi=float(cfg.gamma_phi),
        phase0=float(cfg.phase0),
        visibility=float(cfg.visibility),
    )

    x_mu = ramsey_signal(t=t_s, omega=float(cfg.omega), params=params)

    device = AffineDeviceResponse(gain=float(cfg.device_gain), offset=float(cfg.device_offset))
    y_mu = device.apply(x_mu)

    noise = GaussianReadoutNoise(sigma=float(cfg.sigma_y))
    rng = np.random.default_rng(int(cfg.seed)) if cfg.seed is not None else np.random.default_rng()
    y = (y_mu + rng.normal(loc=0.0, scale=float(cfg.sigma_y), size=y_mu.shape)).astype(np.float64, copy=False)

    ll = noise.loglik(y=y, mu=y_mu)
    return ReadoutChainResult(t_s=t_s, x_mu=x_mu, y_mu=y_mu, y=y, loglik=float(ll))
