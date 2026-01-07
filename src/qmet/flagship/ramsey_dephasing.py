"""
Flagship: Ramsey under dephasing + estimation link.

This flagship is a minimal end-to-end story:
- choose a time grid
- define a Ramsey model with dephasing
- compute mu(t), dmu/domega
- compute Fisher/CRB for omega given Gaussian readout noise
- optionally simulate a noisy record y(t)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..protocols.ramsey import RamseyParams
from ..workflows.pipelines import RamseyFrequencyPipelineResult, run_ramsey_frequency_pipeline

__all__ = [
    "FlagshipRamseyDephasingConfig",
    "run_flagship_ramsey_dephasing",
]


@dataclass(frozen=True, slots=True)
class FlagshipRamseyDephasingConfig:
    """
    Configuration for the Ramsey dephasing flagship.
    """

    omega: float
    gamma_phi: float
    sigma_y: float
    t_min_s: float = 0.0
    t_max_s: float = 1e-3
    n: int = 200
    phase0: float = 0.0
    visibility: float = 1.0
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if float(self.t_max_s) <= float(self.t_min_s):
            raise ValueError("t_max_s must be > t_min_s.")
        if int(self.n) < 2:
            raise ValueError("n must be >= 2.")
        if float(self.sigma_y) <= 0.0:
            raise ValueError("sigma_y must be > 0.")
        if float(self.gamma_phi) < 0.0:
            raise ValueError("gamma_phi must be >= 0.")
        if not (0.0 <= float(self.visibility) <= 1.0):
            raise ValueError("visibility must be in [0, 1].")


def run_flagship_ramsey_dephasing(cfg: FlagshipRamseyDephasingConfig) -> RamseyFrequencyPipelineResult:
    """
    Run the Ramsey dephasing flagship.

    Args:
        cfg: FlagshipRamseyDephasingConfig.

    Returns:
        RamseyFrequencyPipelineResult with simulated y if cfg.seed is not None.
    """
    t_s = np.linspace(float(cfg.t_min_s), float(cfg.t_max_s), int(cfg.n), dtype=np.float64)
    params = RamseyParams(gamma_phi=float(cfg.gamma_phi), phase0=float(cfg.phase0), visibility=float(cfg.visibility))

    rng = None
    if cfg.seed is not None:
        rng = np.random.default_rng(int(cfg.seed))

    return run_ramsey_frequency_pipeline(
        t_s=t_s,
        omega=float(cfg.omega),
        params=params,
        sigma_y=float(cfg.sigma_y),
        rng=rng,
    )
