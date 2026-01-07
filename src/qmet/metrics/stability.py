"""
Stability metrics.

This module provides a small wrapper for Allan deviation, returning a convenient
result structure for workflows and reports.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..noise.allan import overlapping_allan_deviation

__all__ = [
    "AllanResult",
    "allan_result",
]


@dataclass(frozen=True, slots=True)
class AllanResult:
    """Container for Allan deviation curves."""
    tau_s: npt.NDArray[np.float64]
    adev: npt.NDArray[np.float64]


def allan_result(
    y: npt.NDArray[np.float64],
    tau0: float,
    m_values: npt.NDArray[np.int64],
) -> AllanResult:
    """
    Compute Allan deviation curve as an AllanResult.

    Args:
        y: time series (n,).
        tau0: sample period (s).
        m_values: averaging factors (k,).

    Returns:
        AllanResult(tau_s, adev).
    """
    tau_s, adev = overlapping_allan_deviation(y=y, tau0=tau0, m_values=m_values)
    return AllanResult(tau_s=np.asarray(tau_s, dtype=np.float64), adev=np.asarray(adev, dtype=np.float64))
