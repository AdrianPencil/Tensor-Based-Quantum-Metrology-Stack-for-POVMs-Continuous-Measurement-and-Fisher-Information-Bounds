"""
Allan deviation (stability) utilities.

This module implements the overlapping Allan deviation for a uniformly sampled
time series y_k at sampling period tau0.

Overlapping Allan variance for averaging factor m:
    sigma_A^2(m) = (1 / (2 * (N - 2m))) * sum_{k=0}^{N-2m-1} (ȳ_{k+m} - ȳ_k)^2

where ȳ_k is the average over m consecutive samples starting at k.
"""

import numpy as np
import numpy.typing as npt

__all__ = [
    "overlapping_allan_deviation",
]


def overlapping_allan_deviation(
    y: npt.NDArray[np.float64],
    tau0: float,
    m_values: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute overlapping Allan deviation for multiple averaging factors m.

    Args:
        y: uniformly sampled series (n,).
        tau0: sampling period in seconds (> 0).
        m_values: integer averaging factors (k,).

    Returns:
        taus: tau = m * tau0 (k,)
        adev: Allan deviation (k,)
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValueError("y must be 1D.")
    t0 = float(tau0)
    if t0 <= 0.0:
        raise ValueError("tau0 must be > 0.")

    m_arr = np.asarray(m_values, dtype=np.int64)
    if m_arr.ndim != 1:
        raise ValueError("m_values must be 1D.")
    if np.any(m_arr <= 0):
        raise ValueError("m_values must be positive integers.")

    n = int(y_arr.size)
    taus = np.empty_like(m_arr, dtype=np.float64)
    adev = np.empty_like(m_arr, dtype=np.float64)

    c = np.cumsum(y_arr, dtype=np.float64)
    c = np.concatenate([np.array([0.0], dtype=np.float64), c])

    for i, m in enumerate(m_arr):
        m_i = int(m)
        taus[i] = float(m_i) * t0

        if n < 2 * m_i + 1:
            adev[i] = np.nan
            continue

        start = np.arange(0, n - m_i + 1, dtype=np.int64)
        sums = c[start + m_i] - c[start]
        ybar = sums / float(m_i)

        diffs = ybar[m_i:] - ybar[:-m_i]
        denom = 2.0 * float(diffs.size)
        avar = float(np.sum(diffs * diffs) / denom) if diffs.size > 0 else np.nan
        adev[i] = float(np.sqrt(avar)) if np.isfinite(avar) else np.nan

    return taus, adev
