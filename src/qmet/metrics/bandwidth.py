"""
Bandwidth metrics.

This module provides a minimal -3 dB bandwidth estimator from a frequency response.

Given a transfer magnitude |H(f)| sampled on a frequency grid f (Hz):
- normalize to DC or max magnitude
- find the first frequency where magnitude <= 1/sqrt(2)
"""

import numpy as np
import numpy.typing as npt

__all__ = [
    "bandwidth_3db_hz",
]


def bandwidth_3db_hz(
    freq_hz: npt.NDArray[np.float64],
    mag: npt.NDArray[np.float64],
    normalize: str = "max",
) -> float:
    """
    Estimate -3 dB bandwidth in Hz.

    Args:
        freq_hz: frequency grid (n,), nonnegative, increasing preferred.
        mag: magnitude response (n,), nonnegative.
        normalize: "max" or "dc".

    Returns:
        f_3db: bandwidth in Hz (float), or nan if not found.
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    m = np.asarray(mag, dtype=np.float64)
    if f.ndim != 1 or m.ndim != 1 or f.shape != m.shape:
        raise ValueError("freq_hz and mag must be 1D arrays of the same shape.")
    if f.size < 2:
        raise ValueError("Need at least 2 points to estimate bandwidth.")
    if np.any(m < 0.0):
        raise ValueError("mag must be nonnegative.")

    if normalize == "max":
        ref = float(np.max(m))
    elif normalize == "dc":
        ref = float(m[0])
    else:
        raise ValueError('normalize must be "max" or "dc".')

    if ref <= 0.0:
        return float("nan")

    m_n = m / ref
    thr = 1.0 / float(np.sqrt(2.0))
    idx = np.where(m_n <= thr)[0]
    if idx.size == 0:
        return float("nan")
    return float(f[int(idx[0])])
