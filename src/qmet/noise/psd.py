"""
Power spectral density (PSD) utilities.

This module estimates one-sided PSD from a real-valued time series using Welch's method.
It is a compact, practical tool for noise characterization in case studies.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import welch

__all__ = [
    "PSDResult",
    "welch_psd",
]


@dataclass(frozen=True, slots=True)
class PSDResult:
    """Container for a one-sided PSD estimate."""
    freq_hz: npt.NDArray[np.float64]
    psd: npt.NDArray[np.float64]


def welch_psd(
    x: npt.NDArray[np.float64],
    fs_hz: float,
    nperseg: int | None = None,
) -> PSDResult:
    """
    Estimate one-sided PSD using Welch.

    Args:
        x: real-valued signal (n,).
        fs_hz: sampling frequency in Hz (> 0).
        nperseg: Welch segment length.

    Returns:
        PSDResult(freq_hz, psd).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("x must be 1D.")
    fs = float(fs_hz)
    if fs <= 0.0:
        raise ValueError("fs_hz must be > 0.")

    f, p = welch(x_arr, fs=fs, nperseg=nperseg, return_onesided=True, scaling="density")
    return PSDResult(freq_hz=np.asarray(f, dtype=np.float64), psd=np.asarray(p, dtype=np.float64))
