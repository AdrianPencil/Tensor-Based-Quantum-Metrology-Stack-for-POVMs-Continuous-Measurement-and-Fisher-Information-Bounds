"""
Calibration utilities.

A calibration step typically fits a small parametric map between:
- a known/controlled reference signal x_true
- a measured device output y_meas

This file provides an affine calibrator with:
- fit (least squares)
- apply / invert
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "AffineCalibrator",
]


@dataclass(frozen=True, slots=True)
class AffineCalibrator:
    """
    Affine calibration model:

        y ≈ gain * x + offset

    The fit is ordinary least squares, vectorized and stable for typical calibration
    regimes. Use invert() when you want to map measured values back to sensor units.
    """

    gain: float
    offset: float

    @classmethod
    def fit(
        cls,
        x_true: npt.NDArray[np.float64],
        y_meas: npt.NDArray[np.float64],
    ) -> "AffineCalibrator":
        """
        Fit gain and offset with least squares.

        Args:
            x_true: reference input values (n,).
            y_meas: measured outputs (n,).

        Returns:
            AffineCalibrator(gain, offset)
        """
        x = np.asarray(x_true, dtype=np.float64)
        y = np.asarray(y_meas, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("x_true and y_meas must be 1D arrays of the same length.")

        a = np.column_stack([x, np.ones_like(x)])
        params, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
        gain = float(params[0])
        offset = float(params[1])
        return cls(gain=gain, offset=offset)

    def apply(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map sensor units -> measured units."""
        x_arr = np.asarray(x, dtype=np.float64)
        return (float(self.gain) * x_arr + float(self.offset)).astype(np.float64, copy=False)

    def invert(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map measured units -> sensor units."""
        if float(self.gain) == 0.0:
            raise ValueError("Cannot invert an affine calibration with gain=0.")
        y_arr = np.asarray(y, dtype=np.float64)
        return ((y_arr - float(self.offset)) / float(self.gain)).astype(np.float64, copy=False)

    def params(self) -> Tuple[float, float]:
        """Return (gain, offset) as plain floats."""
        return (float(self.gain), float(self.offset))
