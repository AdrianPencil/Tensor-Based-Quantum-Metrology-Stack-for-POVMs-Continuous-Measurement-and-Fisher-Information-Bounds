"""
Device response models.

These map a "sensor-level" quantity (typically an expectation value) to a
measured device output in physical units (e.g., volts, counts, arbitrary units).

Keep it simple and explicit:
- affine gain + offset
- optional clipping (saturation)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = [
    "AffineDeviceResponse",
]


@dataclass(frozen=True, slots=True)
class AffineDeviceResponse:
    """
    y = gain * x + offset, optionally clipped to [y_min, y_max].

    This is a deliberately small building block because most practical readout
    chains can be locally approximated as affine around an operating point.
    """

    gain: float = 1.0
    offset: float = 0.0
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    def apply(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply the device response to a vector of inputs.

        Args:
            x: array of sensor-level values.

        Returns:
            y: array of device outputs.
        """
        y = float(self.gain) * x + float(self.offset)
        if self.y_min is not None:
            y = np.maximum(y, float(self.y_min))
        if self.y_max is not None:
            y = np.minimum(y, float(self.y_max))
        return y.astype(np.float64, copy=False)
