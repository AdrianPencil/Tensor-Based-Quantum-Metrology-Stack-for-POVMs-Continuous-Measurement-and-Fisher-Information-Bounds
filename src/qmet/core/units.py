"""
Units and physical constants used across qmet.

Goal:
- Make unit conversions explicit and uniform
- Keep constants centralized to avoid silent inconsistencies

This is intentionally small: only conversions/constants we expect to use broadly.
"""

from typing import Final

import numpy as np
from scipy import constants as _c

__all__ = [
    "h",
    "hbar",
    "kB",
    "pi",
    "two_pi",
    "hz_to_rad_s",
    "rad_s_to_hz",
    "db_to_linear_power",
    "linear_power_to_db",
]

pi: Final[float] = float(np.pi)
two_pi: Final[float] = float(2.0 * np.pi)

h: Final[float] = float(_c.h)
hbar: Final[float] = float(_c.hbar)
kB: Final[float] = float(_c.k)


def hz_to_rad_s(f_hz: float) -> float:
    """Convert frequency in Hz to angular frequency in rad/s."""
    return two_pi * float(f_hz)


def rad_s_to_hz(omega: float) -> float:
    """Convert angular frequency in rad/s to frequency in Hz."""
    return float(omega) / two_pi


def db_to_linear_power(db: float) -> float:
    """Convert dB (power) to linear power ratio."""
    return float(10.0 ** (float(db) / 10.0))


def linear_power_to_db(p: float) -> float:
    """Convert linear power ratio to dB (power)."""
    p_float = float(p)
    if p_float <= 0.0:
        raise ValueError("linear power must be > 0 to convert to dB.")
    return float(10.0 * np.log10(p_float))
