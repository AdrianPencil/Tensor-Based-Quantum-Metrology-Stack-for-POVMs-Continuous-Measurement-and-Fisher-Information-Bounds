"""
Global configuration for qmet.

This centralizes numerical choices (dtype, tolerances, RNG seed) so that:
- experiments are reproducible
- numeric stability can be managed from one place
- downstream code does not hardcode tolerances everywhere
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Final, Mapping, MutableMapping, Optional

import numpy as np

from .types import DTypeName

__all__ = [
    "GlobalConfig",
    "load_config",
    "save_config",
]

_DEFAULT_ATOL: Final[float] = 1e-12
_DEFAULT_RTOL: Final[float] = 1e-12


@dataclass(slots=True)
class GlobalConfig:
    """
    Project-wide numeric and reproducibility knobs.

    dtype_real / dtype_complex are stored as strings to keep JSON clean.
    Use .np_dtype_real / .np_dtype_complex for NumPy dtype objects.
    """

    seed: Optional[int] = 0
    dtype_real: DTypeName = "float64"
    dtype_complex: DTypeName = "complex128"
    atol: float = _DEFAULT_ATOL
    rtol: float = _DEFAULT_RTOL

    def rng(self) -> np.random.Generator:
        """Return a reproducible RNG based on seed (or an entropy RNG if seed is None)."""
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(self.seed))

    @property
    def np_dtype_real(self) -> np.dtype:
        return np.dtype(self.dtype_real)

    @property
    def np_dtype_complex(self) -> np.dtype:
        return np.dtype(self.dtype_complex)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GlobalConfig":
        return cls(
            seed=d.get("seed", 0),
            dtype_real=d.get("dtype_real", "float64"),
            dtype_complex=d.get("dtype_complex", "complex128"),
            atol=float(d.get("atol", _DEFAULT_ATOL)),
            rtol=float(d.get("rtol", _DEFAULT_RTOL)),
        )


def load_config(path: str | Path) -> GlobalConfig:
    """Load GlobalConfig from a JSON file."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config JSON must be an object.")
    return GlobalConfig.from_dict(data)


def save_config(cfg: GlobalConfig, path: str | Path) -> None:
    """Save GlobalConfig to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
