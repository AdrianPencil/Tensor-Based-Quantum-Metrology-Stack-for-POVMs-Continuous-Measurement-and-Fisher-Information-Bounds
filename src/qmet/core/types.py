"""
Core typing utilities for qmet.

This module keeps "what shapes/types are we passing around?" in one place.
It aims to be small and dependency-light, while still making code self-documenting.

Conventions:
- Matrices are (d, d)
- State vectors are (d,)
- Time series are (n,)
- Stacked batches are (..., d, d) where useful
"""

from dataclasses import dataclass
from typing import Any, Literal, NewType, TypeAlias

import numpy as np
import numpy.typing as npt

Float: TypeAlias = np.float64
Complex: TypeAlias = np.complex128

RealArray: TypeAlias = npt.NDArray[np.floating[Any]]
ComplexArray: TypeAlias = npt.NDArray[np.complexfloating[Any]]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

DTypeName = Literal["float64", "complex128"]

Radian: TypeAlias = float
Second: TypeAlias = float
Hertz: TypeAlias = float

Dim = NewType("Dim", int)


@dataclass(frozen=True, slots=True)
class LinearOp:
    """
    A small container for linear operators used throughout qmet.

    This keeps a consistent naming scheme:
    - op: the matrix (d, d)
    - dim: Hilbert space dimension
    """

    op: ComplexArray
    dim: int

    def __post_init__(self) -> None:
        if self.op.ndim != 2 or self.op.shape[0] != self.op.shape[1]:
            raise ValueError("LinearOp.op must be a square 2D array.")
        if self.op.shape[0] != self.dim:
            raise ValueError("LinearOp.dim must match op.shape[0].")
