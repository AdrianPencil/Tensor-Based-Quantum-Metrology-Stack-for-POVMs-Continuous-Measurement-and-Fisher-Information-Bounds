"""
Flagship experiments.

These are canonical, end-to-end demonstrations that exercise multiple layers:
sensor -> measurement -> estimation -> metrics, with clear defaults.

They are designed to be:
- stable reference points for tests
- readable examples for users
"""

from .ramsey_dephasing import run_flagship_ramsey_dephasing
from .weak_measurement_qubit import run_flagship_weak_measurement_qubit
from .quantum_limited_sensing import quantum_limit_gap_for_ramsey

__all__ = [
    "run_flagship_ramsey_dephasing",
    "run_flagship_weak_measurement_qubit",
    "quantum_limit_gap_for_ramsey",
]
