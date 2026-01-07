"""
Case studies.

These are grounded "application stories" that connect the library components
into narratives that look like real experiments:
- choose a physical-ish scenario
- run a protocol + readout chain
- compute sensitivity / stability / bounds
"""

from .nv_ramsey_sensitivity import run_nv_ramsey_sensitivity
from .readout_chain_example import run_readout_chain_example

__all__ = [
    "run_nv_ramsey_sensitivity",
    "run_readout_chain_example",
]
