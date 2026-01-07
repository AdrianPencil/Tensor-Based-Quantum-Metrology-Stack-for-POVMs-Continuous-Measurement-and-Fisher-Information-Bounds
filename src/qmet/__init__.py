"""
qmet: a modular quantum metrology stack.

This package is organized as:
- core: foundational types, units, linear algebra, and global configuration
- sensor / measurement / estimation: the core "sensor -> measurement -> estimator" chain
- protocols / noise / metrics: reusable building blocks
- workflows / viz: orchestration and visualization
- flagship / case_studies: canonical experiments and grounded examples
"""

from .core.config import GlobalConfig

__all__ = [
    "GlobalConfig",
]

__version__ = "0.1.0"
