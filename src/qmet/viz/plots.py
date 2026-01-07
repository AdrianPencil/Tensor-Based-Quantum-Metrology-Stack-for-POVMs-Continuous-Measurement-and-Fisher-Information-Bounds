"""
Plotting helpers.

This module keeps plotting optional. It imports matplotlib lazily so core
numerics remain usable in environments without plotting backends.

Initial scope:
- time-series plot for (t, mu) and optional (t, y)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = [
    "TimeSeriesPlotData",
    "plot_time_series",
]


@dataclass(frozen=True, slots=True)
class TimeSeriesPlotData:
    """Container for plot-ready time series arrays."""
    t_s: npt.NDArray[np.float64]
    mu: npt.NDArray[np.float64]
    y: Optional[npt.NDArray[np.float64]] = None


def plot_time_series(data: TimeSeriesPlotData, *, title: str = "qmet time series") -> object:
    """
    Plot mean signal and optional noisy observations.

    Args:
        data: TimeSeriesPlotData.
        title: plot title.

    Returns:
        matplotlib Figure object.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from exc

    t = np.asarray(data.t_s, dtype=np.float64)
    mu = np.asarray(data.mu, dtype=np.float64)
    if t.ndim != 1 or mu.ndim != 1 or t.shape != mu.shape:
        raise ValueError("t_s and mu must be 1D arrays with the same shape.")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, mu, label="mu(t)")
    if data.y is not None:
        y = np.asarray(data.y, dtype=np.float64)
        if y.shape != t.shape:
            raise ValueError("y must match t_s shape.")
        ax.plot(t, y, linestyle="none", marker=".", label="y (noisy)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("signal")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
