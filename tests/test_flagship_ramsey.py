"""
Tests for the Ramsey dephasing flagship.

These tests enforce:
- shapes are consistent
- Fisher is nonnegative
- CRB variance is consistent with 1/Fisher for the scalar case
- optional simulated data is produced when a seed is provided
"""

import numpy as np

from qmet.flagship.ramsey_dephasing import FlagshipRamseyDephasingConfig, run_flagship_ramsey_dephasing


def test_flagship_ramsey_dephasing_shapes_and_bounds() -> None:
    cfg = FlagshipRamseyDephasingConfig(
        omega=2.0 * np.pi * 1e6,
        gamma_phi=2e3,
        sigma_y=0.05,
        t_min_s=0.0,
        t_max_s=1e-3,
        n=200,
        phase0=0.1,
        visibility=0.9,
        seed=0,
    )
    res = run_flagship_ramsey_dephasing(cfg)

    assert res.t_s.ndim == 1
    assert res.mu.shape == res.t_s.shape
    assert res.dmu_domega.shape == res.t_s.shape

    assert np.isfinite(res.fisher_omega)
    assert res.fisher_omega >= 0.0

    if res.fisher_omega > 0.0:
        assert np.isclose(res.crb_var_omega, 1.0 / res.fisher_omega, rtol=1e-12, atol=0.0)

    assert res.y is not None
    assert res.y.shape == res.t_s.shape
