"""
Tests for the weak measurement qubit flagship.

These tests enforce basic physical sanity of the trajectory:
- rho remains Hermitian
- trace is ~ 1
- eigenvalues are nonnegative up to a small numerical tolerance
"""

import numpy as np

from qmet.flagship.weak_measurement_qubit import WeakMeasurementConfig, run_flagship_weak_measurement_qubit


def test_flagship_weak_measurement_trajectory_sanity() -> None:
    cfg = WeakMeasurementConfig(n=300, dt=1e-6, kappa=2e5, eta=0.8, seed=0)
    res = run_flagship_weak_measurement_qubit(cfg)

    assert res.t_s.shape == (cfg.n,)
    assert res.d_y.shape == (cfg.n,)
    assert res.rho_traj.shape == (cfg.n + 1, 2, 2)
    assert np.isfinite(res.d_y).all()

    for k in (0, cfg.n // 2, cfg.n):
        rho = res.rho_traj[k]
        assert np.allclose(rho, rho.conjugate().T, atol=1e-10, rtol=0.0)
        tr = np.trace(rho)
        assert np.isclose(tr, 1.0, atol=1e-10, rtol=0.0)

        w = np.linalg.eigvalsh(rho)
        assert float(np.min(w)) >= -1e-9
