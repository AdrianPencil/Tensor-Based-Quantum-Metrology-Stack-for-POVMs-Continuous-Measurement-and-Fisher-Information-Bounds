"""
Consistency tests for Fisher and CRB utilities.

These tests enforce the basic identities:
- scalar: CRB variance = 1 / Fisher
- matrix: CRB covariance = inv(Fisher) (optionally with ridge)
"""

import numpy as np

from qmet.estimation.crb import crb_covariance, crb_variance_scalar
from qmet.estimation.fisher import fisher_gaussian_iid, fisher_scalar_gaussian_iid


def test_scalar_fisher_crb_identity() -> None:
    rng = np.random.default_rng(0)
    dmu = rng.normal(size=200).astype(np.float64)
    sigma = 0.2

    fisher = fisher_scalar_gaussian_iid(dmu_dtheta=dmu, sigma=sigma)
    var = crb_variance_scalar(fisher)

    assert fisher > 0.0
    assert np.isclose(var, 1.0 / fisher, rtol=1e-12, atol=0.0)


def test_matrix_fisher_crb_identity() -> None:
    rng = np.random.default_rng(1)
    n = 300
    p = 3
    jac = rng.normal(size=(n, p)).astype(np.float64)
    sigma = 0.3

    fim = fisher_gaussian_iid(jac=jac, sigma=sigma)
    cov = crb_covariance(fisher=fim)

    eye = np.eye(p, dtype=np.float64)
    assert np.allclose(fim @ cov, eye, atol=1e-10, rtol=0.0)
    assert np.allclose(cov @ fim, eye, atol=1e-10, rtol=0.0)


def test_matrix_crb_with_ridge_is_stable() -> None:
    rng = np.random.default_rng(2)
    n = 50
    p = 2
    jac = rng.normal(size=(n, p)).astype(np.float64)
    jac[:, 1] = jac[:, 0]

    sigma = 0.5
    fim = fisher_gaussian_iid(jac=jac, sigma=sigma)

    cov = crb_covariance(fisher=fim, ridge=1e-9)
    assert np.isfinite(cov).all()
    assert cov.shape == (p, p)
