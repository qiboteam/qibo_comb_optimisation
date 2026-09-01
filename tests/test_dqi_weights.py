"""Tests for qiboopt.dqi.weights."""

import numpy as np
import pytest

from qiboopt.dqi.weights import optimal_weights


def test_basic_shape():
    w = optimal_weights(8, 3)
    assert w.shape == (4,)


def test_unit_norm():
    for m, ell in [(4, 2), (8, 3), (12, 4), (16, 5)]:
        w = optimal_weights(m, ell)
        assert abs(np.linalg.norm(w) - 1.0) < 1e-12


def test_sign_convention():
    """Acceptance criterion 1.6: w[0] > 0 by convention."""
    for m, ell in [(4, 2), (6, 3), (8, 3), (10, 4)]:
        w = optimal_weights(m, ell)
        assert w[0] > 0, f"w[0]={w[0]} should be positive for m={m}, ell={ell}"


@pytest.mark.parametrize(
    "m,ell",
    [(4, 2), (6, 3), (8, 3)],
)
def test_matches_dense_eig(m, ell):
    """Acceptance criterion 1.1: match dense numpy eigvec to 1e-10."""
    w = optimal_weights(m, ell)

    # Build the dense tridiagonal matrix and solve via numpy.
    A = np.zeros((ell + 1, ell + 1))
    for k in range(ell):
        A[k, k + 1] = A[k + 1, k] = np.sqrt((k + 1) * (m - k))
    eigvals, eigvecs = np.linalg.eigh(A)
    expected = eigvecs[:, -1]
    if expected[0] < 0:
        expected = -expected

    assert np.allclose(w, expected, atol=1e-10)


def test_invalid_args():
    with pytest.raises(TypeError):
        optimal_weights(4.0, 2)
    with pytest.raises(ValueError):
        optimal_weights(0, 2)
    with pytest.raises(ValueError):
        optimal_weights(4, 0)
    with pytest.raises(ValueError):
        optimal_weights(4, 5)
