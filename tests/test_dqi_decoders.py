"""Tests for qiboopt.dqi.decoders."""

import numpy as np
import pytest

from qiboopt.dqi.decoders import LUTDecoder, get_decoder
from qiboopt.dqi.max_xorsat import MaxXORSAT


def test_lut_table_entries_satisfy_syndrome_relation():
    """Every (y, S) pair returned must satisfy y = B^T S (mod 2) and |S| <= ell."""
    problem = MaxXORSAT.random_sparse(n=6, m=8, row_weight=2, seed=42)
    ell = 2
    decoder = LUTDecoder(problem, ell)
    for y, S in decoder.decode_table():
        assert y.shape == (problem.n,)
        assert S.shape == (problem.m,)
        assert int(S.sum()) <= ell
        recomputed = (problem.B.T @ S) % 2
        assert np.array_equal(recomputed, y)


def test_lut_table_unique_y_per_entry():
    """The table must have at most one S per syndrome y."""
    problem = MaxXORSAT.random_sparse(n=6, m=8, row_weight=2, seed=42)
    decoder = LUTDecoder(problem, ell=2)
    seen = set()
    for y, _ in decoder.decode_table():
        key = tuple(y.tolist())
        assert key not in seen
        seen.add(key)


def test_lut_is_exact_when_full_coverage():
    """is_exact is True iff every weight-<=ell S has a distinct syndrome."""
    # Construction with no collisions: m=3, n=2, ell=1 gives 1+3=4 syndromes
    # and 4 unique error patterns mapping bijectively to {0,1}^2.
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    s = np.array([0, 0, 0], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    decoder = LUTDecoder(problem, ell=1)
    assert decoder.is_exact is True


def test_lut_is_exact_false_when_collisions():
    """The reviewer's example: m=3, n=2, ell=2 with B=[[1,0],[0,1],[1,1]]
    has 7 weight-<=2 patterns mapping to only 4 syndromes — collisions
    exist, so is_exact must be False."""
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    s = np.array([0, 0, 0], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    decoder = LUTDecoder(problem, ell=2)
    assert decoder.is_exact is False


def test_lut_too_many_error_patterns():
    """Decoder rejects instances whose enumeration exceeds the safety limit."""
    problem = MaxXORSAT.random_sparse(n=4, m=30, row_weight=2, seed=0)
    with pytest.raises(ValueError):
        LUTDecoder(problem, ell=10)


@pytest.mark.parametrize("ell", [-1, 4])
def test_lut_rejects_cutoff_outside_problem_size(ell):
    problem = MaxXORSAT.random_sparse(n=3, m=3, row_weight=2, seed=0)
    with pytest.raises(ValueError):
        LUTDecoder(problem, ell=ell)


def test_decoder_registry():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    decoder = get_decoder("lut", problem, 2)
    assert isinstance(decoder, LUTDecoder)


def test_unknown_decoder():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    with pytest.raises(KeyError):
        get_decoder("nonexistent", problem, 2)


def test_lut_no_runtime_decode_method():
    """Confirm the old runtime decode(y) method is gone."""
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    decoder = LUTDecoder(problem, ell=2)
    assert not hasattr(decoder, "decode")
