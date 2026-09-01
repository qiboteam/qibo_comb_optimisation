"""Tests for qiboopt.dqi.max_xorsat."""

import numpy as np
import pytest

from qiboopt.dqi.max_xorsat import MaxXORSAT


def test_construction_and_shape():
    B = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    assert problem.n == 3
    assert problem.m == 2


def test_evaluate():
    B = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    # x=[1,0,0]: Bx=[1,0]=s, both sat → 2
    assert problem.evaluate(np.array([1, 0, 0], dtype=np.uint8)) == 2
    # x=[0,0,0]: Bx=[0,0]; s=[1,0]; only second sat → 1
    assert problem.evaluate(np.array([0, 0, 0], dtype=np.uint8)) == 1
    # x=[1,1,1]: Bx=[0,0]; s=[1,0]; only second sat → 1
    assert problem.evaluate(np.array([1, 1, 1], dtype=np.uint8)) == 1


def test_brute_force():
    B = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    best_x, best_v = problem.brute_force()
    assert best_v == 3
    assert problem.evaluate(best_x) == 3


def test_random_sparse_shape():
    problem = MaxXORSAT.random_sparse(n=8, m=12, row_weight=3, seed=42)
    assert problem.n == 8
    assert problem.m == 12
    assert (problem.B.sum(axis=1) == 3).all()


def test_random_sparse_reproducible():
    a = MaxXORSAT.random_sparse(n=8, m=12, row_weight=3, seed=42)
    b = MaxXORSAT.random_sparse(n=8, m=12, row_weight=3, seed=42)
    assert np.array_equal(a.B, b.B)
    assert np.array_equal(a.s, b.s)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        MaxXORSAT(np.zeros((2,), dtype=np.uint8), np.zeros((2,), dtype=np.uint8))
    with pytest.raises(ValueError):
        MaxXORSAT(np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 1), dtype=np.uint8))
    with pytest.raises(ValueError):
        MaxXORSAT(np.zeros((2, 3), dtype=np.uint8), np.zeros((3,), dtype=np.uint8))
    with pytest.raises(ValueError):
        MaxXORSAT(np.array([[2, 0], [0, 0]]), np.array([0, 0]))
    with pytest.raises(ValueError):
        MaxXORSAT(np.array([[1, 0], [0, 1]]), np.array([2, 0]))


def test_to_qubo_round_trip_row_weight_2():
    """For row weights 1 and 2, to_qubo + minimisation = max-XOR-SAT maximiser."""
    B = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    s = np.array([0, 1], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    qubo = problem.to_qubo()
    # The QUBO objective should equal m - sat_count for every x.
    for bits in [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]:
        x = np.array(bits, dtype=np.uint8)
        sat = problem.evaluate(x)
        cost = qubo.evaluate_f(list(bits))
        assert cost == problem.m - sat, (bits, sat, cost)


def test_to_qubo_round_trip_row_weight_1_targets():
    """Unit clauses become linear penalties for violating x_j = target."""
    B = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    s = np.array([0, 1], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    qubo = problem.to_qubo(scale=2.0)

    for bits in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        x = np.array(bits, dtype=np.uint8)
        assert qubo.evaluate_f(list(bits)) == 2.0 * (problem.m - problem.evaluate(x))


def test_to_qubo_unsupported_row_weight():
    B = np.array([[1, 1, 1]], dtype=np.uint8)
    s = np.array([0], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    with pytest.raises(NotImplementedError):
        problem.to_qubo()


def test_rejects_non_integer_floats_in_B():
    """Regression: non-integer floats must NOT silently truncate to uint8."""
    with pytest.raises(ValueError):
        MaxXORSAT(np.array([[1.9, 0.0]]), np.array([0]))


def test_rejects_non_integer_floats_in_s():
    with pytest.raises(ValueError):
        MaxXORSAT(np.array([[1, 0]]), np.array([0.5]))


def test_accepts_integer_valued_floats():
    """Float arrays whose values are exactly integer should be accepted."""
    problem = MaxXORSAT(np.array([[1.0, 0.0]]), np.array([1.0]))
    assert problem.B.dtype == np.uint8
    assert problem.s.dtype == np.uint8


def test_evaluate_rejects_non_integer_floats():
    """Regression: evaluate(x) must NOT silently coerce non-integer floats."""
    problem = MaxXORSAT(np.array([[1, 1, 0]]), np.array([0]))
    with pytest.raises(ValueError):
        problem.evaluate(np.array([1.9, 0.0, 1.0]))


def test_evaluate_rejects_wrong_shape():
    problem = MaxXORSAT(np.array([[1, 1, 0]]), np.array([0]))
    with pytest.raises(ValueError):
        problem.evaluate(np.array([[1, 0, 1]], dtype=np.uint8))


def test_evaluate_rejects_out_of_range_integers():
    """Values outside {0, 1} must be rejected after coercion."""
    problem = MaxXORSAT(np.array([[1, 1, 0]]), np.array([0]))
    with pytest.raises(ValueError):
        problem.evaluate(np.array([2, 0, 1]))


def test_evaluate_accepts_integer_valued_floats():
    """Float values exactly equal to integers are valid."""
    problem = MaxXORSAT(np.array([[1, 1, 0]]), np.array([0]))
    # x=[1, 1, 0] : Bx = 1+1 = 0 (mod 2), s = 0, 1 sat -> 1.
    assert problem.evaluate(np.array([1.0, 1.0, 0.0])) == 1


@pytest.mark.parametrize("bad_value", [-255, 257, 256, -1, 2])
def test_constructor_rejects_uint8_wraparound_values(bad_value):
    """Regression: values that wrap to 0 or 1 modulo 256 must be rejected.

    Before the fix, ``bad_value`` was cast to uint8 first and then checked,
    so e.g. -255 (-> 1) and 257 (-> 1) silently passed.
    """
    with pytest.raises(ValueError):
        MaxXORSAT(np.array([[bad_value, 0]]), np.array([0]))
    with pytest.raises(ValueError):
        MaxXORSAT(np.array([[1, 0]]), np.array([bad_value]))


@pytest.mark.parametrize("bad_value", [-255, 257, 256, -1, 2])
def test_evaluate_rejects_uint8_wraparound_values(bad_value):
    """Regression for the same wraparound issue, on evaluate(x)."""
    problem = MaxXORSAT(np.array([[1, 1, 0]]), np.array([0]))
    with pytest.raises(ValueError):
        problem.evaluate(np.array([bad_value, 0, 0]))


def test_brute_force_rejects_large_variable_count():
    """Exhaustive search is deliberately capped before 2**n becomes excessive."""
    problem = MaxXORSAT(
        np.zeros((1, 21), dtype=np.uint8), np.array([0], dtype=np.uint8)
    )
    with pytest.raises(ValueError):
        problem.brute_force()


@pytest.mark.parametrize("row_weight", [0, 5])
def test_random_sparse_rejects_impossible_row_weight(row_weight):
    with pytest.raises(ValueError):
        MaxXORSAT.random_sparse(n=4, m=3, row_weight=row_weight, seed=0)
