"""Tests for qiboopt.dqi.circuit."""

import math

import numpy as np
import pytest

from qiboopt.dqi.circuit import dqi_circuit
from qiboopt.dqi.max_xorsat import MaxXORSAT
from qiboopt.dqi.weights import optimal_weights


def _binary_krawtchouk(k, d, m):
    """Binary Krawtchouk polynomial K_k(d; m) = sum_j (-1)^j C(d,j) C(m-d, k-j)."""
    total = 0
    for j in range(k + 1):
        total += (-1) ** j * math.comb(d, j) * math.comb(m - d, k - j)
    return total


def test_circuit_qubit_count():
    """Circuit acts on m + n qubits."""
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    circuit = dqi_circuit(problem, ell=2)
    assert circuit.nqubits == problem.m + problem.n


def test_circuit_no_measurements():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    circuit = dqi_circuit(problem, ell=2, include_measurements=False)
    assert all(g.name != "measure" for g in circuit.queue)


def test_circuit_measures_all_qubits_for_postselect():
    """include_measurements=True measures ALL m + n qubits.

    The error register is post-selected on |0> by DQISolver to project away
    decoder-failure branches (paper §8.1.2 / Fig. 4 caption: "postselect on |0>").
    """
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    circuit = dqi_circuit(problem, ell=2, include_measurements=True)
    measure_gates = [g for g in circuit.queue if g.name == "measure"]
    assert len(measure_gates) >= 1
    measured_qubits = set()
    for g in measure_gates:
        measured_qubits.update(g.target_qubits)
    expected = set(range(problem.m + problem.n))
    assert measured_qubits == expected


def test_invalid_ell():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    with pytest.raises(TypeError):
        dqi_circuit(problem, ell=1.0)
    with pytest.raises(ValueError):
        dqi_circuit(problem, ell=0)
    with pytest.raises(ValueError):
        dqi_circuit(problem, ell=problem.m + 1)


def test_circuit_rejects_wrong_length_weights():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    with pytest.raises(ValueError):
        dqi_circuit(problem, ell=2, weights=np.array([1.0, 0.0]))


def test_circuit_rejects_decoder_for_different_problem():
    """dqi_circuit must reject a pre-built decoder bound to a different
    MaxXORSAT instance (would silently build a wrong circuit)."""
    from qiboopt.dqi.decoders import LUTDecoder

    problem_a = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    problem_b = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=1)
    decoder_for_a = LUTDecoder(problem_a, ell=2)
    with pytest.raises(ValueError):
        dqi_circuit(problem_b, ell=2, decoder=decoder_for_a)


def test_circuit_rejects_decoder_with_different_ell():
    from qiboopt.dqi.decoders import LUTDecoder

    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    decoder_ell2 = LUTDecoder(problem, ell=2)
    with pytest.raises(ValueError):
        dqi_circuit(problem, ell=3, decoder=decoder_ell2)


def test_circuit_executes(backend):
    """Circuit runs under the numpy backend and produces (m + n)-bit samples."""
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    circuit = dqi_circuit(problem, ell=2)
    result = backend.execute_circuit(circuit, nshots=16)
    samples = np.asarray(result.samples(binary=True), dtype=np.uint8)
    assert samples.shape == (16, problem.m + problem.n)


def test_b_dependence_regression():
    """Two MaxXORSAT instances with same m, s but different B must produce different states.

    This is the regression for the original P0 bug: the buggy circuit
    didn't use B at all, and these two states were identical.
    """
    B1 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    B2 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    problem1 = MaxXORSAT(B1, s)
    problem2 = MaxXORSAT(B2, s)
    c1 = dqi_circuit(problem1, ell=1, include_measurements=False)
    c2 = dqi_circuit(problem2, ell=1, include_measurements=False)
    state1 = c1().state()
    state2 = c2().state()
    assert np.linalg.norm(state1 - state2) > 1e-6


def test_amplitudes_match_krawtchouk_formula_full_coverage():
    """For an instance with full LUT coverage (no collisions), the
    error-register-zero slice of the joint state matches the closed-form
    Krawtchouk amplitude formula to 1e-10.

    Construction: m=3, n=2, ell=1, B=[[1,0],[0,1],[1,1]]. The four
    weight-<=1 errors {(0,0,0), (1,0,0), (0,1,0), (0,0,1)} map bijectively
    onto {0,1}^2, so the LUT decoder is exact and the post-selection
    succeeds with probability 1.

    The DQI amplitude on x is (paper Eq. 25 + 27 + 22 + Krawtchouk
    rewriting):

        amp(x) = (1 / sqrt(2^n)) * sum_k (w_k / sqrt(C(m, k))) * K_k(|s + Bx|, m)
    """
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    ell = 1
    m, n = problem.m, problem.n
    w = optimal_weights(m, ell)

    circuit = dqi_circuit(problem, ell, include_measurements=False)
    state = circuit().state()
    n_register_amps = state[: 2**n]

    expected = np.zeros(2**n, dtype=complex)
    for x_int in range(2**n):
        x_bits = [(x_int >> (n - 1 - j)) & 1 for j in range(n)]
        x = np.array(x_bits, dtype=np.uint8)
        Bx = (B @ x) % 2
        d = int(np.sum(s ^ Bx))
        amp = 0.0
        for k in range(ell + 1):
            amp += w[k] * _binary_krawtchouk(k, d, m) / math.sqrt(math.comb(m, k))
        amp /= math.sqrt(2**n)
        expected[x_int] = amp

    assert np.allclose(n_register_amps, expected, atol=1e-10)
    assert abs(np.linalg.norm(n_register_amps) - 1.0) < 1e-10


@pytest.mark.parametrize(
    "B,s,ell",
    [
        (
            np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8),
            np.array([1, 0, 1], dtype=np.uint8),
            1,
        ),
        (
            np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8),
            np.array([0, 1, 1], dtype=np.uint8),
            2,
        ),
        (
            np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8),
            np.array([1, 0, 1, 0], dtype=np.uint8),
            2,
        ),
    ],
)
def test_amplitudes_match_chosen_S_sum(B, s, ell):
    """For arbitrary (B, s, ell), the error-register-zero slice of the
    joint state matches the partial sum over LUT-chosen ``S`` (paper Eq.
    34 with the perfect-decoder assumption replaced by the chosen-S
    selection of the LUT).

    For each LUT entry (y, S):
        amp_partial(x) = (1 / sqrt(2^n))
                         * sum_S (w_{|S|} / sqrt(C(m, |S|)))
                                 * (-1)^{1_S . (s + Bx)}

    where 1_S is the indicator vector of S. Stronger than the
    Krawtchouk-polynomial test because it covers ell >= 2 with collisions.
    """
    from qiboopt.dqi.decoders import LUTDecoder

    problem = MaxXORSAT(B, s)
    m, n = problem.m, problem.n
    w = optimal_weights(m, ell)

    decoder = LUTDecoder(problem, ell)
    table = decoder.decode_table()
    circuit = dqi_circuit(problem, ell, include_measurements=False)
    state = circuit().state()
    n_register_amps = state[: 2**n]

    expected = np.zeros(2**n, dtype=complex)
    for x_int in range(2**n):
        x_bits = [(x_int >> (n - 1 - j)) & 1 for j in range(n)]
        x = np.array(x_bits, dtype=np.uint8)
        u = (s ^ ((B @ x) % 2)).astype(int)
        amp = 0.0
        for _, S in table:
            k = int(S.sum())
            sign = (-1) ** int((S.astype(int) * u).sum())
            amp += sign * w[k] / math.sqrt(math.comb(m, k))
        amp /= math.sqrt(2**n)
        expected[x_int] = amp

    assert np.allclose(n_register_amps, expected, atol=1e-10)
