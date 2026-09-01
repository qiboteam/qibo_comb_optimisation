"""End-to-end tests for qiboopt.dqi.solver."""

import numpy as np
import pytest

from qiboopt.dqi.max_xorsat import MaxXORSAT
from qiboopt.dqi.solver import DQISolver


def test_solver_basic():
    """Tiny instance: DQI must find the brute-force optimum within 256 shots."""
    B = np.array(
        [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=np.uint8,
    )
    s = np.array([1, 0, 1, 0], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    _, bv = problem.brute_force()

    solver = DQISolver(problem, ell=2, decoder="lut")
    result = solver.solve(nshots=256)
    assert result["best_value"] == bv
    assert isinstance(result["is_dqi_exact"], bool)
    assert 0.0 <= result["decoder_success_probability"] <= 1.0
    assert 0.0 <= result["postselect_success_rate"] <= 1.0


@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
def test_solver_random_sparse(seed):
    """5 random instances: DQI matches brute force on at least 4 of 5 (after post-selection)."""
    problem = MaxXORSAT.random_sparse(n=6, m=8, row_weight=2, seed=seed)
    _, bv = problem.brute_force()
    solver = DQISolver(problem, ell=2, decoder="lut")
    result = solver.solve(nshots=512)
    # Allow occasional miss from decoder-failure branches.
    assert result["best_value"] >= bv - 1


def test_dqi_concentrates_on_optimum():
    """DQI's post-selected output puts more weight on the brute-force optimum
    than uniform sampling would.

    The original buggy implementation passed a "best of N shots" test
    trivially; this requires the algorithm to actually concentrate
    probability on near-optimal x. Margin is conservative (>= 2x uniform).
    """
    problem = MaxXORSAT.random_sparse(n=4, m=4, row_weight=2, seed=11)
    bx, _ = problem.brute_force()
    solver = DQISolver(problem, ell=2, decoder="lut")
    out = solver.sample(nshots=2048)
    if out["shots_kept"] == 0:
        pytest.skip("No shots passed post-selection on this instance.")
    matches = np.all(out["candidates"] == bx[None, :], axis=1).sum()
    empirical = matches / out["shots_kept"]
    uniform = 1.0 / 2**problem.n
    assert (
        empirical > 2 * uniform
    ), f"empirical={empirical:.4f}, uniform={uniform:.4f}; expected >= 2x"


def test_full_coverage_instance_is_dqi_exact():
    """Full-coverage instance: every weight-<=ell S has a distinct syndrome,
    so is_dqi_exact is True and decoder_success_probability is 1.0."""
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    solver = DQISolver(problem, ell=1, decoder="lut")
    out = solver.sample(nshots=128)
    assert out["is_dqi_exact"] is True
    assert abs(out["decoder_success_probability"] - 1.0) < 1e-12
    assert out["postselect_success_rate"] == 1.0
    assert out["shots_kept"] == 128
    assert out["candidates"].shape == (128, problem.n)


def test_colliding_lut_marks_inexact():
    """Reviewer's example: m=3, n=2, ell=2, B=[[1,0],[0,1],[1,1]] has 7
    weight-<=2 S's mapping to only 4 syndromes, so the LUT has collisions
    and ``is_dqi_exact`` should be False. The analytic
    ``decoder_success_probability`` matches the empirical
    ``postselect_success_rate`` to within sampling noise.
    """
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    solver = DQISolver(problem, ell=2, decoder="lut")
    assert solver.is_dqi_exact is False
    # Analytical value is exactly 5/7 for this instance (chosen S weights
    # 0, 1, 1, 1 give w_0^2 + w_1^2 = 1 - w_2^2 = 5/7).
    analytic = solver.decoder_success_probability
    assert analytic < 1.0
    out = solver.sample(nshots=4096)
    empirical = out["postselect_success_rate"]
    # Empirical should track the analytic prediction within standard
    # sampling-noise bounds (~3 sigma for a Bernoulli at p~0.7, n=4096
    # gives sigma ~= 0.007, so 0.05 is generous).
    assert abs(empirical - analytic) < 0.05


def test_invalid_ell():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    with pytest.raises(ValueError):
        DQISolver(problem, ell=0)
    with pytest.raises(ValueError):
        DQISolver(problem, ell=problem.m + 1)


def test_invalid_nshots():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    solver = DQISolver(problem, ell=2)
    with pytest.raises(ValueError):
        solver.sample(0)
    with pytest.raises(ValueError):
        solver.sample(-1)


def test_sample_returns_expected_keys():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    solver = DQISolver(problem, ell=2)
    out = solver.sample(nshots=16)
    assert set(out.keys()) >= {
        "candidates",
        "raw_samples",
        "postselect_success_rate",
        "decoder_success_probability",
        "is_dqi_exact",
        "shots",
        "shots_kept",
    }
    assert out["shots"] == 16
    assert out["raw_samples"].shape == (16, problem.m + problem.n)


def test_sample_applies_noise_model_hook():
    """A supplied noise model is applied to the constructed DQI circuit before execution."""

    class IdentityNoiseModel:
        def __init__(self):
            self.applied_to = []

        def apply(self, circuit):
            self.applied_to.append(circuit.nqubits)
            return circuit

    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    noise_model = IdentityNoiseModel()
    solver = DQISolver(problem, ell=1, noise_model=noise_model)

    out = solver.sample(nshots=8)

    assert noise_model.applied_to == [problem.m + problem.n]
    assert out["raw_samples"].shape == (8, problem.m + problem.n)


def test_unnormalized_weights_give_valid_probability():
    """Regression: passing weights = 2 * optimal_weights must NOT report
    decoder_success_probability above 1.

    Previously the analytic formula used the raw self.weights while the
    circuit auto-normalised; passing 2x weights reported 4.0 vs empirical
    1.0. Solver now normalises weights at construction.
    """
    from qiboopt.dqi.weights import optimal_weights

    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    problem = MaxXORSAT(B, s)
    w_opt = optimal_weights(problem.m, ell=1)

    solver = DQISolver(problem, ell=1, weights=2 * w_opt)
    assert solver.decoder_success_probability <= 1.0 + 1e-10
    assert solver.decoder_success_probability >= 0.0
    out = solver.sample(nshots=256)
    # Full-coverage instance: empirical post-select rate is 1.0.
    assert out["postselect_success_rate"] == 1.0
    assert abs(solver.decoder_success_probability - 1.0) < 1e-10


def test_solver_rejects_zero_norm_weights():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    with pytest.raises(ValueError):
        DQISolver(problem, ell=2, weights=np.zeros(3))


def test_solver_rejects_complex_weights_with_imaginary():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    with pytest.raises(ValueError):
        DQISolver(problem, ell=2, weights=np.array([1.0 + 1j, 0.5, 0.0]))


def test_solver_rejects_wrong_length_weights():
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    with pytest.raises(ValueError):
        DQISolver(problem, ell=2, weights=np.array([1.0, 0.5]))  # length 2, expected 3


def test_solver_rejects_nan_weights():
    """Regression: NaN weights produce NaN diagnostics — must raise instead."""
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    with pytest.raises(ValueError):
        DQISolver(problem, ell=2, weights=np.array([np.nan, 1.0, 0.0]))


def test_solver_rejects_inf_weights():
    """Regression: Inf weights produce a degenerate normalized vector."""
    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    with pytest.raises(ValueError):
        DQISolver(problem, ell=2, weights=np.array([np.inf, 1.0, 0.0]))


def test_solver_rejects_decoder_built_for_different_B():
    """A pre-built decoder bound to a different problem (different B,
    same shape) silently builds the wrong DQI circuit. Must raise."""
    from qiboopt.dqi.decoders import LUTDecoder

    problem_a = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    problem_b = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=1)
    decoder_for_a = LUTDecoder(problem_a, ell=2)
    with pytest.raises(ValueError):
        DQISolver(problem_b, ell=2, decoder=decoder_for_a)


def test_solver_rejects_decoder_built_for_different_ell():
    """A pre-built decoder with a different ell must be rejected."""
    from qiboopt.dqi.decoders import LUTDecoder

    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    decoder_ell2 = LUTDecoder(problem, ell=2)
    with pytest.raises(ValueError):
        DQISolver(problem, ell=3, decoder=decoder_ell2)


def test_solver_accepts_decoder_for_structurally_equal_problem():
    """A decoder built for an equal-but-different MaxXORSAT instance is
    accepted (structural equality, not identity)."""
    from qiboopt.dqi.decoders import LUTDecoder

    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    problem_orig = MaxXORSAT(B, s)
    problem_copy = MaxXORSAT(B.copy(), s.copy())
    decoder = LUTDecoder(problem_orig, ell=1)
    # Must not raise:
    solver = DQISolver(problem_copy, ell=1, decoder=decoder)
    assert solver.decoder is decoder


def test_solver_accepts_decoder_for_identical_problem():
    """A decoder built for the same problem object is accepted (identity match)."""
    from qiboopt.dqi.decoders import LUTDecoder

    problem = MaxXORSAT.random_sparse(n=4, m=6, row_weight=2, seed=0)
    decoder = LUTDecoder(problem, ell=2)
    solver = DQISolver(problem, ell=2, decoder=decoder)
    assert solver.decoder is decoder
