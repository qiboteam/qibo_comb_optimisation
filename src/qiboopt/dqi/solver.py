"""
Top-level DQI solver: build the (m + n)-qubit circuit, sample the
solution register, return the best candidate.
"""

from math import comb

import numpy as np
from qibo.backends import _check_backend
from qibo.config import raise_error

from qiboopt.dqi.circuit import dqi_circuit
from qiboopt.dqi.decoders import get_decoder
from qiboopt.dqi.decoders.base import SyndromeDecoder, validate_decoder_compatibility
from qiboopt.dqi.weights import optimal_weights


class DQISolver:
    r"""Decoded Quantum Interferometry solver for a max-XOR-SAT instance.

    Builds the DQI circuit on ``problem.m + problem.n`` qubits — error
    register on the first :math:`m`, solution register on the last
    :math:`n` — executes it under a Qibo backend, post-selects the error
    register on :math:`\ket{0^m}`, and returns the candidate :math:`x`
    measured directly from the solution register. The classical decoder
    is consumed at *circuit-construction* time (its ``decode_table`` is
    compiled into in-circuit multi-controlled gates); there is no
    classical post-processing of measurement outcomes beyond the
    error-register post-selection.

    **DQI exactness.** The canonical DQI procedure of Jordan et al.
    requires that the in-circuit decoder uniquely recover :math:`S` from
    :math:`B^\top S` for every :math:`|S| \le \ell`. When that holds
    (:attr:`is_dqi_exact` is ``True``), the post-selected solution-register
    samples follow :math:`|P(f)|^2`, the canonical DQI distribution. When
    it does not (the LUT has collisions on the Dicke support), the
    post-selected samples follow a *modified* distribution restricted to
    the LUT-chosen :math:`S` patterns; these may still concentrate near
    optima but lose the paper's quantitative guarantees.

    Args:
        problem (:class:`qiboopt.dqi.max_xorsat.MaxXORSAT`): The max-XOR-SAT
            instance.
        ell (int): Hamming-weight cutoff for the DQI superposition. Must
            satisfy ``1 <= ell <= problem.m``.
        decoder (str | :class:`SyndromeDecoder`, optional): Either a
            registered decoder name (currently only ``"lut"``) or a
            pre-instantiated :class:`SyndromeDecoder`. Defaults to
            ``"lut"``.
        weights (np.ndarray, optional): Length-``ell + 1`` weight vector. If
            ``None``, the optimal weights are computed via
            :func:`qiboopt.dqi.weights.optimal_weights`.
        backend: Qibo backend to execute the circuit. If ``None``, Qibo's
            default backend is used.
        noise_model: Optional Qibo noise model attached to circuit execution.
            DQI noise robustness is open research; passing a noise model
            works mechanically but the resulting samples are not guaranteed
            to retain DQI's quantitative properties. See arXiv:2511.20016
            (kernelized DQI) for ongoing work.

    Note:
        Practical scale on a state-vector simulator is approximately
        ``problem.m + problem.n <= 16``.
    """

    def __init__(
        self,
        problem,
        ell,
        *,
        decoder="lut",
        weights=None,
        backend=None,
        noise_model=None,
    ):
        if ell < 1 or ell > problem.m:
            raise_error(
                ValueError,
                f"ell must satisfy 1 <= ell <= m, got ell={ell}, m={problem.m}.",
            )
        self.problem = problem
        self.ell = ell
        if weights is None:
            weights_arr = optimal_weights(problem.m, ell)
        else:
            weights_arr = np.asarray(weights)
            if np.iscomplexobj(weights_arr) and not np.allclose(weights_arr.imag, 0):
                raise_error(
                    ValueError,
                    "weights must be real-valued; got complex array with "
                    "non-zero imaginary part.",
                )
            weights_arr = np.asarray(weights_arr.real, dtype=float)
            if weights_arr.ndim != 1 or weights_arr.shape[0] != ell + 1:
                raise_error(
                    ValueError,
                    f"weights must be a 1-D array of length ell + 1 = {ell + 1}, "
                    f"got shape {weights_arr.shape}.",
                )
            if not np.all(np.isfinite(weights_arr)):
                raise_error(
                    ValueError,
                    "weights must contain only finite values; got NaN or inf.",
                )
        norm = float(np.linalg.norm(weights_arr))
        if norm == 0:
            raise_error(ValueError, "weights must have non-zero norm.")
        # Normalise to match the circuit's auto-normalisation in
        # weighted_dicke_amplitudes; this keeps decoder_success_probability
        # in [0, 1] regardless of how the caller scales the input.
        self.weights = weights_arr / norm
        if isinstance(decoder, SyndromeDecoder):
            validate_decoder_compatibility(decoder, problem, ell)
            self.decoder = decoder
        else:
            self.decoder = get_decoder(decoder, problem, ell)
        self.backend = _check_backend(backend)
        self.noise_model = noise_model

    @property
    def is_dqi_exact(self):
        """``True`` iff the in-circuit decoder uniquely recovers every
        weight-:math:`\\le \\ell` :math:`S` from its syndrome.

        Equivalent to the decoder's :attr:`is_exact`. When ``False``, the
        post-selected samples follow a chosen-S-restricted distribution,
        not the canonical DQI :math:`|P(f)|^2`.
        """
        return getattr(self.decoder, "is_exact", None)

    @property
    def decoder_success_probability(self):
        r"""Analytic probability that the in-circuit decoder succeeds
        (i.e. that a shot will pass error-register post-selection).

        Computed as :math:`\sum_{(y, S) \in \mathrm{LUT}} w_{|S|}^2 /
        \binom{m}{|S|}`. Equals 1 iff :attr:`is_dqi_exact`. The empirical
        ``postselect_success_rate`` from a real run should match this to
        within sampling noise.
        """
        m = self.problem.m
        total = 0.0
        for _, S in self.decoder.decode_table():
            k = int(S.sum())
            total += float(self.weights[k]) ** 2 / comb(m, k)
        return total

    def _build_circuit(self):
        return dqi_circuit(
            self.problem,
            self.ell,
            weights=self.weights,
            decoder=self.decoder,
            include_measurements=True,
        )

    def sample(self, nshots):
        """Run the DQI circuit, post-select on the error register, and
        return solution-register samples.

        The DQI procedure (paper §8.1.2 / Fig. 4) leaves the m-qubit error
        register in :math:`\\ket{0^m}` on branches where the in-circuit
        decoder succeeds; on failure branches the error register is in a
        non-zero state that contaminates the solution-register marginal.
        We post-select shots on ``error register == 0`` to recover the
        clean (modified-)DQI distribution.

        Args:
            nshots (int): Number of measurement shots.

        Returns:
            dict: A dictionary with keys

            - ``"candidates"``: ``(nkept, n)`` ``np.uint8`` array of
              post-selected ``x`` bitstrings (one row per kept shot).
              Empty if every shot's error register read non-zero.
            - ``"raw_samples"``: ``(nshots, m + n)`` ``np.uint8`` array of
              all shot outcomes (error register first, then solution
              register), retained for diagnostics.
            - ``"postselect_success_rate"``: float in :math:`[0, 1]`,
              empirical fraction of shots whose error register read
              all-zeros.
            - ``"decoder_success_probability"``: float in :math:`[0, 1]`,
              analytic post-selection success probability. Equal to 1 iff
              the LUT has no collisions on the Dicke support.
            - ``"is_dqi_exact"``: bool, ``True`` iff
              ``decoder_success_probability == 1`` (no LUT collisions).
              Marks whether the post-selected samples follow the
              canonical DQI :math:`|P(f)|^2` distribution or a
              chosen-S-restricted variant.
            - ``"shots"``: ``nshots`` (echo back).
            - ``"shots_kept"``: number of post-selected shots.
        """
        if nshots <= 0:
            raise_error(ValueError, f"nshots must be positive, got {nshots}.")
        circuit = self._build_circuit()
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)
        result = self.backend.execute_circuit(circuit, nshots=nshots)
        raw_samples = np.asarray(result.samples(binary=True), dtype=np.uint8)
        m, _ = self.problem.m, self.problem.n
        error_bits = raw_samples[:, :m]
        solution_bits = raw_samples[:, m:]
        passed = error_bits.sum(axis=1) == 0
        candidates = solution_bits[passed]
        return {
            "candidates": candidates,
            "raw_samples": raw_samples,
            "postselect_success_rate": float(passed.mean()),
            "decoder_success_probability": self.decoder_success_probability,
            "is_dqi_exact": self.is_dqi_exact,
            "shots": nshots,
            "shots_kept": int(passed.sum()),
        }

    def solve(self, nshots):
        """Run :meth:`sample` and return the best post-selected candidate.

        Args:
            nshots (int): Number of measurement shots.

        Returns:
            dict: A dictionary with keys

            - ``"best_x"``: the candidate ``x`` (as ``np.uint8`` array)
              maximising :meth:`MaxXORSAT.evaluate`. ``None`` if every
              shot failed post-selection.
            - ``"best_value"``: the corresponding number of satisfied
              constraints, or ``-1`` if no shot passed post-selection.
            - ``"postselect_success_rate"``: float.
            - ``"decoder_success_probability"``: float.
            - ``"is_dqi_exact"``: bool.
            - ``"shots"``: ``nshots``.
            - ``"shots_kept"``: number of post-selected shots.
        """
        out = self.sample(nshots)
        candidates = out["candidates"]
        best_x = None
        best_value = -1
        for x in candidates:
            value = self.problem.evaluate(x)
            if value > best_value:
                best_value = value
                best_x = x
        return {
            "best_x": best_x,
            "best_value": best_value,
            "postselect_success_rate": out["postselect_success_rate"],
            "decoder_success_probability": out["decoder_success_probability"],
            "is_dqi_exact": out["is_dqi_exact"],
            "shots": nshots,
            "shots_kept": out["shots_kept"],
        }
