"""
DQI quantum circuit construction.

The DQI circuit acts on :math:`m + n` qubits, organised into two registers:

- **error register**: qubits ``0 .. m - 1``, one per constraint.
- **solution register**: qubits ``m .. m + n - 1``, one per variable.

Stages:

1. Prepare :math:`\\sum_{k=0}^{\\ell} w_k \\ket{D^m_k}` on the error
   register.
2. Apply :math:`Z^{s_i}` to error qubit :math:`i` for each :math:`s_i = 1`.
3. **B-transpose multiplication**: for every :math:`(i, j)` with
   :math:`B_{ij} = 1`, apply ``CNOT(error_i, solution_j)``. This computes
   :math:`\\ket{S}_e \\ket{0}_s \\to \\ket{S}_e \\ket{B^\\top S}_s`.
4. **In-circuit syndrome decoder**: for each ``(y, S)`` pair from the
   chosen :class:`SyndromeDecoder`, apply :math:`X` on every error qubit
   :math:`i` with :math:`S_i = 1`, multi-controlled on the solution
   register equal to :math:`y` (mixed |0>/|1> controls). This uncomputes
   the error register on covered syndromes; on un-covered syndromes the
   error register is left in a non-zero state ("decoder failure"
   branches).
5. Apply :math:`H^{\\otimes n}` on the solution register.
6. Measure the solution register; the bitstring is the candidate
   :math:`x`.

Reference: Jordan et al., *Optimization by Decoded Quantum
Interferometry*, `arXiv:2408.08292 v5 <https://arxiv.org/abs/2408.08292>`_,
*Nature* **646**:831-836, 2025.
"""

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error

from qiboopt.dqi.decoders import get_decoder
from qiboopt.dqi.decoders.base import SyndromeDecoder, validate_decoder_compatibility
from qiboopt.dqi.dicke import _unitary_with_first_column, weighted_dicke_amplitudes
from qiboopt.dqi.weights import optimal_weights


def dqi_circuit(
    problem,
    ell,
    *,
    weights=None,
    decoder="lut",
    include_measurements=True,
):
    r"""Construct the DQI circuit for a :class:`MaxXORSAT` problem.

    Args:
        problem (:class:`qiboopt.dqi.max_xorsat.MaxXORSAT`): The max-XOR-SAT
            instance with parity-check matrix :math:`B` and target :math:`s`.
        ell (int): Hamming-weight cutoff. Must satisfy ``1 <= ell <= problem.m``.
        weights (np.ndarray, optional): Length-``ell + 1`` weight vector. If
            ``None``, computed via :func:`qiboopt.dqi.weights.optimal_weights`.
        decoder (str | :class:`SyndromeDecoder`, optional): Either a registered
            decoder name (currently only ``"lut"``) or a pre-instantiated
            :class:`SyndromeDecoder`. Defaults to ``"lut"``.
        include_measurements (bool, optional): If ``True``, append measurement
            gates on **all** ``m + n`` qubits (both error and solution
            registers). The error register is post-selected on
            :math:`\\ket{0^m}` by :class:`DQISolver`. Defaults to ``True``.

    Returns:
        :class:`qibo.models.Circuit`: The DQI circuit on ``problem.m +
        problem.n`` qubits.

    Note:
        Practical scale on a state-vector simulator is approximately
        ``problem.m + problem.n <= 16``.
    """
    m = problem.m
    n = problem.n
    if not isinstance(ell, int):
        raise_error(TypeError, "ell must be an integer.")
    if ell < 1 or ell > m:
        raise_error(
            ValueError, f"ell must satisfy 1 <= ell <= m, got ell={ell}, m={m}."
        )

    if weights is None:
        weights = optimal_weights(m, ell)
    if len(weights) != ell + 1:
        raise_error(
            ValueError,
            f"weights must have length ell + 1 = {ell + 1}, got {len(weights)}.",
        )

    if isinstance(decoder, SyndromeDecoder):
        validate_decoder_compatibility(decoder, problem, ell)
        decoder_obj = decoder
    else:
        decoder_obj = get_decoder(decoder, problem, ell)

    total_qubits = m + n
    circuit = Circuit(total_qubits)

    # Stage 1: weighted Dicke prep on the error register only.
    amps = weighted_dicke_amplitudes(m, ell, weights)
    U_dicke = _unitary_with_first_column(amps)
    circuit.add(gates.Unitary(U_dicke, *range(m)))

    # Stage 2: phase flips by s on the error register.
    for i in range(m):
        if problem.s[i]:
            circuit.add(gates.Z(i))

    # Stage 3: B-transpose CNOT network: CNOT(error_i, solution_j) for B[i,j]=1.
    B = problem.B
    for i in range(m):
        for j in range(n):
            if B[i, j]:
                circuit.add(gates.CNOT(i, m + j))

    # Stage 4: in-circuit syndrome decoder.
    n_qubits_solution = list(range(m, m + n))
    for y, S in decoder_obj.decode_table():
        if not np.any(S):
            # The all-zeros error pattern leaves the error register at |0>;
            # nothing to uncompute. Skip to avoid emitting useless multi-CX gates.
            continue
        zero_controls = [m + j for j in range(n) if y[j] == 0]
        # X-conjugation pre-flip so |0> controls fire as expected.
        for q in zero_controls:
            circuit.add(gates.X(q))
        for i in range(m):
            if S[i]:
                circuit.add(gates.X(i).controlled_by(*n_qubits_solution))
        # Unflip to restore the solution register.
        for q in zero_controls:
            circuit.add(gates.X(q))

    # Stage 5: Hadamards on the solution register.
    for j in range(n):
        circuit.add(gates.H(m + j))

    # Stage 6: measure both registers. The error register is post-selected
    # on |0> by :class:`DQISolver` to project away decoder-failure branches
    # (paper §8.1.2 / Fig. 4 caption: "postselect on |0>").
    if include_measurements:
        circuit.add(gates.M(*range(m + n)))

    return circuit
