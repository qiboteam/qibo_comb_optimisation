"""
Decoded Quantum Interferometry (DQI) for max-LIN over GF(2).

Implements the non-variational quantum optimisation algorithm of Jordan,
Shutty, Wootters, Zalcman, Schmidhuber, King, Isakov, Khattar, Babbush,
*Optimization by Decoded Quantum Interferometry*, `arXiv:2408.08292 v5
<https://arxiv.org/abs/2408.08292>`_, *Nature* **646**:831-836, 2025.

Phase 1 scope:

- :class:`MaxXORSAT` problem class (parity-check matrix + target).
- Optimal weight vector :func:`optimal_weights` from the tridiagonal
  eigenvalue problem of Theorem 3.4.
- :func:`dqi_circuit` builder on ``m + n`` qubits (error + solution
  registers, with the syndrome decoder compiled into the circuit).
- :class:`LUTDecoder` brute-force lookup-table decoder.
- :class:`DQISolver` end-to-end solver, with error-register
  post-selection and ``is_dqi_exact`` /
  ``decoder_success_probability`` diagnostics.

Where DQI helps:

- Provable polynomial speedup on the Optimal Polynomial Intersection (OPI)
  family (Section 6 of arXiv:2408.08292).
- Heuristic advantage on sparse / LDPC-style max-LIN.
- Worst-case generic dense max-LINSAT has tight inapproximability results
  (arXiv:2603.04540), so this implementation should not be presented as a
  generic dense max-LIN speedup.

Phase 1 limitations:

- Practical only for ``m + n <= 16`` on a state-vector simulator (the DQI
  circuit acts on the joint error + solution register).
- The brute-force Dicke prep materialises a dense ``2**m`` x ``2**m``
  unitary, capping ``m`` at 12; the decoder enumerates all ``S`` of weight
  ``<= ell``, capping the number of error patterns at :math:`10^6`.
- The in-circuit LUT decoder is exact only when every weight-:math:`\\le \\ell`
  error pattern :math:`S` has a distinct syndrome (no collisions on the
  Dicke support). When it is not, post-selected samples follow a
  chosen-S-restricted distribution, **not** the canonical
  :math:`|P(f)|^2`. ``DQISolver`` exposes ``is_dqi_exact`` and
  ``decoder_success_probability`` to make this observable.
- No QUBO ``solve_dqi`` entry point (most QUBOs are not faithfully
  expressible as max-LIN).
"""

from qiboopt.dqi.circuit import dqi_circuit
from qiboopt.dqi.decoders import LUTDecoder, SyndromeDecoder, get_decoder
from qiboopt.dqi.dicke import dicke_circuit, weighted_dicke_amplitudes
from qiboopt.dqi.max_xorsat import MaxXORSAT
from qiboopt.dqi.solver import DQISolver
from qiboopt.dqi.weights import optimal_weights

__all__ = [
    "MaxXORSAT",
    "optimal_weights",
    "dicke_circuit",
    "weighted_dicke_amplitudes",
    "dqi_circuit",
    "DQISolver",
    "SyndromeDecoder",
    "LUTDecoder",
    "get_decoder",
]
