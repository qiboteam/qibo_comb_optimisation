.. _dqi:

Decoded Quantum Interferometry
------------------------------

This module implements **Decoded Quantum Interferometry (DQI)**, the
non-variational quantum optimisation algorithm of Jordan, Shutty, Wootters,
Zalcman, Schmidhuber, King, Isakov, Khattar, Babbush,
*Optimization by Decoded Quantum Interferometry*,
`arXiv:2408.08292 v5 <https://arxiv.org/abs/2408.08292>`_,
*Nature* **646**:831-836, 2025.

DQI takes a max-LIN over GF(2) instance — equivalently a max-XOR-SAT
instance — given by a parity-check matrix :math:`B \in \mathbb{F}_2^{m
\times n}` and a target :math:`s \in \{0, 1\}^m`, and produces samples
:math:`x \in \{0, 1\}^n` concentrated on near-optimal solutions of

.. math::

    \max_{x \in \{0, 1\}^n} \#\{i : v_i \cdot x = s_i \pmod 2\},

where :math:`v_i` is the :math:`i`-th row of :math:`B`.

**Algorithm.** Following the construction of arXiv:2408.08292 §8.1.2, the
DQI circuit acts on :math:`m + n` qubits — an :math:`m`-qubit *error
register* and an :math:`n`-qubit *solution register*. The procedure is:

1. Prepare the weighted Dicke superposition
   :math:`\sum_{k=0}^{\ell} w_k \ket{D^m_k}` on the error register, where
   :math:`w` is the principal eigenvector of the tridiagonal matrix
   :math:`A^{(m, \ell, 0)}` of paper Eq. 70.
2. Apply :math:`Z^{s_i}` to error qubit :math:`i` for each :math:`s_i = 1`,
   where :math:`s` is the right-hand-side target vector.
3. Reversibly compute :math:`B^\top y` into the solution register via a
   CNOT network: for every :math:`(i, j)` with :math:`B_{ij} = 1`, apply
   ``CNOT(error_i, solution_j)``.
4. Apply the **in-circuit syndrome decoder**: for each ``(y, S)`` pair
   from the chosen :class:`SyndromeDecoder`, multi-controlled-X on the
   error qubits in :math:`S`, controlled on the solution register
   holding :math:`y`. This uncomputes the error register on
   decoder-success branches.
5. Apply :math:`H^{\otimes n}` on the solution register.
6. Measure both registers; **post-select** shots whose error register
   reads :math:`\ket{0^m}` (paper Fig. 4 caption: "postselect on
   :math:`\ket{0}`"). The post-selected solution-register samples follow
   the canonical DQI :math:`|P(f)|^2` distribution **iff** the in-circuit
   decoder is exact (no LUT collisions on the Dicke support); see
   :attr:`DQISolver.is_dqi_exact` and
   :attr:`DQISolver.decoder_success_probability` for the exactness
   diagnostics.

**Where DQI helps.** DQI gives a provable polynomial speedup on the Optimal
Polynomial Intersection (OPI) family (Section 6 of arXiv:2408.08292) and
heuristic advantage on sparse/LDPC max-LIN. For worst-case generic dense
max-LINSAT, arXiv:2603.04540 gives tight inapproximability results, so this
implementation should not be presented as a generic dense max-LIN speedup.

**Phase 1 scope.** Currently implemented: :class:`MaxXORSAT` problem class,
optimal weight vector, weighted Dicke-state preparation, DQI circuit, the
brute-force lookup-table decoder, and end-to-end :class:`DQISolver`.

**Phase 1 limitations.**

- Practical only for ``m + n <= 16`` on a state-vector simulator.
- The brute-force amplitude-encoded Dicke prep materialises a dense
  :math:`2^m \times 2^m` unitary and is hard-capped at ``m = 12``.
  Replacing this with a count-register + Bartschi-Eidenbenz construction
  is a future phase.
- The LUT decoder is exact only when every weight-:math:`\le \ell` error
  pattern :math:`S` has a distinct syndrome
  (:attr:`LUTDecoder.is_exact`). When the LUT has collisions, the
  post-selected samples follow a chosen-S-restricted distribution
  rather than the canonical :math:`|P(f)|^2`; the analytic post-selection
  success probability is reported via
  :attr:`DQISolver.decoder_success_probability`.
- Noise robustness is open research; ``DQISolver`` accepts a Qibo noise
  model but does not guarantee results retain DQI's quantitative
  properties.

MaxXORSAT
^^^^^^^^^

.. autoclass:: qiboopt.dqi.max_xorsat.MaxXORSAT
    :members:
    :member-order: bysource

Optimal weights
^^^^^^^^^^^^^^^

.. autofunction:: qiboopt.dqi.weights.optimal_weights

Weighted Dicke preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qiboopt.dqi.dicke.weighted_dicke_amplitudes

.. autofunction:: qiboopt.dqi.dicke.dicke_circuit

DQI circuit
^^^^^^^^^^^

.. autofunction:: qiboopt.dqi.circuit.dqi_circuit

Decoders
^^^^^^^^

.. autoclass:: qiboopt.dqi.decoders.base.SyndromeDecoder
    :members:

.. autoclass:: qiboopt.dqi.decoders.lut.LUTDecoder
    :members:

.. autofunction:: qiboopt.dqi.decoders.get_decoder

Solver
^^^^^^

.. autoclass:: qiboopt.dqi.solver.DQISolver
    :members:
    :member-order: bysource
