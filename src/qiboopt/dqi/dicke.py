"""
Weighted Dicke-state preparation for Decoded Quantum Interferometry.

For Phase 1 we use brute-force amplitude encoding via :class:`qibo.gates.Unitary`:
the :math:`2^m`-dimensional amplitude vector of
:math:`\\sum_{k=0}^{\\ell} w_k \\ket{D^m_k}` is computed explicitly, then a
unitary whose first column equals that vector is applied to :math:`\\ket{0^m}`.
This materialises a dense :math:`2^m \\times 2^m` matrix, so memory grows as
:math:`O(4^m)`. The implementation is intentionally simple — it matches the
construction used by the PennyLane DQI demo — and is sufficient for the v1
practical ceiling of :math:`m \\le 10` once combined with the in-circuit
decoder. We hard-cap :math:`m` at :data:`_MAX_DICKE_M` to fail loudly rather
than thrash on out-of-memory matrices.

For larger :math:`m` a count-register + Bartschi-Eidenbenz construction is
the natural successor (deferred to a future phase).
"""

import math

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error

#: Maximum number of error qubits supported by the brute-force amplitude
#: encoder. Beyond this the dense :math:`2^m \times 2^m` unitary is too large
#: for typical developer / CI machines.
_MAX_DICKE_M = 12


def weighted_dicke_amplitudes(m, ell, w):
    r"""Return the :math:`2^m`-dimensional amplitude vector of

    .. math::

        \ket{\psi} = \sum_{k=0}^{\ell} w_k \ket{D^m_k}
                  = \sum_{k=0}^{\ell} \frac{w_k}{\sqrt{\binom{m}{k}}}
                    \sum_{|S| = k} \ket{S}.

    Basis ordering follows Qibo's convention: the integer index of a
    computational-basis state encodes the bitstring with qubit ``0`` as the
    most significant bit.

    Args:
        m (int): Number of qubits in the Dicke register.
        ell (int): Hamming-weight cutoff. Must satisfy ``1 <= ell <= m``.
        w (np.ndarray): Weights of length ``ell + 1``. Need not be unit-normalised
            on input; the returned vector is normalised to unit :math:`\\ell^2`
            norm.

    Returns:
        np.ndarray: Complex amplitude vector of length ``2 ** m``.
    """
    if len(w) != ell + 1:
        raise_error(
            ValueError,
            f"w must have length ell + 1 = {ell + 1}, got {len(w)}.",
        )
    if ell < 1 or ell > m:
        raise_error(
            ValueError, f"ell must satisfy 1 <= ell <= m, got ell={ell}, m={m}."
        )
    if m > _MAX_DICKE_M:
        raise_error(
            ValueError,
            f"Brute-force Dicke prep is capped at m <= {_MAX_DICKE_M} (got m={m}). "
            "A count-register + Bartschi-Eidenbenz construction is the planned "
            "successor for larger m.",
        )

    amps = np.zeros(2**m, dtype=complex)
    for state in range(2**m):
        k = bin(state).count("1")
        if k <= ell:
            amps[state] = w[k] / math.sqrt(math.comb(m, k))
    norm = np.linalg.norm(amps)
    if norm == 0:
        raise_error(ValueError, "All weights are zero; cannot prepare a state.")
    return amps / norm


def _unitary_with_first_column(v):
    """Return a unitary matrix whose first column equals the unit vector ``v``.

    Implemented by completing ``v`` to an orthonormal basis via a stable
    Householder-augmented QR factorisation.
    """
    d = v.shape[0]
    M = np.eye(d, dtype=complex)
    M[:, 0] = v
    Q, R = np.linalg.qr(M)
    # Fix sign so that Q[:, 0] equals v (np.linalg.qr is sign-ambiguous).
    phase = R[0, 0] / abs(R[0, 0]) if abs(R[0, 0]) > 0 else 1.0
    Q = Q * np.conj(phase)
    return Q


def dicke_circuit(m, ell, w):
    r"""Build a Qibo circuit that prepares
    :math:`\sum_{k=0}^{\ell} w_k \ket{D^m_k}` on ``m`` qubits.

    Args:
        m (int): Number of qubits in the Dicke register.
        ell (int): Hamming-weight cutoff.
        w (np.ndarray): Weights of length ``ell + 1``.

    Returns:
        :class:`qibo.models.Circuit`: A circuit on ``m`` qubits that prepares
        the weighted Dicke superposition starting from :math:`\ket{0^m}`.
    """
    amps = weighted_dicke_amplitudes(m, ell, w)
    U = _unitary_with_first_column(amps)
    circuit = Circuit(m)
    circuit.add(gates.Unitary(U, *range(m)))
    return circuit
