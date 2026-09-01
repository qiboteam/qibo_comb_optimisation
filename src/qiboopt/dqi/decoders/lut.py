"""
Brute-force lookup-table syndrome decoder for the in-circuit DQI decoder.

Phase-1 decoder. For a max-XOR-SAT instance with parity-check matrix
:math:`B \\in \\mathbb{F}_2^{m \\times n}` and Hamming-weight cutoff
:math:`\\ell`, the decoder enumerates every :math:`S \\in \\{0, 1\\}^m` with
:math:`|S| \\le \\ell`, computes the syndrome :math:`y = B^\\top S \\pmod 2`,
and stores the inverse map :math:`y \\mapsto S`. When several :math:`S`
yield the same :math:`y` (a *collision*), the entry with the smallest
:math:`|S|` wins (ties broken by the lexicographic order of :math:`S`).

The total table size is at most :math:`\\sum_{k = 0}^{\\ell} \\binom{m}{k}`
entries; the decoder is independent of :math:`n`.

**Decoder exactness.** The DQI procedure (paper §8.1.2 / Eqs. 33–34)
requires that the in-circuit decoder uniquely recover the original
:math:`S` from :math:`B^\\top S` for every :math:`|S| \\le \\ell`. This
holds iff the LUT has no collisions, equivalently iff the number of
chosen LUT entries equals :math:`\\sum_{k=0}^{\\ell} \\binom{m}{k}`. We
expose this as :attr:`is_exact`. When ``is_exact`` is ``False``, the
post-selected output of the DQI circuit follows a *modified*
distribution restricted to the LUT-chosen :math:`S` patterns and is
**not** the canonical DQI :math:`|P(f)|^2` distribution.
"""

import itertools
from math import comb

import numpy as np
from qibo.config import raise_error

from qiboopt.dqi.decoders.base import SyndromeDecoder

_TABLE_SIZE_LIMIT = 10**6


class LUTDecoder(SyndromeDecoder):
    """Brute-force lookup-table decoder.

    Args:
        problem (:class:`qiboopt.dqi.max_xorsat.MaxXORSAT`): The max-XOR-SAT
            instance.
        ell (int): Hamming-weight cutoff used at circuit construction time.

    Attributes:
        is_exact (bool): True iff every weight-:math:`\\le \\ell` error
            pattern :math:`S` has a distinct syndrome :math:`B^\\top S`,
            i.e. the LUT has no collisions on the Dicke support. The
            dual-distance condition for this is :math:`2\\ell < d^\\perp`
            (equivalently :math:`2\\ell + 1 \\le d^\\perp` for integer
            distances). When ``is_exact`` is ``True``, the in-circuit decoder
            is the unitary required by the canonical DQI procedure and the
            post-selected solution-register samples follow :math:`|P(f)|^2`.

    Raises:
        ValueError: If the projected number of error patterns exceeds
            an internal safety limit (currently :math:`10^6`).
    """

    def __init__(self, problem, ell):
        super().__init__(problem, ell)
        m = problem.m
        if ell < 0 or ell > m:
            raise_error(
                ValueError,
                f"ell must satisfy 0 <= ell <= m, got ell={ell}, m={m}.",
            )
        n_low_weight = sum(comb(m, k) for k in range(ell + 1))
        if n_low_weight > _TABLE_SIZE_LIMIT:
            raise_error(
                ValueError,
                f"LUT enumerates {n_low_weight} error patterns, exceeding limit "
                f"{_TABLE_SIZE_LIMIT}; use a different decoder for m={m}, ell={ell}.",
            )
        self._table = self._build_table()
        self._n_low_weight = n_low_weight
        self.is_exact = len(self._table) == n_low_weight

    def _build_table(self):
        """Construct the y -> S map keyed by the packed integer of y.

        Smaller |S| wins per y; ties broken by lexicographic (numpy default)
        order on the packed S.
        """
        problem = self.problem
        ell = self.ell
        n = problem.n
        m = problem.m
        B = problem.B.astype(np.uint8)

        # y -> (weight, S_packed_int, S_array, y_array)
        table = {}
        for k in range(ell + 1):
            for support in itertools.combinations(range(m), k):
                S = np.zeros(m, dtype=np.uint8)
                for i in support:
                    S[i] = 1
                y = (B.T @ S) % 2  # length-n syndrome
                y_key = self._pack(y, n)
                S_int = self._pack(S, m)
                prev = table.get(y_key)
                if prev is None:
                    table[y_key] = (k, S_int, S, y)
                elif (k, S_int) < (prev[0], prev[1]):
                    table[y_key] = (k, S_int, S, y)
        return table

    @staticmethod
    def _pack(bits, length):
        """Pack a bit array (qubit 0 most-significant) into a Python int."""
        out = 0
        for i in range(length):
            out = (out << 1) | int(bits[i])
        return out

    def decode_table(self):
        """Return the list of ``(y, S)`` pairs for the in-circuit decoder.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: One pair per covered
            syndrome ``y``. Both ``y`` and ``S`` are ``np.uint8`` arrays.
        """
        return [(entry[3].copy(), entry[2].copy()) for entry in self._table.values()]
