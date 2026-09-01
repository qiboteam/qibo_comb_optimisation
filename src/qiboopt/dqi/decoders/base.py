"""
Abstract base class for DQI in-circuit syndrome decoders.

A decoder produces a *table* of (syndrome, error) pairs that the
:func:`qiboopt.dqi.circuit.dqi_circuit` builder turns into a sequence of
mixed-control multi-controlled-X gates. The decoder is therefore a
classical *circuit-construction helper*, not a runtime mapping.

For a max-XOR-SAT instance with parity-check matrix :math:`B \\in
\\mathbb{F}_2^{m \\times n}`, the table maps each :math:`y \\in \\{0, 1\\}^n`
to a low-weight :math:`S \\in \\{0, 1\\}^m` satisfying :math:`B^\\top S =
y \\pmod 2` and :math:`|S| \\le \\ell`. The DQI circuit uses each entry to
uncompute the m-qubit error register conditional on the n-qubit solution
register holding ``y``.
"""

from abc import ABC, abstractmethod

import numpy as np
from qibo.config import raise_error


def validate_decoder_compatibility(decoder, problem, ell):
    """Ensure ``decoder`` was built for the same ``(problem, ell)`` pair.

    A pre-built decoder bound to a different problem or cutoff produces a
    silently-incorrect DQI circuit (its decode table addresses the wrong
    parity-check matrix). This helper guards :class:`DQISolver` and
    :func:`qiboopt.dqi.circuit.dqi_circuit` against that footgun.

    Compatibility holds iff:

    - ``decoder.ell == ell``;
    - ``decoder.problem is problem`` (identity match), **or**
    - ``decoder.problem.B`` and ``decoder.problem.s`` are element-wise
      equal to ``problem.B`` and ``problem.s`` (structural match).

    Args:
        decoder (:class:`SyndromeDecoder`): The pre-built decoder.
        problem: The :class:`MaxXORSAT` instance the solver / circuit is
            being built for.
        ell (int): The Hamming-weight cutoff being requested.

    Raises:
        ValueError: If ``decoder`` was built for a different ``(problem,
        ell)`` pair.
    """
    if decoder.ell != ell:
        raise_error(
            ValueError,
            f"Pre-built decoder ell={decoder.ell} does not match requested "
            f"ell={ell}.",
        )
    if decoder.problem is problem:
        return
    if not (
        decoder.problem.B.shape == problem.B.shape
        and decoder.problem.s.shape == problem.s.shape
        and np.array_equal(decoder.problem.B, problem.B)
        and np.array_equal(decoder.problem.s, problem.s)
    ):
        raise_error(
            ValueError,
            "Pre-built decoder was constructed for a different MaxXORSAT "
            "instance (B or s differ).",
        )


class SyndromeDecoder(ABC):
    """Interface for in-circuit DQI decoders.

    A decoder is bound to a specific :class:`MaxXORSAT` instance and a
    Hamming-weight cutoff :math:`\\ell`.

    Subclasses must implement :meth:`decode_table`, which returns the list
    of ``(y, S)`` pairs the in-circuit decoder will apply.
    """

    def __init__(self, problem, ell):
        self.problem = problem
        self.ell = ell

    @abstractmethod
    def decode_table(self):
        """Return the in-circuit decode table.

        Yields:
            tuple[np.ndarray, np.ndarray]: ``(y, S)`` pairs where
            ``y`` is a length-``n`` bitstring (``np.uint8``) and ``S`` is a
            length-``m`` bitstring (``np.uint8``) with :math:`|S| \\le \\ell`
            and :math:`B^\\top S = y \\pmod 2`. At most one ``S`` per
            ``y``; the chosen ``S`` should be the smallest-weight (lex
            tiebreak).
        """
