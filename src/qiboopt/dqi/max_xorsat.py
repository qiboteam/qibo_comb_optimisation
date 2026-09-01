"""
Max-LIN over GF(2) / max-XOR-SAT problem class for Decoded Quantum
Interferometry.
"""

import itertools

import numpy as np
from qibo.config import raise_error


def _coerce_binary(arr_like, name):
    """Validate that every entry is exactly ``0`` or ``1`` and return a
    ``uint8`` ndarray.

    The check runs on the raw array *before* any ``uint8`` cast, so
    out-of-range integers (negative or > 1) and non-integer floats are
    rejected rather than silently wrapped modulo 256.
    """
    arr_raw = np.asarray(arr_like)
    if not np.all((arr_raw == 0) | (arr_raw == 1)):
        raise_error(
            ValueError,
            f"{name} entries must be exactly 0 or 1; got values outside that set.",
        )
    return arr_raw.astype(np.uint8)


class MaxXORSAT:
    r"""A max-LIN-over-GF(2) (a.k.a. max-XOR-SAT) optimisation problem.

    Given a parity-check matrix :math:`B \in \mathbb{F}_2^{m \times n}` and a
    target :math:`s \in \{0, 1\}^m`, the goal is to find :math:`x \in
    \{0, 1\}^n` maximising the number of constraints
    :math:`v_i \cdot x = s_i \pmod 2` that are satisfied, where :math:`v_i`
    is the :math:`i`-th row of :math:`B`.

    This is the native input class for Decoded Quantum Interferometry (DQI),
    introduced in Jordan et al., *Optimization by Decoded Quantum
    Interferometry*, `arXiv:2408.08292 v5
    <https://arxiv.org/abs/2408.08292>`_, *Nature* **646**:831-836, 2025.

    Note:
        DQI on a state-vector simulator is practical only for instances with
        ``m + n <= 16`` (the DQI circuit acts on the joint
        :math:`m + n`-qubit register). The brute-force amplitude-encoded
        Dicke prep additionally hard-caps :math:`m` at 12 because it
        materialises a dense :math:`2^m \times 2^m` unitary.

    Args:
        B (np.ndarray): Parity-check matrix of shape ``(m, n)`` with
            ``uint8`` / boolean entries in :math:`\{0, 1\}`.
        s (np.ndarray): Target vector of length ``m`` with entries in
            :math:`\{0, 1\}`.

    Example:
        .. testcode::

            import numpy as np
            from qiboopt.dqi.max_xorsat import MaxXORSAT

            B = np.array([[1, 1, 0],
                          [0, 1, 1],
                          [1, 0, 1]], dtype=np.uint8)
            s = np.array([1, 0, 1], dtype=np.uint8)
            problem = MaxXORSAT(B, s)
            print(problem.n, problem.m)

        .. testoutput::

            3 3

        .. testcode::

            print(problem.evaluate(np.array([1, 0, 0], dtype=np.uint8)))

        .. testoutput::

            3
    """

    def __init__(self, B, s):
        # Shape checks first so we always have a meaningful error message.
        B_raw = np.asarray(B)
        s_raw = np.asarray(s)
        if B_raw.ndim != 2:
            raise_error(ValueError, f"B must be 2-D, got shape {B_raw.shape}.")
        if s_raw.ndim != 1:
            raise_error(ValueError, f"s must be 1-D, got shape {s_raw.shape}.")
        if B_raw.shape[0] != s_raw.shape[0]:
            raise_error(
                ValueError,
                f"B and s must have matching first dimension, got "
                f"{B_raw.shape[0]} != {s_raw.shape[0]}.",
            )
        # Value validation runs on the raw array; out-of-range ints and
        # non-integer floats are rejected before any uint8 cast.
        self.B = _coerce_binary(B_raw, "B")
        self.s = _coerce_binary(s_raw, "s")

    @property
    def m(self):
        """Number of constraints."""
        return self.B.shape[0]

    @property
    def n(self):
        """Number of variables."""
        return self.B.shape[1]

    def evaluate(self, x):
        """Number of constraints satisfied by the candidate ``x``.

        Args:
            x (np.ndarray): Binary vector of length ``n``.

        Returns:
            int: Number of constraints :math:`v_i \\cdot x = s_i \\pmod 2`
            that hold.
        """
        x_raw = np.asarray(x)
        if x_raw.shape != (self.n,):
            raise_error(
                ValueError,
                f"x must have shape ({self.n},), got {x_raw.shape}.",
            )
        x_arr = _coerce_binary(x_raw, "x")
        residual = (self.B @ x_arr + self.s) % 2
        return int(self.m - residual.sum())

    def brute_force(self):
        """Find the optimum by exhaustive enumeration. Use only for ``n <= 20``.

        Returns:
            tuple: A pair ``(x, value)`` where ``x`` is the optimal binary
            vector (as ``np.ndarray``) and ``value`` the number of satisfied
            constraints.
        """
        if self.n > 20:
            raise_error(
                ValueError,
                f"brute_force is only intended for n <= 20, got n = {self.n}.",
            )
        best_x = None
        best_value = -1
        for bits in itertools.product([0, 1], repeat=self.n):
            x = np.array(bits, dtype=np.uint8)
            value = self.evaluate(x)
            if value > best_value:
                best_value = value
                best_x = x
        return best_x, best_value

    def to_qubo(self, scale=1.0):
        """Express the max-XOR-SAT objective as a QUBO.

        Each parity constraint :math:`v_i \\cdot x = s_i` is rewritten as a
        quadratic penalty :math:`(v_i \\cdot x - s_i)^2 \\bmod 2` and then
        linearised. Minimising the resulting QUBO is equivalent to maximising
        the number of satisfied constraints (up to a constant offset).

        Args:
            scale (float): Multiplicative factor applied to all coefficients.

        Returns:
            :class:`qiboopt.opt_class.opt_class.QUBO`: QUBO whose minimiser
            equals the max-XOR-SAT maximiser.
        """
        from qiboopt.opt_class.opt_class import QUBO

        Qdict = {}
        offset = 0.0
        for i in range(self.m):
            row = np.flatnonzero(self.B[i])
            target = int(self.s[i])
            # (sum_{j in row} x_j - target)^2 expanded over reals; under
            # x_j in {0,1} we then reduce to GF(2) by adding -2 for each pair
            # of distinct indices in row that both equal 1, mod 2.
            # The standard trick: penalty for parity constraint is
            #   P_i(x) = sum_{j in row} x_j - 2 * sum_{j<k in row} x_j x_k
            # gives parity(x|row) but only for two-variable rows. For general
            # row weight w we lift via auxiliary booleans; for w<=2 (e.g.
            # MaxCut) the direct expansion below is exact.
            if len(row) == 1:
                j = int(row[0])
                if target == 0:
                    Qdict[(j, j)] = Qdict.get((j, j), 0.0) + scale
                else:
                    Qdict[(j, j)] = Qdict.get((j, j), 0.0) - scale
                    offset += scale
            elif len(row) == 2:
                j, k = int(row[0]), int(row[1])
                # constraint x_j XOR x_k = target
                # XOR expressed as x_j + x_k - 2 x_j x_k
                # we want to penalise (x_j XOR x_k) != target, i.e. cost
                # equals 1 when XOR != target. Reformulating as a linear
                # combination gives the closed form below.
                if target == 0:
                    # penalise XOR=1: cost = x_j + x_k - 2 x_j x_k
                    Qdict[(j, j)] = Qdict.get((j, j), 0.0) + scale
                    Qdict[(k, k)] = Qdict.get((k, k), 0.0) + scale
                    key = (min(j, k), max(j, k))
                    Qdict[key] = Qdict.get(key, 0.0) - 2.0 * scale
                else:
                    # penalise XOR=0: cost = 1 - (x_j + x_k - 2 x_j x_k)
                    offset += scale
                    Qdict[(j, j)] = Qdict.get((j, j), 0.0) - scale
                    Qdict[(k, k)] = Qdict.get((k, k), 0.0) - scale
                    key = (min(j, k), max(j, k))
                    Qdict[key] = Qdict.get(key, 0.0) + 2.0 * scale
            else:
                raise_error(
                    NotImplementedError,
                    f"to_qubo currently supports row weights <= 2; row {i} has weight {len(row)}.",
                )
        return QUBO(offset, Qdict)

    @classmethod
    def random_sparse(cls, n, m, row_weight, *, seed=None):
        """Generate a random sparse max-XOR-SAT instance.

        Each of the ``m`` rows of :math:`B` has exactly ``row_weight``
        non-zero entries placed uniformly at random; ``s`` is uniform on
        :math:`\\{0, 1\\}^m`.

        Args:
            n (int): Number of variables.
            m (int): Number of constraints.
            row_weight (int): Number of non-zero entries in each row of
                :math:`B`. Must satisfy ``1 <= row_weight <= n``.
            seed (int, optional): Seed for ``numpy.random.default_rng``.

        Returns:
            :class:`MaxXORSAT`: Random sparse instance.
        """
        if row_weight < 1 or row_weight > n:
            raise_error(
                ValueError,
                f"row_weight must satisfy 1 <= row_weight <= n, got {row_weight}.",
            )
        rng = np.random.default_rng(seed)
        B = np.zeros((m, n), dtype=np.uint8)
        for i in range(m):
            cols = rng.choice(n, size=row_weight, replace=False)
            B[i, cols] = 1
        s = rng.integers(0, 2, size=m, dtype=np.uint8)
        return cls(B, s)
