"""
Optimal interference weights for Decoded Quantum Interferometry.
"""

import numpy as np
from qibo.config import raise_error
from scipy.linalg import eigh_tridiagonal


def optimal_weights(m, ell):
    r"""Compute the optimal DQI interference weights :math:`w_0, \ldots, w_\ell`.

    These are the amplitudes used in the weighted-Dicke superposition
    :math:`\sum_{k=0}^{\ell} w_k \ket{D^m_k}` that prepares the DQI state
    achieving the maximum expected energy ratio at cutoff :math:`\ell` on a
    max-LIN problem with :math:`m` constraints. Following Theorem 3.4 of
    Jordan et al. (`arXiv:2408.08292 v5
    <https://arxiv.org/abs/2408.08292>`_, *Nature* **646**:831-836, 2025),
    the optimum is the top eigenvector of the symmetric tridiagonal matrix
    :math:`A` with entries

    .. math::

        A_{k, k+1} = A_{k+1, k} = \sqrt{(k + 1)(m - k)},
        \quad k = 0, \ldots, \ell - 1,

    and zeros on the diagonal.

    The returned vector is unit-normalised and sign-fixed so that
    :math:`w_0 > 0`.

    Args:
        m (int): Number of constraints in the max-LIN instance.
        ell (int): Hamming-weight cutoff of the DQI superposition. Must satisfy
            ``0 < ell <= m``.

    Returns:
        np.ndarray: Vector of length ``ell + 1`` containing the optimal
        weights.

    Example:
        .. testcode::

            from qiboopt.dqi.weights import optimal_weights
            import numpy as np

            w = optimal_weights(m=4, ell=2)
            print(np.round(w, 4))

        .. testoutput::

            [0.4472 0.7071 0.5477]
    """
    if not isinstance(m, int) or not isinstance(ell, int):
        raise_error(TypeError, "m and ell must be integers.")
    if m <= 0:
        raise_error(ValueError, f"m must be positive, got {m}.")
    if ell <= 0 or ell > m:
        raise_error(ValueError, f"ell must satisfy 0 < ell <= m, got ell={ell}, m={m}.")

    off = np.array(
        [np.sqrt((k + 1) * (m - k)) for k in range(ell)],
        dtype=float,
    )
    diag = np.zeros(ell + 1, dtype=float)
    eigvals, eigvecs = eigh_tridiagonal(diag, off)
    w = eigvecs[:, -1]
    if w[0] < 0:
        w = -w
    return w
