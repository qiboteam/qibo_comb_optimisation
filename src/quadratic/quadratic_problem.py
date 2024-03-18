from qibo import hamiltonians
from qibo.symbols import X, Z

def qubo_to_ising(Q, constant = 0.0):
    """Convert a QUBO problem to an Ising problem.

    Map a quadratic unconstrained binary optimization (QUBO) problem  defined over binary variables
    (0 or 1 values), where the linear term is contained along x' Qx
    the diagonal of Q, to an Ising model defined on spins (variables with {-1, +1} values).
    Return h and J that define the Ising model as well as the offset in energy
    between the two problem formulations:

    .. math::

         x'  Q  x  = constant + s'  J  s + h'  s

    Args:
        Q (dict[(variable, variable), coefficient]):
            QUBO coefficients in a dict of form {(u, v): coefficient, ...}, where keys
            are 2-tuples of variables of the model and values are biases
            associated with the pair of variables. Tuples (u, v) represent interactions
            and (v, v) linear biases.
        constant:
            Constant offset to be applied to the energy. Default 0.

    Returns:
        (dict, dict, float): A 3-tuple containing:

            dict: Linear coefficients of the Ising problem.

            dict: Quadratic coefficients of the Ising problem.

            float: New energy offset.

    Examples:
        This example converts a QUBO problem of two variables that have positive
        biases of value 1 and are positively coupled with an interaction of value 1
        to an Ising problem, and shows the new energy offset.


    """
    h = {}
    J = {}
    linear_offset = 0.0
    quadratic_offset = 0.0

    for (u, v), bias in Q.items():
        if u == v:
            if u in h:
                h[u] +=  bias/2
            else:
                h[u] = bias/2
            linear_offset += bias

        else:
            if bias != 0.0:
                J[(u, v)] = bias/4

            if u in h:
                h[u] += bias/4
            else:
                h[u] = bias/4

            if v in h:
                h[v] += bias/4
            else:
                h[v] = bias/4

            quadratic_offset += bias

    constant += .5 * linear_offset + .25 * quadratic_offset

    return h, J, constant


def ising_to_qubo(h, J, offset=0.0):
    """Convert an Ising problem to a QUBO problem.

    Map an Ising model defined on spins (variables with {-1, +1} values) to quadratic
    unconstrained binary optimization (QUBO) formulation :math:`x'  Q  x` defined over
    binary variables (0 or 1 values), where the linear term is contained along the diagonal of Q.
    Return matrix Q that defines the model as well as the offset in energy between the two
    problem formulations:

    .. math::

         s'  J  s + h'  s = offset + x'  Q

    Args:
        h (dict[variable, bias]):
            Linear biases as a dict of the form {v: bias, ...}, where keys are variables of
            the model and values are biases.
        J (dict[(variable, variable), bias]):
           Quadratic biases as a dict of the form {(u, v): bias, ...}, where keys
           are 2-tuples of variables of the model and values are quadratic biases
           associated with the pair of variables (the interaction).
        offset (numeric, optional, default=0):
            Constant offset to be applied to the energy. Default 0.

    Returns:
        (dict, float): A 2-tuple containing:

            dict: QUBO coefficients.

            float: New energy offset.

    Examples:
        This example converts an Ising problem of two variables that have positive
        biases of value 1 and are positively coupled with an interaction of value 1
        to a QUBO problem and prints the resulting energy offset.

    """
    # the linear biases are the easiest
    q = {(v, v): 2. * bias for v, bias in h.items()}

    # next the quadratic biases
    for (u, v), bias in J.items():
        if bias == 0.0:
            continue
        q[(u, v)] = 4. * bias
        q[(u, u)] = q.setdefault((u, u), 0) - 2. * bias
        q[(v, v)] = q.setdefault((v, v), 0) - 2. * bias

    # finally calculate the offset
    offset += sum(J.values()) - sum(h.values())

    return q, offset


def construct_symbolic_Hamiltonian_from_QUBO(Q):
    """

    :param Q: Q, a matrix describing the QUBO
    :return: a symbolic hamiltonian that corresponds to the QUBO
    """
    h, J, constant = qubo_to_ising(Q)
    symbolic_ham = sum(h[i]*Z[i] for i in h)
    symbolic_ham += sum(J[u,v] * Z[u]*Z[v] for (u,v) in J)
    symbolic_ham += constant
    ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
    return ham


