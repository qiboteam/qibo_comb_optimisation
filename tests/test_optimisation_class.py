import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.models import QAOA
from qibo.optimizers import optimize as optimize
from qibo.quantum_info import infidelity

from qibo_comb_optimisation.optimisation_class.optimisation_class import (
    QUBO,
    linear_problem,
)


# Test initialization of the QUBO class
def test_initialization():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)

    assert qp.Qdict == Qdict
    assert qp.offset == 0.0
    assert qp.n == 2  # Maximum variable index in Qdict keys


def test_multiply_scalar():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)
    qp.multiply_scalar(2)
    assert qp.Qdict == {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}


def test_add():
    Qdict1 = {(0, 0): 1.0, (0, 1): 0.5}
    Qdict2 = {(0, 0): -1.0, (1, 1): 2.0}
    qp1 = QUBO(0, Qdict1)
    qp2 = QUBO(0, Qdict2)
    qp1 + qp2
    assert qp1.Qdict == {(0, 0): 0.0, (0, 1): 0.5, (1, 1): 2.0}


@pytest.mark.parametrize(
    "h, J",
    [
        (
            {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0},
            {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0},
        ),
        ({(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}, {3: 1.0, 4: 0.82, 5: 0.23}),
        (15, 13),
    ],
)
def test_invalid_input_qubo(h, J):
    with pytest.raises(TypeError):
        qp = QUBO(0, h, J)


def test_qubo_to_ising():
    Qdict = {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}
    qp = QUBO(0, Qdict)

    h, J, constant = qp.qubo_to_ising()

    assert h == {0: 1.25, 1: -0.75}
    assert J == {(0, 1): 0.25}
    assert constant == 0.25


def test_evaluate_f():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)

    x = [1, 1]
    f_value = qp.evaluate_f(x)

    assert f_value == 0.5


def test_evaluate_grad_f():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)

    x = [1, 1]
    grad = qp.evaluate_grad_f(x)
    assert np.array_equal(grad, [1.5, -0.5])


def test_tabu_search():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)

    best_solution, best_obj_value = qp.tabu_search(max_iterations=50, tabu_size=5)

    assert len(best_solution) == 2
    assert isinstance(best_obj_value, float)


def test_brute_force():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)

    opt_vector, min_value = qp.brute_force()

    assert len(opt_vector) == 2
    assert isinstance(min_value, float)


def test_initialization_with_h_and_J():
    # Define example h and J for the Ising model
    h = {0: 1.0, 1: -1.5}
    J = {(0, 1): 0.5}
    offset = 2.0

    # Initialize QUBO instance with Ising h and J
    qubo_instance = QUBO(offset, h, J)
    expected_Qdict = {(0, 0): 0.0, (1, 1): 0.0}
    assert (
        qubo_instance.Qdict == expected_Qdict
    ), "Qdict should be created based on h and J conversion"

    # Check that `n` was set correctly (it should be the max variable index + 1)
    assert qubo_instance.n == 2, "n should be the number of variables (max index + 1)"


def test_offset_calculation():
    # Define example h and J for offset calculation
    h = {0: 1.0, 1: -1.5}
    J = {(0, 1): 0.5}
    offset = 2.0

    # Initialize QUBO instance with Ising h and J
    qubo_instance = QUBO(offset, h, J)

    # Expected offset after adjustment: offset + sum(J) - sum(h)
    expected_offset = offset + sum(J.values()) - sum(h.values())

    # Verify the offset value
    assert (
        qubo_instance.offset == expected_offset
    ), "Offset should be adjusted based on sum of h and J values"


def test_isolated_terms_in_h_and_J():
    # Case with no interactions (only diagonal terms in h)
    h = {0: 1.5, 1: -2.0, 2: 0.5}
    J = {}
    offset = 1.0

    qubo_instance = QUBO(offset, h, J)
    print(qubo_instance.Qdict)
    print("check above")
    # Expected Qdict should only contain diagonal terms based on h
    expected_Qdict = {(0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0}
    assert (
        qubo_instance.Qdict == expected_Qdict
    ), "Qdict should reflect only h terms when J is empty"

    # Expected offset should only adjust based on sum of h values since J is empty
    expected_offset = offset - sum(h.values())
    assert (
        qubo_instance.offset == expected_offset
    ), "Offset should adjust only with h values when J is empty"


def test_consistent_terms_in_ham():
    # Run construct_symbolic_Hamiltonian_from_QUBO
    qubo_instance = QUBO(0, {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0})
    ham = qubo_instance.construct_symbolic_Hamiltonian_from_QUBO()

    # Expected terms based on qubo_to_ising output
    h, J, constant = qubo_instance.qubo_to_ising()

    # Extract terms from the symbolic Hamiltonian (converting to string for easier term extraction)
    ham_str = str(ham.form).replace(" ", "")

    # Verify linear terms from h are present
    for i, coeff in h.items():
        term = f"{coeff}*Z{i}"
        assert (
            term in ham_str
        ), f"Expected linear term '{term}' not found in Hamiltonian."

    # Verify quadratic terms from J are present
    for (u, v), coeff in J.items():
        term = f"{coeff}*Z{u}*Z{v}"
        assert (
            term in ham_str
        ), f"Expected quadratic term '{term}' not found in Hamiltonian."


def test_combine_pairs():
    # Populate Qdict with both (i, j) and (j, i) pairs
    qubo_instance = QUBO(0, {(0, 1): 2, (1, 0): 3, (1, 2): 5, (2, 1): -1})
    # Run canonical_q
    result = qubo_instance.canonical_q()

    # Expected outcome after combining pairs
    expected_result = {(0, 1): 5, (1, 2): 4}
    assert (
        result == expected_result
    ), "canonical_q should combine (i, j) and (j, i) pairs"


@pytest.mark.parametrize(
    "gammas, betas, alphas",
    [
        ([0.1, 0.2], [0.3, 0.4], None),
        ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6]),
    ],
)
def test_qubo_to_qaoa_circuit(gammas, betas, alphas):
    h = {0: 1, 1: -1}
    J = {(0, 1): 0.5}
    qubo = QUBO(0, h, J)

    gammas = [0.1, 0.2]
    betas = [0.3, 0.4]
    circuit = qubo.qubo_to_qaoa_circuit(gammas=gammas, betas=betas, alphas=alphas)
    assert isinstance(circuit, Circuit)
    assert circuit.nqubits == qubo.n


@pytest.mark.parametrize(
    "gammas, betas",
    [
        ([0.1, 0.2], [0.3]),
        ([0.1, 0.2], [0.3, 0.4]),
        ([0.1, 0.2], [0.3, 0.4, 0.5]),
    ],
)
def test_qubo_to_qaoa_svp_mixer(gammas, betas):

    def _get_svp_zero_representation(name_to_index):
        """
        :return: a set of indices where it takes values 1, this is to help constructing the mixer
        """
        active_set = set()
        for key in name_to_index:
            if "x" in key or "y" in key:
                active_set.add(name_to_index[key])
        return active_set

    def _create_svp_mixer(name_to_index, beta):
        """
        :param name_to_index: a name to index mapping required to create mixer to preserve probability of 0
        :return: mixer circuit
        """
        n = 0
        for i in name_to_index:
            n += 1
        mixer = Circuit(n)
        active_set = _get_svp_zero_representation(name_to_index)
        for i in range(n):
            if i in active_set:
                mixer.add(gates.X(i))
            mixer.add(gates.RY((i + 1) % n, beta))
            mixer.add(gates.CZ(i, (i + 1) % n))
            if i in active_set:
                mixer.add(gates.X(i))
        return mixer

    numeric_qubo = {
        (0, 4): 4.0,
        (2, 4): 4.0,
        (3, 1): 6.0,
        (1, 1): -3.0,
        (3, 5): 2.0,
        (4, 4): -1.0,
        (3, 3): -3.0,
        (1, 5): 6.0,
        (2, 0): 8.0,
        (5, 5): -3.0,
    }
    offset = 5.0
    name_to_index = {"w[1]": 0, "w[2]": 1, "x_1_0": 2, "x_2_0": 3, "y[1]": 4, "y[2]": 5}

    SVP_mixers = []
    for idx_beta in range(0, len(betas)):
        SVP_mixers.append(_create_svp_mixer(name_to_index, betas[idx_beta]))

    if len(betas) > 2:
        with pytest.raises(ValueError):
            circuit = QUBO(0, numeric_qubo).qubo_to_qaoa_circuit(
                gammas, betas, alphas=None, custom_mixer=SVP_mixers
            )
    else:
        circuit = QUBO(0, numeric_qubo).qubo_to_qaoa_circuit(
            gammas, betas, alphas=None, custom_mixer=SVP_mixers
        )
        assert isinstance(circuit, Circuit)
        assert circuit.nqubits == QUBO(0, numeric_qubo).n


@pytest.mark.parametrize(
    "gammas, betas, alphas",
    [
        ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6]),
        ([0.1, 0.2], [0.3, 0.4], None),
    ],
)
@pytest.mark.parametrize(
    "reg_loss, cvar_delta",
    [
        (True, None),
        (False, 0.1),
    ],
)
def test_train_QAOA(gammas, betas, alphas, reg_loss, cvar_delta):
    h = {0: 1, 1: -1}
    J = {(0, 1): 0.5}
    qubo = QUBO(0, h, J)

    result = qubo.train_QAOA(
        gammas=gammas,
        betas=betas,
        alphas=alphas,
        nshots=10,
        regular_loss=reg_loss,
        cvar_delta=cvar_delta,
    )
    assert isinstance(result[0], float)
    assert isinstance(result[1], np.ndarray)
    assert isinstance(result[3], Circuit)
    assert isinstance(result[4], dict)


@pytest.mark.parametrize(
    "gammas, betas, alphas, reg_loss, cvar_delta",
    [
        ([0.1, 0.2], [0.3, 0.4], None, True, None),
        ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6], False, 0.1),
    ],
)
def test_train_QAOA_svp_mixer(gammas, betas, alphas, reg_loss, cvar_delta):

    def _get_svp_zero_representation(name_to_index):
        """
        :return: a set of indices where it takes values 1, this is to help constructing the mixer
        """
        active_set = set()
        for key in name_to_index:
            if "x" in key or "y" in key:
                active_set.add(name_to_index[key])
        return active_set

    def _create_svp_mixer(name_to_index, beta):
        """
        :param name_to_index: a name to index mapping required to create mixer to preserve probability of 0
        :return: mixer circuit
        """
        n = 0
        for i in name_to_index:
            n += 1
        mixer = Circuit(n)
        active_set = _get_svp_zero_representation(name_to_index)
        for i in range(n):
            if i in active_set:
                mixer.add(gates.X(i))
            mixer.add(gates.RY((i + 1) % n, beta))
            mixer.add(gates.CZ(i, (i + 1) % n))
            if i in active_set:
                mixer.add(gates.X(i))
        return mixer

    numeric_qubo = {
        (0, 4): 4.0,
        (2, 4): 4.0,
        (3, 1): 6.0,
        (1, 1): -3.0,
        (3, 5): 2.0,
        (4, 4): -1.0,
        (3, 3): -3.0,
        (1, 5): 6.0,
        (2, 0): 8.0,
        (5, 5): -3.0,
    }
    offset = 5.0
    name_to_index = {"w[1]": 0, "w[2]": 1, "x_1_0": 2, "x_2_0": 3, "y[1]": 4, "y[2]": 5}

    SVP_mixers = []
    for idx_beta in range(0, len(betas)):
        SVP_mixers.append(_create_svp_mixer(name_to_index, betas[idx_beta]))

    result = QUBO(0, numeric_qubo).train_QAOA(
        gammas=gammas,
        betas=betas,
        alphas=alphas,
        nshots=10,
        regular_loss=reg_loss,
        cvar_delta=cvar_delta,
        custom_mixer=SVP_mixers,
    )
    assert isinstance(result[0], float)
    assert isinstance(result[1], np.ndarray)
    assert isinstance(result[3], Circuit)
    assert isinstance(result[4], dict)


def test_qubo_to_qaoa_object():
    h = {0: 1, 1: -1}
    J = {(0, 1): 0.5}
    qubo = QUBO(0, h, J)

    qaoa = qubo.qubo_to_qaoa_object()
    assert isinstance(qaoa, QAOA)
    assert hasattr(qaoa, "hamiltonian")


def test_qubo_to_qaoa_object_params():
    params = [0.1, 0.2]
    h = {0: 1, 1: -1}
    J = {(0, 1): 0.5}
    qubo = QUBO(0, h, J)

    qaoa = qubo.qubo_to_qaoa_object(params=np.array(params))

    assert isinstance(qaoa, QAOA)
    assert hasattr(qaoa, "hamiltonian")


def test_linear_initialization():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    lp = linear_problem(A, b)
    assert np.array_equal(lp.A, A)
    assert np.array_equal(lp.b, b)
    assert lp.n == 2


def test_linear_multiply_scalar():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    lp = linear_problem(A, b)
    lp.multiply_scalar(2)
    assert np.array_equal(lp.A, np.array([[2, 4], [6, 8]]))
    assert np.array_equal(lp.b, np.array([10, 12]))


def test_linear_addition():
    A1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([5, 6])
    lp1 = linear_problem(A1, b1)
    A2 = np.array([[1, 1], [1, 1]])
    b2 = np.array([1, 1])
    lp2 = linear_problem(A2, b2)
    lp1 + lp2
    assert np.array_equal(lp1.A, np.array([[2, 3], [4, 5]]))
    assert np.array_equal(lp1.b, np.array([6, 7]))


def test_linear_evaluate_f():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    lp = linear_problem(A, b)
    x = np.array([1, 1])
    result = lp.evaluate_f(x)
    assert np.array_equal(result, np.array([8, 13]))


def test_linear_square():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    lp = linear_problem(A, b)
    Quadratic = lp.square()
    Qdict = Quadratic.Qdict
    offset = Quadratic.offset
    expected_Qdict = {(0, 0): 56, (0, 1): 14, (1, 0): 14, (1, 1): 88}
    expected_offset = 61
    assert Qdict == expected_Qdict
    assert offset == expected_offset
