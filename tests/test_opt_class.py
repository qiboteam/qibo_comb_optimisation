import importlib.util
import itertools

import numpy as np
import pytest
from qibo import Circuit, gates, get_backend, set_backend
from qibo.models import QAOA
from qibo.noise import DepolarizingError, NoiseModel
from qibo.optimizers import optimize
from qibo.quantum_info import infidelity

from qiboopt.opt_class.opt_class import (
    QUBO,
    LinearProblem,
    variable_dict_to_ind_dict,
    variable_to_ind,
)


def _qiboml_available():
    if importlib.util.find_spec("qiboml") is None:
        return False
    if importlib.util.find_spec("torch") is None:
        return False
    return True


ENGINES = ["legacy"] + (["qiboml"] if _qiboml_available() else [])


def test_initialization():
    """Test initialization of the QUBO class"""
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)

    assert qp.Qdict == Qdict
    assert qp.offset == 0.0
    assert qp.n == 2  # Maximum variable index in Qdict keys


def test_add_multiplication_operators():
    """Test addition and multiplication operators for QUBO"""
    qp1 = QUBO(0, {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0})
    qp2 = QUBO(1.0, {(0, 0): 2.0, (1, 0): 2.0, (1, 1): -2.0})

    qp3 = 2 * qp1 + qp2 * 0.5
    assert qp3.Qdict == {(0, 0): 3.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): -3.0}
    assert qp3.offset == 0.5

    # Test type error
    with pytest.raises(TypeError):
        qp3 = qp1 * "invalid"


@pytest.mark.parametrize(
    "h, J",
    [
        (
            {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0},
            {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0},
        ),
        ({(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}, {3: 1.0, 4: 0.82, 5: 0.23}),
        (15, 13),
        ({0: 1, 1: 2}, {(0, "x"): 2}),
    ],
)
def test_invalid_input_qubo(h, J):
    """Test invalid initialization of the QUBO class"""
    with pytest.raises(TypeError):
        _qp = QUBO(0, h, J)


def test_invalid_number_arguments_qubo():
    with pytest.raises(
        NotImplementedError, match="Invalid number of args in the QUBO constructor."
    ):
        QUBO(0, {}, {}, {})


def test_qubo_to_ising():
    Qdict = {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}
    qp = QUBO(0, Qdict)

    h, J, constant = qp.qubo_to_ising()

    assert h == {0: -1.25, 1: 0.75}
    assert J == {(0, 1): 0.25}
    assert constant == 0.25


def test_evaluate_f():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)
    x = [1, 1]
    assert qp.evaluate_f(x) == 0.5


def test_evaluate_grad_f():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)
    x = [1, 1]
    assert np.array_equal(qp.evaluate_grad_f(x), [1.5, -0.5])


def test_tabu_search():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)

    best_solution, best_obj_value = qp.tabu_search(max_iterations=50, tabu_size=5)

    assert len(best_solution) == 2
    assert isinstance(best_obj_value, float)


def test_tabu_search_updates_best_solution(monkeypatch):
    qp = QUBO(0, {(0, 0): 1.0, (1, 1): 1.0})
    monkeypatch.setattr(
        "qiboopt.opt_class.opt_class.np.random.randint",
        lambda *args, **kwargs: np.ones(qp.n, dtype=int),
    )
    best_solution, best_obj_value = qp.tabu_search(max_iterations=1, tabu_size=1)
    assert np.array_equal(best_solution, np.array([0, 1]))
    assert best_obj_value == pytest.approx(1.0)


def test_brute_force():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)

    opt_vector, min_value = qp.brute_force()

    assert len(opt_vector) == 2
    assert isinstance(min_value, float)
    assert abs(min_value + 1.0) < 0.001


def test_initialization_with_h_and_J():
    # Define example h and J for the Ising model
    h = {0: 1.0, 1: -1.5}
    J = {(0, 1): 0.5}
    offset = 2.0

    # Initialize QUBO instance with Ising h and J
    qubo_instance = QUBO(offset, h, J)
    expected_Qdict = {(0, 0): -3.0, (1, 1): 2.0, (0, 1): 2.0}
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
    expected_offset = offset + sum(J.values()) + sum(h.values())

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
    # Expected Qdict should only contain diagonal terms based on h
    expected_Qdict = {(0, 0): -3.0, (1, 1): 4.0, (2, 2): -1.0}
    assert (
        qubo_instance.Qdict == expected_Qdict
    ), "Qdict should reflect only h terms when J is empty"

    # Expected offset should only adjust based on sum of h values since J is empty
    expected_offset = offset + sum(h.values())
    assert (
        qubo_instance.offset == expected_offset
    ), "Offset should adjust only with h values when J is empty"


def test_consistent_terms_in_ham():
    # Run construct_symbolic_Hamiltonian_from_QUBO
    qubo_instance = QUBO(0, {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0})
    ham = qubo_instance.construct_symbolic_Hamiltonian_from_QUBO()

    # Expected terms based on qubo_to_ising output
    h, J, _constant = qubo_instance.qubo_to_ising()

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


def test_qubo_to_qaoa_circuit_without_measurements():
    qubo = QUBO(0, {0: 1, 1: -1}, {(0, 1): 0.5})
    circuit = qubo.qubo_to_qaoa_circuit(
        gammas=[0.1, 0.2],
        betas=[0.3, 0.4],
        include_measurements=False,
    )
    assert isinstance(circuit, Circuit)
    assert len(circuit.measurements) == 0


@pytest.mark.parametrize(
    "gammas, betas",
    [
        ([0.1], [0.2]),
        ([0.1, 0.2], [0.3]),
        ([0.1, 0.2], [0.3, 0.4]),
        ([0.1, 0.2], [0.3, 0.4, 0.5]),
    ],
)
def test_qubo_to_qaoa_svp_mixer(gammas, betas):

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

    # SVP_mixers is now a list of functions that take beta and return a circuit
    svp_mixers = [
        lambda beta, idx=idx: create_svp_mixer(name_to_index, beta)
        for idx in range(len(betas))
    ]

    if len(betas) != len(gammas):
        with pytest.raises(ValueError):
            circuit = QUBO(offset, numeric_qubo).qubo_to_qaoa_circuit(
                gammas, betas, alphas=None, custom_mixer=svp_mixers
            )
    else:
        circuit = QUBO(offset, numeric_qubo).qubo_to_qaoa_circuit(
            gammas, betas, alphas=None, custom_mixer=svp_mixers
        )
        assert isinstance(circuit, Circuit)
        assert circuit.nqubits == QUBO(offset, numeric_qubo).n


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
@pytest.mark.parametrize("noise_model", [(True, False)])
@pytest.mark.parametrize("engine", ENGINES)
def test_train_QAOA(gammas, betas, alphas, reg_loss, cvar_delta, noise_model, engine):
    h = {0: 1, 1: -1}
    J = {(0, 1): 0.5}
    qubo = QUBO(0, h, J)
    if noise_model:
        lam = 0.1
        noise_model = NoiseModel()
        noise_model.add(DepolarizingError(lam))

    result = qubo.train_QAOA(
        gammas=gammas,
        betas=betas,
        alphas=alphas,
        nshots=10,
        regular_loss=reg_loss,
        cvar_delta=cvar_delta,
        noise_model=noise_model,
        engine=engine,
        epochs=5,
    )
    assert isinstance(result[0], float)
    assert isinstance(result[1], np.ndarray)
    assert isinstance(result[3], Circuit)
    assert isinstance(result[4], dict)


@pytest.mark.parametrize("engine", ENGINES)
def test_train_QAOA_convex_qubo(engine):

    Qdict = {(0, 0): 2.0, (1, 1): 2.0}

    qp = QUBO(0, Qdict)
    # The minimum is at x = [0, 0], f([0,0]) = 0
    # Use a small number of layers and shots for a fast test
    gammas = [0.1, 0.21, 0.15]
    betas = [0.2, 0.3, 0.15]

    # Train QAOA with 100 iterations. Should be enough to find the minimum.
    best, params, extra, circuit, freqs = qp.train_QAOA(
        gammas, betas, nshots=1000, maxiter=100, engine=engine, epochs=50
    )
    # Convert result keys to bitstrings
    most_freq = max(freqs, key=freqs.get)
    # The bitstring should be '00' (for x0=0, x1=0)
    assert most_freq == "00", f"Expected ground state '00', got {most_freq}"


def test_train_QAOA_edge_cases():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = QUBO(0, Qdict)

    # 1. Neither p nor gammas provided: should raise ValueError
    with pytest.raises(ValueError, match="Either p or gammas must be provided"):
        qp.train_QAOA()

    # 2. gammas provided with wrong length for p: should raise ValueError
    with pytest.raises(ValueError, match="gammas must be of length 2"):
        qp.train_QAOA(gammas=[0.1], betas=[0.2], p=2)

    # 3. Only p provided: should generate random gammas/betas and run
    # Should not raise, just check output types
    result = qp.train_QAOA(p=2)
    assert isinstance(result[0], float)
    assert isinstance(result[1], np.ndarray)
    assert isinstance(result[3], Circuit)
    assert isinstance(result[4], dict)


def test_train_qaoa_unsupported_engine_raises():
    qp = QUBO(0, {(0, 0): 1.0, (1, 1): 1.0})
    with pytest.raises(ValueError, match="Unsupported engine"):
        qp.train_QAOA(gammas=[0.1], betas=[0.2], engine="invalid")


def test_train_qaoa_qiboml_cvar_fallback_warns():
    qp = QUBO(0, {(0, 0): 1.0, (1, 1): 1.0})
    with pytest.warns(UserWarning, match="CVaR loss"):
        best, params, extra, circuit, stats = qp.train_QAOA(
            gammas=[0.1, 0.2],
            betas=[0.2, 0.3],
            nshots=20,
            regular_loss=False,
            cvar_delta=0.5,
            engine="qiboml",
            maxiter=5,
        )
    assert np.isfinite(best)
    assert isinstance(params, np.ndarray)
    assert isinstance(extra, dict)
    assert isinstance(circuit, Circuit)
    assert isinstance(stats, dict)


def test_train_qaoa_with_noise_model_returns_original_circuit():
    qp = QUBO(0, {(0, 0): 1.0, (1, 1): 1.0})
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(0.05))

    result = qp.train_QAOA(
        gammas=[0.1, 0.2],
        betas=[0.2, 0.3],
        nshots=20,
        noise_model=noise_model,
        maxiter=5,
        engine="legacy",
    )

    assert len(result) == 6
    best, params, extra, circuit, stats, original_circuit = result
    assert np.isfinite(best)
    assert isinstance(params, np.ndarray)
    assert isinstance(extra, dict)
    assert isinstance(circuit, Circuit)
    assert isinstance(stats, dict)
    assert isinstance(original_circuit, Circuit)
    assert circuit is not original_circuit


@pytest.mark.parametrize("nshots", [None, 0])
@pytest.mark.parametrize("regular_loss", [True, False])
def test_train_qaoa_exact_mode_returns_probabilities(nshots, regular_loss):
    qp = QUBO(0, {(0, 0): 1.0, (1, 1): 1.0})
    kwargs = {}
    if not regular_loss:
        kwargs["cvar_delta"] = 0.5
    best, params, extra, circuit, stats = qp.train_QAOA(
        gammas=[0.1, 0.2],
        betas=[0.2, 0.3],
        nshots=nshots,
        regular_loss=regular_loss,
        maxiter=5,
        engine="legacy",
        **kwargs,
    )
    assert np.isfinite(best)
    assert isinstance(params, np.ndarray)
    assert isinstance(extra, dict)
    assert isinstance(circuit, Circuit)
    assert isinstance(stats, dict)
    assert all(isinstance(value, float) for value in stats.values())
    assert np.isclose(sum(stats.values()), 1.0)


def test_train_qaoa_cvar_delta_validation():
    qp = QUBO(0, {(0, 0): 1.0})
    with pytest.raises(ValueError, match="cvar_delta must satisfy 0 < cvar_delta <= 1"):
        qp.train_QAOA(
            gammas=[0.1],
            betas=[0.2],
            regular_loss=False,
            cvar_delta=0.0,
        )


def create_svp_mixer(name_to_index, beta):
    """
    Helper function to create a mixer circuit

    Args:
        name_to_index (dict): a name to index mapping required to create mixer to preserve probability of 0
        beta (float): Circuit parameter

    Returns:
        :class:`qibo.models.Circuit`: Mixer circuit
    """
    n = len(name_to_index)
    mixer = Circuit(n, density_matrix=True)
    # Get the set of indices where it takes values 1; to help construct the mixer
    active_set = {
        value
        for key, value in name_to_index.items()
        if any(_x in key for _x in ("x", "y"))
    }
    for i in range(n):
        if i in active_set:
            mixer.add(gates.X(i))
        mixer.add(gates.RY((i + 1) % n, beta))
        mixer.add(gates.CZ(i, (i + 1) % n))
        if i in active_set:
            mixer.add(gates.X(i))
    return mixer


@pytest.mark.parametrize(
    "gammas, betas, alphas, reg_loss, cvar_delta",
    [
        ([0.1, 0.2], [0.3, 0.4], None, True, None),
        ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6], False, 0.1),
    ],
)
def test_train_QAOA_svp_mixer(gammas, betas, alphas, reg_loss, cvar_delta):
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

    # SVP_mixers is now a list of functions that take beta and return a circuit
    svp_mixers = [
        lambda beta, idx=idx: create_svp_mixer(name_to_index, beta)
        for idx in range(len(betas))
    ]

    result = QUBO(0, numeric_qubo).train_QAOA(
        gammas=gammas,
        betas=betas,
        alphas=alphas,
        nshots=10,
        regular_loss=reg_loss,
        cvar_delta=cvar_delta,
        custom_mixer=svp_mixers,
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
def test_train_QAOA_svp_mixer_lambda(gammas, betas, alphas, reg_loss, cvar_delta):

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

    mixer_lambda = lambda beta: create_svp_mixer(name_to_index, beta)

    result = QUBO(0, numeric_qubo).train_QAOA(
        gammas=gammas,
        betas=betas,
        alphas=alphas,
        nshots=10,
        regular_loss=reg_loss,
        cvar_delta=cvar_delta,
        custom_mixer=[mixer_lambda],
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
def test_train_QAOA_svp_mixer_noise_model(gammas, betas, alphas, reg_loss, cvar_delta):

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

    lam = 0.1
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))

    # SVP_mixers is now a list of functions that take beta and return a circuit
    svp_mixers = [
        lambda beta, idx=idx: create_svp_mixer(name_to_index, beta)
        for idx in range(len(betas))
    ]

    result = QUBO(0, numeric_qubo).train_QAOA(
        gammas=gammas,
        betas=betas,
        alphas=alphas,
        nshots=10,
        regular_loss=reg_loss,
        cvar_delta=cvar_delta,
        custom_mixer=svp_mixers,
        noise_model=noise_model,
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


def test_qubo_energy_paths_consistency_single_qubit():
    """Regression test for energy consistency across sampled/symbolic/exact paths.

    QUBO: f(x) = x  => f(0)=0, f(1)=1
    State prepared: |1>
    """
    set_backend("numpy")
    backend = get_backend()

    qubo = QUBO(0.0, {(0, 0): 1.0})

    # Prepare |1> and measure
    circ = Circuit(1)
    circ.add(gates.X(0))
    circ.add(gates.M(0))

    nshots = 1000
    freqs = backend.execute_circuit(circ, nshots=nshots).frequencies(binary=True)
    sampled = (
        sum(
            qubo.evaluate_f([int(b) for b in bitstr]) * count
            for bitstr, count in freqs.items()
        )
        / nshots
    )

    # Prepare |1> without measurement for expectation values
    circ_no_m = Circuit(1)
    circ_no_m.add(gates.X(0))

    ham_expect = float(
        qubo.construct_symbolic_Hamiltonian_from_QUBO().expectation(circ_no_m)
    )
    exact_loss = float(qubo.qubo_to_qaoa_object().hamiltonian.expectation(circ_no_m))

    true_f1 = qubo.evaluate_f([1])

    # Keep the original diagnostic values visible in pytest output on failure
    debug_msg = (
        f"\ntrue f(1)          : {true_f1}\n"
        f"[A] sampled        : {sampled}\n"
        f"[B] H expectation  : {ham_expect}\n"
        f"[C] exact-mode loss: {exact_loss}\n"
        f"qubo_to_ising      : {qubo.qubo_to_ising()}\n"
    )

    # Ground truth check
    assert true_f1 == 1.0, debug_msg

    # Sampling path should match ground truth for this deterministic preparation
    assert sampled == pytest.approx(1.0, abs=1e-12), debug_msg

    # The next two assertions enforce consistency goals.
    # If current implementation is inconsistent, these will fail and expose the gap.
    assert ham_expect == pytest.approx(true_f1, abs=1e-12), debug_msg
    assert exact_loss == pytest.approx(true_f1, abs=1e-12), debug_msg


@pytest.mark.skipif(not _qiboml_available(), reason="qiboml/torch not installed")
def test_qiboml_energy_consistency_with_direct_evaluation():
    """The qiboml path should return energies consistent with direct QUBO evaluation."""
    # Simple 2-qubit QUBO: f(x0, x1) = 10 + x0 + x1 + x0*x1
    # f(0,0)=0, f(1,0)=1, f(0,1)=1, f(1,1)=3
    qp = QUBO(0.0, {(0, 0): 1.0, (1, 1): 1.0, (0, 1): 1.0})
    all_values = [qp.evaluate_f([x0, x1]) for x0 in (0, 1) for x1 in (0, 1)]
    min_f = min(all_values)
    max_f = max(all_values)

    best, params, extra, circuit, freqs = qp.train_QAOA(
        gammas=[0.1],
        betas=[0.2],
        nshots=500,
        engine="qiboml",
        optimizer="adam",
        lr=0.05,
        epochs=5,
    )

    # The best loss must fall within [min_f, max_f]; a constant offset would push it outside.
    assert min_f <= best <= max_f, (
        f"qiboml best={best:.6f} is outside the QUBO range [{min_f}, {max_f}]. "
        "This likely means an extra energy_shift is being applied."
    )

    # All losses in the history should also be within QUBO range.
    for i, loss in enumerate(extra["loss_history"]):
        assert (
            min_f <= loss <= max_f
        ), f"loss_history[{i}]={loss:.6f} is outside the QUBO range [{min_f}, {max_f}]."


def test_linear_initialization():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    lp = LinearProblem(A, b)
    assert np.array_equal(lp.A, A)
    assert np.array_equal(lp.b, b)
    assert lp.n == 2


def test_linear_add_multiplication_operators():
    """Test addition and multiplication operators for LinearProblem"""
    lp1 = LinearProblem(np.array([[1, 2], [3, 4]]), np.array([5, 6]))
    lp2 = LinearProblem(np.array([[2, 2], [2, 2]]), np.array([2, 2]))
    lp3 = 2 * lp1 + lp2 * 0.5

    assert np.array_equal(lp3.A, np.array([[3, 5], [7, 9]]))
    assert np.array_equal(lp3.b, np.array([11, 13]))

    # Test type error
    with pytest.raises(TypeError):
        lp3 = lp1 * "invalid"


def test_linear_evaluate_f():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    lp = LinearProblem(A, b)
    x = np.array([1, 1])
    result = lp.evaluate_f(x)
    assert np.array_equal(result, np.array([8, 13]))


def test_linear_square():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    lp = LinearProblem(A, b)
    quadratic = lp.square()
    Qdict = quadratic.Qdict
    offset = quadratic.offset
    expected_Qdict = {(0, 0): 56, (0, 1): 14, (1, 0): 14, (1, 1): 88}
    expected_offset = 61
    assert Qdict == expected_Qdict
    assert offset == expected_offset


@pytest.mark.parametrize(
    "variable_dict, var_to_idx, expected",
    [
        (
            {("x1", "x2"): 1.5, ("x2", "x3"): -0.5, "x3": 2.0},
            {"x1": 0, "x2": 1, "x3": 2},
            {(0, 1): 1.5, (1, 2): -0.5, 2: 2.0},
        ),
        ({((0, 1), (1, 0)): 1}, {(0, 1): 0, (1, 0): 1}, {(0, 1): 1}),
    ],
)
def test_variable_dict_to_ind(variable_dict, var_to_idx, expected):
    ind_dict = variable_dict_to_ind_dict(variable_dict, var_to_idx)
    assert expected == ind_dict


def test_construct_symbolic_hamiltonian_matches_qubo():
    qdict = {(0, 0): 0.25, (0, 1): -0.5, (1, 1): 1.5}
    qubo = QUBO(0.75, qdict)
    ham = qubo.construct_symbolic_Hamiltonian_from_QUBO()

    diag = np.diag(ham.matrix).real
    h, J, constant = qubo.qubo_to_ising()
    expected = []
    for bits in itertools.product([0, 1], repeat=qubo.n):
        spins = [1 if bit else -1 for bit in bits]
        energy = constant
        for idx, coeff in h.items():
            energy += coeff * spins[idx]
        for (u, v), coeff in J.items():
            energy += coeff * spins[u] * spins[v]
        expected.append(energy)

    assert np.allclose(np.sort(diag), np.sort(expected))
    assert ham.nqubits == qubo.n


def test_canonical_q_merges_symmetric_terms():
    qdict = {(0, 1): 1.0, (1, 0): 0.25, (1, 1): -1.0}
    qubo = QUBO(0.0, qdict)
    canonical = qubo.canonical_q()

    assert canonical[(0, 1)] == pytest.approx(1.25)
    assert canonical[(1, 1)] == pytest.approx(2 * qdict[(1, 1)])
    assert all(i <= j for i, j in canonical)


def test_train_qaoa_requires_layer_information():
    qubo = QUBO(0.0, {(0, 0): 1.0})
    with pytest.raises(
        ValueError,
        match="Either p or gammas must be provided to define the number of layers\\.",
    ):
        qubo.train_QAOA(betas=[0.1])


def test_variable_to_ind_round_trip():
    variables = ["x1", "x2", "x3"]
    var_to_idx, idx_to_var = variable_to_ind(variables)

    assert var_to_idx == {"x1": 0, "x2": 1, "x3": 2}
    assert idx_to_var == {0: "x1", 1: "x2", 2: "x3"}
