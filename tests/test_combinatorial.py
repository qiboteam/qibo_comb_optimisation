import networkx as nx
import numpy as np
import pytest
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models import QAOA
from test_models_variational import assert_regression_fixture
from enum import Enum

from qiboopt.combinatorial.combinatorial import (
    MIS,
    MWVC,
    QAP,
    TSP,
    MaxCut,
    _calculate_two_to_one,
    _edge_list_from_W,
    _ensure_weight_matrix,
    _maxcut_mixer,
    _maxcut_phaser,
    _normalize,
    _tsp_mixer,
    _tsp_phaser,
)
from qiboopt.opt_class.opt_class import QUBO, LinearProblem


def test__calculate_two_to_one():
    num_cities = 3
    result, _ = _calculate_two_to_one(num_cities)
    expected_array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    expected_dict = {
        (i, j): int(expected_array[i, j])
        for i in range(num_cities)
        for j in range(num_cities)
    }
    assert (
        expected_dict == result
    ), "calculate_two_to_one did not return the expected result"


def test__tsp_phaser():
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    hamiltonian = _tsp_phaser(distance_matrix)
    assert isinstance(
        hamiltonian, SymbolicHamiltonian
    ), "tsp_phaser did not return a SymbolicHamiltonian"
    assert (
        hamiltonian.terms is not None
    ), "tsp_phaser returned a Hamiltonian with no terms"


def test__tsp_mixer():
    num_cities = 3
    hamiltonian = _tsp_mixer(num_cities)
    assert isinstance(
        hamiltonian, SymbolicHamiltonian
    ), "tsp_mixer did not return a SymbolicHamiltonian"
    assert (
        hamiltonian.terms is not None
    ), "tsp_mixer returned a Hamiltonian with no terms"


def test_tsp_class():
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    tsp = TSP(distance_matrix)
    assert tsp.num_cities == 3, "TSP class did not set the number of cities correctly"
    assert np.array_equal(
        tsp.distance_matrix, distance_matrix
    ), "TSP class did not set the distance matrix correctly"

    obj_hamil, mixer = tsp.hamiltonians()
    assert isinstance(
        obj_hamil, SymbolicHamiltonian
    ), "TSP.hamiltonians did not return a SymbolicHamiltonian for the objective Hamiltonian"
    assert isinstance(
        mixer, SymbolicHamiltonian
    ), "TSP.hamiltonians did not return a SymbolicHamiltonian for the mixer"

    ordering = [0, 1, 2]
    initial_state = tsp.prepare_initial_state(ordering)
    assert (
        initial_state is not None
    ), "TSP.prepare_initial_state did not return a valid state"


@pytest.fixture
def setup_tsp():
    num_cities = 3
    distance_matrix = [[0, 10, 15], [10, 0, 20], [15, 20, 0]]
    two_to_one = lambda u, j: u * num_cities + j
    return num_cities, distance_matrix, two_to_one


def test_qubo_objective_function(setup_tsp):
    num_cities, distance_matrix, two_to_one = setup_tsp

    # Initialize q_dict and compute objective function part
    q_dict = {}
    for u in range(num_cities):
        for v in range(num_cities):
            if v != u:
                for j in range(num_cities):
                    q_dict[two_to_one(u, j), two_to_one(v, (j + 1) % num_cities)] = (
                        distance_matrix[u][v]
                    )

    # Check that the q_dict is correctly populated
    expected_q_dict = {
        (0, 4): 10,
        (1, 5): 10,
        (2, 3): 10,
        (0, 7): 15,
        (1, 8): 15,
        (2, 6): 15,
        (3, 1): 10,
        (4, 2): 10,
        (5, 0): 10,
        (3, 7): 20,
        (4, 8): 20,
        (5, 6): 20,
        (6, 1): 15,
        (7, 2): 15,
        (8, 0): 15,
        (6, 4): 20,
        (7, 5): 20,
        (8, 3): 20,
    }
    assert q_dict == expected_q_dict, f"Expected {expected_q_dict}, but got {q_dict}"


def test_row_constraints(setup_tsp):
    num_cities, _, two_to_one = setup_tsp
    penalty = 10
    qp = QUBO(0, {})

    for v in range(num_cities):
        row_constraint = np.array([0 for _ in range(num_cities**2)])
        for j in range(num_cities):
            row_constraint[two_to_one(v, j)] = 1
        lp = LinearProblem(row_constraint, -1)
        tmp_qp = lp.square()
        tmp_qp *= penalty
        qp += tmp_qp

    # Check that the row constraints are correctly applied to the QUBO
    expected_row_constraints = {
        (0, 0): -10,
        (0, 1): 10,
        (0, 2): 10,
        (0, 3): 0,
        (0, 4): 0,
        (0, 5): 0,
        (0, 6): 0,
        (0, 7): 0,
        (0, 8): 0,
        (1, 0): 10,
        (1, 1): -10,
        (1, 2): 10,
        (1, 3): 0,
        (1, 4): 0,
        (1, 5): 0,
        (1, 6): 0,
        (1, 7): 0,
        (1, 8): 0,
        (2, 0): 10,
        (2, 1): 10,
        (2, 2): -10,
        (2, 3): 0,
        (2, 4): 0,
        (2, 5): 0,
        (2, 6): 0,
        (2, 7): 0,
        (2, 8): 0,
        (3, 0): 0,
        (3, 1): 0,
        (3, 2): 0,
        (3, 3): -10,
        (3, 4): 10,
        (3, 5): 10,
        (3, 6): 0,
        (3, 7): 0,
        (3, 8): 0,
        (4, 0): 0,
        (4, 1): 0,
        (4, 2): 0,
        (4, 3): 10,
        (4, 4): -10,
        (4, 5): 10,
        (4, 6): 0,
        (4, 7): 0,
        (4, 8): 0,
        (5, 0): 0,
        (5, 1): 0,
        (5, 2): 0,
        (5, 3): 10,
        (5, 4): 10,
        (5, 5): -10,
        (5, 6): 0,
        (5, 7): 0,
        (5, 8): 0,
        (6, 0): 0,
        (6, 1): 0,
        (6, 2): 0,
        (6, 3): 0,
        (6, 4): 0,
        (6, 5): 0,
        (6, 6): -10,
        (6, 7): 10,
        (6, 8): 10,
        (7, 0): 0,
        (7, 1): 0,
        (7, 2): 0,
        (7, 3): 0,
        (7, 4): 0,
        (7, 5): 0,
        (7, 6): 10,
        (7, 7): -10,
        (7, 8): 10,
        (8, 0): 0,
        (8, 1): 0,
        (8, 2): 0,
        (8, 3): 0,
        (8, 4): 0,
        (8, 5): 0,
        (8, 6): 10,
        (8, 7): 10,
        (8, 8): -10,
    }
    assert (
        qp.Qdict == expected_row_constraints
    ), f"Expected {expected_row_constraints}, but got {qp.Qdict}"


def test_column_constraints(setup_tsp):
    num_cities, _, two_to_one = setup_tsp
    penalty = 10
    qp = QUBO(0, {})

    for j in range(num_cities):
        col_constraint = np.array([0 for _ in range(num_cities**2)])
        for v in range(num_cities):
            col_constraint[two_to_one(v, j)] = 1
        lp = LinearProblem(col_constraint, -1)
        tmp_qp = lp.square()
        tmp_qp *= penalty
        qp += tmp_qp

    # Check that the column constraints are correctly applied to the QUBO
    expected_col_constraints = {
        (0, 0): -10,
        (0, 1): 0,
        (0, 2): 0,
        (0, 3): 10,
        (0, 4): 0,
        (0, 5): 0,
        (0, 6): 10,
        (0, 7): 0,
        (0, 8): 0,
        (1, 0): 0,
        (1, 1): -10,
        (1, 2): 0,
        (1, 3): 0,
        (1, 4): 10,
        (1, 5): 0,
        (1, 6): 0,
        (1, 7): 10,
        (1, 8): 0,
        (2, 0): 0,
        (2, 1): 0,
        (2, 2): -10,
        (2, 3): 0,
        (2, 4): 0,
        (2, 5): 10,
        (2, 6): 0,
        (2, 7): 0,
        (2, 8): 10,
        (3, 0): 10,
        (3, 1): 0,
        (3, 2): 0,
        (3, 3): -10,
        (3, 4): 0,
        (3, 5): 0,
        (3, 6): 10,
        (3, 7): 0,
        (3, 8): 0,
        (4, 0): 0,
        (4, 1): 10,
        (4, 2): 0,
        (4, 3): 0,
        (4, 4): -10,
        (4, 5): 0,
        (4, 6): 0,
        (4, 7): 10,
        (4, 8): 0,
        (5, 0): 0,
        (5, 1): 0,
        (5, 2): 10,
        (5, 3): 0,
        (5, 4): 0,
        (5, 5): -10,
        (5, 6): 0,
        (5, 7): 0,
        (5, 8): 10,
        (6, 0): 10,
        (6, 1): 0,
        (6, 2): 0,
        (6, 3): 10,
        (6, 4): 0,
        (6, 5): 0,
        (6, 6): -10,
        (6, 7): 0,
        (6, 8): 0,
        (7, 0): 0,
        (7, 1): 10,
        (7, 2): 0,
        (7, 3): 0,
        (7, 4): 10,
        (7, 5): 0,
        (7, 6): 0,
        (7, 7): -10,
        (7, 8): 0,
        (8, 0): 0,
        (8, 1): 0,
        (8, 2): 10,
        (8, 3): 0,
        (8, 4): 0,
        (8, 5): 10,
        (8, 6): 0,
        (8, 7): 0,
        (8, 8): -10,
    }
    assert (
        qp.Qdict == expected_col_constraints
    ), f"Expected {expected_col_constraints}, but got {qp.Qdict}"


@pytest.mark.parametrize(
    "distance, penalty",
    [
        (np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]]), 10),
        (
            np.array(
                [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
            ),
            0,
        ),
        (
            np.array(
                [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
            ),
            1000.0,
        ),
        (np.array([[0, 10], [10, 0]]), 1.0),
        (np.array([0]), 1),
    ],
)
def test_tsp_penalty(distance, penalty):
    tsp = TSP(distance)
    qp = tsp.penalty_method(penalty)
    two_to_one, one_to_two = _calculate_two_to_one(tsp.num_cities)
    assert isinstance(qp, QUBO), "TSP.penalty_method() did not return a QUBO."
    assert qp.Qdict, "QUBO dictionary from TSP.penalty_method() is empty."
    for key, value in qp.Qdict.items():
        # key is a pair of indices of the QUBO, we have to get the corresponding city.
        origin, origin_slot = one_to_two[
            key[0]
        ]  # recall the two indices are of the form of (city, slot)
        destination, destination_slot = one_to_two[key[1]]
        if (
            origin != destination
            and (destination_slot - origin_slot) % tsp.num_cities == 1
        ):
            expected_value = distance[origin][destination]
        elif origin == destination and origin_slot == destination_slot:
            expected_value = -2 * penalty
        elif (origin == destination and origin_slot != destination_slot) or (
            origin != destination and origin_slot == destination_slot
        ):
            expected_value = penalty
        else:
            expected_value = 0.0
        assert (
            value == expected_value
        ), f"penalty test: Expected {expected_value} but got {value} for key {key}."


@pytest.mark.parametrize(
    "distance, penalty", [(np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]]), -1)]
)
def test_tsp_negative_penalty(distance, penalty):
    """Test negative penalty value"""
    tsp = TSP(distance)
    with pytest.raises(ValueError):
        qp = tsp.penalty_method(penalty)


def qaoa_function_of_layer(backend, layer):
    """
    This is a function to study the impact of the number of layers on QAOA, it takes
    in the number of layers and compute the distance of the mode of the histogram obtained
    from QAOA
    """
    num_cities = 3
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    # there are two possible cycles, one with distance 1, one with distance 1.9
    distance_matrix = distance_matrix.round(1)

    small_tsp = TSP(distance_matrix, backend=backend)
    initial_state = small_tsp.prepare_initial_state([i for i in range(num_cities)])
    obj_hamil, mixer = small_tsp.hamiltonians()
    qaoa = QAOA(obj_hamil, mixer=mixer)
    initial_state = backend.cast(initial_state, copy=True)
    best_energy, final_parameters, extra = qaoa.minimize(
        initial_p=[0.1 for i in range(layer)],
        initial_state=initial_state,
        method="BFGS",
        options={"maxiter": 1},
    )
    qaoa.set_parameters(final_parameters)
    return qaoa.execute(initial_state)


@pytest.mark.parametrize("nlayers", [2, 4])
def test_tsp(backend, nlayers):
    if nlayers == 4 and backend.platform in ("cupy", "cuquantum"):
        pytest.skip("Failing for cupy and cuquantum.")
    final_state = backend.to_numpy(qaoa_function_of_layer(backend, nlayers))
    atol = 4e-5 if backend.platform in ("cupy", "cuquantum") else 1e-5
    assert_regression_fixture(
        backend, final_state.real, f"tsp_layer{nlayers}_real.out", rtol=1e-3, atol=atol
    )
    assert_regression_fixture(
        backend, final_state.imag, f"tsp_layer{nlayers}_imag.out", rtol=1e-3, atol=atol
    )


def test_mis_class():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    mis = MIS(g)
    assert mis.n == 3, "MIS class did not set the number of nodes correctly"
    assert mis.g == g, "MIS class did not set the graph correctly"

    penalty = 10
    qp = mis.penalty_method(penalty)
    assert isinstance(qp, QUBO), "MIS.penalty_method did not return a QUBO"

    mis_str = str(mis)
    assert mis_str == "MIS", "MIS.__str__ did not return the expected string"


def test_qap():
    f = np.array([[0, 1], [1, 0]], dtype=float)
    d = np.array([[0, 2], [2, 0]], dtype=float)
    qap = QAP(f, d)
    answer = dict()
    for i in range(4):
        for j in range(4):
            answer[(i, j)] = 0.0
    answer[(0, 3)] = 2.0
    answer[(1, 2)] = 2.0
    answer[(2, 1)] = 2.0
    answer[(3, 0)] = 2.0
    assert qap.qp.Qdict == answer
    penalty_value = qap.suggest_penalty()
    penalized_qap = qap.penalty_method(penalty_value)
    assert penalized_qap.Qdict[(0, 1)] != 0


def test_qap_mismatched_shapes():
    f = np.array([[0, 1], [1, 0]], dtype=float)
    d = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]], dtype=float)  # 3x3 vs 2x2
    with pytest.raises(ValueError, match="same shape"):
        QAP(f, d)


def test_qap_non_square():
    f = np.array([[0, 1, 2], [1, 0, 3]], dtype=float)  # 2x3, not square
    d = np.array([[0, 1, 2], [1, 0, 3]], dtype=float)
    with pytest.raises(ValueError, match="square"):
        QAP(f, d)


def test_qap_custom_two_to_one():
    """Test that a custom two_to_one mapping is used instead of the default."""
    f = np.array([[0, 1], [1, 0]], dtype=float)
    d = np.array([[0, 2], [2, 0]], dtype=float)

    num_cities = 2
    custom_mapping = {
        (i, j): i * num_cities + j for i in range(num_cities) for j in range(num_cities)
    }
    custom_two_to_one = (custom_mapping, None)
    qap = QAP(f, d, two_to_one=custom_two_to_one)

    assert qap.two_to_one == custom_mapping


def test_mwvc():
    g = nx.Graph()
    g.add_nodes_from([(0, {"weight": 2}), (1, {"weight": 3}), (2, {"weight": 4})])
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    mwvc = MWVC(g)
    assert (
        mwvc.qp.Qdict[(0, 0)] == 2
        and mwvc.qp.Qdict[(1, 1)] == 3
        and mwvc.qp.Qdict[2, 2] == 4
    )
    penalty = mwvc.suggest_penalty(0.01)
    qp = mwvc.penalty_method(penalty)
    assert qp.Qdict[(0, 1)] != 0


def test_maxcut_helper_functions():
    weights = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
    matrix_form, nodes = _ensure_weight_matrix(weights)
    assert np.array_equal(matrix_form, weights)
    assert nodes == [0, 1, 2]

    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 1.5), ("b", "c", 2.0)])
    graph_matrix, graph_nodes = _ensure_weight_matrix(graph)
    idx = {node: i for i, node in enumerate(graph_nodes)}
    assert pytest.approx(graph_matrix[idx["a"], idx["b"]]) == 1.5
    assert pytest.approx(graph_matrix[idx["b"], idx["c"]]) == 2.0

    with pytest.raises(ValueError):
        _ensure_weight_matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        _ensure_weight_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    loop_graph = nx.Graph()
    loop_graph.add_weighted_edges_from([("a", "a", 5.0), ("a", "b", 2.5)])
    loop_matrix, loop_nodes = _ensure_weight_matrix(loop_graph)
    loop_idx = {node: i for i, node in enumerate(loop_nodes)}
    assert loop_matrix[loop_idx["a"], loop_idx["a"]] == 0.0
    assert pytest.approx(loop_matrix[loop_idx["a"], loop_idx["b"]]) == 2.5
    with pytest.raises(TypeError):
        _ensure_weight_matrix(123)

    same, same_scale = _normalize(weights, "none")
    assert np.array_equal(same, weights) and same_scale == 1.0

    maxdeg_scaled, maxdeg_scale = _normalize(weights, "maxdeg")
    expected_maxdeg = np.max(np.sum(np.abs(weights), axis=1))
    assert pytest.approx(maxdeg_scale) == expected_maxdeg
    assert np.allclose(maxdeg_scaled, weights / maxdeg_scale)

    sum_scaled, sum_scale = _normalize(weights, "sum")
    expected_sum = np.sum(np.abs(np.triu(weights, 1)))
    assert pytest.approx(sum_scale) == expected_sum
    assert np.allclose(sum_scaled, weights / sum_scale)

    zeros = np.zeros((2, 2), dtype=float)
    zero_maxdeg, zero_maxdeg_scale = _normalize(zeros, "maxdeg")
    assert zero_maxdeg_scale == pytest.approx(1.0)
    assert np.array_equal(zero_maxdeg, zeros)
    zero_sum, zero_sum_scale = _normalize(zeros, "sum")
    assert zero_sum_scale == pytest.approx(1.0)
    assert np.array_equal(zero_sum, zeros)

    with pytest.raises(ValueError):
        _normalize(weights, "invalid")

    edges = _edge_list_from_W(weights)
    assert set(edges) == {(0, 1), (1, 2)}


def test_maxcut_mixer_variants():
    ham_x = _maxcut_mixer(3, mode="x")
    assert isinstance(ham_x, SymbolicHamiltonian)

    edges = [(0, 1), (1, 2)]
    ham_xy = _maxcut_mixer(3, mode="xy", edges=edges)
    assert isinstance(ham_xy, SymbolicHamiltonian)
    fallback_xy = _maxcut_mixer(3, mode="xy")
    assert isinstance(fallback_xy, SymbolicHamiltonian)
    assert len(fallback_xy.terms) == 6

    with pytest.raises(ValueError):
        _maxcut_mixer(2, mode="z")


def test_maxcut_prepare_initial_state_variants(monkeypatch):
    weights = np.zeros((2, 2), dtype=float)
    mc = MaxCut(weights)

    plus_state = mc.prepare_initial_state()
    assert np.allclose(plus_state, np.full(4, 0.5))

    basis_state = mc.prepare_initial_state(init="bitstring", bitstring="10")
    assert np.array_equal(basis_state, np.array([0, 0, 1, 0], dtype=complex))

    with pytest.raises(ValueError):
        mc.prepare_initial_state(init="bitstring")
    with pytest.raises(ValueError):
        mc.prepare_initial_state(init="bitstring", bitstring="1")

    class _DeterministicRNG:
        def integers(self, low, high=None, size=None):
            return np.array([1, 0], dtype=int)

    monkeypatch.setattr(
        "qiboopt.combinatorial.combinatorial.np.random.default_rng",
        lambda: _DeterministicRNG(),
    )
    random_state = mc.prepare_initial_state(init="random")
    assert np.array_equal(random_state, basis_state)

    with pytest.raises(ValueError):
        mc.prepare_initial_state(init="invalid")


def test_maxcut_qubo_and_cut_metrics():
    weights = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    mc = MaxCut(weights, normalize="maxdeg", mixer="x")

    obj_h, mix_h = mc.hamiltonians()
    assert isinstance(obj_h, SymbolicHamiltonian)
    assert isinstance(mix_h, SymbolicHamiltonian)

    qubo_max = mc.to_qubo(maximize=True, use_scaled=False)
    qubo_min = mc.to_qubo(maximize=False, use_scaled=False)

    assert qubo_max.Qdict[(0, 0)] == pytest.approx(-3.0)
    assert qubo_max.Qdict[(0, 1)] == pytest.approx(2.0)
    assert qubo_min.Qdict[(0, 1)] == pytest.approx(-2.0)

    qubo_scaled = mc.to_qubo(maximize=True, use_scaled=True)
    scale = np.max(np.sum(np.abs(weights), axis=1))
    assert qubo_scaled.Qdict[(0, 1)] == pytest.approx(qubo_max.Qdict[(0, 1)] / scale)

    bitstring = "010"
    cut_true = mc.cut_value(bitstring)
    cut_scaled = mc.cut_value(bitstring, use_scaled=True)
    assert cut_true == pytest.approx(4.0)
    assert cut_scaled == pytest.approx(cut_true / scale)

    S, Sp = mc.partition_from_bits(bitstring)
    assert S == {0, 2} and Sp == {1}

    with pytest.raises(ValueError):
        mc.cut_value("01")
    with pytest.raises(ValueError):
        mc.partition_from_bits("01")

    assert mc.rescale_energy(0.5) == pytest.approx(mc.energy_scale * 0.5)
    assert str(mc) == "MaxCut(n=3, normalize='maxdeg', mixer='x')"

    zero_weights = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=float)
    zero_mc = MaxCut(zero_weights)
    zero_qubo = zero_mc.to_qubo()
    assert (0, 2) not in zero_qubo.Qdict
    assert zero_mc.cut_value("010") == pytest.approx(3.0)


def test_maxcut_phaser_skips_zero_weights():
    weights = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=float)
    ham = _maxcut_phaser(weights)
    assert isinstance(ham, SymbolicHamiltonian)
    assert len(ham.terms) == 2
