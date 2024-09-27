import numpy as np
import networkx as nx
from qibo.hamiltonians import SymbolicHamiltonian
from qibo_comb_optimisation.combinatorial_classes.combinatorial_classes import calculate_two_to_one, tsp_phaser, tsp_mixer, TSP, Mis
from qibo_comb_optimisation.optimization_class.optimization_class import quadratic_problem


def test_calculate_two_to_one():
    num_cities = 3
    result = calculate_two_to_one(num_cities)
    expected = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert np.array_equal(result, expected), "calculate_two_to_one did not return the expected result"


def test_tsp_phaser():
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    hamiltonian = tsp_phaser(distance_matrix)
    assert isinstance(hamiltonian, SymbolicHamiltonian), "tsp_phaser did not return a SymbolicHamiltonian"
    assert hamiltonian.terms is not None, "tsp_phaser returned a Hamiltonian with no terms"


def test_tsp_mixer():
    num_cities = 3
    hamiltonian = tsp_mixer(num_cities)
    assert isinstance(hamiltonian, SymbolicHamiltonian), "tsp_mixer did not return a SymbolicHamiltonian"
    assert hamiltonian.terms is not None, "tsp_mixer returned a Hamiltonian with no terms"


def test_tsp_class():
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    tsp = TSP(distance_matrix)
    assert tsp.num_cities == 3, "TSP class did not set the number of cities correctly"
    assert np.array_equal(tsp.distance_matrix, distance_matrix), "TSP class did not set the distance matrix correctly"

    obj_hamil, mixer = tsp.hamiltonians()
    assert isinstance(obj_hamil, SymbolicHamiltonian), "TSP.hamiltonians did not return a SymbolicHamiltonian for the objective Hamiltonian"
    assert isinstance(mixer, SymbolicHamiltonian), "TSP.hamiltonians did not return a SymbolicHamiltonian for the mixer"

    ordering = [0, 1, 2]
    initial_state = tsp.prepare_initial_state(ordering)
    assert initial_state is not None, "TSP.prepare_initial_state did not return a valid state"


def run_tests():
    # Test Setup
    num_cities = 4
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    two_to_one = lambda u, j: u * num_cities + j  # Example two_to_one mapping
    tsp = TSP(num_cities, distance_matrix, two_to_one)  # Assuming TSP class exists

    # Test 1: Basic functionality with a moderate penalty value
    penalty = 1.0
    qp = tsp.penalty_method(penalty)
    assert isinstance(qp, quadratic_problem), "Test 1 Failed: Returned object is not a quadratic_problem."
    assert len(qp.q_dict) > 0, "Test 1 Failed: QUBO dictionary is empty."

    # Test 2: Zero penalty
    penalty = 0.0
    qp = tsp.penalty_method(penalty)
    for key, value in qp.q_dict.items():
        expected_value = distance_matrix[key[0] // num_cities][key[1] // num_cities]
        assert value == expected_value, f"Test 2 Failed: Expected {expected_value} but got {value} for key {key}."

    # Test 3: High penalty
    penalty = 1000.0
    qp = tsp.penalty_method(penalty)
    for key, value in qp.q_dict.items():
        assert abs(value) >= 1000, f"Test 3 Failed: Value {value} is less than expected penalty."

    # Test 4: Single city (edge case)
    tsp.num_cities = 1
    tsp.distance_matrix = [[0]]
    qp = tsp.penalty_method(penalty=1.0)
    assert len(qp.q_dict) == 0, "Test 4 Failed: QUBO dictionary should be empty for a single city."

    # Test 5: Two cities (small problem)
    tsp.num_cities = 2
    tsp.distance_matrix = [
        [0, 10],
        [10, 0]
    ]
    qp = tsp.penalty_method(penalty=1.0)
    assert len(qp.q_dict) > 0, "Test 5 Failed: QUBO dictionary should not be empty for two cities."


def test_mis_class():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    mis = Mis(g)
    assert mis.n == 3, "Mis class did not set the number of nodes correctly"
    assert mis.g == g, "Mis class did not set the graph correctly"

    penalty = 10
    qp = mis.penalty_method(penalty)
    assert isinstance(qp, quadratic_problem), "Mis.penalty_method did not return a quadratic_problem"

    mis_str = str(mis)
    assert mis_str == "Mis", "Mis.__str__ did not return the expected string"
