import numpy as np
import networkx as nx
from qibo.hamiltonians import SymbolicHamiltonian
from qibo_comb_optimisation.combinatorial_classes.combinatorial_classes import calculate_two_to_one, tsp_phaser, tsp_mixer, TSP, Mis
from qibo_comb_optimisation.optimisation_class.optimisation_class import quadratic_problem


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
