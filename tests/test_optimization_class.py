import numpy as np
from qibo_comb_optimisation.optimization_class.optimization_class import quadratic_problem, linear_problem


# Test initialization of the quadratic_problem class
def test_initialization():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = quadratic_problem(0, Qdict)

    assert qp.Qdict == Qdict
    assert qp.offset == 0.0
    assert qp.n == 2  # Maximum variable index in Qdict keys


def test_multiply_scalar():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = quadratic_problem(0, Qdict)
    qp.multiply_scalar(2)
    assert qp.Qdict == {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}


def test_add():
    Qdict1 = {(0, 0): 1.0, (0, 1): 0.5}
    Qdict2 = {(0, 0): -1.0, (1, 1): 2.0}
    qp1 = quadratic_problem(0, Qdict1)
    qp2 = quadratic_problem(0, Qdict2)
    qp1 + qp2
    assert qp1.Qdict == {(0, 0): 0.0, (0, 1): 0.5, (1, 1): 2.0}


def test_qubo_to_ising():
    Qdict = {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}
    qp = quadratic_problem(0, Qdict)

    h, J, constant = qp.qubo_to_ising()

    assert h == {0: 1.25, 1: -0.75}
    assert J == {(0, 1): 0.25}
    assert constant == 0.25


def test_evaluate_f():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = quadratic_problem(0, Qdict)

    x = [1, 1]
    f_value = qp.evaluate_f(x)

    assert f_value == 0.5


def test_evaluate_grad_f():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = quadratic_problem(0, Qdict)

    x = [1, 1]
    grad = qp.evaluate_grad_f(x)
    assert np.array_equal(grad, [1.5, -0.5])


def test_tabu_search():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = quadratic_problem(0, Qdict)

    best_solution, best_obj_value = qp.tabu_search(max_iterations=50, tabu_size=5)

    assert len(best_solution) == 2
    assert isinstance(best_obj_value, float)


def test_brute_force():
    Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    qp = quadratic_problem(0, Qdict)

    opt_vector, min_value = qp.brute_force()

    assert len(opt_vector) == 2
    assert isinstance(min_value, float)


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
