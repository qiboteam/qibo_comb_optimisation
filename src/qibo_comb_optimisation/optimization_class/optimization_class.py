import itertools

import numpy as np
from qibo import hamiltonians
from qibo.symbols import Z


class QUBO:
    """
    A class used to represent either a Quadratic Unconstrained Binary Optimization (QUBO) problem or Ising model.
    Attributes
    ----------
    offset : float
        The constant offset of the QUBO problem.
    args : dict or 2 numpy arrays
        A dictionary representing the quadratic coefficient for len of args is 1
        Two numpy arrays represent the coefficient for the Ising model.
    n : int
        Number of variables involved in the problem.

    Methods
    -------
    multiply_scalar(scalar: float):
        Multiplies all the coefficients by a scalar value.

    qubo_to_ising() -> Tuple[dict, dict, float]:
        Converts the QUBO problem into Ising model parameters.

    evaluate_f(x: List[int]) -> float:
        Evaluates the quadratic function for a given binary vector.

    evaluate_grad_f(x: List[int]) -> List[float]:
        Evaluates the gradient of the quadratic function at a given binary vector.

    tabu_search(max_iterations: int, tabu_size: int) -> Tuple[List[int], float]:
        Solves the QUBO problem using the Tabu search algorithm.

    brute_force() -> Tuple[List[int], float]:
        Solves the QUBO problem by exhaustively evaluating all possible solutions.
    """

    def __init__(self, offset, *args):
        """
        Initializes the QUBO class

        Parameters
        ________

        offset : float
            The constant offset of the QUBO problem.
        args: this can be of length 1 or length 2.
             If the length is one, it takes in a dictionary and it will be assigned to the QDict object, the coefficeint
             of QUBO. It represents the matrix Q.
             If the length is 2, it takes in h and J respectively for Ising formulation.

             We have the following relation

                .. math::

             s'  J  s + h'  s = offset + x'  Q x
        Args:
            h (dict[variable, bias]):
                Linear biases as a dict of the form {v: bias, ...}, where keys are variables of
                the model and values are biases.
            J (dict[(variable, variable), bias]):
               Quadratic biases as a dict of the form {(u, v): bias, ...}, where keys
               are 2-tuples of variables of the model and values are optimization_class biases
               associated with the pair of variables (the interaction).
        Example
        -------
        >>> Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
        >>> qp = QUBO(0, Qdict)
        >>> qp.Qdict
        {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
        """

        self.offset = offset
        if len(args) == 1 and isinstance(args[0], dict):
            self.Qdict = args[0]
            self.n = 0
            for key in self.Qdict:
                self.n = max([self.n, key[0], key[1]])
            self.n += 1
        else:
            h = args[0]
            J = args[1]
            self.Qdict = {(v, v): 2.0 * bias for v, bias in h.items()}
            self.n = 0

            # next the optimization_class biases
            for (u, v), bias in self.Qdict.items():
                if bias == 0.0:
                    continue
                self.Qdict[(u, v)] = 4.0 * bias
                self.Qdict[(u, u)] = self.Qdict.setdefault((u, u), 0) - 2.0 * bias
                self.Qdict[(v, v)] = self.Qdict.setdefault((v, v), 0) - 2.0 * bias
                self.n = max([self.n, u, v])
            self.n += 1
            # finally adjust the offset based on QUBO definitions rather than Ising formulation
            self.offset += sum(J.values()) - sum(h.values())

    def multiply_scalar(self, scalar_multiplier):
        """
        Multiplies all the quadratic coefficients by a scalar value.

        Parameters
        ----------
        scalar : float
            The scalar value by which to multiply the coefficients.

        Example
        -------
        >>> Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
        >>> qp = QUBO(0, Qdict)
        >>> qp.multiply_scalar(2)
        >>> qp.Qdict
        {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}
        """
        for key in self.Qdict:
            self.Qdict[key] *= scalar_multiplier
        self.offset *= scalar_multiplier

    def __add__(self, other_Quadratic):
        """
        :param other_Quadratic: another optimization_class class object
        :return: updating the optimization_class function to obtain the sum
        """
        for key in other_Quadratic.Qdict:
            if key in self.Qdict:
                self.Qdict[key] += other_Quadratic.Qdict[key]
            else:
                self.Qdict[key] = other_Quadratic.Qdict[key]
        self.n = max(self.n, other_Quadratic.n)
        self.offset += other_Quadratic.offset

    def qubo_to_ising(self, constant=0.0):
        """Convert a QUBO problem to an Ising problem.

        Map a optimization_class unconstrained binary optimization (QUBO) problem  defined over binary variables
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

        for (u, v), bias in self.Qdict.items():
            if u == v:
                if u in h:
                    h[u] += bias / 2
                else:
                    h[u] = bias / 2
                linear_offset += bias

            else:
                if bias != 0.0:
                    J[(u, v)] = bias / 4

                if u in h:
                    h[u] += bias / 4
                else:
                    h[u] = bias / 4

                if v in h:
                    h[v] += bias / 4
                else:
                    h[v] = bias / 4

                quadratic_offset += bias

        constant += 0.5 * linear_offset + 0.25 * quadratic_offset

        return h, J, constant

    def construct_symbolic_Hamiltonian_from_QUBO(self):
        """
        Constructs a symbolic Hamiltonian from the QUBO problem by converting it
        to an Ising model.

        The method calls the qubo_to_ising function to convert the QUBO formulation
        into an Ising Hamiltonian with linear and quadratic terms. Then, it creates
        a symbolic Hamiltonian using the qibo library.

        Returns:
            A symbolic Hamiltonian that corresponds to the QUBO problem.
        """
        # Correct the call to qubo_to_ising (no need to pass self.Qdict)
        h, J, constant = self.qubo_to_ising()

        # Create a symbolic Hamiltonian using qibo symbols
        symbolic_ham = sum(h[i] * Z(i) for i in h)
        symbolic_ham += sum(J[u, v] * Z(u) * Z(v) for (u, v) in J)
        symbolic_ham += constant

        # Return the symbolic Hamiltonian using qibo's Hamiltonian object
        ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
        return ham

    def evaluate_f(self, x):
        """
        Evaluates the quadratic function for a given binary vector.

        Parameters
        ----------
        x : list of int
            A binary vector for which to evaluate the function.

        Returns
        -------
        float
            The evaluated function value.

        Example
        -------
        >>> Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
        >>> qp = QUBO(0, Qdict)
        >>> x = [1, 1]
        >>> qp.evaluate_f(x)
        0.5
        """
        f_value = self.offset
        for i in range(self.n):
            if x[i] != 0:
                # manage diagonal term first
                if (i, i) in self.Qdict:
                    f_value += self.Qdict[(i, i)]
                    for j in range(i + 1, self.n):
                        if x[j] != 0:
                            f_value += self.Qdict.get((i, j), 0) + self.Qdict.get(
                                (j, i), 0
                            )
        return f_value

    def evaluate_grad_f(self, x):
        """
        Evaluates the gradient of the quadratic function at a given binary vector.

        Parameters
        ----------
        x : list of int
            A binary vector for which to evaluate the gradient.

        Returns
        -------
        list of float
            The gradient vector.

        Example
        -------
        >>> Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
        >>> qp = QUBO(0, Qdict)
        >>> x = [1, 1]
        >>> qp.evaluate_grad_f(x)
        [1.5, -0.5]
        """
        grad = np.asarray([self.Qdict.get((i, i), 0) for i in range(self.n)])
        for i in range(self.n):
            for j in range(self.n):
                if j != i and x[j] == 1:
                    grad[i] += self.Qdict.get((i, j), 0) + self.Qdict.get((j, i), 0)
        return grad

    def tabu_search(self, max_iterations=100, tabu_size=10):
        """
        Solves the QUBO problem using the Tabu search algorithm.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations to run the Tabu search.
        tabu_size : int
            Size of the Tabu list.

        Returns
        -------
        best_solution : list of int
            The best binary vector found.
        best_obj_value : float
            The corresponding objective value.

        Example
        -------
        >>> Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
        >>> qp = QUBO(0, Qdict)
        >>> best_solution, best_obj_value = qp.tabu_search(50, 5)
        >>> best_solution
        [0, 1]
        >>> best_obj_value
        0.5
        """
        x = np.random.randint(2, size=self.n)  # Initial solution
        best_solution = x.copy()
        best_obj_value = self.evaluate_f(x)
        tabu_list = []

        for _ in range(max_iterations):
            neighbors = []
            for i in range(self.n):
                neighbor = x.copy()
                neighbor[i] = 1 - neighbor[i]  # Flip a bit
                neighbors.append((neighbor, self.evaluate_f(neighbor)))

            # Choose the best neighbor that is not tabu
            best_neighbor = min(neighbors, key=lambda x: x[1])
            best_neighbor_solution, best_neighbor_obj = best_neighbor

            # Update the current solution if it's better than the previous best and not tabu
            if (
                best_neighbor_obj < best_obj_value
                and best_neighbor_solution.tolist() not in tabu_list
            ):
                x = best_neighbor_solution
                best_solution = x.copy()
                best_obj_value = best_neighbor_obj

            # Add the best neighbor to the tabu list
            tabu_list.append(best_neighbor_solution.tolist())
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

        return best_solution, best_obj_value

    def brute_force(self):
        """
        Solves the QUBO problem by evaluating all possible binary vectors. Note that this approach is very slow.

        Returns
        -------
        opt_vector : list of int
            The optimal binary vector.
        min_value : float
            The minimum value of the objective function.

        Example
        -------
        >>> Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
        >>> qp = QUBO(0, Qdict)
        >>> opt_vector, min_value = qp.brute_force()
        >>> opt_vector
        [1, 0]
        >>> min_value
        -0.5
        """
        possible_values = {}
        # A list of all the possible permutations for x vector
        vec_permutations = itertools.product([0, 1], repeat=self.n)

        for permutation in vec_permutations:
            x = np.array(
                [[var] for var in permutation]
            )  # Converts the permutation into a column vector
            value = self.evaluate_f(x)
            possible_values[value] = (
                x  # Adds the value and its vector to the dictionary
            )

        min_value = min(
            possible_values.keys()
        )  # Lowest value of the objective function
        opt_vector = tuple(
            possible_values[min_value].T[0]
        )  # Optimum vector x that produces the lowest value

        return opt_vector, min_value

    def canonical_q(self):
        """
        We want to keep non-zero component when i < j.
        :return: a dictionary and also update Qdict
        """
        for i in range(self.n):
            for j in range(i, self.n):
                if (j, i) in self.Qdict:
                    if (i, j) in self.Qdict:
                        self.Qdict[(i,j)] += self.Qdict.pop((j,i))
                    else:
                        self.Qdict[(i,j)] = self.Qdict.pop((j,i))
                    self.Qdict.pop((j, i), None)
        return self.Qdict


class linear_problem:
    """
    A class used to represent a linear problem of the form Ax + b.

    Attributes
    ----------
    A : np.ndarray
        Coefficient matrix.
    b : np.ndarray
        Constant vector.
    n : int
        Dimension of the problem, inferred from the size of b.

    Methods
    -------
    multiply_scalar(scalar):
        Multiplies the matrix A and vector b by a scalar.

    __add__(other):
        Adds another linear_problem to the current one.

    evaluate_f(x):
        Evaluates the linear function at a given point x.

    square():
        Squares the linear problem, returning a quadratic problem.
    """

    def __init__(self, A, b):
        """
        Initializes the linear problem.

        Parameters
        ----------
        A : np.ndarray
            Coefficient matrix.
        b : np.ndarray
            Constant vector.

        Raises
        ------
        ValueError
            If A and b have incompatible dimensions.

        Examples
        --------
        >>> A = np.array([[1, 2], [3, 4]])
        >>> b = np.array([5, 6])
        >>> lp = linear_problem(A, b)
        """
        self.A = np.atleast_2d(A)
        self.b = np.array([b]) if np.isscalar(b) else np.asarray(b)
        self.n = self.A.shape[1]

    def multiply_scalar(self, scalar_multiplier):
        """
        Multiplies the matrix A and vector b by a scalar.

        Parameters
        ----------
        scalar : float
            The scalar value to multiply the matrix A and vector b.

        Examples
        --------
        >>> A = np.array([[1, 2], [3, 4]])
        >>> b = np.array([5, 6])
        >>> lp = linear_problem(A, b)
        >>> lp.multiply_scalar(2)
        >>> print(lp.A)
        [[2 4]
         [6 8]]
        >>> print(lp.b)
        [10 12]
        """
        self.A *= scalar_multiplier
        self.b *= scalar_multiplier

    def __add__(self, other_linear):
        """
        Adds another linear_problem to the current one.

        Parameters
        ----------
        other : linear_problem
            Another linear_problem to be added.

        Raises
        ------
        ValueError
            If the dimensions of the two linear problems do not match.

        Examples
        --------
        >>> A1 = np.array([[1, 2], [3, 4]])
        >>> b1 = np.array([5, 6])
        >>> lp1 = linear_problem(A1, b1)
        >>> A2 = np.array([[1, 1], [1, 1]])
        >>> b2 = np.array([1, 1])
        >>> lp2 = linear_problem(A2, b2)
        >>> lp1 + lp2
        >>> print(lp1.A)
        [[2 3]
         [4 5]]
        >>> print(lp1.b)
        [6 7]
        """
        self.A += other_linear.A
        self.b += other_linear.b
        return self

    def evaluate_f(self, x):
        """
        Evaluates the linear function Ax + b at a given point x.

        Parameters
        ----------
        x : np.ndarray
            Input vector at which to evaluate the linear function.

        Returns
        -------
        np.ndarray
            The value of the linear function Ax + b at the given x.

        Examples
        --------
        >>> A = np.array([[1, 2], [3, 4]])
        >>> b = np.array([5, 6])
        >>> lp = linear_problem(A, b)
        >>> x = np.array([1, 1])
        >>> result = lp.evaluate_f(x)
        >>> print(result)
        [ 8 13]
        """
        return self.A @ x + self.b

    def square(self):
        """
        Squares the linear problem to obtain a quadratic problem.

        Returns
        -------
        QUBO
            A quadratic problem corresponding to squaring the linear function.

        Examples
        --------
        >>> A = np.array([[1, 2], [3, 4]])
        >>> b = np.array([5, 6])
        >>> lp = linear_problem(A, b)
        >>> Quadratic = lp.square()
        >>> print(Quadratic.Qdict)
        {(0, 0): 56, (0, 1): 14, (1, 0): 14, (1, 1): 88}
        >>> print(Quadratic.offset)
        61
        """
        quadraticpart = self.A.T @ self.A + np.diag(2 * (self.b @ self.A))
        offset = np.dot(self.b, self.b)
        num_rows, num_cols = quadraticpart.shape
        Qdict = dict()
        for i in range(num_rows):
            for j in range(num_cols):
                Qdict[(i, j)] = quadraticpart[i, j]
        return QUBO(offset, Qdict)
