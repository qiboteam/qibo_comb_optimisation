from qibo import hamiltonians
from qibo.symbols import Z
import numpy as np
import itertools

class quadratic_problem:
    def __init__(self, offset, *args):
        """
        :param args: this can either be a dictionary or h and J for typical ising formulation
        :param offset: an integer

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
            offset (numeric, optional, default=0):
                Constant offset to be applied to the energy. Default 0.
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
            self.Qdict = {(v, v): 2. * bias for v, bias in h.items()}
            self.n = 0

            # next the optimization_class biases
            for (u, v), bias in self.Qdict.items():
                if bias == 0.0:
                    continue
                self.Qdict[(u, v)] = 4. * bias
                self.Qdict[(u, u)] = self.Qdict.setdefault((u, u), 0) - 2. * bias
                self.Qdict[(v, v)] = self.Qdict.setdefault((v, v), 0) - 2. * bias
                self.n = max([self.n, u, v])
            self.n += 1
            # finally adjust the offset based on QUBO definitions rather than Ising formulation
            self.offset += sum(args.values()) - sum(h.values())


    def multiply_scalar(self, scalar_multiplier):
        """
        :param scalar: this is the scalar that we want to multiply the optimization_class function to
        :return: None, just updating the optimization_class function
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

    def qubo_to_ising(self, constant = 0.0):
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
                    h[u] += bias/2
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



    def construct_symbolic_Hamiltonian_from_QUBO(self):
        """

        :param Q: Q, a matrix describing the QUBO
        :return: a symbolic hamiltonian that corresponds to the QUBO
        """
        h, J, constant = self.Qdict.qubo_to_ising(self.Qdict)
        symbolic_ham = sum(h[i]*Z[i] for i in h)
        symbolic_ham += sum(J[u,v] * Z[u]*Z[v] for (u,v) in J)
        symbolic_ham += constant
        ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
        return ham

    def evaluate_f(self, x):
        f_value = self.offset
        for i in range(self.n):
            if x[i] != 0:
                # manage diagonal term first
                if (i,i) in self.Qdict:
                    f_value += self.Qdict[(i,i)]
                    for j in range(i+1, self.n):
                        if x[j] != 0:
                            f_value += self.Qdict.get((i,j),0) + self.Qdict.get((j,i),0)
        return f_value


    def evaluate_grad_f(self, x):
        """
        :param x:
        :return: grad, the corresponding gradient
        """
        grad = np.asarray([self.Qdict.get((i,i),0) for i in range(self.n)])
        for i in range(self.n):
            for j in range(self.n):
                if j != i and x[j] == 1:
                    grad[i] += self.Qdict.get((i,j),0) + self.Qdict.get((j,i),0)
        return grad

    def tabu_search(self, max_iterations=100, tabu_size=10):
        """
        Perform tabu search to find a minimizer for a QUBO problem.

        Args:
        - max_iterations (int): Maximum number of iterations to perform.
        - tabu_size (int): Size of the tabu list.

        Returns:
        - numpy.array: Binary vector representing the minimizer found.
        - float: Objective value of the minimizer found.
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
            if best_neighbor_obj < best_obj_value and best_neighbor_solution.tolist() not in tabu_list:
                x = best_neighbor_solution
                best_solution = x.copy()
                best_obj_value = best_neighbor_obj

            # Add the best neighbor to the tabu list
            tabu_list.append(best_neighbor_solution.tolist())
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

        return best_solution, best_obj_value

    def brute_force(self):
        # this is to solve the QUBO in a brute force fashion for benchmarking purpose.
        # note that this approach is very slow.
        possible_values = {}
        # A list of all the possible permutations for x vector
        vec_permutations = itertools.product([0, 1], repeat=self.n)

        for permutation in vec_permutations:
            x = np.array(
                [[var] for var in permutation]
            )  # Converts the permutation into a column vector
            value = self.evaluate_f(x)
            possible_values[value] = x  # Adds the value and its vector to the dictionary

        min_value = min(possible_values.keys())  # Lowest value of the objective function
        opt_vector = tuple(
            possible_values[min_value].T[0]
        )  # Optimum vector x that produces the lowest value

        return opt_vector, min_value


class linear_problem:
    def __init__(self, A, b):
        """
        :param A: A is a numpy matrix, possibly a row vector
        """
        self.A = A
        self.b = b
        self.n = b.len


    def multiply_scalar(self, scalar_multiplier):
        """
        :param scalar: this is the scalar that we want to multiply the optimization_class function to
        :return: None, just updating the optimization_class function
        """
        self.A *=scalar_multiplier
        self.b *= scalar_multiplier

    def __add__(self, other_linear):
        """
        :param other_Quadratic: another optimization_class class object
        :return: updating the optimization_class function to obtain the sum
        """
        self.A += other_linear.A
        self.b += other_linear.b


    def evaluate_f(self, x):
        return self.A @ x + self.b


    def square(self):
        '''
        if we square a linear term, we should obtain a QUBO
        ||Ax+b||^2 = (Ax+b)^T(Ax+b) =x^TA^TAx =+ 2b^TAx + b^Tb
        =X^T(A^TA - diag(A^TA))x + (diag(A^TA)+2b^TA)x + b^Tb
        :return: a QUBO object
        '''
        quadraticpart = self.A.T @ self.A + np.diag(2 * (self.b@self.A))
        offset = np.dot(self.b, self.b)
        num_rows, num_cols = quadraticpart.shape
        Qdict = dict()
        for i in range(num_rows):
            for j in range(num_cols):
                Qdict[(i,j)] = quadraticpart[i,j]
        return quadraticpart(Qdict, offset)






