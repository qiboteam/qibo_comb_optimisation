"""
Optimisation classes
"""

import inspect
import itertools
from collections import defaultdict
from enum import Enum
import numpy as np
from qibo import Circuit, gates, hamiltonians
from qibo.backends import _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models import QAOA
from qibo.optimizers import optimize
from qibo.symbols import Z
import warnings


class QUBO:
    """Initializes a ``QUBO`` class. The ``QUBO`` class can be multiplied by a scalar factor, and multiple ``QUBO``
    instances can be added together.

    Args:
        offset (float): Constant offset of the QUBO problem.
        args (dict): Input parameters for the QUBO or Ising formulation. If ``len(args) == 1``,
            ``args`` has to be a dictionary representing the quadratic coefficient assigned to the ``QUBO.QDict``
            attribute, which represents the :math:`Q` matrix. If ``len(args) == 2``, both objects have to be
            dictionaries representing the inputs :math:`h` and :math:`J` for an Ising formulation.

            We have the following relation:

            .. math::

                s'  J  s + h'  s = \\text{offset} + x'  Q x

            where:

            - ``h`` (dict): Linear biases as a dictionary of the form ``{v: bias, ...}``, where keys are variables
              of the model and values are biases.
            - ``J`` (dict): Quadratic biases as a dictionary of the form ``{(u, v): bias, ...}``, where keys are
              two-tuples of variables of the model and values are biases associated with the interaction between the
              pair of variables.

    Example:
        .. testcode::

            from qiboopt.opt_class.opt_class import QUBO

            Qdict1 = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
            qp1 = QUBO(0, Qdict1)
            print(qp1.Qdict)

        .. testoutput::

            {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}

        .. testcode::

            qp1 *= 2
            print(qp1.Qdict)

        .. testoutput::

            {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}

        .. testcode::

            Qdict2 = {(0, 0): 2.0, (1, 1): 1.0}
            qp2 = QUBO(1, Qdict2)
            qp3 = qp1 + qp2
            print(qp3.Qdict)

        .. testoutput::

            {(0, 0): 4.0, (0, 1): 1.0, (1, 1): -1.0}

        .. testcode::

            print(qp3.offset)

        .. testoutput::

            1

        .. testcode::

            h = {3: 1.0, 4: 0.82, 5: 0.23}
            J = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
            qp = QUBO(0, h, J)
            print(qp.Qdict)

        .. testoutput::

            {(3, 3): -2.0, (4, 4): -1.64, (5, 5): -0.46, (0, 1): 2.0, (0, 0): -1.0, (1, 1): -1.0}

    """

    def __init__(self, offset, *args):
        self.offset = offset
        # Check that all of *args are dictionaries
        if not all(isinstance(arg, dict) for arg in args):
            raise_error(
                TypeError, "args in a QUBO constructor can only be dictionaries."
            )
        if len(args) == 1:
            self.Qdict = args[0]
            self.h, self.J, self.ising_constant = self.qubo_to_ising()
        elif len(args) == 2:
            h = args[0]
            J = args[1]
            if not all(isinstance(k, int) for k in h.keys()):
                raise TypeError("All keys in the dictionary must be integers.")
            if not all(isinstance(u, int) and isinstance(v, int) for u, v in J.keys()):
                raise TypeError("All keys in J dictionary must be tuples of integers")
            self.h = h
            self.J = J
            self.Qdict = {(v, v): -2.0 * bias for v, bias in h.items()}

            # next the opt_class biases
            for (u, v), bias in self.J.items():
                if bias and u != v:
                    self.Qdict[(u, v)] = 4.0 * bias
                    self.Qdict[(u, u)] = self.Qdict.get((u, u), 0) - 2.0 * bias
                    self.Qdict[(v, v)] = self.Qdict.get((v, v), 0) - 2.0 * bias

            # finally adjust the offset based on QUBO definitions rather than Ising formulation
            self.offset += sum(J.values()) + sum(h.values())
        else:
            raise_error(
                NotImplementedError, "Invalid number of args in the QUBO constructor."
            )

        self.n = max(max(key) for key in self.Qdict) + 1 if self.Qdict else 0

        # Define other class attributes
        self.n_layers = None
        self.num_betas = None

    def __add__(self, other_quadratic):
        # Create a deep copy of the current QUBO's Qdict
        new_Qdict = self.Qdict.copy()

        # Add the other QUBO's coefficients
        for key, value in other_quadratic.Qdict.items():
            new_Qdict[key] = new_Qdict.get(key, 0.0) + value

        # Calculate the new offset
        new_offset = self.offset + other_quadratic.offset

        # Create and return a new QUBO object
        return self.__class__(new_offset, new_Qdict)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply QUBO by scalar (int or float)")

        new_Qdict = {key: value * scalar for key, value in self.Qdict.items()}
        new_offset = self.offset * scalar
        return self.__class__(new_offset, new_Qdict)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def _phase_separation(self, circuit, gamma):
        """
        Applies the phase separation layer (corresponding to the Ising model Hamiltonian).
        This step encodes the interaction terms into the quantum circuit.
        """
        # Apply R_z gates for diagonal terms (h_i)
        circuit.add(
            gates.RZ(i, -2 * gamma * self.h[i]) for i in range(self.n)
        )  # -2 * gamma * h_i

        # Apply CNOT and R_z for off-diagonal terms (J_ij)
        for i in range(self.n):
            for j in range(self.n):
                if (i, j) in self.J:
                    weight = self.J[(i, j)]
                    if weight:
                        circuit.add(gates.CNOT(i, j))
                        circuit.add(
                            gates.RZ(j, -2 * gamma * weight)
                        )  # -2 * gamma * J_ij
                        circuit.add(gates.CNOT(i, j))

    def _default_mixer(self, circuit, beta, alpha=None):
        """
        Applies the mixer layer (uniform superposition evolution).
        This step applies RX rotations on each qubit to spread the superposition.
        """
        for i in range(self.n):
            circuit.add(gates.RX(i, 2 * beta))  # Apply RX gates for mixer
            if alpha:
                circuit.add(gates.RY(i, 2 * alpha))

    def _build(
        self,
        gammas,
        betas,
        alphas=None,
        custom_mixer=None,
        include_measurements=True,
        density_matrix=False,
    ):
        """
        Constructs the full QAOA circuit for the Ising model with p layers.
        custom_mixer (List[:class:`qibo.models.Circuit`]): An optional function that takes as input custom mixers.
            If len(custom_mixer) == 1, then use this one circuit as mixer for all layers.
            If len(custom_mixer) == len(gammas), then use each circuit as mixer for each layer.
            If len(custom_mixer) != 1 and != len(gammas), raise an error.
        """
        p = len(gammas)

        # Apply initial Hadamard gates (uniform superposition)
        circuit = Circuit(self.n, density_matrix=density_matrix)
        circuit.add(gates.H(i) for i in range(self.n))

        for layer in range(p):
            self._phase_separation(
                circuit, gammas[layer]
            )  # Phase separation (Ising model encoding)
            if alphas is not None:
                self._default_mixer(circuit, betas[layer], alphas[layer])
            else:
                if custom_mixer:
                    if len(gammas) != len(betas):
                        raise_error(
                            ValueError, f"Input {len(gammas) = } != {len(betas) = }."
                        )

                    # Extract number of betas per layer
                    betas_per_layer = len(betas) // p
                    if (
                        custom_mixer[0](
                            betas[
                                layer * betas_per_layer : (layer + 1) * betas_per_layer
                            ]
                        ).density_matrix
                        != circuit.density_matrix
                    ):
                        raise_error(
                            ValueError,
                            f"Ensure density_matrix in custom_mixer is the same as density_matrix argument in QAOA circuit.",
                        )
                    if len(custom_mixer) == 1:
                        circuit += custom_mixer[0](
                            betas[
                                layer * betas_per_layer : (layer + 1) * betas_per_layer
                            ]
                        )
                    elif len(custom_mixer) == len(gammas):
                        circuit += custom_mixer[layer](
                            betas[
                                layer * betas_per_layer : (layer + 1) * betas_per_layer
                            ]
                        )
                else:
                    self._default_mixer(circuit, betas[layer])

        if include_measurements:
            circuit.add(gates.M(i) for i in range(self.n))

        return circuit

    def qubo_to_ising(self):
        """Convert a QUBO problem to an Ising problem.

        Maps a quadratic unconstrained binary optimisation (QUBO) problem defined over binary variables
        (:math:`\\{0, 1\\}`), where the linear term is contained along the diagonal of :math:`Q` (:math:`x' Qx`), to an
        Ising model defined on spin variables (:math:`\\{-1, +1\\}`). More specifically, returns the the :math:`h` and
        :math:`J` variables defining the Ising model as well as the constant value representing the offset in energy
        between the two problem formulations.

        .. math::

             x'  Q  x  = \\text{constant} + s'  J  s + h'  s

        Returns:
            (dict, dict, float): A 3-tuple containing: ``h``: the linear coefficients of the Ising problem, ``J``:
            the quadratic coefficients of the Ising problem, and constant: the new energy offset.
        """
        h = {}
        J = {}
        constant = self.offset

        for (u, v), bias in self.Qdict.items():
            if not bias:
                continue

            if u == v:
                # x_i = (1 - s_i)/2  =>  bias * x_i contributes:
                # constant += bias/2, h_i += -bias/2
                constant += bias / 2
                h[u] = h.get(u, 0.0) - bias / 2
            else:
                # x_u x_v = (1 - s_u - s_v + s_u s_v)/4
                constant += bias / 4
                h[u] = h.get(u, 0.0) - bias / 4
                h[v] = h.get(v, 0.0) - bias / 4
                J[u, v] = J.get((u, v), 0.0) + bias / 4

        return h, J, constant

    def construct_symbolic_Hamiltonian_from_QUBO(self):
        """Constructs a symbolic Hamiltonian from the QUBO problem by converting it to an Ising model.

        The method calls the qubo_to_ising function to convert the QUBO formulation into an Ising Hamiltonian with
        linear and quadratic terms, before creating a symbolic Hamiltonian using the main ``qibo`` library.

        Returns:
            :class:`qibo.hamiltonians.hamiltonians.SymbolicHamiltonian`: Hamiltonian corresponding to the QUBO problem
        """
        h, J, constant = self.qubo_to_ising()

        # Create a symbolic Hamiltonian using qibo symbols
        symbolic_ham = sum(h[i] * Z(i) for i in h)
        symbolic_ham += sum(J[u, v] * Z(u) * Z(v) for (u, v) in J)
        symbolic_ham += constant

        # Return the symbolic Hamiltonian using qibo's Hamiltonian object
        ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
        return ham

    def evaluate_f(self, x):
        """Evaluates the quadratic function for a given binary vector.

        Args:
            x (list): A list representing the binary vector for which to evaluate the function.

        Returns:
            float: Value of the given binary vector.

        Example:
            .. testcode::

                from qiboopt.opt_class.opt_class import QUBO


                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                x = [1, 1]
                print(qp.evaluate_f(x))

            .. testoutput::

                0.5
        """
        f_value = self.offset
        for i in range(self.n):
            if x[i]:
                f_value += self.Qdict.get((i, i), 0.0)  # manage diagonal term first
                f_value += sum(
                    self.Qdict.get((i, j), 0) + self.Qdict.get((j, i), 0)
                    for j in range(i + 1, self.n)
                    if x[j]
                )
        return f_value

    def evaluate_grad_f(self, x):
        """Evaluates the gradient of the quadratic function at a given binary vector.

        Args:
            x (List[int]): A list representing the binary vector for which to evaluate the gradient.

        Returns:
            list: List of floats representing the gradient vector.

        Example:
            .. testcode::

                from qiboopt.opt_class.opt_class import QUBO


                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                x = [1, 1]
                print(qp.evaluate_grad_f(x))

            .. testoutput::

                [ 1.5 -0.5]
        """
        grad = np.asarray([self.Qdict.get((i, i), 0) for i in range(self.n)])
        for i in range(self.n):
            for j in range(self.n):
                if j != i and x[j] == 1:
                    grad[i] += self.Qdict.get((i, j), 0) + self.Qdict.get((j, i), 0)
        return grad

    def tabu_search(self, max_iterations=100, tabu_size=10):
        """Solves the QUBO problem using the Tabu search algorithm.

        Args:
            max_iterations (int): Maximum number of iterations to run the Tabu search.
                Defaults to 100.
            tabu_size (int): Size of the Tabu list.

        Returns:
            (list, float): A list of integers representing the best binary vector found and its corresponding value

        Example:
            .. testcode::

                from qiboopt.opt_class.opt_class import QUBO


                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                best_solution, best_obj_value = qp.tabu_search(50, 5)
                print(best_solution)

            .. testoutput::

                [0 1]

            .. testcode::

                print(best_obj_value)

            .. testoutput::

                -1.0
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
        """Solves the QUBO problem by evaluating all possible binary vectors. Note that this approach is very slow.

        Returns:
            (list, float): A list of integers representing the optimal binary vector and its corresponding value

        Example:

            .. testcode::

                from qiboopt.opt_class.opt_class import QUBO


                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                opt_vector, min_value = qp.brute_force()
                print(opt_vector)

            .. testoutput::

                (0, 1)

            .. testcode::

                print(min_value)

            .. testoutput::

                -1.0
        """
        opt_vector = min(itertools.product([0, 1], repeat=self.n), key=self.evaluate_f)
        return opt_vector, self.evaluate_f(opt_vector)

    def canonical_q(self):
        """Converts the ``Qdict`` attribute (QUBO matrix) to canonical form whereby only terms with ``i < j``
        are retained.

        Returns:
            dict: Updated QUBO matrix
        """
        Qdict = {
            (i, j): self.Qdict.get((i, j), 0) + self.Qdict.get((j, i), 0)
            for i in range(self.n)
            for j in range(i, self.n)
            if (i, j) in self.Qdict or (j, i) in self.Qdict
        }
        self.Qdict = Qdict
        return self.Qdict

    # ---------- Deprecated: delegates to UnifiedQAOA internally ---------- #

    def qubo_to_qaoa_circuit(
        self,
        gammas,
        betas,
        alphas=None,
        custom_mixer=None,
        include_measurements=True,
        density_matrix=False,
    ):
        """
        Constructs a QAOA or XQAOA circuit for the given QUBO problem.

        Args:
            gammas (List[float]): parameters for phasers
            betas (List[float]): parameters for X mixers
            alphas (List[float], optional): parameters for Y mixers for XQAOA
            custom_mixer (List[:class:`qibo.models.Circuit`]): optional argument that takes as input custom mixers.
                If len(custom_mixer) == 1, then use this one circuit as mixer for all layers.
                If len(custom_mixer) == len(gammas), then use each circuit as mixer for each layer.
                If len(custom_mixer) != 1 and != len(gammas), raise an error.
            include_measurements (bool, optional): If ``True``, append measurement gates to all qubits.
                Defaults to ``True``.
            density_matrix (bool): Enables `density_matrix` argument when constructing QAOA circuits to allow
                :class:`qibo.noise.NoiseModel` to be added to the circuit. Defaults to ``False``.

        Returns:
            :class:`qibo.models.Circuit`: The QAOA or XQAOA circuit corresponding to the QUBO problem.
        """
        # Preserve exact legacy behaviour

        if alphas is not None:  # Use XQAOA, ignore mixer_function
            circuit = self._build(
                gammas,
                betas,
                alphas,
                include_measurements=include_measurements,
                density_matrix=density_matrix,
            )
        else:
            if custom_mixer:
                circuit = self._build(
                    gammas,
                    betas,
                    alphas=None,
                    custom_mixer=custom_mixer,
                    include_measurements=include_measurements,
                    density_matrix=density_matrix,
                )
            else:
                circuit = self._build(
                    gammas,
                    betas,
                    include_measurements=include_measurements,
                    density_matrix=density_matrix,
                )
        return circuit

    def _split_qaoa_parameters(self, parameters, p, has_alphas=False):
        """Unpack a flat QAOA parameter vector in block format."""
        gammas = parameters[:p]
        betas = parameters[p : 2 * p]
        unpacked_alphas = parameters[2 * p : 3 * p] if has_alphas else None
        return gammas, betas, unpacked_alphas

    def qaoa_circuit_from_parameters(
        self,
        parameters,
        p,
        custom_mixer=None,
        include_measurements=True,
        has_alphas=False,
        density_matrix=False,
    ):
        """Build a QAOA circuit directly from flat block-ordered parameters."""
        gammas, betas, unpacked_alphas = self._split_qaoa_parameters(
            parameters, p, has_alphas=has_alphas
        )
        return self.qubo_to_qaoa_circuit(
            gammas=gammas,
            betas=betas,
            alphas=unpacked_alphas,
            custom_mixer=custom_mixer,
            include_measurements=include_measurements,
            density_matrix=density_matrix,
        )

    def make_qaoa_circuit_callable(
        self,
        p,
        custom_mixer=None,
        has_alphas=False,
        include_measurements=False,
        density_matrix=False,
    ):
        """Create a fixed-arity callable for qiboml circuit tracing."""
        nparams = 3 * p if has_alphas else 2 * p
        param_names = [f"theta_{i}" for i in range(nparams)]
        signature = inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in param_names
            ]
        )

        def qaoa_circuit(*angles):
            return self.qaoa_circuit_from_parameters(
                parameters=angles,
                p=p,
                custom_mixer=custom_mixer,
                include_measurements=include_measurements,
                has_alphas=has_alphas,
                density_matrix=density_matrix,
            )

        qaoa_circuit.__signature__ = signature
        return qaoa_circuit

    def train_QAOA(
            self,
            gammas=None,
            betas=None,
            alphas=None,
            p=None,
            nshots=int(1e3),
            regular_loss=True,
            maxiter=10,
            method="cobyla",
            cvar_delta=0.25,
            custom_mixer=None,
            density_matrix=False,
            backend=None,
            noise_model=None,
            engine="legacy",
            optimizer="adam",
            lr=0.05,
            epochs=100,
            differentiation=None,
    ):
        """Train QAOA.

        .. deprecated::
            Use ``qubo.to_unified_qaoa().train(...)`` instead.

        Returns the same tuple as before for full backward compatibility.
        """
        # Determine variant from alphas
        variant = "xqaoa" if alphas is not None else "standard"
        mixer_type = "xy" if alphas is not None else None

        uqaoa = self.to_unified_qaoa(
            variant=variant,
            mixer_type=mixer_type,
            custom_mixer=custom_mixer,
        )
        return uqaoa.train(
            gammas=gammas,
            betas=betas,
            alphas=alphas,
            p=p,
            nshots=nshots,
            regular_loss=regular_loss,
            maxiter=maxiter,
            method=method,
            cvar_delta=cvar_delta,
            density_matrix=density_matrix,
            backend=backend,
            noise_model=noise_model,
            engine=engine,
            optimizer=optimizer,
            lr=lr,
            epochs=epochs,
            differentiation=differentiation,
        )

    def qubo_to_qaoa_object(self, params: list = None):
        """
        Generates a QAOA object for the QUBO problem.

        Args:
            params (List[float]): Parameters of the QAOA given in block format:
                e.g. [all_gammas, all_betas, all_alphas] (if alphas is not None)
        Returns:
            `qibo.models.QAOA`: QAOA circuit for the QUBO problem.
        """

        # Convert QUBO to Ising Hamiltonian
        h, J, _constant = self.qubo_to_ising()

        # Create the Ising Hamiltonian using Qibo
        symbolic_ham = sum(h[i] * Z(i) for i in h)
        symbolic_ham += sum(value * Z(u) * Z(v) for (u, v), value in J.items())
        symbolic_ham += _constant

        # Define the QAOA model
        hamiltonian = SymbolicHamiltonian(symbolic_ham)
        qaoa = QAOA(hamiltonian)

        # Optionally set parameters
        if params is not None:
            qaoa.set_parameters(np.array(params))
        return qaoa

    def to_unified_qaoa(self, variant="standard", **kwargs):
        """Create a :class:`UnifiedQAOA` from this QUBO.

        Args:
            variant (str): QAOA variant (``"standard"``, ``"xqaoa"``,
                ``"lr"``, ``"ma"``).
            **kwargs: Forwarded to :class:`UnifiedQAOA`.

        Returns:
            :class:`UnifiedQAOA`
        """
        return UnifiedQAOA(self, variant=variant, **kwargs)


class MixerType(Enum):
    """XQAOA mixer variant types."""

    XY = "xy"  # Full XQAOA with RX + RY
    X_EQUALS_Y = "x_equals_y"  # Constrained: alpha = beta
    Y = "y"  # Pure Y-rotation: beta = 0
    X = "x"  # Pure X-rotation: alpha = 0


class ParameterType(Enum):
    """MA-QAOA parameter type."""

    PER_QUBIT = "per_qubit"
    PER_EDGE = "per_edge"


class UnifiedQAOA:
    """Unified QAOA implementation supporting Standard, XQAOA, LR-QAOA, and MA-QAOA.

    Takes a :class:`QUBO` instance and builds the corresponding QAOA circuit for
    the chosen variant.

    Args:
        qubo (:class:`QUBO`): The QUBO problem instance.
        variant (str): One of ``"standard"``, ``"xqaoa"``, ``"lr"``, ``"ma"``.
        mixer_type (str, optional): For ``variant="xqaoa"``: ``"xy"``, ``"x_equals_y"``,
            ``"y"``, or ``"x"``.
        lr_variant (str, optional): For ``variant="lr"``: ``"standard"`` or ``"xqaoa"``.
        ma_parameter_type (str, optional): For ``variant="ma"``: ``"per_qubit"`` or
            ``"per_edge"``.
        graph (optional): Graph with an ``edges`` attribute. Required for
            ``ma_parameter_type="per_edge"``.
        initial_state (:class:`qibo.models.Circuit`, optional): Circuit that prepares
            the initial state.  When ``None`` (default), Hadamard gates on every qubit
            are used.
        custom_mixer: An optional callable or list of :class:`qibo.models.Circuit`.
            If a single-element list, the same mixer is reused for every layer.
            If its length equals the depth, each element is used for the corresponding
            layer.

    Example:
        .. testcode::

            from qiboopt.opt_class.opt_class import QUBO, UnifiedQAOA
            import numpy as np

            Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
            qp = QUBO(0, Qdict)

            qaoa = UnifiedQAOA(qp, variant="standard")
            params = np.random.rand(qaoa.get_param_count(depth=3)) * 2 * np.pi
            circuit = qaoa.build_circuit(params, depth=3)
    """

    def __init__(
        self,
        qubo,
        variant="standard",
        mixer_type=None,
        lr_variant=None,
        ma_parameter_type=None,
        graph=None,
        initial_state=None,
        custom_mixer=None,
    ):
        if not isinstance(qubo, QUBO):
            raise_error(TypeError, "qubo must be an instance of QUBO.")

        self.qubo = qubo
        self.n = qubo.n
        self.variant = variant.lower()
        self.graph = graph
        self.initial_state = initial_state
        self.custom_mixer = custom_mixer

        if self.variant == "xqaoa":
            self.mixer_type = MixerType(mixer_type or "xy")
        elif self.variant == "lr":
            self.lr_variant = lr_variant or "standard"
        elif self.variant == "ma":
            self.ma_parameter_type = ParameterType(ma_parameter_type or "per_qubit")
        elif self.variant != "standard":
            raise_error(
                ValueError,
                f"Unknown variant '{self.variant}'. "
                "Choose from 'standard', 'xqaoa', 'lr', 'ma'.",
            )

    # ------------------------------------------------------------------ #
    #  Parameter helpers                                                  #
    # ------------------------------------------------------------------ #

    def train(
            self,
            gammas=None,
            betas=None,
            alphas=None,
            p=None,
            nshots=int(1e3),
            regular_loss=True,
            maxiter=10,
            method="cobyla",
            cvar_delta=0.25,
            density_matrix=False,
            backend=None,
            noise_model=None,
            engine="legacy",
            optimizer="adam",
            lr=0.05,
            epochs=100,
            differentiation=None,
    ):
        backend = _check_backend(backend)
        use_exact = (nshots is None) or (nshots == 0)

        if p is None and gammas is None:
            raise_error(
                ValueError,
                "Either p or gammas must be provided to define the number of layers.",
            )
        elif p is None:
            p = len(gammas)
        elif gammas is None:
            gammas = np.random.rand(p) * 2 * np.pi
            betas = np.random.rand(p) * 2 * np.pi
        else:
            if len(gammas) != p:
                raise_error(
                    ValueError,
                    f"gammas must be of length {p}, but got {len(gammas)}.",
                )

        if betas is None:
            raise_error(ValueError, "betas must be provided when gammas are given.")

        has_alphas = alphas is not None
        parameters = list(gammas) + list(betas)
        if has_alphas:
            parameters += list(alphas)

        if engine not in ("legacy", "qiboml"):
            raise_error(
                ValueError,
                f"Unsupported engine '{engine}'. Use 'legacy' or 'qiboml'.",
            )

        if not regular_loss and not (0 < cvar_delta <= 1):
            raise_error(
                ValueError,
                f"cvar_delta must satisfy 0 < cvar_delta <= 1, but got {cvar_delta}.",
            )

        if engine == "qiboml" and not regular_loss:
            warnings.warn(
                "engine='qiboml' does not yet support CVaR loss (regular_loss=False). "
                "Falling back to engine='legacy'.",
                UserWarning,
                stacklevel=2,
            )
            engine = "legacy"

        def _probability_dict_from_state(result):
            probabilities = np.asarray(result.probabilities()).ravel()
            return {
                format(index, f"0{self.n}b"): float(probability)
                for index, probability in enumerate(probabilities)
                if probability > 0
            }

        if use_exact:
            _hamiltonian = self.qubo.qubo_to_qaoa_object().hamiltonian

        def _circuit_from_flat(params, include_measurements):
            return self.qubo.qaoa_circuit_from_parameters(
                parameters=params,
                p=p,
                custom_mixer=self.custom_mixer,
                include_measurements=include_measurements,
                has_alphas=has_alphas,
                density_matrix=density_matrix,
            )

        if regular_loss:

            def myloss(parameters):
                circuit = _circuit_from_flat(parameters, include_measurements=not use_exact)
                if noise_model is not None:
                    if not density_matrix:
                        raise_error(ValueError, "noise_model requires density_matrix=True.")
                    circuit = noise_model.apply(circuit)

                if use_exact:
                    return _hamiltonian.expectation(circuit, nshots=None)

                result = backend.execute_circuit(circuit, nshots=nshots)
                result_counter = result.frequencies(binary=True)
                energy_dict = defaultdict(int)
                for key in result_counter:
                    x = [int(sub_key) for sub_key in key]
                    energy_dict[self.qubo.evaluate_f(x)] += result_counter[key]
                return sum(key * energy_dict[key] / nshots for key in energy_dict)

        else:

            def myloss(parameters, delta=cvar_delta):
                circuit = _circuit_from_flat(parameters, include_measurements=not use_exact)
                if noise_model is not None:
                    if not density_matrix:
                        raise_error(ValueError, "noise_model requires density_matrix=True.")
                    circuit = noise_model.apply(circuit)

                if use_exact:
                    result = backend.execute_circuit(circuit)
                    result_probs = _probability_dict_from_state(result)
                    energy_probs = defaultdict(float)
                    for key, probability in result_probs.items():
                        x = [int(sub_key) for sub_key in key]
                        energy_probs[self.qubo.evaluate_f(x)] += probability
                else:
                    result = backend.execute_circuit(circuit, nshots=nshots)
                    result_counter = result.frequencies(binary=True)
                    energy_dict = defaultdict(int)
                    for key in result_counter:
                        x = [int(sub_key) for sub_key in key]
                        energy_dict[self.qubo.evaluate_f(x)] += result_counter[key]
                    total_counts = sum(energy_dict.values())
                    energy_probs = {
                        key: value / total_counts for key, value in energy_dict.items()
                    }

                sorted_energies = sorted(energy_probs.items())
                cumulative_prob = 0
                selected_energies = []

                for energy, prob in sorted_energies:
                    if cumulative_prob + prob > delta:
                        excess_prob = delta - cumulative_prob
                        selected_energies.append((energy, excess_prob))
                        cumulative_prob = delta
                        break
                    selected_energies.append((energy, prob))
                    cumulative_prob += prob

                return sum(energy * prob for energy, prob in selected_energies) / delta

        if engine == "qiboml":
            from qiboopt.integrations.qiboml_adapter import optimize_qaoa_with_qiboml

            best, params, extra = optimize_qaoa_with_qiboml(
                qubo=self.qubo,
                parameters=parameters,
                p=p,
                nshots=nshots,
                noise_model=noise_model,
                custom_mixer=self.custom_mixer,
                has_alphas=has_alphas,
                optimizer=optimizer,
                lr=lr,
                epochs=epochs,
                differentiation=differentiation,
                backend=backend,
                density_matrix=density_matrix,
            )
        else:
            best, params, extra = optimize(
                myloss, parameters, method=method, options={"maxiter": maxiter}
            )

        circuit = _circuit_from_flat(params, include_measurements=not use_exact)
        original_circuit = Circuit.copy(circuit)

        if noise_model is not None:
            circuit = noise_model.apply(circuit)

        if use_exact:
            result = backend.execute_circuit(circuit)
            statistics = _probability_dict_from_state(result)
        else:
            result = backend.execute_circuit(circuit, nshots=nshots)
            statistics = result.frequencies(binary=True)

        if noise_model is not None:
            return best, params, extra, circuit, statistics, original_circuit
        return best, params, extra, circuit, statistics

    def get_param_count(self, depth):
        """Return the number of optimisation parameters for a given *depth*.

        Args:
            depth (int): Number of QAOA layers.

        Returns:
            int: Total parameter count.
        """
        if self.variant == "standard":
            return 2 * depth

        if self.variant == "xqaoa":
            return 3 * depth if self.mixer_type == MixerType.XY else 2 * depth

        if self.variant == "lr":
            return 3 if self.lr_variant == "xqaoa" else 2

        if self.variant == "ma":
            if self.ma_parameter_type == ParameterType.PER_QUBIT:
                return depth * (1 + self.n)
            if self.ma_parameter_type == ParameterType.PER_EDGE:
                if self.graph is None:
                    raise_error(ValueError, "Graph required for per-edge MA-QAOA.")
                n_edges = (
                    len(self.graph.edges)
                    if hasattr(self.graph, "edges")
                    else len(self.graph)
                )
                return depth * (1 + n_edges)

    def unpack_parameters(self, flat_params, depth):
        """Convert a flat parameter vector to a variant-specific structure.

        Uses block ordering to stay consistent with the existing QUBO API:
        - standard: [gammas..., betas...]
        - xqaoa XY: [gammas..., betas..., alphas...]
        - xqaoa X=Y: [gammas..., thetas...]
        - xqaoa Y: [gammas..., alphas...]
        - xqaoa X: [gammas..., betas...]
        - lr: compact max-parameter form, then linearly ramped
        - ma: per-layer blocks
        """
        flat_params = np.asarray(flat_params, dtype=float)
        expected = self.get_param_count(depth)
        if len(flat_params) != expected:
            raise_error(
                ValueError,
                f"Expected {expected} parameters, got {len(flat_params)}.",
            )

        param_dict = {}

        if self.variant == "standard":
            param_dict["gammas"] = flat_params[:depth]
            param_dict["betas"] = flat_params[depth: 2 * depth]

        elif self.variant == "xqaoa":
            if self.mixer_type == MixerType.XY:
                param_dict["gammas"] = flat_params[:depth]
                param_dict["betas"] = flat_params[depth: 2 * depth]
                param_dict["alphas"] = flat_params[2 * depth: 3 * depth]

            elif self.mixer_type == MixerType.X_EQUALS_Y:
                param_dict["gammas"] = flat_params[:depth]
                thetas = flat_params[depth: 2 * depth]
                param_dict["betas"] = thetas
                param_dict["alphas"] = thetas.copy()

            elif self.mixer_type == MixerType.Y:
                param_dict["gammas"] = flat_params[:depth]
                param_dict["alphas"] = flat_params[depth: 2 * depth]
                param_dict["betas"] = np.zeros(depth)

            elif self.mixer_type == MixerType.X:
                param_dict["gammas"] = flat_params[:depth]
                param_dict["betas"] = flat_params[depth: 2 * depth]
                param_dict["alphas"] = np.zeros(depth)

        elif self.variant == "lr":
            ramp = np.arange(1, depth + 1, dtype=float) / depth
            if self.lr_variant == "xqaoa":
                gamma_max, beta_max, alpha_max = flat_params
                param_dict["gammas"] = gamma_max * ramp
                param_dict["betas"] = beta_max * ramp
                param_dict["alphas"] = alpha_max * ramp
            else:
                gamma_max, beta_max = flat_params
                param_dict["gammas"] = gamma_max * ramp
                param_dict["betas"] = beta_max * ramp

        elif self.variant == "ma":
            if self.ma_parameter_type == ParameterType.PER_QUBIT:
                params_per_layer = 1 + self.n
                gammas = []
                betas = []
                for layer in range(depth):
                    start = layer * params_per_layer
                    gammas.append(flat_params[start])
                    betas.append(flat_params[start + 1: start + params_per_layer])
                param_dict["gammas"] = np.array(gammas)
                param_dict["betas"] = np.array(betas)

            elif self.ma_parameter_type == ParameterType.PER_EDGE:
                n_edges = (
                    len(self.graph.edges)
                    if hasattr(self.graph, "edges")
                    else len(self.graph)
                )
                params_per_layer = 1 + n_edges
                gammas = []
                betas = []
                for layer in range(depth):
                    start = layer * params_per_layer
                    gammas.append(flat_params[start])
                    betas.append(flat_params[start + 1: start + params_per_layer])
                param_dict["gammas"] = np.array(gammas)
                param_dict["betas"] = np.array(betas)

        return param_dict


    def _apply_initial_state(self, circuit):
        """Prepend the initial-state preparation to *circuit*."""
        if self.initial_state is not None:
            circuit += self.initial_state
        else:
            circuit.add(gates.H(i) for i in range(self.n))

    def _apply_phase_separation(self, circuit, gamma):
        """Delegate to the QUBO's phase-separation routine."""
        self.qubo._phase_separation(circuit, gamma)

    def _apply_mixer(self, circuit, layer, param_dict, depth):
        """Apply the mixer layer for the current *layer*.

        Handles custom mixers, MA-QAOA per-qubit betas, and the standard /
        XQAOA default mixer.
        """
        # --- custom mixer takes priority (except for MA-QAOA) ---
        if self.custom_mixer is not None and self.variant != "ma":
            betas = param_dict["betas"]
            if len(self.custom_mixer) == 1:
                mixer_circuit = self.custom_mixer[0]
            elif len(self.custom_mixer) == depth:
                mixer_circuit = self.custom_mixer[layer]
            else:
                raise_error(
                    ValueError,
                    f"custom_mixer length must be 1 or {depth}, "
                    f"got {len(self.custom_mixer)}.",
                )
            # If the mixer is callable (takes betas), call it
            if callable(mixer_circuit):
                circuit += mixer_circuit(betas[layer : layer + 1])
            else:
                circuit += mixer_circuit
            return

        # --- MA-QAOA: per-qubit (or per-edge) betas ---
        if self.variant == "ma":
            if self.ma_parameter_type == ParameterType.PER_QUBIT:
                layer_betas = param_dict["betas"][layer]  # shape (n_qubits,)
                for i in range(self.n):
                    circuit.add(gates.RX(i, 2 * layer_betas[i]))
            elif self.ma_parameter_type == ParameterType.PER_EDGE:
                # Apply RX with per-edge angle on the target qubit of each edge
                layer_betas = param_dict["betas"][layer]
                edge_list = (
                    list(self.graph.edges)
                    if hasattr(self.graph, "edges")
                    else list(self.graph)
                )
                for idx, (u, v) in enumerate(edge_list):
                    circuit.add(gates.RX(v, 2 * layer_betas[idx]))
            return

        # --- Standard / XQAOA / LR default mixer ---
        beta = param_dict["betas"][layer]
        alpha = param_dict.get("alphas")
        alpha_val = alpha[layer] if alpha is not None else None

        for i in range(self.n):
            circuit.add(gates.RX(i, 2 * beta))
            if alpha_val is not None and alpha_val != 0.0:
                circuit.add(gates.RY(i, 2 * alpha_val))

    def build_circuit(
        self,
        parameters,
        depth,
        include_measurements=True,
        density_matrix=False,
    ):
        """Build the full QAOA circuit for the chosen variant.

        Args:
            parameters (array-like): Flat parameter vector whose length must
                match :meth:`get_param_count(depth)`.
            depth (int): Number of QAOA layers.
            include_measurements (bool): Append measurement gates. Defaults
                to ``True``.
            density_matrix (bool): Use density-matrix simulation mode.
                Defaults to ``False``.

        Returns:
            :class:`qibo.models.Circuit`: The constructed circuit.
        """
        param_dict = self.unpack_parameters(parameters, depth)

        circuit = Circuit(self.n, density_matrix=density_matrix)

        # Initial state
        self._apply_initial_state(circuit)

        # QAOA layers
        for layer in range(depth):
            self._apply_phase_separation(circuit, param_dict["gammas"][layer])
            self._apply_mixer(circuit, layer, param_dict, depth)

        if include_measurements:
            circuit.add(gates.M(i) for i in range(self.n))

        return circuit


    def random_parameters(self, depth, seed=None):
        """Sample a random parameter vector uniformly in :math:`[0, 2\\pi)`.
        Args:
            depth (int): Number of QAOA layers.
            seed (int, optional): Random seed.
        Returns:
            numpy.ndarray: Random parameter vector.
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(0, 2 * np.pi, size=self.get_param_count(depth))

    def summary(self, depth):
        """Return a human-readable summary of the configuration.

        Args:
            depth (int): Circuit depth.

        Returns:
            str
        """
        lines = [
            "UnifiedQAOA Configuration",
            "=" * 50,
            f"Variant           : {self.variant}",
            f"Number of qubits  : {self.n}",
            f"Circuit depth     : {depth}",
            f"Total parameters  : {self.get_param_count(depth)}",
            f"Custom initial st.: {self.initial_state is not None}",
            f"Custom mixer      : {self.custom_mixer is not None}",
        ]
        if self.variant == "xqaoa":
            lines.append(f"Mixer type        : {self.mixer_type.value}")
        elif self.variant == "lr":
            lines.append(f"LR base variant   : {self.lr_variant}")
            lines.append("Note: parameter count is independent of depth!")
        elif self.variant == "ma":
            lines.append(f"MA parameter type : {self.ma_parameter_type.value}")
            if self.ma_parameter_type == ParameterType.PER_EDGE:
                n_edges = (
                    len(self.graph.edges)
                    if hasattr(self.graph, "edges")
                    else len(self.graph)
                )
                lines.append(f"Number of edges   : {n_edges}")
        return "\n".join(lines)


class LinearProblem:
    """Initializes a ``LinearProblem`` class, which represents a linear problem of the form :math:`Ax + b`. The
    ``LinearProblem`` class can be multiplied by a scalar factor, and multiple ``LinearProblem`` instances can be added
    together.

    Args:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Constant vector.

    Example:
        .. testcode::

            import numpy as np
            from qiboopt.opt_class.opt_class import LinearProblem


            A1 = np.array([[1, 2], [3, 4]])
            b1 = np.array([5, 6])
            lp1 = LinearProblem(A1, b1)
            lp1 *= 2
            A2 = np.array([[1, 1], [1, 1]])
            b2 = np.array([1, 1])
            lp2 = LinearProblem(A2, b2)
            lp3 = lp1 + lp2
            print(lp3.A)

        .. testoutput::

            [[3 5]
             [7 9]]

        .. testcode::

            print(lp3.b)

        .. testoutput::

            [11 13]
    """

    def __init__(self, A, b):
        # TODO: raise ValueError if A and b have incompatible dimensions.
        self.A = np.atleast_2d(A)
        self.b = np.array([b]) if np.isscalar(b) else np.asarray(b)
        self.n = self.A.shape[1]

    def __add__(self, other_linear):
        new_A = self.A + other_linear.A
        new_b = self.b + other_linear.b
        return self.__class__(new_A, new_b)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply LinearProblem by scalars (int or float)")

        new_A = self.A * scalar
        new_b = self.b * scalar
        return self.__class__(new_A, new_b)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def evaluate_f(self, x):
        """Evaluates the linear function :math:`Ax + b` at a given point :math:`x`.

        Args:
            x (np.ndarray): Input vector at which to evaluate the linear function.

        Example:
            .. testcode::

                import numpy as np
                from qiboopt.opt_class.opt_class import LinearProblem


                A = np.array([[1, 2], [3, 4]])
                b = np.array([5, 6])
                lp = LinearProblem(A, b)
                x = np.array([1, 1])
                result = lp.evaluate_f(x)
                print(result)

            .. testoutput::

                [ 8 13]

        Returns:
            numpy.ndarray: The value of the linear function :math:`Ax + b` at :math:`x`.
        """
        return self.A @ x + self.b

    def square(self):
        """Squares the linear problem to obtain a quadratic problem.
        Returns:
            :class:`qiboopt.opt_class.opt_class.QUBO`: Quadratic problem corresponding to squaring the linear function.

        Example:
            .. testcode::

                import numpy as np
                from qiboopt.opt_class.opt_class import LinearProblem

                A = np.array([[1, 2], [3, 4]])
                b = np.array([5, 6])
                lp = LinearProblem(A, b)
                Quadratic = lp.square()
                print(Quadratic.Qdict)

            .. testoutput::

                {(0, 0): 56, (0, 1): 14, (1, 0): 14, (1, 1): 88}

            .. testcode::

                print(Quadratic.offset)

            .. testoutput::

                61
        """
        quadratic_part = self.A.T @ self.A + np.diag(2 * (self.b @ self.A))
        offset = np.dot(self.b, self.b)
        num_rows, num_cols = quadratic_part.shape
        Qdict = {
            (i, j): quadratic_part[i, j].item()
            for i in range(num_rows)
            for j in range(num_cols)
        }
        return QUBO(offset, Qdict)


def variable_to_ind(variable_list):
    """
    given a list of variable, returns a dictionary from the variables to integers and also
    a dictionary to map the integer to the variables
    Args: a list of objects, typically strings
    returns:
    Two dictionaries, one map from the variable to the indices and it performs the reverse
    Example:
    .. testcode::
       variables = ["x1", "x2", "x3"]
       v2i, i2v = variable_to_ind(variables)

        print(v2i)
        # >>> {'x1': 0, 'x2': 1, 'x3': 2}
        print(i2v)
        # >>> {0: 'x1', 1: 'x2', 2: 'x3'}
    """
    var_to_idx = {var: i for i, var in enumerate(variable_list)}
    idx_to_var = {i: var for i, var in enumerate(variable_list)}
    return var_to_idx, idx_to_var


def variable_dict_to_ind_dict(variable_dict, var_to_idx):
    """
    This functions take in a dictionary that maps from (variable, variable) to float
    or variable to float convert the dictionary key to the corresponding indices
    Args:
        variable_dict: dictionary
        var_to_idx: a mapping from the variable to an index
    Returns:
        a dictionary that maps from indices to
    Example:
    .. testcode::
    var_to_idx = {'x1': 0, 'x2': 1, 'x3': 2}

    variable_dict = {
        ('x1', 'x2'): 1.5,
        ('x2', 'x3'): -0.5,
        'x3': 2.0
    }

    ind_dict = variable_dict_to_ind_dict(variable_dict, var_to_idx)
    print(ind_dict)
    # >>> {(0, 1): 1.5, (1, 2): -0.5, 2: 2.0}
    """
    ind_dict = {}
    for key, value in variable_dict.items():
        ind_key = None
        if key in var_to_idx:
            ind_key = var_to_idx[key]
        elif isinstance(key, tuple):
            ind_key = tuple(var_to_idx[var] for var in key)
        if ind_key is not None:
            ind_dict[ind_key] = value
    return ind_dict
