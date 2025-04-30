# Qibo Combinatorial Optimisation

Qibo Comb Optimisation is a Python library for formulating, analyzing, and solving combinatorial optimisation problems using both classical and quantum algorithms. It is built on top of [Qibo](https://qibo.science/), providing tools to construct Hamiltonians, quantum circuits, and workflows for problems such as QUBO, Ising, TSP, and MIS.

## Features
- **QUBO and Ising Model Support**: Define, manipulate, and convert between QUBO and Ising formulations.
- **Hamiltonian Construction**: Automatic symbolic Hamiltonian generation for quantum algorithms.
- **Quantum Algorithms**: Build and train QAOA circuits for combinatorial problems, with support for custom mixers and CVaR loss.
- **Classical Solvers**: Includes brute-force and Tabu search algorithms for benchmarking and validation.
- **Combinatorial Problem Classes**: Ready-to-use classes for TSP (Traveling Salesman Problem) and MIS (Maximum Independent Set), with penalty methods and constraint handling.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install the package and its dependencies:

```bash
poetry install
```

To install all optional dependencies (analysis, tests, docs):

```bash
poetry install --with analysis,tests,docs
```

## Usage Example

### QUBO Problem (Classical and Quantum)

```python
from qibo_comb_optimisation.optimisation_class.optimisation_class import QUBO

# Define a QUBO problem: f(x) = x0^2 + 0.5*x0*x1 - x1^2
Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
qp = QUBO(0, Qdict)

# Classical brute-force solution
opt_vector, min_value = qp.brute_force()
print(f"Optimal solution: {opt_vector}, value: {min_value}")

# Quantum QAOA circuit (requires Qibo backend)
gammas = [0.1, 0.2]
betas = [0.3, 0.4]
circuit = qp.qubo_to_qaoa_circuit(gammas, betas)
```

### Traveling Salesman Problem (TSP)

```python
from qibo_comb_optimisation.combinatorial_classes.combinatorial_classes import TSP
import numpy as np

distance_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
tsp = TSP(distance_matrix)
qubo = tsp.penalty_method(penalty=10.0)
# Now use qubo with QUBO methods or quantum algorithms
```

### Maximum Independent Set (MIS)

```python
import networkx as nx
from qibo_comb_optimisation.combinatorial_classes.combinatorial_classes import Mis

g = nx.cycle_graph(4)
mis = Mis(g)
qubo = mis.penalty_method(penalty=5.0)
```

## Supported Problems
- **QUBO**: Quadratic Unconstrained Binary Optimization
- **Ising Model**: Spin glass and related problems
- **TSP**: Traveling Salesman Problem
- **MIS**: Maximum Independent Set

## Development

- Source code: `src/qibo_comb_optimisation/`
- Tests: `tests/`
- Utilities: `src/utils/`
- Documentation: `doc/`



## Documentation

To build the documentation locally:

```bash
poetry install --with docs
poetry run poe docs
```

The HTML docs will be available in `doc/build/html/index.html`.

## Contributing

Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/qiboteam/qibo-comb-optimisation/).

## License

This project is licensed under the Apache-2.0 License.

