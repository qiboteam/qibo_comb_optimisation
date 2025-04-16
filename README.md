# qibo-comb-optimisation

![Tests](https://github.com/qiboteam/qibo_comb_optimisation/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibo_comb_optimisation/graph/badge.svg?token=2CMDZP1GU2)](https://codecov.io/gh/qiboteam/qibo_comb_optimisation)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10473173.svg)](https://doi.org/10.5281/zenodo.10473173) -->

qibo-comb-optimisation is a sub-repository of [Qibo](https://github.com/qiboteam/qibo) that focuses on methods to prepare Hamiltonians for common combinatorial optimisation problems to be run as a variational quantum algorithm.

## Documentation

The qibo-comb-optimisation documentation can be found [here](https://qibo.science/qibo_comb_optimisation/stable).

Install using `pip install qibo_comb_optimisation`.

## Minimum Working Examples

Import the necessary packages.
```python
import numpy as np
from qibo_comb_optimisation.optimisation_class.optimisation_class import linear_problem, QUBO
```

To use :ref:`Travelling Salesman Problem <TSP>`,
```python
from qibo_comb_optimisation.combinatorial_classes.combinatorial_classes import TSP as TSP

np.random.seed(42)
num_cities = 3
distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1],[0, 0.7, 0]])
distance_matrix = distance_matrix.round(1)
small_tsp = TSP(distance_matrix)
initial_parameters = np.random.uniform(0, 1, 2)
initial_state = small_tsp.prepare_initial_state([i for i in range(num_cities)])
qaoa_function_of_layer(2, distance_matrix)
```

To use :ref:`Maximum Independent Set <MIS>`,

To use :ref:`QUBO <QUBO>`,

For linear problems :`linear problems <LP>`

## Citation policy

If you use the qibo-comb-optimisation plugin please refer to [the documentation](https://qibo.science/qibo/stable/appendix/citing-qibo.html#publications) for citation instructions.

## Contact

To get in touch with the community and the developers, consider joining the Qibo workspace on Matrix:

[![Matrix](https://img.shields.io/matrix/qibo%3Amatrix.org?logo=matrix)](https://matrix.to/#/#qibo:matrix.org)

If you have a question about the project, contact us at [📫](mailto:qiboteam@qibo.science).

## Contributing

Contributions, issues and feature requests are welcome.




