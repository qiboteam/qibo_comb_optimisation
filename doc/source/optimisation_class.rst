.. _optimisation_class:

Optimisation class
------------------

This module contains classes for formulating and solving QUBO and linear problems.

.. _QUBO:

QUBO
^^^^

QUBO, short for Quadratic Unconstrained Binary Optimisation, are a class of problems which are NP-complete.

When formulated carefully, QUBO problems can be mapped to solve a host of optimisation problems such as :ref:`Travelling Salesman Problem <TSP>`, :ref:`Maximum Independent Set <MIS>`, Quadratic Assignment Problem, Maximum Clique problem, Maximum Cut problem, etc.


.. autoclass:: qibo_comb_optimisation.optimisation_class.optimisation_class.QUBO
    :members:
    :member-order: bysource


.. _LP:

Linear Problems
^^^^^^^^^^^^^^^

Linear problem write up goes here.

.. autoclass:: qibo_comb_optimisation.optimisation_class.optimisation_class.linear_problem
    :members:
    :member-order: bysource
