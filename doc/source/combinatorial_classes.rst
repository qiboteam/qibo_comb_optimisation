.. _combinatorial_classes:

Combinatorial classes
---------------------

Listed here are two Quadratic Unconstrained Binary Optimisation (QUBO) problems, the :ref:`Travelling Salesman Problem <TSP>` and the :ref:`Maximum Independent Set <MIS>`.

.. _TSP:

Travelling Salesman Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Travelling Salesman Problem (sometimes referred to as the Travelling Salesperson Problem), commonly abbreviated as TSP, is a NP-hard problem in combinatorial optimisation.

Briefly, the problem revolves around finding the shortest possible route for a salesman to visit some cities before returning to the origin. TSP is usually formulated as a graph problem with nodes specifying the cities and edges denoting the distances between each city.

The idea behind TSP can be mapped to similar-type problems. For instance, what is the optimal route for the salesman to take in order to minimise something.

In this module, the TSP class follows Hadfield's 2017 paper `arxiv:1709.03489 <https://arxiv.org/abs/1709.03489>`.

.. autoclass:: qibo_comb_optimisation.combinatorial_classes.combinatorial_classes.TSP
    :members:
    :member-order: bysource

.. _MIS:

Maximum Independent Set
^^^^^^^^^^^^^^^^^^^^^^^

The MIS problem involves selecting the largest subset of non-adjacent vertices in a graph.

.. autoclass:: qibo_comb_optimisation.combinatorial_classes.combinatorial_classes.Mis
    :members:
    :member-order: bysource
