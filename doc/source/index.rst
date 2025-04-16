.. title::
      Qibo-comb-optimisation

What is qibo-comb-optimisation?
============================

qibo-comb-optimisation is a Qibo plugin that has tools to solve combinatorial optimisation problems.

Installation instructions
=========================

Install first the package dependencies with the following commands.

We recommend to start with a fresh virtual environment to avoid dependencies conflicts with previously installed packages.

.. code-block:: bash

   $ python -m venv ./env
   source activate ./env/bin/activate

The qibo-comb-optimisation package can be installed through pip:

.. code-block:: bash

   pip install qibo_comb_optimisation


Quickstart
==========

Once installed, the plugin allows for... 

.. code-block:: python
   
   from qibo_comb_optimisation.optimisation_class.optimisation_class import QUBO
   from qibo_comb_optimisation.combinatorial_classes.combinatorial_classes import TSP as TSP

   

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   optimisation_class
   combinatorial_classes

.. toctree::
    :maxdepth: 1
    :caption: Documentation links

    Qibo docs <https://qibo.science/qibo/stable/>
    Qibolab docs <https://qibo.science/qibolab/stable/>
    Qibocal docs <https://qibo.science/qibocal/stable/>
    Qibosoq docs <https://qibo.science/qibosoq/stable/>
    Qibochem docs <https://qibo.science/qibochem/stable/>
    Qibotn docs <https://qibo.science/qibotn/stable/>
    Qibo-cloud-backends docs <https://qibo.science/qibo-cloud-backends/stable/>
    Qibo-comb-optimisation docs <https://qibo.science/qibo-comb-optimisation/stable/>
