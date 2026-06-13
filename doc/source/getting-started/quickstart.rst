Quickstart
----------

Once installed, ``qiboopt`` allows the general user to solve QUBO problems with the built-in ``QUBO`` class.
Along with the ``QUBO`` class, there are some combinatorial classes found in :class:`qiboopt.combinatorial`.

Formulating a QUBO problem:

- Maximal Independent Set:

.. code-block:: python

   import networkx as nx
   from qiboopt.combinatorial.combinatorial import MIS

   g = nx.Graph()
   g.add_edges_from([(0, 1), (1, 2), (2, 0)])
   mis = MIS(g)
   penalty = 10
   qp = mis.penalty_method(penalty)

- Shortest Vector Problem:

.. code-block:: python

   Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
   qp = QUBO(0, Qdict)

   # Brute force search by evaluating all possible binary vectors.
   opt_vector, min_value = qp.brute_force()

QUBO problems can be solved using the `QAOA <https://arxiv.org/abs/1709.03489>`_ method:

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   # Train 2 layers of regular QAOA
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   output = qp.train_QAOA(gammas=gammas, betas=betas)

By default, qiboopt builds QAOA circuits with ``density_matrix=False`` so that
standard state-vector and tensor-network backends can execute larger circuits.
Set ``density_matrix=True`` only when a density-matrix circuit is required:

.. code-block:: python

   circuit = qp.qubo_to_qaoa_circuit(
      gammas=gammas,
      betas=betas,
      include_measurements=False,
      density_matrix=True,
   )

When ``train_QAOA`` receives a Qibo ``noise_model``, qiboopt automatically uses
density-matrix circuits because Qibo noisy simulation requires them for
exact/no-measurement execution.

or the more modern `XQAOA <https://arxiv.org/abs/2302.04479>`_ approach:

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   # Train 2 layers of XQAOA
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   alphas = [0.5, 0.6]
   output = qp.train_QAOA(gammas=gammas, betas=betas, alphas=alphas)

The Conditional Variance at Risk (CVaR) can also be used as an alternative loss function in solving the QUBO problem:

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   # Train 2 layers of regular QAOA with CVaR
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   output = qp.train_QAOA(gammas=gammas, betas=betas, regular_loss=False, cvar_delta=0.1)

To use qiboml's pytorch training loop instead of the qibo optimizer, set ``engine="qiboml"``:

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   qp = QUBO(0, {(0, 0): 1.0, (1, 1): 1.0})
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   output = qp.train_QAOA(
      gammas=gammas,
      betas=betas,
      engine="qiboml",
      optimizer="adam",
      lr=0.05,
      epochs=100,
   )

You can also run in exact (no-shot) mode by setting ``nshots=None`` (or ``nshots=0``):

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   output = qp.train_QAOA(gammas=gammas, betas=betas, nshots=None)

In sampled mode (``nshots > 0``), the returned dictionary contains bitstring counts.
In exact mode (``nshots is None`` or ``nshots == 0``), it contains exact bitstring probabilities.
