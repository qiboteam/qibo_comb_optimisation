"""qiboml integration helpers for QAOA training."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def _energy_shift(qubo) -> float:
    """Constant shift between Ising expectation and QUBO objective value."""
    _h, _J, constant = qubo.qubo_to_ising()
    return float(constant)


def _get_differentiation_class(name: str | None):
    """
    Args:
        name (str | None): Name of the differentiation backend. Supported values are
        - "None" or "torch"L use qiboml's default PyTorch autograd
        - "psr": parameter-shift rule
        - "jax" JAX-based differentiation
        - "adjoint": adjoint differentiation.
    """
    if name is None or name == "torch":
        return None

    try:
        differentiation_module = importlib.import_module(
            "qiboml.operations.differentiation"
        )
    except ImportError as exc:
        raise ImportError(
            "engine='qiboml' differentiation backend requires "
            "`qiboml.operations.differentiation`."
        ) from exc

    mapping = {
        "psr": differentiation_module.PSR,
        "jax": differentiation_module.Jax,
        "adjoint": differentiation_module.Adjoint,
    }
    diff = mapping.get(name.lower())
    if diff is None:
        raise ValueError(
            "Unknown qiboml differentiation method. "
            "Supported values are: None, 'psr', 'jax', 'adjoint', 'torch'."
        )
    return diff


def optimize_qaoa_with_qiboml(
    *,
    qubo,
    parameters,
    p: int,
    nshots: int | None,
    noise_model,
    custom_mixer,
    has_alphas: bool,
    optimizer: str,
    lr: float,
    epochs: int,
    differentiation: str | None,
    backend,
    density_matrix: bool = False,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """
    Optimize QAOA parameters using qiboml's pytorch interface.

    Args:
        qubo: A QUBO object
        parameters: Parameters for initialization for QAOA circuit
        p (int): Number of layers
        nshots: Number of shots. Use "None" to use exact statevector.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): a ``NoiseModel`` of Qibo,
            which is applied to the given circuit to perform noisy simulations.
            In case a `transpiler` is passed, the noise model is applied to the transpiled
            circuit. Defaults to ``None``, and no noise is added.
        custom_mixer (List[:class:`qibo.models.Circuit`]): An optional function that takes as input custom mixers.
            If len(custom_mixer) == 1, then use this one circuit as mixer for all layers.
            If len(custom_mixer) == len(gammas), then use each circuit as mixer for each layer.
            If len(custom_mixer) != 1 and != len(gammas), raise an error.
        has_alphas (bool): Indicate whether XQAOA mode is being used.
        optimizer (string): indicate whether to use adam or sgd
        lr (float): learning rate
        epochs (int): number of epochs
        differentiation (str): qiboml differentiation method. Possible strings are None, 'psr', 'jax',
            'adjoint', 'torch'.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the execution.
                If ``None``, it uses the current backend. Defaults to ``None``.
        density_matrix (bool): Indicate whether to use density matrix circuit.

    Returns:
        best (float): Lowest energy (including the constant energy shift) observed over all
            epochs.
        best_params (np.ndarray): Variational parameters that achieved ``best``, as a ``float64`` array
            of the same shape as ``parameters``.
        extra (dict):
            A dictionary with the following information, engine, optimizer, learning_rate, epochs and loss_history.

    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "engine='qiboml' requires torch. Install optional dependencies, "
            "for example with `poetry install --with qiboml`."
        ) from exc

    try:
        from qiboml.interfaces.pytorch import QuantumModel
        from qiboml.models.decoding import Expectation
    except ImportError as exc:
        raise ImportError(
            "engine='qiboml' requires qiboml. Install optional dependencies, "
            "for example with `poetry install --with qiboml`."
        ) from exc

    # Reuse qiboopt's own QAOA-object construction path for the Hamiltonian.
    hamiltonian = qubo.qubo_to_qaoa_object().hamiltonian
    if not hasattr(hamiltonian, "expectation_from_circuit"):
        # TODO: remove once minimum required qiboml version is > 0.1.0.
        # qiboml 0.1.0 uses `expectation_from_circuit`; older builds expose only `expectation`.
        hamiltonian.expectation_from_circuit = hamiltonian.expectation
    circuit_builder = qubo.make_qaoa_circuit_callable(
        p=p,
        custom_mixer=custom_mixer,
        has_alphas=has_alphas,
        include_measurements=False,
        density_matrix=density_matrix,
    )
    decoder = Expectation(
        nqubits=qubo.n,
        observable=hamiltonian,
        nshots=nshots,
        noise_model=noise_model,
        backend=backend,
    )
    energy_shift = _energy_shift(qubo)

    diff_class = _get_differentiation_class(differentiation)
    model = QuantumModel(
        circuit_structure=[circuit_builder],
        decoding=decoder,
        parameters_initialization=np.asarray(parameters, dtype=np.float64),
        differentiation=diff_class,
    )
    model = model.to(dtype=torch.float64)

    if isinstance(optimizer, str):
        opt_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }
        opt_cls = opt_map.get(optimizer.lower())
        if opt_cls is None:
            raise ValueError(
                f"Unknown optimizer string '{optimizer}'. "
                f"Pass a torch.optim.Optimizer subclass directly, or use one of: {list(opt_map)}."
            )
    else:
        if not callable(optimizer):
            raise ValueError(
                f"optimizer must be a string ('adam', 'sgd') or a callable "
                f"torch.optim.Optimizer subclass, got {type(optimizer)!r}."
            )
        opt_cls = optimizer  # user supplied a class directly

    torch_optimizer = opt_cls(model.parameters(), lr=lr)

    losses = []
    best = float("inf")
    best_params = np.asarray(parameters, dtype=np.float64)
    for _ in range(epochs):
        torch_optimizer.zero_grad()
        loss = model()
        if loss.ndim > 0:
            loss = loss.squeeze()
        loss_value = float(loss.detach().cpu().item()) + energy_shift
        losses.append(loss_value)
        if loss_value < best:
            best = loss_value
            current_parameters = model.circuit_parameters
            if isinstance(current_parameters, torch.Tensor):
                # pylint: disable-next=not-callable
                current_parameters = current_parameters.detach().cpu().numpy()
            best_params = np.asarray(current_parameters, dtype=np.float64).copy()
        loss.backward()
        torch_optimizer.step()

    extra = {
        "engine": "qiboml",
        "optimizer": getattr(opt_cls, "__name__", str(opt_cls)),
        "learning_rate": lr,
        "epochs": epochs,
        "loss_history": losses,
    }
    return best, best_params, extra
