import builtins
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from qiboopt.integrations.qiboml_adapter import (
    _energy_shift,
    _get_differentiation_class,
    optimize_qaoa_with_qiboml,
)
from qiboopt.opt_class.opt_class import QUBO


def _qiboml_available():
    return (
        importlib.util.find_spec("qiboml") is not None
        and importlib.util.find_spec("torch") is not None
    )


def test_get_differentiation_class_none_and_torch_return_none():
    assert _get_differentiation_class(None) is None
    assert _get_differentiation_class("torch") is None


def test_get_differentiation_class_import_error(monkeypatch):
    def _raise_import_error(_name):
        raise ImportError("mocked import failure")

    monkeypatch.setattr(
        "qiboopt.integrations.qiboml_adapter.importlib.import_module",
        _raise_import_error,
    )

    with pytest.raises(ImportError, match="differentiation backend requires"):
        _get_differentiation_class("psr")


def test_get_differentiation_class_invalid_value(monkeypatch):
    fake_module = SimpleNamespace(PSR=object, Jax=object, Adjoint=object)
    monkeypatch.setattr(
        "qiboopt.integrations.qiboml_adapter.importlib.import_module",
        lambda _name: fake_module,
    )

    with pytest.raises(ValueError, match="Unknown qiboml differentiation method"):
        _get_differentiation_class("invalid")


def test_get_differentiation_class_valid_value(monkeypatch):
    fake_psr = object()
    fake_module = SimpleNamespace(PSR=fake_psr, Jax=object, Adjoint=object)
    monkeypatch.setattr(
        "qiboopt.integrations.qiboml_adapter.importlib.import_module",
        lambda _name: fake_module,
    )
    assert _get_differentiation_class("psr") is fake_psr


def test_energy_shift_matches_qubo_to_ising_constant():
    qp = QUBO(0, {(0, 0): 2.0, (0, 1): 1.5, (1, 1): -0.5})
    _h, _J, constant = qp.qubo_to_ising()
    assert _energy_shift(qp) == float(constant)


def test_optimize_qaoa_with_qiboml_raises_when_torch_missing(monkeypatch):
    original_import = builtins.__import__

    def _mocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("mocked missing torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _mocked_import)

    with pytest.raises(ImportError, match="requires torch"):
        optimize_qaoa_with_qiboml(
            qubo=None,
            parameters=[0.1, 0.2],
            p=1,
            nshots=10,
            noise_model=None,
            custom_mixer=None,
            has_alphas=False,
            optimizer="adam",
            lr=0.05,
            epochs=1,
            differentiation=None,
            backend=None,
        )


def test_optimize_qaoa_with_qiboml_raises_when_qiboml_missing(monkeypatch):
    original_import = builtins.__import__
    fake_torch = SimpleNamespace()

    def _mocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("qiboml"):
            raise ImportError("mocked missing qiboml")
        if name == "torch":
            return fake_torch
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _mocked_import)

    with pytest.raises(ImportError, match="requires qiboml"):
        optimize_qaoa_with_qiboml(
            qubo=None,
            parameters=[0.1, 0.2],
            p=1,
            nshots=10,
            noise_model=None,
            custom_mixer=None,
            has_alphas=False,
            optimizer="adam",
            lr=0.05,
            epochs=1,
            differentiation=None,
            backend=None,
        )


def test_qibo_engine_does_not_require_qiboml(monkeypatch):
    import qiboopt.integrations.qiboml_adapter as adapter_module

    def _should_not_be_called(**kwargs):
        raise AssertionError("qiboml adapter should not be called for qibo engine")

    monkeypatch.setattr(
        adapter_module, "optimize_qaoa_with_qiboml", _should_not_be_called
    )
    qp = QUBO(0, {(0, 0): 1.0, (1, 1): 1.0})
    best, params, extra, circuit, freqs = qp.train_QAOA(
        gammas=[0.1, 0.2],
        betas=[0.3, 0.4],
        nshots=100,
        maxiter=5,
        engine="qibo",
    )
    assert isinstance(best, float)
    assert isinstance(params, np.ndarray)
    assert isinstance(freqs, dict)


@pytest.mark.skipif(not _qiboml_available(), reason="qiboml/torch not installed")
def test_qiboml_engine_output_contract():
    qp = QUBO(0, {(0, 0): 2.0, (1, 1): 2.0})
    best, params, extra, circuit, freqs = qp.train_QAOA(
        gammas=[0.1, 0.2],
        betas=[0.2, 0.3],
        nshots=200,
        engine="qiboml",
        optimizer="adam",
        lr=0.05,
        epochs=20,
    )
    assert isinstance(best, float)
    assert isinstance(params, np.ndarray)
    assert isinstance(extra, dict)
    assert isinstance(freqs, dict)
    assert extra.get("engine") == "qiboml"
    assert "loss_history" in extra


@pytest.mark.skipif(not _qiboml_available(), reason="qiboml/torch not installed")
def test_qiboml_engine_updates_parameters():
    qp = QUBO(0, {(0, 0): 2.0, (1, 1): 2.0})
    gammas = [0.1, 0.2]
    betas = [0.3, 0.4]
    init_params = np.array(gammas + betas, dtype=float)
    best, params, extra, circuit, freqs = qp.train_QAOA(
        gammas=gammas,
        betas=betas,
        nshots=200,
        engine="qiboml",
        optimizer="adam",
        lr=0.05,
        epochs=30,
    )
    assert np.linalg.norm(params - init_params) > 0
    assert np.isfinite(best)


@pytest.mark.skipif(not _qiboml_available(), reason="qiboml/torch not installed")
@pytest.mark.parametrize("nshots", [None, 0])
def test_qiboml_engine_supports_exact_mode(nshots):
    qp = QUBO(0, {(0, 0): 1.0, (1, 1): 1.0})
    best, params, extra, circuit, stats = qp.train_QAOA(
        gammas=[0.1, 0.2],
        betas=[0.2, 0.3],
        nshots=nshots,
        engine="qiboml",
        optimizer="adam",
        lr=0.05,
        epochs=10,
    )
    assert np.isfinite(best)
    assert isinstance(params, np.ndarray)
    assert isinstance(extra, dict)
    assert isinstance(stats, dict)
    assert all(isinstance(value, float) for value in stats.values())
    assert np.isclose(sum(stats.values()), 1.0)


@pytest.mark.skipif(not _qiboml_available(), reason="qiboml/torch not installed")
def test_qiboml_engine_supports_sgd_optimizer():
    qp = QUBO(0, {(0, 0): 2.0, (1, 1): 2.0})
    best, params, extra, circuit, freqs = qp.train_QAOA(
        gammas=[0.1, 0.2],
        betas=[0.2, 0.3],
        nshots=150,
        engine="qiboml",
        optimizer="sgd",
        lr=0.05,
        epochs=15,
    )
    assert np.isfinite(best)
    assert isinstance(params, np.ndarray)
    assert isinstance(extra, dict)
    assert isinstance(freqs, dict)
    assert extra.get("optimizer") == "SGD"


@pytest.mark.skipif(not _qiboml_available(), reason="qiboml/torch not installed")
def test_qiboml_engine_unknown_optimizer_raises():
    qp = QUBO(0, {(0, 0): 2.0, (1, 1): 2.0})
    with pytest.raises(ValueError, match="Unknown optimizer string"):
        qp.train_QAOA(
            gammas=[0.1, 0.2],
            betas=[0.2, 0.3],
            nshots=100,
            engine="qiboml",
            optimizer="rmsprop",
            epochs=5,
        )


@pytest.mark.skipif(not _qiboml_available(), reason="qiboml/torch not installed")
def test_qiboml_engine_noncallable_optimizer_raises():
    qp = QUBO(0, {(0, 0): 2.0, (1, 1): 2.0})
    with pytest.raises(ValueError, match="optimizer must be a string"):
        qp.train_QAOA(
            gammas=[0.1, 0.2],
            betas=[0.2, 0.3],
            nshots=100,
            engine="qiboml",
            optimizer=42,
            epochs=5,
        )


@pytest.mark.skipif(not _qiboml_available(), reason="qiboml/torch not installed")
def test_qiboml_engine_accepts_optimizer_class():
    import torch

    qp = QUBO(0, {(0, 0): 2.0, (1, 1): 2.0})
    best, params, extra, circuit, freqs = qp.train_QAOA(
        gammas=[0.1, 0.2],
        betas=[0.2, 0.3],
        nshots=100,
        engine="qiboml",
        optimizer=torch.optim.SGD,
        lr=0.05,
        epochs=5,
    )
    assert abs(best) < 1.5
    assert freqs['00'] > 0.7
    assert isinstance(params, np.ndarray)
    assert isinstance(extra, dict)
    assert isinstance(freqs, dict)
    assert extra.get("optimizer") == "SGD"
