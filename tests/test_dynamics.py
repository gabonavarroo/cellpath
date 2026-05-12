"""Dynamics-model tests: construction, forward shapes, loss stubs.

Until Agent B implements the model body, ``test_forward_shape`` is xfail (the constructor
populates ``n_latent``/``n_genes``/``d_emb`` attributes but the forward pass raises).
"""

from __future__ import annotations

import pytest


def _torch_or_skip():
    return pytest.importorskip("torch")


class TestDynamicsConstruction:
    def test_module_imports(self) -> None:
        from src.models.dynamics import PerturbationDynamicsModel  # noqa: F401

    def test_construction_sets_basic_attrs(self) -> None:
        _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel

        model = PerturbationDynamicsModel(n_latent=32, n_genes=100, d_emb=64, n_hidden=128)
        assert model.n_latent == 32
        assert model.n_genes == 100
        assert model.d_emb == 64
        assert model.log_var_min < model.log_var_max


class TestDynamicsForward:
    """Forward-pass tests — all implemented for Day 0."""

    def test_forward_shape(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel

        model = PerturbationDynamicsModel(n_latent=32, n_genes=100)
        z = torch.randn(8, 32)
        g = torch.randint(1, 101, (8,))  # 1-indexed gene indices per Contract 2
        z_next, mu, log_var = model(z, g)
        assert z_next.shape == (8, 32)
        assert mu.shape == (8, 32)
        assert log_var.shape == (8, 32)

    def test_log_var_clamped(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel

        model = PerturbationDynamicsModel(
            n_latent=32, n_genes=10, log_var_min=-5.0, log_var_max=3.0
        )
        z = torch.randn(4, 32)
        g = torch.randint(1, 11, (4,))  # 1-indexed
        _, _, log_var = model(z, g)
        assert (log_var >= -5.0 - 1e-3).all()
        assert (log_var <= 3.0 + 1e-3).all()

    def test_residual_returns_z_plus_mu(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel

        model = PerturbationDynamicsModel(n_latent=32, n_genes=10)
        z = torch.randn(4, 32)
        g = torch.randint(1, 11, (4,))  # 1-indexed
        z_next, mu, _ = model(z, g)
        assert torch.allclose(z_next, z + mu, atol=1e-5)


class TestDynamicsLosses:
    def test_heteroscedastic_nll_is_stubbed(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import heteroscedastic_nll

        mu = torch.zeros(4, 32)
        log_var = torch.zeros(4, 32)
        target = torch.zeros(4, 32)
        with pytest.raises(NotImplementedError, match="Agent B"):
            heteroscedastic_nll(mu, log_var, target)
