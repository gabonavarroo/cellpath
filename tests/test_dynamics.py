"""Dynamics-model tests: construction, forward shapes, loss functions, gradient flow.

Phase 0 (scaffold): forward-pass tests pass.
Phase 1 (this patch): loss-function tests and gradient-flow tests added; smoke-train test
verifies that optimizer + loss are correctly wired together on mock data.
"""

from __future__ import annotations

from itertools import islice

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
    """Tests for heteroscedastic_nll and composition_loss."""

    def test_heteroscedastic_nll_returns_finite_scalar(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import heteroscedastic_nll

        B, D = 8, 32
        mu     = torch.randn(B, D)
        lv     = torch.zeros(B, D)
        target = torch.randn(B, D)
        loss   = heteroscedastic_nll(mu, lv, target)
        assert loss.ndim == 0, "loss must be a scalar"
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"

    def test_heteroscedastic_nll_decreases_with_better_mu(self) -> None:
        """mu == target should give strictly lower NLL than mu == target + 1."""
        torch = _torch_or_skip()
        from src.models.dynamics import heteroscedastic_nll

        B, D   = 8, 32
        target = torch.randn(B, D)
        lv     = torch.zeros(B, D)
        loss_good = heteroscedastic_nll(target,       lv, target, log_var_reg=0.0)
        loss_bad  = heteroscedastic_nll(target + 1.0, lv, target, log_var_reg=0.0)
        assert loss_good < loss_bad, (
            f"Perfect-mu NLL ({loss_good:.4f}) should be below shifted-mu NLL ({loss_bad:.4f})"
        )

    def test_heteroscedastic_nll_grad_flows_to_mu_and_log_var(self) -> None:
        """Backward pass must produce non-zero gradients for both mu and log_var."""
        torch = _torch_or_skip()
        from src.models.dynamics import heteroscedastic_nll

        B, D   = 8, 32
        mu     = torch.randn(B, D, requires_grad=True)
        lv     = torch.zeros(B, D, requires_grad=True)
        target = torch.randn(B, D)
        loss   = heteroscedastic_nll(mu, lv, target)
        loss.backward()
        assert mu.grad is not None and mu.grad.abs().sum() > 0, "No gradient into mu"
        assert lv.grad is not None and lv.grad.abs().sum() > 0, "No gradient into log_var"

    def test_composition_loss_returns_finite_scalar(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel, composition_loss

        model    = PerturbationDynamicsModel(n_latent=32, n_genes=10)
        B        = 4
        z_ctrl   = torch.randn(B, 32)
        g_a      = torch.randint(1, 11, (B,))   # 1-indexed, range [1, 10]
        g_b      = torch.randint(1, 11, (B,))
        z_pert   = torch.randn(B, 32)
        loss     = composition_loss(model, z_ctrl, g_a, g_b, z_pert)
        assert loss.ndim == 0, "composition loss must be a scalar"
        assert torch.isfinite(loss), f"composition loss is not finite: {loss.item()}"

    def test_composition_loss_grad_flows_through_chain(self) -> None:
        """Both forward passes must contribute gradient to head_mu and gene_embedding."""
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel, composition_loss

        model  = PerturbationDynamicsModel(n_latent=32, n_genes=10)
        B      = 4
        z_ctrl = torch.randn(B, 32)
        g_a    = torch.randint(1, 11, (B,))
        g_b    = torch.randint(1, 11, (B,))
        z_pert = torch.randn(B, 32)
        loss   = composition_loss(model, z_ctrl, g_a, g_b, z_pert)
        loss.backward()

        mu_grad  = model.head_mu.weight.grad
        emb_grad = model.gene_embedding.weight.grad
        assert mu_grad  is not None and mu_grad.abs().sum()  > 0, "No gradient into head_mu"
        assert emb_grad is not None and emb_grad.abs().sum() > 0, "No gradient into gene_embedding"

    def test_smoke_train_step_decreases_loss(self, mock_pairs_npz) -> None:
        """20 optimizer steps on learnable mock pairs must reduce NLL by ≥10%."""
        torch = _torch_or_skip()
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        from src.models.dynamics import PerturbationDynamicsModel, heteroscedastic_nll

        data     = np.load(mock_pairs_npz)
        z_ctrl   = torch.from_numpy(data["z_ctrl"])
        gene_idx = torch.from_numpy(data["gene_idx"]).long()   # int32 → int64
        z_pert   = torch.from_numpy(data["z_pert"])
        n_genes  = int(gene_idx.max().item())   # 4 in conftest fixture

        loader = DataLoader(
            TensorDataset(z_ctrl, gene_idx, z_pert),
            batch_size=64,
            shuffle=True,
        )
        model     = PerturbationDynamicsModel(n_latent=32, n_genes=n_genes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses: list[float] = []
        for z_c, g, z_p in islice(loader, 20):
            _, mu, lv = model(z_c, g)
            loss = heteroscedastic_nll(mu, lv, z_p - z_c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.9, (
            f"Loss did not decrease by ≥10% over 20 steps: "
            f"{losses[0]:.4f} → {losses[-1]:.4f}"
        )
