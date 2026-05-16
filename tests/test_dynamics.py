"""Dynamics-model tests: construction, forward shapes, loss functions, gradient flow.

Phase 0 (scaffold): forward-pass tests pass.
Phase 1 (this patch): loss-function tests and gradient-flow tests added; smoke-train test
verifies that optimizer + loss are correctly wired together on mock data.
"""

from __future__ import annotations

from itertools import cycle, islice

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


class TestEmbeddingHelpers:
    def test_noop_embedding_is_zero_vector_on_requested_device(self) -> None:
        torch = _torch_or_skip()
        from src.models.embeddings import noop_embedding

        emb = noop_embedding(7, device="cpu")
        assert emb.shape == (7,)
        assert emb.device.type == "cpu"
        assert torch.count_nonzero(emb).item() == 0


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
        """20 optimizer steps (cycling over the fixture) must reduce NLL."""
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
            shuffle=False,   # deterministic for the test
        )
        model     = PerturbationDynamicsModel(n_latent=32, n_genes=n_genes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses: list[float] = []
        for z_c, g, z_p in islice(cycle(loader), 20):   # exactly 20 steps
            _, mu, lv = model(z_c, g)
            loss = heteroscedastic_nll(mu, lv, z_p - z_c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease over 20 steps: "
            f"{losses[0]:.4f} → {losses[-1]:.4f}"
        )


class TestDynamicsFlags:
    """Architecture-ablation flags: state-linear skip and per-gene delta bias.

    Defaults must preserve the legacy baseline exactly; non-default combinations
    must still satisfy the forward contract (z_next = z + mu; correct shapes).
    """

    def _seeded_model(self, **kwargs):
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel
        torch.manual_seed(0)
        return PerturbationDynamicsModel(n_latent=32, n_genes=10, **kwargs)

    def test_forward_with_both_flags_false(self) -> None:
        torch = _torch_or_skip()
        model = self._seeded_model(
            use_state_linear_skip=False, use_gene_delta_bias=False,
        )
        z = torch.randn(8, 32)
        g = torch.randint(1, 11, (8,))
        z_next, mu, log_var = model(z, g)
        assert z_next.shape == (8, 32)
        assert mu.shape == (8, 32)
        assert log_var.shape == (8, 32)
        assert torch.allclose(z_next, z + mu, atol=1e-5)
        assert model.state_linear is None
        assert model.gene_delta is None

    def test_forward_with_state_skip_only(self) -> None:
        torch = _torch_or_skip()
        from torch import nn
        model = self._seeded_model(
            use_state_linear_skip=True, use_gene_delta_bias=False,
        )
        z = torch.randn(4, 32)
        g = torch.randint(1, 11, (4,))
        z_next, mu, log_var = model(z, g)
        assert z_next.shape == (4, 32)
        assert mu.shape == (4, 32)
        assert log_var.shape == (4, 32)
        assert torch.allclose(z_next, z + mu, atol=1e-5)
        assert isinstance(model.state_linear, nn.Linear)
        assert model.gene_delta is None

    def test_forward_with_gene_bias_only(self) -> None:
        torch = _torch_or_skip()
        from torch import nn
        model = self._seeded_model(
            use_state_linear_skip=False, use_gene_delta_bias=True,
        )
        z = torch.randn(4, 32)
        g = torch.randint(1, 11, (4,))
        z_next, mu, log_var = model(z, g)
        assert z_next.shape == (4, 32)
        assert mu.shape == (4, 32)
        assert log_var.shape == (4, 32)
        assert torch.allclose(z_next, z + mu, atol=1e-5)
        assert model.state_linear is None
        assert isinstance(model.gene_delta, nn.Embedding)

    def test_forward_with_both_flags_true(self) -> None:
        torch = _torch_or_skip()
        from torch import nn
        model = self._seeded_model(
            use_state_linear_skip=True, use_gene_delta_bias=True,
        )
        z = torch.randn(4, 32)
        g = torch.randint(1, 11, (4,))
        z_next, mu, log_var = model(z, g)
        assert z_next.shape == (4, 32)
        assert mu.shape == (4, 32)
        assert log_var.shape == (4, 32)
        assert torch.allclose(z_next, z + mu, atol=1e-5)
        assert isinstance(model.state_linear, nn.Linear)
        assert isinstance(model.gene_delta, nn.Embedding)

    def test_gene_delta_index_0_initialized_to_zero(self) -> None:
        torch = _torch_or_skip()
        model = self._seeded_model(
            use_state_linear_skip=False, use_gene_delta_bias=True,
        )
        assert model.gene_delta is not None
        assert torch.all(model.gene_delta.weight[0] == 0.0)
        # Other rows should NOT be zero (default Embedding init is N(0, 1)).
        assert model.gene_delta.weight[1:].abs().sum() > 0

    def test_state_skip_adds_extra_params(self) -> None:
        _torch_or_skip()
        base = self._seeded_model(
            use_state_linear_skip=False, use_gene_delta_bias=False,
        )
        skip = self._seeded_model(
            use_state_linear_skip=True, use_gene_delta_bias=False,
        )
        n_base = sum(p.numel() for p in base.parameters())
        n_skip = sum(p.numel() for p in skip.parameters())
        # nn.Linear(32, 32) adds 32*32 + 32 = 1056 params.
        assert n_skip - n_base == 32 * 32 + 32

    def test_gene_bias_adds_extra_params(self) -> None:
        _torch_or_skip()
        base = self._seeded_model(
            use_state_linear_skip=False, use_gene_delta_bias=False,
        )
        bias = self._seeded_model(
            use_state_linear_skip=False, use_gene_delta_bias=True,
        )
        n_base = sum(p.numel() for p in base.parameters())
        n_bias = sum(p.numel() for p in bias.parameters())
        # nn.Embedding(n_genes+1=11, n_latent=32) adds 11*32 = 352 params.
        assert n_bias - n_base == 11 * 32

    # ----- Baseline-invariance contract: both flags False == legacy baseline -----

    def test_baseline_invariance_param_count(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel
        torch.manual_seed(7)
        legacy = PerturbationDynamicsModel(n_latent=32, n_genes=10)
        torch.manual_seed(7)
        explicit = PerturbationDynamicsModel(
            n_latent=32, n_genes=10,
            use_state_linear_skip=False, use_gene_delta_bias=False,
        )
        assert sum(p.numel() for p in legacy.parameters()) == sum(
            p.numel() for p in explicit.parameters()
        )

    def test_baseline_invariance_state_dict_keys(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel
        torch.manual_seed(7)
        legacy = PerturbationDynamicsModel(n_latent=32, n_genes=10)
        torch.manual_seed(7)
        explicit = PerturbationDynamicsModel(
            n_latent=32, n_genes=10,
            use_state_linear_skip=False, use_gene_delta_bias=False,
        )
        assert set(legacy.state_dict().keys()) == set(explicit.state_dict().keys())

    def test_baseline_invariance_forward_output(self) -> None:
        """Same seed + same kwargs (defaults) ⇒ same forward output."""
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel

        torch.manual_seed(11)
        legacy = PerturbationDynamicsModel(n_latent=32, n_genes=10)
        torch.manual_seed(11)
        explicit = PerturbationDynamicsModel(
            n_latent=32, n_genes=10,
            use_state_linear_skip=False, use_gene_delta_bias=False,
        )
        legacy.eval()
        explicit.eval()
        torch.manual_seed(99)
        z = torch.randn(6, 32)
        g = torch.randint(1, 11, (6,))
        with torch.no_grad():
            l_out = legacy(z, g)
            e_out = explicit(z, g)
        for a, b in zip(l_out, e_out, strict=True):
            assert torch.equal(a, b), (
                "both-flags-False output diverged from the legacy default — "
                "the no-flag and flags=False code paths must be operationally identical."
            )


class TestHybridLoss:
    """Hybrid loss: NLL + lambda_mse_delta * MSE(mu, target_delta)."""

    def test_lambda_zero_matches_pure_nll(self) -> None:
        """Adding 0.0 * MSE to NLL must leave the value unchanged."""
        torch = _torch_or_skip()
        from src.models.dynamics import heteroscedastic_nll

        B, D   = 8, 32
        mu     = torch.randn(B, D)
        lv     = torch.zeros(B, D)
        target = torch.randn(B, D)
        nll_only = heteroscedastic_nll(mu, lv, target, log_var_reg=0.0)
        hybrid   = (
            heteroscedastic_nll(mu, lv, target, log_var_reg=0.0)
            + 0.0 * torch.nn.functional.mse_loss(mu, target)
        )
        assert torch.equal(nll_only, hybrid), (
            "lambda=0.0 hybrid loss must equal bare NLL bit-for-bit"
        )

    def test_lambda_positive_changes_loss(self) -> None:
        """lambda > 0 with mu ≠ target must strictly increase total loss above NLL."""
        torch = _torch_or_skip()
        from src.models.dynamics import heteroscedastic_nll

        B, D   = 8, 32
        target = torch.zeros(B, D)
        mu     = torch.ones(B, D)   # clearly far from target
        lv     = torch.zeros(B, D)
        lmda   = 0.1
        nll    = heteroscedastic_nll(mu, lv, target, log_var_reg=0.0)
        hybrid = nll + lmda * torch.nn.functional.mse_loss(mu, target)
        assert hybrid.item() > nll.item(), (
            f"hybrid ({hybrid.item():.4f}) must exceed NLL ({nll.item():.4f}) when mu ≠ target"
        )

    def test_lambda_positive_pulls_mu_toward_target(self) -> None:
        """A few gradient steps with the hybrid loss must reduce ||mu - target||^2."""
        torch = _torch_or_skip()
        from src.models.dynamics import PerturbationDynamicsModel, heteroscedastic_nll

        B, D   = 16, 32
        torch.manual_seed(42)
        model  = PerturbationDynamicsModel(n_latent=D, n_genes=10)
        z_ctrl = torch.randn(B, D)
        g_idx  = torch.randint(1, 11, (B,))
        target_delta = torch.randn(B, D)
        opt    = torch.optim.SGD(model.parameters(), lr=0.05)
        lmda   = 1.0

        errors: list[float] = []
        for _ in range(6):
            _, mu, lv = model(z_ctrl, g_idx)
            loss = (
                heteroscedastic_nll(mu, lv, target_delta, log_var_reg=0.0)
                + lmda * torch.nn.functional.mse_loss(mu, target_delta)
            )
            errors.append(torch.nn.functional.mse_loss(mu, target_delta).item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert errors[-1] < errors[0], (
            f"Hybrid loss did not reduce ||mu - target||^2: {errors[0]:.4f} -> {errors[-1]:.4f}"
        )


class TestEpochMetrics:
    """Per-epoch metric records must be JSON-serializable."""

    def test_epoch_record_json_serializable(self) -> None:
        import json
        record = {
            "epoch":     5,
            "train_nll": 1.234,
            "val_nll":   1.189,
            "val_mlp_r2":                  0.38,
            "val_mlp_pearson":             0.595,
            "val_ridge_r2":                0.383,
            "val_ridge_pearson":           0.601,
            "val_mlp_minus_ridge_pearson": -0.006,
            "uncertainty_spearman":        0.247,
            "margin_checks": {
                "margin_vs_noop_r2":
                    {"value": 0.28,   "threshold": 0.10, "pass": True},
                "margin_vs_global_mean_r2":
                    {"value": 0.15,   "threshold": 0.05, "pass": True},
                "margin_vs_per_gene_mean_r2":
                    {"value": 0.02,   "threshold": 0.0,  "pass": True},
                "margin_vs_linear_ridge_pearson":
                    {"value": -0.006, "threshold": 0.03, "pass": False},
                "margin_vs_knn_r2":
                    {"value": 0.05,   "threshold": 0.03, "pass": True},
            },
            "all_margins_pass": False,
            "lr": 1e-3,
        }
        serialized = json.dumps(record)
        recovered  = json.loads(serialized)
        assert recovered["epoch"] == record["epoch"]
        assert recovered["all_margins_pass"] is False
        assert abs(recovered["val_mlp_pearson"] - record["val_mlp_pearson"]) < 1e-9


class TestCorrelationLoss:
    """Tests for src.analysis.metrics.correlation_loss."""

    def test_perfect_correlation_gives_zero(self) -> None:
        """When pred == target, every per-dim corr is 1.0, so loss should be ~0."""
        torch = _torch_or_skip()
        from src.analysis.metrics import correlation_loss

        torch.manual_seed(0)
        B, D   = 32, 16
        target = torch.randn(B, D)
        loss   = correlation_loss(target, target)
        assert loss.item() < 1e-4, f"perfect correlation loss should be ~0, got {loss.item()}"

    def test_anti_correlation_gives_near_two(self) -> None:
        """When pred == -target (anti-correlated), each per-dim corr is -1.0, loss ≈ 2."""
        torch = _torch_or_skip()
        from src.analysis.metrics import correlation_loss

        torch.manual_seed(1)
        B, D   = 64, 8
        target = torch.randn(B, D)
        loss   = correlation_loss(-target, target)
        assert loss.item() > 1.9, f"anti-correlation loss should be ≈2, got {loss.item()}"

    def test_zero_variance_dim_excluded(self) -> None:
        """A dimension with constant target must be excluded (no NaN/Inf)."""
        torch = _torch_or_skip()
        from src.analysis.metrics import correlation_loss

        B, D   = 16, 4
        target = torch.randn(B, D)
        target[:, 2] = 5.0       # constant column → zero variance → should be excluded
        pred   = torch.randn(B, D)
        loss   = correlation_loss(pred, target)
        assert torch.isfinite(loss), f"loss must be finite even with zero-var dim, got {loss.item()}"

    def test_output_is_finite_scalar(self) -> None:
        """Random inputs should always produce a finite scalar."""
        torch = _torch_or_skip()
        from src.analysis.metrics import correlation_loss

        torch.manual_seed(42)
        B, D   = 128, 32
        pred   = torch.randn(B, D)
        target = torch.randn(B, D)
        loss   = correlation_loss(pred, target)
        assert loss.ndim == 0, "output must be a scalar (0-dim tensor)"
        assert torch.isfinite(loss), f"loss must be finite for random inputs, got {loss.item()}"
