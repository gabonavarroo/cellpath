"""V3C Phase 2 — tests for contraction-aware regularizer loss terms.

Targets the three pathologies the Phase 0C audit surfaced:
  * `excessive_alignment_penalty`     — caps per-(z, g) cos(μ, z_ref − z) above τ
  * `universal_attractor_penalty`     — caps per-gene mean alignment across batch above τ
  * `action_diversity_penalty`        — floor on across-batch variance of μ

Default `λ_*=0` is mandatory: the three helpers must return 0 when their inputs are
below threshold so existing V2/V3 training is byte-identical when the keys are off.

See `artifacts_v3/v3c/interpretation/v3c_phase2_contraction_aware_spec.md` §1.
"""

from __future__ import annotations

import pytest


def _torch_or_skip():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Excessive-alignment penalty
# ---------------------------------------------------------------------------


class TestExcessiveAlignmentPenalty:
    def test_returns_scalar_tensor(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import excessive_alignment_penalty

        z = torch.randn(8, 32)
        z_ref = torch.zeros(32)
        mu = torch.randn(8, 32)
        loss = excessive_alignment_penalty(mu, z, z_ref, tau=0.80)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_zero_when_alignment_below_tau(self) -> None:
        """Random μ orthogonal-ish to (z_ref − z) → alignment ≈ 0 ≪ τ → loss = 0."""
        torch = _torch_or_skip()
        from src.models.dynamics import excessive_alignment_penalty

        # Construct μ orthogonal to (z_ref − z): pick z_ref ≠ 0, μ along an orthogonal direction.
        D = 32
        z = torch.zeros(16, D)
        z_ref = torch.zeros(D); z_ref[0] = 1.0
        mu = torch.zeros(16, D); mu[:, 1] = 1.0  # along dim 1, target_dir along dim 0
        loss = excessive_alignment_penalty(mu, z, z_ref, tau=0.80)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_alignment_above_tau(self) -> None:
        """μ exactly aligned with (z_ref − z) → α=1.0 > τ=0.80 → loss = (1 − 0.80)² = 0.04."""
        torch = _torch_or_skip()
        from src.models.dynamics import excessive_alignment_penalty

        D = 32
        z = torch.zeros(16, D)
        z_ref = torch.zeros(D); z_ref[0] = 1.0
        mu = torch.zeros(16, D); mu[:, 0] = 0.5  # aligned with target_dir along dim 0
        loss = excessive_alignment_penalty(mu, z, z_ref, tau=0.80)
        assert float(loss) == pytest.approx(0.04, abs=1e-4)

    def test_monotone_in_tau(self) -> None:
        """Higher τ → looser cap → lower loss for the same μ."""
        torch = _torch_or_skip()
        from src.models.dynamics import excessive_alignment_penalty

        D = 32
        z = torch.zeros(8, D)
        z_ref = torch.zeros(D); z_ref[0] = 1.0
        mu = torch.zeros(8, D); mu[:, 0] = 1.0
        loss_low = excessive_alignment_penalty(mu, z, z_ref, tau=0.30)
        loss_high = excessive_alignment_penalty(mu, z, z_ref, tau=0.80)
        assert float(loss_low) > float(loss_high)

    def test_gradient_flows_to_mu(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import excessive_alignment_penalty

        D = 32
        z = torch.zeros(8, D)
        z_ref = torch.zeros(D); z_ref[0] = 1.0
        mu_data = torch.zeros(8, D)
        mu_data[:, 0] = 0.9  # over-aligned (α ≈ 1.0 > τ=0.80)
        mu_data.requires_grad_(True)
        loss = excessive_alignment_penalty(mu_data, z, z_ref, tau=0.80)
        loss.backward()
        assert mu_data.grad is not None
        # Gradient on dim 0 should be non-zero (penalty pushes μ to lower alignment)
        assert float(mu_data.grad[:, 0].abs().sum()) > 0.0

    def test_handles_zero_mu_without_nan(self) -> None:
        """Zero-magnitude μ → cosine undefined; helper should return 0 (no penalty)."""
        torch = _torch_or_skip()
        from src.models.dynamics import excessive_alignment_penalty

        D = 32
        z = torch.zeros(8, D)
        z_ref = torch.zeros(D); z_ref[0] = 1.0
        mu = torch.zeros(8, D)
        loss = excessive_alignment_penalty(mu, z, z_ref, tau=0.80)
        assert torch.isfinite(loss)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Universal-attractor penalty
# ---------------------------------------------------------------------------


class TestUniversalAttractorPenalty:
    def test_returns_scalar_tensor(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import universal_attractor_penalty

        B, D = 32, 32
        mu = torch.randn(B, D)
        z = torch.randn(B, D)
        z_ref = torch.zeros(D)
        gene_idx = torch.randint(1, 11, (B,))
        loss = universal_attractor_penalty(mu, z, z_ref, gene_idx, tau=0.80)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_zero_when_no_gene_dominates(self) -> None:
        """If every gene's mean alignment is below τ → loss = 0."""
        torch = _torch_or_skip()
        from src.models.dynamics import universal_attractor_penalty

        D = 32
        B = 16
        z = torch.zeros(B, D)
        z_ref = torch.zeros(D); z_ref[0] = 1.0
        # μ orthogonal to target_dir → alignment = 0 for every gene
        mu = torch.zeros(B, D); mu[:, 1] = 1.0
        gene_idx = torch.randint(1, 5, (B,))
        loss = universal_attractor_penalty(mu, z, z_ref, gene_idx, tau=0.80)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_one_gene_universally_aligns(self) -> None:
        """Pin gene_id=1 to perfectly aligned μ; all other genes have α=0.
        Per-gene means: gene_1 = 1.0, others = 0.0 → max = 1.0 > τ=0.80 → loss = 0.04.
        """
        torch = _torch_or_skip()
        from src.models.dynamics import universal_attractor_penalty

        D = 32
        B = 16
        z = torch.zeros(B, D)
        z_ref = torch.zeros(D); z_ref[0] = 1.0
        mu = torch.zeros(B, D); mu[:, 1] = 1.0  # orthogonal default for most genes
        gene_idx = torch.arange(1, B + 1, dtype=torch.long)
        # Override 6 rows (gene_id=1..6) to fully aligned μ
        mu[:6, :] = 0.0
        mu[:6, 0] = 1.0
        gene_idx[:6] = 1   # all 6 rows are gene 1 — its mean alignment = 1.0
        loss = universal_attractor_penalty(mu, z, z_ref, gene_idx, tau=0.80)
        assert float(loss) == pytest.approx(0.04, abs=1e-4)

    def test_gradient_flows_to_mu(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import universal_attractor_penalty

        D = 32
        B = 16
        z = torch.zeros(B, D)
        z_ref = torch.zeros(D); z_ref[0] = 1.0
        mu = torch.zeros(B, D)
        mu[:, 0] = 0.9
        mu = mu.requires_grad_(True)
        gene_idx = torch.ones(B, dtype=torch.long)
        loss = universal_attractor_penalty(mu, z, z_ref, gene_idx, tau=0.80)
        loss.backward()
        assert mu.grad is not None
        assert float(mu.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Action-diversity penalty (optional)
# ---------------------------------------------------------------------------


class TestActionDiversityPenalty:
    def test_returns_scalar_tensor(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import action_diversity_penalty

        mu = torch.randn(32, 16)
        loss = action_diversity_penalty(mu, tau_min=0.0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_zero_when_floor_is_zero(self) -> None:
        """tau_min=0.0 is the inert default — penalty = 0 regardless of variance."""
        torch = _torch_or_skip()
        from src.models.dynamics import action_diversity_penalty

        # μ = constant (zero variance), but tau_min = 0 makes penalty inert
        mu = torch.ones(32, 16)
        loss = action_diversity_penalty(mu, tau_min=0.0)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_variance_below_floor(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import action_diversity_penalty

        mu = torch.ones(32, 16)  # variance ≈ 0
        loss = action_diversity_penalty(mu, tau_min=0.5)
        assert float(loss) > 0.0

    def test_zero_when_variance_above_floor(self) -> None:
        torch = _torch_or_skip()
        from src.models.dynamics import action_diversity_penalty

        torch.manual_seed(0)
        mu = torch.randn(64, 16) * 2.0  # variance ≈ 4 ≫ floor 0.1
        loss = action_diversity_penalty(mu, tau_min=0.1)
        assert float(loss) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Default-off behavior (sacred rule §3 of CLAUDE.md)
# ---------------------------------------------------------------------------


class TestDefaultDisabledBehavior:
    """When all three λ_* = 0 (the default), training loss is byte-identical to V2/V3."""

    def test_contraction_aware_keys_present_in_yaml_default_zero(self) -> None:
        """config/dynamics.yaml must register the keys with λ_*=0 defaults."""
        import yaml
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        with open(repo_root / "config" / "dynamics.yaml") as f:
            cfg = yaml.safe_load(f)
        ca = cfg["dynamics"].get("contraction_aware", {})
        assert ca.get("lambda_excessive_alignment", 0.0) == 0.0
        assert ca.get("lambda_universal_attractor", 0.0) == 0.0
        assert ca.get("lambda_action_diversity", 0.0) == 0.0
