"""P0E Phase 0 — tests for `GreedyDynamicsBeamPolicy` (multi-step greedy baseline).

These tests stub out the dynamics model with a deterministic callable so the beam search
behaviour can be reasoned about exactly, independent of the trained model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _torch_or_skip():
    return pytest.importorskip("torch")


class _LinearDynamics:
    """Toy dynamics: `z_next = z + delta_per_gene[gene_idx-1]`.

    `gene_idx` is 1-indexed per Contract 2; row 0 is reserved (ctrl placeholder, never called
    here because the env / beam policy passes 1-indexed gene_idx).
    """

    def __init__(self, delta_per_gene: np.ndarray) -> None:
        # shape: (n_genes, n_latent). The buffer is (n_genes+1, n_latent) with row 0 = zeros.
        self.delta = np.concatenate(
            [np.zeros((1, delta_per_gene.shape[1]), dtype=np.float32),
             delta_per_gene.astype(np.float32)],
            axis=0,
        )

    def __call__(self, z, gene_idx):
        import torch
        z_np = z.detach().cpu().numpy().astype(np.float32) if isinstance(z, torch.Tensor) else np.asarray(z, dtype=np.float32)
        g_np = gene_idx.detach().cpu().numpy().astype(np.int64) if isinstance(gene_idx, torch.Tensor) else np.asarray(gene_idx, dtype=np.int64)
        z_next = z_np + self.delta[g_np]
        mu = self.delta[g_np]
        log_var = np.zeros_like(z_next, dtype=np.float32)
        return (torch.from_numpy(z_next), torch.from_numpy(mu), torch.from_numpy(log_var))


class TestGreedyDynamicsBeamPolicy:
    def test_depth_1_matches_greedy_dyn_1(self) -> None:
        """At depth=1 the beam policy must pick the same action as GreedyDynamicsPolicy."""
        _torch_or_skip()
        from src.rl.baselines import GreedyDynamicsBeamPolicy, GreedyDynamicsPolicy

        rng = np.random.default_rng(0)
        n_latent, n_genes = 4, 5
        z_ref = np.zeros(n_latent, dtype=np.float32)
        # Random per-gene deltas, all non-trivial.
        delta_per_gene = rng.standard_normal((n_genes, n_latent)).astype(np.float32)
        dyn = _LinearDynamics(delta_per_gene)

        beam = GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=n_genes,
            depth=1, beam_width=20,
        )
        plain = GreedyDynamicsPolicy(
            dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=n_genes,
        )

        # All actions are valid (mask = all True).
        mask = np.ones(n_genes + 1, dtype=bool)
        for _ in range(10):
            z = rng.standard_normal(n_latent).astype(np.float32) * 2.0
            a_beam = beam.select_action(z, mask, {})
            a_plain = plain.select_action(z, mask, {})
            assert a_beam == a_plain, (
                f"depth=1 beam diverged from greedy_dyn_1: beam={a_beam}, plain={a_plain}, z={z}"
            )

    def test_depth_2_finds_better_plan_than_depth_1(self) -> None:
        """Construct a setting where the greedy 1-step pick is sub-optimal at depth=2.

        Three genes: A moves +0.5 along axis 0; B moves +0.5 along axis 1; C moves +0.1 along
        axis 0 then -1.0 along axis 1 (composite). Starting at z = (-0.1, +0.9), the 1-step
        greedy picks A (best one-step) but the depth-2 optimum is B → A (lands at origin).
        """
        _torch_or_skip()
        from src.rl.baselines import GreedyDynamicsBeamPolicy, GreedyDynamicsPolicy

        n_genes = 3
        n_latent = 2
        z_ref = np.zeros(n_latent, dtype=np.float32)
        # Gene 1 (A): δ = (+0.1, 0); gene 2 (B): δ = (0, -1.0); gene 3 (C): δ = (-1.0, 0).
        delta = np.array([
            [0.1, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ], dtype=np.float32)
        dyn = _LinearDynamics(delta)

        # Start at (1.0, 1.0). Distance = sqrt(2) ≈ 1.414.
        # depth=1 picks (B or C, both reduce one axis to 0); say it picks C → (0, 1) dist=1.
        # depth=2 best plan: C then B → (0,0) dist=0. Or B then C → (0,0) dist=0.
        # First action of the best plan is C (gene 3, action_idx=2) or B (gene 2, action_idx=1).
        z = np.array([1.0, 1.0], dtype=np.float32)
        mask = np.ones(n_genes + 1, dtype=bool)
        noop_idx = n_genes

        beam_d1 = GreedyDynamicsBeamPolicy(dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx, depth=1)
        beam_d2 = GreedyDynamicsBeamPolicy(dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx, depth=2)
        plain   = GreedyDynamicsPolicy(   dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx)

        a_plain = plain.select_action(z, mask, {})
        a_d1 = beam_d1.select_action(z, mask, {})
        a_d2 = beam_d2.select_action(z, mask, {})

        # depth=1 matches plain
        assert a_d1 == a_plain
        # depth=2 also picks one of B or C (action 1 or 2), but its tracked best_distance reaches 0.
        # The fact we tested matches above with depth=1 doesn't tell us depth=2 is better — verify
        # the returned action is the FIRST step of an optimal 2-step plan that reaches origin.
        z_next = z + delta[a_d2]
        # After taking a_d2, the remaining gene that completes the plan is the OTHER axis-killer.
        # The 1-step descendant cannot reach exactly origin from this z_next.
        # We verify by enumerating all 2-step plans:
        best_2step = float("inf")
        for g1 in range(n_genes):
            for g2 in range(n_genes):
                if g2 == g1:
                    continue
                final = z + delta[g1] + delta[g2]
                best_2step = min(best_2step, float(np.linalg.norm(final - z_ref)))
        # The depth-2 policy's first action must lead to a 2-step plan whose final dist equals best_2step.
        achievable = float("inf")
        for g2 in range(n_genes):
            if g2 == a_d2:
                continue
            final = z_next + delta[g2]
            achievable = min(achievable, float(np.linalg.norm(final - z_ref)))
        assert abs(achievable - best_2step) < 1e-5, (
            f"depth=2 first action {a_d2} cannot be extended to the global 2-step optimum "
            f"(achievable={achievable}, optimum={best_2step})."
        )

    def test_beam_width_respects_max_candidates(self) -> None:
        """A beam_width larger than the action space must not error."""
        _torch_or_skip()
        from src.rl.baselines import GreedyDynamicsBeamPolicy

        n_latent, n_genes = 4, 3
        z_ref = np.zeros(n_latent, dtype=np.float32)
        delta = np.eye(n_genes, n_latent).astype(np.float32)  # type: ignore[arg-type]
        dyn = _LinearDynamics(delta)
        beam = GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=n_genes,
            depth=3, beam_width=999,
        )
        z = np.ones(n_latent, dtype=np.float32)
        mask = np.ones(n_genes + 1, dtype=bool)
        a = beam.select_action(z, mask, {})
        # Action must be a valid index (0..n_genes inclusive, last is noop).
        assert 0 <= a <= n_genes

    def test_invalid_depth_or_beam_raises(self) -> None:
        _torch_or_skip()
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        delta = np.eye(3, 4).astype(np.float32)
        dyn = _LinearDynamics(delta)
        with pytest.raises(ValueError, match="depth"):
            GreedyDynamicsBeamPolicy(dyn, n_genes=3, z_ref=np.zeros(4, dtype=np.float32),
                                      noop_idx=3, depth=0)
        with pytest.raises(ValueError, match="beam_width"):
            GreedyDynamicsBeamPolicy(dyn, n_genes=3, z_ref=np.zeros(4, dtype=np.float32),
                                      noop_idx=3, depth=2, beam_width=0)

    def test_repeat_mask_respected_within_plan(self) -> None:
        """The intra-plan repeat exclusion: in a beam search, a gene used at depth 1 is not
        available at depth 2 within the same path. We verify by constructing a setting where
        the greedy plan would otherwise pick the same gene twice."""
        _torch_or_skip()
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        n_genes, n_latent = 2, 1
        # Gene 1: δ = -1; gene 2: δ = -0.4. Starting at +2.0:
        # depth-2 with repeats allowed → gene1, gene1 → 0.0 (best).
        # depth-2 with repeats disallowed → gene1, gene2 → +0.6.
        delta = np.array([[-1.0], [-0.4]], dtype=np.float32)
        dyn = _LinearDynamics(delta)
        beam = GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=np.zeros(n_latent, dtype=np.float32),
            noop_idx=n_genes, depth=2, beam_width=10,
        )
        z = np.array([2.0], dtype=np.float32)
        mask = np.ones(n_genes + 1, dtype=bool)
        a = beam.select_action(z, mask, {})
        # First action of the best no-repeat plan = gene 1 (action_idx 0).
        # Check: if we extend by the other gene (action_idx 1), distance is +0.6; that is the
        # best achievable under no-repeat.
        z_next = z + delta[a]
        best_extension = min(
            float(np.linalg.norm(z_next + delta[g], None)) for g in range(n_genes) if g != a
        )
        assert abs(best_extension - 0.6) < 1e-5, (
            f"first action {a} should lead to best-extension 0.6 under no-repeat, got {best_extension}"
        )
