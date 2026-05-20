"""Tests for V3C dynamics utility metrics (Bucket U-A through U-G).

Each metric is exercised against synthetic dynamics with known geometry so
the audit's diagnostic claims are reproducible. We avoid loading the real
PerturbationDynamicsModel here — we use lightweight callables that share
its forward-signature contract: ``f(z, gene_idx) → (z_next, μ, log_var)``
where ``gene_idx`` is 1-indexed.

See V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md §3 for the
expected metric structure under each bucket.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Synthetic dynamics fixtures (no torch.nn.Module — just callables with the
# right forward-signature contract). Each fixture targets a known failure
# mode the audit must detect.
# ---------------------------------------------------------------------------


class _SyntheticDynamics:
    """Minimal stand-in for PerturbationDynamicsModel.

    Subclasses override ``mu_fn``. forward() returns the three-tuple the env
    and beam policy expect; log_var defaults to zeros (σ = 1).
    """

    def __init__(self, n_latent: int):
        self.n_latent = int(n_latent)

    def mu_fn(self, z: torch.Tensor, gene_idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, z, gene_idx):
        return self.forward(z, gene_idx)

    def forward(self, z, gene_idx):
        z_t = torch.as_tensor(z, dtype=torch.float32)
        g_t = torch.as_tensor(gene_idx, dtype=torch.long)
        mu = self.mu_fn(z_t, g_t)
        z_next = z_t + mu
        log_var = torch.zeros_like(mu)
        return z_next, mu, log_var

    def eval(self):
        return self


class NoOpDynamics(_SyntheticDynamics):
    """μ(z, g) ≈ 0 — the literal Soft-OT-no-op failure mode."""

    def mu_fn(self, z, gene_idx):
        return torch.zeros_like(z)


class UniversalAttractorDynamics(_SyntheticDynamics):
    """μ(z, g) = α · (z_ref − z) for *every* gene — universal-contraction case.

    All genes do the same thing: pull cells toward z_ref. This is the V2
    saturation signature.
    """

    def __init__(self, n_latent: int, z_ref: np.ndarray, alpha: float = 0.5):
        super().__init__(n_latent)
        self.z_ref = torch.as_tensor(z_ref, dtype=torch.float32)
        self.alpha = float(alpha)

    def mu_fn(self, z, gene_idx):
        return self.alpha * (self.z_ref.unsqueeze(0) - z)


class ActionDiscriminatingDynamics(_SyntheticDynamics):
    """μ(z, g) lives on a gene-specific direction.

    Gene 1 contracts toward z_ref; gene 2 *anti-contracts* (pushes away);
    gene 3 is near-null. Action choice MATTERS for the sign of motion —
    the bucket-U-D ideal signature where a planner must pick the right
    gene to make progress.
    """

    def __init__(self, n_latent: int, z_ref: np.ndarray):
        super().__init__(n_latent)
        self.z_ref = torch.as_tensor(z_ref, dtype=torch.float32)
        self.n_latent = n_latent

    def mu_fn(self, z, gene_idx):
        out = torch.zeros_like(z)
        toward = self.z_ref.unsqueeze(0) - z                       # (B, n_latent)
        for i, g in enumerate(gene_idx.tolist()):
            if g == 1:
                out[i] = 0.5 * toward[i]
            elif g == 2:
                out[i] = -0.3 * toward[i]                          # anti-contraction
            elif g == 3:
                out[i] = 0.001 * toward[i]                         # ~null magnitude
        return out


# ---------------------------------------------------------------------------
# U-D: Contraction geometry
# ---------------------------------------------------------------------------


def _sample_starts(n: int, n_latent: int, z_ref: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Place starts in a shell around z_ref so (z_ref - z) is non-degenerate.
    directions = rng.normal(size=(n, n_latent)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    radii = rng.uniform(2.0, 6.0, size=(n, 1)).astype(np.float32)
    return z_ref[None, :] + radii * directions


class TestContractionGeometry:
    def test_noop_dynamics_yields_zero_magnitude_and_undefined_alignment(self):
        from src.analysis.dynamics_utility import compute_contraction_geometry

        z_ref = np.zeros(8, dtype=np.float32)
        starts = _sample_starts(64, 8, z_ref, seed=0)
        result = compute_contraction_geometry(
            dynamics=NoOpDynamics(n_latent=8),
            z_starts=starts,
            z_ref=z_ref,
            n_genes=5,
            sample_label="ood_pool",
        )
        # No-op field: μ ≈ 0 everywhere, so ||μ|| median is ~0
        assert result["delta_magnitude_median"] == pytest.approx(0.0, abs=1e-5)
        # contraction_fraction is reported as null OR ~0.0 for zero-magnitude
        cf = result["contraction_fraction"]
        assert cf is None or cf == pytest.approx(0.0, abs=1e-3)
        # Diagnostic flag identifies the no-op pathology
        assert result["null_gene_fraction"] == pytest.approx(1.0, abs=1e-3)
        assert result["sample_label"] == "ood_pool"

    def test_universal_attractor_has_high_contraction_low_action_diversity(self):
        from src.analysis.dynamics_utility import compute_contraction_geometry

        rng = np.random.default_rng(1)
        z_ref = rng.normal(size=8).astype(np.float32)
        starts = _sample_starts(64, 8, z_ref, seed=2)
        result = compute_contraction_geometry(
            dynamics=UniversalAttractorDynamics(8, z_ref, alpha=0.5),
            z_starts=starts,
            z_ref=z_ref,
            n_genes=5,
            sample_label="ood_pool",
        )
        # All genes contract toward z_ref → contraction_fraction → 1.0
        assert result["contraction_fraction"] == pytest.approx(1.0, abs=1e-3)
        # All genes do the same thing → action_diversity_per_state ≈ 0
        assert result["action_diversity_per_state"] == pytest.approx(0.0, abs=1e-3)
        # Every gene is an attractor → gene_universality_max ≈ 1.0
        assert result["gene_universality_max"] == pytest.approx(1.0, abs=1e-3)
        # No "null" genes — every gene does the same contraction
        assert result["null_gene_fraction"] == pytest.approx(0.0, abs=1e-3)

    def test_action_discriminating_dynamics_has_moderate_diversity(self):
        from src.analysis.dynamics_utility import compute_contraction_geometry

        z_ref = np.zeros(8, dtype=np.float32)
        starts = _sample_starts(64, 8, z_ref, seed=3)
        result = compute_contraction_geometry(
            dynamics=ActionDiscriminatingDynamics(8, z_ref),
            z_starts=starts,
            z_ref=z_ref,
            n_genes=3,
            sample_label="ood_pool",
        )
        # Gene 1 contracts, gene 2 pushes orthogonally, gene 3 ≈ null:
        # contraction_fraction should be in (0.3, 0.7) — discriminating
        assert 0.30 < result["contraction_fraction"] < 0.70
        # Per-state std across genes should be clearly non-zero
        assert result["action_diversity_per_state"] > 0.20
        # gene_universality_max — gene 1 dominates contraction across states
        assert result["gene_universality_max"] >= 0.70
        # Gene 3 ≈ null → null_gene_fraction > 0
        assert result["null_gene_fraction"] >= 1 / 3 - 1e-3

    def test_required_output_keys_present(self):
        from src.analysis.dynamics_utility import compute_contraction_geometry

        z_ref = np.zeros(4, dtype=np.float32)
        starts = _sample_starts(16, 4, z_ref, seed=4)
        result = compute_contraction_geometry(
            dynamics=NoOpDynamics(n_latent=4),
            z_starts=starts,
            z_ref=z_ref,
            n_genes=3,
            sample_label="val_pairs",
        )
        required = {
            "sample_label",
            "n_starts", "n_genes",
            "alignment_cos_median", "alignment_cos_p25", "alignment_cos_p75",
            "alignment_cos_frac_above_0_5", "alignment_cos_frac_below_0",
            "contraction_fraction",
            "delta_magnitude_median", "delta_magnitude_p10", "delta_magnitude_p90",
            "action_diversity_per_state",
            "state_diversity_per_action",
            "gene_universality_max",
            "gene_universality_gini",
            "null_gene_fraction",
            "per_gene_mean_alignment",
        }
        assert required.issubset(set(result.keys()))
        assert result["sample_label"] == "val_pairs"
        assert result["n_starts"] == 16
        assert result["n_genes"] == 3


# ---------------------------------------------------------------------------
# U-E: Action heterogeneity and path diversity
# ---------------------------------------------------------------------------


class TestActionHeterogeneity:
    def test_single_gene_dominates_yields_low_entropy(self):
        from src.analysis.dynamics_utility import compute_action_heterogeneity

        # 100 starts, distance-greedy first action always picks gene 1
        first_actions_distance = np.ones(100, dtype=np.int64)
        first_actions_fused = np.ones(100, dtype=np.int64)
        beam_plans_d2 = [(1,)] * 100
        beam_plans_d3 = [(1, 1)] * 100

        result = compute_action_heterogeneity(
            n_genes=10,
            first_actions_distance=first_actions_distance,
            first_actions_fused=first_actions_fused,
            beam_plans_depth2_distance=beam_plans_d2,
            beam_plans_depth3_distance=beam_plans_d3,
        )

        assert result["first_action_entropy_distance"] == pytest.approx(0.0, abs=1e-9)
        assert result["first_action_entropy_fused"] == pytest.approx(0.0, abs=1e-9)
        assert result["first_action_top1_freq_fused"] == pytest.approx(1.0, abs=1e-9)
        assert result["first_action_top5_freq_fused"] == pytest.approx(1.0, abs=1e-9)
        assert result["first_action_gini_fused"] == pytest.approx(1.0 - 1.0 / 10, abs=1e-3)
        # Distance and fused agree completely
        assert result["distance_vs_fused_first_action_overlap"] == pytest.approx(1.0, abs=1e-9)
        # All plans identical → 1 unique plan over 100 starts → diversity 1/100
        assert result["path_diversity_depth2_distance"] == pytest.approx(1 / 100, abs=1e-9)

    def test_uniform_action_distribution_yields_max_entropy(self):
        from src.analysis.dynamics_utility import compute_action_heterogeneity

        n_genes = 8
        # 80 starts, each gene picked exactly 10 times → uniform
        first_actions = np.repeat(np.arange(1, n_genes + 1, dtype=np.int64), 10)
        beam_plans_d2 = [(int(g), int(g)) for g in first_actions]
        beam_plans_d3 = [(int(g), int(g), int(g)) for g in first_actions]

        result = compute_action_heterogeneity(
            n_genes=n_genes,
            first_actions_distance=first_actions,
            first_actions_fused=first_actions,
            beam_plans_depth2_distance=beam_plans_d2,
            beam_plans_depth3_distance=beam_plans_d3,
        )

        assert result["first_action_entropy_fused"] == pytest.approx(math.log(n_genes), abs=1e-6)
        assert result["first_action_top1_freq_fused"] == pytest.approx(1.0 / n_genes, abs=1e-9)
        assert result["first_action_top5_freq_fused"] == pytest.approx(5.0 / n_genes, abs=1e-9)
        assert result["first_action_gini_fused"] == pytest.approx(0.0, abs=1e-3)
        # All 80 starts have distinct (gene, gene) plans? No — only 8 unique tuples
        # path_diversity = unique_plans / n_starts = 8 / 80 = 0.1
        assert result["path_diversity_depth2_distance"] == pytest.approx(8 / 80, abs=1e-6)

    def test_distance_and_fused_differ_signals_reward_leverage(self):
        from src.analysis.dynamics_utility import compute_action_heterogeneity

        # 100 starts. Distance-greedy picks gene 1 always; fused-greedy picks
        # gene 2 in 40% of starts (e.g. because gene 1 has high tox).
        first_d = np.ones(100, dtype=np.int64)
        first_f = first_d.copy()
        first_f[:40] = 2  # 40 starts diverge to gene 2 under fused

        result = compute_action_heterogeneity(
            n_genes=10,
            first_actions_distance=first_d,
            first_actions_fused=first_f,
            beam_plans_depth2_distance=[(1,)] * 100,
            beam_plans_depth3_distance=[(1, 1)] * 100,
        )

        assert result["distance_vs_fused_first_action_overlap"] == pytest.approx(0.6, abs=1e-6)
        # Fused has two distinct actions (2 with freq 0.4 and 1 with freq 0.6)
        # → entropy = -(0.4 log 0.4 + 0.6 log 0.6) ≈ 0.6730
        expected_h = -(0.4 * math.log(0.4) + 0.6 * math.log(0.6))
        assert result["first_action_entropy_fused"] == pytest.approx(expected_h, abs=1e-6)

    def test_required_output_keys_present(self):
        from src.analysis.dynamics_utility import compute_action_heterogeneity

        result = compute_action_heterogeneity(
            n_genes=5,
            first_actions_distance=np.array([1, 2, 3, 1, 1], dtype=np.int64),
            first_actions_fused=np.array([1, 2, 3, 1, 2], dtype=np.int64),
            beam_plans_depth2_distance=[(1, 1), (2, 3), (3, 1), (1, 2), (1, 1)],
            beam_plans_depth3_distance=[(1, 1, 1), (2, 3, 4), (3, 1, 2), (1, 2, 3), (1, 1, 1)],
        )
        required = {
            "n_starts",
            "first_action_entropy_distance",
            "first_action_entropy_fused",
            "first_action_top1_freq_fused",
            "first_action_top5_freq_fused",
            "first_action_top10_freq_fused",
            "first_action_gini_fused",
            "distance_vs_fused_first_action_overlap",
            "path_diversity_depth2_distance",
            "path_diversity_depth3_distance",
            "top10_genes_fused",  # rank table
        }
        assert required.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# U-F: Reward leverage + Norman measured-combo realism
# ---------------------------------------------------------------------------


class TestRewardLeverage:
    def test_fused_reduces_safety_and_uncertainty_with_minor_success_cost(self):
        """The canonical V3C-desired field: D is load-bearing."""
        from src.analysis.dynamics_utility import compute_reward_leverage

        rollouts_distance = {
            "success_rate": 0.62,
            "mean_final_distance": 2.8,
            "mean_T_at_success": 2.1,
            "mean_tox_path": 0.45,
            "mean_common_essential_count": 0.18,
            "mean_unc_path_max": 0.91,
        }
        rollouts_fused = {
            "success_rate": 0.59,                  # −0.03 raw success (within Pareto ±0.03)
            "mean_final_distance": 2.85,           # only slightly worse
            "mean_T_at_success": 2.3,
            "mean_tox_path": 0.10,                 # ↓
            "mean_common_essential_count": 0.0,    # ↓ to zero
            "mean_unc_path_max": 0.75,             # ↓
        }
        result = compute_reward_leverage(
            cell_id="k3_bin8-10_splitood",
            rollouts_distance=rollouts_distance,
            rollouts_fused=rollouts_fused,
        )
        assert result["cell_id"] == "k3_bin8-10_splitood"
        assert result["delta_success_fused_minus_distance"] == pytest.approx(-0.03, abs=1e-9)
        assert result["delta_tox_path"] == pytest.approx(-0.35, abs=1e-9)
        assert result["delta_common_essential_count"] == pytest.approx(-0.18, abs=1e-9)
        assert result["delta_unc_path_max"] == pytest.approx(-0.16, abs=1e-9)
        # Pareto check: ≥ 2 axes improve, success degrades by ≤ 0.03, distance not blown out
        assert result["pareto_axes_improved"] >= 2
        assert result["raw_success_within_pareto_tolerance"] is True
        assert result["pareto_signal"] is True

    def test_fused_kills_raw_success_blocks_pareto(self):
        from src.analysis.dynamics_utility import compute_reward_leverage

        rollouts_distance = {
            "success_rate": 0.62, "mean_final_distance": 2.8, "mean_T_at_success": 2.1,
            "mean_tox_path": 0.45, "mean_common_essential_count": 0.18, "mean_unc_path_max": 0.91,
        }
        rollouts_fused = {
            "success_rate": 0.40,                   # −0.22, out of Pareto tolerance
            "mean_final_distance": 4.5,             # also worse
            "mean_T_at_success": 5.5, "mean_tox_path": 0.05,
            "mean_common_essential_count": 0.0, "mean_unc_path_max": 0.5,
        }
        result = compute_reward_leverage(
            cell_id="k3_bin8-10_splitood",
            rollouts_distance=rollouts_distance,
            rollouts_fused=rollouts_fused,
        )
        # Even though axes improve, the raw-success delta and distance regression
        # mean the field is being over-shaped, not displaying useful leverage.
        assert result["raw_success_within_pareto_tolerance"] is False
        assert result["pareto_signal"] is False
        assert result["concern_over_shaped"] is True

    def test_no_reward_leverage_when_distance_and_fused_identical(self):
        from src.analysis.dynamics_utility import compute_reward_leverage

        same = {
            "success_rate": 0.5, "mean_final_distance": 2.5, "mean_T_at_success": 2.0,
            "mean_tox_path": 0.2, "mean_common_essential_count": 0.05, "mean_unc_path_max": 0.6,
        }
        result = compute_reward_leverage(
            cell_id="k3_bin8-10_splitood",
            rollouts_distance=same,
            rollouts_fused=dict(same),
        )
        assert result["pareto_axes_improved"] == 0
        assert result["pareto_signal"] is False
        # Not "over-shaped" — just no leverage.
        assert result["concern_over_shaped"] is False


class TestNormanComboConsistency:
    def test_no_paths_in_combo_overlap_returns_na(self):
        """If no greedy/PPO 2-step paths land in measured combo set, return n/a."""
        from src.analysis.dynamics_utility import compute_norman_combo_consistency

        # Measured combo set has (1,2), (3,4); planner picks (5,6) (no overlap).
        z_dim = 4
        measured_combos = {
            "gene_idx_a": np.array([1, 3], dtype=np.int32),
            "gene_idx_b": np.array([2, 4], dtype=np.int32),
            "z_ctrl": np.zeros((2, z_dim), dtype=np.float32),
            "z_pert_ab": np.ones((2, z_dim), dtype=np.float32),
        }
        plans = [
            {"path": (5, 6), "z_predicted_post": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)},
        ]
        result = compute_norman_combo_consistency(
            plans=plans,
            measured_combos=measured_combos,
            check_ordered=True,
        )
        assert result["fraction_paths_with_measured_combo_overlap"] == pytest.approx(0.0)
        assert result["n_overlapping_paths"] == 0
        assert result["measured_combo_latent_consistency"] is None
        assert result["measured_combo_distance_consistency"] is None

    def test_overlapping_paths_compute_latent_consistency(self):
        from src.analysis.dynamics_utility import compute_norman_combo_consistency

        z_dim = 4
        measured_combos = {
            "gene_idx_a": np.array([1, 3], dtype=np.int32),
            "gene_idx_b": np.array([2, 4], dtype=np.int32),
            "z_ctrl": np.zeros((2, z_dim), dtype=np.float32),
            # Measured post-combo latents
            "z_pert_ab": np.stack(
                [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])],
                dtype=np.float32,
            ),
        }
        # Planner produces (1, 2) and (3, 4) — both in measured set
        plans = [
            {"path": (1, 2), "z_predicted_post": np.array([0.95, 0.0, 0.0, 0.0], dtype=np.float32)},
            {"path": (3, 4), "z_predicted_post": np.array([0.0, 0.95, 0.0, 0.0], dtype=np.float32)},
            {"path": (1, 2), "z_predicted_post": np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)},
        ]
        result = compute_norman_combo_consistency(
            plans=plans,
            measured_combos=measured_combos,
            check_ordered=True,
        )
        assert result["fraction_paths_with_measured_combo_overlap"] == pytest.approx(1.0)
        assert result["n_overlapping_paths"] == 3
        # All paths cosine ≈ 1.0 (predicted aligned with measured direction)
        assert result["measured_combo_latent_consistency"] == pytest.approx(1.0, abs=1e-5)

    def test_unordered_match_finds_swapped_pair(self):
        from src.analysis.dynamics_utility import compute_norman_combo_consistency

        measured_combos = {
            "gene_idx_a": np.array([1], dtype=np.int32),
            "gene_idx_b": np.array([2], dtype=np.int32),
            "z_ctrl": np.zeros((1, 4), dtype=np.float32),
            "z_pert_ab": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        }
        # Planner picks (2, 1) — reverse of measured (1, 2)
        plans = [{"path": (2, 1), "z_predicted_post": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}]
        ordered = compute_norman_combo_consistency(
            plans=plans, measured_combos=measured_combos, check_ordered=True,
        )
        unordered = compute_norman_combo_consistency(
            plans=plans, measured_combos=measured_combos, check_ordered=False,
        )
        assert ordered["fraction_paths_with_measured_combo_overlap"] == pytest.approx(0.0)
        assert unordered["fraction_paths_with_measured_combo_overlap"] == pytest.approx(1.0)

    def test_skips_cleanly_when_no_combo_data_available(self):
        from src.analysis.dynamics_utility import compute_norman_combo_consistency

        # mean_delta / soft_ot / 64D fields may not have combo_pairs available
        result = compute_norman_combo_consistency(
            plans=[{"path": (1, 2), "z_predicted_post": np.zeros(4, dtype=np.float32)}],
            measured_combos=None,
            check_ordered=True,
        )
        assert result["fraction_paths_with_measured_combo_overlap"] is None
        assert result["status"] == "no_combo_data"


# ---------------------------------------------------------------------------
# util_score composite — ranking aid only, never a verdict
# ---------------------------------------------------------------------------


class TestUtilScore:
    def _ideal_bucket_inputs(self) -> dict[str, Any]:
        """A field that maxes out every U-bucket. util_score → 1.0."""
        return {
            "u_a": {"val_pearson": 1.0},
            "u_b": {"beam_reach_at_K4_bin8_10_p15": 1.0,
                    "beam_reach_at_K5_bin8_10_p15": 1.0,
                    "beam_reach_at_K8_bin8_10_p15": 1.0},
            "u_c": {"cumulative_depth_leverage": 0.5},
            "u_d": {"contraction_fraction": 1.0, "gene_universality_max": 0.0},
            "u_e": {"first_action_entropy_fused": math.log(105),
                    "first_action_entropy_max_nats": math.log(105)},
            "u_f": {"distance_vs_fused_first_action_overlap": 0.0},
            "u_g": {"all_preconditions_pass": True},
        }

    def test_ideal_field_scores_1_0(self):
        from src.analysis.dynamics_utility import compute_utility_score

        score = compute_utility_score(self._ideal_bucket_inputs())
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_dead_field_scores_0(self):
        from src.analysis.dynamics_utility import compute_utility_score

        inputs = {
            "u_a": {"val_pearson": 0.0},
            "u_b": {"beam_reach_at_K4_bin8_10_p15": 0.0,
                    "beam_reach_at_K5_bin8_10_p15": 0.0,
                    "beam_reach_at_K8_bin8_10_p15": 0.0},
            "u_c": {"cumulative_depth_leverage": 0.0},
            "u_d": {"contraction_fraction": 0.0, "gene_universality_max": 1.0},
            "u_e": {"first_action_entropy_fused": 0.0, "first_action_entropy_max_nats": 1.0},
            "u_f": {"distance_vs_fused_first_action_overlap": 1.0},
            "u_g": {"all_preconditions_pass": False},
        }
        score = compute_utility_score(inputs)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_weights_sum_to_one(self):
        from src.analysis.dynamics_utility import UTIL_SCORE_WEIGHTS

        assert sum(UTIL_SCORE_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)

    def test_missing_bucket_returns_none_not_zero(self):
        """Robust to missing metrics (guardrail #5): null, not silent zero."""
        from src.analysis.dynamics_utility import compute_utility_score

        inputs = self._ideal_bucket_inputs()
        del inputs["u_d"]  # drop U-D entirely
        score = compute_utility_score(inputs)
        # When a foundational bucket is missing, return None (don't pretend).
        assert score is None

    def test_partial_inputs_with_skip_missing_returns_partial_score(self):
        """If caller opts in to partial scoring, we re-normalize over present buckets."""
        from src.analysis.dynamics_utility import compute_utility_score, UTIL_SCORE_WEIGHTS

        inputs = self._ideal_bucket_inputs()
        del inputs["u_d"]
        score = compute_utility_score(inputs, allow_missing=True)
        # All present buckets max out → score = 1.0 (renormalized)
        assert score == pytest.approx(1.0, abs=1e-9)
        # And missing U-D's 0.15 weight should be excluded — sanity check
        assert UTIL_SCORE_WEIGHTS["u_d"] == pytest.approx(0.15, abs=1e-9)

    def test_docstring_explicitly_marks_ranking_aid(self):
        """Guardrail #1: util_score docstring must mark it ranking-aid-not-verdict."""
        from src.analysis.dynamics_utility import compute_utility_score

        assert "ranking aid" in (compute_utility_score.__doc__ or "").lower()
        assert "not a verdict" in (compute_utility_score.__doc__ or "").lower()
