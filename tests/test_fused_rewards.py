"""Tests for V3B Phase 4 fused rewards: safety_path_freeband (B+C),
uncertainty_aware (D), biorealistic_fused (B+C+D).

Validates:
* Reductions: B+C → B when safety=0; D → V2 terminal_only when unc=0; B+C+D → B+C when λ_unc=0.
* Additivity of fused terms.
* Numerical finite-ness on truncation/failure.
* Missing-Chronos neutrality (env tox/ce arrays missing → zero penalty).
* Env unc accumulator: max across non-NOOP steps; mean across same.
* Reward-aware greedy with uncertainty: name suffix, scoring respects λ_unc_path.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rl.biology_rewards import (
    biorealistic_fused_reward,
    path_length_freeband_reward,
    per_step_uncertainty_scalar,
    safety_aware_reward,
    safety_path_freeband_reward,
    uncertainty_aware_reward,
)
from src.rl.reward import compute_reward


class TestUncertaintyScalar:
    def test_zero_log_var_gives_unit_sigma_mean(self) -> None:
        # exp(0.5·0) = 1 → mean = 1.0
        out = per_step_uncertainty_scalar(np.zeros(8), reduce="mean_sigma")
        assert out == pytest.approx(1.0)

    def test_max_sigma_picks_largest(self) -> None:
        log_var = np.array([0.0, 0.0, 2.0, 0.0])  # exp(0.5·2)=2.718
        out = per_step_uncertainty_scalar(log_var, reduce="max_sigma")
        assert out == pytest.approx(np.exp(1.0))

    def test_clip_bounds_apply(self) -> None:
        # log_var = 100 should clip to clip_max
        out = per_step_uncertainty_scalar(np.full(4, 100.0), reduce="mean_sigma", clip_max=3.0)
        assert out == pytest.approx(np.exp(1.5))

    def test_mean_log_var_signed(self) -> None:
        out = per_step_uncertainty_scalar(np.array([-1.0, 0.0, 1.0]), reduce="mean_log_var")
        assert out == pytest.approx(0.0)

    def test_unknown_reduce_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reduce"):
            per_step_uncertainty_scalar(np.zeros(4), reduce="bogus")


class TestUncertaintyAwareReward:
    def test_zero_unc_recovers_v2_terminal_only(self) -> None:
        # Variant D with λ_unc=0 + unc_path_max=0 should match terminal_only_step_cost.
        z = np.zeros(4)
        r_d = compute_reward(
            z, z, action=2, noop_idx=10, reward_mode="uncertainty_aware",
            terminated=True, is_success=True, step_idx=3, beta_step_cost=0.05,
            lambda_unc_path=0.0, unc_path_max=0.0,
        )
        r_v2 = compute_reward(
            z, z, action=2, noop_idx=10, reward_mode="terminal_only_step_cost",
            terminated=True, is_success=True, step_idx=3, beta_step_cost=0.05,
            lambda_sparse=0.0,
        )
        assert r_d == pytest.approx(r_v2)

    def test_monotonic_in_unc_path_max(self) -> None:
        r0 = uncertainty_aware_reward(
            is_success=True, terminated=True, truncated=False, step_idx=1,
            unc_path_max=0.0, lambda_unc=0.10,
        )
        r1 = uncertainty_aware_reward(
            is_success=True, terminated=True, truncated=False, step_idx=1,
            unc_path_max=2.0, lambda_unc=0.10,
        )
        assert r1 < r0
        assert r0 - r1 == pytest.approx(0.20)  # 0.10 · 2.0

    def test_truncation_finite(self) -> None:
        r = uncertainty_aware_reward(
            is_success=False, terminated=False, truncated=True, step_idx=8,
            unc_path_max=4.5, lambda_unc=0.05,
        )
        assert np.isfinite(r)


class TestSafetyPathFreebandReward:
    def test_zero_safety_recovers_freeband(self) -> None:
        r_bc = safety_path_freeband_reward(
            is_success=True, terminated=True, truncated=False, step_idx=4,
            tox_path=0.0, common_essential_count=0,
            free_steps=3, mild_until=5, mild_beta=0.02, heavy_beta=0.10,
            lambda_tox=0.10, lambda_ce=0.05, success_bonus=1.0,
        )
        r_b = path_length_freeband_reward(
            is_success=True, terminated=True, truncated=False, step_idx=4,
            free_steps=3, mild_until=5, mild_beta=0.02, heavy_beta=0.10,
            success_bonus=1.0,
        )
        assert r_bc == pytest.approx(r_b)

    def test_safety_penalty_subtracts(self) -> None:
        # T=3 (free band), tox=1.5, ce=2 → R = 1 − 0 − 0.10·1.5 − 0.05·2 = 1 − 0.25 = 0.75
        r = safety_path_freeband_reward(
            is_success=True, terminated=True, truncated=False, step_idx=3,
            tox_path=1.5, common_essential_count=2,
            lambda_tox=0.10, lambda_ce=0.05,
        )
        assert r == pytest.approx(0.75)

    def test_truncation_finite(self) -> None:
        r = safety_path_freeband_reward(
            is_success=False, terminated=False, truncated=True, step_idx=8,
            tox_path=5.0, common_essential_count=4,
            lambda_tox=0.10, lambda_ce=0.05,
        )
        # T=8 in heavy band → penalty=0.04+0.30=0.34. R = 0 − 0.34 − 0.5 − 0.2 = −1.04
        assert r == pytest.approx(-1.04)
        assert np.isfinite(r)


class TestBiorealisticFusedReward:
    def test_zero_all_reduces_to_freeband(self) -> None:
        # λ_tox = λ_ce = λ_unc = 0 → reduces to path_length_freeband
        r_bcd = biorealistic_fused_reward(
            is_success=True, terminated=True, truncated=False, step_idx=4,
            tox_path=1.0, common_essential_count=1, unc_path_max=2.0,
            lambda_tox=0.0, lambda_ce=0.0, lambda_unc=0.0,
        )
        r_b = path_length_freeband_reward(
            is_success=True, terminated=True, truncated=False, step_idx=4,
        )
        assert r_bcd == pytest.approx(r_b)

    def test_lambda_unc_zero_reduces_to_BC(self) -> None:
        r_bcd = biorealistic_fused_reward(
            is_success=True, terminated=True, truncated=False, step_idx=4,
            tox_path=1.5, common_essential_count=2, unc_path_max=2.0,
            lambda_tox=0.10, lambda_ce=0.05, lambda_unc=0.0,
        )
        r_bc = safety_path_freeband_reward(
            is_success=True, terminated=True, truncated=False, step_idx=4,
            tox_path=1.5, common_essential_count=2,
            lambda_tox=0.10, lambda_ce=0.05,
        )
        assert r_bcd == pytest.approx(r_bc)

    def test_additivity_of_fused_terms(self) -> None:
        # R = 1·success − path − λ_tox·tox − λ_ce·ce − λ_unc·unc
        # T=5 (mild band): penalty = 0.02·2 = 0.04
        # safety: 0.10·1.0 + 0.05·1 = 0.15
        # unc: 0.05·3.0 = 0.15
        # R = 1 − 0.04 − 0.15 − 0.15 = 0.66
        r = biorealistic_fused_reward(
            is_success=True, terminated=True, truncated=False, step_idx=5,
            tox_path=1.0, common_essential_count=1, unc_path_max=3.0,
            lambda_tox=0.10, lambda_ce=0.05, lambda_unc=0.05,
        )
        assert r == pytest.approx(0.66)

    def test_truncation_finite(self) -> None:
        r = biorealistic_fused_reward(
            is_success=False, terminated=False, truncated=True, step_idx=8,
            tox_path=3.0, common_essential_count=3, unc_path_max=4.0,
            lambda_tox=0.10, lambda_ce=0.05, lambda_unc=0.05,
        )
        # T=8 heavy: penalty=0.34; safety=0.30+0.15=0.45; unc=0.20. R = 0 − 0.99 = −0.99
        assert r == pytest.approx(-0.99)
        assert np.isfinite(r)

    def test_mid_episode_is_zero(self) -> None:
        r = biorealistic_fused_reward(
            is_success=False, terminated=False, truncated=False, step_idx=2,
            tox_path=10.0, common_essential_count=5, unc_path_max=8.0,
            lambda_tox=0.5, lambda_ce=0.5, lambda_unc=0.5,
        )
        assert r == 0.0


class TestComputeRewardDispatchFused:
    def test_dispatch_safety_path_freeband(self) -> None:
        z = np.zeros(4)
        r = compute_reward(
            z, z, action=2, noop_idx=10, reward_mode="safety_path_freeband",
            terminated=True, is_success=True, step_idx=3,
            tox_path=0.5, common_essential_count=1, lambda_tox=0.10, lambda_ce=0.05,
        )
        # T=3 free band, R = 1 − 0 − 0.10·0.5 − 0.05·1 = 0.90
        assert r == pytest.approx(0.90)

    def test_dispatch_uncertainty_aware(self) -> None:
        z = np.zeros(4)
        r = compute_reward(
            z, z, action=2, noop_idx=10, reward_mode="uncertainty_aware",
            terminated=True, is_success=True, step_idx=2, beta_step_cost=0.05,
            unc_path_max=3.0, lambda_unc_path=0.10,
        )
        # R = 1 − 0.10 − 0.30 = 0.60
        assert r == pytest.approx(0.60)

    def test_dispatch_biorealistic_fused(self) -> None:
        z = np.zeros(4)
        r = compute_reward(
            z, z, action=2, noop_idx=10, reward_mode="biorealistic_fused",
            terminated=True, is_success=True, step_idx=4,
            tox_path=1.0, common_essential_count=1, unc_path_max=2.0,
            lambda_tox=0.10, lambda_ce=0.05, lambda_unc_path=0.05,
        )
        # T=4 mild: penalty=0.02; safety=0.10+0.05=0.15; unc=0.10. R = 1 − 0.27 = 0.73
        assert r == pytest.approx(0.73)

    def test_multi_objective_alias_works(self) -> None:
        z = np.zeros(4)
        r1 = compute_reward(
            z, z, action=2, noop_idx=10, reward_mode="biorealistic_fused",
            terminated=True, is_success=True, step_idx=3,
            tox_path=1.0, common_essential_count=1, unc_path_max=2.0,
        )
        r2 = compute_reward(
            z, z, action=2, noop_idx=10, reward_mode="multi_objective",
            terminated=True, is_success=True, step_idx=3,
            tox_path=1.0, common_essential_count=1, unc_path_max=2.0,
        )
        assert r1 == pytest.approx(r2)

    def test_unknown_mode_raises_with_full_list(self) -> None:
        z = np.zeros(4)
        with pytest.raises(ValueError, match="Unknown reward_mode"):
            compute_reward(z, z, action=0, noop_idx=10, reward_mode="bogus_mode",
                           terminated=True, is_success=True)

    def test_v2_modes_unchanged(self) -> None:
        # Ensure absolute_distance & terminal_only_step_cost still produce same output.
        z = np.zeros(4)
        r_abs = compute_reward(z, z, action=5, noop_idx=10, reward_mode="absolute_distance",
                                terminated=True, is_success=True, lambda_sparse=0.0)
        assert r_abs == pytest.approx(0.0)
        r_term = compute_reward(z, z, action=5, noop_idx=10, reward_mode="terminal_only_step_cost",
                                 terminated=True, is_success=True, step_idx=3, lambda_sparse=0.0)
        assert r_term == pytest.approx(1.0 - 0.05 * 3)


class TestEnvUncertaintyAccumulator:
    """Verify env accumulates unc_path_max / unc_path_mean correctly over non-NOOP steps."""

    def _make_env_with_unc_dynamics(self):
        """Mock dynamics that returns a stable log_var per call (deterministic)."""
        import torch
        class _UncDyn:
            def __init__(self):
                self.call_count = 0
            def __call__(self, z, g):
                self.call_count += 1
                mu = -0.1 * z
                # Vary log_var per call so unc_step changes: call k → log_var = k * 0.5
                log_var = torch.full_like(z, float(self.call_count) * 0.5)
                return z + mu, mu, log_var
            def eval(self): return self
            def to(self, device): return self

        from src.rl.environment import CellReprogrammingEnv
        n_latent = 4
        z_ref = np.zeros(n_latent, dtype=np.float32)
        start_pool = np.ones((5, n_latent), dtype=np.float32) * 5.0
        env = CellReprogrammingEnv(
            dynamics_model=_UncDyn(),
            z_reference_centroid=z_ref,
            epsilon_success=0.01,
            n_genes=10,
            max_steps=3,
            lambda_sparse=0.0,
            start_state_strategy="random_perturbation",
            distance_metric="l2",
            start_pool_latents=start_pool,
            reward_mode="uncertainty_aware",
            lambda_unc_path=0.05,
            uncertainty_reduce="mean_sigma",
            seed=42,
        )
        return env

    def test_reset_zeroes_unc_accumulators(self) -> None:
        env = self._make_env_with_unc_dynamics()
        _, info = env.reset(seed=0)
        assert info["unc_path_max"] == 0.0
        assert info["unc_path_mean"] == 0.0

    def test_unc_max_tracks_largest_step(self) -> None:
        env = self._make_env_with_unc_dynamics()
        env.reset(seed=0)
        # Each call gives log_var = k*0.5 → unc_step = exp(0.25k) — monotonically increasing.
        _, _, _, _, info1 = env.step(0)
        u1 = info1["unc_path_max"]
        _, _, _, _, info2 = env.step(1)
        u2 = info2["unc_path_max"]
        assert u2 > u1, f"expected unc_max to grow: u1={u1}, u2={u2}"

    def test_unc_mean_is_running_average(self) -> None:
        env = self._make_env_with_unc_dynamics()
        env.reset(seed=0)
        env.step(0)
        _, _, _, _, info = env.step(1)
        # mean = (unc1 + unc2)/2
        u_max = info["unc_path_max"]
        u_mean = info["unc_path_mean"]
        assert 0 < u_mean <= u_max

    def test_v2_mode_does_not_track_unc(self) -> None:
        """In V2 modes, unc accumulators should stay 0 (no log_var extraction)."""
        import torch
        class _MockDyn:
            def __call__(self, z, g):
                return z + (-0.1 * z), -0.1 * z, torch.zeros_like(z)
            def eval(self): return self
            def to(self, device): return self

        from src.rl.environment import CellReprogrammingEnv
        n_latent = 4
        env = CellReprogrammingEnv(
            dynamics_model=_MockDyn(),
            z_reference_centroid=np.zeros(n_latent, dtype=np.float32),
            epsilon_success=0.01,
            n_genes=10,
            max_steps=2,
            start_pool_latents=np.ones((5, n_latent), dtype=np.float32) * 5.0,
            reward_mode="terminal_only_step_cost",
            seed=42,
        )
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        # Even though dynamics emits log_var, V2 mode doesn't track it (lambda_unc=0, mode != phase4).
        assert info["unc_path_max"] == 0.0


class TestGreedyUncertaintyAware:
    def _toy_dynamics(self):
        """Toy dynamics with stable log_var per gene action."""
        import torch
        class _Toy:
            def __call__(self, z_batch, g_batch):
                delta = -0.5 * z_batch
                # log_var depends on gene index: gene 0 has low unc, gene 1 high unc
                lv = torch.zeros_like(z_batch)
                for i, g in enumerate(g_batch):
                    if int(g.item()) == 2:  # gene_idx 2 ≡ action 1 (after +1 shift)
                        lv[i] = 3.0  # high uncertainty
                return z_batch + delta, delta, lv
            def eval(self): return self
            def to(self, device): return self
        return _Toy()

    def test_name_suffix_includes_unc(self) -> None:
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        n_genes = 4
        policy = GreedyDynamicsBeamPolicy(
            self._toy_dynamics(), n_genes=n_genes, z_ref=np.zeros(4),
            noop_idx=n_genes, depth=3,
            lambda_unc_path=0.5,
        )
        assert policy.name == "greedy_dyn_3_unc"

    def test_unc_disabled_when_lambda_zero(self) -> None:
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        n_genes = 4
        policy = GreedyDynamicsBeamPolicy(
            self._toy_dynamics(), n_genes=n_genes, z_ref=np.zeros(4),
            noop_idx=n_genes, depth=3,
            lambda_unc_path=0.0,
        )
        # No "unc" suffix when λ=0
        assert "unc" not in policy.name

    def test_unc_aware_greedy_prefers_low_unc_action(self) -> None:
        """Two equal-distance-reducing actions; one has high σ — greedy with λ_unc should avoid it."""
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        z = np.ones(4, dtype=np.float32) * 5.0
        n_genes = 4
        mask = np.zeros(n_genes + 1, dtype=bool)
        mask[0] = True  # gene action 0 (gene_idx=1, low unc)
        mask[1] = True  # gene action 1 (gene_idx=2, high unc)
        mask[n_genes] = True  # noop

        # Moderate λ_unc forces the policy to prefer action 0 (low unc gene) over action 1
        # (high unc gene), without making noop look better than the lower-distance gene action.
        # gene 0: dist ≈ 5, unc ≈ 1.0 → score = 5 + 0.5·1 = 5.5
        # gene 1: dist ≈ 5, unc ≈ exp(1.5) ≈ 4.48 → score = 5 + 0.5·4.48 = 7.24
        # noop:   dist = 10, unc = 0 → score = 10
        policy = GreedyDynamicsBeamPolicy(
            self._toy_dynamics(), n_genes=n_genes,
            z_ref=np.zeros(4, dtype=np.float32),
            noop_idx=n_genes, depth=1,
            lambda_unc_path=0.5,
            uncertainty_reduce="mean_sigma",
        )
        a = policy.select_action(z, mask, {})
        assert a == 0, f"expected action 0 (low-unc); picked {a}"
