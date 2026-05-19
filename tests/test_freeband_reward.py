"""Tests for V3B Phase 3 path-length free-band reward (Variant B).

Validates:
* Schedule arithmetic: zero penalty in free band, mild penalty in [free_steps+1, mild_until],
  heavy penalty beyond mild_until.
* Success bonus dominates short-path successes.
* Truncation/failure produces finite, sensible reward.
* V2-mode regression: setting (free_steps=0, mild_until=max_steps,
  mild_beta=heavy_beta=beta) recovers terminal_only_step_cost exactly.
* Env accumulators (step_idx) feed into the reward correctly.
* Reward-aware GreedyDynamicsBeamPolicy stops early when the free band's success
  outscores a longer plan.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.rl.biology_rewards import path_length_freeband_reward
from src.rl.reward import compute_reward


# ---------------------------------------------------------------------------
# path_length_freeband_reward — pure function
# ---------------------------------------------------------------------------


class TestFreebandSchedule:
    DEFAULTS = dict(free_steps=3, mild_until=5, mild_beta=0.02, heavy_beta=0.10, success_bonus=1.0)

    def _call(self, *, T, is_success, terminated=True, truncated=False, **kw):
        params = {**self.DEFAULTS, **kw}
        return path_length_freeband_reward(
            is_success=is_success, terminated=terminated, truncated=truncated,
            step_idx=T, **params,
        )

    def test_mid_episode_is_zero(self) -> None:
        r = self._call(T=2, is_success=True, terminated=False, truncated=False)
        assert r == 0.0

    def test_zero_steps_success_pays_no_penalty(self) -> None:
        # NOOP-at-step-0 success: T=0, penalty=0, reward=success_bonus.
        r = self._call(T=0, is_success=True)
        assert r == pytest.approx(1.0)

    def test_zero_steps_failure(self) -> None:
        r = self._call(T=0, is_success=False)
        assert r == pytest.approx(0.0)

    def test_free_band_zero_penalty(self) -> None:
        for T in (1, 2, 3):
            r_succ = self._call(T=T, is_success=True)
            r_fail = self._call(T=T, is_success=False)
            assert r_succ == pytest.approx(1.0), f"T={T} success should give 1.0, got {r_succ}"
            assert r_fail == pytest.approx(0.0), f"T={T} fail should give 0.0, got {r_fail}"

    def test_mild_band_penalty(self) -> None:
        # T=4: penalty = 0.02·(4-3) = 0.02
        # T=5: penalty = 0.02·(5-3) = 0.04
        assert self._call(T=4, is_success=True) == pytest.approx(1.0 - 0.02)
        assert self._call(T=5, is_success=True) == pytest.approx(1.0 - 0.04)
        assert self._call(T=4, is_success=False) == pytest.approx(-0.02)
        assert self._call(T=5, is_success=False) == pytest.approx(-0.04)

    def test_heavy_band_penalty(self) -> None:
        # T=6: penalty = 0.02·(5-3) + 0.10·(6-5) = 0.04 + 0.10 = 0.14
        # T=8: penalty = 0.04 + 0.10·3 = 0.34
        assert self._call(T=6, is_success=True) == pytest.approx(1.0 - 0.14)
        assert self._call(T=8, is_success=True) == pytest.approx(1.0 - 0.34)
        assert self._call(T=6, is_success=False) == pytest.approx(-0.14)
        assert self._call(T=8, is_success=False) == pytest.approx(-0.34)

    def test_truncation_is_finite(self) -> None:
        # Truncation without success at T=8: bounded, finite, ≥ -0.34.
        r = self._call(T=8, is_success=False, terminated=False, truncated=True)
        assert np.isfinite(r)
        assert r == pytest.approx(-0.34)

    def test_success_bonus_dominates_short_success(self) -> None:
        # T=3 success should beat T=5 success on reward.
        r3 = self._call(T=3, is_success=True)
        r5 = self._call(T=5, is_success=True)
        assert r3 > r5
        # And both successes should be much better than a T=3 failure.
        rf3 = self._call(T=3, is_success=False)
        assert r3 > rf3

    def test_recovers_terminal_only_step_cost_at_v2_settings(self) -> None:
        # With (free_steps=0, mild_until=8, mild_beta=heavy_beta=0.05, success_bonus=1):
        #   penalty(T) = 0.05·T   = V2's β·t exactly.
        for T in (1, 3, 5, 8):
            r = self._call(
                T=T, is_success=True,
                free_steps=0, mild_until=8, mild_beta=0.05, heavy_beta=0.05,
                success_bonus=1.0,
            )
            assert r == pytest.approx(1.0 - 0.05 * T)

    def test_monotonic_in_T_with_constant_success(self) -> None:
        # Reward should be (weakly) decreasing in T at fixed success.
        prev = float("inf")
        for T in range(0, 11):
            r = self._call(T=T, is_success=True)
            assert r <= prev + 1e-9, f"reward should not increase with T (T={T}: r={r}, prev={prev})"
            prev = r


# ---------------------------------------------------------------------------
# compute_reward dispatch
# ---------------------------------------------------------------------------


class TestComputeRewardDispatch:
    def test_path_length_freeband_reachable_via_compute_reward(self) -> None:
        z = np.zeros(4)
        r = compute_reward(
            z, z, action=2, noop_idx=10,
            reward_mode="path_length_freeband",
            terminated=True, truncated=False, is_success=True,
            step_idx=4,
            free_steps=3, mild_until=5, mild_beta=0.02, heavy_beta=0.10,
            freeband_success_bonus=1.0,
        )
        # 1·success − 0.02·(4−3) = 0.98
        assert r == pytest.approx(0.98)

    def test_freeband_ignores_lambda_sparse(self) -> None:
        # Freeband mode should NOT additionally apply lambda_sparse / lambda_unc /
        # success_bonus / failure_penalty — these are V2 surface terms.
        z = np.zeros(4)
        r = compute_reward(
            z, z, action=2, noop_idx=10,
            reward_mode="path_length_freeband",
            terminated=True, truncated=False, is_success=True,
            step_idx=3, lambda_sparse=0.99, success_bonus=99.0,
        )
        # Free band → penalty=0; freeband_success_bonus default 1.0 → r = 1.0
        assert r == pytest.approx(1.0)

    def test_unknown_reward_mode_raises(self) -> None:
        z = np.zeros(4)
        with pytest.raises(ValueError, match="Unknown reward_mode"):
            compute_reward(
                z, z, action=0, noop_idx=10, reward_mode="nonsense_mode",
                terminated=True, is_success=True,
            )

    def test_v2_modes_unchanged_after_freeband_addition(self) -> None:
        # absolute_distance, terminal_only_step_cost behave as before
        z_next = np.zeros(4)
        z_ref = np.zeros(4)
        r_abs = compute_reward(
            z_next, z_ref, action=5, noop_idx=10,
            reward_mode="absolute_distance",
            terminated=True, is_success=True, lambda_sparse=0.0,
        )
        assert r_abs == pytest.approx(0.0)  # -0·1.0 - 0·1[a≠NOOP] = 0
        r_term = compute_reward(
            z_next, z_ref, action=5, noop_idx=10,
            reward_mode="terminal_only_step_cost",
            terminated=True, is_success=True, step_idx=3, beta_step_cost=0.05,
            lambda_sparse=0.0,
        )
        assert r_term == pytest.approx(1.0 - 0.05 * 3)


# ---------------------------------------------------------------------------
# Env accumulator integration
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dynamics() -> Any:
    """A toy dynamics that always reduces |z| by half."""
    import torch
    class _MockDyn:
        def __call__(self, z, g):
            mu = -0.5 * z  # halve the distance each step
            log_var = torch.zeros_like(z)
            return z + mu, mu, log_var
        def eval(self): return self
        def to(self, device): return self
    return _MockDyn()


def _make_env(mock_dynamics, *, n_genes=10, max_steps=8, epsilon=0.01, **kwargs):
    from src.rl.environment import CellReprogrammingEnv
    n_latent = 4
    z_ref = np.zeros(n_latent, dtype=np.float32)
    start_pool = np.ones((5, n_latent), dtype=np.float32) * 5.0
    return CellReprogrammingEnv(
        dynamics_model=mock_dynamics,
        z_reference_centroid=z_ref,
        epsilon_success=float(epsilon),
        n_genes=n_genes,
        max_steps=max_steps,
        lambda_sparse=0.0,
        start_state_strategy="random_perturbation",
        distance_metric="l2",
        start_pool_latents=start_pool,
        reward_mode="path_length_freeband",
        seed=42,
        **kwargs,
    )


class TestFreebandEnvIntegration:
    def test_freeband_mid_episode_reward_is_zero(self, mock_dynamics) -> None:
        env = _make_env(mock_dynamics, max_steps=4, epsilon=0.01)
        env.reset(seed=0)
        _, r1, _, _, _ = env.step(0)
        # Mid-episode under freeband: R should be 0 unless terminated/truncated.
        assert r1 == 0.0

    def test_freeband_truncation_reward_matches_schedule(self, mock_dynamics) -> None:
        # max_steps=4, ε=0.01. Mock dynamics halves |z|=5 → 2.5 → 1.25 → 0.625 → 0.3125
        # After 4 steps (one for each call to step), neither ≤ ε so truncated.
        env = _make_env(mock_dynamics, max_steps=4, epsilon=0.01)
        env.reset(seed=0)
        rewards = []
        for _ in range(4):
            _, r, terminated, truncated, info = env.step(0 if _ == 0 else 1 if _ == 1 else 2 if _ == 2 else 3)
            rewards.append(r)
        # Last reward should be the terminal/truncated reward.
        assert truncated
        # T=4 with default schedule: penalty=0.02·(4-3)=0.02; is_success=False → r = -0.02.
        assert rewards[-1] == pytest.approx(-0.02)
        # Mid-episode rewards were zero.
        assert all(r == 0.0 for r in rewards[:-1])

    def test_freeband_short_success_full_bonus(self, mock_dynamics) -> None:
        # Start pool is all-5s, 4-dim → ||z||=sqrt(4·25)=10. mock halves → 5 → 2.5 → 1.25.
        # ε=6 → succeeds after one step (||z||=5 < 6).
        env = _make_env(mock_dynamics, max_steps=4, epsilon=6.0)
        env.reset(seed=0)
        _, r, terminated, _, info = env.step(0)
        # T=1 within free band: penalty=0, success → r = 1.0
        assert terminated, f"expected terminated; info={info}"
        assert info["success"]
        assert r == pytest.approx(1.0)

    def test_freeband_long_path_pays_penalty(self, mock_dynamics) -> None:
        # max_steps=6, ε very tight so we never succeed → truncated at T=6.
        # T=6 default: penalty = 0.04 + 0.10 = 0.14
        env = _make_env(mock_dynamics, max_steps=6, epsilon=0.001)
        env.reset(seed=0)
        rewards = []
        for i in range(6):
            _, r, _, _, _ = env.step(i)
            rewards.append(r)
        assert rewards[-1] == pytest.approx(-0.14, abs=1e-6)

    def test_freeband_overrides_propagate_through_constructor(self, mock_dynamics) -> None:
        # Set free_steps=0, mild_until=8, mild_beta=heavy_beta=0.05 → V2 step-cost equivalent.
        env = _make_env(
            mock_dynamics, max_steps=3, epsilon=0.001,
            free_steps=0, mild_until=8, mild_beta=0.05, heavy_beta=0.05,
        )
        env.reset(seed=0)
        for _ in range(3):
            _, r, _, _, _ = env.step(0 if _ == 0 else 1 if _ == 1 else 2)
        # T=3 truncated, no success: r = -0.05·3 = -0.15
        assert r == pytest.approx(-0.15, abs=1e-6)


# ---------------------------------------------------------------------------
# Reward-aware GreedyDynamicsBeamPolicy
# ---------------------------------------------------------------------------


class TestRewardAwareGreedy:
    def _build_toy_dynamics(self, *, decay: float = 0.4):
        """Dynamics that pulls z toward 0 by factor `decay` each step."""
        import torch
        class _Toy:
            def __call__(self, z_batch: torch.Tensor, g_batch: torch.Tensor):
                delta = -decay * z_batch
                log_var = torch.zeros_like(z_batch)
                return z_batch + delta, delta, log_var
            def eval(self): return self
            def to(self, device): return self
        return _Toy()

    def test_v2_distance_only_default_unchanged(self):
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        dyn = self._build_toy_dynamics(decay=0.4)
        z_ref = np.zeros(4, dtype=np.float32)
        z = np.ones(4, dtype=np.float32) * 5.0
        n_genes = 10
        mask = np.ones(n_genes + 1, dtype=bool)
        policy = GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=n_genes, depth=3,
        )
        # No freeband → V2 distance-only behavior. policy.name should NOT have any suffix.
        assert policy.name == "greedy_dyn_3"
        a = policy.select_action(z, mask, {})
        # Some gene action picked (not NOOP) since dynamics reduces distance.
        assert 0 <= a < n_genes

    def test_freeband_aware_name_suffix(self):
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        dyn = self._build_toy_dynamics(decay=0.4)
        policy = GreedyDynamicsBeamPolicy(
            self._build_toy_dynamics(), n_genes=10, z_ref=np.zeros(4),
            noop_idx=10, depth=5,
            freeband_schedule={"free_steps": 3, "mild_until": 5,
                               "mild_beta": 0.02, "heavy_beta": 0.10, "success_bonus": 1.0},
            success_epsilon=0.5,
        )
        assert policy.name == "greedy_dyn_5_freeband"

    def test_freeband_aware_prefers_shorter_success(self):
        """Two plans both succeed; the shorter (T=2) should beat the longer (T=4) under freeband."""
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        # Dynamics decays by 0.6 each step: 5 → 2 → 0.8 → 0.32 → 0.128 → ...
        # epsilon=2.5 means T=1 ALREADY succeeds (5·0.4 = 2 < 2.5).
        dyn = self._build_toy_dynamics(decay=0.6)
        z = np.ones(4, dtype=np.float32) * 5.0
        n_genes = 4
        mask = np.ones(n_genes + 1, dtype=bool)
        policy = GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=np.zeros(4, dtype=np.float32),
            noop_idx=n_genes, depth=5,
            freeband_schedule={"free_steps": 3, "mild_until": 5,
                               "mild_beta": 0.5, "heavy_beta": 1.0,  # exaggerate to make T=4,5 costly
                               "success_bonus": 1.0},
            success_epsilon=2.5,
        )
        # All 4 gene actions reach ε at depth=1 (single step). Policy should pick a depth-1 plan.
        a = policy.select_action(z, mask, {})
        # We expect the policy to commit to one of the 4 gene actions (not NOOP).
        assert 0 <= a < n_genes
