"""Tests for V3B Variant C (safety-aware) reward and env safety accumulators.

Validates:
* V2-mode regression: with λ_tox = λ_ce = 0, ``safety_aware`` ≡ ``terminal_only_step_cost``.
* Scale invariance and monotonicity in λ_tox / λ_ce.
* Env accumulators correctly track ``tox_path`` and ``common_essential_count`` across steps.
* The permute_chronos null preserves the marginal (n_essential) but destroys per-gene identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from src.rl.biology_rewards import build_safety_arrays, safety_aware_reward
from src.rl.reward import compute_reward


# ---------------------------------------------------------------------------
# safety_aware_reward — pure function
# ---------------------------------------------------------------------------


class TestSafetyAwareReward:
    def test_mid_episode_is_zero(self) -> None:
        r = safety_aware_reward(
            is_success=False, terminated=False, truncated=False,
            step_idx=2, tox_path=10.0, common_essential_count=3,
        )
        assert r == 0.0

    def test_terminal_success_no_safety_cost(self) -> None:
        # Successful, no tox, no essential → R = 1 - β·t
        r = safety_aware_reward(
            is_success=True, terminated=True, truncated=False,
            step_idx=2, tox_path=0.0, common_essential_count=0,
        )
        assert r == pytest.approx(1.0 - 0.05 * 2)

    def test_terminal_failure(self) -> None:
        # Truncated without success
        r = safety_aware_reward(
            is_success=False, terminated=False, truncated=True,
            step_idx=3, tox_path=0.0, common_essential_count=0,
        )
        assert r == pytest.approx(0.0 - 0.05 * 3)

    def test_safety_penalty_subtracts(self) -> None:
        # 1 success - 0.05·2 step - 0.10·1.5 tox - 0.05·1 ce = 1 - 0.10 - 0.15 - 0.05 = 0.70
        r = safety_aware_reward(
            is_success=True, terminated=True, truncated=False,
            step_idx=2, tox_path=1.5, common_essential_count=1,
            lambda_tox=0.10, lambda_ce=0.05,
        )
        assert r == pytest.approx(0.70)

    def test_v2_regression_lambda_zero_matches_terminal_only(self) -> None:
        # λ_tox = λ_ce = 0 should exactly reproduce terminal_only_step_cost.
        from src.rl.reward import compute_reward
        # V2 terminal_only_step_cost
        z_next = np.zeros(8)
        z_ref = np.zeros(8)
        v2 = compute_reward(
            z_next, z_ref, action=5, noop_idx=10,
            reward_mode="terminal_only_step_cost",
            terminated=True, truncated=False, is_success=True,
            step_idx=2, beta_step_cost=0.05,
            lambda_sparse=0.0,  # turn off sparsity for clean comparison
        )
        v3b = compute_reward(
            z_next, z_ref, action=5, noop_idx=10,
            reward_mode="safety_aware",
            terminated=True, truncated=False, is_success=True,
            step_idx=2, beta_step_cost=0.05,
            tox_path=10.0, common_essential_count=5,
            lambda_tox=0.0, lambda_ce=0.0,  # safety inactive
        )
        assert v3b == pytest.approx(v2)

    def test_monotonic_in_lambda_tox(self) -> None:
        # Increasing λ_tox should decrease reward when tox_path > 0.
        r0 = safety_aware_reward(
            is_success=True, terminated=True, truncated=False,
            step_idx=1, tox_path=1.0, common_essential_count=0,
            lambda_tox=0.0, lambda_ce=0.0,
        )
        r1 = safety_aware_reward(
            is_success=True, terminated=True, truncated=False,
            step_idx=1, tox_path=1.0, common_essential_count=0,
            lambda_tox=0.5, lambda_ce=0.0,
        )
        assert r1 < r0

    def test_monotonic_in_lambda_ce(self) -> None:
        r0 = safety_aware_reward(
            is_success=True, terminated=True, truncated=False,
            step_idx=1, tox_path=0.0, common_essential_count=2,
            lambda_tox=0.0, lambda_ce=0.0,
        )
        r1 = safety_aware_reward(
            is_success=True, terminated=True, truncated=False,
            step_idx=1, tox_path=0.0, common_essential_count=2,
            lambda_tox=0.0, lambda_ce=0.5,
        )
        assert r1 == pytest.approx(r0 - 1.0)


# ---------------------------------------------------------------------------
# build_safety_arrays — array materialisation + permutation null
# ---------------------------------------------------------------------------


def _synthetic_gene_safety(n: int = 10) -> pl.DataFrame:
    rows = []
    for i in range(n):
        c = -1.0 + 0.2 * i  # spans -1.0 to +0.8
        tox = max(0.0, -c - 0.5)
        rows.append({
            "gene_symbol": f"GENE_{i:02d}",
            "action_idx": i,
            "chronos": c,
            "is_essential": bool(c < -0.5),
            "tox_raw": tox,
            "tox_norm": min(1.0, tox),
            "missing_chronos": False,
        })
    return pl.DataFrame(rows)


class TestBuildSafetyArrays:
    def test_shapes(self) -> None:
        tox, ess = build_safety_arrays(_synthetic_gene_safety(10), n_genes=10)
        assert tox.shape == (10,)
        assert ess.shape == (10,)
        assert tox.dtype == np.float32
        assert ess.dtype == bool

    def test_values_match_dataframe(self) -> None:
        df = _synthetic_gene_safety(10)
        tox, ess = build_safety_arrays(df, n_genes=10)
        # Gene 0: chronos=-1.0 → tox=0.5, essential
        assert tox[0] == pytest.approx(0.5)
        assert ess[0] is np.True_ or ess[0] == True  # noqa: E712
        # Gene 9: chronos=0.8 → tox=0, not essential
        assert tox[9] == pytest.approx(0.0)
        assert not ess[9]

    def test_permute_preserves_n_essential(self) -> None:
        df = _synthetic_gene_safety(10)
        tox, ess = build_safety_arrays(df, n_genes=10)
        tox_p, ess_p = build_safety_arrays(df, n_genes=10, permute_chronos=True, seed=42)
        assert ess.sum() == ess_p.sum()
        assert tox.sum() == pytest.approx(tox_p.sum())

    def test_permute_changes_per_gene_identity(self) -> None:
        df = _synthetic_gene_safety(20)
        _, ess = build_safety_arrays(df, n_genes=20)
        _, ess_p = build_safety_arrays(df, n_genes=20, permute_chronos=True, seed=42)
        # With n=20 and 2-3 essentials, permutation almost certainly changes the indices.
        # If it doesn't, the test is uninformative — re-seed.
        assert not np.array_equal(ess, ess_p), "Permutation produced identical array; re-seed."


# ---------------------------------------------------------------------------
# Env accumulator integration
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dynamics() -> Any:
    """A toy dynamics that moves z toward zero by a fraction per step."""
    import torch
    class _MockDyn:
        def __call__(self, z, g):
            mu = -0.1 * z  # always make progress toward origin
            log_var = torch.zeros_like(z)
            return z + mu, mu, log_var
        def eval(self): return self
        def to(self, device): return self
    return _MockDyn()


def _make_env(mock_dynamics, *, reward_mode: str, n_genes: int = 10,
              tox_arr=None, ess_arr=None, lambda_tox=0.10, lambda_ce=0.05):
    from src.rl.environment import CellReprogrammingEnv
    n_latent = 4
    z_ref = np.zeros(n_latent, dtype=np.float32)
    start_pool = np.ones((5, n_latent), dtype=np.float32) * 5.0  # 5 units from ref
    return CellReprogrammingEnv(
        dynamics_model=mock_dynamics,
        z_reference_centroid=z_ref,
        epsilon_success=0.01,  # very strict — we won't reach it
        n_genes=n_genes,
        max_steps=3,
        lambda_sparse=0.0,  # turn off so safety is the only signal
        start_state_strategy="random_perturbation",
        distance_metric="l2",
        start_pool_latents=start_pool,
        reward_mode=reward_mode,
        beta_step_cost=0.05,
        safety_tox_per_action=tox_arr,
        safety_essential_per_action=ess_arr,
        lambda_tox=lambda_tox,
        lambda_ce=lambda_ce,
        seed=42,
    )


class TestEnvSafetyAccumulators:
    def test_reset_zeroes_accumulators(self, mock_dynamics) -> None:
        df = _synthetic_gene_safety(10)
        tox, ess = build_safety_arrays(df, n_genes=10)
        env = _make_env(mock_dynamics, reward_mode="safety_aware", tox_arr=tox, ess_arr=ess)
        _, info = env.reset(seed=0)
        assert info["tox_path"] == 0.0
        assert info["common_essential_count"] == 0

    def test_step_accumulates_tox(self, mock_dynamics) -> None:
        df = _synthetic_gene_safety(10)
        tox, ess = build_safety_arrays(df, n_genes=10)
        env = _make_env(mock_dynamics, reward_mode="safety_aware", tox_arr=tox, ess_arr=ess)
        env.reset(seed=0)
        # Action 0: tox = 0.5, essential
        _, _, _, _, info = env.step(0)
        assert info["tox_path"] == pytest.approx(0.5)
        assert info["common_essential_count"] == 1
        # Action 1: tox = 0.3, essential. Cumulative: tox=0.8, ce=2
        _, _, _, _, info = env.step(1)
        assert info["tox_path"] == pytest.approx(0.8)
        assert info["common_essential_count"] == 2

    def test_noop_does_not_change_safety_accumulators(self, mock_dynamics) -> None:
        df = _synthetic_gene_safety(10)
        tox, ess = build_safety_arrays(df, n_genes=10)
        env = _make_env(mock_dynamics, reward_mode="safety_aware", tox_arr=tox, ess_arr=ess)
        env.reset(seed=0)
        env.step(0)  # tox += 0.5, ce += 1
        _, _, _, _, info = env.step(10)  # NOOP
        # NOOP is the terminal action; accumulators should not advance.
        assert info["tox_path"] == pytest.approx(0.5)
        assert info["common_essential_count"] == 1

    def test_safety_reward_terminal_value(self, mock_dynamics) -> None:
        df = _synthetic_gene_safety(10)
        tox, ess = build_safety_arrays(df, n_genes=10)
        env = _make_env(
            mock_dynamics, reward_mode="safety_aware",
            tox_arr=tox, ess_arr=ess, lambda_tox=0.1, lambda_ce=0.05,
        )
        env.reset(seed=0)
        # Gene 0 (tox=0.5, essential), then NOOP — episode terminates without success.
        _, r1, _, _, _ = env.step(0)
        assert r1 == 0.0  # mid-episode
        # NOOP terminate; pre-step distance ~ 4.5 ≫ ε=0.01, so is_success=False.
        # R_T = 0·success - β·t - λ_tox·tox_path - λ_ce·ce
        #     = 0 - 0.05·1 - 0.1·0.5 - 0.05·1 = -0.05 - 0.05 - 0.05 = -0.15
        _, r2, terminated, _, _ = env.step(10)
        assert terminated
        assert r2 == pytest.approx(-0.15)

    def test_safety_arrays_none_means_zero_accumulators(self, mock_dynamics) -> None:
        # When safety arrays are None, reward_mode=safety_aware still works:
        # tox_path stays 0, ce_count stays 0 → equivalent to terminal_only_step_cost.
        env = _make_env(
            mock_dynamics, reward_mode="safety_aware",
            tox_arr=None, ess_arr=None, lambda_tox=0.1, lambda_ce=0.05,
        )
        env.reset(seed=0)
        env.step(0)
        _, r, terminated, _, info = env.step(10)
        assert terminated
        # is_success=False (dist ~ 4.5 ≫ ε=0.01); R = -β·t = -0.05 (one gene step).
        assert info["tox_path"] == 0.0
        assert info["common_essential_count"] == 0
        assert r == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# compute_reward dispatch
# ---------------------------------------------------------------------------


class TestSafetyAwareGreedy:
    """V3B Phase 2 — greedy_dyn_K with safety scoring."""

    def _toy_dynamics(self):
        """Deterministic: action 0 makes large progress; action 1 makes equal progress."""
        import torch
        class _Toy:
            def __init__(self) -> None:
                # 4-d latent. delta[action 0]=−0.4·z; delta[action 1]=−0.4·z (equal).
                pass
            def __call__(self, z_batch: torch.Tensor, g_batch: torch.Tensor):
                # g_batch is 1-indexed gene_idx (env action + 1).
                delta = -0.4 * z_batch
                log_var = torch.zeros_like(z_batch)
                return z_batch + delta, delta, log_var
            def eval(self): return self
            def to(self, device): return self
        return _Toy()

    def test_lambda_zero_matches_v2_behavior(self) -> None:
        """λ_tox = λ_ce = 0 should not change the chosen action vs V2 greedy."""
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        dyn = self._toy_dynamics()
        z_ref = np.zeros(4, dtype=np.float32)
        z = np.ones(4, dtype=np.float32) * 5.0
        n_genes = 10
        mask = np.ones(n_genes + 1, dtype=bool)

        v2_policy = GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=n_genes, depth=2,
        )
        v3b_policy = GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=n_genes, depth=2,
            safety_tox_per_action=np.zeros(n_genes, dtype=np.float32),
            safety_essential_per_action=np.zeros(n_genes, dtype=bool),
            lambda_tox=0.0, lambda_ce=0.0,
        )
        a_v2 = v2_policy.select_action(z, mask, {})
        a_v3b = v3b_policy.select_action(z, mask, {})
        assert a_v2 == a_v3b

    def test_safety_aware_avoids_essential(self) -> None:
        """Two actions reduce distance equally; only action 0 is essential.
        Safety-aware greedy should pick action 1 (non-essential) when essentially tied."""
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        dyn = self._toy_dynamics()
        z_ref = np.zeros(4, dtype=np.float32)
        z = np.ones(4, dtype=np.float32) * 5.0
        n_genes = 10

        tox = np.zeros(n_genes, dtype=np.float32)
        tox[0] = 0.5  # essential
        ess = np.zeros(n_genes, dtype=bool)
        ess[0] = True

        # Restrict mask to actions {0, 1} so the choice is unambiguous.
        mask = np.zeros(n_genes + 1, dtype=bool)
        mask[0] = True
        mask[1] = True
        mask[n_genes] = True  # noop

        safety_policy = GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=n_genes, depth=1,
            safety_tox_per_action=tox, safety_essential_per_action=ess,
            lambda_tox=10.0, lambda_ce=10.0,  # heavy safety weighting
        )
        a = safety_policy.select_action(z, mask, {})
        assert a == 1, f"Safety-aware greedy should avoid essential gene 0; picked {a}"

    def test_safety_aware_name_suffix(self) -> None:
        """Policy.name should advertise safety mode."""
        from src.rl.baselines import GreedyDynamicsBeamPolicy
        n_genes = 10
        policy = GreedyDynamicsBeamPolicy(
            self._toy_dynamics(), n_genes=n_genes, z_ref=np.zeros(4), noop_idx=n_genes,
            depth=2,
            safety_tox_per_action=np.zeros(n_genes, dtype=np.float32),
            safety_essential_per_action=np.zeros(n_genes, dtype=bool),
            lambda_tox=0.1, lambda_ce=0.05,
        )
        assert policy.name == "greedy_dyn_2_safety"

        v2_policy = GreedyDynamicsBeamPolicy(
            self._toy_dynamics(), n_genes=n_genes, z_ref=np.zeros(4), noop_idx=n_genes, depth=2,
        )
        assert v2_policy.name == "greedy_dyn_2"


class TestComputeRewardDispatch:
    def test_safety_aware_reachable_via_compute_reward(self) -> None:
        z = np.zeros(4)
        r = compute_reward(
            z, z, action=2, noop_idx=10,
            reward_mode="safety_aware",
            terminated=True, truncated=False, is_success=True,
            step_idx=1, beta_step_cost=0.05,
            tox_path=0.5, common_essential_count=1,
            lambda_tox=0.1, lambda_ce=0.05,
        )
        # 1 - 0.05·1 - 0.1·0.5 - 0.05·1 = 1 - 0.05 - 0.05 - 0.05 = 0.85
        assert r == pytest.approx(0.85)

    def test_safety_aware_ignores_lambda_sparse(self) -> None:
        # Safety mode should NOT apply lambda_sparse on top.
        z = np.zeros(4)
        r1 = compute_reward(
            z, z, action=2, noop_idx=10, reward_mode="safety_aware",
            terminated=True, truncated=False, is_success=True,
            step_idx=1, beta_step_cost=0.05,
            tox_path=0.0, common_essential_count=0,
            lambda_tox=0.0, lambda_ce=0.0,
            lambda_sparse=0.99,  # huge, should be ignored in safety_aware
        )
        # R = 1 - 0.05·1 = 0.95 (sparse not applied)
        assert r1 == pytest.approx(0.95)
