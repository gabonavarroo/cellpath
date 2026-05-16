"""Tests for src/rl/reward.py — distance_to_reference and compute_reward.

Agent B Phase 3 implementation tests.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zeros(n: int = 32) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _ones(n: int = 32) -> np.ndarray:
    return np.ones(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# TestDistanceToReference
# ---------------------------------------------------------------------------


class TestDistanceToReference:
    def test_l2_zero_vector_to_itself(self) -> None:
        from src.rl.reward import distance_to_reference

        z = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert distance_to_reference(z, z) == pytest.approx(0.0, abs=1e-6)

    def test_l2_matches_numpy_norm(self) -> None:
        from src.rl.reward import distance_to_reference

        rng = np.random.default_rng(0)
        z     = rng.normal(size=32).astype(np.float32)
        z_ref = rng.normal(size=32).astype(np.float32)
        expected = float(np.linalg.norm(z - z_ref))
        assert distance_to_reference(z, z_ref, metric="l2") == pytest.approx(expected, rel=1e-5)

    def test_cosine_identical_vectors_is_zero(self) -> None:
        from src.rl.reward import distance_to_reference

        z = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert distance_to_reference(z, z, metric="cosine") == pytest.approx(0.0, abs=1e-6)

    def test_cosine_orthogonal_vectors_is_one(self) -> None:
        from src.rl.reward import distance_to_reference

        z     = np.array([1.0, 0.0], dtype=np.float32)
        z_ref = np.array([0.0, 1.0], dtype=np.float32)
        assert distance_to_reference(z, z_ref, metric="cosine") == pytest.approx(1.0, abs=1e-6)

    def test_cosine_zero_z_returns_one_not_nan(self) -> None:
        from src.rl.reward import distance_to_reference

        d = distance_to_reference(_zeros(), _ones(), metric="cosine")
        assert np.isfinite(d)
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_cosine_zero_z_ref_returns_one_not_nan(self) -> None:
        from src.rl.reward import distance_to_reference

        d = distance_to_reference(_ones(), _zeros(), metric="cosine")
        assert np.isfinite(d)
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_invalid_metric_raises(self) -> None:
        from src.rl.reward import distance_to_reference

        with pytest.raises(ValueError, match="Unknown distance metric"):
            distance_to_reference(_zeros(), _ones(), metric="manhattan")

    def test_returns_python_float(self) -> None:
        from src.rl.reward import distance_to_reference

        assert isinstance(distance_to_reference(_zeros(), _ones()), float)


# ---------------------------------------------------------------------------
# TestComputeReward
# ---------------------------------------------------------------------------


class TestComputeReward:
    """Reward formula: r = -distance_scale*d - lambda_sparse*[a≠noop]
                            - lambda_unc*mean(exp(lv)) + success_bonus*[terminated∧success]
                            - failure_penalty*[truncated]
    """

    def _base(self, **kwargs) -> float:
        from src.rl.reward import compute_reward

        defaults = dict(
            z_next=_zeros(),
            z_ref=_zeros(),
            action=0,
            noop_idx=10,
            distance_scale=1.0,
            lambda_sparse=0.05,
            lambda_unc=0.0,
            success_bonus=0.0,
            failure_penalty=0.0,
            terminated=False,
            truncated=False,
            is_success=False,
            distance_metric="l2",
        )
        defaults.update(kwargs)
        return compute_reward(**defaults)

    def test_distance_term_l2(self) -> None:
        from src.rl.reward import compute_reward

        z_ref = _zeros()
        z_next = np.array([3.0, 4.0] + [0.0] * 30, dtype=np.float32)
        # l2 = 5.0; no other terms
        r = compute_reward(z_next, z_ref, action=10, noop_idx=10)
        assert r == pytest.approx(-5.0, rel=1e-5)

    def test_distance_scale_multiplies_distance(self) -> None:
        z_ref  = _zeros()
        z_next = np.array([1.0] + [0.0] * 31, dtype=np.float32)
        # Use noop action to isolate the distance term (no sparsity penalty)
        r = self._base(z_next=z_next, z_ref=z_ref, distance_scale=3.0, action=10, noop_idx=10)
        assert r == pytest.approx(-3.0, rel=1e-5)

    def test_noop_does_not_pay_sparsity(self) -> None:
        z_ref = _zeros()
        noop  = 10
        # distance = 0; noop action → no sparsity penalty
        r = self._base(z_ref=z_ref, action=noop, noop_idx=noop, lambda_sparse=0.05)
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_non_noop_pays_sparsity(self) -> None:
        z_ref = _zeros()
        # distance = 0; non-noop action → -lambda_sparse
        r_noop    = self._base(z_ref=z_ref, action=10, noop_idx=10, lambda_sparse=0.05)
        r_non_noop = self._base(z_ref=z_ref, action=3,  noop_idx=10, lambda_sparse=0.05)
        assert r_non_noop - r_noop == pytest.approx(-0.05, abs=1e-6)

    def test_success_bonus_only_when_terminated_and_success(self) -> None:
        z_ref = _zeros()
        bonus = 10.0
        # not terminated → no bonus
        r_no  = self._base(z_ref=z_ref, success_bonus=bonus, terminated=False, is_success=True)
        # terminated + success → bonus
        r_yes = self._base(z_ref=z_ref, success_bonus=bonus, terminated=True,  is_success=True)
        # terminated but not success → no bonus
        r_fail = self._base(z_ref=z_ref, success_bonus=bonus, terminated=True, is_success=False)

        assert r_yes - r_no  == pytest.approx(bonus, abs=1e-6)
        assert r_fail == r_no

    def test_failure_penalty_on_truncation(self) -> None:
        z_ref   = _zeros()
        penalty = 5.0
        r_no   = self._base(z_ref=z_ref, failure_penalty=penalty, truncated=False)
        r_yes  = self._base(z_ref=z_ref, failure_penalty=penalty, truncated=True)
        assert r_no - r_yes == pytest.approx(penalty, abs=1e-6)

    def test_uncertainty_penalty_when_lambda_unc_positive(self) -> None:
        """Per locked semantics in src/rl/reward.py:6-11 the penalty is ``λ_unc · ‖σ‖``
        where ``σ = exp(½ · log_var)`` (i.e. L2 norm of per-dim std-devs)."""
        from src.rl.reward import compute_reward

        z_ref   = _zeros()
        log_var = np.full(32, -1.0, dtype=np.float32)
        sigma = np.exp(0.5 * log_var.astype(np.float64))
        expected_penalty = float(np.linalg.norm(sigma))

        r_no_unc = compute_reward(
            _zeros(), z_ref, action=10, noop_idx=10,
            log_var=log_var, lambda_unc=0.0,
        )
        r_with_unc = compute_reward(
            _zeros(), z_ref, action=10, noop_idx=10,
            log_var=log_var, lambda_unc=1.0,
        )
        assert r_no_unc - r_with_unc == pytest.approx(expected_penalty, rel=1e-5)

    def test_uncertainty_penalty_zero_when_lambda_zero(self) -> None:
        from src.rl.reward import compute_reward

        z_ref   = _zeros()
        log_var = np.full(32, 5.0, dtype=np.float32)  # large variance
        r_no_lv  = compute_reward(_zeros(), z_ref, action=10, noop_idx=10, lambda_unc=0.0)
        r_with_lv = compute_reward(_zeros(), z_ref, action=10, noop_idx=10,
                                   log_var=log_var, lambda_unc=0.0)
        assert r_no_lv == pytest.approx(r_with_lv, abs=1e-6)

    def test_returns_python_float(self) -> None:
        from src.rl.reward import compute_reward

        r = compute_reward(_zeros(), _zeros(), action=0, noop_idx=10)
        assert isinstance(r, float)

    def test_cosine_distance_metric(self) -> None:
        from src.rl.reward import compute_reward, distance_to_reference

        rng   = np.random.default_rng(7)
        z     = rng.normal(size=32).astype(np.float32)
        z_ref = rng.normal(size=32).astype(np.float32)

        r_l2  = compute_reward(z, z_ref, action=10, noop_idx=10, distance_metric="l2")
        r_cos = compute_reward(z, z_ref, action=10, noop_idx=10, distance_metric="cosine")

        expected_l2  = -distance_to_reference(z, z_ref, metric="l2")
        expected_cos = -distance_to_reference(z, z_ref, metric="cosine")
        assert r_l2  == pytest.approx(expected_l2,  rel=1e-5)
        assert r_cos == pytest.approx(expected_cos, rel=1e-5)


# ---------------------------------------------------------------------------
# TestComputeRewardModes (P0D Track B)
# ---------------------------------------------------------------------------


class TestComputeRewardModes:
    """P0D Track B — reward_mode ∈ {absolute_distance, delta_distance, terminal_only_step_cost}.

    The default ``absolute_distance`` mode must be bit-for-bit equivalent to the legacy
    behaviour (verified by ``TestComputeReward`` above remaining green). These tests verify
    the two new modes plus error paths.
    """

    def test_absolute_distance_mode_matches_legacy(self) -> None:
        """Setting reward_mode='absolute_distance' explicitly must not change behaviour."""
        from src.rl.reward import compute_reward
        z_next = np.array([3.0, 4.0] + [0.0] * 30, dtype=np.float32)
        z_ref  = np.zeros(32, dtype=np.float32)
        r_legacy = compute_reward(z_next, z_ref, action=10, noop_idx=10)
        r_new    = compute_reward(z_next, z_ref, action=10, noop_idx=10,
                                  reward_mode="absolute_distance")
        assert r_legacy == pytest.approx(r_new, abs=1e-9)

    def test_delta_distance_rewards_progress(self) -> None:
        from src.rl.reward import compute_reward
        z_next = np.array([1.0] + [0.0] * 31, dtype=np.float32)
        z_ref  = np.zeros(32, dtype=np.float32)
        # d_prev = 5.0, d_next = 1.0 → reward = (5.0 - 1.0) = 4.0 (no sparsity bc NO-OP).
        r = compute_reward(z_next, z_ref, action=10, noop_idx=10,
                           reward_mode="delta_distance", prev_distance=5.0,
                           distance_scale=1.0, lambda_sparse=0.0)
        assert r == pytest.approx(4.0, abs=1e-6)

    def test_delta_distance_negative_when_worse(self) -> None:
        from src.rl.reward import compute_reward
        z_next = np.array([5.0] + [0.0] * 31, dtype=np.float32)
        z_ref  = np.zeros(32, dtype=np.float32)
        # d_prev = 1.0, d_next = 5.0 → reward = (1.0 - 5.0) = -4.0 (no sparsity bc NO-OP).
        r = compute_reward(z_next, z_ref, action=10, noop_idx=10,
                           reward_mode="delta_distance", prev_distance=1.0,
                           lambda_sparse=0.0)
        assert r == pytest.approx(-4.0, abs=1e-6)

    def test_delta_distance_requires_prev_distance(self) -> None:
        from src.rl.reward import compute_reward
        z_next = np.zeros(32, dtype=np.float32)
        z_ref  = np.zeros(32, dtype=np.float32)
        with pytest.raises(ValueError, match="prev_distance"):
            compute_reward(z_next, z_ref, action=0, noop_idx=10,
                           reward_mode="delta_distance", prev_distance=None)

    def test_terminal_only_zero_mid_episode(self) -> None:
        from src.rl.reward import compute_reward
        z_next = np.array([7.0] + [0.0] * 31, dtype=np.float32)  # large distance
        z_ref  = np.zeros(32, dtype=np.float32)
        r = compute_reward(z_next, z_ref, action=3, noop_idx=10,
                           reward_mode="terminal_only_step_cost",
                           terminated=False, truncated=False,
                           is_success=False, lambda_sparse=0.0, step_idx=2)
        # Mid-episode: 0 base reward; sparsity is 0 since lambda_sparse=0.
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_terminal_only_success_at_terminal(self) -> None:
        from src.rl.reward import compute_reward
        z_next = np.zeros(32, dtype=np.float32)  # at goal
        z_ref  = np.zeros(32, dtype=np.float32)
        # Terminal step at step_idx=2 with β=0.05 → R = 1·1 - 0.05·2 = 0.9.
        r = compute_reward(z_next, z_ref, action=10, noop_idx=10,
                           reward_mode="terminal_only_step_cost",
                           terminated=True, is_success=True,
                           beta_step_cost=0.05, step_idx=2,
                           lambda_sparse=0.0)
        assert r == pytest.approx(0.9, abs=1e-6)

    def test_terminal_only_truncation_no_success(self) -> None:
        from src.rl.reward import compute_reward
        z_next = np.array([7.0] + [0.0] * 31, dtype=np.float32)
        z_ref  = np.zeros(32, dtype=np.float32)
        # Truncated at step 3, no success → R = 0 - 0.05·3 = -0.15.
        r = compute_reward(z_next, z_ref, action=3, noop_idx=10,
                           reward_mode="terminal_only_step_cost",
                           terminated=False, truncated=True,
                           is_success=False, beta_step_cost=0.05, step_idx=3,
                           lambda_sparse=0.0)
        assert r == pytest.approx(-0.15, abs=1e-6)

    def test_unknown_reward_mode_raises(self) -> None:
        from src.rl.reward import compute_reward
        with pytest.raises(ValueError, match="Unknown reward_mode"):
            compute_reward(_zeros(), _zeros(), action=0, noop_idx=10,
                           reward_mode="quadratic")
