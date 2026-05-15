"""Tests for :class:`src.rl.environment.CellReprogrammingEnv`.

Locked semantics tested here:
- Gymnasium API compliance (check_env).
- NO-OP terminates the episode; success conditional on ``||z - z_ref|| < ε`` (D7).
- Repeat-mask correctness.
- Action space includes NO-OP at the final index.
- Reward composition (D7 — NO-OP no sparsity, sparsity only for gene actions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _make_env(mock_z_reference_centroid: np.ndarray, mock_epsilon_success: float) -> Any:
    """Instantiate the env with a stub dynamics model returning z unchanged."""
    pytest.importorskip("gymnasium")
    from src.rl.environment import CellReprogrammingEnv

    class _NoopDynamics:
        """Identity dynamics: z_next = z, mu = 0, log_var = 0.

        Returns torch tensors with batch dim to match the real ``PerturbationDynamicsModel`` API.
        """

        def __call__(self, z, gene_idx):
            import torch
            mu = torch.zeros_like(z)
            log_var = torch.zeros_like(z)
            return z.clone(), mu, log_var

    # Provide a non-trivial start pool so reset() doesn't fall back to Gaussian noise
    start_pool = np.tile(
        mock_z_reference_centroid + 1.0, (10, 1)
    ).astype(np.float32)

    env = CellReprogrammingEnv(
        dynamics_model=_NoopDynamics(),
        z_reference_centroid=mock_z_reference_centroid,
        epsilon_success=mock_epsilon_success,
        n_genes=4,
        max_steps=5,
        start_pool_latents=start_pool,
    )
    return env


class TestEnvironmentConstruction:
    def test_construction_basic_attrs(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        assert env.n_genes == 4
        assert env.noop_idx == 4
        assert env.epsilon == pytest.approx(mock_epsilon_success)
        assert env.max_steps == 5


class TestEnvironmentAPI:
    def test_gymnasium_api_compliance(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        gym = pytest.importorskip("gymnasium")
        from gymnasium.utils.env_checker import check_env

        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        # check_env does its own resets/steps; the env must survive them
        check_env(env, skip_render_check=True)

    def test_reset_returns_obs_info(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        obs, info = env.reset(seed=42)
        assert obs.shape == mock_z_reference_centroid.shape
        assert obs.dtype == np.float32
        assert "action_mask" in info
        assert info["action_mask"].shape == (env.n_genes + 1,)
        assert info["action_mask"].dtype == bool

    def test_noop_terminates_episode(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(env.noop_idx)
        assert terminated is True

    def test_noop_success_requires_distance_below_epsilon(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        """NO-OP from a state ABOVE ε must NOT count as success."""
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        # Place the env far from the reference centroid (above ε) before NO-OP.
        env._z = (
            mock_z_reference_centroid
            + np.ones_like(mock_z_reference_centroid) * (mock_epsilon_success + 5.0)
        ).astype(np.float32)
        _, _, terminated, _, info = env.step(env.noop_idx)
        assert terminated is True
        assert info["success"] is False

    def test_noop_success_at_centroid(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        """NO-OP from a state AT the centroid must count as success."""
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        env._z = mock_z_reference_centroid.astype(np.float32).copy()
        _, _, terminated, _, info = env.step(env.noop_idx)
        assert terminated is True
        assert info["success"] is True

    def test_repeat_mask(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        obs, _, _, _, info = env.step(1)
        # The action used at step 1 should now be masked.
        assert info["action_mask"][1] == False  # noqa: E712

    def test_action_masks_method(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        """sb3-contrib's MaskablePPO polls env.action_masks() — must work."""
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.shape == (env.n_genes + 1,)
        assert mask[env.noop_idx] == True  # NO-OP always available  # noqa: E712

    def test_noop_never_masked(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        # Use all gene actions; NO-OP must remain available
        for a in range(env.n_genes):
            if env.action_masks()[a]:
                obs, _, term, trunc, info = env.step(a)
                if term or trunc:
                    break
        assert env.action_masks()[env.noop_idx] == True  # noqa: E712


class TestReward:
    """Pure reward function tests — no env needed."""

    def test_compute_reward_noop_pays_no_sparsity(self) -> None:
        from src.rl.reward import compute_reward

        z_far = np.array([10.0] * 32, dtype=np.float32)
        z_ref = np.zeros(32, dtype=np.float32)
        # NO-OP at action == noop_idx=10. Sparsity penalty must NOT apply.
        r = compute_reward(
            z_next=z_far, z_ref=z_ref, action=10, noop_idx=10,
            lambda_sparse=999.0,  # huge — if applied, reward would be hugely negative
            terminated=True, is_success=False,
        )
        # Reward ≈ -||z_far|| = -sqrt(32 * 100) ≈ -56.57 — NO 999 sparsity.
        assert r > -100.0, f"NO-OP should NOT pay sparsity penalty, got {r}"

    def test_compute_reward_gene_action_pays_sparsity(self) -> None:
        from src.rl.reward import compute_reward

        z = np.zeros(32, dtype=np.float32)
        z_ref = np.zeros(32, dtype=np.float32)
        # At z_ref, distance = 0. Gene action (action=0, noop_idx=10) → pays λ_sparse.
        r = compute_reward(
            z_next=z, z_ref=z_ref, action=0, noop_idx=10,
            lambda_sparse=0.05,
        )
        assert r == pytest.approx(-0.05)

    def test_compute_reward_success_bonus_applied(self) -> None:
        from src.rl.reward import compute_reward

        z = np.zeros(32, dtype=np.float32)
        z_ref = np.zeros(32, dtype=np.float32)
        r_no_bonus = compute_reward(
            z_next=z, z_ref=z_ref, action=10, noop_idx=10,
            terminated=True, is_success=True, success_bonus=0.0,
        )
        r_with_bonus = compute_reward(
            z_next=z, z_ref=z_ref, action=10, noop_idx=10,
            terminated=True, is_success=True, success_bonus=10.0,
        )
        assert r_with_bonus == pytest.approx(r_no_bonus + 10.0)

    def test_compute_reward_failure_penalty_applied(self) -> None:
        from src.rl.reward import compute_reward

        z = np.zeros(32, dtype=np.float32)
        z_ref = np.zeros(32, dtype=np.float32)
        r_no_pen = compute_reward(
            z_next=z, z_ref=z_ref, action=0, noop_idx=10,
            truncated=True, failure_penalty=0.0, lambda_sparse=0.0,
        )
        r_with_pen = compute_reward(
            z_next=z, z_ref=z_ref, action=0, noop_idx=10,
            truncated=True, failure_penalty=5.0, lambda_sparse=0.0,
        )
        assert r_with_pen == pytest.approx(r_no_pen - 5.0)

    def test_distance_l2(self) -> None:
        from src.rl.reward import distance_to_reference

        a = np.array([3.0, 4.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        assert distance_to_reference(a, b, metric="l2") == pytest.approx(5.0)

    def test_distance_cosine(self) -> None:
        from src.rl.reward import distance_to_reference

        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        # Identical vectors → cosine distance = 0
        assert distance_to_reference(a, b, metric="cosine") == pytest.approx(0.0)

        c = np.array([0.0, 1.0], dtype=np.float32)
        # Orthogonal → cosine distance = 1
        assert distance_to_reference(a, c, metric="cosine") == pytest.approx(1.0)

    def test_distance_cosine_zero_vector_safe(self) -> None:
        from src.rl.reward import distance_to_reference

        a = np.zeros(32, dtype=np.float32)
        b = np.ones(32, dtype=np.float32)
        # Should not raise; returns 1.0 by convention
        d = distance_to_reference(a, b, metric="cosine")
        assert d == pytest.approx(1.0)

    def test_distance_unknown_metric_raises(self) -> None:
        from src.rl.reward import distance_to_reference

        with pytest.raises(ValueError, match="Unknown distance metric"):
            distance_to_reference(
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                metric="manhattan",
            )
