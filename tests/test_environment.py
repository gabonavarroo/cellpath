"""Tests for :class:`src.rl.environment.CellReprogrammingEnv` once Agent B implements it.

Key tests (all xfail until B implements):
- Gymnasium API compliance.
- NO-OP semantics: terminates the episode, success conditional on distance < ε.
- Repeat-mask correctness.
- Action space includes NO-OP at the final index.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _make_env(mock_z_reference_centroid: np.ndarray, mock_epsilon_success: float) -> Any:
    """Instantiate the env with a stub dynamics model.

    Used in xfail-marked tests; will be flipped on once Agent B implements ``CellReprogrammingEnv``.
    """
    pytest.importorskip("gymnasium")
    from src.rl.environment import CellReprogrammingEnv

    class _NoopDynamics:
        def __call__(self, z, gene_idx):
            return z, np.zeros_like(z), np.zeros_like(z)

    env = CellReprogrammingEnv(
        dynamics_model=_NoopDynamics(),
        z_reference_centroid=mock_z_reference_centroid,
        epsilon_success=mock_epsilon_success,
        n_genes=4,
        max_steps=5,
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
    @pytest.mark.xfail(reason="Agent B has not implemented reset/step yet.")
    def test_gymnasium_api_compliance(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        gym = pytest.importorskip("gymnasium")
        from gymnasium.utils.env_checker import check_env

        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        check_env(env)

    @pytest.mark.xfail(reason="Agent B has not implemented reset yet.")
    def test_reset_returns_obs_info(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        obs, info = env.reset(seed=42)
        assert obs.shape == mock_z_reference_centroid.shape
        assert "action_mask" in info
        assert info["action_mask"].shape == (env.n_genes + 1,)

    @pytest.mark.xfail(reason="Agent B has not implemented step yet.")
    def test_noop_terminates_episode(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(env.noop_idx)
        assert terminated is True

    @pytest.mark.xfail(reason="Agent B has not implemented step yet.")
    def test_noop_success_requires_distance_below_epsilon(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        """NO-OP from a state ABOVE ε must NOT count as success."""
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        # Place the env far from the reference centroid (above ε) before NO-OP.
        env._z = mock_z_reference_centroid + np.ones_like(mock_z_reference_centroid) * (
            mock_epsilon_success + 5.0
        )
        _, _, terminated, _, info = env.step(env.noop_idx)
        assert terminated is True
        assert info["success"] is False

    @pytest.mark.xfail(reason="Agent B has not implemented step yet.")
    def test_noop_success_at_centroid(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        """NO-OP from a state AT the centroid must count as success."""
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        env._z = mock_z_reference_centroid.copy()
        _, _, terminated, _, info = env.step(env.noop_idx)
        assert terminated is True
        assert info["success"] is True

    @pytest.mark.xfail(reason="Agent B has not implemented step yet.")
    def test_repeat_mask(
        self, mock_z_reference_centroid: Any, mock_epsilon_success: float
    ) -> None:
        env = _make_env(mock_z_reference_centroid, mock_epsilon_success)
        env.reset(seed=42)
        obs, _, _, _, info = env.step(1)
        # The action used at step 1 should now be masked.
        assert info["action_mask"][1] == False  # noqa: E712


