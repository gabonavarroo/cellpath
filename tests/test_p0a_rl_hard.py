from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class _MoveToOriginModel:
    def __call__(self, z, gene_idx):
        import torch

        scale = torch.where(gene_idx[:, None] == 1, 1.0, 0.25)
        mu = -scale * z
        return z + mu, mu, torch.zeros_like(z)


def test_baseline_policies_respect_masks_and_choose_expected_actions(tmp_path: Path) -> None:
    from src.rl.baselines import (
        AlwaysNoopPolicy,
        GreedyDynamicsPolicy,
        MeanDeltaGreedyPolicy,
        RandomUniformValidPolicy,
        RidgeGreedyPolicy,
    )

    z = np.array([2.0, 0.0], dtype=np.float32)
    z_ref = np.zeros(2, dtype=np.float32)
    mask = np.array([False, True, True], dtype=bool)

    assert AlwaysNoopPolicy(noop_idx=2).select_action(z, mask, {}) == 2
    assert RandomUniformValidPolicy(seed=0).select_action(z, mask, {}) in {1, 2}

    mean_delta = np.array([[-0.25, 0.0], [-2.0, 0.0]], dtype=np.float32)
    assert MeanDeltaGreedyPolicy(mean_delta, z_ref=z_ref, noop_idx=2).select_action(z, mask, {}) == 1

    ridge_path = tmp_path / "ridge.npz"
    np.savez(
        ridge_path,
        W_z=np.zeros((2, 2), dtype=np.float32),
        W_gene=np.array([[-0.25, 0.0], [-2.0, 0.0]], dtype=np.float32),
        b=np.zeros(2, dtype=np.float32),
    )
    assert RidgeGreedyPolicy.from_npz(ridge_path, z_ref=z_ref, noop_idx=2).select_action(z, mask, {}) == 1

    greedy = GreedyDynamicsPolicy(_MoveToOriginModel(), n_genes=2, z_ref=z_ref, noop_idx=2)
    assert greedy.select_action(z, mask, {}) == 1


def test_evaluate_rl_hard_rollout_summary_on_fake_env() -> None:
    from scripts.evaluate_rl_hard import run_policy_episodes, wilson_ci

    class OneStepEnv:
        noop_idx = 1

        def reset(self, seed=None, options=None):
            self._done = False
            return np.array([2.0], dtype=np.float32), {
                "action_mask": np.array([True, True], dtype=bool),
                "distance": 2.0,
            }

        def step(self, action):
            self._done = True
            success = int(action) == 0
            return np.array([0.0 if success else 2.0], dtype=np.float32), 0.0, success, not success, {
                "action_mask": np.array([False, True], dtype=bool),
                "distance": 0.0 if success else 2.0,
                "success": success,
            }

    class PickGene:
        name = "pick_gene"

        def select_action(self, z, mask, info):
            return 0

    summary = run_policy_episodes(OneStepEnv(), PickGene(), n_episodes=5, gene_lookup={0: "G1", 1: "NO_OP"})
    assert summary["success_rate"] == pytest.approx(1.0)
    assert summary["mean_steps"] == pytest.approx(1.0)
    assert summary["top_actions"][0]["gene_symbol"] == "G1"

    lo, hi = wilson_ci(5, 5)
    assert 0.0 <= lo <= hi <= 1.0


def test_empty_start_pool_summaries_are_recorded_for_every_policy() -> None:
    from scripts.evaluate_rl_hard import empty_start_pool_summaries

    summaries = empty_start_pool_summaries(
        policy_names=["ppo_deterministic", "random_uniform_valid"],
        cell={
            "k": 3,
            "epsilon_label": "p25",
            "epsilon_value": 3.1,
            "distance_bin": "10-12",
            "gene_split": "ood",
        },
        reason="Empty start pool for split=ood, distance_bin=10-12",
    )

    assert set(summaries) == {"ppo_deterministic", "random_uniform_valid"}
    assert summaries["ppo_deterministic"]["status"] == "skipped_empty_start_pool"
    assert summaries["random_uniform_valid"]["success_rate"] is None
