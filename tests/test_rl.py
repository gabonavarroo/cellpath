"""Phase 3 RL tests — gate check, factory, train_ppo smoke, evaluate_policy contract.

Smoke tests run for a few thousand timesteps to verify the pipeline end-to-end without
requiring real dynamics. Marked ``slow`` so they can be skipped in fast CI runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# =============================================================================
# check_dynamics_gate
# =============================================================================


class TestCheckDynamicsGate:
    def test_missing_file_no_skip_raises(self, tmp_path: Any) -> None:
        from src.rl.train_ppo import check_dynamics_gate

        with pytest.raises(FileNotFoundError):
            check_dynamics_gate(tmp_path / "nonexistent.json", skip=False)

    def test_missing_file_with_skip_returns_override(self, tmp_path: Any) -> None:
        from src.rl.train_ppo import check_dynamics_gate

        result = check_dynamics_gate(tmp_path / "nonexistent.json", skip=True)
        assert result["passed"] is True
        assert result["override"] is True

    def test_passed_gate_returns_dict(self, tmp_path: Any) -> None:
        from src.rl.train_ppo import check_dynamics_gate

        gate = {"passed": True, "primary": {"r2": 0.85}}
        gate_path = tmp_path / "gate.json"
        gate_path.write_text(json.dumps(gate))

        result = check_dynamics_gate(gate_path, skip=False)
        assert result["passed"] is True
        assert result["primary"]["r2"] == 0.85

    def test_failed_gate_no_skip_exits(self, tmp_path: Any) -> None:
        from src.rl.train_ppo import check_dynamics_gate

        gate = {"passed": False}
        gate_path = tmp_path / "gate.json"
        gate_path.write_text(json.dumps(gate))

        with pytest.raises(SystemExit) as exc_info:
            check_dynamics_gate(gate_path, skip=False)
        assert exc_info.value.code == 2

    def test_failed_gate_with_skip_overrides(self, tmp_path: Any) -> None:
        from src.rl.train_ppo import check_dynamics_gate

        gate = {"passed": False, "primary": {"r2": 0.10}}
        gate_path = tmp_path / "gate.json"
        gate_path.write_text(json.dumps(gate))

        result = check_dynamics_gate(gate_path, skip=True)
        assert result["override"] is True


# =============================================================================
# Environment factory (heavy — uses real artifacts if present)
# =============================================================================


def _real_artifacts_present() -> bool:
    root = Path(__file__).resolve().parents[1]
    needed = [
        root / "artifacts/vae/z_reference_centroid.npy",
        root / "artifacts/vae/epsilon_success.json",
        root / "artifacts/vae/gene_vocab.json",
        root / "artifacts/vae/latents.h5ad",
    ]
    return all(p.exists() for p in needed)


class TestMakeEnvFactory:
    @pytest.mark.slow
    def test_factory_constructs_env_with_real_artifacts_smoke(
        self, hydra_cfg: Any
    ) -> None:
        if not _real_artifacts_present():
            pytest.skip("VAE artifacts not present — run `make vae` first.")

        from omegaconf import OmegaConf, open_dict
        from src.rl.environment import make_env_factory

        # Use smoke-with-untrained-dynamics fallback so the test doesn't need real dynamics
        cfg = OmegaConf.create(OmegaConf.to_container(hydra_cfg, resolve=True))
        with open_dict(cfg):
            cfg.rl.train.smoke_with_untrained_dynamics = True

        factory = make_env_factory(cfg)
        env = factory()

        assert env.n_genes == 105
        assert env.noop_idx == 105
        obs, info = env.reset(seed=0)
        assert obs.shape == (32,)
        assert obs.dtype == np.float32
        assert info["action_mask"].shape == (106,)


# =============================================================================
# evaluate_policy — contract schema test (no real model needed)
# =============================================================================


class TestEvaluatePolicyContract:
    """Test that evaluate_policy produces Contract 4 rollouts."""

    @pytest.mark.slow
    def test_rollouts_match_contract4_schema(self, tmp_path: Any) -> None:
        if not _real_artifacts_present():
            pytest.skip("VAE artifacts not present — run `make vae` first.")

        from omegaconf import OmegaConf
        from hydra import compose, initialize_config_dir

        from src.rl.environment import make_env_factory

        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "config")):
            cfg = compose(
                config_name="default",
                overrides=[
                    "device.force=cpu",
                    f"paths.root={repo_root}",
                    "rl.train.skip_gate=true",
                    "rl.train.smoke_with_untrained_dynamics=true",
                    "rl.env.n_envs=1",
                    "rl.env.vec_env=dummy",
                ],
            )

        # Redirect rl output to tmp_path
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.paths.rl_dir = str(tmp_path)
            cfg.paths.rl_ppo_zip = str(tmp_path / "ppo.zip")
            cfg.paths.rl_rollouts_parquet = str(tmp_path / "rollouts.parquet")
            cfg.paths.rl_action_freq_json = str(tmp_path / "action_freq.json")
            cfg.paths.rl_success_curves_png = str(tmp_path / "success.png")

        # Quick smoke train (300 timesteps — enough for callbacks to fire)
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        factory = make_env_factory(cfg)
        env = DummyVecEnv([factory])
        model = MaskablePPO(
            "MlpPolicy", env,
            n_steps=64, batch_size=32, n_epochs=2,
            policy_kwargs={"net_arch": [32, 32]},
            seed=42, verbose=0, device="cpu",
        )
        # No need to call .learn(); evaluate_policy works on an untrained model too

        from src.rl.train_ppo import evaluate_policy
        single_env = factory()
        metrics = evaluate_policy(model, single_env, n_episodes=3, deterministic=True, cfg=cfg)

        # Check return-dict schema
        assert "success_rate" in metrics
        assert "mean_steps" in metrics
        assert "mean_reward" in metrics
        assert "action_freq" in metrics

        # Check rollouts.parquet schema (Contract 4)
        import polars as pl
        rollouts = pl.read_parquet(str(tmp_path / "rollouts.parquet"))
        expected_cols = {
            "episode_id", "step", "action", "gene_symbol",
            "z_norm", "reward", "terminated", "success", "z_vector",
        }
        assert expected_cols.issubset(set(rollouts.columns))
        assert len(rollouts) > 0
        # z_vector should be 32 elements
        assert len(rollouts["z_vector"][0]) == 32


# =============================================================================
# train_ppo end-to-end smoke (very slow — explicitly opt-in)
# =============================================================================


class TestTrainPPOSmoke:
    @pytest.mark.slow
    def test_train_ppo_short_run_writes_all_artifacts(self, tmp_path: Any) -> None:
        if not _real_artifacts_present():
            pytest.skip("VAE artifacts not present — run `make vae` first.")

        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict
        from src.rl.train_ppo import train_ppo

        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "config")):
            cfg = compose(
                config_name="default",
                overrides=[
                    "device.force=cpu",
                    f"paths.root={repo_root}",
                    "rl.train.skip_gate=true",
                    "rl.train.smoke_with_untrained_dynamics=true",
                    "rl.env.n_envs=1",
                    "rl.env.vec_env=dummy",
                    "rl.ppo.total_timesteps=512",
                    "rl.ppo.n_steps=64",
                    "rl.ppo.batch_size=32",
                    "rl.ppo.n_epochs=2",
                    "rl.train.eval_freq=256",
                    "rl.train.n_eval_episodes=2",
                    "rl.eval.n_rollout_episodes=2",
                ],
            )

        with open_dict(cfg):
            cfg.paths.rl_dir = str(tmp_path)
            cfg.paths.rl_ppo_zip = str(tmp_path / "ppo.zip")
            cfg.paths.rl_rollouts_parquet = str(tmp_path / "rollouts.parquet")
            cfg.paths.rl_action_freq_json = str(tmp_path / "action_freq.json")
            cfg.paths.rl_success_curves_png = str(tmp_path / "success.png")

        model = train_ppo(cfg)
        assert model is not None

        # Verify Contract 4 artifacts written
        assert (tmp_path / "ppo.zip").exists()
        assert (tmp_path / "rollouts.parquet").exists()
        assert (tmp_path / "action_freq.json").exists()
