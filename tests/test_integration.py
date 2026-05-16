"""End-to-end integration tests.

Two integration paths:
1. ``test_mock_pipeline`` — validates the pipeline CLI dry-run path without real-data retrain.
2. ``test_pipeline_with_real_data`` — slow, skipped in CI; full real-data run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


class TestHydraConfig:
    def test_default_config_composes(self, hydra_cfg) -> None:
        assert "vae" in hydra_cfg
        assert "dynamics" in hydra_cfg
        assert "rl" in hydra_cfg
        assert "paths" in hydra_cfg

    def test_paths_resolves(self, hydra_cfg) -> None:
        """Sanity: ``${paths.root}`` interpolates."""
        from omegaconf import OmegaConf

        resolved = OmegaConf.to_container(hydra_cfg, resolve=True)
        assert "vae_dir" in resolved["paths"]
        assert resolved["paths"]["vae_dir"].endswith("/artifacts/vae")

    def test_action_space_defaults(self, hydra_cfg) -> None:
        """Sacred config defaults from §1.1 of the audit."""
        assert hydra_cfg.rl.action_space.enable_knockout is False
        assert hydra_cfg.rl.action_space.include_noop is True
        assert hydra_cfg.rl.reference.source == "unperturbed_k562"

    def test_v1_epsilon_threshold_is_p50(self, hydra_cfg) -> None:
        """V1 intentionally keeps p50 as the canonical RL success threshold."""
        assert hydra_cfg.vae.epsilon_percentile == 50
        assert hydra_cfg.rl.reference.epsilon_percentile == 50

    def test_data_pipeline_keeps_raw_counts(self, hydra_cfg) -> None:
        """The scVI setup config must point at the counts layer."""
        assert hydra_cfg.vae.setup.layer == "counts"


def test_mock_pipeline(tmp_path) -> None:
    """Pipeline dry-run should use the real CLI path, not direct Typer invocation."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "KMP_INIT_AT_FORK": "FALSE",
        "KMP_USE_SHM": "0",
    })
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline", "run", "--config-name", "default", "--dry-run"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY RUN" in result.stdout
    assert "config=default" in result.stdout


def test_pipeline_experiment_config_dry_run() -> None:
    """Experiment config names should compose at the top level and dry-run cleanly."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "KMP_INIT_AT_FORK": "FALSE",
        "KMP_USE_SHM": "0",
    })
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.pipeline",
            "run",
            "--config-name",
            "experiments/rl_sparse",
            "--dry-run",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "config=experiments/rl_sparse" in result.stdout
    assert "rl.ppo.total_timesteps=1000000" in result.stdout


@pytest.mark.slow
def test_pipeline_with_real_data() -> None:
    pytest.skip("Real-data integration is run manually on the cluster.")
