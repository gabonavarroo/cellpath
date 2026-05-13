"""End-to-end integration tests.

Two integration paths:
1. ``test_mock_pipeline`` — runs the four components on tiny mock data; xfail-marked until all
   four agents' work is implemented.
2. ``test_pipeline_with_real_data`` — slow, skipped in CI; full real-data run.
"""

from __future__ import annotations

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

    def test_data_pipeline_keeps_raw_counts(self, hydra_cfg) -> None:
        """The scVI setup config must point at the counts layer."""
        assert hydra_cfg.vae.setup.layer == "counts"


@pytest.mark.slow
@pytest.mark.xfail(reason="Pipeline implementation is shared and not done yet.")
def test_mock_pipeline(tmp_path) -> None:
    """End-to-end on synthetic data: should be possible once all stubs are filled."""
    from src.pipeline import run

    run(config_name="default", dry_run=True)


@pytest.mark.slow
def test_pipeline_with_real_data() -> None:
    pytest.skip("Real-data integration is run manually on the cluster.")
