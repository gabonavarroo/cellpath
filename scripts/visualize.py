"""scripts/visualize.py — generate every presentation figure from artifacts.

Owner: Agent A (with Agent B contributing trajectory-data plumbing).

Every figure used in the thesis defense MUST be produced by this script (CLAUDE.md sacred
rule: notebooks visualize but presentation figures are reproducible from artifacts).

Figures produced
----------------
- ``fig_umap_perturbations.png``  — UMAP of latent space, colored by perturbation, with
  ``z_reference_centroid`` overlay.
- ``fig_dynamics_gate.png``        — bar chart of MLP vs baselines on primary + OOD.
- ``fig_rl_success_curve.png``     — PPO success rate over training timesteps.
- ``fig_rl_action_freq.png``       — top-K gene frequency bar chart.
- ``fig_rl_trajectories.png``      — rollout trajectories projected to UMAP.
- ``fig_depmap_enrichment.png``    — q-value heatmap across panels.

Usage
-----
::

    python scripts/visualize.py --config-name default
    python scripts/visualize.py --config-name default figure=fig_umap_perturbations
"""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra-driven main."""
    from src.utils.seeding import set_seed

    set_seed(int(cfg.seed))

    if cfg.get("dry_run", False):
        print(f"DRY RUN — would write figures to {cfg.paths.eval_figures_dir}")
        return 0

    raise NotImplementedError(
        "Agent A: implement figure orchestrator. "
        "Each figure is a top-level function in this module that reads artifacts and saves PNG."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
