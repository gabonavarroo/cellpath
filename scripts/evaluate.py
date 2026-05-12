"""scripts/evaluate.py — full evaluation suite.

Owner: Agent A. See AGENTS.md §1 Phase 3 + Phase 5.

Aggregates artifacts from all four components and produces the final evaluation tables:

1. Latent quality: silhouette + ARI (from :mod:`src.analysis.latent_space`).
2. Dynamics: re-load ``gate.json`` + ``val_metrics.json`` + ``ood_metrics.json``.
3. RL: success rate, mean steps, action frequencies (from :mod:`src.analysis.metrics`).
4. DepMap enrichment: hypergeometric + GSEA + null comparison
   (from :mod:`src.analysis.depmap_validation`).
5. Trajectory rendering (used by :mod:`scripts.visualize`).
6. Aggregate across multiple ablation runs if invoked with ``--aggregate-ablations``.

Usage
-----
::

    python scripts/evaluate.py --config-name default
    python scripts/evaluate.py --config-name default --aggregate-ablations
"""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra-driven main. See module docstring."""
    from src.utils.device import device_summary
    from src.utils.seeding import set_seed

    set_seed(int(cfg.seed))
    print(device_summary())

    if cfg.get("dry_run", False):
        print(f"DRY RUN — would write evaluation outputs to {cfg.paths.eval_dir}")
        return 0

    raise NotImplementedError(
        "Agent A: implement evaluation orchestrator. "
        "Read all four component artifacts, run analyses, write to artifacts/eval/."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
