"""scripts/train_vae.py — standalone VAE training entry point.

Owner: Agent A. See AGENTS.md §1 Phase 1.

Usage
-----
::

    python scripts/train_vae.py --config-name default
    python scripts/train_vae.py vae.n_latent=64 vae.max_epochs=200
    python scripts/train_vae.py --multirun --config-name vae_ablation

Workflow
--------
1. Compose Hydra config.
2. ``set_seed(cfg.seed)``.
3. Log device summary.
4. Check for existing checkpoint; skip training if present (unless save_overwrite=True).
5. Run preprocessing if processed h5ad is missing.
6. Call :func:`src.models.vae.train_vae`.
"""

from __future__ import annotations

import logging
import sys

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    from pathlib import Path

    from src.utils.device import device_summary
    from src.utils.logging import setup_logging
    from src.utils.seeding import set_seed

    setup_logging(level=str(cfg.log.level))
    set_seed(int(cfg.seed))
    log.info(device_summary())

    # ------------------------------------------------------------------ #
    # Dry-run: validate config + resolve paths, then exit
    # ------------------------------------------------------------------ #
    if cfg.get("dry_run", False):
        log.info(
            "DRY RUN — VAE config: n_latent=%d, likelihood=%s, max_epochs=%d",
            cfg.vae.n_latent, cfg.vae.gene_likelihood, cfg.vae.max_epochs,
        )
        log.info("DRY RUN — would write artifacts to: %s", cfg.paths.vae_dir)
        return 0

    # ------------------------------------------------------------------ #
    # Preprocessing: run if processed h5ad is missing
    # ------------------------------------------------------------------ #
    processed_path = Path(cfg.paths.norman_processed_h5ad)
    if not processed_path.exists():
        log.info("Processed h5ad not found — running preprocessing first...")
        from src.data.preprocess import run_preprocessing
        adata = run_preprocessing(cfg)
    else:
        log.info("Preprocessed h5ad found at %s — skipping preprocessing.", processed_path)
        adata = None  # train_vae will load it

    # ------------------------------------------------------------------ #
    # Train VAE
    # ------------------------------------------------------------------ #
    from src.models.vae import train_vae
    model, adata = train_vae(cfg, adata=adata)

    log.info("VAE training complete. Artifacts at %s", cfg.paths.vae_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
