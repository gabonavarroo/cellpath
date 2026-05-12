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
2. ``src.utils.seeding.set_seed(cfg.seed)``.
3. Log device summary.
4. Check for an existing checkpoint at ``cfg.paths.vae_model_dir``; if present, skip and
   exit 0 (use ``--force`` env var to override).
5. Load preprocessed AnnData from ``cfg.paths.norman_processed_h5ad`` (runs preprocessing if
   missing and ``--auto-preprocess`` is on).
6. Call :func:`src.models.vae.train_vae`.
7. Write the four Contract-1 artifacts: model dir, latents.h5ad, gene_vocab.json,
   z_reference_centroid.npy, epsilon_success.json.
"""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra-driven main. See module docstring for workflow."""
    from src.utils.device import device_summary
    from src.utils.seeding import set_seed

    set_seed(int(cfg.seed))
    print(device_summary())

    # ----- Dry-run -----
    if cfg.get("dry_run", False):
        print(f"DRY RUN — would train scVI with n_latent={cfg.vae.n_latent}, "
              f"likelihood={cfg.vae.gene_likelihood}, max_epochs={cfg.vae.max_epochs}")
        print(f"DRY RUN — would write to {cfg.paths.vae_dir}")
        return 0

    # ----- Real run -----
    raise NotImplementedError(
        "Agent A: implement the 7-step workflow above. "
        "Use src.models.vae.train_vae(cfg, adata=None). "
        "Verify Contract 1 artifacts after training."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
