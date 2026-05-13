"""Build OT pseudo-pairs for the dynamics model.

Hydra entry point — called by ``make pairs``.

Usage::

    # Default (OT pairing, config/default.yaml)
    python scripts/build_pairs.py --config-name default

    # Switch pairing method on the fly
    python scripts/build_pairs.py --config-name default pairing.method=random
    python scripts/build_pairs.py --config-name default pairing.method=mean_delta

    # Dry-run: validate config and paths, exit before any heavy computation
    python scripts/build_pairs.py --config-name default --dry-run
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    from src.utils.seeding import set_seed
    from src.utils.logging import setup_logging

    setup_logging(
        level=str(cfg.log.level),
        rich_traceback=bool(cfg.log.rich_traceback),
    )
    set_seed(int(cfg.seed))

    # ------------------------------------------------------------------
    # Dry-run: validate paths and exit
    # ------------------------------------------------------------------
    dry_run = cfg.get("dry_run", False)

    processed_path = Path(cfg.paths.norman_processed_h5ad)
    latents_path   = Path(cfg.paths.vae_latents_h5ad)

    if dry_run:
        log.info("[dry-run] Preprocessed h5ad: %s (exists=%s)", processed_path, processed_path.exists())
        log.info("[dry-run] Latents h5ad:       %s (exists=%s)", latents_path,   latents_path.exists())
        log.info("[dry-run] Pairs output dir:    %s", cfg.paths.pairs_dir)
        log.info("[dry-run] Pairing method:      %s", cfg.pairing.method)
        log.info("[dry-run] OT epsilon:          %.4f", cfg.pairing.ot_epsilon)
        log.info("[dry-run] Exiting (dry-run mode).")
        return

    # ------------------------------------------------------------------
    # Validate required inputs exist
    # ------------------------------------------------------------------
    missing = [p for p in (processed_path, latents_path) if not p.exists()]
    if missing:
        log.error(
            "Required inputs not found: %s\n"
            "Run `make vae` first to produce latents.h5ad.",
            [str(p) for p in missing],
        )
        sys.exit(1)

    log.info("Preprocessed h5ad: %s", processed_path)
    log.info("Latents h5ad:      %s", latents_path)
    log.info("Output dir:        %s", cfg.paths.pairs_dir)
    log.info("Pairing method:    %s", cfg.pairing.method)

    # ------------------------------------------------------------------
    # Build pairs
    # ------------------------------------------------------------------
    from src.data.perturbation_pairs import build_pairs

    paths = build_pairs(cfg)

    log.info("Pairs written:")
    for key, path in paths.items():
        size_mb = Path(path).stat().st_size / 1e6 if Path(path).exists() else 0
        log.info("  %-12s → %s  (%.1f MB)", key, path, size_mb)

    log.info("Done. Run `make dynamics` next.")


if __name__ == "__main__":
    main()
