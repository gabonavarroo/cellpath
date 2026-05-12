"""scripts/train_dynamics.py — standalone dynamics-model training entry point.

Owner: Agent B. See AGENTS.md §2 Phase 1 + Phase 2.

Usage
-----
::

    python scripts/train_dynamics.py --config-name default
    python scripts/train_dynamics.py dynamics.lambda_combo=0.0  pairing.method=random
    python scripts/train_dynamics.py --multirun --config-name vae_ablation

Workflow
--------
1. Compose Hydra config.
2. ``src.utils.seeding.set_seed(cfg.seed)``.
3. Log device summary.
4. Check for an existing checkpoint at ``cfg.paths.dynamics_model``; if present, skip
   training and run only the validation gate (unless ``--force`` is set).
5. Load Contract 2 pairs from ``cfg.paths.pairs_train/val/ood/combo``. If those files don't
   exist, fall back to :func:`src.data.perturbation_pairs.generate_mock_pairs` (Day 0 use case).
6. Train the residual heteroscedastic MLP per ``cfg.dynamics``.
7. Run :func:`src.analysis.metrics.dynamics_validation_gate` and write ``gate.json``.
8. Exit 0 on success; exit 1 if the gate fails (so CI / next pipeline step sees the failure).
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

    if cfg.get("dry_run", False):
        print(f"DRY RUN — would train dynamics with d_emb={cfg.dynamics.d_emb}, "
              f"n_hidden={cfg.dynamics.n_hidden}, n_layers={cfg.dynamics.n_layers}")
        print(f"DRY RUN — would write to {cfg.paths.dynamics_dir}")
        return 0

    raise NotImplementedError(
        "Agent B: implement the 8-step workflow above. "
        "Train with heteroscedastic NLL + composition loss for combo pairs. "
        "Write gate.json — exit 1 if gate not passed."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
