"""V3B Phase 2 — thin wrapper around ``scripts/train_rl.py`` for safety-aware PPO.

Enforces:
* ``rl.reward.mode=safety_aware`` (Variant C of V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §4).
* ``rl.reward.safety_table_path`` resolves to the V3B biology layer parquet.
* Output goes under ``artifacts_v3/rl_v3b_<reward>_<latent>_seed<N>[...]/``.

The actual training is delegated to ``scripts/train_rl.py`` via subprocess so that the
Hydra config composition is identical to the V2 primary path (no parallel pipeline).

Typical use::

    # Real-Chronos PPO_C (the canonical Phase 2 run)
    python scripts/train_rl_v3b.py --seed 42

    # Permuted-Chronos null control (acceptance criterion 4)
    python scripts/train_rl_v3b.py --seed 42 --permute_chronos

    # Smoke (200k timesteps for quick sanity)
    python scripts/train_rl_v3b.py --seed 42 --total_timesteps 200000
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


LOG = logging.getLogger("train_rl_v3b")


def build_overrides(args: argparse.Namespace, repo_root: Path) -> list[str]:
    """Construct Hydra override list for the V3B safety-aware retrain."""
    safety_path = repo_root / "artifacts_v3/v3b_biology/gene_safety.parquet"
    if not safety_path.exists():
        raise FileNotFoundError(
            f"V3B biology layer not built: {safety_path}. "
            f"Run scripts/build_v3b_biology_layer.py first."
        )

    # Naming convention: rl_v3b_<reward>_<latent>_seed<N>[_permuted]
    permute_tag = "_permuted_chronos" if args.permute_chronos else ""
    rl_dir = (
        f"artifacts_v3/rl_v3b_safety_aware_v2primary_seed{args.seed}{permute_tag}"
    )

    overrides: list[str] = [
        # Reward switch
        "rl.reward.mode=safety_aware",
        f"rl.reward.lambda_tox={args.lambda_tox}",
        f"rl.reward.lambda_ce={args.lambda_ce}",
        f"rl.reward.safety_table_path={safety_path}",
        f"rl.reward.permute_chronos={'true' if args.permute_chronos else 'false'}",
        f"rl.reward.permute_chronos_seed={args.seed}",
        # Output path
        f"paths.rl_dir={rl_dir}",
        # Seed
        f"seed={args.seed}",
    ]
    if args.total_timesteps is not None:
        overrides.append(f"rl.ppo.total_timesteps={int(args.total_timesteps)}")

    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_tox", type=float, default=0.10,
                        help="V3B Variant C λ_tox (default per plan §4).")
    parser.add_argument("--lambda_ce", type=float, default=0.05,
                        help="V3B Variant C λ_ce (default per plan §4).")
    parser.add_argument("--permute_chronos", action="store_true",
                        help="Null control: randomly permute Chronos labels.")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override rl.ppo.total_timesteps (default: V2 primary 1M).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the resolved command without executing.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    overrides = build_overrides(args, repo_root)

    cmd = [
        ".venv/bin/python",
        "scripts/train_rl.py",
        "--config-name", "default",
        *overrides,
    ]

    LOG.info("V3B Phase 2 training command:")
    LOG.info("  cwd: %s", repo_root)
    LOG.info("  cmd: %s", " ".join(cmd))

    if args.dry_run:
        LOG.info("Dry run; not executing.")
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
