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
    """Construct Hydra override list for V3B retrains.

    Modes:
    * ``safety_aware`` — Variant C (Phase 2): toxicity + common-essential penalty.
    * ``path_length_freeband`` — Variant B (Phase 3): nonlinear path-length penalty,
      no biology, default env.max_steps=8 to enable longer-horizon exploration.
    """
    overrides: list[str] = []

    if args.mode == "safety_aware":
        safety_path = repo_root / "artifacts_v3/v3b_biology/gene_safety.parquet"
        if not safety_path.exists():
            raise FileNotFoundError(
                f"V3B biology layer not built: {safety_path}. "
                f"Run scripts/build_v3b_biology_layer.py first."
            )
        permute_tag = "_permuted_chronos" if args.permute_chronos else ""
        rl_dir = f"artifacts_v3/rl_v3b_safety_aware_seed{args.seed}{permute_tag}"
        overrides += [
            "rl.reward.mode=safety_aware",
            f"rl.reward.lambda_tox={args.lambda_tox}",
            f"rl.reward.lambda_ce={args.lambda_ce}",
            f"rl.reward.safety_table_path={safety_path}",
            f"rl.reward.permute_chronos={'true' if args.permute_chronos else 'false'}",
            f"rl.reward.permute_chronos_seed={args.seed}",
        ]
    elif args.mode == "path_length_freeband":
        rl_dir = f"artifacts_v3/rl_v3b_path_freeband_seed{args.seed}"
        overrides += [
            "rl.reward.mode=path_length_freeband",
            f"rl.reward.freeband.free_steps={args.free_steps}",
            f"rl.reward.freeband.mild_until={args.mild_until}",
            f"rl.reward.freeband.mild_beta={args.mild_beta}",
            f"rl.reward.freeband.heavy_beta={args.heavy_beta}",
            f"rl.reward.freeband.success_bonus={args.success_bonus}",
            f"rl.env.max_steps={args.max_steps}",
        ]
    elif args.mode in ("safety_path_freeband", "uncertainty_aware", "biorealistic_fused"):
        # V3B Phase 4 — fused rewards. All write under artifacts_v3/rl_v3b_<mode>_eps<label>_seed<N>/.
        safety_path = repo_root / "artifacts_v3/v3b_biology/gene_safety.parquet"
        eps_tag = args.epsilon_label or "p25"
        rl_dir = f"artifacts_v3/rl_v3b_{args.mode}_eps{eps_tag}_seed{args.seed}"
        overrides += [
            f"rl.reward.mode={args.mode}",
            f"rl.reward.freeband.free_steps={args.free_steps}",
            f"rl.reward.freeband.mild_until={args.mild_until}",
            f"rl.reward.freeband.mild_beta={args.mild_beta}",
            f"rl.reward.freeband.heavy_beta={args.heavy_beta}",
            f"rl.reward.freeband.success_bonus={args.success_bonus}",
            f"rl.env.max_steps={args.max_steps}",
            f"rl.reward.lambda_unc_path={args.lambda_unc_path}",
        ]
        # Variants B+C and B+C+D need the safety table.
        if args.mode in ("safety_path_freeband", "biorealistic_fused"):
            if not safety_path.exists():
                raise FileNotFoundError(
                    f"V3B biology layer not built: {safety_path}. "
                    f"Run scripts/build_v3b_biology_layer.py first."
                )
            overrides += [
                f"rl.reward.lambda_tox={args.lambda_tox}",
                f"rl.reward.lambda_ce={args.lambda_ce}",
                f"rl.reward.safety_table_path={safety_path}",
            ]
        # Stricter epsilon overrides (Phase 4 calibrated).
        if args.epsilon_value is not None:
            overrides.append(f"rl.env.epsilon_override={args.epsilon_value}")
    else:
        raise ValueError(f"Unknown V3B mode: {args.mode}")

    overrides += [
        f"paths.rl_dir={rl_dir}",
        f"seed={args.seed}",
    ]
    if args.total_timesteps is not None:
        overrides.append(f"rl.ppo.total_timesteps={int(args.total_timesteps)}")
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=(
            "safety_aware", "path_length_freeband",
            "safety_path_freeband", "uncertainty_aware", "biorealistic_fused",
        ),
        default="safety_aware",
        help=(
            "V3B reward variant. Phase 2 Variant C = safety_aware; "
            "Phase 3 Variant B = path_length_freeband; "
            "Phase 4: B+C = safety_path_freeband, D = uncertainty_aware, "
            "B+C+D = biorealistic_fused."
        ),
    )
    # Variant C (safety) knobs
    parser.add_argument("--lambda_tox", type=float, default=0.10,
                        help="V3B Variant C λ_tox (default per plan §4).")
    parser.add_argument("--lambda_ce", type=float, default=0.05,
                        help="V3B Variant C λ_ce (default per plan §4).")
    parser.add_argument("--permute_chronos", action="store_true",
                        help="Null control: randomly permute Chronos labels.")
    # Variant B (freeband) knobs
    parser.add_argument("--free_steps", type=int, default=3,
                        help="V3B Variant B free-band upper bound (T ≤ free_steps pays 0 penalty).")
    parser.add_argument("--mild_until", type=int, default=5,
                        help="V3B Variant B mild-band upper bound.")
    parser.add_argument("--mild_beta", type=float, default=0.02,
                        help="V3B Variant B mild-band slope per step.")
    parser.add_argument("--heavy_beta", type=float, default=0.10,
                        help="V3B Variant B heavy-band slope per step.")
    parser.add_argument("--success_bonus", type=float, default=1.0,
                        help="V3B Variant B terminal success bonus.")
    parser.add_argument("--max_steps", type=int, default=8,
                        help="V3B Variant B env.max_steps (default 8 to allow K∈{4,5,8} exploration).")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override rl.ppo.total_timesteps (default: V2 primary 1M).")
    # Phase 4 fused-mode knobs
    parser.add_argument("--lambda_unc_path", type=float, default=0.05,
                        help="V3B Variant D λ_unc_path (default per config/rl.yaml).")
    parser.add_argument("--epsilon_value", type=float, default=None,
                        help="Stricter ε override (e.g. p15=2.9898, p10=2.8846, p5=2.7362). "
                             "If None, uses default from epsilon_success.json + config.")
    parser.add_argument("--epsilon_label", type=str, default=None,
                        help="Label for output dir naming (e.g. 'p15'). Defaults to 'p25'.")
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
