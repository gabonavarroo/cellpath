"""V3C final pipeline / evaluation entrypoint.

Default: evaluate the locked V3C champion (cheap, ~10 min on CPU).
Modes:
  eval     — re-run the champion's 7-cell PPO + greedy baseline evaluation.
  demo     — evaluate at K=2/bin8-10/OOD only (fastest demo).
  baseline — re-run the V2 anchor evaluation for comparison.
  audit    — re-run the V3C utility audit on the champion's dynamics field.
  figures  — regenerate figures from existing audit / eval outputs.

Reads `artifacts_v3/v3c/final_champion_manifest.json` to resolve all paths;
no hardcoded artifact paths in this file. Refuses to retrain anything by default.

Usage
-----
::

    python scripts/run_final_v3c_pipeline.py --mode eval
    python scripts/run_final_v3c_pipeline.py --mode demo
    python scripts/run_final_v3c_pipeline.py --mode baseline
    python scripts/run_final_v3c_pipeline.py --mode figures

This is the canonical V3C runnable target. The `make final-v3c-eval` target wraps it.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = "artifacts_v3/v3c/final_champion_manifest.json"

LOG = logging.getLogger("v3c.final_pipeline")


def _load_manifest() -> dict:
    p = REPO_ROOT / MANIFEST_REL
    if not p.exists():
        raise SystemExit(
            f"Final champion manifest not found at {p}. "
            "Run Stage 4.5 of the V3C session to generate it, or specify --manifest."
        )
    return json.loads(p.read_text())


def _run(cmd: list[str], dry: bool) -> int:
    LOG.info("→ %s", " ".join(cmd))
    if dry:
        return 0
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def mode_eval(manifest: dict, args: argparse.Namespace) -> int:
    """Re-run the canonical 7-cell evaluation on the champion."""
    champ = manifest["champion"]
    out = REPO_ROOT / f"artifacts_v3/v3c/eval_final_champion_repro_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/evaluate_rl_v3b_phase4.py",
        "--vae_dir",       champ["vae_dir"],
        "--dynamics_dir",  champ["dynamics_dir"],
        "--pairs_dir",     champ["pairs_dir"],
        "--ppo",           f"PPO_CHAMP:{champ['ppo_checkpoint']}",
        "--out_dir",       str(out),
        "--n_episodes",    str(args.n_episodes),
        "--epsilon_value", str(champ["epsilon_scalar"]),
        "--epsilon_label", champ["epsilon_label"],
        "--lambda_tox",    str(champ["reward_coefficients"]["lambda_tox"]),
        "--lambda_ce",     str(champ["reward_coefficients"]["lambda_ce"]),
        "--lambda_unc_path", str(champ["reward_coefficients"]["lambda_unc_path"]),
    ]
    return _run(cmd, args.dry_run)


def mode_demo(manifest: dict, args: argparse.Namespace) -> int:
    """Fast 1-cell demo at the discriminating cell K=2/bin8-10/OOD."""
    champ = manifest["champion"]
    out = REPO_ROOT / f"artifacts_v3/v3c/eval_final_champion_demo_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/evaluate_rl_v3b_phase4.py",
        "--vae_dir",       champ["vae_dir"],
        "--dynamics_dir",  champ["dynamics_dir"],
        "--pairs_dir",     champ["pairs_dir"],
        "--ppo",           f"PPO_CHAMP:{champ['ppo_checkpoint']}",
        "--out_dir",       str(out),
        "--n_episodes",    str(min(args.n_episodes, 100)),
        "--epsilon_value", str(champ["epsilon_scalar"]),
        "--epsilon_label", champ["epsilon_label"],
        "--cells",         "k2_bin8-10_splitood",
        "--lambda_tox",    str(champ["reward_coefficients"]["lambda_tox"]),
        "--lambda_ce",     str(champ["reward_coefficients"]["lambda_ce"]),
        "--lambda_unc_path", str(champ["reward_coefficients"]["lambda_unc_path"]),
    ]
    return _run(cmd, args.dry_run)


def mode_baseline(manifest: dict, args: argparse.Namespace) -> int:
    """Re-run V2 anchor evaluation for side-by-side."""
    anchor = manifest["baselines"]["anchor"]
    out = REPO_ROOT / f"artifacts_v3/v3c/eval_anchor_baseline_repro_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/evaluate_rl_v3b_phase4.py",
        "--vae_dir",       anchor["vae_dir"],
        "--dynamics_dir",  anchor["dynamics_dir"],
        "--pairs_dir",     anchor["pairs_dir"],
        "--ppo",           f"PPO_ANCHOR:{anchor['ppo_checkpoint']}",
        "--out_dir",       str(out),
        "--n_episodes",    str(args.n_episodes),
        "--epsilon_value", str(anchor["epsilon_scalar"]),
        "--epsilon_label", anchor["epsilon_label"],
    ]
    return _run(cmd, args.dry_run)


def mode_audit(manifest: dict, args: argparse.Namespace) -> int:
    """Re-run the V3C utility audit on the champion's dynamics field."""
    champ = manifest["champion"]
    field_id = champ["dynamics_field_id"]
    cmd = [
        sys.executable, "scripts/audit_dynamics_utility_v3c.py", "all",
        "--field-id", field_id, "--n-episodes", str(args.n_episodes),
    ]
    return _run(cmd, args.dry_run)


def mode_figures(manifest: dict, args: argparse.Namespace) -> int:
    """Regenerate figures from existing audit/eval outputs."""
    cmd = [sys.executable, "scripts/generate_v3c_figures.py", "all"]
    return _run(cmd, args.dry_run)


MODES = {
    "eval": mode_eval, "demo": mode_demo, "baseline": mode_baseline,
    "audit": mode_audit, "figures": mode_figures,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=list(MODES), default="eval",
                   help="What to run. Default: eval (full 7-cell champion repro).")
    p.add_argument("--n-episodes", type=int, default=200,
                   help="Episodes per (policy, cell). Default 200; use 50–100 for demo speed.")
    p.add_argument("--tag", default="default",
                   help="Output-dir suffix to avoid overwriting prior runs.")
    p.add_argument("--manifest", default=None,
                   help="Override manifest path.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
    )
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
    else:
        manifest = _load_manifest()

    return MODES[args.mode](manifest, args) or 0


if __name__ == "__main__":
    sys.exit(main())
