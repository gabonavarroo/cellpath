"""scripts/train_rl.py — standalone PPO training entry point.

Owner: Agent B. See AGENTS.md §2 Phase 3.

**Sacred gate (CLAUDE.md §3 rule #9):** this script refuses to start unless the dynamics
validation gate is recorded as passed in ``artifacts/dynamics/gate.json``. Override with
``rl.train.skip_gate=true`` (Hydra) and a loud warning is logged.

Usage
-----
::

    python scripts/train_rl.py --config-name default
    python scripts/train_rl.py rl.reward.lambda_sparse=0.1
    python scripts/train_rl.py --multirun --config-name rl_sparse
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


def _gate_check(cfg: DictConfig) -> bool:
    """Read gate.json and return ``passed``.

    Honor ``cfg.rl.train.skip_gate`` with a loud warning.
    """
    gate_path = Path(cfg.paths.dynamics_gate)
    skip = bool(cfg.rl.train.skip_gate)

    if skip:
        print("⚠️  P0 WARNING: rl.train.skip_gate=true. Proceeding without gate verification.")
        return True

    if not gate_path.exists():
        print(f"❌ Gate file not found at {gate_path}. Run scripts/train_dynamics.py first.")
        return False

    try:
        gate = json.loads(gate_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"❌ Could not parse {gate_path}: {exc}")
        return False

    passed = bool(gate.get("passed", False))
    if not passed:
        print(f"❌ Dynamics gate not passed (see {gate_path}). RL training refuses to start.")
    return passed


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra-driven main."""
    from src.utils.device import device_summary
    from src.utils.seeding import set_seed

    set_seed(int(cfg.seed))
    print(device_summary())

    if not _gate_check(cfg):
        return 2

    if cfg.get("dry_run", False):
        print(f"DRY RUN — would train MaskablePPO for {cfg.rl.ppo.total_timesteps} timesteps")
        print(f"DRY RUN — n_envs={cfg.rl.env.n_envs}, lambda_sparse={cfg.rl.reward.lambda_sparse}")
        print(f"DRY RUN — would write to {cfg.paths.rl_dir}")
        return 0

    from src.rl.train_ppo import train_ppo
    train_ppo(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
