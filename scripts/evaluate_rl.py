"""scripts/evaluate_rl.py — evaluate an existing MaskablePPO checkpoint.

Loads a saved policy, builds the env from current Hydra config (with optional eval-time
overrides), and runs one or more rollout passes. By default produces both a deterministic
and a stochastic eval folder side-by-side so reports can show both honestly.

Each output folder contains:
- ``rollouts.parquet`` (Contract 4 schema, identical to ``train_ppo``'s output)
- ``action_freq.json``
- ``metadata.json``  (full provenance: epsilon source/value, gate status + override flag,
  dynamics ckpt SHA-256, PPO hparams snapshot, deterministic flag, n_episodes, git SHA, …)

Usage
-----
::

    # Default: evaluate artifacts/rl/ppo.zip → artifacts/rl/eval_{det,stoch}/
    python scripts/evaluate_rl.py --config-name default \\
        rl.train.skip_gate=true

    # Custom checkpoint + isolated output dir
    python scripts/evaluate_rl.py --config-name default \\
        rl.train.skip_gate=true \\
        +eval_rl.ppo_path=artifacts/rl_sweeps/p50_start8_noopfix_500k_detfinal/ppo.zip \\
        +eval_rl.out_dir=artifacts/rl_sweeps/p50_start8_noopfix_500k_detfinal/eval_re \\
        +eval_rl.n_episodes=500 \\
        +eval_rl.min_start_distance=8.0 \\
        +eval_rl.epsilon_override=2.8

    # Only one mode
    python scripts/evaluate_rl.py --config-name default rl.train.skip_gate=true \\
        '+eval_rl.modes=[deterministic]'
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict


log = logging.getLogger(__name__)


def _coerce_modes(modes_raw: Any) -> list[str]:
    """Accept ``[deterministic, stochastic]`` (list), ``"deterministic,stochastic"``,
    or a single string ``"deterministic"``. Normalize to a list of valid mode names."""
    if modes_raw is None:
        modes = ["deterministic", "stochastic"]
    elif isinstance(modes_raw, str):
        if "," in modes_raw:
            modes = [m.strip() for m in modes_raw.split(",") if m.strip()]
        else:
            modes = [modes_raw.strip()]
    else:
        modes = [str(m).strip() for m in modes_raw]

    valid = {"deterministic", "stochastic"}
    bad = [m for m in modes if m not in valid]
    if bad:
        raise ValueError(f"Unknown eval mode(s) {bad!r}; valid modes: {sorted(valid)}")
    return modes


def _apply_eval_overrides(cfg: DictConfig) -> dict[str, Any]:
    """Apply ``+eval_rl.*`` knobs onto ``cfg`` in-place; return the resolved settings.

    Knobs honored:
      - ``eval_rl.ppo_path``           (default ``cfg.paths.rl_ppo_zip``)
      - ``eval_rl.out_dir``            (default ``cfg.paths.rl_dir``)
      - ``eval_rl.n_episodes``         (default ``cfg.rl.eval.n_rollout_episodes``)
      - ``eval_rl.min_start_distance`` (default ``cfg.rl.env.min_start_distance``)
      - ``eval_rl.epsilon_override``   (default ``cfg.rl.env.epsilon_override``)
      - ``eval_rl.modes``              (default ``[deterministic, stochastic]``)
      - ``eval_rl.seed``               (default ``cfg.seed``)
    """
    eval_blob = cfg.get("eval_rl", None)
    eval_d: dict[str, Any] = (
        OmegaConf.to_container(eval_blob, resolve=True) if eval_blob is not None else {}
    )

    ppo_path = eval_d.get("ppo_path", None) or str(cfg.paths.rl_ppo_zip)
    out_dir = eval_d.get("out_dir", None) or str(cfg.paths.rl_dir)
    n_episodes = int(eval_d.get("n_episodes", cfg.rl.eval.n_rollout_episodes))
    seed = int(eval_d.get("seed", cfg.get("seed", 42)))
    modes = _coerce_modes(eval_d.get("modes", None))

    # Apply env-level overrides into the cfg so make_env_factory picks them up
    with open_dict(cfg):
        if "min_start_distance" in eval_d and eval_d["min_start_distance"] is not None:
            cfg.rl.env.min_start_distance = eval_d["min_start_distance"]
        if "epsilon_override" in eval_d and eval_d["epsilon_override"] is not None:
            cfg.rl.env.epsilon_override = float(eval_d["epsilon_override"])

    return {
        "ppo_path": str(ppo_path),
        "out_dir": str(out_dir),
        "n_episodes": n_episodes,
        "seed": seed,
        "modes": modes,
    }


def _run_one_mode(
    cfg: DictConfig,
    *,
    mode: str,
    ppo_path: Path,
    out_dir: Path,
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run a single eval pass; write rollouts/action_freq/metadata into ``out_dir``."""
    from sb3_contrib import MaskablePPO

    from src.rl.environment import make_env_factory
    from src.rl.train_ppo import _write_run_metadata, evaluate_policy

    deterministic = (mode == "deterministic")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reroute the artifact paths Contract-4 expects so each mode lands in its own dir.
    # Use a local OmegaConf copy so subsequent modes don't see the previous override.
    mode_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(mode_cfg):
        mode_cfg.rl.eval.deterministic = deterministic
        mode_cfg.rl.eval.n_rollout_episodes = n_episodes
        mode_cfg.paths.rl_dir = str(out_dir)
        mode_cfg.paths.rl_rollouts_parquet = str(out_dir / "rollouts.parquet")
        mode_cfg.paths.rl_action_freq_json = str(out_dir / "action_freq.json")
        # Stable path for the (loaded, not retrained) policy artifact; we don't rewrite it.
        mode_cfg.paths.rl_ppo_zip = str(ppo_path)
        mode_cfg.paths.rl_success_curves_png = str(out_dir / "success_curves.png")
        # seed used by env reset() in evaluate_policy is the episode index, not cfg.seed,
        # but propagate cfg.seed for any downstream RNG.
        mode_cfg.seed = int(seed)

    log.info(
        "[%s] Loading policy %s and evaluating %d episodes → %s",
        mode, ppo_path, n_episodes, out_dir,
    )
    model = MaskablePPO.load(str(ppo_path), device="cpu")

    factory = make_env_factory(mode_cfg)
    env = factory()
    metrics = evaluate_policy(
        model=model,
        env=env,
        n_episodes=n_episodes,
        deterministic=deterministic,
        cfg=mode_cfg,
    )
    log.info(
        "[%s] success_rate=%.3f  mean_steps=%.2f  mean_reward=%.3f",
        mode, metrics["success_rate"], metrics["mean_steps"], metrics["mean_reward"],
    )

    _write_run_metadata(
        mode_cfg,
        out_dir,
        deterministic=deterministic,
        n_episodes=n_episodes,
        extras={
            "stage": "evaluate_rl",
            "eval_mode": mode,
            "policy_path": str(ppo_path),
            "policy_sha256": None,  # cheap; only the ckpt file path is needed for tracing
            "final_eval_metrics": {
                "success_rate": metrics["success_rate"],
                "mean_steps": metrics["mean_steps"],
                "mean_reward": metrics["mean_reward"],
            },
        },
    )
    return metrics


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra entry point."""
    from src.utils.device import device_summary
    from src.utils.seeding import set_seed

    set_seed(int(cfg.get("seed", 42)))
    print(device_summary())

    settings = _apply_eval_overrides(cfg)
    ppo_path = Path(settings["ppo_path"])
    out_dir = Path(settings["out_dir"])

    if cfg.get("dry_run", False):
        print("DRY RUN — would evaluate policy without retraining:")
        print(f"  ppo_path           = {ppo_path}")
        print(f"  out_dir            = {out_dir}")
        print(f"  n_episodes         = {settings['n_episodes']}")
        print(f"  modes              = {settings['modes']}")
        print(f"  min_start_distance = {cfg.rl.env.get('min_start_distance', 'auto')}")
        print(f"  epsilon_override   = {cfg.rl.env.get('epsilon_override', None)}")
        return 0

    if not ppo_path.exists():
        print(f"ERROR: PPO checkpoint not found at {ppo_path}", file=sys.stderr)
        return 2

    for mode in settings["modes"]:
        mode_out = out_dir / f"eval_{mode}"
        _run_one_mode(
            cfg,
            mode=mode,
            ppo_path=ppo_path,
            out_dir=mode_out,
            n_episodes=settings["n_episodes"],
            seed=settings["seed"],
        )

    print(f"Done. Wrote eval folders under {out_dir} for modes={settings['modes']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
