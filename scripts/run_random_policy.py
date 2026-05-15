"""scripts/run_random_policy.py — random-policy baseline matched to PPO eval settings.

Builds the same ``CellReprogrammingEnv`` PPO eval uses (same epsilon, same
``min_start_distance``, same max_steps, same reward shaping) and runs a configurable
number of rollouts under a non-learned policy. The headline use case is "is PPO actually
beating a random policy under the dynamics we trained, or is the task contractive?".

Each invocation writes (into ``random_policy.out_dir``):
- ``rollouts.parquet`` — Contract 4 schema, same columns as PPO eval
- ``action_freq.json`` — gene-symbol → count
- ``summary.json``    — success rate, mean steps, mean final / min distance, top actions
- ``metadata.json``   — full provenance via the shared writer in ``src.rl.train_ppo``

Usage
-----
::

    python scripts/run_random_policy.py --config-name default \\
        rl.train.skip_gate=true \\
        +random_policy.out_dir=artifacts/rl_sweeps/p50_start8_noopfix_500k_detfinal/random_baseline \\
        +random_policy.n_episodes=500 \\
        +random_policy.min_start_distance=8.0 \\
        +random_policy.epsilon_override=2.8 \\
        +random_policy.policy_kind=uniform_valid

Policy kinds
------------
- ``uniform_valid``           (default) — uniform over ``mask=True`` actions (including NO-OP).
- ``always_noop``             — always emit NO-OP at step 0; terminal.
- ``uniform_genes_then_noop`` — uniform over **gene** actions for the first step, then
                                proceed uniformly over valid actions. Avoids the trivial
                                "NO-OP first → fail" failure mode.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict


log = logging.getLogger(__name__)


def _apply_overrides(cfg: DictConfig) -> dict[str, Any]:
    """Apply ``+random_policy.*`` knobs onto ``cfg`` in-place; return resolved settings."""
    rp_blob = cfg.get("random_policy", None)
    rp_d: dict[str, Any] = (
        OmegaConf.to_container(rp_blob, resolve=True) if rp_blob is not None else {}
    )

    out_dir = rp_d.get("out_dir", None)
    if out_dir is None:
        raise ValueError(
            "random_policy.out_dir is required. Pass +random_policy.out_dir=<path>."
        )

    n_episodes = int(rp_d.get("n_episodes", cfg.rl.eval.n_rollout_episodes))
    policy_kind = str(rp_d.get("policy_kind", "uniform_valid"))
    seed = int(rp_d.get("seed", cfg.get("seed", 42)))

    valid_kinds = {"uniform_valid", "always_noop", "uniform_genes_then_noop"}
    if policy_kind not in valid_kinds:
        raise ValueError(
            f"random_policy.policy_kind={policy_kind!r} not in {sorted(valid_kinds)}"
        )

    # Push env-level overrides into the cfg so make_env_factory respects them.
    with open_dict(cfg):
        if "min_start_distance" in rp_d and rp_d["min_start_distance"] is not None:
            cfg.rl.env.min_start_distance = rp_d["min_start_distance"]
        if "epsilon_override" in rp_d and rp_d["epsilon_override"] is not None:
            cfg.rl.env.epsilon_override = float(rp_d["epsilon_override"])

    return {
        "out_dir": str(out_dir),
        "n_episodes": n_episodes,
        "policy_kind": policy_kind,
        "seed": seed,
    }


def _sample_action(
    mask: np.ndarray, noop_idx: int, *, step_idx: int, kind: str, rng: np.random.Generator
) -> int:
    """Pick an action under one of three baseline policies (all respect the env's mask)."""
    if kind == "always_noop":
        return int(noop_idx)

    valid = np.where(mask)[0]
    if len(valid) == 0:
        # Safety: shouldn't happen because NO-OP is always available, but defend anyway.
        return int(noop_idx)

    if kind == "uniform_genes_then_noop" and step_idx == 0:
        gene_valid = valid[valid != noop_idx]
        if len(gene_valid) > 0:
            return int(rng.choice(gene_valid))
        # Fall through to uniform_valid if all gene actions are somehow masked.

    return int(rng.choice(valid))


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra entry point."""
    import polars as pl

    from src.rl.environment import make_env_factory
    from src.rl.train_ppo import _write_run_metadata
    from src.utils.device import device_summary
    from src.utils.seeding import set_seed

    set_seed(int(cfg.get("seed", 42)))
    print(device_summary())

    settings = _apply_overrides(cfg)
    out_dir = Path(settings["out_dir"])

    if cfg.get("dry_run", False):
        print("DRY RUN — would run random-policy baseline:")
        print(f"  out_dir            = {out_dir}")
        print(f"  n_episodes         = {settings['n_episodes']}")
        print(f"  policy_kind        = {settings['policy_kind']}")
        print(f"  min_start_distance = {cfg.rl.env.get('min_start_distance', 'auto')}")
        print(f"  epsilon_override   = {cfg.rl.env.get('epsilon_override', None)}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # Match the PPO-eval output dir convention by also redirecting Contract-4 paths.
    with open_dict(cfg):
        cfg.paths.rl_dir = str(out_dir)
        cfg.paths.rl_rollouts_parquet = str(out_dir / "rollouts.parquet")
        cfg.paths.rl_action_freq_json = str(out_dir / "action_freq.json")
        cfg.paths.rl_success_curves_png = str(out_dir / "success_curves.png")

    # Gene-symbol lookup
    gene_lookup: dict[int, str] = {}
    try:
        with open(cfg.paths.vae_gene_vocab_json) as f:
            vocab = json.load(f)
        for i, g in enumerate(vocab["genes"]):
            gene_lookup[i] = str(g)
        gene_lookup[int(vocab["noop_idx"])] = "NO_OP"
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load gene_vocab.json (%s) — gene_symbol will be 'gene_<idx>'", exc)

    rng = np.random.default_rng(settings["seed"])
    factory = make_env_factory(cfg)
    env = factory()
    noop_idx = int(env.noop_idx)

    rows: list[dict[str, Any]] = []
    action_freq: dict[str, int] = {}
    success_count = 0
    step_counts: list[int] = []
    reward_totals: list[float] = []
    final_distances: list[float] = []
    min_distances: list[float] = []
    noop_first_action_failures = 0

    n_episodes = settings["n_episodes"]
    policy_kind = settings["policy_kind"]

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        terminated = False
        truncated = False
        ep_steps = 0
        ep_reward = 0.0
        ep_min_dist = float(info.get("distance", float("inf")))
        ep_terminal_success = False
        first_action: int | None = None

        while not (terminated or truncated):
            mask = info["action_mask"]
            action = _sample_action(
                mask, noop_idx, step_idx=ep_steps, kind=policy_kind, rng=rng,
            )
            if first_action is None:
                first_action = action

            sym = gene_lookup.get(action, f"gene_{action}")
            action_freq[sym] = action_freq.get(sym, 0) + 1

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            d = float(info["distance"])
            ep_min_dist = min(ep_min_dist, d)

            rows.append({
                "episode_id": int(ep),
                "step": int(ep_steps),
                "action": int(action),
                "gene_symbol": sym,
                "z_norm": d,
                "reward": float(reward),
                "terminated": bool(terminated),
                "success": bool(info.get("success", False)),
                "z_vector": obs.astype(np.float32).tolist(),
            })
            ep_terminal_success = bool(info.get("success", False))

        if ep_terminal_success and terminated:
            success_count += 1
        if first_action == noop_idx and not ep_terminal_success:
            noop_first_action_failures += 1
        step_counts.append(ep_steps)
        reward_totals.append(ep_reward)
        final_distances.append(float(info.get("distance", float("nan"))))
        min_distances.append(ep_min_dist)

    success_rate = success_count / max(n_episodes, 1)
    mean_steps = float(np.mean(step_counts)) if step_counts else 0.0
    mean_reward = float(np.mean(reward_totals)) if reward_totals else 0.0
    mean_final_dist = float(np.mean(final_distances)) if final_distances else 0.0
    mean_min_dist = float(np.mean(min_distances)) if min_distances else 0.0

    # Write Contract-4 artifacts (rollouts + action_freq).
    pl.DataFrame(rows).write_parquet(str(out_dir / "rollouts.parquet"))
    with open(out_dir / "action_freq.json", "w") as f:
        json.dump(action_freq, f, indent=2)

    summary = {
        "policy_kind": policy_kind,
        "n_episodes": n_episodes,
        "success_rate": float(success_rate),
        "successes": int(success_count),
        "failures": int(n_episodes - success_count),
        "mean_steps": mean_steps,
        "mean_total_reward": mean_reward,
        "mean_final_distance": mean_final_dist,
        "mean_min_distance": mean_min_dist,
        "noop_first_action_failures": int(noop_first_action_failures),
        "noop_first_action_rate": float(noop_first_action_failures / max(n_episodes, 1)),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "Random policy '%s': success_rate=%.3f  mean_steps=%.2f  mean_final_d=%.3f",
        policy_kind, success_rate, mean_steps, mean_final_dist,
    )

    _write_run_metadata(
        cfg,
        out_dir,
        deterministic=None,
        n_episodes=n_episodes,
        extras={
            "stage": "random_policy",
            "policy_kind": policy_kind,
            "policy_path": "random_policy://" + policy_kind,
            "summary": summary,
        },
    )

    print(f"Done. Random-policy baseline → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
