"""MaskablePPO training loop using ``sb3-contrib``.

Owner: Agent B. See ARCHITECTURE.md Concept 4 and AGENTS.md §2 Phase 3.

Hard gate
---------
:func:`train_ppo` reads ``artifacts/dynamics/gate.json`` and refuses to start unless
``passed=True``. The override flag ``cfg.rl.train.skip_gate`` exists for dev runs but
emits a P0 warning to the rich console *and* to TensorBoard.

Math summary
------------
- **PPO clipped objective**::

    L^CLIP(θ) = E_t [ min( r_t(θ) · A_t,
                           clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]

  where ``r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)``, ``A_t`` is GAE.

- **GAE-λ advantage**::

    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    A_t = Σ_{l≥0} (γλ)^l δ_{t+l}

- **MaskablePPO**: masked logits → ``-inf`` before softmax, so the policy never assigns
  probability to forbidden actions. Gradient flows only through valid actions.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def check_dynamics_gate(gate_path: str | Path, skip: bool = False) -> dict[str, Any]:
    """Check the dynamics validation gate before RL training.

    Parameters
    ----------
    gate_path
        Path to ``artifacts/dynamics/gate.json``.
    skip
        If True, bypass the gate check (P0 warning).

    Returns
    -------
    dict
        Gate JSON contents (or override marker ``{"passed": True, "override": True}``).
    """
    gate_path = Path(gate_path)

    if skip:
        log.warning(
            "P0 — gate check bypassed via rl.train.skip_gate=True. "
            "Training will proceed without dynamics validation. "
            "DO NOT report metrics from this run as validated."
        )
        if gate_path.exists():
            try:
                with open(gate_path) as f:
                    data = json.load(f)
                data["override"] = True
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"passed": True, "override": True, "skipped": True}

    if not gate_path.exists():
        raise FileNotFoundError(
            f"Dynamics gate not found at {gate_path}. "
            f"Either run `make dynamics` first or set `rl.train.skip_gate=true`."
        )

    with open(gate_path) as f:
        gate = json.load(f)

    if not bool(gate.get("passed", False)):
        log.error(
            "Dynamics gate did NOT pass — refusing to start RL training. "
            "Inspect %s and rerun dynamics with adjusted hyperparameters, "
            "or pass rl.train.skip_gate=true to override.",
            gate_path,
        )
        raise SystemExit(2)

    log.info("Dynamics gate passed ✓ — proceeding with RL training.")
    return gate


def train_ppo(cfg: Any) -> Any:
    """Train MaskablePPO over the ``CellReprogrammingEnv``.

    Returns the trained ``MaskablePPO`` instance.
    """
    from omegaconf import OmegaConf
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from src.rl.environment import make_env_factory
    from src.utils.seeding import set_seed

    set_seed(int(cfg.seed))

    # ------------------------------------------------------------------ #
    # 1. Gate check
    # ------------------------------------------------------------------ #
    check_dynamics_gate(
        gate_path=cfg.paths.dynamics_gate,
        skip=bool(cfg.rl.train.get("skip_gate", False)),
    )

    # ------------------------------------------------------------------ #
    # 2. Build env factory and vectorized env
    # ------------------------------------------------------------------ #
    factory = make_env_factory(cfg)
    n_envs = int(cfg.rl.env.n_envs)

    # macOS Apple-Silicon + fork() + MPS can deadlock — default to DummyVecEnv unless
    # explicitly forced via cfg.rl.env.vec_env="subproc"
    vec_env_kind = str(cfg.rl.env.get("vec_env", "dummy")).lower()
    if vec_env_kind == "subproc" and n_envs > 1:
        log.info("Using SubprocVecEnv with %d workers", n_envs)
        vec_env = SubprocVecEnv([factory for _ in range(n_envs)])
    else:
        log.info("Using DummyVecEnv (single process) with %d copies", n_envs)
        vec_env = DummyVecEnv([factory for _ in range(n_envs)])

    eval_env = DummyVecEnv([factory])

    # ------------------------------------------------------------------ #
    # 3. Instantiate MaskablePPO
    # ------------------------------------------------------------------ #
    policy_kwargs = (
        OmegaConf.to_container(cfg.rl.ppo.policy_kwargs, resolve=True)
        if cfg.rl.ppo.get("policy_kwargs", None) is not None
        else {}
    )
    # Convert string activation_fn → torch.nn class (SB3 expects the class itself)
    if isinstance(policy_kwargs.get("activation_fn"), str):
        import torch.nn as nn
        _act_map = {
            "tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU,
            "silu": nn.SiLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU,
        }
        act_str = policy_kwargs["activation_fn"].lower()
        if act_str not in _act_map:
            raise ValueError(f"Unknown activation_fn={act_str!r}. Use one of {list(_act_map)}")
        policy_kwargs["activation_fn"] = _act_map[act_str]

    tb_dir = Path(cfg.log.tensorboard_dir) / f"rl_{int(time.time())}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO(
        cfg.rl.ppo.policy,
        vec_env,
        learning_rate=float(cfg.rl.ppo.lr),
        n_steps=int(cfg.rl.ppo.n_steps),
        batch_size=int(cfg.rl.ppo.batch_size),
        n_epochs=int(cfg.rl.ppo.n_epochs),
        gamma=float(cfg.rl.ppo.gamma),
        gae_lambda=float(cfg.rl.ppo.gae_lambda),
        clip_range=float(cfg.rl.ppo.clip_range),
        ent_coef=float(cfg.rl.ppo.ent_coef),
        vf_coef=float(cfg.rl.ppo.vf_coef),
        max_grad_norm=float(cfg.rl.ppo.max_grad_norm),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        seed=int(cfg.seed),
        verbose=1,
        device="cpu",  # Tiny network; CPU is fine and avoids MPS+SB3 quirks
    )

    log.info(
        "MaskablePPO initialised: policy=%s, lr=%.1e, n_steps=%d, batch=%d, γ=%.3f, GAE λ=%.3f",
        cfg.rl.ppo.policy, float(cfg.rl.ppo.lr), int(cfg.rl.ppo.n_steps),
        int(cfg.rl.ppo.batch_size), float(cfg.rl.ppo.gamma), float(cfg.rl.ppo.gae_lambda),
    )

    # ------------------------------------------------------------------ #
    # 4. Callbacks
    # ------------------------------------------------------------------ #
    rl_dir = Path(cfg.paths.rl_dir)
    rl_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(rl_dir / "best"),
        log_path=str(rl_dir / "eval_logs"),
        eval_freq=int(cfg.rl.train.eval_freq),
        n_eval_episodes=int(cfg.rl.train.n_eval_episodes),
        deterministic=False,
        render=False,
    )
    callbacks = CallbackList([eval_cb])

    # ------------------------------------------------------------------ #
    # 5. Train
    # ------------------------------------------------------------------ #
    total_steps = int(cfg.rl.ppo.total_timesteps)
    log.info("Training for %d timesteps...", total_steps)
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0
    log.info("Training complete in %.1f min", elapsed / 60.0)

    # ------------------------------------------------------------------ #
    # 6. Save policy + evaluate + dump rollouts
    # ------------------------------------------------------------------ #
    ppo_path = Path(cfg.paths.rl_ppo_zip)
    ppo_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ppo_path))
    log.info("Saved MaskablePPO → %s", ppo_path)

    # Plot success curve from eval logs
    _plot_success_curve(
        eval_logs_dir=rl_dir / "eval_logs",
        out_path=Path(cfg.paths.rl_success_curves_png),
    )

    # Final evaluation rollouts → Contract 4
    single_env = factory()
    metrics = evaluate_policy(
        model=model,
        env=single_env,
        n_episodes=int(cfg.rl.eval.n_rollout_episodes),
        deterministic=bool(cfg.rl.eval.deterministic),
        cfg=cfg,
    )
    log.info(
        "Final eval: success_rate=%.3f  mean_steps=%.2f  mean_reward=%.3f",
        metrics["success_rate"], metrics["mean_steps"], metrics["mean_reward"],
    )

    vec_env.close()
    eval_env.close()
    return model


def evaluate_policy(
    model: Any,
    env: Any,
    n_episodes: int = 500,
    deterministic: bool = False,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a trained policy and dump Contract 4 rollouts.

    Returns
    -------
    dict
        ``{"success_rate", "mean_steps", "mean_reward", "action_freq"}``.
    """
    import polars as pl

    # Gene-symbol lookup (action idx → symbol)
    gene_lookup: dict[int, str] = {}
    if cfg is not None:
        try:
            with open(cfg.paths.vae_gene_vocab_json) as f:
                vocab = json.load(f)
            genes = vocab["genes"]
            noop_idx = int(vocab["noop_idx"])
            for i, g in enumerate(genes):
                gene_lookup[i] = str(g)
            gene_lookup[noop_idx] = "NO_OP"
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load gene_vocab.json (%s) — gene_symbol will be 'gene_<idx>'.", exc)

    rows: list[dict[str, Any]] = []
    action_freq: dict[str, int] = {}
    success_count = 0
    step_counts: list[int] = []
    reward_totals: list[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        terminated = False
        truncated = False
        ep_steps = 0
        ep_reward = 0.0
        ep_terminal_success = False

        while not (terminated or truncated):
            mask = info["action_mask"]
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
            action = int(np.asarray(action).item())
            sym = gene_lookup.get(action, f"gene_{action}")
            action_freq[sym] = action_freq.get(sym, 0) + 1

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1

            rows.append({
                "episode_id": int(ep),
                "step": int(ep_steps),
                "action": int(action),
                "gene_symbol": sym,
                "z_norm": float(info["distance"]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "success": bool(info.get("success", False)),
                "z_vector": obs.astype(np.float32).tolist(),
            })

            ep_terminal_success = bool(info.get("success", False))

        if ep_terminal_success and terminated:
            success_count += 1
        step_counts.append(ep_steps)
        reward_totals.append(ep_reward)

    success_rate = success_count / max(n_episodes, 1)
    mean_steps = float(np.mean(step_counts)) if step_counts else 0.0
    mean_reward = float(np.mean(reward_totals)) if reward_totals else 0.0

    # Write Contract 4 artifacts
    if cfg is not None:
        rollouts_df = pl.DataFrame(rows) if rows else pl.DataFrame()
        rollouts_path = Path(cfg.paths.rl_rollouts_parquet)
        rollouts_path.parent.mkdir(parents=True, exist_ok=True)
        rollouts_df.write_parquet(str(rollouts_path))
        log.info("Saved rollouts → %s (%d rows, %d episodes)", rollouts_path, len(rows), n_episodes)

        action_freq_path = Path(cfg.paths.rl_action_freq_json)
        action_freq_path.parent.mkdir(parents=True, exist_ok=True)
        with open(action_freq_path, "w") as f:
            json.dump(action_freq, f, indent=2)
        log.info("Saved action_freq → %s", action_freq_path)

    return {
        "success_rate": float(success_rate),
        "mean_steps": float(mean_steps),
        "mean_reward": float(mean_reward),
        "action_freq": action_freq,
    }


# ---------------------------------------------------------------------- #
# Plot helper — eval success curve
# ---------------------------------------------------------------------- #


def _plot_success_curve(eval_logs_dir: Path, out_path: Path) -> None:
    """Parse SB3 eval evaluations.npz and plot mean reward over training."""
    import matplotlib.pyplot as plt

    npz_path = eval_logs_dir / "evaluations.npz"
    if not npz_path.exists():
        log.warning("Eval log %s not found — skipping success-curve plot.", npz_path)
        return

    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]  # (n_evals, n_episodes) — per-episode rewards

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timesteps, mean_rewards, label="mean reward")
    ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Mean episode reward (eval)")
    ax.set_title("MaskablePPO — training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved success curve → %s", out_path)
