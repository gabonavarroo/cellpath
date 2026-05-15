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

import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# =============================================================================
# Per-run metadata
# =============================================================================


_METADATA_SCHEMA_VERSION = 1


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str | None:
    """Compute the SHA-256 of a file, or return ``None`` if it's missing."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except OSError as exc:
        log.warning("Could not hash %s: %s", p, exc)
        return None


def _git_commit() -> str | None:
    """Return the current HEAD SHA, or ``None`` if not in a git repo / git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("git rev-parse HEAD failed: %s", exc)
    return None


def _safe_to_container(node: Any) -> Any:
    """Convert an OmegaConf node to a plain Python container; pass through otherwise."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(node):
            return OmegaConf.to_container(node, resolve=True)
    except Exception:  # noqa: BLE001
        pass
    return node


def _read_gate_status(cfg: Any) -> tuple[bool | None, bool | None]:
    """Best-effort read of ``gate.json`` — returns ``(passed, overridden)``.

    Both fields may be ``None`` if the gate file is missing or unreadable.
    ``overridden`` is derived from ``cfg.rl.train.skip_gate`` and the presence of the
    ``"override"`` key written by :func:`check_dynamics_gate` when ``skip=True``.
    """
    gate_path = Path(cfg.paths.dynamics_gate) if hasattr(cfg.paths, "dynamics_gate") else None
    passed: bool | None = None
    if gate_path is not None and gate_path.exists():
        try:
            with open(gate_path) as f:
                blob = json.load(f)
            passed = bool(blob.get("passed", False))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not parse %s: %s", gate_path, exc)

    overridden = None
    try:
        overridden = bool(cfg.rl.train.get("skip_gate", False))
    except Exception:  # noqa: BLE001
        pass
    return passed, overridden


def _read_dynamics_arch_flags(cfg: Any) -> dict[str, Any]:
    """Read dynamics architecture flags from saved config.json (preferred) or cfg.dynamics."""
    out: dict[str, Any] = {}
    cfg_path = Path(cfg.paths.dynamics_config) if hasattr(cfg.paths, "dynamics_config") else None
    if cfg_path is not None and cfg_path.exists():
        try:
            with open(cfg_path) as f:
                dyn_blob = json.load(f)
            out["use_state_linear_skip"] = bool(dyn_blob.get("use_state_linear_skip", False))
            out["use_gene_delta_bias"] = bool(dyn_blob.get("use_gene_delta_bias", False))
            out["n_latent"] = int(dyn_blob.get("n_latent", -1))
            out["n_hidden"] = int(dyn_blob.get("n_hidden", -1))
            out["n_layers"] = int(dyn_blob.get("n_layers", -1))
            out["d_emb"] = int(dyn_blob.get("d_emb", -1))
            return out
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not parse %s: %s", cfg_path, exc)
    # Fall back to live cfg
    try:
        out["use_state_linear_skip"] = bool(cfg.dynamics.get("use_state_linear_skip", False))
        out["use_gene_delta_bias"] = bool(cfg.dynamics.get("use_gene_delta_bias", False))
    except Exception:  # noqa: BLE001
        pass
    return out


def _write_run_metadata(
    cfg: Any,
    out_dir: str | Path,
    *,
    deterministic: bool | None = None,
    n_episodes: int | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Write a ``metadata.json`` capturing the full provenance of an RL run.

    Robust to missing files (gate.json, epsilon_success.json, .git, dynamics config). The
    intent is that every PPO train run and every PPO eval / random-policy eval produces a
    self-contained record that can be cited next to its numbers.

    Parameters
    ----------
    cfg
        Hydra config (after any open_dict overrides — the snapshot taken here is what was
        actually used).
    out_dir
        Directory the metadata is written into. Created if needed.
    deterministic, n_episodes
        Eval flags. ``None`` for training-only runs (e.g. ``train_ppo()`` itself).
    extras
        Additional free-form fields merged at the top level (e.g. ``{"policy_kind": "uniform_valid"}``
        for the random-policy script). Wins over auto-derived fields on key collision.

    Returns
    -------
    Path
        The full path to the written ``metadata.json``.
    """
    from src.rl.environment import resolve_epsilon

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Epsilon provenance
    try:
        epsilon_value, epsilon_source = resolve_epsilon(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not resolve epsilon for metadata: %s", exc)
        epsilon_value, epsilon_source = None, None

    # Epsilon percentile from the canonical JSON (independent of override; helps comparisons)
    epsilon_percentile_json = None
    try:
        with open(cfg.paths.vae_epsilon_success_json) as f:
            epsilon_percentile_json = json.load(f).get("percentile")
    except (OSError, json.JSONDecodeError):
        pass

    gate_passed, gate_overridden = _read_gate_status(cfg)
    dyn_arch = _read_dynamics_arch_flags(cfg)

    # Dynamics checkpoint
    dyn_ckpt = (
        str(cfg.paths.dynamics_model) if hasattr(cfg.paths, "dynamics_model") else None
    )
    dyn_sha = _sha256_of_file(Path(dyn_ckpt)) if dyn_ckpt else None

    # Policy path
    policy_path = (
        str(cfg.paths.rl_ppo_zip) if hasattr(cfg.paths, "rl_ppo_zip") else None
    )

    # VAE
    vae_n_latent = None
    try:
        vae_n_latent = int(cfg.vae.n_latent)
    except Exception:  # noqa: BLE001
        pass
    vae_dir = str(cfg.paths.vae_dir) if hasattr(cfg.paths, "vae_dir") else None

    # min_start_distance — preserve the string form ("auto" / "none") faithfully
    try:
        min_start_distance = cfg.rl.env.get("min_start_distance", "auto")
        # OmegaConf may give us a non-JSON-safe wrapper; coerce
        if not isinstance(min_start_distance, (str, int, float, type(None))):
            min_start_distance = str(min_start_distance)
    except Exception:  # noqa: BLE001
        min_start_distance = None

    meta: dict[str, Any] = {
        "schema_version": _METADATA_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": int(cfg.get("seed", 42)) if hasattr(cfg, "get") else None,
        "epsilon_value": epsilon_value,
        "epsilon_source": epsilon_source,
        "epsilon_percentile_json": epsilon_percentile_json,
        "min_start_distance": min_start_distance,
        "max_steps": int(cfg.rl.env.max_steps) if hasattr(cfg.rl.env, "max_steps") else None,
        "vae_n_latent": vae_n_latent,
        "vae_dir": vae_dir,
        "dynamics_checkpoint": dyn_ckpt,
        "dynamics_checkpoint_sha256": dyn_sha,
        "dynamics_gate_passed": gate_passed,
        "dynamics_gate_overridden": gate_overridden,
        "dynamics_arch": dyn_arch,
        "ppo_hparams": _safe_to_container(getattr(cfg.rl, "ppo", None)),
        "reward_hparams": _safe_to_container(getattr(cfg.rl, "reward", None)),
        "reward_mode": "absolute_distance",  # only mode implemented in P0
        "deterministic_eval": deterministic,
        "n_episodes": n_episodes,
        "policy_path": policy_path,
        "notes": None,
    }

    if extras:
        # Caller wins on key collision (intentional: lets the random-policy script overwrite
        # policy_path with "random_policy" or similar).
        meta.update(extras)

    out_path = out_dir / "metadata.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info("Wrote metadata → %s", out_path)
    return out_path


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

    # Per-run provenance — written next to ppo.zip / rollouts.parquet.
    try:
        _write_run_metadata(
            cfg,
            rl_dir,
            deterministic=bool(cfg.rl.eval.deterministic),
            n_episodes=int(cfg.rl.eval.n_rollout_episodes),
            extras={
                "stage": "train_ppo",
                "training_total_timesteps": int(cfg.rl.ppo.total_timesteps),
                "training_elapsed_sec": float(elapsed),
                "final_eval_metrics": {
                    "success_rate": metrics["success_rate"],
                    "mean_steps": metrics["mean_steps"],
                    "mean_reward": metrics["mean_reward"],
                },
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write run metadata: %s", exc)

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
