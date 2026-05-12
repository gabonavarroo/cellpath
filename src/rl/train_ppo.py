"""MaskablePPO training loop using ``sb3-contrib``.

Owner: Agent B. See ARCHITECTURE.md Concept 4 and AGENTS.md §2 Phase 3.

Hard gate
---------
:func:`train_ppo` reads ``artifacts/dynamics/gate.json`` and refuses to start unless
``passed=True``. The override flag ``cfg.rl.train.skip_gate`` exists for dev runs but
emits a P0 warning to the rich console *and* to TensorBoard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def check_dynamics_gate(gate_path: str | Path, skip: bool = False) -> dict[str, Any]:
    """Check the dynamics validation gate before RL training.

    Parameters
    ----------
    gate_path
        Path to ``artifacts/dynamics/gate.json``.
    skip
        If True, bypass the gate check and return a synthetic ``{"passed": True, "override": True}``.
        Caller must log a loud warning.

    Returns
    -------
    dict
        Gate JSON contents (or override marker).

    Raises
    ------
    NotImplementedError
        Agent B: implement.
    FileNotFoundError
        If ``gate_path`` is missing.
    SystemExit
        If the gate is failed and ``skip=False`` — exit code 2.
    """
    raise NotImplementedError(
        "Agent B: read gate.json; if passed=False and skip=False, exit(2) with a clear message. "
        "If skip=True, log a P0-tagged warning to rich console + tb."
    )


def train_ppo(cfg: Any) -> Any:
    """Train MaskablePPO over the ``CellReprogrammingEnv``.

    Workflow
    --------
    1. :func:`check_dynamics_gate` — exit if gate not passed (unless overridden).
    2. Load dynamics model + VAE artifacts (``z_reference_centroid``, ``epsilon_success``,
       ``gene_vocab``).
    3. Build vectorized envs via :func:`src.rl.environment.make_env_factory` and ``SubprocVecEnv``.
    4. Instantiate ``sb3_contrib.MaskablePPO`` with ``cfg.rl.ppo`` hyperparameters.
    5. Train for ``cfg.rl.ppo.total_timesteps``, periodically evaluating and saving.
    6. Final rollout dump to ``cfg.paths.rl_rollouts_parquet``.

    Parameters
    ----------
    cfg
        Hydra config (must have ``rl``, ``paths``, ``seed``).

    Returns
    -------
    sb3_contrib.MaskablePPO
        Trained policy.

    Raises
    ------
    NotImplementedError
        Agent B: implement the 6-step workflow.
    """
    raise NotImplementedError(
        "Agent B: implement MaskablePPO training. "
        "Refuse to start unless dynamics gate passed (or skip_gate=True with loud warning). "
        "Use SubprocVecEnv with n_envs from config; TensorBoard logging from step 0."
    )


def evaluate_policy(
    model: Any,
    env: Any,
    n_episodes: int = 500,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Evaluate a trained policy and return per-episode metrics.

    Parameters
    ----------
    model
        Trained MaskablePPO.
    env
        Single (non-vectorized) eval env.
    n_episodes
        Number of rollout episodes.
    deterministic
        If True, take argmax action; else sample.

    Returns
    -------
    dict
        ``{"success_rate": float, "mean_steps": float, "mean_reward": float,
           "action_freq": dict[gene_symbol, int]}``.

    Raises
    ------
    NotImplementedError
        Agent B: implement, including action-mask handling at each step.
    """
    raise NotImplementedError(
        "Agent B: rollouts with action masking; aggregate success/length/reward/action_freq."
    )
