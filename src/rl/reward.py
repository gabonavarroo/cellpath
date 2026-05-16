"""Reward function for ``CellReprogrammingEnv``.

Owner: Agent B. See ARCHITECTURE.md Concept 5 and decision D7.

Composition (locked, do NOT change without coordinating with AGENTS.md §4)::

    R(z, a, z', σ) = − distance(z', z_ref) · λ_dist
                      − λ_sparse · 𝟙[a ≠ NO_OP]
                      − λ_unc   · ‖σ‖
                      + success_bonus · 𝟙[terminal ∧ ‖z' − z_ref‖ < ε]
                      − failure_penalty · 𝟙[truncation]

The distance term provides dense shaping; the sparsity term encodes the "fewer interventions
preferred" prior. The uncertainty term (optional, off by default) discourages the agent from
visiting high-uncertainty regions of the dynamics model.

**NO-OP semantics (sacred, D7):** NO-OP earns no sparsity penalty — it is the "stop" action.
Without this exemption, the policy would be penalised for the rational choice of stopping,
which contradicts D7 and the unit tests in tests/test_environment.py.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_EPS = 1e-12  # numerical guard for cosine norm denominators


def compute_reward(
    z_next: np.ndarray,
    z_ref: np.ndarray,
    action: int,
    noop_idx: int,
    log_var: np.ndarray | None = None,
    *,
    lambda_sparse: float = 0.05,
    lambda_unc: float = 0.0,
    distance_scale: float = 1.0,
    success_bonus: float = 0.0,
    failure_penalty: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    is_success: bool = False,
    distance_metric: str = "l2",
    reward_mode: str = "absolute_distance",
    prev_distance: float | None = None,
    beta_step_cost: float = 0.05,
    step_idx: int = 0,
) -> float:
    """Scalar reward for one transition.

    Parameters
    ----------
    z_next
        Post-step latent (for non-NO-OP) or current latent (for NO-OP),
        shape ``(n_latent,)``.
    z_ref
        Reference centroid, shape ``(n_latent,)``.
    action
        Action taken at this step. If ``action == noop_idx``, the sparsity
        penalty is NOT applied (D7).
    noop_idx
        Index of the NO-OP action.
    log_var
        Optional predicted ``log σ²`` for the action, shape ``(n_latent,)``. Used only if
        ``lambda_unc > 0``.
    lambda_sparse, lambda_unc, distance_scale, success_bonus, failure_penalty
        Hyperparameters (see ``config/rl.yaml::reward``).
    terminated
        True if the episode terminated this step (NO-OP or in-step success).
    truncated
        True if the step budget was reached without termination.
    is_success
        True if the terminal state is within ε of ``z_ref``.
    distance_metric
        ``"l2"`` (default) or ``"cosine"``.
    reward_mode
        P0D Track B. One of:
          * ``"absolute_distance"`` (default, V1 behaviour): ``R = -d_next·distance_scale``.
          * ``"delta_distance"``: ``R = (d_prev - d_next)·distance_scale`` — rewards progress
            per step. Requires ``prev_distance`` to be provided.
          * ``"terminal_only_step_cost"``: ``R = 0`` mid-episode; on terminal step,
            ``R = 1·is_success - beta_step_cost·step_idx``.
        ``lambda_sparse``, ``lambda_unc``, ``success_bonus``, ``failure_penalty`` are
        still applied additively as in the absolute_distance mode (they shape, not
        replace, the chosen base reward). The exception is ``terminal_only_step_cost``,
        which already encodes the success signal — passing ``success_bonus > 0`` would
        double-count.
    prev_distance
        Distance from the *pre-step* state to z_ref. Required for ``reward_mode="delta_distance"``.
        For NO-OP and ``absolute_distance``/``terminal_only_step_cost`` modes it is ignored.
    beta_step_cost
        Per-step cost in the ``terminal_only_step_cost`` mode. Default 0.05.
    step_idx
        Current step index (0-based), used by ``terminal_only_step_cost``.

    Returns
    -------
    float
        Scalar reward for this transition.
    """
    d_next = distance_to_reference(z_next, z_ref, metric=distance_metric)

    # --- 1. Base reward by mode -------------------------------------------
    if reward_mode == "absolute_distance":
        reward = -float(distance_scale) * d_next
    elif reward_mode == "delta_distance":
        if prev_distance is None:
            raise ValueError(
                "reward_mode='delta_distance' requires prev_distance to be provided. "
                "Pass distance_to_reference(z_pre, z_ref) from the environment."
            )
        reward = float(distance_scale) * (float(prev_distance) - d_next)
    elif reward_mode == "terminal_only_step_cost":
        if terminated or truncated:
            reward = float(distance_scale) * (1.0 if is_success else 0.0)
            reward -= float(beta_step_cost) * float(step_idx)
        else:
            reward = 0.0
    else:
        raise ValueError(
            f"Unknown reward_mode {reward_mode!r}. Choose from "
            "{'absolute_distance', 'delta_distance', 'terminal_only_step_cost'}."
        )

    # --- 2. Sparsity penalty (gene actions only, D7) ----------------------
    if action != noop_idx:
        reward -= float(lambda_sparse)

    # --- 3. Optional uncertainty penalty ---------------------------------
    if lambda_unc > 0.0 and log_var is not None:
        # ‖σ‖ where σ = exp(½ · log_var). Using L2 norm of σ over latent dims.
        sigma = np.exp(0.5 * np.asarray(log_var, dtype=np.float64))
        reward -= float(lambda_unc) * float(np.linalg.norm(sigma))

    # --- 4. Terminal bonuses / penalties ---------------------------------
    if terminated and is_success and success_bonus > 0.0:
        reward += float(success_bonus)
    if truncated and failure_penalty > 0.0:
        reward -= float(failure_penalty)

    return float(reward)


def distance_to_reference(
    z: np.ndarray,
    z_ref: np.ndarray,
    metric: str = "l2",
) -> float:
    """Distance between a latent state and the reference centroid.

    Parameters
    ----------
    z
        Shape ``(n_latent,)``.
    z_ref
        Shape ``(n_latent,)``.
    metric
        ``"l2"`` (default — Euclidean) or ``"cosine"`` (in [0, 2]).

    Returns
    -------
    float

    Notes
    -----
    Cosine guards against zero-norm vectors via a small epsilon; near-zero vectors return a
    distance of 1.0 (orthogonal-by-convention).
    """
    z = np.asarray(z, dtype=np.float64)
    z_ref = np.asarray(z_ref, dtype=np.float64)

    if metric == "l2":
        return float(np.linalg.norm(z - z_ref))

    if metric == "cosine":
        nz = np.linalg.norm(z)
        nr = np.linalg.norm(z_ref)
        if nz < _EPS or nr < _EPS:
            return 1.0  # by convention — undefined direction
        cos_sim = float(np.dot(z, z_ref) / (nz * nr))
        # Clamp for numerical safety (dot can drift outside [-1, 1] for unit-norm float64).
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return 1.0 - cos_sim

    raise ValueError(f"Unknown distance metric: {metric!r}. Use 'l2' or 'cosine'.")
