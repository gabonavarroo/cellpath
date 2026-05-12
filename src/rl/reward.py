"""Reward function for ``CellReprogrammingEnv``.

Owner: Agent B. See ARCHITECTURE.md Concept 5.

Composition::

    R(z, a, z', sigma) = − distance(z', z_ref) * λ_dist
                          − λ_sparse · 1[a ≠ NO_OP]
                          − λ_unc   · ||sigma||
                          + success_bonus · 1[terminal ∧ ||z' − z_ref|| < ε]
                          − failure_penalty · 1[truncation]

The distance term provides dense shaping; the sparsity term encodes the "fewer interventions
preferred" prior. The uncertainty term (optional, off by default) discourages the agent from
visiting high-uncertainty regions of the dynamics model.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
) -> float:
    """Scalar reward for one transition.

    Parameters
    ----------
    z_next
        Post-step latent, shape ``(n_latent,)``.
    z_ref
        Reference centroid, shape ``(n_latent,)``.
    action
        Action taken at this step.
    noop_idx
        Index of the NO-OP action.
    log_var
        Optional predicted ``log σ²`` for the action, shape ``(n_latent,)``. Used only if
        ``lambda_unc > 0``.
    lambda_sparse, lambda_unc, distance_scale, success_bonus, failure_penalty
        Hyperparameters (see ``config/rl.yaml::reward``).
    terminated
        True if the episode terminated this step.
    truncated
        True if the step budget was reached.
    is_success
        True if the terminal state is within ε of ``z_ref``.
    distance_metric
        ``"l2"`` (default) or ``"cosine"``.

    Returns
    -------
    float
        Scalar reward for this transition.

    Raises
    ------
    NotImplementedError
        Agent B: implement per the formula in the module docstring.
    """
    raise NotImplementedError(
        "Agent B: assemble reward per the docstring composition. "
        "Note: NO-OP does NOT pay the sparsity penalty (it is the 'stop' action)."
    )


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
        ``"l2"`` (default) or ``"cosine"``.

    Returns
    -------
    float

    Raises
    ------
    NotImplementedError
        Agent B: implement.
    """
    raise NotImplementedError("Agent B: numpy L2 / cosine distance.")
