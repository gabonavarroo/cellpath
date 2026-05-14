"""Reward function for ``CellReprogrammingEnv``.

Owner: Agent B. See ARCHITECTURE.md Concept 5.

Composition::

    R(z, a, z', sigma) = − distance(z', z_ref) * λ_dist
                          − λ_sparse · 1[a ≠ NO_OP]
                          − λ_unc   · mean(exp(log_var))
                          + success_bonus · 1[terminal ∧ ||z' − z_ref|| < ε]
                          − failure_penalty · 1[truncation]

The distance term provides dense shaping; the sparsity term encodes the "fewer interventions
preferred" prior. The uncertainty term (optional, off by default) discourages the agent from
visiting high-uncertainty regions of the dynamics model.
"""

from __future__ import annotations

import numpy as np


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
        Non-negative distance. For cosine, result is in ``[0, 2]``; for L2, ``[0, ∞)``.
        Zero vectors under cosine return ``1.0`` (maximum uncertainty) rather than NaN.

    Raises
    ------
    ValueError
        If ``metric`` is not ``"l2"`` or ``"cosine"``.
    """
    z    = np.asarray(z,    dtype=np.float32)
    z_ref = np.asarray(z_ref, dtype=np.float32)

    if metric == "l2":
        return float(np.linalg.norm(z - z_ref))

    if metric == "cosine":
        norm_z    = float(np.linalg.norm(z))
        norm_ref  = float(np.linalg.norm(z_ref))
        if norm_z == 0.0 or norm_ref == 0.0:
            return 1.0
        cos_sim = float(np.dot(z, z_ref) / (norm_z * norm_ref))
        # clamp to [-1, 1] to guard against floating-point drift
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return 1.0 - cos_sim

    raise ValueError(
        f"Unknown distance metric: {metric!r}. Choose 'l2' or 'cosine'."
    )


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

    Notes
    -----
    NO-OP does not pay the sparsity penalty — it is the "stop" action and should never
    be penalised for frugality. The uncertainty penalty requires ``lambda_unc > 0`` AND
    a non-None ``log_var``; both conditions must hold.
    """
    d = distance_to_reference(z_next, z_ref, metric=distance_metric)
    r = -distance_scale * d

    if action != noop_idx:
        r -= lambda_sparse

    if lambda_unc > 0.0 and log_var is not None:
        var = np.exp(np.asarray(log_var, dtype=np.float64))
        r -= lambda_unc * float(np.mean(var))

    if terminated and is_success:
        r += success_bonus

    if truncated:
        r -= failure_penalty

    return float(r)
