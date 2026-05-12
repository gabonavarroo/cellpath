"""Gymnasium env that wraps the learned dynamics model.

Owner: Agent B. See ARCHITECTURE.md Concepts 4 + 5 and AGENTS.md §2 (Phase 3).

Semantics (locked by Contract 3 + AGENTS.md):
- ``action_space = Discrete(n_genes + 1)``. The final index is **NO-OP / terminate**.
- ``reset()`` samples ``z₀`` from a random perturbation cluster (off-target).
- ``step(action)``:
    * If ``action == noop_idx``: ``terminated = True``;
      **success := (||z − z_ref|| < ε_success)**. NO-OP does **not** count as success unless
      the current state is already within ε.
    * Else: pass ``(z, gene_idx)`` through the dynamics model; update the repeat-mask;
      recompute success.
- ``info["action_mask"]: (n_genes + 1,) bool`` is populated every step for ``MaskablePPO``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class CellReprogrammingEnv:
    """Latent-space steering environment.

    Subclass of :class:`gymnasium.Env`. Agent B implements the full body; this stub fixes the
    signature so other modules can import and reference type hints.

    Parameters
    ----------
    dynamics_model
        Trained ``PerturbationDynamicsModel``. Eval mode; no gradients during rollouts.
    z_reference_centroid
        Shape ``(n_latent,)``. Loaded from ``artifacts/vae/z_reference_centroid.npy``.
    epsilon_success
        Threshold for success. Loaded from ``artifacts/vae/epsilon_success.json`` (computed
        as a percentile of the NT-control distance distribution; see
        :func:`src.models.vae.compute_epsilon_success`).
    n_genes
        Action-space size excluding NO-OP. Loaded from ``gene_vocab.json["n_genes"]``.
    max_steps
        Episode budget K.
    lambda_sparse
        Per-step sparsity penalty.
    lambda_unc
        Per-step uncertainty penalty (set to 0 to disable).
    repeat_mask
        If True (default), genes used earlier in the episode are masked.
    start_state_strategy
        How to pick ``z₀`` in ``reset()``.
    distance_metric
        ``"l2"`` (default) or ``"cosine"``.
    start_pool_latents
        Optional ``(N, n_latent)`` matrix of starting-state candidates. Defaults to all
        non-control cells from ``latents.h5ad`` at construction time.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dynamics_model: Any,
        z_reference_centroid: np.ndarray,
        epsilon_success: float,
        n_genes: int,
        max_steps: int = 10,
        lambda_sparse: float = 0.05,
        lambda_unc: float = 0.0,
        repeat_mask: bool = True,
        start_state_strategy: str = "random_perturbation",
        distance_metric: str = "l2",
        start_pool_latents: np.ndarray | None = None,
        success_bonus: float = 0.0,
        failure_penalty: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.dynamics_model = dynamics_model
        self.z_ref = np.asarray(z_reference_centroid, dtype=np.float32)
        self.epsilon = float(epsilon_success)
        self.n_genes = int(n_genes)
        self.noop_idx = int(n_genes)            # the final action index
        self.max_steps = int(max_steps)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_unc = float(lambda_unc)
        self.repeat_mask_enabled = bool(repeat_mask)
        self.start_strategy = str(start_state_strategy)
        self.distance_metric = str(distance_metric)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self._start_pool = start_pool_latents
        # Subclass / agent B implementation populates:
        #   self.action_space = gymnasium.spaces.Discrete(n_genes + 1)
        #   self.observation_space = gymnasium.spaces.Box(-inf, inf, (n_latent,))
        #   self._rng = np.random.default_rng(seed)
        #   self._z, self._step_idx, self._used_genes

    # ---------------------------------------------------------------------
    # gymnasium API
    # ---------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a new starting state.

        Parameters
        ----------
        seed
            Episode seed; if provided, re-seed the internal RNG.
        options
            Optional dict with ``"start_z"`` key to override the sampled start.

        Returns
        -------
        obs : np.ndarray
            Shape ``(n_latent,)`` float32 — the starting latent.
        info : dict
            Must include ``"action_mask": (n_genes + 1,) bool`` for ``MaskablePPO``.

        Raises
        ------
        NotImplementedError
            Agent B: implement start-state sampling per ``start_state_strategy`` and populate
            ``info["action_mask"]``.
        """
        raise NotImplementedError(
            "Agent B: implement reset() per gymnasium API. Populate info['action_mask']."
        )

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action; advance the environment.

        Parameters
        ----------
        action
            Integer in ``[0, n_genes]``. ``action == noop_idx`` terminates the episode.

        Returns
        -------
        obs : np.ndarray
            Next latent, shape ``(n_latent,)``.
        reward : float
            See :mod:`src.rl.reward`.
        terminated : bool
            True on NO-OP or success.
        truncated : bool
            True on step budget K reached.
        info : dict
            Must include:
              - ``"action_mask": (n_genes + 1,) bool`` (next step's mask).
              - ``"success": bool``.
              - ``"distance": float`` (current ``||z - z_ref||``).
              - ``"step": int`` (post-increment).

        Raises
        ------
        NotImplementedError
            Agent B: implement per ARCHITECTURE.md §1 RL block. NO-OP success rule is in D7.

        Notes
        -----
        **NO-OP semantics (sacred, locked):**
          - If ``action == noop_idx``: ``terminated = True``;
            ``success := (||z - z_ref|| < epsilon_success)`` evaluated on the
            *current* state, NOT after any displacement.
          - NO-OP earns no sparsity penalty (it is the "stop" action).
        """
        raise NotImplementedError(
            "Agent B: implement step(). NO-OP terminates with success conditional on distance. "
            "Repeat-mask updates after non-NO-OP actions. Reward via src.rl.reward."
        )

    def action_masks(self) -> np.ndarray:
        """Current action mask, for ``MaskablePPO``.

        Returns
        -------
        np.ndarray
            Shape ``(n_genes + 1,) bool``. ``False`` for masked (forbidden) actions.

        Raises
        ------
        NotImplementedError
            Agent B: return the current repeat-mask + NO-OP always available.
        """
        raise NotImplementedError(
            "Agent B: return action_mask. Repeat-mask zeros out used genes; NO-OP always True."
        )


def make_env_factory(cfg: Any) -> Any:
    """Return a callable that constructs a ``CellReprogrammingEnv`` from artifacts.

    The returned factory is what ``stable_baselines3.common.vec_env.SubprocVecEnv`` consumes.

    Parameters
    ----------
    cfg
        Hydra config (must have ``rl``, ``paths``).

    Returns
    -------
    callable
        ``factory() -> CellReprogrammingEnv``. Loads dynamics model, ``z_reference_centroid``,
        ``epsilon_success``, and ``gene_vocab`` from ``cfg.paths`` on each call.

    Raises
    ------
    NotImplementedError
        Agent B: implement. Must load all four artifacts on each call so subprocess envs work.
    """
    raise NotImplementedError(
        "Agent B: factory loads dynamics + centroid + epsilon + gene_vocab. "
        "Used by SubprocVecEnv. See AGENTS.md §4 Contract 1 + 3."
    )
