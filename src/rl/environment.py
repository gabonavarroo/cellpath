"""Gymnasium env that wraps the learned dynamics model.

Owner: Agent B. See ARCHITECTURE.md Concepts 4 + 5 and AGENTS.md §2 (Phase 3).

Semantics (locked by Contract 3 + AGENTS.md):
- ``action_space = Discrete(n_genes + 1)``. The final index is **NO-OP / terminate**.
- ``reset()`` samples ``z₀`` from a random perturbation cluster (off-target).
- ``step(action)``:
    * If ``action == noop_idx``: ``terminated = True``;
      **success := (||z − z_ref|| < ε_success)**. NO-OP does **not** count as success unless
      the current state is already within ε. (D7)
    * Else: pass ``(z, gene_idx)`` through the dynamics model; update the repeat-mask;
      recompute success.
- ``info["action_mask"]: (n_genes + 1,) bool`` is populated every step for ``MaskablePPO``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.rl.reward import compute_reward, distance_to_reference

log = logging.getLogger(__name__)


class CellReprogrammingEnv(gym.Env):
    """Latent-space steering environment.

    See module docstring for the locked semantic contract.

    Parameters
    ----------
    dynamics_model
        Trained ``PerturbationDynamicsModel`` (eval mode). May be a callable returning
        ``(z_next, mu, log_var)`` for testing.
    z_reference_centroid
        Shape ``(n_latent,)``. Loaded from ``artifacts/vae/z_reference_centroid.npy``.
    epsilon_success
        Threshold for success. Loaded from ``artifacts/vae/epsilon_success.json``.
    n_genes
        Action-space size excluding NO-OP. Loaded from ``gene_vocab.json["n_genes"]``.
    max_steps
        Episode budget K.
    lambda_sparse
        Per-step sparsity penalty (only paid by gene actions, not NO-OP).
    lambda_unc
        Per-step uncertainty penalty (set to 0 to disable).
    repeat_mask
        If True (default), genes used earlier in the episode are masked.
    start_state_strategy
        How to pick ``z₀`` in ``reset()``. ``"random_perturbation"`` samples uniformly from
        ``start_pool_latents``.
    distance_metric
        ``"l2"`` (default) or ``"cosine"``.
    start_pool_latents
        Optional ``(N, n_latent)`` matrix of starting-state candidates. If None and the
        strategy needs it, callers must pass it (the factory loads it from latents.h5ad).
    success_bonus, failure_penalty
        Terminal shaping (default 0).
    seed
        RNG seed for start-state sampling.
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
        super().__init__()

        self.dynamics_model = dynamics_model
        self.z_ref = np.asarray(z_reference_centroid, dtype=np.float32)
        self.epsilon = float(epsilon_success)
        self.n_genes = int(n_genes)
        self.noop_idx = int(n_genes)
        self.max_steps = int(max_steps)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_unc = float(lambda_unc)
        self.repeat_mask_enabled = bool(repeat_mask)
        self.start_strategy = str(start_state_strategy)
        self.distance_metric = str(distance_metric)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self._start_pool = (
            np.asarray(start_pool_latents, dtype=np.float32)
            if start_pool_latents is not None
            else None
        )

        n_latent = int(self.z_ref.shape[0])

        # gymnasium spaces — dtype MUST match obs dtype for check_env compliance
        self.action_space = spaces.Discrete(self.n_genes + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_latent,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)

        # Episode state (populated in reset)
        self._z: np.ndarray = np.zeros(n_latent, dtype=np.float32)
        self._step_idx: int = 0
        self._used_genes: set[int] = set()
        self._current_mask: np.ndarray = np.ones(self.n_genes + 1, dtype=bool)

    # ------------------------------------------------------------------ #
    # gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a new starting state.

        Sampling strategy:
        - If ``options["start_z"]`` is provided, use it (deterministic eval).
        - Else if ``start_state_strategy == "random_perturbation"``, sample uniformly from
          ``_start_pool``.
        - If ``_start_pool`` is None, fall back to random Gaussian (smoke testing only).
        """
        # Required by gymnasium API: super().reset() seeds self.np_random correctly
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Resolve starting latent
        if options is not None and "start_z" in options:
            z0 = np.asarray(options["start_z"], dtype=np.float32)
        elif self.start_strategy == "random_perturbation" and self._start_pool is not None:
            idx = int(self._rng.integers(0, len(self._start_pool)))
            z0 = self._start_pool[idx].astype(np.float32, copy=True)
        else:
            # Fallback for smoke testing without a pool
            z0 = self._rng.standard_normal(self.z_ref.shape[0]).astype(np.float32)

        self._z = z0
        self._step_idx = 0
        self._used_genes = set()
        self._current_mask = self._compute_mask()

        info = {
            "action_mask": self._current_mask.copy(),
            "distance": float(distance_to_reference(self._z, self.z_ref, metric=self.distance_metric)),
            "step": 0,
        }
        return self._z.copy(), info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action; advance the environment."""
        action = int(action)

        if action == self.noop_idx:
            # --- NO-OP: terminate using CURRENT state (D7) ---
            d = float(distance_to_reference(self._z, self.z_ref, metric=self.distance_metric))
            is_success = bool(d < self.epsilon)
            terminated = True
            truncated = False

            reward = compute_reward(
                z_next=self._z,
                z_ref=self.z_ref,
                action=action,
                noop_idx=self.noop_idx,
                log_var=None,
                lambda_sparse=self.lambda_sparse,
                lambda_unc=self.lambda_unc,
                distance_scale=1.0,
                success_bonus=self.success_bonus,
                failure_penalty=self.failure_penalty,
                terminated=terminated,
                truncated=truncated,
                is_success=is_success,
                distance_metric=self.distance_metric,
            )

            # Mask not relevant on terminal step; leave it consistent
            self._current_mask = self._compute_mask()
            info = {
                "action_mask": self._current_mask.copy(),
                "success": is_success,
                "distance": d,
                "step": self._step_idx,
            }
            return self._z.copy(), reward, terminated, truncated, info

        # --- Gene action: apply dynamics, update repeat-mask ---
        # gene_idx is 1-indexed in dynamics (0 reserved for ctrl placeholder)
        gene_idx = action + 1

        log_var_np: np.ndarray | None = None
        with torch.no_grad():
            z_t = torch.from_numpy(self._z).float().unsqueeze(0)  # (1, n_latent)
            g_t = torch.tensor([gene_idx], dtype=torch.long)
            out = self.dynamics_model(z_t, g_t)

        # The dynamics model returns (z_next, mu, log_var); test stubs may return same shape
        if isinstance(out, tuple) and len(out) >= 1:
            z_next = out[0]
            if len(out) >= 3 and self.lambda_unc > 0.0:
                log_var_t = out[2]
                if isinstance(log_var_t, torch.Tensor):
                    log_var_np = log_var_t.detach().cpu().numpy().squeeze(0)
                else:
                    log_var_np = np.asarray(log_var_t).squeeze(0)
        else:
            z_next = out

        if isinstance(z_next, torch.Tensor):
            self._z = z_next.detach().cpu().numpy().squeeze(0).astype(np.float32)
        else:
            self._z = np.asarray(z_next, dtype=np.float32).squeeze(0)

        d = float(distance_to_reference(self._z, self.z_ref, metric=self.distance_metric))
        is_success = bool(d < self.epsilon)

        self._step_idx += 1
        self._used_genes.add(action)

        terminated = is_success  # reaching ε ends the episode successfully
        truncated = (self._step_idx >= self.max_steps) and not terminated

        reward = compute_reward(
            z_next=self._z,
            z_ref=self.z_ref,
            action=action,
            noop_idx=self.noop_idx,
            log_var=log_var_np,
            lambda_sparse=self.lambda_sparse,
            lambda_unc=self.lambda_unc,
            distance_scale=1.0,
            success_bonus=self.success_bonus,
            failure_penalty=self.failure_penalty,
            terminated=terminated,
            truncated=truncated,
            is_success=is_success,
            distance_metric=self.distance_metric,
        )

        self._current_mask = self._compute_mask()
        info = {
            "action_mask": self._current_mask.copy(),
            "success": is_success,
            "distance": d,
            "step": self._step_idx,
        }
        return self._z.copy(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the current action mask for MaskablePPO."""
        return self._current_mask.copy()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _compute_mask(self) -> np.ndarray:
        """Mask: True = available. NO-OP always True. Used genes always False if enabled."""
        mask = np.ones(self.n_genes + 1, dtype=bool)
        if self.repeat_mask_enabled:
            for g in self._used_genes:
                if 0 <= g < self.n_genes:
                    mask[g] = False
        # NO-OP (final index) is always available
        mask[self.noop_idx] = True
        return mask


# ---------------------------------------------------------------------- #
# Factory for SubprocVecEnv / DummyVecEnv
# ---------------------------------------------------------------------- #


def _load_dynamics_model(cfg: Any, *, allow_untrained: bool = False) -> Any:
    """Load PerturbationDynamicsModel from artifacts/dynamics/model.pt.

    Falls back to a fresh untrained model if ``allow_untrained=True`` and the checkpoint
    is missing — for smoke testing only. Logs a P0 warning in that case.
    """
    from src.models.dynamics import PerturbationDynamicsModel

    model_path = Path(cfg.paths.dynamics_model)
    config_path = Path(cfg.paths.dynamics_config) if hasattr(cfg.paths, "dynamics_config") else None

    n_genes = _peek_n_genes(cfg)
    n_latent = _peek_n_latent(cfg)

    if not model_path.exists():
        if not allow_untrained:
            raise FileNotFoundError(
                f"Dynamics model not found at {model_path}. Either run `make dynamics` first "
                f"or set `rl.train.smoke_with_untrained_dynamics=true` to bypass."
            )
        log.warning(
            "P0 — dynamics model.pt missing; instantiating an UNTRAINED model for smoke testing. "
            "All metrics from this run are MEANINGLESS."
        )
        model = PerturbationDynamicsModel(
            n_latent=n_latent,
            n_genes=n_genes,
            d_emb=int(cfg.dynamics.d_emb),
            n_hidden=int(cfg.dynamics.n_hidden),
            n_layers=int(cfg.dynamics.n_layers),
            dropout=float(cfg.dynamics.dropout),
        )
    else:
        # Source the architecture from the saved config.json (preferred), else cfg.dynamics.
        if config_path is not None and config_path.exists():
            with open(config_path) as f:
                dyn_cfg = json.load(f)
            log.info("Loading dynamics architecture from saved config.json")
        else:
            dyn_cfg = {
                "n_latent": n_latent,
                "n_genes":  n_genes,
                "d_emb":    int(cfg.dynamics.d_emb),
                "n_hidden": int(cfg.dynamics.n_hidden),
                "n_layers": int(cfg.dynamics.n_layers),
                "dropout":  float(cfg.dynamics.dropout),
            }
            log.info("Loading dynamics architecture from cfg.dynamics (no config.json found)")

        # Forward every architecture flag the model accepts; tolerate older configs missing keys.
        import inspect
        sig = inspect.signature(PerturbationDynamicsModel.__init__)
        accepted = set(sig.parameters.keys())

        kwargs: dict[str, Any] = {
            "n_latent": int(dyn_cfg["n_latent"]),
            "n_genes":  int(dyn_cfg["n_genes"]),
            "d_emb":    int(dyn_cfg.get("d_emb", 64)),
            "n_hidden": int(dyn_cfg.get("n_hidden", 256)),
            "n_layers": int(dyn_cfg.get("n_layers", 3)),
            "dropout":  float(dyn_cfg.get("dropout", 0.1)),
        }
        # Optional flags introduced after the initial design (e.g. state_linear_skip).
        for opt_key, opt_default in [
            ("activation",            "silu"),
            ("use_layernorm",         True),
            ("log_var_min",           -5.0),
            ("log_var_max",           3.0),
            ("log_var_init_bias",     -2.0),
            ("use_state_linear_skip", False),
            ("use_gene_delta_bias",   False),
        ]:
            if opt_key in accepted:
                kwargs[opt_key] = dyn_cfg.get(opt_key, opt_default)

        model = PerturbationDynamicsModel(**kwargs)
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        # Unwrap if saved as a checkpoint dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

    model.eval()
    model.to("cpu")  # RL rollouts are single-cell; CPU avoids MPS overhead
    return model


def _peek_n_genes(cfg: Any) -> int:
    """Read n_genes from gene_vocab.json."""
    with open(cfg.paths.vae_gene_vocab_json) as f:
        return int(json.load(f)["n_genes"])


def _peek_n_latent(cfg: Any) -> int:
    """Read n_latent from VAE config (n_latent = 32 by default)."""
    return int(cfg.vae.n_latent)


def _build_start_pool(
    cfg: Any,
    max_size: int = 50_000,
    min_distance: float | None = None,
) -> np.ndarray:
    """Build the non-control start-state pool from latents.h5ad.

    If ``min_distance`` is provided, the pool is restricted to cells where
    ``||z - z_ref|| > min_distance``. This prevents the trivial reward-hacking
    failure mode where cells already within ε_success of z_ref make the RL task
    artificially easy (any/no action → "success").

    Subsampled to ``max_size`` to keep subprocess memory reasonable.
    """
    import anndata as ad

    adata = ad.read_h5ad(str(cfg.paths.vae_latents_h5ad))
    Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    pert_idx = np.asarray(adata.obs["perturbation_idx"].values)
    pool = Z[pert_idx != 0]
    n_perturbed = len(pool)

    if min_distance is not None and min_distance > 0:
        z_ref = np.load(str(cfg.paths.vae_z_reference_centroid)).astype(np.float32)
        dists = np.linalg.norm(pool - z_ref, axis=1)
        pool = pool[dists > min_distance]
        log.info(
            "Start pool filtered to cells with ||z - z_ref|| > %.3f: %d → %d cells (%.1f%% kept). "
            "Distance stats: median=%.2f  p90=%.2f  max=%.2f",
            min_distance, n_perturbed, len(pool),
            100 * len(pool) / max(n_perturbed, 1),
            float(np.median(dists)), float(np.percentile(dists, 90)), float(dists.max()),
        )
        if len(pool) == 0:
            raise ValueError(
                f"Start pool is empty after filtering with min_distance={min_distance}. "
                f"Max distance in pool is {float(dists.max()):.2f}. Lower min_distance."
            )

    if len(pool) > max_size:
        rng = np.random.default_rng(42)
        pool = pool[rng.choice(len(pool), max_size, replace=False)]

    log.info("Start pool: %d non-control latents (from %d perturbed cells)", len(pool), n_perturbed)
    return pool


def resolve_epsilon(cfg: Any) -> tuple[float, str]:
    """Resolve the success-distance threshold ε and report its provenance.

    Precedence: ``cfg.rl.env.epsilon_override`` (when set, a float) wins over the value cached
    in ``artifacts/vae/epsilon_success.json``. The override path lets eval-time scripts switch
    percentiles (e.g. p90 ↔ p50) without mutating the canonical JSON.

    Returns
    -------
    (epsilon, source) :
        ``epsilon`` is the float to use as the success threshold; ``source`` is a short
        human-readable string suitable for logging and the per-run ``metadata.json``
        (``"override(<value>)"`` or ``"json(p<percentile>)"``).
    """
    override = cfg.rl.env.get("epsilon_override", None) if hasattr(cfg.rl.env, "get") else None
    if override is not None:
        eps = float(override)
        return eps, f"override({eps:.6g})"

    with open(cfg.paths.vae_epsilon_success_json) as f:
        blob = json.load(f)
    eps = float(blob["value"])
    pct = blob.get("percentile", "?")
    return eps, f"json(p{pct})"


def make_env_factory(cfg: Any) -> Callable[[], CellReprogrammingEnv]:
    """Return a zero-arg factory that constructs a fresh ``CellReprogrammingEnv``.

    Loads heavy artifacts ONCE (in the parent process) and closes over them so the returned
    factory is cheap to call. Suitable for both ``DummyVecEnv`` and ``SubprocVecEnv``.
    """
    # Load all artifacts once
    z_ref = np.load(str(cfg.paths.vae_z_reference_centroid))
    epsilon, epsilon_source = resolve_epsilon(cfg)
    log.info("Env epsilon = %.4f  (source: %s)", epsilon, epsilon_source)
    n_genes = _peek_n_genes(cfg)

    allow_untrained = bool(cfg.rl.train.get("smoke_with_untrained_dynamics", False))
    dynamics_model = _load_dynamics_model(cfg, allow_untrained=allow_untrained)

    # Filter start pool to cells that NEED steering (avoid reward-hacking from already-near-target starts).
    # "auto" → use epsilon (cells at-or-near goal are trivially successful).
    # A float → exact distance threshold.
    # "none" → no filter (legacy behavior; not recommended).
    min_start_dist_cfg = cfg.rl.env.get("min_start_distance", "auto")
    if isinstance(min_start_dist_cfg, str) and min_start_dist_cfg.lower() == "auto":
        min_start_distance: float | None = epsilon
    elif isinstance(min_start_dist_cfg, str) and min_start_dist_cfg.lower() == "none":
        min_start_distance = None
    else:
        min_start_distance = float(min_start_dist_cfg)
    start_pool = _build_start_pool(cfg, min_distance=min_start_distance)

    rl_cfg = cfg.rl
    env_cfg = rl_cfg.env
    reward_cfg = rl_cfg.reward

    log.info(
        "Env factory ready: n_genes=%d, ε=%.4f, max_steps=%d, λ_sparse=%.3f, distance=%s",
        n_genes, epsilon, int(env_cfg.max_steps), float(reward_cfg.lambda_sparse), str(env_cfg.distance_metric),
    )

    def factory() -> CellReprogrammingEnv:
        # Note: each subprocess gets a different seed via the Python RNG state
        return CellReprogrammingEnv(
            dynamics_model=dynamics_model,
            z_reference_centroid=z_ref,
            epsilon_success=epsilon,
            n_genes=n_genes,
            max_steps=int(env_cfg.max_steps),
            lambda_sparse=float(reward_cfg.lambda_sparse),
            lambda_unc=float(reward_cfg.lambda_unc),
            repeat_mask=bool(rl_cfg.action_space.repeat_mask),
            start_state_strategy=str(env_cfg.start_state),
            distance_metric=str(env_cfg.distance_metric),
            start_pool_latents=start_pool,
            success_bonus=float(reward_cfg.success_bonus),
            failure_penalty=float(reward_cfg.failure_penalty),
            seed=int(np.random.default_rng().integers(0, 2**31 - 1)),
        )

    return factory
