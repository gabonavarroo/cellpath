"""P0D Track B — distance-bin curriculum callback for PPO training.

A Stable-Baselines3 callback that gradually shifts the env's ``min_start_distance``
from an easy band (e.g. 4.0) to a hard band (e.g. 10.0) across a fraction of the
training run. The callback owns a *raw* perturbed-cell pool loaded from
``latents.h5ad`` once at training start, and mutates every env's ``_start_pool``
via :meth:`CellReprogrammingEnv.set_start_pool` whenever the schedule says the
filter has moved by more than a small threshold.

Sacred rules respected:
- No edits to V1 artifacts (the callback reads only).
- No reward shape changes here (the reward layer is in ``src/rl/reward.py``).
- No path hardcoding; the callback takes ``vae_dir`` and ``z_ref_path`` parameters
  resolved by the trainer from Hydra config.

This is the only mechanism the trainer uses to evolve start distributions during
training; PPO/SB3 are otherwise unaware of curriculum existence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:                              # pragma: no cover - SB3 missing in some envs
    class BaseCallback:                        # type: ignore[no-redef]
        """Stub used when SB3 is unavailable (tests import BaseCallback at import time)."""

        def __init__(self, verbose: int = 0) -> None:
            self.verbose = verbose
            self.num_timesteps = 0
            self.training_env = None

        def _on_step(self) -> bool:
            return True


class DistanceCurriculumCallback(BaseCallback):
    """Linear (or step) schedule on ``min_start_distance`` over training.

    The callback maintains a cached "raw" start pool (all perturbed cells, no
    distance filter) and recomputes the filtered pool whenever the schedule says
    ``min_start_distance`` has moved by more than ``apply_threshold``. Updates
    are pushed to every env in the VecEnv via ``training_env.env_method``.

    Parameters
    ----------
    vae_latents_h5ad
        Path to the latents h5ad file. ``adata.obsm["X_scVI"]`` is read; the
        perturbed-cell subset is the start pool.
    z_ref_path
        Path to ``z_reference_centroid.npy``.
    start_d, end_d
        Start and end of the linear schedule. ``min_start_distance`` is
        interpolated linearly from ``start_d`` to ``end_d`` over the first
        ``end_fraction`` of ``total_timesteps``; after that, it stays at ``end_d``.
    total_timesteps
        Total PPO timesteps (used to compute schedule progress).
    end_fraction
        Fraction of total_timesteps at which the schedule reaches ``end_d``.
        Default 0.7 (last 30 % of training is on the hard distribution).
    apply_threshold
        Minimum change in ``min_start_distance`` (units of latent L2) before the
        callback bothers re-filtering the pool. Default 0.25.
    check_every
        Minimum step gap between checks. Default 10 000.
    """

    def __init__(
        self,
        vae_latents_h5ad: str | Path,
        z_ref_path: str | Path,
        start_d: float,
        end_d: float,
        total_timesteps: int,
        end_fraction: float = 0.7,
        apply_threshold: float = 0.25,
        check_every: int = 10_000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.vae_latents_h5ad = str(vae_latents_h5ad)
        self.z_ref_path = str(z_ref_path)
        self.start_d = float(start_d)
        self.end_d = float(end_d)
        self.total_timesteps = int(total_timesteps)
        self.end_fraction = float(end_fraction)
        self.apply_threshold = float(apply_threshold)
        self.check_every = int(check_every)

        self._raw_pool: np.ndarray | None = None
        self._z_ref: np.ndarray | None = None
        self._last_d: float | None = None
        self._last_update_step: int = -10**9
        self.events: list[tuple[int, float, int]] = []  # (timestep, min_d, pool_size) audit log

    # ----- SB3 hook --------------------------------------------------------

    def _on_training_start(self) -> None:
        import anndata as ad
        adata = ad.read_h5ad(self.vae_latents_h5ad)
        Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
        pert_idx = np.asarray(adata.obs["perturbation_idx"].values)
        self._raw_pool = Z[pert_idx != 0]
        self._z_ref = np.load(self.z_ref_path).astype(np.float32)
        self._apply(self.start_d)
        self._last_d = self.start_d
        if self.verbose:
            print(f"[DistanceCurriculumCallback] training_start: "
                  f"raw_pool={len(self._raw_pool)}, start_d={self.start_d:.2f}, "
                  f"end_d={self.end_d:.2f}, end_fraction={self.end_fraction:.2f}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update_step < self.check_every:
            return True
        self._last_update_step = self.num_timesteps
        d = self._scheduled_distance(self.num_timesteps)
        if self._last_d is None or abs(d - self._last_d) >= self.apply_threshold:
            self._apply(d)
            self._last_d = d
        return True

    # ----- internals -------------------------------------------------------

    def _scheduled_distance(self, t: int) -> float:
        """Linear schedule: start_d → end_d over first end_fraction × total_timesteps."""
        if self.total_timesteps <= 0:
            return self.end_d
        denom = max(1, int(self.total_timesteps * self.end_fraction))
        frac = min(1.0, max(0.0, float(t) / float(denom)))
        return self.start_d + frac * (self.end_d - self.start_d)

    def _apply(self, d: float) -> None:
        """Filter the cached raw pool and push to every env."""
        if self._raw_pool is None or self._z_ref is None:
            return
        dists = np.linalg.norm(self._raw_pool - self._z_ref, axis=1)
        pool = self._raw_pool[dists > d]
        if len(pool) == 0:
            if self.verbose:
                print(f"[DistanceCurriculumCallback] skip apply: filter d={d:.2f} "
                      f"leaves empty pool (raw max dist={float(dists.max()):.2f})")
            return
        # In tests we may not have a real VecEnv; gracefully fall back to logging only.
        env: Any = None
        try:
            env = getattr(self, "training_env", None)
        except (AssertionError, AttributeError):
            env = None
        if env is not None and hasattr(env, "env_method"):
            env.env_method("set_start_pool", pool)
        self.events.append((int(self.num_timesteps), float(d), int(len(pool))))
        if self.verbose:
            print(f"[DistanceCurriculumCallback] t={self.num_timesteps}: "
                  f"min_start_distance={d:.2f}, pool_size={len(pool)}")
