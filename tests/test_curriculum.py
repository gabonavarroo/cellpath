"""P0D Track B — distance-curriculum callback tests.

Verifies the linear schedule formula, the apply-threshold gating, the empty-pool
guard, and that ``training_env.env_method('set_start_pool', ...)`` is the only
side-effect on env state. The callback is exercised without a real PPO/VecEnv to
keep the tests fast and SB3-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


class _FakeVecEnv:
    """Minimal stand-in for SB3's VecEnv. Records env_method calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any]]] = []

    def env_method(self, name: str, *args: Any, **kwargs: Any) -> list[Any]:
        self.calls.append((name, list(args)))
        return [None]


class _FakeModel:
    """Stand-in for SB3 model so ``callback.training_env`` resolves correctly.

    SB3's ``BaseCallback.training_env`` is a property that delegates to ``self.model.get_env()``,
    so tests cannot assign to it directly; they assign a fake model with a ``get_env`` method
    instead.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def get_env(self) -> Any:
        return self._env


@pytest.fixture()
def fake_assets(tmp_path: Path) -> dict[str, Path]:
    """Build a tiny latents.h5ad + z_ref.npy on disk for the callback to load."""
    anndata = pytest.importorskip("anndata")
    n = 200
    n_latent = 32
    rng = np.random.default_rng(0)
    z_ref = np.zeros(n_latent, dtype=np.float32)
    # Mix of nearby + far cells, plus controls (perturbation_idx == 0) that must be excluded.
    near = rng.standard_normal((n // 2, n_latent)).astype(np.float32) * 2.0     # ‖z‖ ≈ small
    far = rng.standard_normal((n // 2, n_latent)).astype(np.float32) * 5.0      # ‖z‖ ≈ larger
    Z = np.concatenate([near, far], axis=0)
    pert_idx = np.array([0] * 50 + [1] * 150, dtype=np.int64)   # first 50 are control
    obs = {"perturbation_idx": pert_idx}
    import pandas as pd
    adata = anndata.AnnData(X=np.zeros((n, 4), dtype=np.float32),
                             obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(n)]))
    adata.obsm["X_scVI"] = Z

    h5_path  = tmp_path / "latents.h5ad"
    ref_path = tmp_path / "z_reference_centroid.npy"
    adata.write_h5ad(h5_path)
    np.save(ref_path, z_ref)
    return {"h5ad": h5_path, "ref": ref_path}


class TestDistanceCurriculumSchedule:
    def test_linear_schedule_endpoints(self, fake_assets: dict[str, Path]) -> None:
        from src.rl.curriculum import DistanceCurriculumCallback
        cb = DistanceCurriculumCallback(
            vae_latents_h5ad=fake_assets["h5ad"], z_ref_path=fake_assets["ref"],
            start_d=4.0, end_d=10.0, total_timesteps=1000, end_fraction=0.5,
        )
        # At t=0, scheduled d == start_d. At t = end_fraction × total, == end_d.
        assert cb._scheduled_distance(0) == pytest.approx(4.0, abs=1e-6)
        assert cb._scheduled_distance(500) == pytest.approx(10.0, abs=1e-6)
        # After that, plateau at end_d.
        assert cb._scheduled_distance(1000) == pytest.approx(10.0, abs=1e-6)

    def test_linear_schedule_midpoint(self, fake_assets: dict[str, Path]) -> None:
        from src.rl.curriculum import DistanceCurriculumCallback
        cb = DistanceCurriculumCallback(
            vae_latents_h5ad=fake_assets["h5ad"], z_ref_path=fake_assets["ref"],
            start_d=0.0, end_d=10.0, total_timesteps=1000, end_fraction=1.0,
        )
        # Linear from 0→10 over 1000 steps → at 500, d=5.
        assert cb._scheduled_distance(500) == pytest.approx(5.0, abs=1e-6)


class TestDistanceCurriculumApply:
    def test_training_start_loads_raw_pool_and_applies_filter(
        self, fake_assets: dict[str, Path]
    ) -> None:
        from src.rl.curriculum import DistanceCurriculumCallback
        cb = DistanceCurriculumCallback(
            vae_latents_h5ad=fake_assets["h5ad"], z_ref_path=fake_assets["ref"],
            start_d=2.0, end_d=10.0, total_timesteps=1000, verbose=0,
        )
        fake_env = _FakeVecEnv()
        cb.model = _FakeModel(fake_env)
        cb._on_training_start()
        # Raw pool must exclude perturbation_idx=0 cells (150, not 200).
        assert cb._raw_pool is not None and len(cb._raw_pool) == 150
        # Exactly one env_method call ("set_start_pool") was issued with a non-empty pool.
        assert len(fake_env.calls) == 1
        name, args = fake_env.calls[0]
        assert name == "set_start_pool"
        assert isinstance(args[0], np.ndarray) and len(args[0]) > 0
        # All cells in the filtered pool are at distance > start_d.
        dists = np.linalg.norm(args[0] - cb._z_ref, axis=1)
        assert float(dists.min()) > cb.start_d - 1e-6

    def test_apply_threshold_skips_small_changes(
        self, fake_assets: dict[str, Path]
    ) -> None:
        from src.rl.curriculum import DistanceCurriculumCallback
        cb = DistanceCurriculumCallback(
            vae_latents_h5ad=fake_assets["h5ad"], z_ref_path=fake_assets["ref"],
            start_d=4.0, end_d=10.0, total_timesteps=10_000, end_fraction=1.0,
            apply_threshold=1.0,  # large threshold → only sizeable updates apply
            check_every=100,      # check often
            verbose=0,
        )
        fake_env = _FakeVecEnv()
        cb.model = _FakeModel(fake_env)
        cb._on_training_start()
        # Advance a tiny bit: d ≈ 4.06 — under apply_threshold, no new call.
        n_calls_before = len(fake_env.calls)
        cb.num_timesteps = 100
        cb._on_step()
        assert len(fake_env.calls) == n_calls_before  # no update
        # Advance enough so d ≈ 5.5 — should now update.
        cb.num_timesteps = 2500
        cb._on_step()
        assert len(fake_env.calls) == n_calls_before + 1

    def test_empty_pool_is_skipped_safely(self, fake_assets: dict[str, Path]) -> None:
        """If the schedule yields a filter so strict the pool is empty, skip silently."""
        from src.rl.curriculum import DistanceCurriculumCallback
        cb = DistanceCurriculumCallback(
            vae_latents_h5ad=fake_assets["h5ad"], z_ref_path=fake_assets["ref"],
            start_d=0.0, end_d=10_000.0, total_timesteps=1000, end_fraction=0.001,
            check_every=10, apply_threshold=0.01, verbose=0,
        )
        fake_env = _FakeVecEnv()
        cb.model = _FakeModel(fake_env)
        cb._on_training_start()
        # After many "training" steps, the schedule has long since pushed d > all distances.
        n_calls_before = len(fake_env.calls)
        cb.num_timesteps = 5000
        # Should not raise; should not append an env_method call.
        cb._on_step()
        # No new applies were issued (raw pool's max distance is finite).
        for _, args in fake_env.calls[n_calls_before:]:
            pool = args[0]
            assert len(pool) > 0


class TestDistanceCurriculumWithoutVecEnv:
    def test_gracefully_handles_missing_training_env(
        self, fake_assets: dict[str, Path]
    ) -> None:
        """If training_env is None (unit-test smoke), _apply must not raise."""
        from src.rl.curriculum import DistanceCurriculumCallback
        cb = DistanceCurriculumCallback(
            vae_latents_h5ad=fake_assets["h5ad"], z_ref_path=fake_assets["ref"],
            start_d=4.0, end_d=10.0, total_timesteps=1000, verbose=0,
        )
        # Leave cb.model unset (SB3's training_env property will return None → AttributeError;
        # the callback's _apply uses getattr(..., None) to handle this case.)
        cb.model = _FakeModel(None)
        # Should not raise.
        cb._on_training_start()
        cb.num_timesteps = 500
        cb._on_step()
        # Events accumulate even without a real VecEnv.
        assert len(cb.events) >= 1
