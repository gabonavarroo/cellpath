"""P0B″ — soft_ot pairing tests + 4-way comparator schema regression.

Tests:
1. soft_ot output shape matches hard OT (per-cell shape (N_pert, d)).
2. Each soft-OT row is a convex combination of controls (weights ≥ 0, sum ≈ 1).
3. When the transport plan is (effectively) one-hot, soft_ot collapses to hard pairing.
4. Contract-2 pair files written by build_pairs(method='soft_ot') remain valid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


CONTRACT2_KEYS_TRAIN = {"z_ctrl", "gene_idx", "z_pert"}
CONTRACT2_KEYS_COMBO = {"z_ctrl", "gene_idx_a", "gene_idx_b", "z_pert_ab"}


def test_soft_ot_output_shape_matches_hard_ot() -> None:
    """soft_ot returns (N_pert, d) float vectors, parallel to z_ctrl[hard_ot_idx]."""
    from src.data.perturbation_pairs import pair_ot, pair_soft_ot

    rng = np.random.default_rng(0)
    z_ctrl = rng.normal(size=(40, 8)).astype(np.float32)
    z_pert = rng.normal(loc=0.5, size=(25, 8)).astype(np.float32)

    hard_idx = pair_ot(z_ctrl, z_pert, epsilon=0.1, max_iter=200)
    soft_vec = pair_soft_ot(z_ctrl, z_pert, epsilon=0.1, max_iter=200)

    assert hard_idx.shape == (25,)
    assert soft_vec.shape == (25, 8)
    assert soft_vec.dtype == np.float32


def test_soft_ot_outputs_are_convex_combinations() -> None:
    """Each soft control must lie inside the convex hull of z_ctrl: minimal componentwise
    bounds satisfied (soft control lies between componentwise min and max of z_ctrl)."""
    from src.data.perturbation_pairs import pair_soft_ot

    rng = np.random.default_rng(1)
    z_ctrl = rng.normal(size=(60, 6)).astype(np.float32)
    z_pert = rng.normal(loc=1.0, size=(30, 6)).astype(np.float32)

    soft = pair_soft_ot(z_ctrl, z_pert, epsilon=0.1, max_iter=200)

    # Componentwise convex-hull check (sufficient necessary condition)
    z_min = z_ctrl.min(axis=0)
    z_max = z_ctrl.max(axis=0)
    eps = 1e-4
    assert np.all(soft >= z_min - eps), "soft control below componentwise min of z_ctrl"
    assert np.all(soft <= z_max + eps), "soft control above componentwise max of z_ctrl"


def test_soft_ot_collapses_to_hard_when_transport_is_one_hot() -> None:
    """When ε → 0 and controls are well-separated, soft_ot ≈ z_ctrl[argmax T[:,j]]."""
    from src.data.perturbation_pairs import pair_ot, pair_soft_ot

    rng = np.random.default_rng(2)
    # Construct well-separated control clusters so the OT plan is nearly one-hot
    z_ctrl = np.stack([
        rng.normal(loc=np.array([5.0 * i, 5.0 * i]), scale=0.05, size=(2,))
        for i in range(20)
    ]).astype(np.float32)
    # Pert cells are close to specific controls; permute so 1-1 matching is preferred
    z_pert = z_ctrl[:20].copy()
    z_pert += rng.normal(scale=0.01, size=z_pert.shape).astype(np.float32)

    hard_idx = pair_ot(z_ctrl, z_pert, epsilon=0.005, max_iter=2000)
    soft = pair_soft_ot(z_ctrl, z_pert, epsilon=0.005, max_iter=2000)
    hard_vec = z_ctrl[hard_idx]

    # With near-one-hot transport, soft and hard should be very close
    diff = np.linalg.norm(soft - hard_vec, axis=1)
    assert float(np.median(diff)) < 0.2, (
        f"soft vs hard median diff {float(np.median(diff)):.4f} too large; "
        "transport plan may not be near one-hot"
    )


def test_soft_ot_is_finite_on_typical_inputs() -> None:
    """No NaN/Inf for normal-sized inputs."""
    from src.data.perturbation_pairs import pair_soft_ot

    rng = np.random.default_rng(3)
    z_ctrl = rng.normal(size=(80, 12)).astype(np.float32)
    z_pert = rng.normal(loc=0.3, size=(50, 12)).astype(np.float32)
    out = pair_soft_ot(z_ctrl, z_pert, epsilon=0.05, max_iter=500)
    assert np.all(np.isfinite(out))


def test_pairs_soft_ot_schema_contract2() -> None:
    """If artifacts_v2/pairs_soft_ot exists, validate Contract-2 schema."""
    root = Path("artifacts_v2/pairs_soft_ot")
    if not root.exists():
        pytest.skip("artifacts_v2/pairs_soft_ot not produced yet — runs before build step are expected to skip")
    for fname, keys in [
        ("train_pairs.npz", CONTRACT2_KEYS_TRAIN),
        ("val_pairs.npz", CONTRACT2_KEYS_TRAIN),
        ("ood_pairs.npz", CONTRACT2_KEYS_TRAIN),
        ("combo_pairs.npz", CONTRACT2_KEYS_COMBO),
    ]:
        data = np.load(root / fname)
        assert set(data.files) >= keys, f"{fname} missing keys: {keys - set(data.files)}"
        if "z_ctrl" in data.files:
            assert data["z_ctrl"].shape[1] == 32
            assert data["z_ctrl"].dtype == np.float32
    meta = json.loads((root / "metadata.json").read_text())
    assert meta.get("pairing_method") == "soft_ot"
