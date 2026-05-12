"""Shared pytest fixtures.

All fixtures here are designed to let tests run without the real Norman dataset and without a
trained scVI model. Tests that need real data must be marked ``@pytest.mark.slow`` and are
skipped in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Re-seed Python/NumPy/Torch before every test for determinism."""
    try:
        from src.utils.seeding import set_seed

        set_seed(42)
    except ImportError:
        # During very early scaffold runs, torch may not be installed; tolerate.
        np.random.seed(42)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anndata() -> Any:
    """Tiny synthetic AnnData mirroring the processed Norman schema.

    Shape ``(200 cells, 50 HVGs)``; 5 perturbation labels (1 ctrl + 4 genes).
    """
    ad = pytest.importorskip("anndata")
    n_cells, n_genes_hvg = 200, 50
    rng = np.random.default_rng(42)
    # Integer counts in layers["counts"], float log1p in X
    counts = rng.poisson(lam=2.0, size=(n_cells, n_genes_hvg)).astype("int32")
    X = np.log1p(counts.astype("float32"))
    perturbations = np.array(
        ["ctrl"] * 100 + ["GENE_A"] * 25 + ["GENE_B"] * 25 + ["GENE_C"] * 25 + ["GENE_D"] * 25
    )
    pert_idx = np.array(
        [0] * 100 + [1] * 25 + [2] * 25 + [3] * 25 + [4] * 25, dtype="int32"
    )
    adata = ad.AnnData(
        X=X,
        obs={"perturbation": perturbations, "perturbation_idx": pert_idx},
        var={"gene_symbol": [f"G{i}" for i in range(n_genes_hvg)],
             "highly_variable": np.ones(n_genes_hvg, dtype=bool)},
    )
    adata.layers["counts"] = counts
    adata.uns["perturbation_encoder"] = {
        "ctrl": 0, "GENE_A": 1, "GENE_B": 2, "GENE_C": 3, "GENE_D": 4,
    }
    adata.uns["noop_idx"] = 5
    return adata


@pytest.fixture
def mock_latents(mock_anndata: Any) -> Any:
    """Mock 32-dim scVI latent matrix consistent with ``mock_anndata``.

    Each perturbation has a slightly shifted centroid so that downstream "is the latent
    separating perturbations?" tests have signal.
    """
    rng = np.random.default_rng(0)
    n_latent = 32
    n_cells = mock_anndata.n_obs
    Z = rng.normal(size=(n_cells, n_latent)).astype("float32")
    # Add per-perturbation mean shift
    for p_idx in np.unique(mock_anndata.obs["perturbation_idx"]):
        mask = (mock_anndata.obs["perturbation_idx"] == p_idx).values
        shift = rng.normal(scale=0.5, size=(n_latent,)).astype("float32")
        Z[mask] += shift
    mock_anndata.obsm["X_scVI"] = Z
    return Z


@pytest.fixture
def mock_z_reference_centroid(mock_anndata: Any, mock_latents: Any) -> np.ndarray:
    """Mock reference centroid (mean of control latents)."""
    ctrl_mask = (mock_anndata.obs["perturbation_idx"] == 0).values
    return mock_latents[ctrl_mask].mean(axis=0).astype("float32")


@pytest.fixture
def mock_epsilon_success(mock_latents: Any, mock_z_reference_centroid: Any) -> float:
    """Mock ε_success: 90th percentile of control distances."""
    ctrl_dists = np.linalg.norm(
        mock_latents[: 100] - mock_z_reference_centroid, axis=1
    )
    return float(np.percentile(ctrl_dists, 90))


@pytest.fixture
def mock_pairs_npz(tmp_path: Path) -> Path:
    """Synthetic pairs file matching Contract 2 schema.

    Each perturbation has a constant Δz signature so a dynamics model has a learnable target.
    """
    rng = np.random.default_rng(0)
    n, n_latent, n_genes = 500, 32, 4
    z_ctrl = rng.normal(size=(n, n_latent)).astype("float32")
    gene_idx = rng.integers(1, n_genes + 1, size=n).astype("int32")
    per_gene_delta = {g: rng.normal(scale=0.3, size=n_latent).astype("float32")
                      for g in range(1, n_genes + 1)}
    z_pert = np.stack(
        [z_ctrl[i] + per_gene_delta[int(gene_idx[i])] for i in range(n)]
    ).astype("float32")
    path = tmp_path / "train_pairs.npz"
    np.savez(path, z_ctrl=z_ctrl, gene_idx=gene_idx, z_pert=z_pert)
    return path


@pytest.fixture
def mock_gene_vocab(tmp_path: Path) -> Path:
    """Synthetic gene_vocab.json matching Contract 1 schema."""
    vocab = {
        "genes": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
        "ctrl_idx": 0,
        "n_genes": 4,
        "noop_idx": 5,   # 0 = ctrl, 1..4 = genes, 5 = NO-OP at action-space level
    }
    path = tmp_path / "gene_vocab.json"
    path.write_text(json.dumps(vocab))
    return path


# ---------------------------------------------------------------------------
# Hydra config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def hydra_cfg() -> Any:
    """Composed Hydra ``default`` config, with paths rerouted to a temporary dir.

    Tests that need real Hydra config should depend on this fixture. Tests that only need a
    simple namespace can use a plain ``types.SimpleNamespace``.
    """
    hydra = pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="default", overrides=["device.force=cpu"])
    return cfg
