"""Preprocessing / pairing tests.

Stubs that pass once Agent A implements the corresponding functions. Until then they assert
that ``NotImplementedError`` is raised, so the test SUITE stays green during the scaffold phase.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_preprocess_module_imports(self) -> None:
        from src.data import preprocess  # noqa: F401

    def test_run_preprocessing_with_mock(self, tmp_path: Any) -> None:
        """Preprocessing runs end-to-end on a tiny synthetic AnnData."""
        import types
        import anndata
        import numpy as np

        from src.data.preprocess import run_preprocessing

        rng = np.random.default_rng(0)
        n_cells, n_genes = 120, 80
        # Raw integer counts in X — matching the scperturb Norman format
        X = rng.poisson(lam=3.0, size=(n_cells, n_genes)).astype("float32")

        perts = (
            ["control"] * 40
            + ["GENE_A"] * 20
            + ["GENE_B"] * 20
            + ["GENE_A_GENE_B"] * 20  # combo with "_" separator
            + ["GENE_C"] * 20
        )
        nperts = [0] * 40 + [1] * 20 + [1] * 20 + [2] * 20 + [1] * 20

        adata = anndata.AnnData(
            X=X,
            obs={"perturbation": perts, "nperts": nperts},
        )

        cfg = types.SimpleNamespace(
            data=types.SimpleNamespace(
                min_counts=1,        # keep all cells
                min_cells=2,         # keep genes in ≥2 cells
                n_hvg=20,            # small subset of HVGs
                hvg_flavor="seurat_v3",
                normalize_total=10_000,
                log_transform=True,
            ),
            paths=types.SimpleNamespace(
                norman_raw_h5ad=str(tmp_path / "raw.h5ad"),
                norman_processed_h5ad=str(tmp_path / "processed.h5ad"),
            ),
        )

        adata = run_preprocessing(cfg, adata=adata, save=True)

        # Shape: all cells survive (min_counts=1), exactly n_hvg=20 genes
        assert adata.shape[1] == 20, f"Expected 20 HVGs, got {adata.shape[1]}"

        # counts layer must be int32 integers
        counts = adata.layers["counts"]
        import scipy.sparse as sp
        if sp.issparse(counts):
            counts = counts.toarray()
        assert counts.dtype == np.dtype("int32")
        assert np.allclose(counts, np.round(counts))

        # X must be log1p-normalised (not raw integers)
        X_out = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        assert not np.allclose(X_out, np.round(X_out)), "X should be float, not raw counts"

        # Encoding: ctrl=0, singles=1..3, combo=4
        assert 0 in adata.obs["perturbation_idx"].values
        assert adata.uns["noop_idx"] == 3   # 3 single-gene perturbations
        assert adata.uns["ctrl_label"] == "control"

        # File exists and is non-empty
        assert (tmp_path / "processed.h5ad").stat().st_size > 1000

    @pytest.mark.slow
    def test_run_preprocessing_real(self, hydra_cfg: Any) -> None:
        """End-to-end real-data preprocess. Skipped in CI; run manually on real data."""
        pytest.skip("Needs Norman download — run scripts/train_vae.py instead.")


# ---------------------------------------------------------------------------
# Mock pairs schema (Contract 2)
# ---------------------------------------------------------------------------


class TestMockPairs:
    def test_mock_pairs_shape(self, mock_pairs_npz: Any) -> None:
        """The conftest fixture should write a Contract-2 compliant npz."""
        npz = np.load(mock_pairs_npz)
        assert set(npz.files) == {"z_ctrl", "gene_idx", "z_pert"}
        assert npz["z_ctrl"].shape == npz["z_pert"].shape
        assert npz["z_ctrl"].shape[1] == 32
        assert npz["gene_idx"].shape[0] == npz["z_ctrl"].shape[0]
        assert npz["z_ctrl"].dtype == np.float32
        assert npz["z_pert"].dtype == np.float32
        assert npz["gene_idx"].dtype == np.int32

    def test_generate_mock_pairs_is_stubbed_or_works(self, tmp_path: Any) -> None:
        """Agent A's Day-0 deliverable. If implemented, must produce valid npz.

        Until implemented, raises NotImplementedError — that's also fine.
        """
        from src.data.perturbation_pairs import generate_mock_pairs

        try:
            out = generate_mock_pairs(
                n=128, n_genes=5, n_latent=32, n_combo=16, seed=42, out_dir=tmp_path
            )
        except NotImplementedError:
            pytest.skip("generate_mock_pairs is Agent A's Day 0 deliverable.")
        # If we got here, Agent A has implemented it — assert schema.
        assert set(out.keys()) >= {"train", "val", "ood", "combo", "metadata"}
        train = np.load(out["train"])
        assert train["z_ctrl"].shape[1] == 32


# ---------------------------------------------------------------------------
# Stubs for the other pairing methods
# ---------------------------------------------------------------------------


class TestPairing:
    def test_pair_ot_is_stubbed(self) -> None:
        from src.data.perturbation_pairs import pair_ot

        z_ctrl = np.random.randn(10, 32).astype("float32")
        z_pert = np.random.randn(15, 32).astype("float32")
        with pytest.raises(NotImplementedError, match="Agent A"):
            pair_ot(z_ctrl, z_pert)

    def test_pair_random_is_stubbed(self) -> None:
        from src.data.perturbation_pairs import pair_random

        z_ctrl = np.random.randn(10, 32).astype("float32")
        z_pert = np.random.randn(15, 32).astype("float32")
        rng = np.random.default_rng(0)
        with pytest.raises(NotImplementedError, match="Agent A"):
            pair_random(z_ctrl, z_pert, rng)

    def test_pair_mean_delta_is_stubbed(self) -> None:
        from src.data.perturbation_pairs import pair_mean_delta

        z_ctrl = np.random.randn(10, 32).astype("float32")
        z_pert = np.random.randn(15, 32).astype("float32")
        with pytest.raises(NotImplementedError, match="Agent A"):
            pair_mean_delta(z_ctrl, z_pert)
