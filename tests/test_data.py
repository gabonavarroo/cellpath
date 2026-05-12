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

    def test_run_preprocessing_is_stubbed(self, mock_anndata: Any) -> None:
        """Until Agent A implements it, the stub raises with a clear hint."""
        from src.data.preprocess import run_preprocessing

        with pytest.raises(NotImplementedError, match="Agent A"):
            run_preprocessing(cfg=None, adata=mock_anndata, save=False)

    @pytest.mark.slow
    def test_run_preprocessing_real(self, hydra_cfg: Any) -> None:
        """End-to-end real-data preprocess. Skipped in CI."""
        pytest.skip("Implemented post-Phase-1; needs Norman download.")


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
