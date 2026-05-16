from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def test_compute_epsilon_percentile_from_control_latents(tmp_path: Path) -> None:
    from src.utils.epsilon_percentile import compute_epsilon_percentile

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    z_ref = np.array([0.0, 0.0], dtype=np.float32)
    np.save(vae_dir / "z_reference_centroid.npy", z_ref)

    adata = ad.AnnData(
        X=np.zeros((4, 2), dtype=np.float32),
        obs=pd.DataFrame({"perturbation_idx": [0, 0, 1, 1]}),
    )
    adata.obsm["X_scVI"] = np.array(
        [[1.0, 0.0], [3.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
        dtype=np.float32,
    )
    adata.write_h5ad(vae_dir / "latents.h5ad")

    out = compute_epsilon_percentile(vae_dir, p=50)
    assert out["value"] == pytest.approx(2.0)
    assert out["n_ctrl_cells"] == 2
    assert out["percentile"] == 50.0
