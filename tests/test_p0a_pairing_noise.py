from __future__ import annotations

import numpy as np


def test_pairing_noise_ratio_is_zero_for_identical_deltas_and_positive_for_noisy_gene() -> None:
    from scripts.diagnose_pairing_noise import compute_pairing_noise

    z_ctrl = np.zeros((6, 2), dtype=np.float32)
    gene_idx = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    deltas = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    rows, summary = compute_pairing_noise({"z_ctrl": z_ctrl, "gene_idx": gene_idx, "z_pert": z_ctrl + deltas})

    by_gene = {row["gene_idx"]: row for row in rows}
    assert by_gene[1]["noise_ratio"] == 0.0
    assert by_gene[2]["noise_ratio"] > 0.0
    assert summary["n_genes"] == 2
    assert "median_noise_ratio" in summary
