from __future__ import annotations

import numpy as np


def test_gate_breakdown_tables_report_dim_and_gene_margins() -> None:
    from src.analysis.gate_breakdown import build_gate_breakdown_tables

    z_train = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    gene_train = np.array([1, 1, 2, 2], dtype=np.int32)
    z_pert_train = z_train + np.array(
        [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]],
        dtype=np.float32,
    )

    z_val = np.array(
        [[0.2, 0.0], [1.2, 0.0], [0.0, 1.2], [1.0, 1.2]],
        dtype=np.float32,
    )
    gene_val = np.array([1, 1, 2, 2], dtype=np.int32)
    true_delta_val = np.array(
        [[1.0, 0.0], [1.0, 0.2], [0.0, 1.0], [0.2, 1.0]],
        dtype=np.float32,
    )
    z_pert_val = z_val + true_delta_val
    z_pred_val = z_val + true_delta_val

    per_dim, per_gene = build_gate_breakdown_tables(
        train={"z_ctrl": z_train, "gene_idx": gene_train, "z_pert": z_pert_train},
        splits={
            "val": {
                "z_ctrl": z_val,
                "gene_idx": gene_val,
                "z_pert": z_pert_val,
                "z_pred": z_pred_val,
            }
        },
        per_gene_min_for_pearson=2,
    )

    assert set(per_dim["split"]) == {"val"}
    assert set(per_dim["dim"]) == {0, 1}
    assert "mlp_minus_ridge_pearson" in per_dim.columns
    assert np.isfinite(per_dim["mlp_minus_ridge_pearson"]).all()

    assert set(per_gene["gene_idx"]) == {1, 2}
    assert "mlp_minus_ridge_r2" in per_gene.columns
    assert "mlp_minus_ridge_pearson" in per_gene.columns
    assert np.isfinite(per_gene["mlp_minus_ridge_r2"]).all()
