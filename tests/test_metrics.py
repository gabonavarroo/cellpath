"""Tests for Phase 2 dynamics metrics in src/analysis/metrics.py.

Agent B Phase 2 — covers predictive_r2, pearson_r_per_dim,
uncertainty_calibration_spearman, and dynamics_validation_gate.

Agent A's silhouette_perturbation and ari_on_perturbation_clusters are
tested via the broader test_data.py / test_integration.py; they are not
touched here.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic fixture helpers (module-level, no conftest changes needed)
# ---------------------------------------------------------------------------


def _make_synthetic_pairs(
    n: int = 200,
    n_latent: int = 8,
    n_genes: int = 3,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_ctrl, gene_idx, z_pert) with learnable per-gene Δ structure."""
    rng = np.random.default_rng(seed)
    z_ctrl = rng.normal(size=(n, n_latent)).astype("float32")
    gene_idx = rng.integers(1, n_genes + 1, size=n).astype("int32")
    per_gene_delta = {
        g: rng.normal(scale=0.3, size=n_latent).astype("float32")
        for g in range(1, n_genes + 1)
    }
    deltas = np.stack([per_gene_delta[int(g)] for g in gene_idx]).astype("float32")
    z_pert = (
        z_ctrl + deltas + rng.normal(scale=noise, size=(n, n_latent))
    ).astype("float32")
    return z_ctrl, gene_idx, z_pert


def _default_cfg_gate() -> dict:
    return {
        "margin_vs_noop_r2": 0.10,
        "margin_vs_global_mean_r2": 0.05,
        "margin_vs_per_gene_mean_r2": 0.0,
        "margin_vs_linear_ridge_pearson": 0.03,
        "margin_vs_knn_r2": 0.03,
        "min_uncertainty_calibration_spearman": 0.20,
    }


# ---------------------------------------------------------------------------
# TestPredictiveR2
# ---------------------------------------------------------------------------


class TestPredictiveR2:
    def test_perfect_prediction_is_one(self) -> None:
        from src.analysis.metrics import predictive_r2

        rng = np.random.default_rng(0)
        y = rng.normal(size=(100, 8)).astype("float32")
        assert predictive_r2(y, y) == pytest.approx(1.0, abs=1e-6)

    def test_mean_prediction_is_zero(self) -> None:
        from src.analysis.metrics import predictive_r2

        rng = np.random.default_rng(1)
        y_true = rng.normal(size=(100, 8)).astype("float32")
        y_mean = np.full_like(y_true, y_true.mean())
        assert predictive_r2(y_true, y_mean) == pytest.approx(0.0, abs=1e-5)

    def test_bad_prediction_is_negative(self) -> None:
        from src.analysis.metrics import predictive_r2

        rng = np.random.default_rng(2)
        y_true = rng.normal(size=(100, 8)).astype("float32")
        y_noise = rng.normal(scale=5.0, size=y_true.shape).astype("float32")
        assert predictive_r2(y_true, y_noise) < 0.0

    def test_constant_y_true_perfect_pred_returns_one(self) -> None:
        from src.analysis.metrics import predictive_r2

        # sklearn-style: constant y_true, y_pred matches exactly → 1.0
        y = np.full((50, 8), 3.14, dtype="float32")
        assert predictive_r2(y, y) == pytest.approx(1.0, abs=1e-6)

    def test_constant_y_true_wrong_pred_returns_zero(self) -> None:
        from src.analysis.metrics import predictive_r2

        # sklearn-style: constant y_true, y_pred differs → 0.0 (not NaN)
        y_true = np.full((50, 8), 3.14, dtype="float32")
        y_pred = np.full((50, 8), 2.71, dtype="float32")
        result = predictive_r2(y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-6)
        assert np.isfinite(result)

    def test_returns_python_float(self) -> None:
        from src.analysis.metrics import predictive_r2

        y = np.ones((10, 4), dtype="float32")
        assert isinstance(predictive_r2(y, y), float)


# ---------------------------------------------------------------------------
# TestPearsonRPerDim
# ---------------------------------------------------------------------------


class TestPearsonRPerDim:
    def test_output_shape(self) -> None:
        from src.analysis.metrics import pearson_r_per_dim

        rng = np.random.default_rng(0)
        y = rng.normal(size=(100, 8)).astype("float32")
        assert pearson_r_per_dim(y, y).shape == (8,)

    def test_perfect_per_dim(self) -> None:
        from src.analysis.metrics import pearson_r_per_dim

        rng = np.random.default_rng(1)
        y = rng.normal(size=(100, 8)).astype("float32")
        r = pearson_r_per_dim(y, y)
        assert np.allclose(r, 1.0, atol=1e-5)

    def test_constant_column_in_y_true_returns_zero(self) -> None:
        from src.analysis.metrics import pearson_r_per_dim

        rng = np.random.default_rng(2)
        y_true = rng.normal(size=(100, 8)).astype("float32")
        y_pred = y_true.copy()
        y_true[:, 3] = 5.0  # make col 3 constant → Pearson undefined → 0.0
        r = pearson_r_per_dim(y_true, y_pred)
        assert r[3] == pytest.approx(0.0, abs=1e-6)

    def test_constant_column_in_y_pred_returns_zero(self) -> None:
        from src.analysis.metrics import pearson_r_per_dim

        rng = np.random.default_rng(3)
        y_true = rng.normal(size=(100, 8)).astype("float32")
        y_pred = y_true.copy()
        y_pred[:, 5] = 0.0  # constant column in y_pred → denominator zero → 0.0
        r = pearson_r_per_dim(y_true, y_pred)
        assert np.isfinite(r[5])
        assert r[5] == pytest.approx(0.0, abs=1e-6)

    def test_returns_ndarray(self) -> None:
        from src.analysis.metrics import pearson_r_per_dim

        y = np.ones((10, 4), dtype="float32")
        assert isinstance(pearson_r_per_dim(y, y), np.ndarray)

    def test_all_finite(self) -> None:
        from src.analysis.metrics import pearson_r_per_dim

        rng = np.random.default_rng(4)
        y_true = rng.normal(size=(50, 16)).astype("float32")
        y_pred = rng.normal(size=(50, 16)).astype("float32")
        r = pearson_r_per_dim(y_true, y_pred)
        assert np.all(np.isfinite(r))


# ---------------------------------------------------------------------------
# TestUncertaintyCalibrationSpearman
# ---------------------------------------------------------------------------


class TestUncertaintyCalibrationSpearman:
    def test_monotonic_pairs_returns_one(self) -> None:
        from src.analysis.metrics import uncertainty_calibration_spearman

        # exp(log_var) == sq_err when log_var = log(sq_err) → perfect rank match
        rng = np.random.default_rng(0)
        sq_err = np.sort(rng.uniform(0.01, 1.0, size=(200, 8))).astype("float32")
        log_var = np.log(sq_err).astype("float32")
        rho = uncertainty_calibration_spearman(log_var, sq_err)
        assert rho == pytest.approx(1.0, abs=1e-3)

    def test_anti_monotonic_returns_negative(self) -> None:
        from src.analysis.metrics import uncertainty_calibration_spearman

        # linspace over the full (n*d) elements is monotonic in row-major order,
        # so its reverse is a clean rank-inversion after flattening.
        n, d = 200, 8
        sq_err  = np.linspace(0.01, 1.0, n * d).reshape(n, d).astype("float32")
        log_var = np.log(np.linspace(1.0, 0.01, n * d).reshape(n, d)).astype("float32")
        rho = uncertainty_calibration_spearman(log_var, sq_err)
        assert rho < 0.0

    def test_constant_log_var_returns_zero(self) -> None:
        from src.analysis.metrics import uncertainty_calibration_spearman

        log_var = np.full((50, 8), -2.0, dtype="float32")
        sq_err  = np.random.default_rng(2).uniform(0.0, 1.0, size=(50, 8)).astype("float32")
        rho = uncertainty_calibration_spearman(log_var, sq_err)
        assert rho == pytest.approx(0.0, abs=1e-6)

    def test_returns_python_float(self) -> None:
        from src.analysis.metrics import uncertainty_calibration_spearman

        lv = np.zeros((10, 4), dtype="float32")
        se = np.ones((10, 4), dtype="float32")
        assert isinstance(uncertainty_calibration_spearman(lv, se), float)


# ---------------------------------------------------------------------------
# TestDynamicsValidationGate
# ---------------------------------------------------------------------------


class TestDynamicsValidationGate:
    def test_raises_value_error_when_baselines_train_data_none(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs()
        log_var = np.zeros_like(z_pert)
        with pytest.raises(ValueError, match="baselines_train_data"):
            dynamics_validation_gate(
                z_ctrl, gene_idx, z_pert, z_pert, log_var,
                _default_cfg_gate(),
                baselines_train_data=None,
            )

    def test_raises_value_error_when_train_key_missing(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs()
        log_var = np.zeros_like(z_pert)
        with pytest.raises(ValueError, match="z_pert"):
            dynamics_validation_gate(
                z_ctrl, gene_idx, z_pert, z_pert, log_var,
                _default_cfg_gate(),
                baselines_train_data={"z_ctrl": z_ctrl, "gene_idx": gene_idx},
            )

    def test_dict_style_cfg_access(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs(seed=0)
        z_tr, g_tr, zp_tr = _make_synthetic_pairs(n=400, seed=2)
        log_var = np.zeros_like(z_pert)
        out = dynamics_validation_gate(
            z_ctrl, gene_idx, z_pert, z_pert, log_var,
            _default_cfg_gate(),  # plain dict
            baselines_train_data={"z_ctrl": z_tr, "gene_idx": g_tr, "z_pert": zp_tr},
        )
        assert isinstance(out, dict)

    def test_attribute_style_cfg_access(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs(seed=0)
        z_tr, g_tr, zp_tr = _make_synthetic_pairs(n=400, seed=2)
        log_var = np.zeros_like(z_pert)
        cfg_ns = types.SimpleNamespace(**_default_cfg_gate())  # attribute-style access
        out = dynamics_validation_gate(
            z_ctrl, gene_idx, z_pert, z_pert, log_var,
            cfg_ns,
            baselines_train_data={"z_ctrl": z_tr, "gene_idx": g_tr, "z_pert": zp_tr},
        )
        assert isinstance(out, dict)

    def test_returns_expected_top_level_keys(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs(seed=0)
        z_tr, g_tr, zp_tr = _make_synthetic_pairs(n=400, seed=2)
        log_var = np.zeros_like(z_pert)
        out = dynamics_validation_gate(
            z_ctrl, gene_idx, z_pert, z_pert, log_var,
            _default_cfg_gate(),
            baselines_train_data={"z_ctrl": z_tr, "gene_idx": g_tr, "z_pert": zp_tr},
        )
        assert set(out.keys()) == {"passed", "primary", "uncertainty_calibration", "margins_used"}

    def test_primary_has_expected_keys(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs(seed=0)
        z_tr, g_tr, zp_tr = _make_synthetic_pairs(n=400, seed=2)
        log_var = np.zeros_like(z_pert)
        out = dynamics_validation_gate(
            z_ctrl, gene_idx, z_pert, z_pert, log_var,
            _default_cfg_gate(),
            baselines_train_data={"z_ctrl": z_tr, "gene_idx": g_tr, "z_pert": zp_tr},
        )
        assert set(out["primary"].keys()) == {"r2", "pearson_r", "baselines", "margin_checks"}
        expected_baselines = {
            "no_op", "global_mean_delta", "per_gene_mean_delta",
            "linear_ridge", "nearest_neighbor",
        }
        assert set(out["primary"]["baselines"].keys()) == expected_baselines

    def test_result_is_json_serializable(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs(seed=0)
        z_tr, g_tr, zp_tr = _make_synthetic_pairs(n=400, seed=2)
        log_var = np.zeros_like(z_pert)
        out = dynamics_validation_gate(
            z_ctrl, gene_idx, z_pert, z_pert, log_var,
            _default_cfg_gate(),
            baselines_train_data={"z_ctrl": z_tr, "gene_idx": g_tr, "z_pert": zp_tr},
        )
        json.dumps(out)  # must not raise TypeError for numpy scalars / numpy bools

    def test_very_good_mlp_passes_gate(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        # Larger train set for stable baselines
        z_tr, g_tr, zp_tr = _make_synthetic_pairs(
            n=400, n_latent=8, n_genes=3, noise=0.05, seed=2
        )
        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs(
            n=200, n_latent=8, n_genes=3, noise=0.05, seed=0
        )
        delta_true = z_pert - z_ctrl

        # Very-good-but-not-perfect MLP: small structured error of varying magnitudes
        rng = np.random.default_rng(1)
        err = (0.02 * rng.normal(size=delta_true.shape)).astype("float32")
        delta_pred_mlp = delta_true + err
        z_pert_pred = z_ctrl + delta_pred_mlp

        # log_var rank-matches err² → Spearman ≈ 1.0 (deterministic, no float-noise fight)
        squared_error = err ** 2
        log_var = np.log(squared_error + 1e-8).astype("float32")

        out = dynamics_validation_gate(
            z_ctrl, gene_idx, z_pert, z_pert_pred, log_var,
            _default_cfg_gate(),
            baselines_train_data={"z_ctrl": z_tr, "gene_idx": g_tr, "z_pert": zp_tr},
        )

        assert out["passed"] is True, (
            f"Gate failed unexpectedly.\n"
            f"margin_checks: {out['primary']['margin_checks']}\n"
            f"calibration: {out['uncertainty_calibration']}"
        )
        assert out["uncertainty_calibration"]["pass"] is True
        for key, check in out["primary"]["margin_checks"].items():
            assert check["pass"] is True, f"Margin check '{key}' failed: {check}"

    def test_noop_mlp_fails_noop_margin(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs(seed=0)
        z_tr, g_tr, zp_tr = _make_synthetic_pairs(n=400, seed=2)
        log_var = np.zeros_like(z_pert)

        # MLP predicts no displacement → identical to no-op → margin = 0 < 0.10
        z_pert_pred_noop = z_ctrl.copy()

        out = dynamics_validation_gate(
            z_ctrl, gene_idx, z_pert, z_pert_pred_noop, log_var,
            _default_cfg_gate(),
            baselines_train_data={"z_ctrl": z_tr, "gene_idx": g_tr, "z_pert": zp_tr},
        )

        assert out["passed"] is False
        assert out["primary"]["margin_checks"]["margin_vs_noop_r2"]["pass"] is False

    def test_unseen_val_gene_falls_back_to_global_mean(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        # Train: only genes 1, 2, 3
        z_tr, g_tr, zp_tr = _make_synthetic_pairs(n=300, n_genes=3, seed=2)

        # Val: all gene_idx = 4 (unseen in train)
        z_ctrl, _, z_pert = _make_synthetic_pairs(n=50, n_genes=3, seed=0)
        gene_idx_unseen = np.full(50, 4, dtype=np.int32)
        log_var = np.zeros_like(z_pert)

        # Must not raise; per_gene_mean baseline uses global mean for all rows
        out = dynamics_validation_gate(
            z_ctrl, gene_idx_unseen, z_pert, z_pert, log_var,
            _default_cfg_gate(),
            baselines_train_data={"z_ctrl": z_tr, "gene_idx": g_tr, "z_pert": zp_tr},
        )
        r2_pg = out["primary"]["baselines"]["per_gene_mean_delta"]["r2"]
        assert np.isfinite(r2_pg), f"Expected finite R² for unseen-gene fallback, got {r2_pg}"

    def test_small_train_set_clamps_knn_k(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        # Only 3 train rows — kNN must clamp k to 3 (not 5), else sklearn raises
        rng = np.random.default_rng(0)
        z_ctrl_tr = rng.normal(size=(3, 8)).astype("float32")
        g_idx_tr  = np.array([1, 2, 3], dtype=np.int32)
        zp_tr     = (z_ctrl_tr + 0.1).astype("float32")

        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs(n=20, n_genes=3, seed=0)
        log_var = np.zeros_like(z_pert)

        out = dynamics_validation_gate(
            z_ctrl, gene_idx, z_pert, z_pert, log_var,
            _default_cfg_gate(),
            baselines_train_data={"z_ctrl": z_ctrl_tr, "gene_idx": g_idx_tr, "z_pert": zp_tr},
        )
        assert isinstance(out, dict)

    def test_baselines_use_train_not_val_data(self) -> None:
        from src.analysis.metrics import dynamics_validation_gate

        # Train: gene 1 → Δ = +10, gene 2 → Δ = -10 (strong, distinct signals)
        rng = np.random.default_rng(42)
        n, d = 100, 8
        z_ctrl_tr = rng.normal(size=(n, d)).astype("float32")
        g_idx_tr  = np.array([1] * (n // 2) + [2] * (n // 2), dtype=np.int32)
        zp_tr     = z_ctrl_tr.copy()
        zp_tr[: n // 2]  += 10.0
        zp_tr[n // 2 :] -= 10.0

        # Val: only gene 1 cells; true Δ = +10 (matches train gene-1 mean)
        z_ctrl_v = rng.normal(size=(n, d)).astype("float32")
        g_idx_v  = np.ones(n, dtype=np.int32)
        zp_v     = (z_ctrl_v + 10.0).astype("float32")
        log_var  = np.zeros((n, d), dtype="float32")

        out = dynamics_validation_gate(
            z_ctrl_v, g_idx_v, zp_v, zp_v, log_var,
            _default_cfg_gate(),
            baselines_train_data={"z_ctrl": z_ctrl_tr, "gene_idx": g_idx_tr, "z_pert": zp_tr},
        )
        # Per-gene-mean baseline predicts ~+10 for gene 1 → very high R²
        r2_pg = out["primary"]["baselines"]["per_gene_mean_delta"]["r2"]
        assert r2_pg > 0.8, (
            f"Expected per-gene-mean R²>0.8 when train has matching structure, got {r2_pg}"
        )
