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


# ---------------------------------------------------------------------------
# TestGateDiagnostics
# ---------------------------------------------------------------------------


class TestGateDiagnostics:
    """gate_diagnostics: shape + JSON-safety + ridge-consistency with the gate."""

    def _val_inputs(self, seed: int = 0):
        z_ctrl, gene_idx, z_pert = _make_synthetic_pairs(
            n=200, n_latent=8, n_genes=3, noise=0.05, seed=seed
        )
        return z_ctrl, gene_idx, z_pert

    def _train_inputs(self, seed: int = 2):
        return _make_synthetic_pairs(
            n=400, n_latent=8, n_genes=3, noise=0.05, seed=seed
        )

    def test_returns_expected_top_level_keys(self) -> None:
        from src.analysis.metrics import gate_diagnostics

        z_tr, g_tr, zp_tr = self._train_inputs()
        z_v,  g_v,  zp_v  = self._val_inputs()
        out = gate_diagnostics(
            z_ctrl_train=z_tr, gene_idx_train=g_tr, z_pert_train=zp_tr,
            z_ctrl_val=z_v,    gene_idx_val=g_v,    z_pert_val=zp_v,
            z_pert_pred_mlp_val=zp_v,
        )
        assert set(out.keys()) == {"overall", "per_dim", "worst_dims", "per_gene_val"}
        assert set(out["overall"].keys()) == {"val", "ood"}
        assert set(out["per_dim"].keys()) == {"val", "ood"}

    def test_ood_none_when_ood_inputs_missing(self) -> None:
        from src.analysis.metrics import gate_diagnostics

        z_tr, g_tr, zp_tr = self._train_inputs()
        z_v,  g_v,  zp_v  = self._val_inputs()
        out = gate_diagnostics(
            z_ctrl_train=z_tr, gene_idx_train=g_tr, z_pert_train=zp_tr,
            z_ctrl_val=z_v,    gene_idx_val=g_v,    z_pert_val=zp_v,
            z_pert_pred_mlp_val=zp_v,
            z_ctrl_ood=None, gene_idx_ood=None, z_pert_ood=None,
            z_pert_pred_mlp_ood=None,
        )
        assert out["overall"]["ood"] is None
        assert out["per_dim"]["ood"] is None
        assert out["worst_dims"]["ood"] == []

    def test_ood_block_populated_when_provided(self) -> None:
        from src.analysis.metrics import gate_diagnostics

        z_tr, g_tr, zp_tr = self._train_inputs(seed=2)
        z_v,  g_v,  zp_v  = self._val_inputs(seed=0)
        z_o,  g_o,  zp_o  = _make_synthetic_pairs(n=100, n_latent=8, n_genes=3, seed=1)

        out = gate_diagnostics(
            z_ctrl_train=z_tr, gene_idx_train=g_tr, z_pert_train=zp_tr,
            z_ctrl_val=z_v,    gene_idx_val=g_v,    z_pert_val=zp_v,
            z_pert_pred_mlp_val=zp_v,
            z_ctrl_ood=z_o, gene_idx_ood=g_o, z_pert_ood=zp_o,
            z_pert_pred_mlp_ood=zp_o,
        )
        assert isinstance(out["overall"]["ood"], dict)
        assert set(out["overall"]["ood"].keys()) == {
            "mlp_r2", "mlp_pearson", "ridge_r2", "ridge_pearson",
            "mlp_minus_ridge_pearson",
        }
        assert isinstance(out["per_dim"]["ood"], dict)

    def test_result_is_json_serializable(self) -> None:
        from src.analysis.metrics import gate_diagnostics

        z_tr, g_tr, zp_tr = self._train_inputs()
        z_v,  g_v,  zp_v  = self._val_inputs()
        out = gate_diagnostics(
            z_ctrl_train=z_tr, gene_idx_train=g_tr, z_pert_train=zp_tr,
            z_ctrl_val=z_v,    gene_idx_val=g_v,    z_pert_val=zp_v,
            z_pert_pred_mlp_val=zp_v,
        )
        json.dumps(out)  # must not raise

    def test_per_gene_val_has_n_and_r2(self) -> None:
        from src.analysis.metrics import gate_diagnostics

        z_tr, g_tr, zp_tr = self._train_inputs()
        z_v,  g_v,  zp_v  = self._val_inputs()
        out = gate_diagnostics(
            z_ctrl_train=z_tr, gene_idx_train=g_tr, z_pert_train=zp_tr,
            z_ctrl_val=z_v,    gene_idx_val=g_v,    z_pert_val=zp_v,
            z_pert_pred_mlp_val=zp_v,
        )
        assert len(out["per_gene_val"]) == len(np.unique(g_v))
        for entry in out["per_gene_val"]:
            assert {"gene_idx", "n", "mlp_r2", "ridge_r2", "mlp_minus_ridge_r2",
                    "mlp_pearson", "ridge_pearson"} <= set(entry.keys())
            assert isinstance(entry["n"], int) and entry["n"] >= 1

    def test_perfect_mlp_pred_gives_high_val_pearson(self) -> None:
        from src.analysis.metrics import gate_diagnostics

        z_tr, g_tr, zp_tr = self._train_inputs()
        z_v,  g_v,  zp_v  = self._val_inputs()
        out = gate_diagnostics(
            z_ctrl_train=z_tr, gene_idx_train=g_tr, z_pert_train=zp_tr,
            z_ctrl_val=z_v,    gene_idx_val=g_v,    z_pert_val=zp_v,
            z_pert_pred_mlp_val=zp_v,  # perfect prediction
        )
        # Perfect MLP prediction → MLP pearson is ~1; MLP - ridge >= 0.
        assert out["overall"]["val"]["mlp_pearson"] > 0.95
        assert out["overall"]["val"]["mlp_minus_ridge_pearson"] >= -1e-3

    def test_ridge_matches_gate_baseline(self) -> None:
        """gate_diagnostics' ridge must match dynamics_validation_gate's ridge exactly."""
        from src.analysis.metrics import (
            dynamics_validation_gate,
            gate_diagnostics,
        )

        z_tr, g_tr, zp_tr = self._train_inputs()
        z_v,  g_v,  zp_v  = self._val_inputs()
        log_var = np.zeros_like(zp_v)

        gate_out = dynamics_validation_gate(
            z_v, g_v, zp_v, zp_v, log_var,
            _default_cfg_gate(),
            baselines_train_data={"z_ctrl": z_tr, "gene_idx": g_tr, "z_pert": zp_tr},
        )
        diag_out = gate_diagnostics(
            z_ctrl_train=z_tr, gene_idx_train=g_tr, z_pert_train=zp_tr,
            z_ctrl_val=z_v,    gene_idx_val=g_v,    z_pert_val=zp_v,
            z_pert_pred_mlp_val=zp_v,
        )

        gate_ridge_pearson = gate_out["primary"]["baselines"]["linear_ridge"]["pearson_r"]
        diag_ridge_pearson = diag_out["overall"]["val"]["ridge_pearson"]
        assert gate_ridge_pearson == pytest.approx(diag_ridge_pearson, abs=1e-6), (
            f"Ridge baseline diverged between gate ({gate_ridge_pearson}) "
            f"and diagnostics ({diag_ridge_pearson}); they MUST share the same fit."
        )


# ---------------------------------------------------------------------------
# TestAblationRecommend (selection logic, no training required)
# ---------------------------------------------------------------------------


class TestAblationRecommend:
    """Selection logic for run_dynamics_ablation.py — pure-Python, no I/O."""

    def _baseline_row(self, **overrides) -> dict:
        row = {
            "name": "baseline",
            "use_state_linear_skip": False, "use_gene_delta_bias": False,
            "status": "complete", "exit_code": 0, "passed": False,
            "val_mlp_r2": 0.38, "val_mlp_pearson": 0.595,
            "val_ridge_pearson": 0.601, "val_ridge_r2": 0.383,
            "val_mlp_minus_ridge_pearson": -0.006,
            "uncertainty_spearman": 0.25, "uncertainty_pass": True,
            "ood_mlp_r2": -0.01, "ood_mlp_pearson": 0.35,
            "ood_ridge_pearson": 0.44, "ood_ridge_r2": 0.18,
            "ood_mlp_minus_ridge_pearson": -0.09,
        }
        row.update(overrides)
        return row

    def _alt_row(self, name, **overrides) -> dict:
        row = self._baseline_row(name=name, use_state_linear_skip=(name != "gene_bias"),
                                 use_gene_delta_bias=(name != "state_linear"))
        row.update(overrides)
        return row

    def test_keep_baseline_when_no_alt_beats(self) -> None:
        from scripts.run_dynamics_ablation import recommend

        rows = [
            self._baseline_row(),
            self._alt_row("state_linear", val_mlp_minus_ridge_pearson=-0.05,
                          ood_mlp_pearson=0.30, ood_mlp_r2=-0.05),
            self._alt_row("gene_bias", val_mlp_minus_ridge_pearson=-0.04,
                          ood_mlp_pearson=0.10, ood_mlp_r2=-0.20),
            self._alt_row("state_linear_gene_bias", val_mlp_minus_ridge_pearson=-0.03,
                          ood_mlp_pearson=0.15, ood_mlp_r2=-0.15),
        ]
        rec = recommend(rows)
        assert rec["setup"] == "keep_baseline"
        assert rec["fallback_invoked"] is True

    def test_state_linear_wins_when_it_improves_margin_without_ood_loss(self) -> None:
        from scripts.run_dynamics_ablation import recommend

        rows = [
            self._baseline_row(),
            self._alt_row("state_linear",
                          val_mlp_minus_ridge_pearson=0.05,  # now positive
                          passed=True,
                          ood_mlp_pearson=0.40, ood_mlp_r2=0.10),
            self._alt_row("gene_bias",
                          val_mlp_minus_ridge_pearson=-0.05,
                          ood_mlp_pearson=0.20, ood_mlp_r2=-0.10),
            self._alt_row("state_linear_gene_bias",
                          val_mlp_minus_ridge_pearson=0.03,
                          ood_mlp_pearson=0.36, ood_mlp_r2=0.05),
        ]
        rec = recommend(rows)
        assert rec["setup"] == "state_linear"
        assert rec["passed"] is True

    def test_prefer_passed_over_better_margin(self) -> None:
        """state_linear_gene_bias passes; state_linear improves margin more but doesn't pass."""
        from scripts.run_dynamics_ablation import recommend

        rows = [
            self._baseline_row(),
            self._alt_row("state_linear",
                          val_mlp_minus_ridge_pearson=0.10, passed=False,
                          ood_mlp_pearson=0.42, ood_mlp_r2=0.05),
            self._alt_row("gene_bias",
                          val_mlp_minus_ridge_pearson=-0.02,
                          ood_mlp_pearson=0.30, ood_mlp_r2=-0.05),
            self._alt_row("state_linear_gene_bias",
                          val_mlp_minus_ridge_pearson=0.05, passed=True,
                          ood_mlp_pearson=0.42, ood_mlp_r2=0.05),
        ]
        rec = recommend(rows)
        assert rec["setup"] == "state_linear_gene_bias"

    def test_memorization_signature_rejects_gene_bias(self) -> None:
        """gene_bias gets big val gain but no OOD movement → rejected."""
        from scripts.run_dynamics_ablation import recommend

        rows = [
            self._baseline_row(),
            self._alt_row("state_linear",
                          val_mlp_minus_ridge_pearson=-0.05,
                          ood_mlp_pearson=0.34, ood_mlp_r2=-0.02),
            self._alt_row("gene_bias",
                          val_mlp_minus_ridge_pearson=0.10,  # big val gain
                          ood_mlp_pearson=0.35, ood_mlp_r2=-0.01),  # no OOD gain
            self._alt_row("state_linear_gene_bias",
                          val_mlp_minus_ridge_pearson=-0.04,
                          ood_mlp_pearson=0.33, ood_mlp_r2=-0.03),
        ]
        rec = recommend(rows)
        # gene_bias must NOT be selected
        assert rec["setup"] != "gene_bias"

    def test_uncertainty_floor_rejects(self) -> None:
        from scripts.run_dynamics_ablation import recommend

        rows = [
            self._baseline_row(),
            self._alt_row("state_linear",
                          val_mlp_minus_ridge_pearson=0.05, passed=True,
                          ood_mlp_pearson=0.42, ood_mlp_r2=0.05,
                          uncertainty_spearman=0.10),  # below floor
        ]
        rec = recommend(rows)
        assert rec["setup"] == "keep_baseline"
        assert "uncertainty" in rec["rationale"].lower() or "no setup" in rec["rationale"].lower()

    def test_incomplete_baseline_falls_back(self) -> None:
        from scripts.run_dynamics_ablation import recommend

        rows = [
            self._baseline_row(status="incomplete"),
        ]
        rec = recommend(rows)
        assert rec["setup"] == "keep_baseline"
        assert rec["fallback_invoked"] is True
