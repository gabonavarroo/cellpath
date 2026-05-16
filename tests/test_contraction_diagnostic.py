"""Tests for the dynamics contraction diagnostic.

We test the pure-function core (``evaluate_contraction`` + ``aggregate_contraction``) with
synthetic dynamics stubs. The Hydra entry point is exercised only via a dry-run smoke test
elsewhere; the math is what matters.

Locked semantics tested:
- **Strict improvement**: ``improvement > 0``. A zero-mu model has ``d_after == d_before``
  for every (start, gene), so ``fraction_improved == 0.0`` — NOT 1.0 (per the plan).
- A "perfect-contractive" model that always returns ``mu = z_ref - z`` collapses every cell
  to ``z_ref``, so ``fraction_improved == 1.0`` and ``mean_improvement == mean(d_before)``.
- Per-gene rows have the correct count and gene_symbols are propagated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _make_starts(n: int, d: int, *, rng_seed: int = 0) -> np.ndarray:
    """N start states drawn from a Gaussian shifted away from origin (so d_before > 0)."""
    rng = np.random.default_rng(rng_seed)
    return (rng.normal(loc=2.0, scale=1.0, size=(n, d))).astype(np.float32)


def _zero_mu(z: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
    """Stub: predict zero displacement."""
    return np.zeros_like(z, dtype=np.float32)


def _perfect_contractive(z_ref: np.ndarray) -> Any:
    """Return a stub callable that maps every (z, g) → mu = z_ref - z."""
    z_ref = np.asarray(z_ref, dtype=np.float32)

    def _call(z: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
        return (z_ref - z).astype(np.float32)

    return _call


def _gene_specific_contractive(z_ref: np.ndarray, contractive_gene: int) -> Any:
    """Stub that only the ``contractive_gene`` (1-indexed) contracts; all others are zero-mu."""
    z_ref = np.asarray(z_ref, dtype=np.float32)

    def _call(z: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
        mu = np.zeros_like(z, dtype=np.float32)
        m = (gene_idx == contractive_gene)
        mu[m] = z_ref - z[m]
        return mu

    return _call


# Helper to import the script (it's in scripts/, not on sys.path by default).
def _import_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import diagnose_dynamics_contraction as ddc  # noqa: WPS433
    return ddc


# =============================================================================
# evaluate_contraction
# =============================================================================


class TestEvaluateContraction:
    def test_shapes_match_n_starts_x_n_genes(self) -> None:
        ddc = _import_module()
        starts = _make_starts(20, 8)
        z_ref = np.zeros(8, dtype=np.float32)
        out = ddc.evaluate_contraction(starts, z_ref, _zero_mu, n_genes=5, chunk_starts=7)
        assert out["d_before"].shape == (20, 5)
        assert out["d_after"].shape == (20, 5)
        assert out["improvement"].shape == (20, 5)

    def test_zero_mu_yields_zero_improvement(self) -> None:
        """Zero-mu model → d_after == d_before → improvement == 0 everywhere."""
        ddc = _import_module()
        starts = _make_starts(50, 16)
        z_ref = np.zeros(16, dtype=np.float32)
        out = ddc.evaluate_contraction(starts, z_ref, _zero_mu, n_genes=4)
        np.testing.assert_allclose(out["improvement"], 0.0, atol=1e-6)

    def test_perfect_contractive_collapses_everything(self) -> None:
        """A model that always returns mu = z_ref - z → d_after == 0 ∀ (start, gene)."""
        ddc = _import_module()
        starts = _make_starts(40, 8)
        z_ref = np.zeros(8, dtype=np.float32)
        out = ddc.evaluate_contraction(starts, z_ref, _perfect_contractive(z_ref), n_genes=3)
        np.testing.assert_allclose(out["d_after"], 0.0, atol=1e-5)
        # d_before is positive (starts are shifted), so improvement is positive ∀ pair.
        assert (out["improvement"] > 0).all()
        np.testing.assert_allclose(out["improvement"], out["d_before"], atol=1e-5)

    def test_invalid_shapes_raise(self) -> None:
        ddc = _import_module()
        with pytest.raises(ValueError, match="starts must be"):
            ddc.evaluate_contraction(
                np.zeros(8, dtype=np.float32),  # 1-D, not (N, D)
                np.zeros(8, dtype=np.float32),
                _zero_mu, n_genes=2,
            )
        with pytest.raises(ValueError, match="z_ref shape"):
            ddc.evaluate_contraction(
                np.zeros((4, 8), dtype=np.float32),
                np.zeros(7, dtype=np.float32),  # wrong D
                _zero_mu, n_genes=2,
            )
        with pytest.raises(ValueError, match="n_genes"):
            ddc.evaluate_contraction(
                np.zeros((4, 8), dtype=np.float32),
                np.zeros(8, dtype=np.float32),
                _zero_mu, n_genes=0,
            )

    def test_chunking_matches_single_batch(self) -> None:
        """Result must be invariant to chunk size."""
        ddc = _import_module()
        starts = _make_starts(33, 12)
        z_ref = np.zeros(12, dtype=np.float32)
        f = _perfect_contractive(z_ref)
        a = ddc.evaluate_contraction(starts, z_ref, f, n_genes=4, chunk_starts=33)
        b = ddc.evaluate_contraction(starts, z_ref, f, n_genes=4, chunk_starts=5)
        np.testing.assert_allclose(a["improvement"], b["improvement"], atol=1e-6)


# =============================================================================
# aggregate_contraction
# =============================================================================


class TestAggregateContraction:
    def test_zero_mu_fraction_improved_strict_zero(self) -> None:
        """The locked, plan-mandated test: zero-mu → fraction_improved == 0.0 (strict)."""
        ddc = _import_module()
        starts = _make_starts(50, 16)
        z_ref = np.zeros(16, dtype=np.float32)
        out = ddc.evaluate_contraction(starts, z_ref, _zero_mu, n_genes=4)
        agg = ddc.aggregate_contraction(out["improvement"])
        assert agg["summary"]["fraction_improved"] == pytest.approx(0.0)
        assert agg["summary"]["mean_improvement"] == pytest.approx(0.0, abs=1e-6)
        assert agg["summary"]["best_improvement"] == pytest.approx(0.0, abs=1e-6)
        assert agg["summary"]["worst_improvement"] == pytest.approx(0.0, abs=1e-6)
        # Random baseline trivially zero
        assert agg["random_action_baseline"]["mean_improvement_uniform_random"] == pytest.approx(0.0, abs=1e-6)

    def test_perfect_contractive_fraction_improved_one(self) -> None:
        ddc = _import_module()
        starts = _make_starts(40, 8)
        z_ref = np.zeros(8, dtype=np.float32)
        out = ddc.evaluate_contraction(starts, z_ref, _perfect_contractive(z_ref), n_genes=3)
        agg = ddc.aggregate_contraction(out["improvement"])
        assert agg["summary"]["fraction_improved"] == pytest.approx(1.0)
        # Mean improvement equals mean d_before (since d_after == 0)
        expected_mean = float(out["d_before"].mean())
        assert agg["summary"]["mean_improvement"] == pytest.approx(expected_mean, rel=1e-5)
        # Random baseline matches (uniform over genes, all genes equally good here)
        assert (
            agg["random_action_baseline"]["mean_improvement_uniform_random"]
            == pytest.approx(expected_mean, rel=1e-5)
        )

    def test_per_gene_table_has_correct_rows_and_symbols(self) -> None:
        ddc = _import_module()
        starts = _make_starts(20, 4)
        z_ref = np.zeros(4, dtype=np.float32)
        out = ddc.evaluate_contraction(starts, z_ref, _zero_mu, n_genes=3)
        agg = ddc.aggregate_contraction(
            out["improvement"], gene_symbols=["GENE_A", "GENE_B", "GENE_C"],
        )
        rows = agg["per_gene"]
        assert len(rows) == 3
        # Row symbols == gene_symbols passed in (order may differ — sorted by mean_improvement)
        assert {r["gene_symbol"] for r in rows} == {"GENE_A", "GENE_B", "GENE_C"}
        # gene_idx is 1-indexed
        assert {r["gene_idx"] for r in rows} == {1, 2, 3}
        for r in rows:
            assert r["n_starts"] == 20
            assert r["fraction_improved"] == pytest.approx(0.0)

    def test_per_gene_fallback_symbols_when_none(self) -> None:
        ddc = _import_module()
        starts = _make_starts(10, 4)
        z_ref = np.zeros(4, dtype=np.float32)
        out = ddc.evaluate_contraction(starts, z_ref, _zero_mu, n_genes=2)
        agg = ddc.aggregate_contraction(out["improvement"], gene_symbols=None)
        # Fallback uses gene_<1-indexed>
        symbols = {r["gene_symbol"] for r in agg["per_gene"]}
        assert symbols == {"gene_1", "gene_2"}

    def test_gene_specific_contractive_identifies_winner(self) -> None:
        """Only gene_idx == 2 contracts; per-gene table must rank it first by mean_improvement."""
        ddc = _import_module()
        starts = _make_starts(60, 6)
        z_ref = np.zeros(6, dtype=np.float32)
        out = ddc.evaluate_contraction(
            starts, z_ref, _gene_specific_contractive(z_ref, contractive_gene=2), n_genes=4,
        )
        agg = ddc.aggregate_contraction(out["improvement"], gene_symbols=["G1", "G2", "G3", "G4"])
        # Sorted by mean_improvement descending → first row is the contractive one
        first = agg["per_gene"][0]
        assert first["gene_idx"] == 2
        assert first["gene_symbol"] == "G2"
        assert first["fraction_improved"] == pytest.approx(1.0)
        # Other rows: zero-mu, fraction_improved == 0.0
        for r in agg["per_gene"][1:]:
            assert r["fraction_improved"] == pytest.approx(0.0)
        # Overall fraction improved: only column 2 contracts → 1/4
        assert agg["summary"]["fraction_improved"] == pytest.approx(0.25, abs=1e-6)
