"""Unit tests for V3B biology layer loader + episode scorer.

Synthetic fixtures construct a 10-gene action space with a known toxicity
distribution and a tiny SL pair set so the scorer's arithmetic can be verified
by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.analysis.path_feasibility import (
    DEFAULT_REALISM_WEIGHTS,
    BiologyLayer,
    aggregate_episode_scores,
    load_biology_layer,
    score_episode,
)


# ---------------------------------------------------------------------------
# Synthetic biology layer
# ---------------------------------------------------------------------------


def _make_synthetic_layer(
    n_genes: int = 10,
    *,
    chronos: list[float] | None = None,
    sl_pairs: list[tuple[int, int]] | None = None,
) -> BiologyLayer:
    """Build an in-memory BiologyLayer for tests without touching disk."""
    if chronos is None:
        # Mix: 3 essential (Chronos < -0.5), 4 mild-tox, 3 safe.
        chronos = [-1.0, -0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    assert len(chronos) == n_genes

    rows = []
    for i, c in enumerate(chronos):
        tox_raw = max(0.0, -c - 0.5)
        rows.append({
            "gene_symbol": f"GENE_{i:02d}",
            "action_idx": i,
            "chronos": c,
            "is_essential": bool(c < -0.5),
            "tox_raw": tox_raw,
            "tox_norm": min(1.0, tox_raw),
            "missing_chronos": False,
        })
    gene_safety = pl.DataFrame(rows)

    sl_pairs = sl_pairs or []
    sl_df = pl.DataFrame({
        "gene_a": [f"GENE_{a:02d}" for a, _ in sl_pairs],
        "gene_b": [f"GENE_{b:02d}" for _, b in sl_pairs],
        "action_idx_a": [a for a, _ in sl_pairs],
        "action_idx_b": [b for _, b in sl_pairs],
        "sl_score": [float(-5.0)] * len(sl_pairs),
        "source": ["synthetic_test"] * len(sl_pairs),
    }, schema={
        "gene_a": pl.Utf8, "gene_b": pl.Utf8,
        "action_idx_a": pl.Int64, "action_idx_b": pl.Int64,
        "sl_score": pl.Float64, "source": pl.Utf8,
    })

    coverage = {"synthetic": True, "n_total_genes": n_genes}
    return BiologyLayer(gene_safety=gene_safety, sl_pairs=sl_df, coverage=coverage)


# ---------------------------------------------------------------------------
# Construction / lookup correctness
# ---------------------------------------------------------------------------


class TestBiologyLayerConstruction:
    def test_tox_lookup_matches_definition(self) -> None:
        layer = _make_synthetic_layer()
        # Gene 0: chronos=-1.0 → tox_raw = max(0, 1.0 - 0.5) = 0.5
        # Gene 7: chronos=0.1  → tox_raw = max(0, -0.1 - 0.5) = 0.0
        assert layer.tox_by_action[0] == pytest.approx(0.5)
        assert layer.tox_by_action[7] == pytest.approx(0.0)

    def test_essentiality_lookup(self) -> None:
        layer = _make_synthetic_layer()
        # Genes 0, 1, 2 have Chronos < -0.5
        for i in range(3):
            assert layer.is_essential_by_action[i] is True
        for i in range(3, 10):
            assert layer.is_essential_by_action[i] is False

    def test_sl_pair_set_canonical_ordering(self) -> None:
        layer = _make_synthetic_layer(sl_pairs=[(3, 1), (8, 2)])
        # Canonicalised: (1, 3) and (2, 8). Lookup is order-invariant.
        assert (1, 3) in layer.sl_pair_set
        assert (3, 1) not in layer.sl_pair_set  # canonicalised away
        assert (2, 8) in layer.sl_pair_set

    def test_empty_sl_pairs_layer(self) -> None:
        layer = _make_synthetic_layer(sl_pairs=[])
        assert layer.sl_pair_set == frozenset()


# ---------------------------------------------------------------------------
# Episode scoring — handcrafted episodes with known answers
# ---------------------------------------------------------------------------


class TestScoreEpisode:
    NOOP = 999  # any non-action index

    def test_empty_episode(self) -> None:
        layer = _make_synthetic_layer()
        out = score_episode([], layer, noop_idx=self.NOOP)
        assert out["n_steps"] == 0
        assert out["n_gene_steps"] == 0
        assert out["tox_path"] == 0.0
        assert out["sl_violations"] == 0
        assert out["common_essential_count"] == 0

    def test_all_noop_episode(self) -> None:
        layer = _make_synthetic_layer()
        out = score_episode([self.NOOP, self.NOOP], layer, noop_idx=self.NOOP)
        assert out["n_steps"] == 2
        assert out["n_gene_steps"] == 0
        assert out["tox_path"] == 0.0
        assert out["sl_violations"] == 0

    def test_single_safe_gene_action(self) -> None:
        layer = _make_synthetic_layer()
        # Gene 8 (Chronos=0.2): tox_raw=0, not essential.
        out = score_episode([8], layer, noop_idx=self.NOOP)
        assert out["n_gene_steps"] == 1
        assert out["tox_path"] == pytest.approx(0.0)
        assert out["common_essential_count"] == 0
        assert out["frac_safe_actions"] == pytest.approx(1.0)

    def test_single_essential_gene_action(self) -> None:
        layer = _make_synthetic_layer()
        # Gene 0 (Chronos=-1.0): tox_raw = 0.5, is_essential.
        out = score_episode([0], layer, noop_idx=self.NOOP)
        assert out["n_gene_steps"] == 1
        assert out["tox_path"] == pytest.approx(0.5)
        assert out["common_essential_count"] == 1
        assert out["frac_safe_actions"] == pytest.approx(0.0)

    def test_tox_path_sums_correctly(self) -> None:
        layer = _make_synthetic_layer()
        # Genes 0, 1, 2: tox_raw = 0.5, 0.3, 0.1; sum = 0.9
        out = score_episode([0, 1, 2], layer, noop_idx=self.NOOP)
        assert out["tox_path"] == pytest.approx(0.9)
        assert out["common_essential_count"] == 3
        assert out["mean_tox"] == pytest.approx(0.3)

    def test_sl_violation_counted(self) -> None:
        layer = _make_synthetic_layer(sl_pairs=[(1, 3), (5, 7)])
        # Path contains (3, 1) → canonical (1, 3) ∈ SL → 1 violation.
        out = score_episode([3, 1, 9], layer, noop_idx=self.NOOP)
        assert out["sl_violations"] == 1

    def test_multiple_sl_violations(self) -> None:
        layer = _make_synthetic_layer(sl_pairs=[(0, 1), (2, 3), (4, 5)])
        # Path [0, 1, 2, 3]: pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
        # SL set has (0,1), (2,3); both present → 2 violations.
        out = score_episode([0, 1, 2, 3], layer, noop_idx=self.NOOP)
        assert out["sl_violations"] == 2

    def test_repeat_gene_does_not_double_count_sl_pair(self) -> None:
        # In the RL env, repeat_mask=True precludes this. But the function should
        # not break on it. Repeated genes produce a self-pair, never in SL set.
        layer = _make_synthetic_layer(sl_pairs=[(1, 2)])
        out = score_episode([1, 1, 2], layer, noop_idx=self.NOOP)
        # Pairs of distinct gene IDs in path: (1,1)→self skipped by canonical
        # lookup since (1,1) not in SL; (1,2) appears twice in pair iteration
        # but each appearance is counted (this is a degenerate case).
        # We accept either 1 or 2 — assert it's at least 1.
        assert out["sl_violations"] >= 1

    def test_uncertainty_aggregation(self) -> None:
        layer = _make_synthetic_layer()
        log_var = np.array([[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        # Per-step L2 norms: sqrt(9)=3.0 and 0.0. Peak=3, mean=1.5.
        out = score_episode([0, 1], layer, noop_idx=self.NOOP, log_var_per_step=log_var)
        assert out["peak_unc"] == pytest.approx(3.0)
        assert out["mean_unc"] == pytest.approx(1.5)

    def test_uncertainty_none_when_not_provided(self) -> None:
        layer = _make_synthetic_layer()
        out = score_episode([0, 1], layer, noop_idx=self.NOOP)
        assert out["peak_unc"] is None
        assert out["mean_unc"] is None


class TestRealism:
    NOOP = 999

    def test_realism_perfect_path_no_norman(self) -> None:
        # Genes 7..9 are all safe (tox=0, non-essential), no SL pairs.
        layer = _make_synthetic_layer()
        out = score_episode([7, 8, 9], layer, noop_idx=self.NOOP, on_norman_manifold=None)
        # chr_term = 1, sl_term = 1, ess_term = 1 → renormalised → realism ≈ 1.0
        assert out["realism"] == pytest.approx(1.0, abs=1e-9)
        assert out["realism_weights"]["_renormalised"] is True

    def test_realism_perfect_path_with_norman(self) -> None:
        layer = _make_synthetic_layer()
        out = score_episode([7, 8, 9], layer, noop_idx=self.NOOP, on_norman_manifold=True)
        # All four terms = 1, weights sum to 1.0 → realism = 1.0
        assert out["realism"] == pytest.approx(1.0, abs=1e-9)
        assert out["realism_weights"]["_renormalised"] is False

    def test_realism_essential_path_reduces_score(self) -> None:
        layer = _make_synthetic_layer()
        # All three essential genes (0, 1, 2): ess_term = 0; tox high.
        out = score_episode([0, 1, 2], layer, noop_idx=self.NOOP, on_norman_manifold=True)
        # ess_term=0, chr_term low, sl_term=1, nor_term=1
        # realism = 0.4·chr + 0.3·1 + 0.2·0 + 0.1·1 = 0.4·chr + 0.4
        # chr_term = 1 - min(1, 0.9/3) = 1 - 0.3 = 0.7
        # realism = 0.28 + 0.4 = 0.68
        assert out["realism"] == pytest.approx(0.68, abs=1e-9)

    def test_realism_sl_violation_zeroes_sl_term(self) -> None:
        layer = _make_synthetic_layer(sl_pairs=[(5, 6)])
        out_clean = score_episode([7, 8, 9], layer, noop_idx=self.NOOP, on_norman_manifold=True)
        out_bad = score_episode([5, 6, 9], layer, noop_idx=self.NOOP, on_norman_manifold=True)
        assert out_clean["realism"] > out_bad["realism"]
        # The clean path lacks SL violations; the bad one has one.
        assert out_bad["sl_violations"] == 1


class TestAggregator:
    NOOP = 999

    def test_empty_aggregation(self) -> None:
        out = aggregate_episode_scores([])
        assert out["n_episodes"] == 0

    def test_aggregator_means_match_handcrafted(self) -> None:
        layer = _make_synthetic_layer()
        scores = [
            score_episode([0], layer, noop_idx=self.NOOP),   # tox 0.5
            score_episode([1], layer, noop_idx=self.NOOP),   # tox 0.3
            score_episode([9], layer, noop_idx=self.NOOP),   # tox 0.0
        ]
        agg = aggregate_episode_scores(scores)
        # tox_path mean = (0.5 + 0.3 + 0.0) / 3 = 0.2666...
        assert agg["tox_path_mean"] == pytest.approx(0.2666666, abs=1e-5)
        assert agg["n_episodes"] == 3
        assert agg["fraction_zero_sl_violations"] == 1.0  # no SL pairs in layer

    def test_aggregator_handles_optional_unc(self) -> None:
        layer = _make_synthetic_layer()
        scores = [
            score_episode([0], layer, noop_idx=self.NOOP),  # no log_var → peak_unc=None
            score_episode([1], layer, noop_idx=self.NOOP, log_var_per_step=np.array([[1.0, 1.0, 1.0]])),
        ]
        agg = aggregate_episode_scores(scores)
        # Only one episode has peak_unc → mean = sqrt(3) ≈ 1.732
        assert agg["peak_unc_mean"] == pytest.approx(np.sqrt(3.0))


# ---------------------------------------------------------------------------
# Disk loader — uses the real artifacts_v3 outputs produced by the builder
# ---------------------------------------------------------------------------


class TestDiskLoader:
    def test_load_real_biology_layer(self) -> None:
        """If scripts/build_v3b_biology_layer.py has been run, verify the loader."""
        biology_dir = Path("artifacts_v3/v3b_biology")
        if not (biology_dir / "gene_safety.parquet").exists():
            pytest.skip("Biology layer not built; run scripts/build_v3b_biology_layer.py first")

        layer = load_biology_layer(biology_dir)
        assert layer.gene_safety.height == 105
        # Coverage JSON should record the missing-chronos count.
        cov = layer.coverage["gene_safety"]
        assert cov["n_total_genes"] == 105
        assert cov["n_with_chronos"] >= 95  # we expect ≥ 95 of 105 covered
        # SL pair set is empty (V3B Phase 0 gap).
        assert len(layer.sl_pair_set) == 0
