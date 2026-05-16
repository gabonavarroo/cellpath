"""Tests for the DepMap gene-score comparison functions.

Uses synthetic data; no real DepMap parquet or action_freq files are required.
"""

from __future__ import annotations

import json
import math

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_chronos_df():
    """Small DepMap-like table with known Chronos values."""
    return pl.DataFrame({
        "gene_symbol": ["GENEA", "GENEB", "GENEC", "GENED", "GENEE", "GENEF", "GENEG"],
        "chronos": [-0.8, -0.6, -0.3, 0.0, 0.2, 0.4, None],
        "is_essential": [True, True, False, False, False, False, None],
    })


@pytest.fixture()
def mock_ppo_det_freq():
    return {"GENEA": 100, "GENEB": 80, "GENEC": 60, "NO_OP": 5}


@pytest.fixture()
def mock_ppo_stoch_freq():
    return {"GENEA": 95, "GENEB": 75, "GENEC": 55, "GENED": 10, "NO_OP": 3}


@pytest.fixture()
def mock_random_freq():
    return {"GENED": 50, "GENEE": 50, "GENEF": 45, "GENEG": 40, "NO_OP": 10}


@pytest.fixture()
def mock_background():
    return ["GENEA", "GENEB", "GENEC", "GENED", "GENEE", "GENEF", "GENEG"]


@pytest.fixture()
def mock_action_universe():
    return ["GENEA", "GENEB", "GENEC", "GENED", "GENEE", "GENEF"]


# ---------------------------------------------------------------------------
# _chronos_group_stats
# ---------------------------------------------------------------------------

def test_chronos_group_stats_basic():
    from src.analysis.depmap_validation import _chronos_group_stats

    scores = [-0.8, -0.6, 0.0, 0.4]
    s = _chronos_group_stats(scores)
    assert s["n"] == 4
    assert s["n_with_depmap"] == 4
    assert s["n_missing_depmap"] == 0
    assert math.isclose(s["mean_chronos"], sum(scores) / 4, rel_tol=1e-5)
    assert math.isclose(s["fraction_essential"], 0.5, rel_tol=1e-5)  # -0.8, -0.6 < -0.5


def test_chronos_group_stats_with_none():
    from src.analysis.depmap_validation import _chronos_group_stats

    scores = [-0.8, None, 0.3]
    s = _chronos_group_stats(scores)
    assert s["n"] == 3
    assert s["n_with_depmap"] == 2
    assert s["n_missing_depmap"] == 1
    assert s["mean_chronos"] is not None


def test_chronos_group_stats_empty():
    from src.analysis.depmap_validation import _chronos_group_stats

    s = _chronos_group_stats([])
    assert s["n_with_depmap"] == 0
    assert s["mean_chronos"] is None


def test_weighted_mean_chronos():
    from src.analysis.depmap_validation import _chronos_group_stats

    # Gene with count=3 has score -1.0, gene with count=1 has score 0.0
    # Weighted mean = (3*-1 + 1*0) / 4 = -0.75
    scores = [-1.0, 0.0]
    counts = [3, 1]
    s = _chronos_group_stats(scores, counts=counts)
    assert math.isclose(s["weighted_mean_chronos"], -0.75, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# More-negative = more-essential direction
# ---------------------------------------------------------------------------

def test_more_negative_chronos_is_more_essential():
    from src.analysis.depmap_validation import _chronos_group_stats

    # PPO: very negative (more essential)
    ppo = _chronos_group_stats([-0.9, -0.8, -0.7])
    # Random: near zero
    rand = _chronos_group_stats([0.1, 0.2, 0.3])
    assert (ppo["mean_chronos"] or 0) < (rand["mean_chronos"] or 0)
    assert ppo["fraction_essential"] == 1.0
    assert rand["fraction_essential"] == 0.0


# ---------------------------------------------------------------------------
# Mann-Whitney
# ---------------------------------------------------------------------------

def test_mann_whitney_returns_stable_fields():
    from src.analysis.depmap_validation import _mann_whitney_comparison

    a = [-0.9, -0.8, -0.7, -0.6]
    b = [0.1, 0.2, 0.3, 0.4]
    result = _mann_whitney_comparison(a, b, "ppo", "rand")
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["cliffs_delta"] is not None
    assert result["delta_mean_chronos"] < 0  # ppo mean < rand mean


def test_mann_whitney_too_few_genes():
    from src.analysis.depmap_validation import _mann_whitney_comparison

    result = _mann_whitney_comparison([-0.5], [0.1], "tiny_a", "tiny_b")
    assert result["insufficient_data"] is True
    assert result["p_value"] is None


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def test_permutation_returns_valid_p():
    from src.analysis.depmap_validation import _permutation_comparison

    obs = [-0.8, -0.7, -0.6]
    universe = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    result = _permutation_comparison(obs, universe, "test_label", n_permutations=500, seed=0)
    assert not result["insufficient_data"]
    assert 0.0 <= result["empirical_p"] <= 1.0
    assert result["n_permutations"] == 500


# ---------------------------------------------------------------------------
# Missing genes do not crash
# ---------------------------------------------------------------------------

def test_run_depmap_comparison_missing_genes(
    mock_chronos_df, mock_ppo_det_freq, mock_ppo_stoch_freq,
    mock_random_freq, mock_background, mock_action_universe,
):
    """Genes in action_freq that are absent from DepMap are flagged, not crashes."""
    from src.analysis.depmap_validation import run_depmap_comparison

    # GHOST_GENE has the highest count so it lands in top-k; it is absent from DepMap
    ppo_extra = {"GHOST_GENE": 200, "GENEA": 100, "GENEB": 80, "GENEC": 60, "NO_OP": 5}

    gene_df, summary, table_md = run_depmap_comparison(
        ppo_det_freq=ppo_extra,
        ppo_stoch_freq=mock_ppo_stoch_freq,
        random_freq=mock_random_freq,
        background_genes=mock_background,
        action_universe=mock_action_universe,
        chronos_df=mock_chronos_df,
        top_k=3,
        n_permutations=200,
        seed=42,
    )
    missing_rows = gene_df.filter(pl.col("gene_symbol") == "GHOST_GENE")
    assert len(missing_rows) == 1
    assert missing_rows["missing_in_depmap"][0] is True


# ---------------------------------------------------------------------------
# Gene-level CSV has correct flags
# ---------------------------------------------------------------------------

def test_gene_level_scores_flags(
    mock_chronos_df, mock_ppo_det_freq, mock_ppo_stoch_freq,
    mock_random_freq, mock_background, mock_action_universe,
):
    from src.analysis.depmap_validation import run_depmap_comparison

    gene_df, summary, table_md = run_depmap_comparison(
        ppo_det_freq=mock_ppo_det_freq,
        ppo_stoch_freq=mock_ppo_stoch_freq,
        random_freq=mock_random_freq,
        background_genes=mock_background,
        action_universe=mock_action_universe,
        chronos_df=mock_chronos_df,
        top_k=3,
        n_permutations=200,
        seed=42,
    )
    # GENEA is in ppo_det top-3 and in action universe
    genea = gene_df.filter(pl.col("gene_symbol") == "GENEA").to_dicts()[0]
    assert genea["in_ppo_det_top_k"] is True
    assert genea["in_action_universe"] is True
    assert genea["chronos_score"] == pytest.approx(-0.8, rel=1e-4)
    assert genea["is_essential_chronos_lt_minus_0_5"] is True

    # GENED is in random top-3 but not in ppo_det top-3
    gened = gene_df.filter(pl.col("gene_symbol") == "GENED").to_dicts()[0]
    assert gened["in_random_top_k"] is True


# ---------------------------------------------------------------------------
# Markdown table caveats
# ---------------------------------------------------------------------------

def test_markdown_table_has_caveats(
    mock_chronos_df, mock_ppo_det_freq, mock_ppo_stoch_freq,
    mock_random_freq, mock_background, mock_action_universe,
):
    from src.analysis.depmap_validation import run_depmap_comparison

    _, _, table_md = run_depmap_comparison(
        ppo_det_freq=mock_ppo_det_freq,
        ppo_stoch_freq=mock_ppo_stoch_freq,
        random_freq=mock_random_freq,
        background_genes=mock_background,
        action_universe=mock_action_universe,
        chronos_df=mock_chronos_df,
        top_k=3,
        n_permutations=200,
        seed=42,
    )
    assert "plausibility check" in table_md.lower()
    assert "not validation" in table_md.lower() or "not validated" in table_md.lower() or "NOT validation" in table_md
    assert "non-significant" in table_md.lower() or "negative evidence" in table_md.lower()


# ---------------------------------------------------------------------------
# File output round-trip
# ---------------------------------------------------------------------------

def test_run_depmap_comparison_writes_files(
    tmp_path,
    mock_chronos_df, mock_ppo_det_freq, mock_ppo_stoch_freq,
    mock_random_freq, mock_background, mock_action_universe,
):
    from src.analysis.depmap_validation import run_depmap_comparison

    run_depmap_comparison(
        ppo_det_freq=mock_ppo_det_freq,
        ppo_stoch_freq=mock_ppo_stoch_freq,
        random_freq=mock_random_freq,
        background_genes=mock_background,
        action_universe=mock_action_universe,
        chronos_df=mock_chronos_df,
        top_k=3,
        n_permutations=200,
        seed=42,
        out_dir=tmp_path,
    )
    assert (tmp_path / "depmap_gene_level_scores.csv").exists()
    assert (tmp_path / "depmap_comparison_summary.json").exists()
    assert (tmp_path / "depmap_comparison_table.md").exists()

    # Summary JSON must contain the comparison structure
    with open(tmp_path / "depmap_comparison_summary.json") as f:
        s = json.load(f)
    assert "comparisons" in s
    assert "ppo_det_vs_random_top_k" in s["comparisons"]
    assert "chronos_stats" in s
