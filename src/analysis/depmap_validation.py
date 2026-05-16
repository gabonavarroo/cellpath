"""DepMap K562 enrichment of RL-selected genes.

Owner: Agent A. See DATA.md §5 and ARCHITECTURE.md Concept 6.

**Honesty constraint.** This module measures *biological plausibility of selected genes*. It
does NOT validate reprogramming. Norman is CRISPRa (gain-of-function); DepMap is
CRISPR/RNAi loss-of-function. Overlap is plausibility, not therapeutic proof. Read Concept 6
before adding any claim.

Tests
-----
1. Hypergeometric enrichment against (a) DepMap K562 essentials, (b) MSigDB Hallmarks,
   (c) hematopoietic-lineage panels.
2. GSEA preranked test of RL action frequencies against the Chronos distribution.
3. Null comparison: matched-size and matched-expression random sets.

All statistical primitives live in :mod:`src.analysis.metrics`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_MIN_GENES_FOR_TEST = 3  # fewer genes → report NA, do not crash


def load_depmap_k562(path: str | Path) -> Any:
    """Load the DepMap K562 Chronos parquet table.

    Parameters
    ----------
    path
        Path to ``data/processed/depmap_k562_chronos.parquet``.

    Returns
    -------
    polars.DataFrame
        Columns: ``gene_symbol`` (str), ``chronos`` (float32), ``is_essential`` (bool).
    """
    import polars as pl

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"DepMap parquet not found at {path}. Run `make data` first."
        )
    df = pl.read_parquet(str(path))
    required = {"gene_symbol", "chronos", "is_essential"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DepMap parquet missing columns: {missing}. "
            f"Present: {df.columns}. Re-run scripts/download_data.sh."
        )
    log.info(
        "Loaded DepMap K562 Chronos: %d genes, %d essential (Chronos < -0.5)",
        len(df), df["is_essential"].sum(),
    )
    return df


def load_gene_panels(panel_dir: str | Path) -> dict[str, list[str]]:
    """Load curated gene-set panels from a directory of plain-text files.

    Each file ``<panel_name>.txt`` must contain one HGNC gene symbol per line.
    Lines starting with ``#`` are ignored (comments). Missing or empty directory
    returns an empty dict gracefully — the caller falls back to the DepMap panel only.

    Parameters
    ----------
    panel_dir
        Directory containing ``*.txt`` panel files.

    Returns
    -------
    dict
        ``{panel_name: [gene_symbol, ...]}``.
    """
    panel_dir = Path(panel_dir)
    panels: dict[str, list[str]] = {}

    if not panel_dir.exists():
        log.warning(
            "Gene panel directory not found: %s — only DepMap essentials panel will be used.",
            panel_dir,
        )
        return panels

    for txt in sorted(panel_dir.glob("*.txt")):
        genes = [
            line.strip() for line in txt.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if genes:
            panels[txt.stem] = genes
            log.info("Loaded panel '%s': %d genes", txt.stem, len(genes))

    return panels


def run_depmap_enrichment(
    rl_action_freq: dict[str, int],
    background_genes: list[str],
    panels: dict[str, list[str]],
    chronos_df: Any,
    top_k: int = 20,
    n_null: int = 1_000,
    expression_mean_per_gene: dict[str, float] | None = None,
    out_path: str | Path | None = None,
) -> Any:
    """Full DepMap enrichment pipeline.

    For each gene panel (including the auto-built DepMap K562 essentials panel):
    1. Hypergeometric test of top-K RL genes vs panel.
    2. GSEA preranked on RL action frequency.
    3. Null comparison (size-matched + expression-matched if expression provided).
    Applies Benjamini-Hochberg FDR correction across all (panel, test) rows.

    Honesty note (ARCHITECTURE.md Concept 6): a positive enrichment finding is
    biological plausibility — not proof of reprogramming or therapeutic validity.

    Parameters
    ----------
    rl_action_freq
        ``{gene_symbol: count}`` from ``artifacts/rl/action_freq.json``.
    background_genes
        HVG universe (list of gene symbols).
    panels
        Output of :func:`load_gene_panels`.
    chronos_df
        Output of :func:`load_depmap_k562`.
    top_k
        Number of top-frequency RL genes to use as the "selected" set.
    n_null
        Permutations for null comparison.
    expression_mean_per_gene
        Optional ``{gene_symbol: mean_expr}`` for expression-matched null.
    out_path
        CSV destination. If ``None``, not written to disk.

    Returns
    -------
    polars.DataFrame
        One row per (panel, test). Columns:
        ``panel``, ``test``, ``p_value``, ``q_value`` (BH-FDR),
        ``effect_size``, ``null_z_score``, ``null_empirical_p``, ``details_json``.
    """
    import polars as pl
    from statsmodels.stats.multitest import multipletests

    from src.analysis.metrics import (
        gsea_preranked,
        hypergeometric_enrichment,
        null_enrichment_comparison,
    )

    # ------------------------------------------------------------------
    # 1. Select top-K genes by RL action frequency
    # ------------------------------------------------------------------
    sorted_genes = sorted(rl_action_freq, key=lambda g: -rl_action_freq[g])
    selected = [g for g in sorted_genes[:top_k] if g in set(background_genes)]
    log.info("Top-%d RL genes: %s", top_k, selected[:10])

    # ------------------------------------------------------------------
    # 2. Build panels dict — always include DepMap K562 essentials
    # ------------------------------------------------------------------
    essential_genes = (
        chronos_df.filter(pl.col("is_essential"))["gene_symbol"].to_list()
    )
    all_panels = {"depmap_k562_essentials": essential_genes}
    all_panels.update(panels)
    log.info("Panels: %s", list(all_panels.keys()))

    # ------------------------------------------------------------------
    # 3. Build GSEA score vector (action frequency, aligned to background)
    # ------------------------------------------------------------------
    bg_set = set(background_genes)
    gsea_scores = np.array(
        [float(rl_action_freq.get(g, 0)) for g in background_genes],
        dtype=np.float64,
    )

    # Expression mean vector aligned to background (for expression-matched null)
    expr_vec: list[float] | None = None
    if expression_mean_per_gene is not None:
        expr_vec = [float(expression_mean_per_gene.get(g, 0.0)) for g in background_genes]

    # ------------------------------------------------------------------
    # 4. Run tests for each panel
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []

    for panel_name, panel_genes in all_panels.items():
        panel_in_bg = [g for g in panel_genes if g in bg_set]
        if not panel_in_bg:
            log.warning("Panel '%s': no genes overlap with background — skipping.", panel_name)
            continue

        log.info("Testing panel '%s' (%d genes in background)...", panel_name, len(panel_in_bg))

        # a. Hypergeometric
        hyper = hypergeometric_enrichment(selected, panel_in_bg, background_genes)
        null_h = null_enrichment_comparison(
            hyper["log_odds"], selected, background_genes,
            n_null_samples=n_null,
            match_expression=np.array(expr_vec) if expr_vec else None,
        )
        rows.append({
            "panel":           panel_name,
            "test":            "hypergeometric",
            "p_value":         hyper["p_value"],
            "effect_size":     hyper["log_odds"],
            "null_z_score":    null_h["z_score"],
            "null_empirical_p": null_h["empirical_p"],
            "details_json":    json.dumps({**hyper, **{f"null_{k}": v for k, v in null_h.items()}}),
        })

        # b. GSEA preranked
        gsea = gsea_preranked(background_genes, gsea_scores, panel_in_bg)
        rows.append({
            "panel":           panel_name,
            "test":            "gsea_preranked",
            "p_value":         gsea["p_value"],
            "effect_size":     gsea["NES"],
            "null_z_score":    float("nan"),
            "null_empirical_p": gsea["fdr"],
            "details_json":    json.dumps(gsea),
        })

    if not rows:
        log.warning("No enrichment rows produced — check panels and background gene overlap.")
        return pl.DataFrame()

    # ------------------------------------------------------------------
    # 5. Benjamini-Hochberg FDR correction across all rows
    # ------------------------------------------------------------------
    df = pl.DataFrame(rows)
    p_vals = df["p_value"].to_numpy()
    _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
    df = df.with_columns(pl.Series("q_value", q_vals.tolist()))

    # Reorder columns
    df = df.select([
        "panel", "test", "p_value", "q_value",
        "effect_size", "null_z_score", "null_empirical_p", "details_json",
    ])

    n_sig = int((df["q_value"] < 0.05).sum())
    log.info(
        "Enrichment complete: %d rows, %d significant (q < 0.05).",
        len(df), n_sig,
    )

    # ------------------------------------------------------------------
    # 6. Write CSV
    # ------------------------------------------------------------------
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(str(out_path))
        log.info("Saved enrichment table → %s", out_path)

    return df


# =============================================================================
# Gene-score comparison: PPO det / PPO stoch / random / action universe
# =============================================================================


def _top_k_genes(freq: dict[str, int], top_k: int) -> list[str]:
    """Return top-K genes by frequency, excluding NO_OP."""
    freq_clean = {g: c for g, c in freq.items() if g != "NO_OP"}
    return [g for g, _ in sorted(freq_clean.items(), key=lambda x: -x[1])[:top_k]]


def _chronos_group_stats(
    chronos_scores: list[float | None],
    counts: list[int] | None = None,
) -> dict[str, Any]:
    """Compute summary statistics for a list of Chronos scores.

    Parameters
    ----------
    chronos_scores
        Per-gene Chronos values (None for genes missing in DepMap).
    counts
        Optional action counts aligned to ``chronos_scores`` for weighted mean.
    """
    valid = [(c, (counts[i] if counts else 1)) for i, c in enumerate(chronos_scores) if c is not None]
    n_total = len(chronos_scores)
    n_with_depmap = len(valid)
    n_missing = n_total - n_with_depmap

    if not valid:
        return {
            "n": n_total,
            "n_with_depmap": 0,
            "n_missing_depmap": n_missing,
            "mean_chronos": None,
            "median_chronos": None,
            "fraction_essential": None,
            "weighted_mean_chronos": None,
        }

    vals = [c for c, _ in valid]
    wts = [w for _, w in valid]
    mean_c = float(np.mean(vals))
    median_c = float(np.median(vals))
    frac_ess = float(np.mean([1 if c < -0.5 else 0 for c in vals]))
    total_wt = sum(wts)
    weighted_mean = float(sum(c * w for c, w in zip(vals, wts)) / total_wt) if total_wt > 0 else None

    return {
        "n": n_total,
        "n_with_depmap": n_with_depmap,
        "n_missing_depmap": n_missing,
        "mean_chronos": round(mean_c, 6),
        "median_chronos": round(median_c, 6),
        "fraction_essential": round(frac_ess, 6),
        "weighted_mean_chronos": round(weighted_mean, 6) if weighted_mean is not None else None,
    }


def _mann_whitney_comparison(
    scores_a: list[float],
    scores_b: list[float],
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Mann-Whitney U test (alternative='less': A tends toward more-negative Chronos than B).

    Cliff's delta = (2 * U) / (n_a * n_b) - 1.
    Negative delta means group A has systematically lower (more essential) Chronos.
    Returns NA fields if either group is too small.
    """
    if len(scores_a) < _MIN_GENES_FOR_TEST or len(scores_b) < _MIN_GENES_FOR_TEST:
        log.warning(
            "Mann-Whitney %s vs %s: too few genes (%d vs %d) — reporting NA.",
            label_a, label_b, len(scores_a), len(scores_b),
        )
        return {
            "test": "mann_whitney_u",
            "alternative": "less (A more negative than B)",
            "n_a": len(scores_a),
            "n_b": len(scores_b),
            "mwu_stat": None,
            "p_value": None,
            "cliffs_delta": None,
            "delta_mean_chronos": None,
            "insufficient_data": True,
        }

    from scipy import stats

    stat, p = stats.mannwhitneyu(scores_a, scores_b, alternative="less")
    cliffs_d = (2 * stat) / (len(scores_a) * len(scores_b)) - 1
    return {
        "test": "mann_whitney_u",
        "alternative": "less (A more negative than B = more essential)",
        "n_a": len(scores_a),
        "n_b": len(scores_b),
        "mwu_stat": float(stat),
        "p_value": float(p),
        "cliffs_delta": round(float(cliffs_d), 6),
        "delta_mean_chronos": round(float(np.mean(scores_a)) - float(np.mean(scores_b)), 6),
        "insufficient_data": False,
    }


def _permutation_comparison(
    observed_scores: list[float],
    universe_scores: list[float],
    label: str,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Permutation test: is mean(observed) lower than expected from universe?

    Draws ``n_permutations`` random sets of size ``len(observed)`` from
    ``universe_scores`` and computes the fraction with mean ≤ observed mean.
    """
    if len(observed_scores) < _MIN_GENES_FOR_TEST or len(universe_scores) < _MIN_GENES_FOR_TEST:
        log.warning(
            "Permutation test for %s: too few genes — reporting NA.", label,
        )
        return {
            "test": "permutation_mean",
            "n_observed": len(observed_scores),
            "n_universe": len(universe_scores),
            "observed_mean": float(np.mean(observed_scores)) if observed_scores else None,
            "null_mean": None,
            "null_std": None,
            "empirical_p": None,
            "n_permutations": n_permutations,
            "insufficient_data": True,
        }

    rng = np.random.default_rng(seed)
    obs_mean = float(np.mean(observed_scores))
    u_arr = np.array(universe_scores, dtype=np.float64)
    n = len(observed_scores)
    null_means = np.array([rng.choice(u_arr, size=n, replace=False).mean()
                           if n <= len(u_arr)
                           else rng.choice(u_arr, size=n, replace=True).mean()
                           for _ in range(n_permutations)])
    empirical_p = float((null_means <= obs_mean).mean())
    return {
        "test": "permutation_mean",
        "n_observed": n,
        "n_universe": len(universe_scores),
        "observed_mean": round(obs_mean, 6),
        "null_mean": round(float(null_means.mean()), 6),
        "null_std": round(float(null_means.std()), 6),
        "empirical_p": round(empirical_p, 6),
        "n_permutations": n_permutations,
        "insufficient_data": False,
    }


def build_gene_level_scores(
    ppo_det_freq: dict[str, int],
    ppo_stoch_freq: dict[str, int],
    random_freq: dict[str, int],
    background_genes: list[str],
    action_universe: list[str] | None,
    chronos_df: Any,
    top_k: int = 20,
) -> Any:
    """Build one row per gene in the union of PPO top-K, random top-K, and action universe.

    Parameters
    ----------
    chronos_df
        Output of :func:`load_depmap_k562` (polars DataFrame with gene_symbol, chronos, is_essential).

    Returns
    -------
    polars.DataFrame
        One row per gene with counts, group membership flags, and Chronos annotation.
    """
    import polars as pl

    chron_lookup: dict[str, tuple[float | None, bool | None]] = {
        row["gene_symbol"]: (row["chronos"], row["is_essential"])
        for row in chronos_df.iter_rows(named=True)
    }

    # Rank DepMap genes by Chronos ascending (rank 1 = most essential = most negative)
    ranked = sorted(
        [(g, c) for g, (c, _) in chron_lookup.items() if c is not None],
        key=lambda x: x[1],
    )
    rank_dict = {g: i + 1 for i, (g, _) in enumerate(ranked)}
    n_depmap_total = len(rank_dict)

    ppo_det_clean = {g: c for g, c in ppo_det_freq.items() if g != "NO_OP"}
    ppo_stoch_clean = {g: c for g, c in ppo_stoch_freq.items() if g != "NO_OP"}
    random_clean = {g: c for g, c in random_freq.items() if g != "NO_OP"}

    ppo_det_top = set(_top_k_genes(ppo_det_clean, top_k))
    ppo_stoch_top = set(_top_k_genes(ppo_stoch_clean, top_k))
    random_top = set(_top_k_genes(random_clean, top_k))
    bg_set = set(background_genes)
    act_set = set(action_universe) if action_universe else set()

    gene_universe = ppo_det_top | ppo_stoch_top | random_top | act_set

    rows: list[dict[str, Any]] = []
    for gene in sorted(gene_universe):
        chron, ess = chron_lookup.get(gene, (None, None))
        rank = rank_dict.get(gene)
        pct = round(rank / n_depmap_total * 100, 2) if rank is not None else None
        rows.append({
            "gene_symbol": gene,
            "ppo_det_count": int(ppo_det_clean.get(gene, 0)),
            "ppo_stoch_count": int(ppo_stoch_clean.get(gene, 0)),
            "random_count": int(random_clean.get(gene, 0)),
            "in_ppo_det_top_k": gene in ppo_det_top,
            "in_ppo_stoch_top_k": gene in ppo_stoch_top,
            "in_random_top_k": gene in random_top,
            "in_action_universe": gene in act_set,
            "in_hvg_background": gene in bg_set,
            "chronos_score": chron,
            "depmap_rank": rank,
            "depmap_percentile": pct,
            "is_essential_chronos_lt_minus_0_5": ess,
            "missing_in_depmap": gene not in chron_lookup,
        })

    return pl.DataFrame(rows)


def run_depmap_comparison(
    *,
    ppo_det_freq: dict[str, int],
    ppo_stoch_freq: dict[str, int],
    random_freq: dict[str, int],
    background_genes: list[str],
    action_universe: list[str] | None,
    chronos_df: Any,
    top_k: int = 20,
    n_permutations: int = 10_000,
    seed: int = 42,
    out_dir: Path | str | None = None,
) -> tuple[Any, dict[str, Any], str]:
    """Compare Chronos score distributions across PPO det, PPO stoch, random, and action universe.

    DepMap plausibility check — NOT validation of biological reprogramming. CRISPRa ≠ CRISPRko.

    Parameters
    ----------
    ppo_det_freq, ppo_stoch_freq, random_freq
        ``{gene_symbol: count}`` dicts from action_freq.json files.
    background_genes
        HVG universe from latents.h5ad var_names.
    action_universe
        Genes in the RL action space from gene_vocab.json (the natural reference distribution).
    chronos_df
        Output of :func:`load_depmap_k562`.
    top_k
        Number of top-frequency genes per policy to use as the "selected" set.
    n_permutations
        Permutation samples for the null comparison.
    seed
        Random seed for reproducibility.
    out_dir
        If given, writes gene_level_scores.csv, comparison_summary.json, comparison_table.md.

    Returns
    -------
    (gene_level_df, summary_dict, table_md_str)
    """
    from statsmodels.stats.multitest import multipletests

    ppo_det_clean = {g: c for g, c in ppo_det_freq.items() if g != "NO_OP"}
    ppo_stoch_clean = {g: c for g, c in ppo_stoch_freq.items() if g != "NO_OP"}
    random_clean = {g: c for g, c in random_freq.items() if g != "NO_OP"}

    ppo_det_top = _top_k_genes(ppo_det_clean, top_k)
    ppo_stoch_top = _top_k_genes(ppo_stoch_clean, top_k)
    random_top = _top_k_genes(random_clean, top_k)
    act_list = list(action_universe) if action_universe else []

    # ------------------------------------------------------------------
    # Gene-level table
    # ------------------------------------------------------------------
    gene_df = build_gene_level_scores(
        ppo_det_freq, ppo_stoch_freq, random_freq,
        background_genes, action_universe, chronos_df, top_k=top_k,
    )

    # ------------------------------------------------------------------
    # Chronos lookups per group
    # ------------------------------------------------------------------
    chron_lookup: dict[str, float | None] = {
        row["gene_symbol"]: row["chronos"]
        for row in chronos_df.iter_rows(named=True)
    }

    def _scores_and_counts(genes: list[str], freq: dict[str, int]) -> tuple[list[float], list[int]]:
        sc, ct = [], []
        for g in genes:
            c = chron_lookup.get(g)
            if c is not None:
                sc.append(c)
                ct.append(int(freq.get(g, 1)))
        return sc, ct

    ppo_det_sc, ppo_det_ct = _scores_and_counts(ppo_det_top, ppo_det_clean)
    ppo_stoch_sc, ppo_stoch_ct = _scores_and_counts(ppo_stoch_top, ppo_stoch_clean)
    random_sc, random_ct = _scores_and_counts(random_top, random_clean)
    act_sc, _ = _scores_and_counts(act_list, {})

    all_sc = [c for c in chron_lookup.values() if c is not None]

    # ------------------------------------------------------------------
    # Group statistics — counts must be aligned with chronos_scores (including Nones)
    # ------------------------------------------------------------------
    def _aligned_counts(genes: list[str], freq: dict[str, int]) -> list[int]:
        """Return per-gene counts aligned with genes list (for use alongside chron_lookup)."""
        return [int(freq.get(g, 1)) for g in genes]

    ppo_det_stats = _chronos_group_stats(
        [chron_lookup.get(g) for g in ppo_det_top],
        _aligned_counts(ppo_det_top, ppo_det_clean),
    )
    ppo_stoch_stats = _chronos_group_stats(
        [chron_lookup.get(g) for g in ppo_stoch_top],
        _aligned_counts(ppo_stoch_top, ppo_stoch_clean),
    )
    random_stats = _chronos_group_stats(
        [chron_lookup.get(g) for g in random_top],
        _aligned_counts(random_top, random_clean),
    )
    act_stats = _chronos_group_stats([chron_lookup.get(g) for g in act_list])

    # ------------------------------------------------------------------
    # Statistical comparisons
    # ------------------------------------------------------------------
    mw_ppo_vs_rand = _mann_whitney_comparison(ppo_det_sc, random_sc, "ppo_det_top_k", "random_top_k")
    mw_ppo_vs_act = _mann_whitney_comparison(ppo_det_sc, act_sc, "ppo_det_top_k", "action_universe")
    mw_stoch_vs_rand = _mann_whitney_comparison(ppo_stoch_sc, random_sc, "ppo_stoch_top_k", "random_top_k")
    perm_ppo_vs_act = _permutation_comparison(ppo_det_sc, act_sc, "ppo_det_vs_action_universe",
                                               n_permutations=n_permutations, seed=seed)
    perm_rand_vs_act = _permutation_comparison(random_sc, act_sc, "random_vs_action_universe",
                                                n_permutations=n_permutations, seed=seed + 1)

    # BH-FDR on the Mann-Whitney p-values
    raw_ps = [
        mw_ppo_vs_rand.get("p_value"),
        mw_ppo_vs_act.get("p_value"),
        mw_stoch_vs_rand.get("p_value"),
    ]
    valid_ps = [p for p in raw_ps if p is not None]
    if valid_ps:
        _, q_arr, _, _ = multipletests(valid_ps, method="fdr_bh")
        q_iter = iter(q_arr.tolist())
        for comp in (mw_ppo_vs_rand, mw_ppo_vs_act, mw_stoch_vs_rand):
            if comp.get("p_value") is not None:
                comp["q_value_bh"] = round(float(next(q_iter)), 6)
            else:
                comp["q_value_bh"] = None

    # ------------------------------------------------------------------
    # Summary dict
    # ------------------------------------------------------------------
    overlap_ppo_rand = sorted(set(ppo_det_top) & set(random_top))
    summary: dict[str, Any] = {
        "note": (
            "DepMap plausibility check — NOT validation of biological reprogramming. "
            "CRISPRa (gain-of-function) ≠ DepMap CRISPR/RNAi (loss-of-function). "
            "More negative Chronos = stronger K562 dependency (essential threshold: Chronos < −0.5)."
        ),
        "top_k": top_k,
        "n_permutations": n_permutations,
        "seed": seed,
        "group_sizes": {
            "ppo_det_unique_genes": len(ppo_det_clean),
            "ppo_stoch_unique_genes": len(ppo_stoch_clean),
            "random_unique_genes": len(random_clean),
            "action_universe": len(act_list),
            "hvg_background": len(background_genes),
        },
        "top_k_lists": {
            "ppo_det": ppo_det_top,
            "ppo_stoch": ppo_stoch_top,
            "random": random_top,
        },
        "overlap_ppo_det_random_top_k": overlap_ppo_rand,
        "chronos_stats": {
            "ppo_det_top_k": ppo_det_stats,
            "ppo_stoch_top_k": ppo_stoch_stats,
            "random_top_k": random_stats,
            "action_universe": act_stats,
        },
        "comparisons": {
            "ppo_det_vs_random_top_k": mw_ppo_vs_rand,
            "ppo_det_vs_action_universe": mw_ppo_vs_act,
            "ppo_stoch_vs_random_top_k": mw_stoch_vs_rand,
            "ppo_det_permutation_vs_action_universe": perm_ppo_vs_act,
            "random_permutation_vs_action_universe": perm_rand_vs_act,
        },
        "interpretation": _build_interpretation(mw_ppo_vs_rand, ppo_det_stats, random_stats),
    }

    # ------------------------------------------------------------------
    # Markdown table
    # ------------------------------------------------------------------
    table_md = _build_comparison_table_md(summary)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gene_path = out_dir / "depmap_gene_level_scores.csv"
        gene_df.write_csv(str(gene_path))
        log.info("Wrote gene-level scores → %s", gene_path)

        summary_path = out_dir / "depmap_comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Wrote comparison summary → %s", summary_path)

        table_path = out_dir / "depmap_comparison_table.md"
        table_path.write_text(table_md)
        log.info("Wrote comparison table → %s", table_path)

    return gene_df, summary, table_md


def _build_interpretation(
    mw_ppo_vs_rand: dict[str, Any],
    ppo_stats: dict[str, Any],
    rand_stats: dict[str, Any],
) -> str:
    """Generate a short, honest interpretation sentence."""
    ppo_mean = ppo_stats.get("mean_chronos")
    rand_mean = rand_stats.get("mean_chronos")
    p = mw_ppo_vs_rand.get("p_value")
    q = mw_ppo_vs_rand.get("q_value_bh")
    cliffs = mw_ppo_vs_rand.get("cliffs_delta")

    if ppo_mean is None or rand_mean is None:
        return "Insufficient data for interpretation."

    direction = "more negative (more K562-essential)" if ppo_mean < rand_mean else "less negative (less K562-essential)"
    delta = abs(ppo_mean - rand_mean)

    if p is None:
        sig = "significance could not be assessed (too few genes)"
    elif (q or p) < 0.05:
        sig = f"statistically significant after BH-FDR correction (q={q:.3f})" if q is not None else f"statistically significant (p={p:.3f})"
    else:
        q_str = f"q={q:.3f}" if q is not None else f"p={p:.3f}"
        sig = f"not statistically significant ({q_str})"

    cliff_str = f" Cliff's delta={cliffs:.3f}." if cliffs is not None else ""

    return (
        f"PPO det top-{mw_ppo_vs_rand.get('n_a', '?')} genes have {direction} mean Chronos "
        f"(Δ={delta:.4f}) vs random top-{mw_ppo_vs_rand.get('n_b', '?')}. "
        f"The difference is {sig}.{cliff_str} "
        "This is a plausibility check only; non-significant results are reported as negative evidence."
    )


def _build_comparison_table_md(summary: dict[str, Any]) -> str:
    """Render the comparison summary as a defense-ready Markdown document."""
    top_k = summary["top_k"]
    stats = summary["chronos_stats"]
    comps = summary["comparisons"]
    note = summary["note"]

    def _row(label: str, s: dict[str, Any], p: float | None = None, q: float | None = None) -> str:
        n = s.get("n_with_depmap", "—")
        mean = f"{s['mean_chronos']:.4f}" if s.get("mean_chronos") is not None else "—"
        median = f"{s['median_chronos']:.4f}" if s.get("median_chronos") is not None else "—"
        frac = f"{s['fraction_essential']:.3f}" if s.get("fraction_essential") is not None else "—"
        wmean = f"{s['weighted_mean_chronos']:.4f}" if s.get("weighted_mean_chronos") is not None else "—"
        p_str = f"{p:.3g}" if p is not None else "—"
        q_str = f"{q:.3g}" if q is not None else "—"
        return f"| {label} | {n} | {mean} | {median} | {frac} | {wmean} | {p_str} | {q_str} |"

    mw_pr = comps.get("ppo_det_vs_random_top_k", {})
    mw_pa = comps.get("ppo_det_vs_action_universe", {})
    mw_sr = comps.get("ppo_stoch_vs_random_top_k", {})
    perm_pa = comps.get("ppo_det_permutation_vs_action_universe", {})

    lines = [
        "# CellPath — DepMap Gene-Score Comparison (MVP V1)",
        "",
        f"> **{note}**",
        "",
        "## Group Chronos statistics",
        "",
        f"| group | n (with DepMap) | mean Chronos | median Chronos | frac essential | weighted mean | p vs random (MWU) | q (BH-FDR) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _row(f"PPO det top-{top_k}", stats["ppo_det_top_k"],
             mw_pr.get("p_value"), mw_pr.get("q_value_bh")),
        _row(f"PPO stoch top-{top_k}", stats["ppo_stoch_top_k"],
             mw_sr.get("p_value"), mw_sr.get("q_value_bh")),
        _row(f"Random top-{top_k}", stats["random_top_k"]),
        _row("Action universe (all)", stats["action_universe"]),
        "",
        "## Statistical comparisons",
        "",
        "All tests: alternative = PPO has **more negative** Chronos than reference (one-sided). "
        "More negative Chronos = stronger K562 dependency.",
        "",
        "| comparison | test | n_A | n_B | Δ mean Chronos | Cliff's delta | p-value | q (BH-FDR) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def _comp_row(label: str, comp: dict[str, Any]) -> str:
        na = comp.get("n_a", "—")
        nb = comp.get("n_b", "—")
        dmc = f"{comp['delta_mean_chronos']:.4f}" if comp.get("delta_mean_chronos") is not None else "—"
        cd = f"{comp['cliffs_delta']:.3f}" if comp.get("cliffs_delta") is not None else "—"
        pv = f"{comp['p_value']:.3g}" if comp.get("p_value") is not None else "NA"
        qv = f"{comp['q_value_bh']:.3g}" if comp.get("q_value_bh") is not None else "NA"
        return f"| {label} | MWU | {na} | {nb} | {dmc} | {cd} | {pv} | {qv} |"

    def _perm_row(label: str, comp: dict[str, Any]) -> str:
        na = comp.get("n_observed", "—")
        nu = comp.get("n_universe", "—")
        obs = f"{comp['observed_mean']:.4f}" if comp.get("observed_mean") is not None else "—"
        null = f"{comp['null_mean']:.4f}" if comp.get("null_mean") is not None else "—"
        pv = f"{comp['empirical_p']:.3g}" if comp.get("empirical_p") is not None else "NA"
        return f"| {label} | permutation | {na} | {nu} | obs={obs} null={null} | — | {pv} | — |"

    lines.append(_comp_row("PPO det vs random (top-K)", mw_pr))
    lines.append(_comp_row("PPO det vs action universe", mw_pa))
    lines.append(_comp_row("PPO stoch vs random (top-K)", mw_sr))
    lines.append(_perm_row("PPO det permutation vs action universe", perm_pa))

    perm_rd = comps.get("random_permutation_vs_action_universe", {})
    lines.append(_perm_row("Random permutation vs action universe", perm_rd))

    lines += [
        "",
        "## Interpretation",
        "",
        summary.get("interpretation", ""),
        "",
        "## Caveats",
        "",
        "- This is a **DepMap plausibility check**, not validation of biological reprogramming.",
        "- Current V1 tests whether PPO-selected genes are more K562 dependency-relevant than "
        "random/background genes.",
        "- **Non-significant results are reported as negative evidence, not hidden.** "
        "Small top-K sample sizes (n ≈ 20) limit statistical power.",
        "- CRISPRa (activation) in Norman 2019 and CRISPR/RNAi (loss-of-function) in DepMap "
        "measure different biological directions. Overlap is plausibility, not therapeutic proof.",
    ]

    return "\n".join(lines) + "\n"
