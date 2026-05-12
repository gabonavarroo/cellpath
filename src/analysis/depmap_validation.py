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

from pathlib import Path
from typing import Any


def load_depmap_k562(path: str | Path) -> Any:
    """Load the parquet table from ``scripts/download_data.sh`` post-processing.

    Parameters
    ----------
    path
        Path to ``data/processed/depmap_k562_chronos.parquet``.

    Returns
    -------
    polars.DataFrame
        Columns: ``gene_symbol``, ``chronos``, ``is_essential``.

    Raises
    ------
    NotImplementedError
        Agent A: ``polars.read_parquet`` + schema check.
    """
    raise NotImplementedError("Agent A: polars.read_parquet + schema check.")


def load_gene_panels(panel_dir: str | Path) -> dict[str, list[str]]:
    """Load curated gene-set panels for enrichment.

    Parameters
    ----------
    panel_dir
        Directory containing one ``<panel>.txt`` per panel (one gene symbol per line).

    Returns
    -------
    dict
        ``{panel_name: [gene_symbol, ...]}``.

    Raises
    ------
    NotImplementedError
        Agent A: iterate ``.txt`` files; HGNC-symbol normalize.
    """
    raise NotImplementedError(
        "Agent A: load gene panels from .txt files. Suggested panels: "
        "depmap_k562_essentials, msigdb_hallmark_hematopoiesis, leukemia_drivers."
    )


def run_depmap_enrichment(
    rl_action_freq: dict[str, int],
    background_genes: list[str],
    panels: dict[str, list[str]],
    chronos_df: Any,
    top_k: int = 20,
    n_null: int = 1_000,
    expression_mean_per_gene: Any | None = None,
    out_path: str | Path | None = None,
) -> Any:
    """Full DepMap enrichment pipeline.

    Parameters
    ----------
    rl_action_freq
        Dict of ``{gene_symbol: count}`` from ``artifacts/rl/action_freq.json``.
    background_genes
        Universe of genes (typically the HVG list).
    panels
        Output of :func:`load_gene_panels`.
    chronos_df
        Output of :func:`load_depmap_k562`.
    top_k
        Number of top-frequency genes to take as the "selected" set.
    n_null
        Permutations for null comparison.
    expression_mean_per_gene
        Optional per-gene mean expression for expression-matched null.
    out_path
        CSV path; defaults to ``cfg.paths.eval_depmap_csv``.

    Returns
    -------
    polars.DataFrame
        One row per (panel, test). Columns include ``q_value`` (BH-FDR) and
        ``null_z_score``.

    Raises
    ------
    NotImplementedError
        Agent A: orchestrate calls to metrics module + Benjamini-Hochberg correction.

    Notes
    -----
    See ARCHITECTURE.md Concept 6 for the strict-interpretation guidance: a positive enrichment
    is plausibility, not therapeutic validity.
    """
    raise NotImplementedError(
        "Agent A: orchestrate hypergeometric + GSEA + null comparison across panels. "
        "Apply Benjamini-Hochberg via statsmodels.stats.multitest.multipletests. "
        "Write CSV with q-values + z-scores."
    )
