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


