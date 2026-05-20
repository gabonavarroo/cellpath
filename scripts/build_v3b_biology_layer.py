"""V3B Phase 0 — build the biology annotation layer for path safety scoring.

Outputs (under ``cfg.paths.v3b_biology_dir`` / ``artifacts_v3/v3b_biology/``):

* ``gene_safety.parquet`` — one row per gene in the 105-gene action universe with
  Chronos / essentiality columns.
* ``k562_sl_pairs.parquet`` — Horlbeck-2018 K562 synthetic-lethal pairs intersected
  with the 105 genes (if obtainable in this session; otherwise an empty
  zero-row parquet plus a logged gap in the README).
* ``coverage.json`` — provenance + counts for every annotation column.

Honesty caveats:
- DepMap Chronos was generated from CRISPR-Cas9 knockout, **not** CRISPRa activation
  (which is what Norman 2019 uses). Treat Chronos as a prior on "experimental
  disturbance"; do not equate with toxicity. See V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §3.5.
- Horlbeck 2018 SL pairs are K562-specific (≠ Jurkat); we use the K562 partition only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


LOG = logging.getLogger("build_v3b_biology_layer")


# Toxicity score: linear in Chronos with a floor at the essentiality threshold.
# tox(g) = max(0, -Chronos(g) - 0.5)
# Mapping check: Chronos = -0.5 → tox = 0.0; Chronos = -1.5 → tox = 1.0; Chronos = -2.0 → tox = 1.5.
# Normalized version clamps at 1.0 for reward bound preservation.
def _tox_from_chronos(chronos: float | None) -> float | None:
    if chronos is None or not np.isfinite(chronos):
        return None
    return max(0.0, -float(chronos) - 0.5)


def _normalized_tox(tox: float | None, cap: float = 1.0) -> float | None:
    if tox is None:
        return None
    return min(float(cap), max(0.0, float(tox)))


def build_gene_safety_table(
    gene_vocab_path: Path,
    chronos_parquet_path: Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build the per-gene safety table for the 105-gene action universe.

    Columns:
    * ``gene_symbol`` — HGNC symbol matching ``gene_vocab.json``.
    * ``action_idx`` — 0-indexed position in ``gene_vocab.json::genes`` (matches RL action id).
    * ``chronos`` — DepMap K562 Chronos score, or None if not in DepMap.
    * ``is_essential`` — bool, Chronos < -0.5; None if Chronos is None.
    * ``tox_raw`` — max(0, -chronos - 0.5); None if no Chronos.
    * ``tox_norm`` — tox_raw clipped to [0, 1].
    * ``missing_chronos`` — bool sentinel.

    Returns
    -------
    (df, coverage_dict)
    """
    with open(gene_vocab_path) as f:
        vocab = json.load(f)
    genes = list(vocab["genes"])
    n_genes = int(vocab["n_genes"])
    assert len(genes) == n_genes, f"gene_vocab inconsistent: len(genes)={len(genes)} != n_genes={n_genes}"

    chronos = pl.read_parquet(str(chronos_parquet_path))
    required = {"gene_symbol", "chronos", "is_essential"}
    missing_cols = required - set(chronos.columns)
    if missing_cols:
        raise ValueError(f"DepMap parquet missing columns {missing_cols}; got {chronos.columns}")

    # Action index lookup
    action_idx = {g: i for i, g in enumerate(genes)}

    base = pl.DataFrame(
        {"gene_symbol": genes, "action_idx": list(range(len(genes)))}
    )
    joined = base.join(
        chronos.select(["gene_symbol", "chronos", "is_essential"]),
        on="gene_symbol",
        how="left",
    )

    # Compute toxicity columns (Python-side because polars expressions on Option<f64>
    # composed with a max(0, -x - 0.5) clip are clearer in Python for n=105).
    chronos_vals = joined["chronos"].to_list()
    tox_raw = [_tox_from_chronos(c) for c in chronos_vals]
    tox_norm = [_normalized_tox(t, cap=1.0) for t in tox_raw]
    missing_chr = [c is None for c in chronos_vals]

    out = joined.with_columns(
        pl.Series("tox_raw", tox_raw, dtype=pl.Float64),
        pl.Series("tox_norm", tox_norm, dtype=pl.Float64),
        pl.Series("missing_chronos", missing_chr, dtype=pl.Boolean),
    )

    # Sort by action_idx for stable downstream indexing
    out = out.sort("action_idx")

    n_with_chr = int(out.filter(~pl.col("missing_chronos")).height)
    n_essential = int(out.filter(pl.col("is_essential") == True).height)  # noqa: E712 (polars bool col)
    essential_genes = out.filter(pl.col("is_essential") == True)["gene_symbol"].to_list()  # noqa: E712
    missing_genes = out.filter(pl.col("missing_chronos") == True)["gene_symbol"].to_list()  # noqa: E712

    coverage = {
        "n_total_genes": int(len(genes)),
        "n_with_chronos": n_with_chr,
        "fraction_with_chronos": round(n_with_chr / max(len(genes), 1), 4),
        "n_essential_chronos_lt_minus_0_5": n_essential,
        "essential_genes": sorted(essential_genes),
        "missing_chronos_genes": sorted(missing_genes),
        "chronos_stats_on_105": {
            "mean":   float(np.nanmean([c for c in chronos_vals if c is not None])) if n_with_chr else None,
            "median": float(np.nanmedian([c for c in chronos_vals if c is not None])) if n_with_chr else None,
            "min":    float(np.nanmin([c for c in chronos_vals if c is not None])) if n_with_chr else None,
            "max":    float(np.nanmax([c for c in chronos_vals if c is not None])) if n_with_chr else None,
            "p25":    float(np.nanpercentile([c for c in chronos_vals if c is not None], 25)) if n_with_chr else None,
            "p75":    float(np.nanpercentile([c for c in chronos_vals if c is not None], 75)) if n_with_chr else None,
        },
        "essentiality_threshold": -0.5,
        "tox_raw_definition": "max(0, -chronos - 0.5)",
        "tox_norm_definition": "min(1.0, tox_raw)",
        "honesty_caveat": (
            "DepMap Chronos is CRISPR-Cas9 knockout. Norman 2019 is CRISPRa. "
            "Treat Chronos as a prior on K562 experimental disturbance, not as "
            "direct CRISPRa toxicity. See V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §3.5."
        ),
    }

    return out, coverage


def build_horlbeck_sl_table(
    horlbeck_source: Path | None,
    gene_vocab_path: Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build the K562 synthetic-lethal pair table intersected with the 105-gene universe.

    Schema:
    * ``gene_a`` — HGNC symbol of first gene of the SL pair.
    * ``gene_b`` — HGNC symbol of second gene of the SL pair.
    * ``action_idx_a``, ``action_idx_b`` — 0-indexed positions in gene_vocab.
    * ``sl_score`` — float, source-specific magnitude (Horlbeck GI score, more negative
      = stronger SL); None if the source did not provide a magnitude.
    * ``source`` — provenance string (e.g. ``"horlbeck_2018_k562"``).

    If ``horlbeck_source`` is None or not present, returns an empty DataFrame with
    the expected schema and a logged gap. Downstream code must tolerate the empty
    case (no SL penalty applied).

    Notes
    -----
    The Horlbeck 2018 supplementary tables are gated behind the Cell paywall;
    direct programmatic download is unreliable. This loader supports three formats:

    1. ``.csv`` / ``.parquet`` from SLKB-K562 export with columns
       ``gene_a, gene_b, sl_score, source``.
    2. Horlbeck supplementary Table S5 (GI scores) — columns
       ``gene_A_symbol, gene_B_symbol, GI_score_K562``; we threshold |GI| > 3 for
       SL, matching the SLKB Horlbeck-method cutoff.
    3. Empty / missing source → return empty parquet + log a gap.
    """
    with open(gene_vocab_path) as f:
        vocab = json.load(f)
    genes_105 = list(vocab["genes"])
    gene_to_idx = {g: i for i, g in enumerate(genes_105)}
    gene_set = set(genes_105)

    schema = {
        "gene_a": pl.Utf8,
        "gene_b": pl.Utf8,
        "action_idx_a": pl.Int64,
        "action_idx_b": pl.Int64,
        "sl_score": pl.Float64,
        "source": pl.Utf8,
    }
    empty = pl.DataFrame(schema=schema)

    gap = {
        "source_path": str(horlbeck_source) if horlbeck_source else None,
        "source_exists": False,
        "format_used": None,
        "n_raw_sl_pairs_loaded": 0,
        "n_pairs_after_105_intersection": 0,
        "fraction_genes_with_at_least_one_sl_pair": 0.0,
        "honesty_caveat": (
            "Horlbeck 2018 K562 SL pairs are cell-line-specific (only ~82 of "
            "1,523 K562 pairs overlap Jurkat). Pairs are CRISPRi (knockdown); "
            "interpretation under CRISPRa (activation) requires biological "
            "judgement. See V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §3.5."
        ),
    }

    if horlbeck_source is None:
        LOG.warning("No Horlbeck source provided — writing empty SL table.")
        gap["gap_reason"] = "no horlbeck_source argument"
        return empty, gap

    if not horlbeck_source.exists():
        LOG.warning("Horlbeck source %s does not exist — writing empty SL table.", horlbeck_source)
        gap["gap_reason"] = f"file not found: {horlbeck_source}"
        return empty, gap

    gap["source_exists"] = True

    # Format dispatch
    if horlbeck_source.suffix == ".parquet":
        raw = pl.read_parquet(str(horlbeck_source))
        gap["format_used"] = "parquet_slkb_export"
        if {"gene_a", "gene_b"} <= set(raw.columns):
            df = raw
        else:
            raise ValueError(
                f"Horlbeck parquet {horlbeck_source} missing gene_a/gene_b cols; "
                f"got {raw.columns}"
            )
    elif horlbeck_source.suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if horlbeck_source.suffix == ".tsv" else ","
        # Try auto-detect of expected schemas
        try:
            raw = pl.read_csv(str(horlbeck_source), separator=sep, infer_schema_length=10_000)
        except Exception as exc:
            raise ValueError(f"Failed to parse Horlbeck source {horlbeck_source}: {exc}") from exc

        cols_lower = {c.lower(): c for c in raw.columns}
        if {"gene_a", "gene_b"} <= set(c.lower() for c in raw.columns):
            df = raw.rename({cols_lower["gene_a"]: "gene_a", cols_lower["gene_b"]: "gene_b"})
            if "sl_score" not in df.columns and "score" in cols_lower:
                df = df.rename({cols_lower["score"]: "sl_score"})
            gap["format_used"] = "csv_slkb_like"
        elif {"gene_a_symbol", "gene_b_symbol"} <= set(c.lower() for c in raw.columns):
            df = raw.rename(
                {cols_lower["gene_a_symbol"]: "gene_a", cols_lower["gene_b_symbol"]: "gene_b"}
            )
            gi_col = next((cols_lower[c] for c in cols_lower if "gi_score" in c), None)
            if gi_col is None:
                raise ValueError(f"Horlbeck file {horlbeck_source}: no GI score column found")
            df = df.rename({gi_col: "sl_score"})
            # Horlbeck threshold: |GI| > 3 typically marks SL (negative GI in their convention).
            n_pre = df.height
            df = df.filter(pl.col("sl_score") < -3.0)
            LOG.info("Horlbeck SL filter |GI| < -3.0: %d -> %d pairs", n_pre, df.height)
            gap["format_used"] = "horlbeck_supplementary_table_S5"
        else:
            raise ValueError(
                f"Unrecognised Horlbeck schema; got columns {raw.columns}. "
                f"Expected SLKB-like (gene_a, gene_b) or Horlbeck-like (gene_A_symbol, gene_B_symbol)."
            )
    else:
        raise ValueError(f"Unsupported Horlbeck file extension {horlbeck_source.suffix}")

    if "sl_score" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("sl_score"))
    if "source" not in df.columns:
        df = df.with_columns(pl.lit("horlbeck_2018_k562").alias("source"))

    n_raw = df.height
    gap["n_raw_sl_pairs_loaded"] = int(n_raw)

    # Intersect with 105 genes
    df = df.filter(pl.col("gene_a").is_in(genes_105) & pl.col("gene_b").is_in(genes_105))
    n_intersect = df.height
    gap["n_pairs_after_105_intersection"] = int(n_intersect)

    if n_intersect == 0:
        LOG.warning(
            "Horlbeck SL pairs in 105-gene universe: 0/%d after intersection.",
            n_raw,
        )
        gap["gap_reason"] = "zero SL pairs survive 105-gene intersection"
        return empty, gap

    df = df.with_columns(
        pl.col("gene_a").map_elements(lambda g: gene_to_idx[g], return_dtype=pl.Int64).alias("action_idx_a"),
        pl.col("gene_b").map_elements(lambda g: gene_to_idx[g], return_dtype=pl.Int64).alias("action_idx_b"),
        pl.col("sl_score").cast(pl.Float64),
    )

    # Canonical ordering so (a,b) == (b,a) is a single row
    df = df.with_columns(
        pl.when(pl.col("action_idx_a") <= pl.col("action_idx_b"))
        .then(pl.col("gene_a")).otherwise(pl.col("gene_b")).alias("_ga"),
        pl.when(pl.col("action_idx_a") <= pl.col("action_idx_b"))
        .then(pl.col("gene_b")).otherwise(pl.col("gene_a")).alias("_gb"),
    ).drop(["gene_a", "gene_b"]).rename({"_ga": "gene_a", "_gb": "gene_b"})
    df = df.with_columns(
        pl.col("gene_a").map_elements(lambda g: gene_to_idx[g], return_dtype=pl.Int64).alias("action_idx_a"),
        pl.col("gene_b").map_elements(lambda g: gene_to_idx[g], return_dtype=pl.Int64).alias("action_idx_b"),
    ).unique(subset=["gene_a", "gene_b"])

    # Genes involved in at least one SL pair
    sl_genes = set(df["gene_a"].to_list()) | set(df["gene_b"].to_list())
    gap["fraction_genes_with_at_least_one_sl_pair"] = round(
        len(sl_genes & gene_set) / max(len(gene_set), 1), 4
    )

    df = df.select(["gene_a", "gene_b", "action_idx_a", "action_idx_b", "sl_score", "source"])
    return df, gap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gene_vocab",
        default="artifacts/vae/gene_vocab.json",
        help="Path to gene vocabulary JSON (defaults to V1 vocab — identical to V2/V3 by audit).",
    )
    parser.add_argument(
        "--chronos_parquet",
        default="data/processed/depmap_k562_chronos.parquet",
        help="Path to DepMap K562 Chronos parquet.",
    )
    parser.add_argument(
        "--horlbeck_source",
        default=None,
        help="Optional path to Horlbeck 2018 K562 SL table (CSV/TSV/parquet). "
        "If omitted or missing, the SL table is written empty.",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts_v3/v3b_biology",
        help="Output directory under artifacts_v3/.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    gene_vocab_path = (repo_root / args.gene_vocab).resolve() if not Path(args.gene_vocab).is_absolute() else Path(args.gene_vocab)
    chronos_path = (repo_root / args.chronos_parquet).resolve() if not Path(args.chronos_parquet).is_absolute() else Path(args.chronos_parquet)
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horlbeck_path: Path | None = None
    if args.horlbeck_source:
        horlbeck_path = (
            (repo_root / args.horlbeck_source).resolve()
            if not Path(args.horlbeck_source).is_absolute()
            else Path(args.horlbeck_source)
        )

    LOG.info("Building gene_safety table from %s + %s", gene_vocab_path, chronos_path)
    safety_df, safety_cov = build_gene_safety_table(gene_vocab_path, chronos_path)
    safety_out = out_dir / "gene_safety.parquet"
    safety_df.write_parquet(str(safety_out))
    LOG.info(
        "Wrote %s (%d rows, %d/%d with Chronos, %d essential)",
        safety_out,
        safety_df.height,
        safety_cov["n_with_chronos"],
        safety_cov["n_total_genes"],
        safety_cov["n_essential_chronos_lt_minus_0_5"],
    )

    LOG.info("Building Horlbeck SL pair table (source=%s)", horlbeck_path)
    sl_df, sl_cov = build_horlbeck_sl_table(horlbeck_path, gene_vocab_path)
    sl_out = out_dir / "k562_sl_pairs.parquet"
    sl_df.write_parquet(str(sl_out))
    LOG.info(
        "Wrote %s (%d rows after 105-gene intersection)",
        sl_out,
        sl_df.height,
    )

    coverage = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "gene_vocab": str(gene_vocab_path),
            "chronos_parquet": str(chronos_path),
            "horlbeck_source": str(horlbeck_path) if horlbeck_path else None,
        },
        "gene_safety": safety_cov,
        "k562_sl_pairs": sl_cov,
    }
    cov_path = out_dir / "coverage.json"
    cov_path.write_text(json.dumps(coverage, indent=2, default=str))
    LOG.info("Wrote %s", cov_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
