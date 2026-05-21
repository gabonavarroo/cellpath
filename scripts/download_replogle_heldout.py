"""Download + parse the Replogle 2022 K562 Essential Perturb-seq held-out gene set.

Source: Harmonizome mirror of Replogle et al., *Cell* 185, 2559-2575 (2022).
URL:    https://maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.,+Cell,+2022+K562+Essential+Perturb-seq+Gene+Perturbation+Signatures

What this script does (Stage 2 of the V3C final Replogle audit):

1. Downloads the Harmonizome gene-list and gene-attribute files for the
   `reploglek562essential` slug into `data/raw/replogle/` (gzipped, small — <2 MB total).
2. Parses the gene list to build the canonical Replogle-essential gene set
   (CRISPRi knockdown perturbations that produced an essential-screen signature).
3. Intersects with the Norman 105 single-gene CRISPRa action universe (from
   `data/processed/norman_hvg.h5ad`) and with the DepMap K562 essential set
   (`is_essential = True` in `data/processed/depmap_k562_chronos.parquet`) to
   reconstruct the Phase 2c headline counts.
4. Writes small processed JSON/CSV artifacts under `data/processed/replogle/`:
   `replogle_essential_genes.json`, `replogle_essential_genes.csv`,
   `replogle_norman_intersection.json`, `replogle_only_essential_genes.json`,
   `source_metadata.json`.

This is *only* the held-out gene-set source. It is not the Replogle CRISPRi
single-cell h5ad referenced by `paths.replogle_crispri_h5ad` (placeholder for a
future knockout/CRISPRi action-space extension).

Sacred-rule conformance: writes are restricted to `data/raw/replogle/` and
`data/processed/replogle/`. Does not touch frozen tiers. Reads from
`data/processed/norman_hvg.h5ad` and `data/processed/depmap_k562_chronos.parquet`.

Usage:
    python scripts/download_replogle_heldout.py
    python scripts/download_replogle_heldout.py --skip-download   # reuse cached raw files
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import logging
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

LOG = logging.getLogger("download_replogle_heldout")

HARMONIZOME_SLUG = "reploglek562essential"
HARMONIZOME_BASE = (
    f"https://maayanlab.cloud/static/hdfs/harmonizome/data/{HARMONIZOME_SLUG}"
)
SOURCE_PAGE = (
    "https://maayanlab.cloud/Harmonizome/dataset/"
    "Replogle+et+al.,+Cell,+2022+K562+Essential+Perturb-seq+"
    "Gene+Perturbation+Signatures"
)

FILES = {
    "gene_list_terms.txt.gz": "Gene list — one row per Replogle-essential HGNC symbol",
    "attribute_list_entries.txt.gz": "Attribute list — per-perturbation signatures",
    "gene_attribute_edges.txt.gz": "Sparse gene-attribute edge list",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "replogle"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "replogle"
NORMAN_H5AD = REPO_ROOT / "data" / "processed" / "norman_hvg.h5ad"
DEPMAP_PARQUET = REPO_ROOT / "data" / "processed" / "depmap_k562_chronos.parquet"


def _http_get(url: str, dest: Path, timeout: int = 60) -> None:
    LOG.info("GET %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "cellpath/v3c-audit"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    LOG.info("    -> %s  (%d bytes)", dest, len(data))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def download_files(force: bool = False) -> dict[str, dict[str, Any]]:
    """Download the three Harmonizome files. Returns provenance per file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict[str, Any]] = {}
    for filename in FILES:
        url = f"{HARMONIZOME_BASE}/{filename}"
        dest = RAW_DIR / filename
        if dest.exists() and not force:
            LOG.info("cached: %s  (use --force-download to refetch)", dest)
        else:
            try:
                _http_get(url, dest)
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                LOG.error("download failed for %s: %s", url, exc)
                provenance[filename] = {
                    "url": url,
                    "status": "FAILED",
                    "error": str(exc),
                }
                continue
        provenance[filename] = {
            "url": url,
            "status": "OK",
            "bytes": dest.stat().st_size,
            "sha256": _sha256(dest),
        }
    return provenance


_PERTURBATION_RE = re.compile(r"^\d+_(.+?)_(P\d+(?:P\d+)?)$")


def parse_essential_perturbations(path: Path) -> tuple[list[str], list[str]]:
    """Parse Harmonizome `attribute_list_entries.txt.gz` to extract Replogle-essential genes.

    The file is TSV with header `<idx>\tGene Perturbation\tGene Perturbation ID`. Column 1
    contains entries of the form `<numeric_id>_<HGNC_SYMBOL>_P1[P2]` for the genes Replogle
    CRISPRi-perturbed and that produced a K562-essential signature. NON-TARGETING controls
    and a small number of transcript-ID-suffixed rows are excluded — they do not contribute
    a unique HGNC symbol.

    Returns (essential_genes_sorted, unparsed_entries) so the caller can surface anomalies.
    """
    genes: set[str] = set()
    unparsed: list[str] = []
    with gzip.open(path, "rt") as f:
        next(f)  # header row
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            perturbation = parts[1].strip()
            if "NON-TARGETING" in perturbation:
                continue
            m = _PERTURBATION_RE.match(perturbation)
            if m:
                sym = m.group(1).strip()
                if sym and sym not in {"NA", "None", "-"}:
                    genes.add(sym)
            else:
                unparsed.append(perturbation)
    return sorted(genes), unparsed


def load_norman_105() -> list[str]:
    import scanpy as sc

    ad = sc.read_h5ad(NORMAN_H5AD)
    enc = ad.uns["perturbation_encoder"]
    return sorted(g for g in enc if g != "control" and "_" not in g)


def load_depmap_essentials() -> list[str]:
    import polars as pl

    df = pl.read_parquet(DEPMAP_PARQUET)
    ess = df.filter(pl.col("is_essential")).select("gene_symbol").to_series().to_list()
    return sorted(ess)


def write_gene_sets(replogle_essential: list[str]) -> dict[str, Any]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    norman_105 = load_norman_105()
    depmap_essentials = load_depmap_essentials()

    norman_set = set(norman_105)
    depmap_set = set(depmap_essentials)
    replogle_set = set(replogle_essential)

    replogle_in_norman = sorted(replogle_set & norman_set)
    depmap_in_norman = sorted(depmap_set & norman_set)
    both = sorted(set(replogle_in_norman) & set(depmap_in_norman))
    depmap_only_in_norman = sorted(set(depmap_in_norman) - set(replogle_in_norman))
    replogle_only_in_norman = sorted(set(replogle_in_norman) - set(depmap_in_norman))

    # Raw gene list (JSON + CSV)
    (PROCESSED_DIR / "replogle_essential_genes.json").write_text(
        json.dumps(
            {
                "n_genes": len(replogle_essential),
                "genes": replogle_essential,
                "source_slug": HARMONIZOME_SLUG,
                "source_page": SOURCE_PAGE,
            },
            indent=2,
        )
    )
    (PROCESSED_DIR / "replogle_essential_genes.csv").write_text(
        "gene_symbol\n" + "\n".join(replogle_essential) + "\n"
    )

    # Intersection JSON (matches Phase 2c schema for easy diffing)
    intersection_payload = {
        "source": "Harmonizome Replogle 2022 K562 Essential Perturb-seq Gene Perturbation Signatures",
        "source_url": SOURCE_PAGE,
        "source_slug": HARMONIZOME_SLUG,
        "n_replogle_essential_genes_unique": len(replogle_essential),
        "n_norman_105": len(norman_105),
        "n_norman_in_replogle_essential": len(replogle_in_norman),
        "norman_genes_in_replogle_essential": replogle_in_norman,
        "n_depmap_essential_in_norman": len(depmap_in_norman),
        "depmap_essential_in_norman": depmap_in_norman,
        "agreement_replogle_and_depmap_essential": both,
        "depmap_only_essential": depmap_only_in_norman,
        "replogle_only_essential": replogle_only_in_norman,
        "interpretation": (
            "DepMap and Replogle are CRISPR-Cas9 vs CRISPRi screens — different perturbation "
            "directions and different statistical thresholds. Agreement on the small intersection "
            "with Norman 105 is a strong join signal; differences reflect cell-line and assay "
            "specificity. This is the gene-set source for the V3C action-overlap audit."
        ),
    }
    (PROCESSED_DIR / "replogle_norman_intersection.json").write_text(
        json.dumps(intersection_payload, indent=2)
    )

    # Replogle-only essential — the clean Bucket-C set
    (PROCESSED_DIR / "replogle_only_essential_genes.json").write_text(
        json.dumps(
            {
                "n_replogle_only_in_norman_105": len(replogle_only_in_norman),
                "replogle_only_in_norman_105": replogle_only_in_norman,
                "definition": (
                    "Genes that are Replogle K562-essential AND in the Norman 105 single-gene "
                    "CRISPRa action universe AND NOT in the DepMap-Chronos `is_essential = True` "
                    "reward set. These are the held-out essentiality 'traps' that a safety-aware "
                    "controller should avoid if its DepMap prior generalised."
                ),
            },
            indent=2,
        )
    )

    return {
        "n_replogle_essential": len(replogle_essential),
        "n_norman_in_replogle_essential": len(replogle_in_norman),
        "n_depmap_essential_in_norman": len(depmap_in_norman),
        "n_replogle_only_in_norman_105": len(replogle_only_in_norman),
        "norman_in_replogle_essential": replogle_in_norman,
        "replogle_only_in_norman_105": replogle_only_in_norman,
    }


def write_source_metadata(provenance: dict[str, dict[str, Any]], summary: dict[str, Any]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Harmonizome 3.0 — Replogle et al., Cell 2022 K562 Essential Perturb-seq Gene Perturbation Signatures",
        "source_page": SOURCE_PAGE,
        "source_slug": HARMONIZOME_SLUG,
        "harmonizome_base_url": HARMONIZOME_BASE,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": provenance,
        "parsing_assumptions": [
            "attribute_list_entries.txt.gz is TSV with a header; column 1 holds entries of "
            "the form `<numeric_id>_<HGNC_SYMBOL>_P1[P2]` — one row per essentiality-screen "
            "perturbation. The Replogle-essential gene set is the set of unique HGNC symbols "
            "extracted from these rows.",
            "NON-TARGETING control rows are excluded (no HGNC symbol).",
            "A handful of ENST-transcript-suffixed rows (e.g. `5825_NSF_ENST...`) are also "
            "excluded — they refer to isoform-level perturbations that do not add a new HGNC "
            "symbol beyond those already covered by the standard `_P1P2` rows.",
            "Symbol case is preserved as published (HGNC upper-case for protein-coding genes).",
            "No symbol-normalisation alias resolution is applied; intersection with "
            "Norman 105 and DepMap is done on exact-match strings, matching Phase 2c.",
        ],
        "primary_citation": "Replogle JM et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. Cell 185, 2559-2575 (2022). DOI: 10.1016/j.cell.2022.05.013",
        "phase_2c_reference": "artifacts_v3/interpretation/v3b_phase2c_seed_escalation.md §6",
        "summary": summary,
    }
    (PROCESSED_DIR / "source_metadata.json").write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse the cached files under data/raw/replogle/ instead of downloading.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if cached files exist.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    LOG.info("Source: %s", SOURCE_PAGE)
    LOG.info("Slug:   %s", HARMONIZOME_SLUG)
    LOG.info("Raw dir:       %s", RAW_DIR)
    LOG.info("Processed dir: %s", PROCESSED_DIR)

    # Stage 2a: download (or skip)
    if args.skip_download:
        LOG.info("--skip-download: reusing cached files")
        provenance: dict[str, dict[str, Any]] = {}
        for fname in FILES:
            p = RAW_DIR / fname
            if p.exists():
                provenance[fname] = {
                    "url": f"{HARMONIZOME_BASE}/{fname}",
                    "status": "CACHED",
                    "bytes": p.stat().st_size,
                    "sha256": _sha256(p),
                }
            else:
                provenance[fname] = {"status": "MISSING_AND_SKIP_DOWNLOAD_SET"}
    else:
        provenance = download_files(force=args.force_download)

    # Bail cleanly if the essential-perturbation file isn't present.
    perturbation_file = RAW_DIR / "attribute_list_entries.txt.gz"
    if not perturbation_file.exists():
        LOG.error(
            "Required file %s is not present. Download failed or skipped without cache. "
            "Stage 3 cannot proceed; falling back to existing Phase 2c intersection JSON.",
            perturbation_file,
        )
        write_source_metadata(provenance, summary={"status": "DOWNLOAD_FAILED"})
        return 1

    # Stage 2b/3: parse and build gene-set artifacts
    LOG.info("Parsing essential-perturbation list: %s", perturbation_file)
    replogle_essential, unparsed_rows = parse_essential_perturbations(perturbation_file)
    LOG.info("    Replogle-essential genes (unique): %d", len(replogle_essential))
    LOG.info("    unparsed rows skipped: %d  (NON-TARGETING controls + ENST-suffixed entries)",
             len(unparsed_rows))

    summary = write_gene_sets(replogle_essential)
    write_source_metadata(provenance, summary=summary)

    LOG.info("\n=== Summary ===")
    LOG.info("Replogle-essential (unique):           %d", summary["n_replogle_essential"])
    LOG.info("Replogle ∩ Norman 105:                  %d  %s",
             summary["n_norman_in_replogle_essential"],
             summary["norman_in_replogle_essential"])
    LOG.info("DepMap-essential ∩ Norman 105:          %d", summary["n_depmap_essential_in_norman"])
    LOG.info("Replogle-only essential ∩ Norman 105:  %d  %s",
             summary["n_replogle_only_in_norman_105"],
             summary["replogle_only_in_norman_105"])
    LOG.info("Outputs written under %s", PROCESSED_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
