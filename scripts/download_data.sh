#!/usr/bin/env bash
# scripts/download_data.sh — download Norman 2019 + DepMap K562 data.
#
# Primary path:  pertpy.dt.norman_2019()  → data/raw/norman_2019.h5ad
# Fallback:      scperturb zenodo URL     → data/raw/norman_2019.h5ad
# Last resort:   GEO GSE133344 MTX tarball (slow, not auto-attempted)
#
# Also downloads the DepMap Chronos CSV and extracts the K562 column.
#
# Usage:
#   bash scripts/download_data.sh
#   make data                   # convenience wrapper

set -euo pipefail

cd "$(dirname "$0")/.."

DATA_RAW="data/raw"
DATA_PROCESSED="data/processed"
mkdir -p "$DATA_RAW" "$DATA_PROCESSED"

NORMAN_H5AD="$DATA_RAW/norman_2019.h5ad"
DEPMAP_CSV="$DATA_RAW/depmap_chronos.csv"
DEPMAP_VERSION="${DEPMAP_VERSION:-24Q2}"

# ---------------------------------------------------------------------------
# 1. Norman 2019 — primary (pertpy)
# ---------------------------------------------------------------------------
if [ ! -f "$NORMAN_H5AD" ]; then
    echo "[1/2] Norman 2019 not found — attempting pertpy fetch…"
    if python - <<PY
try:
    import pertpy as pt
    adata = pt.dt.norman_2019()
    adata.write_h5ad("$NORMAN_H5AD")
    print(f"OK: shape={adata.shape}")
except Exception as exc:
    print(f"pertpy fetch failed: {exc}")
    raise SystemExit(1)
PY
    then
        echo "  ✓ pertpy fetch succeeded → $NORMAN_H5AD"
    else
        echo "  ✗ pertpy fetch failed; attempting scperturb fallback…"
        SCPERTURB_URL="https://zenodo.org/records/10044268/files/NormanWeissman2019_filtered.h5ad?download=1"
        curl -L -o "$NORMAN_H5AD" "$SCPERTURB_URL"
        if [ ! -s "$NORMAN_H5AD" ]; then
            echo "  ✗ scperturb fallback also failed. Resort to GEO GSE133344 manually."
            exit 2
        fi
        echo "  ✓ scperturb fetch succeeded → $NORMAN_H5AD"
    fi
else
    echo "[1/2] $NORMAN_H5AD already exists; skipping."
fi

# Sanity check (cheap)
python - <<PY
import anndata
ad = anndata.read_h5ad("$NORMAN_H5AD")
assert ad.n_obs > 50_000, f"Suspiciously small: {ad.shape}"
print(f"  Norman h5ad: shape={ad.shape}, obs_keys={list(ad.obs.columns)[:5]}…")
PY

# ---------------------------------------------------------------------------
# 2. DepMap Chronos
# ---------------------------------------------------------------------------
if [ ! -f "$DEPMAP_CSV" ]; then
    echo "[2/2] DepMap Chronos CSV not found — resolving download URL via manifest..."
    # DepMap now uses a two-step API: first fetch a manifest CSV that lists all files
    # with their signed GCS URLs, then download from the real URL in that manifest.
    ACTUAL_URL=$(python - <<PY
import urllib.request, csv, io, sys
manifest_url = "https://depmap.org/portal/api/download/files?file_name=CRISPRGeneEffect.csv"
try:
    with urllib.request.urlopen(manifest_url, timeout=30) as r:
        content = r.read().decode("utf-8")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
reader = csv.DictReader(io.StringIO(content))
matches = [row for row in reader if row["filename"] == "CRISPRGeneEffect.csv"]
if not matches:
    print("ERROR: CRISPRGeneEffect.csv not found in manifest", file=sys.stderr)
    sys.exit(1)
# Pick the first match (latest release — manifest is sorted newest-first)
print(matches[0]["url"])
PY)
    if [ $? -ne 0 ] || [ -z "$ACTUAL_URL" ]; then
        echo "  ✗ Failed to resolve DepMap download URL. Check your internet connection."
        exit 3
    fi
    DEPMAP_RELEASE=$(python - <<PY
import urllib.request, csv, io
with urllib.request.urlopen("https://depmap.org/portal/api/download/files?file_name=CRISPRGeneEffect.csv", timeout=30) as r:
    content = r.read().decode("utf-8")
matches = [row for row in csv.DictReader(io.StringIO(content)) if row["filename"] == "CRISPRGeneEffect.csv"]
print(matches[0]["release"] if matches else "unknown")
PY)
    echo "  Downloading from release: ${DEPMAP_RELEASE}..."
    if ! curl -L -o "$DEPMAP_CSV" "$ACTUAL_URL"; then
        echo "  ✗ DepMap download failed."
        exit 3
    fi
    echo "  ✓ DepMap fetch succeeded → $DEPMAP_CSV"
else
    echo "[2/2] $DEPMAP_CSV already exists — extracting K562 row..."
    python - <<PY
import polars as pl, sys
df = pl.read_csv("$DEPMAP_CSV", truncate_ragged_lines=True)
# DepMap column name for cell-line IDs varies by release:
#   older releases: "ModelID" or "DepMap_ID"
#   newer releases (26Q1+): first column has no name -> polars reads it as ""
id_col = next(
    (c for c in ["ModelID", "DepMap_ID", ""] if c in df.columns),
    None,
)
if id_col is None:
    print(f"  ✗ Cannot find cell-line ID column. Columns: {df.columns[:5]}", file=sys.stderr)
    sys.exit(1)
k562 = df.filter(pl.col(id_col) == "ACH-000551")
if k562.height == 0:
    raise SystemExit(f"K562 row (ACH-000551) not found in column {id_col!r}")
genes = [c for c in df.columns if c != id_col]
long = pl.DataFrame({
    "gene_symbol": [c.split(" ")[0] for c in genes],
    "chronos": k562.select(genes).row(0),
}).with_columns(is_essential=pl.col("chronos") < -0.5)
long.write_parquet("$DATA_PROCESSED/depmap_k562_chronos.parquet")
print(f"  DepMap K562: {long.height} genes; {long['is_essential'].sum()} essential.")
PY
fi

echo ""
echo "=== Data download complete ==="
echo "  Norman:   $NORMAN_H5AD"
echo "  DepMap:   $DEPMAP_CSV → $DATA_PROCESSED/depmap_k562_chronos.parquet"
echo ""
echo "Next: make pipeline   (or run scripts/train_vae.py individually)"
