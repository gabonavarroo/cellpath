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
    echo "[2/2] DepMap Chronos CSV not found — downloading version $DEPMAP_VERSION…"
    DEPMAP_URL="https://depmap.org/portal/api/download/files/CRISPRGeneEffect.csv?release=${DEPMAP_VERSION}"
    if ! curl -L -o "$DEPMAP_CSV" "$DEPMAP_URL"; then
        echo "  ✗ DepMap download failed; please visit https://depmap.org/portal/download/ and place the CSV at $DEPMAP_CSV"
        exit 3
    fi
    echo "  ✓ DepMap fetch succeeded → $DEPMAP_CSV"
else
    echo "[2/2] $DEPMAP_CSV already exists; skipping."
fi

# Extract K562 row into a parquet
python - <<PY
import polars as pl
df = pl.read_csv("$DEPMAP_CSV")
# DepMap rows are cell lines; first column is "ModelID" or "DepMap_ID".
# K562's DepMap ID is ACH-000551 (verify per release notes).
id_col = "ModelID" if "ModelID" in df.columns else "DepMap_ID"
k562 = df.filter(pl.col(id_col) == "ACH-000551")
if k562.height == 0:
    raise SystemExit(f"K562 row (ACH-000551) not found in {id_col!r}")
genes = [c for c in df.columns if c != id_col]
long = pl.DataFrame({
    "gene_symbol": [c.split(" ")[0] for c in genes],
    "chronos": k562.select(genes).row(0),
}).with_columns(is_essential=pl.col("chronos") < -0.5)
long.write_parquet("$DATA_PROCESSED/depmap_k562_chronos.parquet")
print(f"  DepMap K562: {long.height} genes; {long['is_essential'].sum()} essential.")
PY

echo ""
echo "=== Data download complete ==="
echo "  Norman:   $NORMAN_H5AD"
echo "  DepMap:   $DEPMAP_CSV → $DATA_PROCESSED/depmap_k562_chronos.parquet"
echo ""
echo "Next: make pipeline   (or run scripts/train_vae.py individually)"
