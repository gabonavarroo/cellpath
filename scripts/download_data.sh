#!/usr/bin/env bash
# scripts/download_data.sh — download Norman 2019 + DepMap K562 data.
#
# Primary path:  pertpy.dt.norman_2019()  → data/raw/norman_2019.h5ad
# Fallback:      scperturb zenodo URL     → data/raw/norman_2019.h5ad
# Last resort:   GEO GSE133344 MTX tarball (slow, not auto-attempted)
#
# Also downloads the DepMap CRISPRGeneEffect CSV and extracts the K562 column.
#
# Usage:
#   bash scripts/download_data.sh
#   make data                   # convenience wrapper

set -euo pipefail

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Python resolver — prefer venv, honour $PYTHON env override, fallback system
# ---------------------------------------------------------------------------
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN="$PYTHON"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python"
fi
echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

DATA_RAW="data/raw"
DATA_PROCESSED="data/processed"
mkdir -p "$DATA_RAW" "$DATA_PROCESSED"

NORMAN_H5AD="$DATA_RAW/norman_2019.h5ad"
NORMAN_PART="$DATA_RAW/norman_2019.h5ad.part"
DEPMAP_CSV="$DATA_RAW/depmap_chronos.csv"
DEPMAP_PART="$DATA_RAW/depmap_chronos.csv.part"

# scperturb Zenodo record 10044268 — NormanWeissman 2019 filtered h5ad (~660 MB)
SCPERTURB_URL="https://zenodo.org/records/10044268/files/NormanWeissman2019_filtered.h5ad?download=1"

# ---------------------------------------------------------------------------
# 1. Norman 2019 — primary (pertpy), fallback (resumable curl from Zenodo)
# ---------------------------------------------------------------------------
if [ ! -f "$NORMAN_H5AD" ]; then
    echo "[1/2] Norman 2019 not found — attempting pertpy fetch…"
    if "$PYTHON_BIN" - <<PY
try:
    import pertpy as pt
    adata = pt.dt.norman_2019()
    adata.write_h5ad("$NORMAN_H5AD")
    print(f"OK: shape={adata.shape}")
except Exception as exc:
    print(f"pertpy fetch failed (non-fatal): {exc}")
    import sys; sys.exit(1)
PY
    then
        echo "  ✓ pertpy fetch succeeded → $NORMAN_H5AD"
    else
        echo "  ✗ pertpy unavailable — attempting resumable curl from Zenodo…"
        echo "  URL: $SCPERTURB_URL"
        if curl -fL \
                --retry 8 \
                --retry-all-errors \
                --retry-delay 10 \
                --connect-timeout 30 \
                --continue-at - \
                -o "$NORMAN_PART" \
                "$SCPERTURB_URL"; then
            mv "$NORMAN_PART" "$NORMAN_H5AD"
            echo "  ✓ curl fallback succeeded → $NORMAN_H5AD"
        else
            rm -f "$NORMAN_PART"
            echo "  ✗ curl fallback also failed. For manual download:"
            echo "    curl -L -o $NORMAN_H5AD '$SCPERTURB_URL'"
            echo "  Or download GEO GSE133344 MTX tarball manually."
            exit 2
        fi
    fi
else
    echo "[1/2] $NORMAN_H5AD already exists; skipping download."
fi

# Validate h5ad (cheap — just opens the file and checks n_obs)
echo "  Validating Norman h5ad…"
if ! "$PYTHON_BIN" - <<PY
import anndata, sys
try:
    ad = anndata.read_h5ad("$NORMAN_H5AD")
    assert ad.n_obs > 50_000, f"Suspiciously few cells: {ad.n_obs}"
    print(f"  ✓ Norman h5ad valid: shape={ad.shape}, obs_keys={list(ad.obs.columns)[:5]}…")
except Exception as exc:
    print(f"  ✗ Validation error: {exc}", file=sys.stderr)
    sys.exit(1)
PY
then
    echo "  ✗ h5ad validation failed — deleting corrupt or incomplete file."
    rm -f "$NORMAN_H5AD"
    exit 3
fi

# ---------------------------------------------------------------------------
# 2. DepMap CRISPRGeneEffect — resolve signed URL via manifest, then curl
# ---------------------------------------------------------------------------
if [ ! -f "$DEPMAP_CSV" ]; then
    echo "[2/2] DepMap CRISPRGeneEffect.csv not found — resolving download URL via manifest…"

    # DepMap uses a two-step API: fetch a manifest CSV listing all files with
    # their signed GCS URLs, then download from the real URL in the manifest.
    DEPMAP_MANIFEST="https://depmap.org/portal/api/download/files?file_name=CRISPRGeneEffect.csv"

    ACTUAL_URL=$("$PYTHON_BIN" - <<PY
import urllib.request, csv, io, sys
manifest_url = "$DEPMAP_MANIFEST"
try:
    with urllib.request.urlopen(manifest_url, timeout=30) as r:
        content = r.read().decode("utf-8")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
reader = csv.DictReader(io.StringIO(content))
matches = [row for row in reader if row.get("filename") == "CRISPRGeneEffect.csv"]
if not matches:
    print("ERROR: CRISPRGeneEffect.csv not found in manifest", file=sys.stderr)
    sys.exit(1)
# Manifest is sorted newest-first; pick first match (latest release)
print(matches[0]["url"])
PY)

    if [ -z "$ACTUAL_URL" ]; then
        echo "  ✗ Failed to resolve DepMap download URL."
        exit 4
    fi

    DEPMAP_RELEASE=$("$PYTHON_BIN" - <<PY
import urllib.request, csv, io, sys
manifest_url = "$DEPMAP_MANIFEST"
try:
    with urllib.request.urlopen(manifest_url, timeout=30) as r:
        content = r.read().decode("utf-8")
except Exception as e:
    print("unknown", file=sys.stderr)
    sys.exit(0)
matches = [row for row in csv.DictReader(io.StringIO(content)) if row.get("filename") == "CRISPRGeneEffect.csv"]
print(matches[0].get("release", "unknown") if matches else "unknown")
PY)

    echo "  Downloading from release: ${DEPMAP_RELEASE}…"
    echo "  URL: $ACTUAL_URL"

    if curl -fL \
            --retry 8 \
            --retry-all-errors \
            --retry-delay 10 \
            --connect-timeout 30 \
            --continue-at - \
            -o "$DEPMAP_PART" \
            "$ACTUAL_URL"; then
        mv "$DEPMAP_PART" "$DEPMAP_CSV"
        echo "  ✓ DepMap fetch succeeded → $DEPMAP_CSV"
    else
        rm -f "$DEPMAP_PART"
        echo "  ✗ DepMap download failed."
        exit 4
    fi
else
    echo "[2/2] $DEPMAP_CSV already exists; skipping download."
fi

# Extract K562 row to parquet
echo "  Extracting K562 (ACH-000551) Chronos scores…"
"$PYTHON_BIN" - <<PY
import polars as pl, sys
try:
    df = pl.read_csv("$DEPMAP_CSV", truncate_ragged_lines=True)
except Exception as e:
    print(f"  ✗ Failed to read DepMap CSV: {e}", file=sys.stderr)
    sys.exit(1)
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
    print(f"  ✗ K562 row (ACH-000551) not found in column {id_col!r}", file=sys.stderr)
    sys.exit(1)
genes = [c for c in df.columns if c != id_col]
long = pl.DataFrame({
    "gene_symbol": [c.split(" ")[0] for c in genes],
    "chronos": k562.select(genes).row(0),
}).with_columns(is_essential=pl.col("chronos") < -0.5)
long.write_parquet("$DATA_PROCESSED/depmap_k562_chronos.parquet")
print(f"  ✓ DepMap K562: {long.height} genes; {long['is_essential'].sum()} essential (Chronos < -0.5).")
PY

echo ""
echo "=== Data download complete ==="
echo "  Norman:   $NORMAN_H5AD"
echo "  DepMap:   $DEPMAP_CSV → $DATA_PROCESSED/depmap_k562_chronos.parquet"
echo ""
echo "Next: make pipeline   (or run scripts/train_vae.py individually)"
