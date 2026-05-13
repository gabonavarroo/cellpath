"""Norman 2019 + DepMap K562 download helpers.

Owner: Agent A. See DATA.md §1 (Norman download) and §5 (DepMap).

Download strategy
-----------------
Norman 2019:
  Primary  : ``pertpy.dt.norman_2019()``  (requires jax; may fail on some systems)
  Fallback : scperturb Zenodo h5ad  (always works; curl + no Python deps beyond anndata)
  Last resort: GEO MTX tarball (manual; not auto-attempted)

DepMap Chronos:
  DepMap now uses a two-step signed-URL system. The helper downloads a manifest
  CSV, extracts the real Google Cloud Storage URL for CRISPRGeneEffect.csv, then
  fetches the actual file. This is forward-compatible with future DepMap releases.
"""

from __future__ import annotations

import csv
import hashlib
import io
import urllib.request
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Norman 2019
# ---------------------------------------------------------------------------

SCPERTURB_ZENODO_URL = (
    "https://zenodo.org/records/10044268/files/"
    "NormanWeissman2019_filtered.h5ad?download=1"
)


def download_norman(
    target_path: str | Path,
    source: str = "pertpy",
) -> Path:
    """Download Norman 2019 Perturb-seq (GSE133344) to disk.

    Parameters
    ----------
    target_path
        Where to write ``norman_2019.h5ad``. Parent directory is created if missing.
    source
        ``"pertpy"`` (primary) | ``"scperturb"`` (Zenodo fallback) | ``"geo"`` (last resort).
        If ``"pertpy"`` fails (e.g. jaxlib not present), call again with ``"scperturb"``.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if source == "pertpy":
        import pertpy as pt
        adata = pt.dt.norman_2019()
        adata.write_h5ad(str(target_path))

    elif source == "scperturb":
        urllib.request.urlretrieve(SCPERTURB_ZENODO_URL, target_path)
        _assert_h5ad_valid(target_path, min_obs=50_000)

    elif source == "geo":
        raise NotImplementedError(
            "GEO MTX fallback requires manual download. "
            "See DATA.md §1.1 for instructions."
        )
    else:
        raise ValueError(f"Unknown source {source!r}. Choose 'pertpy', 'scperturb', or 'geo'.")

    _assert_h5ad_valid(target_path, min_obs=50_000)
    return target_path.resolve()


def _assert_h5ad_valid(path: Path, min_obs: int) -> None:
    import anndata
    adata = anndata.read_h5ad(str(path))
    if adata.n_obs < min_obs:
        raise RuntimeError(
            f"Downloaded AnnData looks truncated: shape={adata.shape} "
            f"(expected ≥{min_obs:,} cells). Re-download."
        )


# ---------------------------------------------------------------------------
# DepMap Chronos
# ---------------------------------------------------------------------------

DEPMAP_MANIFEST_URL = (
    "https://depmap.org/portal/api/download/files?file_name=CRISPRGeneEffect.csv"
)


def download_depmap_k562(
    target_csv: str | Path,
    release: str = "latest",
) -> Path:
    """Download the DepMap CRISPRGeneEffect.csv via the two-step manifest API.

    DepMap uses signed Google Cloud Storage URLs that expire. The approach:
    1. Fetch the manifest CSV (lists all available files + their signed GCS URLs).
    2. Filter for ``filename == "CRISPRGeneEffect.csv"``, pick the first match
       (newest release first in the manifest).
    3. Download from the GCS URL.

    Parameters
    ----------
    target_csv
        Where to write ``depmap_chronos.csv``.
    release
        ``"latest"`` picks the first (newest) entry in the manifest. Pass an
        explicit release name (e.g. ``"DepMap Public 24Q2"``) to pin a version.

    Returns
    -------
    Path
        Absolute path to the written CSV.
    """
    target_csv = Path(target_csv)
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    # Step 1 — fetch manifest
    with urllib.request.urlopen(DEPMAP_MANIFEST_URL, timeout=30) as r:
        content = r.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    matches = [row for row in reader if row.get("filename") == "CRISPRGeneEffect.csv"]

    if not matches:
        raise RuntimeError(
            "CRISPRGeneEffect.csv not found in the DepMap manifest. "
            f"Manifest URL: {DEPMAP_MANIFEST_URL}"
        )

    if release == "latest":
        chosen = matches[0]
    else:
        candidates = [m for m in matches if m["release"] == release]
        if not candidates:
            available = [m["release"] for m in matches]
            raise ValueError(
                f"Release {release!r} not found. Available: {available}"
            )
        chosen = candidates[0]

    gcs_url = chosen["url"]
    print(f"  Downloading DepMap {chosen['release']} CRISPRGeneEffect.csv...")
    urllib.request.urlretrieve(gcs_url, target_csv)

    if not target_csv.exists() or target_csv.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"DepMap download appears empty or too small: {target_csv.stat().st_size} bytes."
        )
    return target_csv.resolve()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def verify_checksum(path: str | Path, expected_sha256: str) -> bool:
    """Verify a file's SHA-256 checksum.

    Parameters
    ----------
    path
        File to check.
    expected_sha256
        Hex-encoded expected digest.

    Returns
    -------
    bool
        True if checksums match. Never silently updates the digest.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256.lower()


def load_processed_anndata(path: str | Path) -> Any:
    """Load the preprocessed Norman h5ad and validate its schema.

    Parameters
    ----------
    path
        Path to ``data/processed/norman_hvg.h5ad``.

    Returns
    -------
    anndata.AnnData
        With ``layers["counts"]`` (raw integer UMIs) and ``X`` (log-normalised HVG).
    """
    import anndata

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed AnnData not found: {path}. "
            "Run preprocessing first (make pipeline or scripts/train_vae.py)."
        )
    adata = anndata.read_h5ad(str(path))
    _validate_processed_schema(adata, str(path))
    return adata


def _validate_processed_schema(adata: Any, path_hint: str = "") -> None:
    loc = f" in {path_hint}" if path_hint else ""
    if "counts" not in adata.layers:
        raise ValueError(
            f"Missing layers['counts']{loc}. "
            "scVI requires raw integer counts; was the preprocessing step skipped?"
        )
    for col in ["perturbation", "perturbation_idx"]:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing obs['{col}']{loc}.")
    if "perturbation_encoder" not in adata.uns:
        raise ValueError(f"Missing uns['perturbation_encoder']{loc}.")
