"""Norman 2019 + DepMap K562 download helpers.

Owner: Agent A. See DATA.md §1 (Norman download) and §5 (DepMap).

Primary path uses ``pertpy.dt.norman_2019``; ``scperturb`` and GEO MTX paths are documented
fallbacks. Checksums are pinned in ``scripts/download_data.sh`` and verified after each
download — checksum mismatch fails loudly, never silently updated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
        One of ``"pertpy"`` | ``"scperturb"`` | ``"geo"``. Use ``"pertpy"`` unless it fails.

    Returns
    -------
    Path
        Absolute path to the written file.

    Raises
    ------
    NotImplementedError
        Agent A: implement the three branches. For ``"pertpy"``: call
        ``pertpy.dt.norman_2019()`` and ``adata.write_h5ad(target_path)``. For ``"scperturb"``:
        ``urllib`` download from the pinned Zenodo URL. For ``"geo"``: tar + per-sample MTX
        stitching (last resort).

    Notes
    -----
    The downloaded file's expected shape is approximately ``(111_255, 19_018)``. Verify
    with ``anndata.read_h5ad(...).shape`` after download.
    """
    raise NotImplementedError(
        "Agent A: implement Norman 2019 download. Primary: pertpy.dt.norman_2019(). "
        "Fallbacks: scperturb Zenodo URL + GEO MTX. Verify shape ≈ (111255, 19018)."
    )


def download_depmap_k562(
    target_csv: str | Path,
    release: str = "24Q2",
) -> Path:
    """Download the DepMap Chronos table (used to extract K562 essentiality scores).

    Parameters
    ----------
    target_csv
        Where to write ``depmap_chronos.csv``.
    release
        DepMap release identifier (default ``"24Q2"``).

    Returns
    -------
    Path
        Absolute path to the written CSV.

    Raises
    ------
    NotImplementedError
        Agent A: implement using the DepMap public download API.
    """
    raise NotImplementedError(
        "Agent A: implement DepMap Chronos CSV download. See DATA.md §5.1 for the URL pattern."
    )


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
        True if the checksum matches.

    Raises
    ------
    NotImplementedError
        Agent A: implement with ``hashlib.sha256`` streaming over 4 MB chunks.
    FileNotFoundError
        If ``path`` does not exist.
    """
    raise NotImplementedError(
        "Agent A: streaming sha256 over 4 MB chunks. Return bool, never auto-update the digest."
    )


def load_processed_anndata(path: str | Path) -> Any:
    """Load the preprocessed Norman h5ad with schema validation.

    Parameters
    ----------
    path
        Path to ``data/processed/norman_hvg.h5ad``.

    Returns
    -------
    anndata.AnnData
        AnnData with ``layers["counts"]`` (raw integer UMIs) and ``X`` (log-normalized HVG).

    Raises
    ------
    NotImplementedError
        Agent A: implement and verify schema (layers, obs keys, var keys, uns keys).
    FileNotFoundError
        If ``path`` doesn't exist.
    ValueError
        If required schema fields (``layers["counts"]``, ``obs["perturbation_idx"]``, etc.) are
        missing.
    """
    raise NotImplementedError(
        "Agent A: anndata.read_h5ad + schema check. Required fields per DATA.md §2.8."
    )
