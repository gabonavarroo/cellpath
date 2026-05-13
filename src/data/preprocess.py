"""Norman 2019 preprocessing pipeline.

Owner: Agent A. See DATA.md §2 for biological justification of each step.

**Sacred rule (CLAUDE.md + DATA.md):** raw integer counts MUST be preserved in
``adata.layers["counts"]`` throughout the pipeline. scVI's NB likelihood consumes
this layer; normalising in-place into ``adata.X`` and feeding that to scVI silently
corrupts training.

Dataset notes (scperturb Zenodo version, verified 2026-05-12)
--------------------------------------------------------------
- ``adata.X``: float32 sparse matrix of raw UMI counts (integers stored as floats).
  No pre-normalisation in the scperturb build.
- Control label: ``"control"`` (not ``"ctrl"`` as stated in older DATA.md).
- Combo separator: ``"_"`` (e.g. ``"KLF1_BAK1"``), not ``"+"`` as in some docs.
  Reliably detected via the ``nperts`` column (0=ctrl, 1=single, 2=combo).
- Gene count: 33,694 before HVG filtering (not 19,018 — scperturb includes all genes).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def run_preprocessing(
    cfg: Any,
    adata: Any | None = None,
    save: bool = True,
) -> Any:
    """End-to-end Norman preprocessing.

    Steps (DATA.md §2):
    1. Load raw AnnData.
    2. Filter cells: ``sc.pp.filter_cells(min_counts=cfg.data.min_counts)``.
    3. Filter genes: ``sc.pp.filter_genes(min_cells=cfg.data.min_cells)``.
    4. **Preserve raw counts**: ``adata.layers["counts"] = adata.X.copy()`` (int32).
    5. HVG selection: ``sc.pp.highly_variable_genes(flavor="seurat_v3", layer="counts")``.
    6. Normalise + log1p on ``adata.X`` only (layers["counts"] untouched).
    7. Subset to HVGs.
    8. Encode ``perturbation_idx`` (0=ctrl, 1..N=single genes, N+1..=combos).
    9. Save to ``cfg.paths.norman_processed_h5ad``.

    Parameters
    ----------
    cfg
        Hydra config (must have ``data``, ``paths``).
    adata
        Optional pre-loaded raw AnnData. If ``None``, loaded from
        ``cfg.paths.norman_raw_h5ad``.
    save
        Write output to disk.

    Returns
    -------
    anndata.AnnData
        Shape ``(N_cells_post_QC, cfg.data.n_hvg)``.
    """
    import anndata as ad
    import scanpy as sc

    # ------------------------------------------------------------------ #
    # Step 1: load
    # ------------------------------------------------------------------ #
    if adata is None:
        raw_path = Path(cfg.paths.norman_raw_h5ad)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw Norman h5ad not found at {raw_path}. Run `make data` first."
            )
        log.info("Loading raw AnnData from %s", raw_path)
        adata = ad.read_h5ad(str(raw_path))

    log.info("Raw shape: %s", adata.shape)

    # ------------------------------------------------------------------ #
    # Step 2: filter cells
    # ------------------------------------------------------------------ #
    sc.pp.filter_cells(adata, min_counts=int(cfg.data.min_counts))
    log.info("After filter_cells (min_counts=%d): %d cells", cfg.data.min_counts, adata.n_obs)

    # ------------------------------------------------------------------ #
    # Step 3: filter genes
    # ------------------------------------------------------------------ #
    sc.pp.filter_genes(adata, min_cells=int(cfg.data.min_cells))
    log.info("After filter_genes (min_cells=%d): %d genes", cfg.data.min_cells, adata.n_vars)

    # ------------------------------------------------------------------ #
    # Steps 4-7: memory-efficient ordering for large datasets.
    #
    # Naive order (layers["counts"] copy on full matrix) peaks at ~4 GB on an
    # 8 GB Mac (two copies of 111k × 17k). Instead:
    #   4. Identify HVGs on a raw-X subsample (X is still raw counts here)
    #   5. Subset adata to the 2000 HVGs immediately → 111k × 2000 (small)
    #   6. THEN copy layers["counts"] from the small HVG matrix (only ~250 MB)
    #   7. Normalise + log1p on the small matrix
    #
    # scVI only needs layers["counts"] for the HVG subset, so this is correct.
    # ------------------------------------------------------------------ #
    import gc
    import scipy.sparse as sp

    # ------------------------------------------------------------------ #
    # Step 4: HVG selection on raw X (X is still raw counts at this point)
    # seurat_v3 operates directly on raw counts (correct for NB pipeline).
    # Subsample cells to bound the LOESS fit time; variance rankings are
    # stable well before 20k cells.
    # ------------------------------------------------------------------ #
    HVG_SUBSAMPLE = 20_000
    if adata.n_obs > HVG_SUBSAMPLE:
        log.info(
            "Large dataset (%d cells) — HVGs on %d-cell subsample.",
            adata.n_obs, HVG_SUBSAMPLE,
        )
        rng_hvg = np.random.default_rng(42)
        sub_idx = rng_hvg.choice(adata.n_obs, HVG_SUBSAMPLE, replace=False)
        adata_sub = adata[sub_idx].copy()
        sc.pp.highly_variable_genes(
            adata_sub,
            flavor=str(cfg.data.hvg_flavor),
            n_top_genes=int(cfg.data.n_hvg),
        )
        adata.var["highly_variable"] = adata_sub.var["highly_variable"]
        del adata_sub
        gc.collect()
    else:
        sc.pp.highly_variable_genes(
            adata,
            flavor=str(cfg.data.hvg_flavor),
            n_top_genes=int(cfg.data.n_hvg),
        )
    n_hvg = adata.var["highly_variable"].sum()
    log.info("HVG selection: %d genes marked (requested %d)", n_hvg, cfg.data.n_hvg)

    # ------------------------------------------------------------------ #
    # Step 5: subset to HVGs immediately — free the large full-gene matrix
    # ------------------------------------------------------------------ #
    adata = adata[:, adata.var["highly_variable"]].copy()
    gc.collect()
    log.info("Post-HVG subset shape: %s", adata.shape)

    # ------------------------------------------------------------------ #
    # Step 6: preserve raw counts from the HVG subset (small: ~250 MB)
    # ------------------------------------------------------------------ #
    if sp.issparse(adata.X):
        adata.layers["counts"] = adata.X.copy().astype("int32")
    else:
        adata.layers["counts"] = adata.X.copy().astype("int32")

    sample = adata.layers["counts"][:10, :10]
    if sp.issparse(sample):
        sample = sample.toarray()
    assert (sample >= 0).all(), "Counts contain negative values."
    assert np.allclose(sample, np.round(sample)), "Counts contain non-integers."

    # ------------------------------------------------------------------ #
    # Step 7: normalise + log1p on the small HVG matrix
    # ------------------------------------------------------------------ #
    sc.pp.normalize_total(adata, target_sum=float(cfg.data.normalize_total))
    if bool(cfg.data.log_transform):
        sc.pp.log1p(adata)

    # ------------------------------------------------------------------ #
    # Step 8: encode perturbation indices
    # ------------------------------------------------------------------ #
    adata = encode_perturbations(adata)

    n_ctrl = (adata.obs["perturbation_idx"] == 0).sum()
    n_genes = adata.uns["noop_idx"]
    log.info(
        "Perturbation encoding: %d ctrl cells, %d single-gene targets, noop_idx=%d",
        n_ctrl, n_genes, n_genes,
    )

    # ------------------------------------------------------------------ #
    # Step 9: save
    # ------------------------------------------------------------------ #
    if save:
        save_processed(adata, cfg.paths.norman_processed_h5ad)

    return adata


def encode_perturbations(adata: Any) -> Any:
    """Add ``perturbation_idx`` and ``perturbation_encoder`` to the AnnData.

    Encoding:
    - 0  : control (``"control"`` or ``"ctrl"``)
    - 1..N_single : single-gene CRISPRa perturbations (sorted alphabetically)
    - N_single+1..M : dual-gene combinations (sorted)

    The combo separator in the scperturb Norman build is ``"_"`` (e.g. ``"KLF1_BAK1"``),
    detected via the ``nperts`` column when available. The ``noop_idx`` stored in
    ``adata.uns`` equals ``N_single`` — this is the RL NO-OP action index.

    Parameters
    ----------
    adata
        AnnData with ``obs["perturbation"]`` (str) column.

    Returns
    -------
    anndata.AnnData
        Same object mutated in-place; returned for chaining.
    """
    if "perturbation" not in adata.obs.columns:
        raise ValueError(
            "obs['perturbation'] column not found. "
            f"Available obs columns: {list(adata.obs.columns)}"
        )

    unique_perts = adata.obs["perturbation"].unique().tolist()

    # Identify control label
    ctrl_label: str | None = None
    for candidate in ("control", "ctrl", "non-targeting", "NT"):
        if candidate in unique_perts:
            ctrl_label = candidate
            break
    if ctrl_label is None:
        raise ValueError(
            f"No control label found in perturbations. "
            f"Checked: control, ctrl, non-targeting, NT. "
            f"Sample values: {unique_perts[:10]}"
        )

    # Separate singles from combos using the nperts column (most reliable)
    # or the "_" separator (Norman scperturb convention).
    if "nperts" in adata.obs.columns:
        # Build a set of combo perturbation strings from the nperts column
        combo_mask = adata.obs["nperts"] == 2
        combo_labels = set(adata.obs.loc[combo_mask, "perturbation"].unique().tolist())
    else:
        # Fallback: combo labels contain "_" (Norman convention) and aren't ctrl
        combo_labels = {
            p for p in unique_perts
            if "_" in str(p) and p != ctrl_label
        }

    single_labels = sorted(
        p for p in unique_perts
        if p != ctrl_label and p not in combo_labels
    )
    combo_labels_sorted = sorted(combo_labels)

    # Build encoder: ctrl=0, singles=1..N, combos=N+1..M
    encoder: dict[str, int] = {ctrl_label: 0}
    for i, p in enumerate(single_labels, start=1):
        encoder[p] = i
    for i, p in enumerate(combo_labels_sorted, start=len(single_labels) + 1):
        encoder[p] = i

    adata.obs["perturbation_idx"] = (
        adata.obs["perturbation"].map(encoder).astype("int32")
    )
    adata.uns["perturbation_encoder"] = encoder
    adata.uns["ctrl_label"] = ctrl_label
    # noop_idx = number of single-gene perturbations = RL Discrete action space size - 1
    adata.uns["noop_idx"] = len(single_labels)

    return adata


def select_hvgs(adata: Any, n_top_genes: int = 2000, flavor: str = "seurat_v3") -> Any:
    """Compute HVGs on the raw counts layer.

    Parameters
    ----------
    adata
        AnnData with ``layers["counts"]`` containing integer UMIs.
    n_top_genes
        Number of HVGs to mark.
    flavor
        Must be ``"seurat_v3"`` to operate correctly on raw counts.
    """
    import scanpy as sc

    if flavor != "seurat_v3":
        raise ValueError(
            f"flavor={flavor!r} — only 'seurat_v3' correctly operates on raw counts. "
            "See DATA.md §2.4."
        )
    sc.pp.highly_variable_genes(
        adata, flavor=flavor, n_top_genes=n_top_genes, layer="counts"
    )
    return adata


def save_processed(adata: Any, path: str | Path) -> Path:
    """Persist the preprocessed AnnData to disk.

    Parameters
    ----------
    adata
        Output of :func:`run_preprocessing`.
    path
        Destination ``.h5ad``.

    Returns
    -------
    Path
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(path))
    log.info("Saved preprocessed AnnData → %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path
