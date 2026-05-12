"""Norman 2019 preprocessing pipeline.

Owner: Agent A. See DATA.md §2 for biological justification of each step.

**Sacred rule (CLAUDE.md and DATA.md):** raw integer counts MUST be preserved in
``adata.layers["counts"]`` throughout the pipeline. scVI's NB likelihood consumes this layer;
normalizing in-place into ``adata.X`` and feeding that to scVI silently corrupts training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_preprocessing(
    cfg: Any,
    adata: Any | None = None,
    save: bool = True,
) -> Any:
    """End-to-end Norman preprocessing.

    Steps (in order; see DATA.md §2):

    1. Load raw AnnData (from disk or via ``download_norman``).
    2. ``sc.pp.filter_cells(adata, min_counts=cfg.data.min_counts)``.
    3. ``sc.pp.filter_genes(adata, min_cells=cfg.data.min_cells)``.
    4. **Preserve** raw integer counts: ``adata.layers["counts"] = adata.X.copy()``.
    5. HVG selection via ``sc.pp.highly_variable_genes(adata, flavor=cfg.data.hvg_flavor,
       n_top_genes=cfg.data.n_hvg, layer="counts")``.
    6. Normalize-total + log1p on ``adata.X`` (does **NOT** mutate ``layers["counts"]``).
    7. Subset to HVG.
    8. Encode ``perturbation_idx`` integer label (0 = ctrl).
    9. Save to ``cfg.paths.norman_processed_h5ad``.

    Parameters
    ----------
    cfg
        Hydra config (must have ``data``, ``paths``).
    adata
        Optional pre-loaded raw AnnData. If ``None``, loaded from
        ``cfg.paths.norman_raw_h5ad``.
    save
        If True, write to ``cfg.paths.norman_processed_h5ad``.

    Returns
    -------
    anndata.AnnData
        Preprocessed AnnData. Shape ``(N_cells_post_QC, cfg.data.n_hvg)``.
        ``layers["counts"]``: int32 raw counts (input to scVI).
        ``X``: float32 log1p-normalized (visualization / HVG selection).
        ``obs["perturbation_idx"]``: int32, 0 = ctrl.
        ``uns["perturbation_encoder"]``: dict[str, int].

    Raises
    ------
    NotImplementedError
        Agent A: implement the 9 steps above. Each step has a corresponding ``sc.pp.*`` call.

    Examples
    --------
    Expected post-preprocessing shapes::

        adata.shape                                  # (~111_000, 2_000)
        adata.layers["counts"].dtype                 # int32 or uint32
        adata.X.dtype                                # float32
        adata.X.max()                                # ≈ log(1 + 1e4) ≈ 9.21
        adata.obs["perturbation_idx"].nunique()      # ~107 (106 genes + ctrl)
    """
    raise NotImplementedError(
        "Agent A: implement 9-step preprocessing per DATA.md §2. "
        "MUST preserve raw counts in adata.layers['counts'] before normalize_total / log1p."
    )


def encode_perturbations(adata: Any) -> Any:
    """Add ``adata.obs["perturbation_idx"]`` and ``adata.uns["perturbation_encoder"]``.

    Parameters
    ----------
    adata
        Preprocessed AnnData with ``obs["perturbation"]`` as a string column where the
        control label is ``"control"`` or ``"ctrl"``.

    Returns
    -------
    anndata.AnnData
        Same object, mutated in-place. Returns it for chaining.

    Raises
    ------
    NotImplementedError
        Agent A: implement. 0 = ctrl, 1..N = sorted single-gene targets, combos get their own
        indices contiguous after singles (see DATA.md §2.7 and §3).
    """
    raise NotImplementedError(
        "Agent A: implement perturbation encoding. ctrl_idx == 0; single-gene next; combos last."
    )


def select_hvgs(adata: Any, n_top_genes: int = 2000, flavor: str = "seurat_v3") -> Any:
    """Compute HVGs on the raw counts layer.

    Parameters
    ----------
    adata
        AnnData with ``layers["counts"]`` containing integer UMIs.
    n_top_genes
        Number of HVGs to mark.
    flavor
        Must be ``"seurat_v3"`` to operate on raw counts. Other flavors operate on
        log-normalized data and would introduce circular bias.

    Returns
    -------
    anndata.AnnData
        AnnData with ``var["highly_variable"]`` populated.

    Raises
    ------
    NotImplementedError
        Agent A: ``sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=n_top_genes,
        layer="counts")``.
    ValueError
        If ``flavor != "seurat_v3"`` and ``layer="counts"`` is requested (incompatibility).
    """
    raise NotImplementedError(
        "Agent A: HVG selection via sc.pp.highly_variable_genes with layer='counts'."
    )


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

    Raises
    ------
    NotImplementedError
        Agent A: implement ``adata.write_h5ad(path)`` with parent dir creation.
    """
    raise NotImplementedError("Agent A: adata.write_h5ad(path) with parent dir creation.")
