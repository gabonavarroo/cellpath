"""scVI VAE wrapper — train + compute reference centroid + ε_success.

Owner: Agent A. See ARCHITECTURE.md Concepts 1 + 2.

**Sacred rules:**
- ``setup_anndata`` MUST be called with ``layer="counts"`` (raw integer UMIs).
- Saving uses the official scVI API (``model.save() / SCVI.load()``).  Manual ``state_dict``
  serialization is forbidden (CLAUDE.md sacred rule #1, ARCHITECTURE.md D14).
- ``z_reference_centroid`` is computed once after training and persisted to disk.
- ``epsilon_success`` is data-driven: percentile of ``||z_ctrl - z_ref||_2`` distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_vae(
    cfg: Any,
    adata: Any | None = None,
) -> tuple[Any, Any]:
    """Train scVI on the preprocessed Norman AnnData and persist all artifacts.

    Workflow
    --------
    1. ``scvi.model.SCVI.setup_anndata(adata, layer="counts", **cfg.vae.setup)``.
    2. Instantiate ``SCVI(adata, n_latent=cfg.vae.n_latent,
       gene_likelihood=cfg.vae.gene_likelihood, n_layers=cfg.vae.n_layers,
       dropout_rate=cfg.vae.dropout_rate)``.
    3. ``model.train(max_epochs=cfg.vae.max_epochs, early_stopping=cfg.vae.early_stopping, ...)``.
    4. ``model.save(cfg.paths.vae_model_dir, overwrite=cfg.vae.save_overwrite)``  — official API.
    5. ``Z = model.get_latent_representation()``; attach as ``adata.obsm["X_scVI"]``.
    6. Write ``cfg.paths.vae_latents_h5ad`` (full adata + obsm).
    7. Compute ``z_reference_centroid`` from control cells; save to ``cfg.paths.vae_z_reference_centroid``.
    8. Compute ``epsilon_success`` (default 90th percentile); save to ``cfg.paths.vae_epsilon_success_json``.
    9. Write ``gene_vocab.json`` with the perturbation-index → gene-symbol mapping.

    Parameters
    ----------
    cfg
        Hydra config (must have ``vae``, ``paths``, ``seed``).
    adata
        Preprocessed AnnData. If ``None``, loaded from ``cfg.paths.norman_processed_h5ad``.

    Returns
    -------
    tuple[scvi.model.SCVI, anndata.AnnData]
        ``(model, adata_with_latents)``.

    Raises
    ------
    NotImplementedError
        Agent A: implement steps 1–9. Workflow is documented above and in PHASES.md Phase 1.

    Examples
    --------
    Expected post-training artifacts (Contract 1)::

        artifacts/vae/model/         <- scVI's directory
        artifacts/vae/latents.h5ad
        artifacts/vae/gene_vocab.json
        artifacts/vae/z_reference_centroid.npy   # shape (32,)
        artifacts/vae/epsilon_success.json       # {"percentile": 90, "value": ...}
    """
    raise NotImplementedError(
        "Agent A: implement the 9-step VAE training + artifact pipeline. "
        "Use scvi.model.SCVI.setup_anndata(layer='counts') and model.save() — NEVER raw state_dict."
    )


def compute_z_reference_centroid(adata: Any, latent_key: str = "X_scVI") -> Any:
    """Compute the unperturbed-K562 NT-guide centroid in latent space.

    Parameters
    ----------
    adata
        AnnData with ``adata.obsm[latent_key]`` populated.
    latent_key
        Key into ``adata.obsm``. Default ``"X_scVI"``.

    Returns
    -------
    np.ndarray
        Shape ``(n_latent,)`` float32. The mean latent vector of cells with
        ``obs["perturbation_idx"] == 0``.

    Raises
    ------
    NotImplementedError
        Agent A: trivial mean over the control mask.
    """
    raise NotImplementedError(
        "Agent A: mean(adata.obsm[latent_key][adata.obs['perturbation_idx'] == 0], axis=0)."
    )


def compute_epsilon_success(
    Z_ctrl: Any,
    z_ref: Any,
    percentile: float = 90.0,
) -> dict[str, Any]:
    """Data-driven success threshold for the RL env.

    ε_success = percentile(||z_ctrl_i − z_ref||_2, percentile). Default 90% means a state
    closer to ``z_ref`` than 90% of unperturbed cells counts as a success — calibrated to the
    actual geometry of the learned latent rather than a hardcoded constant.

    Parameters
    ----------
    Z_ctrl
        Control latent vectors, shape ``(N_ctrl, n_latent)``.
    z_ref
        Reference centroid, shape ``(n_latent,)``.
    percentile
        Percentile to use (default 90; 80/95/99 supported).

    Returns
    -------
    dict
        ``{"percentile": float, "value": float, "n_ctrl_cells": int, "method": "L2_distance"}``
        — written verbatim to ``artifacts/vae/epsilon_success.json``.

    Raises
    ------
    NotImplementedError
        Agent A: numpy percentile over L2 distances.

    Examples
    --------
    >>> out = compute_epsilon_success(Z_ctrl, z_ref, percentile=90)
    >>> 0.1 < out["value"] < 10.0   # sanity bounds in scVI 32-dim latent
    True
    """
    raise NotImplementedError(
        "Agent A: numpy percentile over ||Z_ctrl - z_ref||_2. Return the dict schema for JSON."
    )


def load_vae_model(model_dir: str | Path, adata: Any) -> Any:
    """Load a trained scVI model via the official API.

    Parameters
    ----------
    model_dir
        Directory previously written by ``model.save(...)``.
    adata
        AnnData with the same schema as the training data.

    Returns
    -------
    scvi.model.SCVI
        Loaded model, ready for ``get_latent_representation``.

    Raises
    ------
    NotImplementedError
        Agent A: delegate to :func:`src.utils.checkpointing.load_scvi_model`.
    """
    raise NotImplementedError(
        "Agent A: delegate to src.utils.checkpointing.load_scvi_model. "
        "Schema mismatch must raise loudly — no silent fallback to fresh model."
    )
