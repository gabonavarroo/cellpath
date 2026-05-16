"""scVI VAE wrapper — train, compute reference centroid, compute ε_success.

Owner: Agent A. See ARCHITECTURE.md Concepts 1 + 2.

Sacred rules:
- ``setup_anndata`` MUST be called with ``layer="counts"`` (raw integer UMIs).
- Save uses the official scVI API (``model.save()``). Manual state_dict saves are
  forbidden (CLAUDE.md rule #1, ARCHITECTURE.md D14).
- ``z_reference_centroid`` is the mean of the NT-control latents — the RL reward target.
- ``epsilon_success`` is data-driven. V1 intentionally uses the p50 control-cell
  distance threshold because that is the threshold used for the best RL result;
  p90 may be retained as a named reference artifact, not the canonical V1 value.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def train_vae(
    cfg: Any,
    adata: Any | None = None,
) -> tuple[Any, Any]:
    """Train scVI on preprocessed Norman AnnData and persist all Contract-1 artifacts.

    Workflow:
    1. Load preprocessed AnnData if not provided.
    2. Check for existing checkpoint; skip training if present (CLAUDE.md rule #1).
    3. ``SCVI.setup_anndata(adata, layer="counts")``.
    4. Instantiate ``SCVI`` with config hyperparameters.
    5. Train (early stopping on val ELBO).
    6. Save via official API → ``cfg.paths.vae_model_dir``.
    7. Encode latents Z = ``model.get_latent_representation()``.
    8. Write ``latents.h5ad``.
    9. Compute and write ``z_reference_centroid.npy``.
    10. Compute and write ``epsilon_success.json``.
    11. Write ``gene_vocab.json``.

    Parameters
    ----------
    cfg
        Hydra config (must have ``vae``, ``paths``, ``seed``).
    adata
        Preprocessed AnnData (HVG-subset, ``layers["counts"]`` present).
        If ``None``, loaded from ``cfg.paths.norman_processed_h5ad``.

    Returns
    -------
    tuple[scvi.model.SCVI, anndata.AnnData]
        ``(model, adata_with_latents)``.
    """
    import scvi
    from src.utils.checkpointing import load_scvi_model, save_scvi_model

    model_dir = Path(cfg.paths.vae_model_dir)

    # ------------------------------------------------------------------ #
    # Step 1: load preprocessed AnnData
    # ------------------------------------------------------------------ #
    if adata is None:
        from src.data.download import load_processed_anndata
        adata = load_processed_anndata(cfg.paths.norman_processed_h5ad)

    # ------------------------------------------------------------------ #
    # Step 2: checkpoint check — avoid retraining if model exists
    # ------------------------------------------------------------------ #
    model_exists = (model_dir / "model.pt").exists()
    if model_exists:
        log.info("Existing VAE checkpoint found at %s — loading instead of retraining.", model_dir)
        scvi.model.SCVI.setup_anndata(adata, layer=str(cfg.vae.setup.layer))
        model = load_scvi_model(model_dir, adata)
    else:

        # ------------------------------------------------------------------ #
        # Step 3: setup AnnData for scVI
        # ------------------------------------------------------------------ #
        setup_kwargs: dict[str, Any] = {"layer": str(cfg.vae.setup.layer)}
        if cfg.vae.setup.batch_key:
            setup_kwargs["batch_key"] = str(cfg.vae.setup.batch_key)
        if cfg.vae.setup.labels_key:
            setup_kwargs["labels_key"] = str(cfg.vae.setup.labels_key)

        scvi.model.SCVI.setup_anndata(adata, **setup_kwargs)
        log.info("setup_anndata complete. layer=%s", setup_kwargs["layer"])

        # ------------------------------------------------------------------ #
        # Step 4: instantiate model
        # ------------------------------------------------------------------ #
        model = scvi.model.SCVI(
            adata,
            n_latent=int(cfg.vae.n_latent),
            gene_likelihood=str(cfg.vae.gene_likelihood),
            n_layers=int(cfg.vae.n_layers),
            n_hidden=int(cfg.vae.n_hidden),
            dropout_rate=float(cfg.vae.dropout_rate),
            dispersion=str(cfg.vae.dispersion),
        )
        log.info(
            "SCVI model: n_latent=%d, likelihood=%s, n_layers=%d, n_hidden=%d",
            cfg.vae.n_latent, cfg.vae.gene_likelihood, cfg.vae.n_layers, cfg.vae.n_hidden,
        )

        # ------------------------------------------------------------------ #
        # Step 5: train
        # ------------------------------------------------------------------ #
        from src.utils.device import get_device
        _dev = get_device()
        _accel_map = {"mps": "mps", "cuda": "gpu", "cpu": "cpu"}
        _accelerator = _accel_map.get(_dev.type, "cpu")

        train_kwargs: dict[str, Any] = {
            "max_epochs": int(cfg.vae.max_epochs),
            "early_stopping": bool(cfg.vae.early_stopping),
            "batch_size": int(cfg.vae.batch_size),
            "accelerator": _accelerator,
        }
        if cfg.vae.early_stopping:
            train_kwargs["early_stopping_patience"] = int(cfg.vae.early_stopping_patience)
            train_kwargs["early_stopping_monitor"] = str(cfg.vae.early_stopping_monitor)

        log.info("Training scVI (max_epochs=%d, early_stopping=%s)...", cfg.vae.max_epochs, cfg.vae.early_stopping)
        model.train(**train_kwargs)
        log.info("Training complete.")

        # ------------------------------------------------------------------ #
        # Step 6: save via official scVI API
        # ------------------------------------------------------------------ #
        save_scvi_model(model, model_dir, overwrite=bool(cfg.vae.save_overwrite))
        log.info("scVI model saved to %s", model_dir)

    # ------------------------------------------------------------------ #
    # Step 7: encode latents
    # ------------------------------------------------------------------ #
    log.info("Encoding latent representations...")
    Z = model.get_latent_representation()  # (N_cells, n_latent) float32
    adata.obsm["X_scVI"] = Z
    log.info("Latents shape: %s", Z.shape)

    # ------------------------------------------------------------------ #
    # Step 8: write latents.h5ad
    # ------------------------------------------------------------------ #
    latents_path = Path(cfg.paths.vae_latents_h5ad)
    latents_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(latents_path))
    log.info("Latents AnnData saved to %s", latents_path)

    # ------------------------------------------------------------------ #
    # Step 9: z_reference_centroid
    # ------------------------------------------------------------------ #
    z_ref = compute_z_reference_centroid(adata)
    z_ref_path = Path(cfg.paths.vae_z_reference_centroid)
    z_ref_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(z_ref_path), z_ref)
    log.info("z_reference_centroid saved (shape %s, norm=%.3f)", z_ref.shape, float(np.linalg.norm(z_ref)))

    # ------------------------------------------------------------------ #
    # Step 10: epsilon_success
    # ------------------------------------------------------------------ #
    ctrl_mask = (adata.obs["perturbation_idx"] == 0).values
    eps_dict = compute_epsilon_success(
        Z[ctrl_mask], z_ref, percentile=float(cfg.vae.epsilon_percentile)
    )
    eps_path = Path(cfg.paths.vae_epsilon_success_json)
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    eps_path.write_text(json.dumps(eps_dict, indent=2))
    log.info("epsilon_success=%.4f (p%.0f, %d ctrl cells)", eps_dict["value"], eps_dict["percentile"], eps_dict["n_ctrl_cells"])

    # ------------------------------------------------------------------ #
    # Step 11: gene_vocab.json
    # ------------------------------------------------------------------ #
    _write_gene_vocab(adata, cfg.paths.vae_gene_vocab_json)

    return model, adata


def compute_z_reference_centroid(adata: Any, latent_key: str = "X_scVI") -> np.ndarray:
    """Mean latent of unperturbed NT-control cells — the RL reward target.

    Parameters
    ----------
    adata
        AnnData with ``obsm[latent_key]`` populated and ``obs["perturbation_idx"]``.
    latent_key
        Key into ``obsm`` (default ``"X_scVI"``).

    Returns
    -------
    np.ndarray
        Shape ``(n_latent,)`` float32.
    """
    ctrl_mask = (adata.obs["perturbation_idx"] == 0).values
    if ctrl_mask.sum() == 0:
        raise ValueError("No control cells (perturbation_idx == 0) found in AnnData.")
    return adata.obsm[latent_key][ctrl_mask].mean(axis=0).astype("float32")


def compute_epsilon_success(
    Z_ctrl: Any,
    z_ref: Any,
    percentile: float = 50.0,
) -> dict[str, Any]:
    """Data-driven RL success threshold.

    ε = percentile(||z_ctrl_i − z_ref||₂, p)

    V1 uses p50 as the canonical success threshold because the reported best RL
    result used that stricter median-control threshold. This calibrates the
    threshold to the geometry of the learned latent space rather than to an
    arbitrary Euclidean constant (ARCHITECTURE.md D8). p90 remains useful as a
    named reference/provenance artifact, but is not the V1 canonical threshold.

    Parameters
    ----------
    Z_ctrl
        Latent vectors of control cells, shape ``(N_ctrl, n_latent)``.
    z_ref
        Reference centroid, shape ``(n_latent,)``.
    percentile
        Percentile to use (default 50 for V1; p90 supported as reference provenance).

    Returns
    -------
    dict
        ``{"percentile", "value", "n_ctrl_cells", "method"}`` — written verbatim
        to ``artifacts/vae/epsilon_success.json``.
    """
    Z_ctrl = np.asarray(Z_ctrl, dtype=np.float32)
    z_ref = np.asarray(z_ref, dtype=np.float32)
    dists = np.linalg.norm(Z_ctrl - z_ref, axis=1)
    eps_value = float(np.percentile(dists, percentile))
    return {
        "percentile": float(percentile),
        "value": eps_value,
        "n_ctrl_cells": int(len(Z_ctrl)),
        "method": "L2_distance",
    }


def load_vae_model(model_dir: str | Path, adata: Any) -> Any:
    """Load a trained scVI model via the official API.

    Parameters
    ----------
    model_dir
        Directory written by ``model.save()``.
    adata
        AnnData with the same schema as the training data.

    Returns
    -------
    scvi.model.SCVI
    """
    from src.utils.checkpointing import load_scvi_model
    return load_scvi_model(model_dir, adata)


def _write_gene_vocab(adata: Any, path: str | Path) -> None:
    """Write ``gene_vocab.json`` (Contract 1).

    Contains the ordered list of single-gene perturbation symbols, the control
    index, the total number of single-gene targets, and the RL NO-OP action index.
    """
    encoder: dict[str, int] = adata.uns["perturbation_encoder"]
    ctrl_label: str = adata.uns["ctrl_label"]
    n_single: int = adata.uns["noop_idx"]

    # Single-gene symbols in encoder order (index 1..n_single)
    single_genes = [
        sym for sym, idx in sorted(encoder.items(), key=lambda kv: kv[1])
        if sym != ctrl_label and idx <= n_single
    ]

    vocab = {
        "genes": single_genes,            # ordered, 1-indexed (gene_idx 1..n_single in pairs)
        "ctrl_idx": 0,                    # perturbation_idx of control cells
        "n_genes": n_single,              # number of RL gene actions
        "noop_idx": n_single,             # RL NO-OP action = Discrete(n_genes+1) last index
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=2))
    log.info("gene_vocab.json written: %d single-gene targets", n_single)
