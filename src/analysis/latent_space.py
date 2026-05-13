"""VAE latent-space analysis: UMAP, silhouette, ARI, centroid distance.

Owner: Agent A. See PHASES.md Phase 2 and EXPERIMENTS.md §7.

All metric *computations* live in :mod:`src.analysis.metrics`. This module orchestrates them
and produces figures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def compute_umap(
    Z: Any,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> Any:
    """UMAP projection of the latent matrix.

    Parameters
    ----------
    Z
        Latent matrix, shape ``(N, n_latent)``.
    n_neighbors, min_dist, random_state
        Standard UMAP knobs.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)`` UMAP coordinates.
    """
    import umap as umap_module

    Z = np.asarray(Z, dtype=np.float32)
    reducer = umap_module.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1,
    )
    return reducer.fit_transform(Z).astype(np.float32)


def plot_latent_umap(
    umap_xy: Any,
    labels: Any,
    z_reference_centroid_xy: Any | None = None,
    out_path: str | Path | None = None,
    title: str | None = None,
) -> Any:
    """Scatter UMAP, colored by perturbation, with the reference centroid marked.

    Top-10 most common perturbations are highlighted in color; all others in light gray.

    Parameters
    ----------
    umap_xy
        Output of :func:`compute_umap`, shape ``(N, 2)``.
    labels
        ``adata.obs["perturbation"]`` (string labels), shape ``(N,)``.
    z_reference_centroid_xy
        Optional ``(2,)`` UMAP coords of the reference centroid.
    out_path
        Where to save the figure. If ``None``, returned to caller without writing.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    umap_xy = np.asarray(umap_xy)
    labels = np.asarray(labels, dtype=str)

    unique_labels = np.unique(labels)
    label_counts = {lbl: int((labels == lbl).sum()) for lbl in unique_labels}
    top_labels = sorted(label_counts, key=lambda x: -label_counts[x])[:10]
    top_set = set(top_labels)

    cmap = plt.cm.get_cmap("tab10", len(top_labels))
    label_to_color = {lbl: cmap(i) for i, lbl in enumerate(top_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))

    other_mask = ~np.isin(labels, list(top_set))
    if other_mask.any():
        ax.scatter(
            umap_xy[other_mask, 0], umap_xy[other_mask, 1],
            c="lightgray", s=1, alpha=0.3, rasterized=True,
        )

    for lbl in top_labels:
        mask = labels == lbl
        ax.scatter(
            umap_xy[mask, 0], umap_xy[mask, 1],
            c=[label_to_color[lbl]], s=2, alpha=0.7,
            label=f"{lbl} (n={label_counts[lbl]})", rasterized=True,
        )

    if z_reference_centroid_xy is not None:
        ax.scatter(
            z_reference_centroid_xy[0], z_reference_centroid_xy[1],
            c="black", s=250, marker="*", zorder=10, label="z_ref_centroid",
        )

    ax.legend(markerscale=5, loc="best", fontsize=8, framealpha=0.7)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if title:
        ax.set_title(title)

    if out_path is not None:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")

    return fig


def analyze_latent_quality(
    adata: Any,
    latent_key: str = "X_scVI",
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Full latent-quality summary used in PHASES.md Phase 1 + Phase 2 success criteria.

    Always computes silhouette + ARI + centroid intra-distances (fast, no UMAP).
    UMAP projection and plots are only produced when ``save_dir`` is provided (Phase 2).

    Parameters
    ----------
    adata
        AnnData with ``obsm[latent_key]`` and ``obs["perturbation_idx"]`` populated.
    latent_key
        Key for the latent matrix in ``adata.obsm``.
    save_dir
        Where to save plots and ``latent_quality.json``; if ``None``, plots are skipped.

    Returns
    -------
    dict
        ``{"silhouette": float, "ari": float, "n_clusters_recovered": int,
           "centroid_intra_distance_mean": float, "silhouette_pass": bool}``.
    """
    from src.analysis.metrics import ari_on_perturbation_clusters, silhouette_perturbation

    Z = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    labels = np.asarray(adata.obs["perturbation_idx"].values)
    n_clusters = int(len(np.unique(labels)))

    # --- Silhouette (Phase 1 gate: ≥ 0.05) ---
    log.info("Computing silhouette score (subsample=10k)...")
    sil = silhouette_perturbation(Z, labels, sample_size=10_000)
    log.info("Silhouette: %.4f  (threshold ≥ 0.05 → %s)", sil, "PASS" if sil >= 0.05 else "FAIL")

    # --- ARI ---
    log.info("Computing ARI (MiniBatchKMeans, subsample=20k, n_clusters=%d)...", n_clusters)
    ari = ari_on_perturbation_clusters(Z, labels, n_clusters=n_clusters)
    log.info("ARI: %.4f", ari)

    # --- Centroid intra-distance ---
    intra = []
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.sum() < 2:
            continue
        Z_lbl = Z[mask]
        centroid = Z_lbl.mean(axis=0)
        intra.append(float(np.linalg.norm(Z_lbl - centroid, axis=1).mean()))
    centroid_intra_distance_mean = float(np.mean(intra))
    log.info("Centroid intra-distance mean: %.4f", centroid_intra_distance_mean)

    results: dict[str, Any] = {
        "silhouette": sil,
        "silhouette_pass": bool(sil >= 0.05),
        "ari": ari,
        "n_clusters_recovered": n_clusters,
        "centroid_intra_distance_mean": centroid_intra_distance_mean,
    }

    # --- UMAP + plots (Phase 2, only when save_dir provided) ---
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            log.info("Computing UMAP (this may take 10–30 min on 111k cells)...")
            umap_xy = compute_umap(Z)
            adata.obsm["X_umap"] = umap_xy

            import matplotlib.pyplot as plt
            fig = plot_latent_umap(
                umap_xy,
                adata.obs["perturbation"].values,
                out_path=save_dir / "umap_perturbation.png",
                title=f"VAE latent UMAP  |  silhouette={sil:.3f}  ARI={ari:.3f}",
            )
            plt.close(fig)
            log.info("UMAP plot saved → %s", save_dir / "umap_perturbation.png")
            results["umap_computed"] = True
        except Exception as exc:
            log.warning("UMAP skipped: %s", exc)
            results["umap_computed"] = False

        (save_dir / "latent_quality.json").write_text(
            json.dumps({k: v for k, v in results.items() if isinstance(v, (int, float, bool))}, indent=2)
        )
        log.info("Metrics saved → %s", save_dir / "latent_quality.json")

    return results
