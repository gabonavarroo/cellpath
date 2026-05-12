"""VAE latent-space analysis: UMAP, silhouette, ARI, centroid distance.

Owner: Agent A. See PHASES.md Phase 2 and EXPERIMENTS.md §7.

All metric *computations* live in :mod:`src.analysis.metrics`. This module orchestrates them
and produces figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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

    Raises
    ------
    NotImplementedError
        Agent A: use ``umap.UMAP`` from ``umap-learn``.
    """
    raise NotImplementedError("Agent A: umap.UMAP fit_transform on Z.")


def plot_latent_umap(
    umap_xy: Any,
    labels: Any,
    z_reference_centroid_xy: Any | None = None,
    out_path: str | Path | None = None,
    title: str | None = None,
) -> Any:
    """Scatter UMAP, colored by perturbation, with the reference centroid marked.

    Parameters
    ----------
    umap_xy
        Output of :func:`compute_umap`.
    labels
        ``adata.obs["perturbation"]`` (string labels).
    z_reference_centroid_xy
        Optional ``(2,)`` UMAP coords of the reference centroid (computed via the same UMAP).
    out_path
        Where to save the figure. If ``None``, returned to caller without writing.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    NotImplementedError
        Agent A: implement with seaborn/matplotlib.
    """
    raise NotImplementedError("Agent A: scatter colored by labels; star for centroid.")


def analyze_latent_quality(
    adata: Any,
    latent_key: str = "X_scVI",
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Full latent-quality summary used in PHASES.md Phase 2 success criteria.

    Parameters
    ----------
    adata
        AnnData with ``obsm[latent_key]`` populated.
    latent_key
        Key for the latent matrix in ``adata.obsm``.
    save_dir
        Where to save plots; if ``None``, plots are not saved.

    Returns
    -------
    dict
        ``{"silhouette": float, "ari": float, "n_clusters_recovered": int,
           "centroid_intra_distance_mean": float, ...}``.

    Raises
    ------
    NotImplementedError
        Agent A: combines :func:`src.analysis.metrics.silhouette_perturbation` and
        :func:`ari_on_perturbation_clusters` and writes plots via :func:`plot_latent_umap`.
    """
    raise NotImplementedError(
        "Agent A: orchestrate silhouette + ARI + UMAP plots. "
        "All metric formulas live in src.analysis.metrics."
    )
