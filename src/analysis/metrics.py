"""Single source of truth for every metric in CellPath.

Owner: shared (Agents A + B add here; coordinate on additions).

Every function in this module has, in its docstring:
- The mathematical definition (formula).
- The expected input shapes.
- The interpretation (what good / bad values mean).
- The component(s) that use it.

CLAUDE.md sacred rule #4: notebooks and training scripts MUST import from here. No ad-hoc
metric computation in entry-point scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =============================================================================
# Dynamics-model metrics
# =============================================================================


def predictive_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R² across all latent dims pooled.

    R² = 1 − (sum_squared_residual / total_sum_squares).

    Parameters
    ----------
    y_true, y_pred
        Each shape ``(N, n_latent)``. Typically ``Δz_true`` and ``Δz_pred``.

    Returns
    -------
    float
        R² ∈ (−∞, 1]. 0 means "no better than predicting the mean"; 1 is perfect.

    Raises
    ------
    NotImplementedError
        Agent B: implement.
    """
    raise NotImplementedError("Agent B: pooled R² across latent dims.")


def pearson_r_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-dim Pearson R, then average.

    Parameters
    ----------
    y_true, y_pred
        Each shape ``(N, n_latent)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_latent,)`` of per-dim Pearson R. Mean is the usual summary statistic.

    Raises
    ------
    NotImplementedError
        Agent B: implement via ``scipy.stats.pearsonr`` per column.
    """
    raise NotImplementedError("Agent B: per-dim Pearson; return shape (n_latent,).")


def uncertainty_calibration_spearman(
    log_var_pred: np.ndarray,
    squared_error: np.ndarray,
) -> float:
    """Spearman correlation between predicted variance and observed squared error.

    Good dynamics models should have higher predicted variance where errors are large.
    Threshold ≥ 0.2 is required to pass the validation gate (see ``config/dynamics.yaml::gate``).

    Parameters
    ----------
    log_var_pred
        Predicted log σ², shape ``(N, n_latent)``.
    squared_error
        Observed (Δz_true − Δz_pred)², shape ``(N, n_latent)``.

    Returns
    -------
    float
        Spearman ρ over flattened arrays.

    Raises
    ------
    NotImplementedError
        Agent B: implement via ``scipy.stats.spearmanr`` over flattened arrays.
    """
    raise NotImplementedError("Agent B: scipy.stats.spearmanr over flattened arrays.")


def dynamics_validation_gate(
    z_ctrl: np.ndarray,
    gene_idx: np.ndarray,
    z_pert_true: np.ndarray,
    z_pert_pred_mlp: np.ndarray,
    log_var_pred: np.ndarray,
    cfg_gate: Any,
    baselines_train_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the primary + OOD dynamics validation gate.

    Implements the comparison matrix in PHASES.md Phase 2 + ``config/dynamics.yaml::gate``:
    no-op, global mean-Δ, per-gene mean-Δ, linear ridge per gene, nearest-neighbor (k=5),
    and uncertainty calibration.

    Parameters
    ----------
    z_ctrl
        Held-out control latents, shape ``(N, n_latent)``.
    gene_idx
        Held-out gene indices, shape ``(N,)``.
    z_pert_true
        Empirical paired latents, shape ``(N, n_latent)``.
    z_pert_pred_mlp
        Dynamics MLP predictions, shape ``(N, n_latent)``.
    log_var_pred
        Predicted log σ², shape ``(N, n_latent)``.
    cfg_gate
        Subset of Hydra config with the per-baseline margins (``cfg.dynamics.gate``).
    baselines_train_data
        Optional precomputed baselines (mean-Δ tables, ridge weights, kNN index) so the gate
        is fast to recompute across multiple validation splits.

    Returns
    -------
    dict
        Gate JSON contract (AGENTS.md §4 Contract 3) — written to ``gate.json``::

            {
              "passed": bool,
              "primary": {"r2": float, "pearson_r": float,
                          "baselines": {<name>: {<metric>: float}}},
              "ood": {"r2": float, "pearson_r": float, "baselines": {...}},
              "uncertainty_calibration": {"spearman": float, "pass": bool},
              "margins_used": cfg_gate,
            }

    Raises
    ------
    NotImplementedError
        Agent B: implement gate logic. Shared metric helpers (R², Pearson, Spearman) are in
        this same module.
    """
    raise NotImplementedError(
        "Agent B: assemble gate. Baselines: no-op, global mean Δ, per-gene mean Δ, "
        "linear ridge per gene, k-NN. Compare margins per cfg_gate. Add uncertainty calibration."
    )


# =============================================================================
# Latent-space / VAE metrics
# =============================================================================


def silhouette_perturbation(
    Z: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 10_000,
) -> float:
    """Silhouette score using perturbation labels as the partition.

    Informational diagnostic only — not a hard gate. Vanilla unsupervised scVI does not
    optimize for perturbation-cluster separation (no label supervision), so values in the
    range [−0.1, 0.05] are expected and do not indicate model failure. The meaningful Phase 1
    gate is ε_success ∈ (0.1, 10). See PHASES.md Phase 2 note for full justification.

    Parameters
    ----------
    Z
        Latent matrix, shape ``(N, n_latent)``.
    labels
        ``adata.obs["perturbation_idx"]`` or string labels, shape ``(N,)``.
    sample_size
        Subsample if too large; default 10,000 cells.

    Returns
    -------
    float
        Mean silhouette over the (sub)sample.
    """
    from sklearn.metrics import silhouette_score

    Z = np.asarray(Z, dtype=np.float32)
    labels = np.asarray(labels)

    # Remove singleton clusters (silhouette undefined for n=1)
    unique, counts = np.unique(labels, return_counts=True)
    valid = unique[counts >= 2]
    keep = np.isin(labels, valid)
    Z, labels = Z[keep], labels[keep]

    if sample_size is not None and len(Z) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Z), sample_size, replace=False)
        Z, labels = Z[idx], labels[idx]
        # Re-drop any singletons created by subsampling
        unique, counts = np.unique(labels, return_counts=True)
        valid = unique[counts >= 2]
        keep = np.isin(labels, valid)
        Z, labels = Z[keep], labels[keep]

    return float(silhouette_score(Z, labels, metric="euclidean"))


def ari_on_perturbation_clusters(
    Z: np.ndarray,
    labels: np.ndarray,
    n_clusters: int | None = None,
    sample_size: int = 20_000,
) -> float:
    """Adjusted Rand Index of a KMeans clustering vs ground-truth perturbation labels.

    ARI ∈ [−0.5, 1]. 0 = random; 1 = perfect. Values > 0.1 indicate meaningful recovery.
    MiniBatchKMeans is used for speed; subsample to ``sample_size`` for tractability.

    Parameters
    ----------
    Z, labels
        See :func:`silhouette_perturbation`.
    n_clusters
        Number of KMeans clusters; defaults to number of unique labels.
    sample_size
        Subsample for KMeans fit; ARI rankings stabilise well before 20k cells.

    Returns
    -------
    float
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score

    Z = np.asarray(Z, dtype=np.float32)
    labels = np.asarray(labels)

    if n_clusters is None:
        n_clusters = int(len(np.unique(labels)))

    if len(Z) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Z), sample_size, replace=False)
        Z, labels = Z[idx], labels[idx]

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=4096)
    pred = km.fit_predict(Z)

    return float(adjusted_rand_score(labels, pred))


# =============================================================================
# RL metrics
# =============================================================================


def success_rate(rollouts: Any) -> float:
    """Fraction of episodes that ended with ``info["success"] == True``.

    Parameters
    ----------
    rollouts
        Polars or pandas DataFrame with the Contract 4 schema for ``rollouts.parquet``.

    Returns
    -------
    float

    Raises
    ------
    NotImplementedError
        Agent B: implement. Group by ``episode_id``, check terminal ``success``.
    """
    raise NotImplementedError("Agent B: group by episode_id, take terminal success.")


def mean_steps_to_success(rollouts: Any) -> float:
    """Average episode length conditional on success.

    Parameters
    ----------
    rollouts
        See :func:`success_rate`.

    Returns
    -------
    float

    Raises
    ------
    NotImplementedError
        Agent B: implement.
    """
    raise NotImplementedError("Agent B: filter to successful episodes; mean(step + 1).")


# =============================================================================
# DepMap-enrichment metrics
# =============================================================================


def hypergeometric_enrichment(
    selected_genes: list[str],
    gene_set: list[str],
    background: list[str],
) -> dict[str, Any]:
    """One-sided hypergeometric test for overlap.

    Parameters
    ----------
    selected_genes
        RL-selected (top-K) genes.
    gene_set
        Reference gene set (e.g. DepMap K562 essentials).
    background
        Universe of genes (e.g. all HVGs, or all genes in DepMap).

    Returns
    -------
    dict
        ``{"k": int, "K": int, "n": int, "N": int, "p_value": float, "log_odds": float}``.

    Raises
    ------
    NotImplementedError
        Agent A: ``scipy.stats.hypergeom`` upper-tail.
    """
    raise NotImplementedError("Agent A: scipy.stats.hypergeom one-sided.")


def gsea_preranked(
    ranked_genes: list[str],
    scores: np.ndarray,
    gene_set: list[str],
) -> dict[str, Any]:
    """GSEA-preranked enrichment.

    Parameters
    ----------
    ranked_genes
        Genes in ranked order.
    scores
        Score vector (e.g. RL action frequency, DepMap Chronos).
    gene_set
        Reference gene set.

    Returns
    -------
    dict
        ``{"ES": float, "NES": float, "p_value": float, "fdr": float}``.

    Raises
    ------
    NotImplementedError
        Agent A: implement classic GSEA-preranked (KS-like running enrichment), with N=1000
        permutations for the null.
    """
    raise NotImplementedError("Agent A: classic GSEA-preranked with permutation null.")


def null_enrichment_comparison(
    observed_enrichment: float,
    selected_genes: list[str],
    background: list[str],
    n_null_samples: int = 1_000,
    match_expression: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare observed enrichment to random gene sets of matched size (+ optional expression).

    Parameters
    ----------
    observed_enrichment
        The observed statistic (e.g. hypergeometric log-odds).
    selected_genes
        The actual RL-selected gene set.
    background
        Universe of genes.
    n_null_samples
        Number of random matched sets.
    match_expression
        Optional per-gene expression mean (for expression-matched sampling).

    Returns
    -------
    dict
        ``{"z_score": float, "empirical_p": float, "null_mean": float, "null_std": float}``.

    Raises
    ------
    NotImplementedError
        Agent A: implement matched-size and matched-expression null distributions.
    """
    raise NotImplementedError(
        "Agent A: matched-size + matched-expression null; report z and empirical p."
    )
