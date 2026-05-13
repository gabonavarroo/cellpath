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
# Dynamics-model metrics (Agent B — Phase 2)
# =============================================================================

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _as_float32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _cfg_value(cfg: Any, key: str) -> Any:
    """Read a config value from dict, OmegaConf DictConfig, dataclass, or SimpleNamespace."""
    if cfg is None:
        return None
    try:
        return cfg[key]
    except (TypeError, KeyError):
        pass
    return getattr(cfg, key)


def _one_hot_genes(gene_idx: np.ndarray, n_genes: int) -> np.ndarray:
    """Convert 1-indexed gene_idx to (N, n_genes) float32 one-hot matrix.

    Indices outside [1, n_genes] (e.g. unseen OOD genes) produce all-zero rows.
    """
    idx = np.asarray(gene_idx, dtype=np.int64)
    out = np.zeros((len(idx), n_genes), dtype=np.float32)
    valid = (idx >= 1) & (idx <= n_genes)
    rows = np.where(valid)[0]
    cols = idx[valid] - 1  # convert to 0-indexed
    out[rows, cols] = 1.0
    return out


def _safe_float(x: Any) -> float:
    """Cast to Python float; NaN/Inf → 0.0."""
    v = float(x)
    if v != v or v == float("inf") or v == float("-inf"):
        return 0.0
    return v


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def predictive_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R² across all latent dims pooled.

    R² = 1 − (sum_squared_residual / total_sum_squares).

    Follows sklearn semantics for the degenerate case: if total_sum_squares == 0
    (constant ``y_true``), returns 1.0 when predictions match exactly, else 0.0.

    Parameters
    ----------
    y_true, y_pred
        Each shape ``(N, n_latent)``. Typically ``Δz_true`` and ``Δz_pred``.

    Returns
    -------
    float
        R² ∈ (−∞, 1]. 0 means "no better than predicting the mean"; 1 is perfect.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        return 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if bool(np.allclose(yt, yp)) else 0.0
    return float(1.0 - ss_res / ss_tot)


def pearson_r_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-dim Pearson R, then average.

    Vectorised formula: r_d = cov(y_true[:,d], y_pred[:,d]) / (std_true_d * std_pred_d).
    Constant or NaN columns yield 0.0 for that dimension.

    Parameters
    ----------
    y_true, y_pred
        Each shape ``(N, n_latent)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_latent,)`` of per-dim Pearson R. Mean is the usual summary statistic.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    yt_c = yt - yt.mean(axis=0)
    yp_c = yp - yp.mean(axis=0)
    num = (yt_c * yp_c).sum(axis=0)
    den = np.sqrt((yt_c ** 2).sum(axis=0) * (yp_c ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0.0, num / den, 0.0)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def uncertainty_calibration_spearman(
    log_var_pred: np.ndarray,
    squared_error: np.ndarray,
) -> float:
    """Spearman correlation between predicted variance and observed squared error.

    Computes Spearman ρ between exp(log_var_pred) and squared_error over all
    flattened elements. Good dynamics models have higher predicted variance where
    errors are large. Threshold ≥ 0.2 is required to pass the validation gate
    (see ``config/dynamics.yaml::gate``).

    Parameters
    ----------
    log_var_pred
        Predicted log σ², shape ``(N, n_latent)``.
    squared_error
        Observed (Δz_true − Δz_pred)², shape ``(N, n_latent)``.

    Returns
    -------
    float
        Spearman ρ ∈ [−1, 1] over flattened arrays. Constant inputs → 0.0.
    """
    from scipy.stats import spearmanr

    var_flat = np.exp(np.asarray(log_var_pred, dtype=np.float64)).ravel()
    err_flat = np.asarray(squared_error, dtype=np.float64).ravel()
    if len(var_flat) == 0:
        return 0.0
    result = spearmanr(var_flat, err_flat)
    rho = float(result.statistic)
    return 0.0 if (rho != rho) else rho  # NaN → 0.0


def dynamics_validation_gate(
    z_ctrl: np.ndarray,
    gene_idx: np.ndarray,
    z_pert_true: np.ndarray,
    z_pert_pred_mlp: np.ndarray,
    log_var_pred: np.ndarray,
    cfg_gate: Any,
    baselines_train_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the dynamics validation gate on one data split (primary or OOD).

    Implements the comparison matrix in PHASES.md Phase 2 + ``config/dynamics.yaml::gate``:
    no-op, global mean-Δ, per-gene mean-Δ, linear ridge, nearest-neighbor (k=5), and
    uncertainty calibration.

    Call this function once per split; the training script merges two calls (val + ood)
    into the final ``gate.json`` — the ``"ood"`` block is filled by the caller.

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
        Supports dict, OmegaConf DictConfig, dataclass, or SimpleNamespace.
    baselines_train_data
        **Required.** Dict with keys ``"z_ctrl"``, ``"gene_idx"``, ``"z_pert"`` from
        the TRAIN split. Baselines are fitted on train data only to avoid validation
        leakage.

    Returns
    -------
    dict
        JSON-safe dict conforming to AGENTS.md Contract 3 ``gate.json`` primary block::

            {
              "passed": bool,
              "primary": {
                "r2": float, "pearson_r": float,
                "baselines": {<name>: {"r2": float, "pearson_r": float}},
                "margin_checks": {<key>: {"value": float, "threshold": float, "pass": bool}},
              },
              "uncertainty_calibration": {"spearman": float, "pass": bool},
              "margins_used": {<key>: float},
            }

    Raises
    ------
    ValueError
        If ``baselines_train_data`` is None or missing required keys.
    """
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import NearestNeighbors

    if baselines_train_data is None:
        raise ValueError(
            "dynamics_validation_gate requires baselines_train_data with keys "
            "'z_ctrl', 'gene_idx', 'z_pert' from the TRAIN split to avoid validation leakage."
        )
    for _k in ("z_ctrl", "gene_idx", "z_pert"):
        if _k not in baselines_train_data:
            raise ValueError(f"baselines_train_data is missing required key '{_k}'.")

    # --- coerce inputs ---
    z_ctrl_v          = _as_float32(z_ctrl)
    gene_idx_v        = np.asarray(gene_idx, dtype=np.int32)
    z_pert_true_v     = _as_float32(z_pert_true)
    z_pert_pred_mlp_v = _as_float32(z_pert_pred_mlp)
    log_var_pred_v    = _as_float32(log_var_pred)

    delta_true     = z_pert_true_v     - z_ctrl_v  # (N, n_latent)
    delta_pred_mlp = z_pert_pred_mlp_v - z_ctrl_v  # (N, n_latent)

    # --- train data (baselines fitted on train only) ---
    z_ctrl_tr  = _as_float32(baselines_train_data["z_ctrl"])
    g_idx_tr   = np.asarray(baselines_train_data["gene_idx"], dtype=np.int32)
    z_pert_tr  = _as_float32(baselines_train_data["z_pert"])
    delta_tr   = z_pert_tr - z_ctrl_tr              # (M, n_latent)
    n_genes_tr = int(g_idx_tr.max())

    # --- MLP metrics ---
    mlp_r2      = predictive_r2(delta_true, delta_pred_mlp)
    mlp_pearson = _safe_float(np.nanmean(pearson_r_per_dim(delta_true, delta_pred_mlp)))

    # --- baseline predictions ---

    # a. no-op: Δ = 0
    delta_noop = np.zeros_like(delta_true)

    # b. global mean Δ (from train)
    global_mean  = delta_tr.mean(axis=0)
    delta_global = np.broadcast_to(global_mean, delta_true.shape).copy()

    # c. per-gene mean Δ (from train); unseen val genes fall back to global mean
    mean_by_gene: dict[int, np.ndarray] = {
        int(g): delta_tr[g_idx_tr == g].mean(axis=0)
        for g in np.unique(g_idx_tr)
    }
    delta_per_gene = np.stack(
        [mean_by_gene.get(int(g), global_mean) for g in gene_idx_v]
    ).astype(np.float32)

    # d. linear ridge on [z_ctrl, one_hot(gene_idx)] → delta
    X_tr  = np.concatenate([z_ctrl_tr, _one_hot_genes(g_idx_tr,   n_genes_tr)], axis=1)
    X_val = np.concatenate([z_ctrl_v,  _one_hot_genes(gene_idx_v, n_genes_tr)], axis=1)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_tr, delta_tr)
    delta_ridge = _as_float32(ridge.predict(X_val))

    # e. kNN (k=5) on train z_ctrl; average k nearest delta_tr (geometry-only baseline)
    k = min(5, len(z_ctrl_tr))
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    knn.fit(z_ctrl_tr)
    _, nn_idx = knn.kneighbors(z_ctrl_v)
    delta_knn = delta_tr[nn_idx].mean(axis=1).astype(np.float32)

    baselines_pred: dict[str, np.ndarray] = {
        "no_op":               delta_noop,
        "global_mean_delta":   delta_global,
        "per_gene_mean_delta": delta_per_gene,
        "linear_ridge":        delta_ridge,
        "nearest_neighbor":    delta_knn,
    }

    # --- per-baseline metrics ---
    baselines_out: dict[str, dict[str, float]] = {
        name: {
            "r2":        predictive_r2(delta_true, dp),
            "pearson_r": _safe_float(np.nanmean(pearson_r_per_dim(delta_true, dp))),
        }
        for name, dp in baselines_pred.items()
    }

    # --- margin checks ---
    _margin_spec = [
        ("margin_vs_noop_r2",              mlp_r2,      baselines_out["no_op"]["r2"]),
        ("margin_vs_global_mean_r2",       mlp_r2,      baselines_out["global_mean_delta"]["r2"]),
        ("margin_vs_per_gene_mean_r2",     mlp_r2,      baselines_out["per_gene_mean_delta"]["r2"]),
        ("margin_vs_linear_ridge_pearson", mlp_pearson, baselines_out["linear_ridge"]["pearson_r"]),
        ("margin_vs_knn_r2",               mlp_r2,      baselines_out["nearest_neighbor"]["r2"]),
    ]
    margin_checks: dict[str, Any] = {}
    all_margins_pass = True
    for key, mlp_val, base_val in _margin_spec:
        threshold = float(_cfg_value(cfg_gate, key))
        value     = float(mlp_val - base_val)
        passes    = bool(value >= threshold)
        margin_checks[key] = {"value": value, "threshold": threshold, "pass": passes}
        if not passes:
            all_margins_pass = False

    # --- uncertainty calibration ---
    squared_error   = (delta_true - delta_pred_mlp) ** 2
    calib_rho       = uncertainty_calibration_spearman(log_var_pred_v, squared_error)
    calib_threshold = float(_cfg_value(cfg_gate, "min_uncertainty_calibration_spearman"))
    calib_pass      = bool(calib_rho >= calib_threshold)

    # --- overall pass ---
    passed = bool(all_margins_pass and calib_pass)

    # --- threshold snapshot for auditability ---
    _all_keys = [
        "margin_vs_noop_r2",
        "margin_vs_global_mean_r2",
        "margin_vs_per_gene_mean_r2",
        "margin_vs_linear_ridge_pearson",
        "margin_vs_knn_r2",
        "min_uncertainty_calibration_spearman",
    ]
    margins_used = {k: float(_cfg_value(cfg_gate, k)) for k in _all_keys}

    return {
        "passed": passed,
        "primary": {
            "r2":        float(mlp_r2),
            "pearson_r": float(mlp_pearson),
            "baselines": {
                name: {"r2": float(v["r2"]), "pearson_r": float(v["pearson_r"])}
                for name, v in baselines_out.items()
            },
            "margin_checks": margin_checks,
        },
        "uncertainty_calibration": {"spearman": float(calib_rho), "pass": calib_pass},
        "margins_used": margins_used,
    }


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
