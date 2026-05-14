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


def _fit_ridge_baseline(
    z_ctrl_tr: np.ndarray,
    gene_idx_tr: np.ndarray,
    delta_tr: np.ndarray,
    n_genes: int,
    *,
    alpha: float = 1.0,
    random_state: int = 42,
) -> Any:
    """Fit ``Ridge`` on ``[z_ctrl, one_hot(gene_idx, n_genes)] → delta``.

    Single source of truth for the linear-ridge baseline used by the dynamics
    validation gate AND the gate-diagnostics report. The ``n_genes`` argument
    is the one-hot width at fit time; the same width MUST be used at predict
    time so feature dimensions line up.

    Parameters
    ----------
    z_ctrl_tr, delta_tr
        Each ``(M, n_latent)``. ``delta_tr = z_pert_tr - z_ctrl_tr``.
    gene_idx_tr
        Shape ``(M,)`` int, 1-indexed per Contract 2.
    n_genes
        One-hot width. Typically ``int(gene_idx_tr.max())``.
    alpha
        L2 regularization strength (default 1.0; matches gate baseline).
    random_state
        Forwarded to sklearn for reproducibility.

    Returns
    -------
    sklearn.linear_model.Ridge
        Fitted estimator with ``predict`` that returns float64.
    """
    from sklearn.linear_model import Ridge

    X_tr = np.concatenate(
        [_as_float32(z_ctrl_tr), _one_hot_genes(gene_idx_tr, n_genes)], axis=1
    )
    ridge = Ridge(alpha=alpha, random_state=random_state)
    ridge.fit(X_tr, _as_float32(delta_tr))
    return ridge


def _predict_ridge_baseline(
    ridge: Any,
    z_ctrl: np.ndarray,
    gene_idx: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    """Predict Δz from the fitted ridge baseline; return float32.

    ``n_genes`` must equal the value passed to :func:`_fit_ridge_baseline`.
    """
    X = np.concatenate(
        [_as_float32(z_ctrl), _one_hot_genes(gene_idx, n_genes)], axis=1
    )
    return _as_float32(ridge.predict(X))


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

    # d. linear ridge on [z_ctrl, one_hot(gene_idx)] → delta (shared helper to keep
    #    the gate baseline and gate_diagnostics consistent)
    ridge = _fit_ridge_baseline(z_ctrl_tr, g_idx_tr, delta_tr, n_genes_tr)
    delta_ridge = _predict_ridge_baseline(ridge, z_ctrl_v, gene_idx_v, n_genes_tr)

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


def gate_diagnostics(
    z_ctrl_train: np.ndarray,
    gene_idx_train: np.ndarray,
    z_pert_train: np.ndarray,
    z_ctrl_val: np.ndarray,
    gene_idx_val: np.ndarray,
    z_pert_val: np.ndarray,
    z_pert_pred_mlp_val: np.ndarray,
    z_ctrl_ood: np.ndarray | None = None,
    gene_idx_ood: np.ndarray | None = None,
    z_pert_ood: np.ndarray | None = None,
    z_pert_pred_mlp_ood: np.ndarray | None = None,
    n_worst_dims: int = 8,
    per_gene_min_for_pearson: int = 30,
) -> dict[str, Any]:
    """Per-dim and per-gene diagnostic breakdown of MLP vs ridge.

    Uses :func:`_fit_ridge_baseline` + :func:`_predict_ridge_baseline` (the same helpers
    consumed by :func:`dynamics_validation_gate`) so the ridge baseline is bit-identical
    between gate.json and gate_diagnostics.json — there is no second ridge implementation
    anywhere.

    The output is intended for an engineer reading ``artifacts/dynamics/gate_diagnostics.json``
    while debugging a failed primary gate. Three lenses:

    1. **Overall:** MLP vs ridge R² and Pearson on val (and OOD if present), plus the
       MLP-minus-ridge Pearson which is exactly the quantity the primary gate compares
       against ``margin_vs_linear_ridge_pearson``.
    2. **Per-dim:** the same comparison projected onto each of the ``n_latent`` axes;
       useful for spotting "the MLP wins 30 dims and loses 2 catastrophically".
    3. **Per-gene (val only):** R² of MLP and ridge per gene_idx, with sample count.
       Pearson is reported only when the per-gene cell count ≥ ``per_gene_min_for_pearson``
       (default 30); below that, per-dim Pearson is too noisy and would mislead.

    Parameters
    ----------
    z_ctrl_train, gene_idx_train, z_pert_train
        Training triples used to fit the ridge baseline. Shapes
        ``(M, n_latent)`` / ``(M,)`` / ``(M, n_latent)``.
    z_ctrl_val, gene_idx_val, z_pert_val, z_pert_pred_mlp_val
        Validation triples + the MLP's predicted ``z_pert`` (NOT Δz). Shapes
        ``(N_val, n_latent)`` for the latent arrays.
    z_ctrl_ood, gene_idx_ood, z_pert_ood, z_pert_pred_mlp_ood
        Optional OOD triples + predictions. If any is ``None``, the ``"ood"``
        sections of the returned dict are ``None`` and the function still succeeds.
    n_worst_dims
        How many of the worst (MLP minus ridge) per-dim Pearson values to expose
        in the ``"worst_dims"`` block.
    per_gene_min_for_pearson
        Threshold below which per-gene Pearson is reported as ``None``.

    Returns
    -------
    dict
        JSON-safe dict with sections ``overall``, ``per_dim``, ``worst_dims``,
        ``per_gene_val``. See the source for the exact schema.
    """
    z_ctrl_tr  = _as_float32(z_ctrl_train)
    g_idx_tr   = np.asarray(gene_idx_train, dtype=np.int32)
    z_pert_tr  = _as_float32(z_pert_train)
    delta_tr   = z_pert_tr - z_ctrl_tr
    n_genes_tr = int(g_idx_tr.max()) if len(g_idx_tr) else 0

    z_ctrl_v       = _as_float32(z_ctrl_val)
    g_idx_v        = np.asarray(gene_idx_val, dtype=np.int32)
    z_pert_v       = _as_float32(z_pert_val)
    z_pert_pred_v  = _as_float32(z_pert_pred_mlp_val)
    delta_true_v   = z_pert_v - z_ctrl_v
    delta_mlp_v    = z_pert_pred_v - z_ctrl_v

    ridge = _fit_ridge_baseline(z_ctrl_tr, g_idx_tr, delta_tr, n_genes_tr)
    delta_ridge_v = _predict_ridge_baseline(ridge, z_ctrl_v, g_idx_v, n_genes_tr)

    # --- per-dim Pearson on val (already vectorised) ---
    per_dim_mlp_v   = pearson_r_per_dim(delta_true_v, delta_mlp_v)
    per_dim_ridge_v = pearson_r_per_dim(delta_true_v, delta_ridge_v)
    per_dim_diff_v  = per_dim_mlp_v - per_dim_ridge_v

    overall_val = {
        "mlp_r2":                  float(predictive_r2(delta_true_v, delta_mlp_v)),
        "mlp_pearson":             _safe_float(np.nanmean(per_dim_mlp_v)),
        "ridge_r2":                float(predictive_r2(delta_true_v, delta_ridge_v)),
        "ridge_pearson":           _safe_float(np.nanmean(per_dim_ridge_v)),
        "mlp_minus_ridge_pearson": _safe_float(np.nanmean(per_dim_diff_v)),
    }
    per_dim_val = {
        "mlp_pearson":     [float(x) for x in per_dim_mlp_v],
        "ridge_pearson":   [float(x) for x in per_dim_ridge_v],
        "mlp_minus_ridge": [float(x) for x in per_dim_diff_v],
    }

    # --- worst dims (val) ---
    n_dim = int(per_dim_diff_v.shape[0])
    k_worst = max(0, min(int(n_worst_dims), n_dim))
    order_worst = np.argsort(per_dim_diff_v)[:k_worst]  # most-negative diff first
    worst_dims_val = [
        {
            "dim":   int(d),
            "mlp":   float(per_dim_mlp_v[d]),
            "ridge": float(per_dim_ridge_v[d]),
            "diff":  float(per_dim_diff_v[d]),
        }
        for d in order_worst
    ]

    # --- per-gene summary (val) ---
    per_gene_val: list[dict[str, Any]] = []
    for g in np.unique(g_idx_v):
        mask = g_idx_v == g
        n_g = int(mask.sum())
        dt = delta_true_v[mask]
        dm = delta_mlp_v[mask]
        dr = delta_ridge_v[mask]
        r2_m = float(predictive_r2(dt, dm))
        r2_r = float(predictive_r2(dt, dr))
        entry: dict[str, Any] = {
            "gene_idx":           int(g),
            "n":                  n_g,
            "mlp_r2":             r2_m,
            "ridge_r2":           r2_r,
            "mlp_minus_ridge_r2": float(r2_m - r2_r),
            "mlp_pearson":        None,
            "ridge_pearson":      None,
        }
        if n_g >= int(per_gene_min_for_pearson):
            pearson_m = _safe_float(np.nanmean(pearson_r_per_dim(dt, dm)))
            pearson_r = _safe_float(np.nanmean(pearson_r_per_dim(dt, dr)))
            entry["mlp_pearson"]   = pearson_m
            entry["ridge_pearson"] = pearson_r
        per_gene_val.append(entry)
    per_gene_val.sort(key=lambda e: e["mlp_minus_ridge_r2"])  # worst first

    # --- OOD (optional) ---
    overall_ood: dict[str, float] | None = None
    per_dim_ood: dict[str, list[float]] | None = None
    worst_dims_ood: list[dict[str, Any]] = []

    ood_inputs = (z_ctrl_ood, gene_idx_ood, z_pert_ood, z_pert_pred_mlp_ood)
    if all(x is not None for x in ood_inputs):
        z_ctrl_o       = _as_float32(z_ctrl_ood)
        g_idx_o        = np.asarray(gene_idx_ood, dtype=np.int32)
        z_pert_o       = _as_float32(z_pert_ood)
        z_pert_pred_o  = _as_float32(z_pert_pred_mlp_ood)
        delta_true_o   = z_pert_o - z_ctrl_o
        delta_mlp_o    = z_pert_pred_o - z_ctrl_o
        delta_ridge_o  = _predict_ridge_baseline(ridge, z_ctrl_o, g_idx_o, n_genes_tr)

        per_dim_mlp_o   = pearson_r_per_dim(delta_true_o, delta_mlp_o)
        per_dim_ridge_o = pearson_r_per_dim(delta_true_o, delta_ridge_o)
        per_dim_diff_o  = per_dim_mlp_o - per_dim_ridge_o

        overall_ood = {
            "mlp_r2":                  float(predictive_r2(delta_true_o, delta_mlp_o)),
            "mlp_pearson":             _safe_float(np.nanmean(per_dim_mlp_o)),
            "ridge_r2":                float(predictive_r2(delta_true_o, delta_ridge_o)),
            "ridge_pearson":           _safe_float(np.nanmean(per_dim_ridge_o)),
            "mlp_minus_ridge_pearson": _safe_float(np.nanmean(per_dim_diff_o)),
        }
        per_dim_ood = {
            "mlp_pearson":     [float(x) for x in per_dim_mlp_o],
            "ridge_pearson":   [float(x) for x in per_dim_ridge_o],
            "mlp_minus_ridge": [float(x) for x in per_dim_diff_o],
        }
        n_dim_o = int(per_dim_diff_o.shape[0])
        k_o = max(0, min(int(n_worst_dims), n_dim_o))
        order_o = np.argsort(per_dim_diff_o)[:k_o]
        worst_dims_ood = [
            {
                "dim":   int(d),
                "mlp":   float(per_dim_mlp_o[d]),
                "ridge": float(per_dim_ridge_o[d]),
                "diff":  float(per_dim_diff_o[d]),
            }
            for d in order_o
        ]

    return {
        "overall": {"val": overall_val, "ood": overall_ood},
        "per_dim": {"val": per_dim_val, "ood": per_dim_ood},
        "worst_dims": {"val": worst_dims_val, "ood": worst_dims_ood},
        "per_gene_val": per_gene_val,
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

    Tests whether the observed overlap k = |selected ∩ gene_set| is larger than
    expected when drawing n = |selected| genes from a population of N = |background|
    genes of which K = |gene_set ∩ background| are "successes":

        p = P(X ≥ k) = hypergeom.sf(k - 1, N, K, n)   (upper-tail, one-sided)
        log_odds = log2( (k/n) / (K/N) )                (effect size; 0 = no enrichment)

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
    """
    from scipy.stats import hypergeom

    bg   = set(background)
    sel  = set(selected_genes) & bg
    gset = set(gene_set) & bg

    N = len(bg)
    K = len(gset)
    n = len(sel)
    k = len(sel & gset)

    p_value = float(hypergeom.sf(k - 1, N, K, n)) if (N > 0 and K > 0 and n > 0) else 1.0

    # log-odds: observed overlap rate vs expected; guard against division by zero
    obs_rate = k / n if n > 0 else 0.0
    exp_rate = K / N if N > 0 else 0.0
    if obs_rate == 0 or exp_rate == 0:
        log_odds = 0.0
    else:
        import math
        log_odds = math.log2(obs_rate / exp_rate)

    return {"k": k, "K": K, "n": n, "N": N, "p_value": p_value, "log_odds": log_odds}


def gsea_preranked(
    ranked_genes: list[str],
    scores: np.ndarray,
    gene_set: list[str],
    n_permutations: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """GSEA-preranked enrichment (Subramanian et al. 2005).

    Algorithm:
    1. Sort genes by ``scores`` (descending); this is the pre-ranked list.
    2. Compute the running enrichment score (ES): walk the ranked list, adding
       ``+|score_i|^p / sum_hit`` when gene i is in the set and
       ``-1 / sum_miss`` otherwise (p=1 weighting).
    3. ES = maximum deviation from zero of the running sum.
    4. Permutation null: shuffle gene labels 1 000× and recompute ES each time.
    5. NES = ES / mean(|null_ES|); p = fraction of |null_ES| ≥ |ES|.

    Parameters
    ----------
    ranked_genes
        Gene symbols in any order; will be sorted by ``scores``.
    scores
        Score vector aligned to ``ranked_genes``
        (e.g. RL action frequency or Chronos).
    gene_set
        Reference gene set.
    n_permutations
        Number of permutations for the null (default 1 000).
    seed
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``{"ES": float, "NES": float, "p_value": float, "fdr": float}``.
        ``fdr`` is a conservative single-test estimate: ``min(p_value * n_tests, 1.0)``
        where n_tests=1; the caller applies BH correction across multiple gene sets.
    """
    scores  = np.asarray(scores, dtype=np.float64)
    order   = np.argsort(-scores)          # descending
    genes_sorted = [ranked_genes[i] for i in order]
    scores_sorted = scores[order]

    gset = set(gene_set)
    hits = np.array([g in gset for g in genes_sorted], dtype=bool)
    n_hit  = hits.sum()
    n_miss = len(hits) - n_hit

    if n_hit == 0 or n_miss == 0:
        return {"ES": 0.0, "NES": 0.0, "p_value": 1.0, "fdr": 1.0}

    def _running_es(hit_mask: np.ndarray, sc: np.ndarray) -> float:
        """Compute ES from a hit mask and score vector."""
        sum_hit = np.abs(sc[hit_mask]).sum()
        if sum_hit == 0:
            return 0.0
        n_miss_loc = (~hit_mask).sum()
        running = np.where(
            hit_mask,
            np.abs(sc) / sum_hit,
            -1.0 / max(n_miss_loc, 1),
        ).cumsum()
        return float(running[np.abs(running).argmax()])

    es_obs = _running_es(hits, scores_sorted)

    # Permutation null — shuffle hit membership, keep scores fixed
    rng = np.random.default_rng(seed)
    null_es = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_hits = np.zeros(len(hits), dtype=bool)
        perm_hits[rng.choice(len(hits), n_hit, replace=False)] = True
        null_es[i] = _running_es(perm_hits, scores_sorted)

    null_pos = null_es[null_es >= 0]
    null_neg = null_es[null_es < 0]

    if es_obs >= 0:
        mean_null = null_pos.mean() if len(null_pos) else 1.0
        p_value   = float((null_pos >= es_obs).mean()) if len(null_pos) else 1.0
    else:
        mean_null = null_neg.mean() if len(null_neg) else -1.0
        p_value   = float((null_neg <= es_obs).mean()) if len(null_neg) else 1.0

    nes = float(es_obs / abs(mean_null)) if mean_null != 0 else 0.0

    return {"ES": float(es_obs), "NES": nes, "p_value": p_value, "fdr": min(p_value, 1.0)}


def null_enrichment_comparison(
    observed_enrichment: float,
    selected_genes: list[str],
    background: list[str],
    n_null_samples: int = 1_000,
    match_expression: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare observed enrichment to random gene sets of matched size (+ optional expression).

    Two null strategies (DATA.md §5.3):
    - **Size-matched**: draw ``len(selected_genes)`` random genes from ``background``.
    - **Expression-matched**: if ``match_expression`` is provided, stratify background by
      expression decile and sample proportionally from the same decile bins as
      ``selected_genes``. Controls for HVG-selection bias.

    Parameters
    ----------
    observed_enrichment
        The observed statistic (e.g. hypergeometric log-odds).
    selected_genes
        The actual RL-selected gene set.
    background
        Universe of genes (aligned to ``match_expression`` if provided).
    n_null_samples
        Number of random matched sets.
    match_expression
        Optional per-gene mean expression aligned to ``background``.
    seed
        RNG seed.

    Returns
    -------
    dict
        ``{"z_score": float, "empirical_p": float, "null_mean": float, "null_std": float}``.
    """
    rng  = np.random.default_rng(seed)
    bg   = list(background)
    n    = min(len(selected_genes), len(bg))

    null_stats: list[float] = []

    if match_expression is not None and len(match_expression) == len(bg):
        # Expression-matched: bin background into deciles, sample from same bins
        expr = np.asarray(match_expression, dtype=np.float64)
        # Decile bin for each background gene
        bin_edges = np.percentile(expr, np.linspace(0, 100, 11))
        bin_edges[0] -= 1e-9  # include minimum
        bg_bins = np.digitize(expr, bin_edges) - 1  # 0..9

        # Determine which deciles the selected genes fall into
        sel_set = set(selected_genes)
        sel_idx = [i for i, g in enumerate(bg) if g in sel_set]
        sel_bins = bg_bins[sel_idx] if sel_idx else np.array([], dtype=int)
        bin_counts = np.bincount(sel_bins, minlength=10) if len(sel_bins) else np.zeros(10, dtype=int)

        for _ in range(n_null_samples):
            null_set: list[str] = []
            for b, cnt in enumerate(bin_counts):
                if cnt == 0:
                    continue
                candidates = np.where(bg_bins == b)[0]
                drawn = rng.choice(candidates, size=min(cnt, len(candidates)), replace=False)
                null_set.extend(bg[i] for i in drawn)
            # Compute hypergeometric log-odds of this null set vs background
            # (we recycle the same gene_set = selected_genes as the reference — proxy for enrichment)
            null_stats.append(float(len(set(null_set) & sel_set)) / max(len(null_set), 1))
    else:
        # Size-matched: uniform random sampling
        bg_arr = np.array(bg)
        sel_set = set(selected_genes)
        for _ in range(n_null_samples):
            null_genes = rng.choice(bg_arr, size=n, replace=False).tolist()
            null_stats.append(float(len(set(null_genes) & sel_set)) / max(n, 1))

    null_arr  = np.array(null_stats, dtype=np.float64)
    null_mean = float(null_arr.mean())
    null_std  = float(null_arr.std()) if null_arr.std() > 0 else 1.0
    z_score   = float((observed_enrichment - null_mean) / null_std)
    emp_p     = float((null_arr >= observed_enrichment).mean())

    return {
        "z_score":    z_score,
        "empirical_p": emp_p,
        "null_mean":  null_mean,
        "null_std":   null_std,
    }
