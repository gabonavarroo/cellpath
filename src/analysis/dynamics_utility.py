"""V3C Bucket U — dynamics-utility metrics.

All Bucket U sub-bucket computations (U-D contraction geometry, U-E action
heterogeneity, U-F reward leverage + Norman-combo realism, util_score
composite) live here so that any script — audit driver, aggregator, ad-hoc
notebook — calls the same canonical implementation. Sacred-rule §4 of
CLAUDE.md.

Naming: V3C sub-buckets are U-A through U-G to avoid collision with the
V3B-legacy Bucket A/B/C/D meanings.

See V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md §3 / §4.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Dynamics protocol — anything callable with the PerturbationDynamicsModel
# forward-signature contract works here (real model OR a synthetic stand-in
# for tests).
# ---------------------------------------------------------------------------


class DynamicsCallable(Protocol):
    def __call__(
        self,
        z: torch.Tensor,
        gene_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


def _to_torch_batch(z_np: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(z_np, dtype=np.float32), dtype=torch.float32)


# ---------------------------------------------------------------------------
# U-D — contraction geometry
# ---------------------------------------------------------------------------


def _gini(values: np.ndarray) -> float:
    """Gini coefficient over a non-negative array; 0 = uniform, 1 = single point."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or float(arr.sum()) <= 0.0:
        return 0.0
    arr = np.sort(np.clip(arr, 0.0, None))
    n = arr.size
    cum = np.cumsum(arr)
    # Standard formula: (2 * Σ i·x_i) / (n · Σ x_i) − (n+1)/n
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * arr)) / (n * cum[-1]) - (n + 1.0) / n)


def compute_contraction_geometry(
    *,
    dynamics: DynamicsCallable,
    z_starts: np.ndarray,
    z_ref: np.ndarray,
    n_genes: int,
    sample_label: str,
    gene_idx_subset: np.ndarray | None = None,
    batch_size: int = 4096,
    null_mag_eps: float = 0.01,
) -> dict[str, Any]:
    """Compute Bucket U-D contraction-geometry metrics on a (z, g) sample.

    Parameters
    ----------
    dynamics
        Frozen dynamics callable with signature ``f(z, gene_idx) → (z_next, μ, log_var)``.
        ``gene_idx`` is **1-indexed** per PerturbationDynamicsModel's contract.
    z_starts
        ``(N, n_latent)`` array of start latents.
    z_ref
        ``(n_latent,)`` reference centroid.
    n_genes
        Action-space width excluding NO-OP. The (z, g) grid is ``N × n_genes``.
    sample_label
        Free-text tag ("ood_pool" or "val_pairs"); echoed in the output for
        downstream divergence reporting.
    gene_idx_subset
        Optional 1-indexed gene IDs to score (default: every gene 1..n_genes).
    batch_size
        Mini-batch size for forward passes; lower if memory-bound.
    null_mag_eps
        Below this fraction of the median ``||μ||``, a gene is considered
        "null" for ``null_gene_fraction``.

    Returns
    -------
    dict
        See test_dynamics_utility.TestContractionGeometry for the required key
        set and semantics. NaN/null returned for metrics that are undefined
        (e.g. ``contraction_fraction`` when ``||μ||`` is identically zero).
    """
    z_starts = np.asarray(z_starts, dtype=np.float32)
    z_ref = np.asarray(z_ref, dtype=np.float32)
    if z_starts.ndim != 2 or z_starts.shape[1] != z_ref.shape[0]:
        raise ValueError(
            f"z_starts shape {z_starts.shape} incompatible with z_ref shape {z_ref.shape}"
        )

    n_starts = z_starts.shape[0]
    if gene_idx_subset is None:
        genes = np.arange(1, n_genes + 1, dtype=np.int64)
    else:
        genes = np.asarray(gene_idx_subset, dtype=np.int64).reshape(-1)
    g_count = genes.shape[0]

    target_dir = z_ref[None, :] - z_starts                          # (N, D)
    target_norm = np.linalg.norm(target_dir, axis=1, keepdims=True) + 1e-12

    # Per-(state, gene) cosine alignment and ||μ||
    alignment = np.zeros((n_starts, g_count), dtype=np.float64)
    mu_norm = np.zeros((n_starts, g_count), dtype=np.float64)

    # Batched forward pass: stack (z, g) over states for one gene at a time
    # to avoid an O(N·G·D) tensor materialization.
    z_tensor = _to_torch_batch(z_starts)
    with torch.no_grad():
        for gi, g in enumerate(genes.tolist()):
            g_batch = torch.full((n_starts,), int(g), dtype=torch.long)
            mus_chunks: list[torch.Tensor] = []
            for start in range(0, n_starts, batch_size):
                end = min(start + batch_size, n_starts)
                _, mu, _ = dynamics(z_tensor[start:end], g_batch[start:end])
                mus_chunks.append(mu.detach().cpu())
            mu_np = torch.cat(mus_chunks, dim=0).numpy().astype(np.float64)
            mu_n = np.linalg.norm(mu_np, axis=1, keepdims=True)
            mu_norm[:, gi] = mu_n[:, 0]
            denom = (mu_n[:, 0] * target_norm[:, 0]) + 1e-12
            alignment[:, gi] = (mu_np * target_dir).sum(axis=1) / denom

    # Per-(state, gene) "contraction" is positive dot-product of μ with (z_ref − z).
    contraction_mask = ((alignment > 0.0) & (mu_norm > 0.0))

    median_mu_norm = float(np.median(mu_norm))
    p10_mu_norm = float(np.quantile(mu_norm, 0.10))
    p90_mu_norm = float(np.quantile(mu_norm, 0.90))

    # Mask zero-magnitude (z, g) pairs out of alignment statistics — cosine is
    # undefined when μ = 0. We report null_gene_fraction separately.
    nonzero_mask = mu_norm > (null_mag_eps * max(median_mu_norm, 1e-9))
    if not nonzero_mask.any():
        # No-op field: every gene effectively does nothing.
        contraction_fraction: float | None = None
        align_median = align_p25 = align_p75 = None
        frac_above = frac_below = None
        action_diversity_per_state = 0.0
        state_diversity_per_action = 0.0
        gene_universality_max = 0.0
        gene_universality_gini = 0.0
    else:
        flat_align = alignment[nonzero_mask]
        contraction_fraction = float(contraction_mask.mean())
        align_median = float(np.median(flat_align))
        align_p25 = float(np.quantile(flat_align, 0.25))
        align_p75 = float(np.quantile(flat_align, 0.75))
        frac_above = float(np.mean(flat_align > 0.5))
        frac_below = float(np.mean(flat_align < 0.0))

        # Per-state std-over-genes — captures "do all genes do the same thing
        # in this state?"
        action_diversity_per_state = float(np.median(np.std(alignment, axis=1)))
        # Per-gene std-over-states — captures "is this gene a universal
        # attractor or universally null?"
        state_diversity_per_action = float(np.median(np.std(alignment, axis=0)))
        gene_means = alignment.mean(axis=0)                          # (G,)
        gene_universality_max = float(np.max(gene_means))
        gene_universality_gini = _gini(np.abs(gene_means))

    # Per-gene null detection: mean ||μ|| across states relative to global median
    mean_mag_per_gene = mu_norm.mean(axis=0)                         # (G,)
    threshold = null_mag_eps * max(median_mu_norm, 1e-9)
    if median_mu_norm <= 1e-9:
        # Whole field is no-op — every gene counts as null.
        null_gene_fraction = 1.0
    else:
        null_gene_fraction = float(np.mean(mean_mag_per_gene < threshold))

    per_gene_mean_alignment = [
        {"gene_idx": int(g), "mean_alignment": float(gene_means[i] if nonzero_mask.any() else 0.0)}
        for i, g in enumerate(genes.tolist())
    ]

    return {
        "sample_label": str(sample_label),
        "n_starts": int(n_starts),
        "n_genes": int(g_count),
        "alignment_cos_median": align_median,
        "alignment_cos_p25": align_p25,
        "alignment_cos_p75": align_p75,
        "alignment_cos_frac_above_0_5": frac_above,
        "alignment_cos_frac_below_0": frac_below,
        "contraction_fraction": contraction_fraction,
        "delta_magnitude_median": median_mu_norm,
        "delta_magnitude_p10": p10_mu_norm,
        "delta_magnitude_p90": p90_mu_norm,
        "action_diversity_per_state": action_diversity_per_state,
        "state_diversity_per_action": state_diversity_per_action,
        "gene_universality_max": gene_universality_max,
        "gene_universality_gini": gene_universality_gini,
        "null_gene_fraction": null_gene_fraction,
        "per_gene_mean_alignment": per_gene_mean_alignment,
    }


# ---------------------------------------------------------------------------
# U-E — action heterogeneity and path diversity
# ---------------------------------------------------------------------------


def _shannon_entropy_nats(counts: np.ndarray) -> float:
    """Natural-log Shannon entropy of a count vector. 0 ≤ H ≤ log(K)."""
    arr = np.asarray(counts, dtype=np.float64)
    total = arr.sum()
    if total <= 0.0:
        return 0.0
    p = arr / total
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum())


def _topk_freq(counts: np.ndarray, k: int) -> float:
    """Fraction of mass in the top-k bins."""
    arr = np.asarray(counts, dtype=np.float64)
    total = arr.sum()
    if total <= 0.0:
        return 0.0
    sorted_desc = np.sort(arr)[::-1]
    return float(sorted_desc[:k].sum() / total)


def _action_counts(actions: np.ndarray, n_genes: int) -> np.ndarray:
    """Bin a 1-indexed action array (NOOP ignored if present) into a count vector."""
    counts = np.zeros(n_genes, dtype=np.int64)
    arr = np.asarray(actions, dtype=np.int64).reshape(-1)
    for a in arr.tolist():
        if 1 <= a <= n_genes:
            counts[a - 1] += 1
    return counts


def compute_action_heterogeneity(
    *,
    n_genes: int,
    first_actions_distance: np.ndarray,
    first_actions_fused: np.ndarray,
    beam_plans_depth2_distance: list[tuple[int, ...]],
    beam_plans_depth3_distance: list[tuple[int, ...]],
    top_k_report: int = 10,
) -> dict[str, Any]:
    """Compute Bucket U-E action heterogeneity from greedy rollout records.

    Inputs are 1-indexed gene actions captured by the greedy sub-audit; NOOPs
    are filtered out (NOOP is not a gene action and would skew the entropy
    estimate). Beam plans are tuples of 1-indexed actions per start state.
    """
    first_d = np.asarray(first_actions_distance, dtype=np.int64).reshape(-1)
    first_f = np.asarray(first_actions_fused, dtype=np.int64).reshape(-1)
    n = int(first_d.shape[0])
    if first_f.shape[0] != n:
        raise ValueError("first_actions_distance and first_actions_fused length mismatch")

    counts_d = _action_counts(first_d, n_genes)
    counts_f = _action_counts(first_f, n_genes)

    overlap_mask = first_d == first_f
    # Restrict overlap to (gene, gene) pairs — NOOP-NOOP starts are uninformative.
    valid_pair = ((first_d >= 1) & (first_d <= n_genes) & (first_f >= 1) & (first_f <= n_genes))
    n_valid = int(valid_pair.sum())
    if n_valid == 0:
        overlap = None
    else:
        overlap = float((overlap_mask & valid_pair).sum() / n_valid)

    h_distance = _shannon_entropy_nats(counts_d)
    h_fused = _shannon_entropy_nats(counts_f)

    top1_f = _topk_freq(counts_f, 1)
    top5_f = _topk_freq(counts_f, 5)
    top10_f = _topk_freq(counts_f, min(10, n_genes))

    gini_f = _gini(counts_f.astype(np.float64))

    # Path-plan diversity: unique-tuple count / n_starts
    def _diversity(plans: list[tuple[int, ...]]) -> float:
        if not plans:
            return 0.0
        # Normalize each plan to a tuple-of-ints
        norm = [tuple(int(x) for x in p) for p in plans]
        return float(len(set(norm)) / max(len(norm), 1))

    div_d2 = _diversity(beam_plans_depth2_distance)
    div_d3 = _diversity(beam_plans_depth3_distance)

    # Top-K genes by fused first-action frequency
    rank = np.argsort(counts_f)[::-1]
    top10_genes = [
        {"gene_idx": int(rank[i] + 1), "count": int(counts_f[rank[i]])}
        for i in range(min(top_k_report, n_genes))
    ]

    return {
        "n_starts": n,
        "n_genes": int(n_genes),
        "first_action_entropy_distance": h_distance,
        "first_action_entropy_fused": h_fused,
        "first_action_entropy_max_nats": float(np.log(max(n_genes, 1))),
        "first_action_top1_freq_fused": top1_f,
        "first_action_top5_freq_fused": top5_f,
        "first_action_top10_freq_fused": top10_f,
        "first_action_gini_fused": gini_f,
        "distance_vs_fused_first_action_overlap": overlap,
        "path_diversity_depth2_distance": div_d2,
        "path_diversity_depth3_distance": div_d3,
        "top10_genes_fused": top10_genes,
    }


# ---------------------------------------------------------------------------
# U-F — reward leverage under locked B+C+D
# ---------------------------------------------------------------------------


# Pareto verdict tunables — match V3C plan §4 Stage 4 / §13 item 8.
PARETO_RAW_SUCCESS_TOLERANCE = 0.03    # |Δ raw success| ≤ this counts as "tied"
PARETO_FINAL_DISTANCE_REGRESSION = 0.10  # |Δ final_distance| ≤ this is acceptable
PARETO_AXES_THRESHOLD = 2              # need ≥ this many axes improved


def compute_reward_leverage(
    *,
    cell_id: str,
    rollouts_distance: dict[str, float],
    rollouts_fused: dict[str, float],
) -> dict[str, Any]:
    """Compute Bucket U-F reward-leverage tabulation for one cell.

    Inputs are per-cell aggregate dicts from the greedy sub-audit (each must
    expose ``success_rate, mean_final_distance, mean_T_at_success,
    mean_tox_path, mean_common_essential_count, mean_unc_path_max``).
    Output includes the Pareto-signal flag — the V3C-locked verdict
    ingredient that credits "reward-axis improvement without success
    regression" (§4 Stage 4 `CANDIDATE_SIGNAL_PARETO`).
    """
    def _g(d: dict[str, float], k: str) -> float:
        v = d.get(k)
        if v is None:
            return float("nan")
        return float(v)

    deltas = {
        "delta_success_fused_minus_distance": _g(rollouts_fused, "success_rate") - _g(rollouts_distance, "success_rate"),
        "delta_final_distance": _g(rollouts_fused, "mean_final_distance") - _g(rollouts_distance, "mean_final_distance"),
        "delta_T_at_success": _g(rollouts_fused, "mean_T_at_success") - _g(rollouts_distance, "mean_T_at_success"),
        "delta_tox_path": _g(rollouts_fused, "mean_tox_path") - _g(rollouts_distance, "mean_tox_path"),
        "delta_common_essential_count": _g(rollouts_fused, "mean_common_essential_count") - _g(rollouts_distance, "mean_common_essential_count"),
        "delta_unc_path_max": _g(rollouts_fused, "mean_unc_path_max") - _g(rollouts_distance, "mean_unc_path_max"),
    }

    # An "improvement" on a safety/uncertainty/path axis is a negative delta
    # (lower tox / lower CE count / lower uncertainty / shorter T at success).
    axes_improvements = [
        deltas["delta_tox_path"] < -1e-9,
        deltas["delta_common_essential_count"] < -1e-9,
        deltas["delta_unc_path_max"] < -1e-9,
        deltas["delta_T_at_success"] < -1e-9,
    ]
    pareto_axes_improved = int(sum(1 for a in axes_improvements if a))

    delta_success = deltas["delta_success_fused_minus_distance"]
    delta_dist = deltas["delta_final_distance"]
    raw_success_within_tol = bool(abs(delta_success) <= PARETO_RAW_SUCCESS_TOLERANCE + 1e-9)
    final_distance_ok = bool(delta_dist <= PARETO_FINAL_DISTANCE_REGRESSION + 1e-9)

    pareto_signal = bool(
        raw_success_within_tol
        and final_distance_ok
        and pareto_axes_improved >= PARETO_AXES_THRESHOLD
    )

    # "Over-shaped": fused field collapses raw success AND distance regresses
    # significantly — reward shaping has gone too far on this dynamics.
    concern_over_shaped = bool(
        delta_success < -PARETO_RAW_SUCCESS_TOLERANCE
        and delta_dist > PARETO_FINAL_DISTANCE_REGRESSION
    )

    out: dict[str, Any] = {
        "cell_id": str(cell_id),
        **deltas,
        "pareto_axes_improved": pareto_axes_improved,
        "raw_success_within_pareto_tolerance": raw_success_within_tol,
        "final_distance_not_regressed": final_distance_ok,
        "pareto_signal": pareto_signal,
        "concern_over_shaped": concern_over_shaped,
        "thresholds": {
            "raw_success_tolerance": PARETO_RAW_SUCCESS_TOLERANCE,
            "final_distance_regression_max": PARETO_FINAL_DISTANCE_REGRESSION,
            "axes_threshold": PARETO_AXES_THRESHOLD,
        },
    }
    return out


# ---------------------------------------------------------------------------
# U-F (cont.) — Norman measured-combo realism diagnostic
# ---------------------------------------------------------------------------


def compute_norman_combo_consistency(
    *,
    plans: list[dict[str, Any]],
    measured_combos: dict[str, np.ndarray] | None,
    check_ordered: bool = True,
) -> dict[str, Any]:
    """Bucket U-F realism diagnostic: planner / greedy 2-step path agreement with Norman 2019 combos.

    Parameters
    ----------
    plans
        List of dicts with keys ``"path"`` (tuple of 1-indexed gene ids,
        length ≥ 2) and ``"z_predicted_post"`` (the dynamics-predicted
        latent after rolling out the first two steps). Plans with `<2`
        actions are ignored (NOOP-only trajectories).
    measured_combos
        Norman ``combo_pairs.npz`` payload (keys ``gene_idx_a``, ``gene_idx_b``,
        ``z_ctrl``, ``z_pert_ab``), or ``None`` if combo data is unavailable
        (e.g. mean_delta / soft_ot / 64D fields without a matching combo file).
    check_ordered
        If True, ``(a, b)`` must match ``(measured_a, measured_b)`` exactly.
        If False, ``(a, b)`` and ``(b, a)`` are both accepted (gene-pair as
        unordered set).
    """
    if measured_combos is None:
        return {
            "status": "no_combo_data",
            "fraction_paths_with_measured_combo_overlap": None,
            "n_overlapping_paths": 0,
            "measured_combo_latent_consistency": None,
            "measured_combo_distance_consistency": None,
        }

    a_arr = np.asarray(measured_combos["gene_idx_a"], dtype=np.int64).reshape(-1)
    b_arr = np.asarray(measured_combos["gene_idx_b"], dtype=np.int64).reshape(-1)
    z_pert_ab = np.asarray(measured_combos["z_pert_ab"], dtype=np.float32)

    if check_ordered:
        measured_set: dict[tuple[int, int], list[int]] = {}
        for i, (a, b) in enumerate(zip(a_arr.tolist(), b_arr.tolist())):
            measured_set.setdefault((int(a), int(b)), []).append(i)
    else:
        measured_set = {}
        for i, (a, b) in enumerate(zip(a_arr.tolist(), b_arr.tolist())):
            key = tuple(sorted((int(a), int(b))))
            measured_set.setdefault(key, []).append(i)

    cos_scores: list[float] = []
    dist_scores: list[float] = []
    n_overlap = 0
    n_paths = 0
    for plan in plans:
        path = tuple(int(x) for x in plan.get("path", ()))
        if len(path) < 2:
            continue
        n_paths += 1
        key = path[:2] if check_ordered else tuple(sorted(path[:2]))
        idxs = measured_set.get(key)
        if not idxs:
            continue
        n_overlap += 1
        # Compare predicted post-combo latent to the empirical Norman
        # post-combo latent (mean across all cells observed for this combo).
        empirical = z_pert_ab[idxs].mean(axis=0)
        predicted = np.asarray(plan["z_predicted_post"], dtype=np.float32).reshape(-1)
        if predicted.shape != empirical.shape:
            # Latent-dim mismatch — combo data not compatible with this field.
            continue
        denom = float(np.linalg.norm(predicted) * np.linalg.norm(empirical))
        if denom <= 1e-12:
            cos = 0.0
        else:
            cos = float(predicted @ empirical / denom)
        cos_scores.append(cos)
        dist_scores.append(float(np.linalg.norm(predicted - empirical)))

    if n_paths == 0:
        frac = None
    else:
        frac = float(n_overlap / n_paths)

    return {
        "status": "ok",
        "fraction_paths_with_measured_combo_overlap": frac,
        "n_overlapping_paths": n_overlap,
        "n_total_paths": n_paths,
        "measured_combo_latent_consistency": (
            float(np.mean(cos_scores)) if cos_scores else None
        ),
        "measured_combo_distance_consistency": (
            float(np.mean(dist_scores)) if dist_scores else None
        ),
    }


# ---------------------------------------------------------------------------
# util_score composite — strictly a ranking aid (§4 Stage 1)
# ---------------------------------------------------------------------------


UTIL_SCORE_WEIGHTS: dict[str, float] = {
    "u_a": 0.20,
    "u_b": 0.20,
    "u_c": 0.20,
    "u_d": 0.15,
    "u_e": 0.10,
    "u_f": 0.10,
    "u_g": 0.05,
}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _coalesce_float(bucket: dict[str, Any], key: str, default: float) -> float:
    """Fetch a numeric value, treating only ``None`` (not ``0.0``) as missing."""
    v = bucket.get(key)
    return float(default) if v is None else float(v)


def _u_a_score(bucket: dict[str, Any]) -> float:
    return _clip01(_coalesce_float(bucket, "val_pearson", 0.0))


def _u_b_score(bucket: dict[str, Any]) -> float:
    keys = ("beam_reach_at_K4_bin8_10_p15", "beam_reach_at_K5_bin8_10_p15", "beam_reach_at_K8_bin8_10_p15")
    vals = [bucket.get(k) for k in keys]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return 0.0
    return _clip01(float(np.mean(vals)))


def _u_c_score(bucket: dict[str, Any]) -> float:
    return _clip01(_coalesce_float(bucket, "cumulative_depth_leverage", 0.0) / 0.5)


def _u_d_score(bucket: dict[str, Any]) -> float:
    cf = _coalesce_float(bucket, "contraction_fraction", 0.0)
    gu = _coalesce_float(bucket, "gene_universality_max", 1.0)
    return _clip01(cf * (1.0 - gu))


def _u_e_score(bucket: dict[str, Any]) -> float:
    h = _coalesce_float(bucket, "first_action_entropy_fused", 0.0)
    h_max = _coalesce_float(bucket, "first_action_entropy_max_nats", 0.0)
    if h_max <= 0.0:
        return 0.0
    return _clip01(h / h_max)


def _u_f_score(bucket: dict[str, Any]) -> float:
    overlap = _coalesce_float(bucket, "distance_vs_fused_first_action_overlap", 1.0)
    return _clip01(1.0 - overlap)


def _u_g_score(bucket: dict[str, Any]) -> float:
    return 1.0 if bucket.get("all_preconditions_pass", False) else 0.0


_SCORERS: dict[str, Callable[[dict[str, Any]], float]] = {
    "u_a": _u_a_score,
    "u_b": _u_b_score,
    "u_c": _u_c_score,
    "u_d": _u_d_score,
    "u_e": _u_e_score,
    "u_f": _u_f_score,
    "u_g": _u_g_score,
}


def compute_utility_score(
    buckets: dict[str, dict[str, Any]],
    *,
    allow_missing: bool = False,
) -> float | None:
    """Composite Bucket-U utility score.

    **This is a ranking aid, not a verdict.** A field's `util_score` orders
    the inventory for triage in `aggregate_v3c_utility_audit.py`, but the
    actual Phase 1 smoke-target selection (Anchor / Best-by-audit /
    Wildcards) requires *written qualitative rationale* per V3C plan §4
    Stage 3 / guardrail #1. A high score is not a smoke ticket; a low
    score does not exclude wildcard promotion.

    Weights are fixed and sum to 1.0 (see ``UTIL_SCORE_WEIGHTS``). The
    formula is documented in V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md
    §4 Stage 1.

    Parameters
    ----------
    buckets
        Dict keyed by ``"u_a"`` through ``"u_g"`` whose values are the
        bucket-summary dicts produced by the per-bucket audit functions.
    allow_missing
        When False (default), a missing bucket returns ``None`` — we do not
        silently pretend a bucket was zero. When True, the score is
        renormalized over present buckets (use only when a caller has an
        explicit reason to score partial fields, e.g. mid-execution
        previews).

    Returns
    -------
    float in [0, 1] or ``None`` if required buckets are missing and
    ``allow_missing`` is False.
    """
    missing = [b for b in UTIL_SCORE_WEIGHTS if b not in buckets]
    if missing and not allow_missing:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for key, weight in UTIL_SCORE_WEIGHTS.items():
        if key not in buckets:
            continue
        weighted_sum += weight * _SCORERS[key](buckets[key])
        weight_total += weight
    if weight_total <= 0.0:
        return None
    return float(weighted_sum / weight_total)


def contraction_divergence(
    sample1: dict[str, Any], sample2: dict[str, Any]
) -> dict[str, Any]:
    """Compare U-D output across two sample distributions (OOD pool vs val pairs).

    Per V3C plan §3 U-D: ``|contraction_fraction(sample1) − contraction_fraction(sample2)|``
    above 0.30 flags the field as geometrically inconsistent between
    distributions.
    """
    def _maybe(v: Any) -> float | None:
        return None if v is None else float(v)

    cf1, cf2 = _maybe(sample1.get("contraction_fraction")), _maybe(sample2.get("contraction_fraction"))
    if cf1 is None or cf2 is None:
        cf_div: float | None = None
    else:
        cf_div = abs(cf1 - cf2)
    return {
        "sample1_label": sample1.get("sample_label"),
        "sample2_label": sample2.get("sample_label"),
        "contraction_fraction_divergence": cf_div,
        "flag_significant_divergence": bool(cf_div is not None and cf_div > 0.30),
    }
