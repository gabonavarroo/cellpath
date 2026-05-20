"""V3C utility-audit aggregator (Phase 0B).

Reads every ``<field_id>/`` sub-audit JSON under
``artifacts_v3/v3c/utility_audit/`` and writes the cross-field matrices:

    prediction_metrics.csv         (Bucket U-A)
    reachability_matrix.csv        (Bucket U-B)
    greedy_saturation_matrix.csv   (Bucket U-C)
    depth_leverage_matrix.csv      (Bucket U-C derived)
    contraction_geometry.csv       (Bucket U-D, OOD pool)
    contraction_geometry_val_pairs.csv  (Bucket U-D, val pairs)
    action_heterogeneity.csv       (Bucket U-E)
    reward_leverage_fused.csv      (Bucket U-F)
    ppo_preconditions.csv          (Bucket U-G)
    utility_summary.md             (cross-bucket synthesis per field)
    candidate_ranking.md           (top-K ranking + flagged duplicates)

Guardrails honored:
- ``util_score`` is reported as a ranking aid only; the candidate-ranking
  markdown explicitly disclaims it as a verdict and reminds the human
  reader that written rationale is required (guardrail #1, #8).
- Duplicate detection per guardrail #4: fields whose (val_pearson,
  ridge_margin, model.pt md5) match are flagged in `candidate_ranking.md`
  so they do not consume separate smoke slots downstream.

See V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md §4 Stage 2, §10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.analysis.dynamics_utility import compute_utility_score

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_REL = "artifacts_v3/v3c/utility_audit/dynamics_inventory.csv"
AUDIT_ROOT_REL = "artifacts_v3/v3c/utility_audit"

LOG = logging.getLogger("v3c.aggregate")

CANONICAL_CELL_IDS = (
    "k2_bin6-8_splitood", "k2_bin8-10_splitood",
    "k3_bin6-8_splitood", "k3_bin8-10_splitood",
    "k4_bin8-10_splitood", "k5_bin8-10_splitood",
    "k8_bin8-10_splitood",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        LOG.warning("Failed to read %s: %s", p, exc)
        return None


def _model_md5(field_path: str) -> str | None:
    p = REPO_ROOT / field_path / "model.pt"
    if not p.exists():
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-bucket flatteners — each returns a list of row dicts (one row / field).
# ---------------------------------------------------------------------------


def _row_u_a(field_id: str, audit_dir: Path) -> dict[str, Any] | None:
    payload = _load_json(audit_dir / "prediction_metrics.json")
    if payload is None:
        return None
    return {
        "field_id": field_id,
        "status": payload.get("status"),
        "val_pearson": payload.get("val_pearson"),
        "ood_pearson": payload.get("ood_pearson"),
        "ridge_margin": payload.get("ridge_margin"),
        "uncertainty_spearman": payload.get("uncertainty_spearman"),
        "per_dim_pearson_median": payload.get("per_dim_pearson_median"),
        "per_dim_pearson_p10": payload.get("per_dim_pearson_p10"),
        "gate_passed": payload.get("gate_passed"),
        "prediction_pathological_flag": payload.get("prediction_pathological_flag"),
    }


def _row_u_b(field_id: str, audit_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(audit_dir / "reachability.json")
    if payload is None:
        return []
    rows: list[dict[str, Any]] = []
    epsilon = payload.get("epsilon")
    for cell in payload.get("cells", []):
        rows.append({
            "field_id": field_id,
            "cell_id": cell.get("cell_id"),
            "K": cell.get("K"),
            "epsilon": epsilon,
            "beam_reach_at_K_p15": cell.get("beam_reach_at_K_p15"),
            "n_pool": cell.get("n_pool"),
            "n_episodes": cell.get("n_episodes"),
            "status": cell.get("status"),
        })
    return rows


def _row_u_c(field_id: str, audit_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(audit_dir / "greedy_saturation.json")
    if payload is None:
        return []
    rows: list[dict[str, Any]] = []
    for cell in payload.get("cells", []):
        if cell.get("status") != "ok":
            rows.append({"field_id": field_id, "cell_id": cell.get("cell_id"),
                         "status": cell.get("status")})
            continue
        depths = cell.get("depths", {})
        row: dict[str, Any] = {
            "field_id": field_id,
            "cell_id": cell.get("cell_id"),
            "K": cell.get("K"),
            "status": "ok",
        }
        for depth_str in ("1", "2", "3", "5", "8"):
            for mode in ("distance", "fused"):
                bucket = depths.get(depth_str, {}).get(mode)
                row[f"g{depth_str}_{mode}_success"] = (bucket or {}).get("success_rate")
        lev = cell.get("depth_leverage", {})
        for k in ("g2_minus_g1", "g3_minus_g2", "g5_minus_g3", "g8_minus_g5",
                  "cumulative_K_max_minus_K1"):
            row[k] = lev.get(k)
        rows.append(row)
    return rows


def _row_u_d(field_id: str, audit_dir: Path, sample_key: str) -> dict[str, Any] | None:
    payload = _load_json(audit_dir / "contraction_geometry.json")
    if payload is None:
        return None
    sub = payload.get(sample_key) or {}
    if not sub or sub.get("status") not in (None, "ok"):
        return {"field_id": field_id, "status": sub.get("status")}
    return {
        "field_id": field_id,
        "status": "ok",
        "sample_label": sub.get("sample_label"),
        "n_starts": sub.get("n_starts"),
        "n_genes": sub.get("n_genes"),
        "contraction_fraction": sub.get("contraction_fraction"),
        "alignment_cos_median": sub.get("alignment_cos_median"),
        "action_diversity_per_state": sub.get("action_diversity_per_state"),
        "state_diversity_per_action": sub.get("state_diversity_per_action"),
        "gene_universality_max": sub.get("gene_universality_max"),
        "gene_universality_gini": sub.get("gene_universality_gini"),
        "null_gene_fraction": sub.get("null_gene_fraction"),
        "delta_magnitude_median": sub.get("delta_magnitude_median"),
    }


def _row_u_e(field_id: str, audit_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(audit_dir / "action_heterogeneity.json")
    if payload is None:
        return []
    rows: list[dict[str, Any]] = []
    for cell in payload.get("cells", []):
        rows.append({
            "field_id": field_id,
            "cell_id": cell.get("cell_id"),
            "status": cell.get("status") or "ok",
            "first_action_entropy_distance": cell.get("first_action_entropy_distance"),
            "first_action_entropy_fused": cell.get("first_action_entropy_fused"),
            "first_action_entropy_max_nats": cell.get("first_action_entropy_max_nats"),
            "first_action_top1_freq_fused": cell.get("first_action_top1_freq_fused"),
            "first_action_top5_freq_fused": cell.get("first_action_top5_freq_fused"),
            "first_action_gini_fused": cell.get("first_action_gini_fused"),
            "distance_vs_fused_first_action_overlap": cell.get("distance_vs_fused_first_action_overlap"),
            "path_diversity_depth2_distance": cell.get("path_diversity_depth2_distance"),
            "path_diversity_depth3_distance": cell.get("path_diversity_depth3_distance"),
        })
    return rows


def _row_u_f(field_id: str, audit_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(audit_dir / "reward_leverage_fused.json")
    if payload is None:
        return []
    rows: list[dict[str, Any]] = []
    norman = payload.get("norman_combo_realism") or {}
    for cell in payload.get("cells", []):
        rows.append({
            "field_id": field_id,
            "cell_id": cell.get("cell_id"),
            "status": cell.get("status") or "ok",
            "delta_success_fused_minus_distance": cell.get("delta_success_fused_minus_distance"),
            "delta_final_distance": cell.get("delta_final_distance"),
            "delta_T_at_success": cell.get("delta_T_at_success"),
            "delta_tox_path": cell.get("delta_tox_path"),
            "delta_common_essential_count": cell.get("delta_common_essential_count"),
            "delta_unc_path_max": cell.get("delta_unc_path_max"),
            "pareto_axes_improved": cell.get("pareto_axes_improved"),
            "pareto_signal": cell.get("pareto_signal"),
            "concern_over_shaped": cell.get("concern_over_shaped"),
            "norman_combo_status": norman.get("status"),
            "fraction_paths_with_measured_combo_overlap": norman.get("fraction_paths_with_measured_combo_overlap"),
            "measured_combo_latent_consistency": norman.get("measured_combo_latent_consistency"),
        })
    return rows


def _row_u_g(field_id: str, audit_dir: Path) -> dict[str, Any] | None:
    payload = _load_json(audit_dir / "ppo_preconditions.json")
    if payload is None:
        return None
    return {"field_id": field_id, **payload}


# ---------------------------------------------------------------------------
# Duplicate detection (guardrail #4)
# ---------------------------------------------------------------------------


def _detect_duplicates(rows_a: list[dict[str, Any]], inventory: pl.DataFrame) -> dict[str, list[str]]:
    """Group fields with identical (val_pearson, ridge_margin, model.pt md5).

    Returns ``{representative_field_id: [duplicate_field_ids...]}`` (rep
    excluded from its own duplicate list).
    """
    by_key: dict[tuple[Any, Any, str | None], list[str]] = {}
    for row in rows_a:
        fid = row["field_id"]
        inv_row = inventory.filter(pl.col("field_id") == fid)
        if inv_row.height == 0:
            continue
        field_path = str(inv_row["path"][0])
        md5 = _model_md5(field_path)
        key = (row.get("val_pearson"), row.get("ridge_margin"), md5)
        by_key.setdefault(key, []).append(fid)
    out: dict[str, list[str]] = {}
    for key, group in by_key.items():
        if len(group) < 2:
            continue
        group_sorted = sorted(group)
        rep = group_sorted[0]
        out[rep] = group_sorted[1:]
    return out


# ---------------------------------------------------------------------------
# Top-line summary writers
# ---------------------------------------------------------------------------


def _compute_util_scores(rows_by_id: dict[str, dict[str, dict[str, Any]]]) -> dict[str, float | None]:
    """Per-field util_score using ``compute_utility_score`` with allow_missing=True."""
    out: dict[str, float | None] = {}
    for field_id, sub in rows_by_id.items():
        buckets: dict[str, dict[str, Any]] = {}
        u_a = sub.get("u_a")
        if u_a and u_a.get("status") == "ok":
            buckets["u_a"] = {"val_pearson": u_a.get("val_pearson", 0.0)}
        u_b = sub.get("u_b_cells") or []
        by_id = {r.get("cell_id"): r for r in u_b if r.get("status") == "ok"}
        if by_id:
            buckets["u_b"] = {
                "beam_reach_at_K4_bin8_10_p15": (by_id.get("k4_bin8-10_splitood") or {}).get("beam_reach_at_K_p15"),
                "beam_reach_at_K5_bin8_10_p15": (by_id.get("k5_bin8-10_splitood") or {}).get("beam_reach_at_K_p15"),
                "beam_reach_at_K8_bin8_10_p15": (by_id.get("k8_bin8-10_splitood") or {}).get("beam_reach_at_K_p15"),
            }
        u_c_cells = sub.get("u_c_cells") or []
        cum_vals = [r.get("cumulative_K_max_minus_K1") for r in u_c_cells if r.get("status") == "ok"]
        cum_vals = [v for v in cum_vals if v is not None]
        if cum_vals:
            buckets["u_c"] = {"cumulative_depth_leverage": float(max(cum_vals))}
        u_d = sub.get("u_d_ood")
        if u_d and u_d.get("status") == "ok":
            buckets["u_d"] = {
                "contraction_fraction": u_d.get("contraction_fraction"),
                "gene_universality_max": u_d.get("gene_universality_max"),
            }
        u_e_cells = sub.get("u_e_cells") or []
        primary = next(
            (r for r in u_e_cells if r.get("cell_id") == "k3_bin8-10_splitood" and r.get("first_action_entropy_fused") is not None),
            None,
        )
        if primary:
            buckets["u_e"] = {
                "first_action_entropy_fused": primary.get("first_action_entropy_fused"),
                "first_action_entropy_max_nats": primary.get("first_action_entropy_max_nats"),
            }
        ov_vals = [r.get("distance_vs_fused_first_action_overlap") for r in u_e_cells]
        ov_vals = [v for v in ov_vals if v is not None]
        if ov_vals:
            buckets["u_f"] = {"distance_vs_fused_first_action_overlap": float(np.median(ov_vals))}
        u_g = sub.get("u_g")
        buckets["u_g"] = {"all_preconditions_pass": bool((u_g or {}).get("all_preconditions_pass", False))}
        out[field_id] = compute_utility_score(buckets, allow_missing=True)
    return out


def _write_summary_md(
    path: Path,
    util_scores: dict[str, float | None],
    inventory: pl.DataFrame,
    duplicates: dict[str, list[str]],
) -> None:
    inv_rows = {r["field_id"]: r for r in inventory.to_dicts()}
    lines = [
        "# V3C dynamics utility audit summary",
        "",
        "_Aggregator output. `util_score` is a **ranking aid only** — it ranks",
        "the inventory for triage; it never decides which fields get PPO smoke._",
        "_Smoke-target selection requires written qualitative rationale citing",
        "specific Bucket U-A through U-G evidence (V3C plan §4 Stage 3 /",
        "guardrails #1, #8)._",
        "",
        "## Fields by `util_score` (ranking aid)",
        "",
        "| Field | n_lat | pair | RoR | λ_corr | util_score | audit_class | duplicate_of |",
        "|---|---:|---|:-:|---:|---:|---|---|",
    ]
    dup_lookup: dict[str, str] = {}
    for rep, members in duplicates.items():
        for m in members:
            dup_lookup[m] = rep
    rows_sorted = sorted(
        util_scores.items(),
        key=lambda kv: (-(kv[1] if kv[1] is not None else -1.0)),
    )
    for field_id, score in rows_sorted:
        inv = inv_rows.get(field_id, {})
        dup = dup_lookup.get(field_id, "")
        score_str = f"{score:.4f}" if isinstance(score, float) else "_n/a_"
        lines.append(
            f"| `{inv.get('path','?')}` | {inv.get('n_latent','?')} | "
            f"{inv.get('pair_source','?')} | {'✓' if inv.get('ror_flag') else ''} | "
            f"{inv.get('lambda_corr',0.0):.2f} | {score_str} | "
            f"{inv.get('audit_class','?')} | {dup} |"
        )
    lines += [
        "",
        "## Duplicate clusters (guardrail #4)",
        "",
    ]
    if duplicates:
        for rep, members in sorted(duplicates.items()):
            lines.append(f"- `{rep}` ≡ {', '.join(f'`{m}`' for m in members)}")
        lines.append("")
        lines.append("Duplicates share `val_pearson`, `ridge_margin`, and `model.pt` md5.")
        lines.append("Each cluster should only consume **one** smoke slot in Phase 1.")
    else:
        lines.append("(none detected)")
    lines.append("")
    lines += [
        "## Next-step protocol (V3C plan §4 Stage 3)",
        "",
        "Selection of smoke targets is interpretive, not algorithmic. Each of",
        "the **four** smoke slots (Anchor + 1–2 Best-by-audit + 1–2 Wildcards)",
        "must be documented in `artifacts_v3/v3c/interpretation/v3c_phase0_utility_audit.md`",
        "with written rationale citing specific U-bucket evidence. A high",
        "`util_score` alone is not sufficient rationale.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_candidate_ranking_md(
    path: Path,
    util_scores: dict[str, float | None],
    inventory: pl.DataFrame,
    sub_by_field: dict[str, dict[str, dict[str, Any]]],
    duplicates: dict[str, list[str]],
) -> None:
    inv_rows = {r["field_id"]: r for r in inventory.to_dicts()}
    dup_lookup: dict[str, str] = {}
    for rep, members in duplicates.items():
        for m in members:
            dup_lookup[m] = rep

    def _flags(field_id: str) -> list[str]:
        sub = sub_by_field.get(field_id, {})
        u_a = sub.get("u_a") or {}
        u_d = sub.get("u_d_ood") or {}
        u_d_vp = sub.get("u_d_val") or {}
        u_g = sub.get("u_g") or {}
        out: list[str] = []
        if u_a.get("prediction_pathological_flag"):
            out.append("PREDICTION_PATHOLOGICAL")
        if u_a.get("gate_passed"):
            out.append("GATE_PASSED")
        if u_d.get("contraction_fraction") is not None and u_d.get("contraction_fraction", 0) > 0.95:
            out.append("CONTRACTION_NEAR_UNIVERSAL")
        if u_d.get("contraction_fraction") is not None and u_d.get("contraction_fraction", 0) < 0.30:
            out.append("CONTRACTION_LOW_or_BARYCENTRIC")
        if u_d.get("gene_universality_max") is not None and u_d.get("gene_universality_max", 0) > 0.85:
            out.append("UNIVERSAL_ATTRACTOR_GENE")
        if u_d_vp.get("contraction_fraction") is not None and u_d.get("contraction_fraction") is not None:
            div = abs(float(u_d_vp["contraction_fraction"]) - float(u_d["contraction_fraction"]))
            if div > 0.30:
                out.append(f"GEOMETRY_DIVERGENCE_{div:.2f}")
        if u_g.get("all_preconditions_pass"):
            out.append("U-G_ALL_PASS")
        if field_id in dup_lookup:
            out.append(f"DUPLICATE_OF[{dup_lookup[field_id]}]")
        return out

    lines = [
        "# V3C candidate ranking and wildcard flags",
        "",
        "_Per-field flags surface unusual signatures that the human researcher_",
        "_should weigh when picking smoke targets. The audit ranks/explains/flags;_",
        "_smoke-target selection is interpretive and requires written rationale._",
        "",
        "## Fields with U-G all-pass (Best-by-audit candidates)",
        "",
    ]
    # Sort fields by util_score desc, partition into pass vs not
    rows_sorted = sorted(
        util_scores.items(),
        key=lambda kv: (-(kv[1] if kv[1] is not None else -1.0)),
    )
    pass_rows = [(fid, sc) for fid, sc in rows_sorted
                 if (sub_by_field.get(fid, {}).get("u_g") or {}).get("all_preconditions_pass")]
    fail_rows = [(fid, sc) for fid, sc in rows_sorted
                 if not (sub_by_field.get(fid, {}).get("u_g") or {}).get("all_preconditions_pass")]
    for fid, sc in pass_rows:
        inv = inv_rows.get(fid, {})
        score_str = f"{sc:.3f}" if isinstance(sc, float) else "n/a"
        flags = ", ".join(_flags(fid)) or "_(no flags)_"
        lines.append(f"- **{score_str}** `{inv.get('path','?')}` — {flags}")
    if not pass_rows:
        lines.append("_(no fields pass all 7 U-G preconditions — wildcard route applies to top candidates)_")
    lines += [
        "",
        "## Fields below U-G (Wildcard candidates — see flags for promotion rationale)",
        "",
    ]
    for fid, sc in fail_rows:
        inv = inv_rows.get(fid, {})
        score_str = f"{sc:.3f}" if isinstance(sc, float) else "n/a"
        flags = ", ".join(_flags(fid)) or "_(no flags)_"
        lines.append(f"- **{score_str}** `{inv.get('path','?')}` — {flags}")
    lines += [
        "",
        "## Anchor",
        "",
        "`artifacts_v2/dynamics_v1ot_ror_corr010` is **always** in the Phase 1",
        "smoke roster regardless of `util_score` (V3C plan §4 Stage 3).",
        "",
        "## Reminder",
        "",
        "- `util_score` is a ranking aid. **Not** a verdict.",
        "- Smoke-target selection requires written qualitative rationale.",
        "- Duplicate clusters consume **one** smoke slot, not many.",
        "- 64D fields use their own latent/pair files (guardrail #8) and a",
        "  per-VAE p15 ε; their reachability is not directly comparable to 32D.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V3C utility-audit aggregator")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG,
                        format="[%(levelname)s] %(message)s")
    root: Path = args.root.resolve()
    audit_root = root / AUDIT_ROOT_REL

    inventory_path = root / INVENTORY_REL
    if not inventory_path.exists():
        LOG.error("missing inventory: %s", inventory_path)
        return 1
    inventory = pl.read_csv(inventory_path)

    # Walk per-field sub-audit directories
    rows_u_a: list[dict[str, Any]] = []
    rows_u_b: list[dict[str, Any]] = []
    rows_u_c: list[dict[str, Any]] = []
    rows_u_d_ood: list[dict[str, Any]] = []
    rows_u_d_val: list[dict[str, Any]] = []
    rows_u_e: list[dict[str, Any]] = []
    rows_u_f: list[dict[str, Any]] = []
    rows_u_g: list[dict[str, Any]] = []
    sub_by_field: dict[str, dict[str, Any]] = {}

    field_dirs = sorted([p for p in audit_root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    LOG.info("found %d audited field directories under %s", len(field_dirs), audit_root)

    for fd in field_dirs:
        field_id = fd.name
        u_a = _row_u_a(field_id, fd)
        u_b_cells = _row_u_b(field_id, fd)
        u_c_cells = _row_u_c(field_id, fd)
        u_d_ood = _row_u_d(field_id, fd, "sample1_ood_pool")
        u_d_val = _row_u_d(field_id, fd, "sample2_val_pairs")
        u_e_cells = _row_u_e(field_id, fd)
        u_f_cells = _row_u_f(field_id, fd)
        u_g = _row_u_g(field_id, fd)

        if u_a:
            rows_u_a.append(u_a)
        rows_u_b.extend(u_b_cells)
        rows_u_c.extend(u_c_cells)
        if u_d_ood:
            rows_u_d_ood.append(u_d_ood)
        if u_d_val:
            rows_u_d_val.append(u_d_val)
        rows_u_e.extend(u_e_cells)
        rows_u_f.extend(u_f_cells)
        if u_g:
            rows_u_g.append(u_g)

        sub_by_field[field_id] = {
            "u_a": u_a, "u_b_cells": u_b_cells, "u_c_cells": u_c_cells,
            "u_d_ood": u_d_ood, "u_d_val": u_d_val,
            "u_e_cells": u_e_cells, "u_f_cells": u_f_cells, "u_g": u_g,
        }

    audit_root.mkdir(parents=True, exist_ok=True)

    def _write(rows: list[dict[str, Any]], name: str) -> None:
        target = audit_root / name
        if not rows:
            target.write_text("")  # empty file is informative
            LOG.info("no rows for %s — empty file", name)
            return
        # Use lenient schema inference (some columns may be all-null)
        pl.DataFrame(rows, strict=False).write_csv(str(target))
        LOG.info("wrote %s (%d rows)", name, len(rows))

    _write(rows_u_a, "prediction_metrics.csv")
    _write(rows_u_b, "reachability_matrix.csv")
    _write(rows_u_c, "greedy_saturation_matrix.csv")
    _write(rows_u_d_ood, "contraction_geometry.csv")
    _write(rows_u_d_val, "contraction_geometry_val_pairs.csv")
    _write(rows_u_e, "action_heterogeneity.csv")
    _write(rows_u_f, "reward_leverage_fused.csv")
    _write(rows_u_g, "ppo_preconditions.csv")

    util_scores = _compute_util_scores(sub_by_field)
    duplicates = _detect_duplicates(rows_u_a, inventory)

    _write_summary_md(audit_root / "utility_summary.md", util_scores, inventory, duplicates)
    _write_candidate_ranking_md(audit_root / "candidate_ranking.md", util_scores, inventory, sub_by_field, duplicates)

    # Single-file JSON dump for downstream consumption
    payload = {
        "field_count": len(field_dirs),
        "util_scores": util_scores,
        "duplicate_clusters": duplicates,
    }
    (audit_root / "aggregate_index.json").write_text(
        json.dumps(payload, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x)
    )

    LOG.info("aggregator complete — see utility_summary.md / candidate_ranking.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
