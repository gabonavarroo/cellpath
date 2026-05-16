"""Result aggregator for the CellPath MVP V1 freeze.

Pure-function library — no Hydra, no file globbing. The Hydra wrapper lives in
``scripts/aggregate_eval.py``; figure rendering lives in ``scripts/visualize.py``;
end-to-end evaluation lives in ``scripts/evaluate.py``. All three depend on this module
as the single source of truth for the composite ``summary.json``.

This module consumes — but does not modify — the JSON artifacts already on disk:

- per-run RL ``summary.json`` produced by :mod:`scripts.summarize_rl_run` (with embedded
  ``metadata.json`` + ``delta_vs_random`` blocks);
- contraction ``summary.json`` + ``metadata.json`` produced by
  :mod:`scripts.diagnose_dynamics_contraction`;
- dynamics ``gate.json`` + ``ood_metrics.json`` + ``checkpoint_comparison.json`` produced
  by :mod:`scripts.train_dynamics`;
- per-run ``action_freq.json`` produced by both PPO eval and the random-policy script.

Outputs (written by the wrapper, not here) are all under ``artifacts/eval/``:

- ``summary.json`` — composite blob keyed by ``provenance``, ``rl``, ``dynamics``,
  ``contraction``, ``ablation_64d``, ``top_actions``.
- ``results_table.md`` — defense-ready Markdown (one table per section).
- ``caveats.md`` — gate-override + contraction + 64D-rejected + biological-honesty.

Sacred constraints (enforced in :func:`build_caveats_md`):
- never assert gate passed; always show ``gate_overridden`` if true;
- never claim biological discovery / therapeutic reprogramming;
- treat 64D as a documented negative ablation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# =============================================================================
# Loaders (tolerant — missing files are warned, never crashed)
# =============================================================================


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    """Read JSON; return None if the path is missing/unreadable/None."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not parse %s: %s", p, exc)
        return None


def load_rl_run_summary(run_dir: str | Path | None) -> dict[str, Any] | None:
    """Read ``<run_dir>/summary.json`` and normalize to the wrapped shape.

    Two on-disk shapes exist:

    1. **Wrapped** (``scripts/summarize_rl_run.py``):
       ``{"run_dir": ..., "metadata": {...}, "metrics": {...},
          "random_baseline": {...}?, "delta_vs_random": {...}?}``
    2. **Flat** (``scripts/run_random_policy.py``):
       ``{"policy_kind": ..., "n_episodes": N, "success_rate": ..., ...}``

    Both are accepted; the flat form is upgraded to the wrapped form so downstream
    consumers see a single schema. A flat summary additionally tries to read the run's
    ``metadata.json`` if present (the random-policy script writes one).
    """
    if run_dir is None:
        return None
    run_dir = Path(run_dir)
    blob = _load_json(run_dir / "summary.json")
    if blob is None:
        return None
    if "metrics" in blob:
        return blob  # already wrapped

    # Flat → wrapped. Map the fields run_random_policy.py emits to the canonical metrics
    # shape. Missing fields (action entropy, count vs failures distinction) are set to None.
    n_episodes = blob.get("n_episodes")
    successes = blob.get("successes")
    failures = blob.get("failures")
    if successes is None and n_episodes is not None and blob.get("success_rate") is not None:
        successes = int(round(float(blob["success_rate"]) * int(n_episodes)))
        failures = int(n_episodes) - successes
    metrics = {
        "n_episodes": n_episodes,
        "successes": successes,
        "failures": failures,
        "success_rate": blob.get("success_rate"),
        "mean_steps": blob.get("mean_steps"),
        "mean_total_reward": blob.get("mean_total_reward"),
        "mean_final_distance": blob.get("mean_final_distance"),
        "mean_min_distance": blob.get("mean_min_distance"),
        # run_random_policy.py writes ..._failures and ..._rate but not _count;
        # the count is at most the failure count + the success-NO_OP count, which we
        # don't have. Surface what we have and leave the rest as None.
        "noop_first_action_count": None,
        "noop_first_action_failures": blob.get("noop_first_action_failures"),
        "noop_first_action_rate": blob.get("noop_first_action_rate"),
        "action_entropy_nats": None,
        "action_entropy_normalized": None,
        "top_actions": [],
    }
    return {
        "run_dir": str(run_dir),
        "metadata": _load_json(run_dir / "metadata.json") or {},
        "metrics": metrics,
    }


def load_rl_action_freq(run_dir: str | Path | None) -> dict[str, int] | None:
    """Read ``<run_dir>/action_freq.json`` (gene_symbol → count)."""
    if run_dir is None:
        return None
    blob = _load_json(Path(run_dir) / "action_freq.json")
    if blob is None:
        return None
    return {str(k): int(v) for k, v in blob.items()}


def load_contraction_summary(contraction_dir: str | Path | None) -> dict[str, Any] | None:
    """Read ``<contraction_dir>/summary.json`` + merge select fields from ``metadata.json``."""
    if contraction_dir is None:
        return None
    d = Path(contraction_dir)
    summary = _load_json(d / "summary.json")
    if summary is None:
        return None
    meta = _load_json(d / "metadata.json") or {}
    out = dict(summary)
    # Expose the provenance fields needed for headline tables / captions.
    diag_block = meta.get("diagnostic", {}) if isinstance(meta, dict) else {}
    out["min_start_distance_used"] = diag_block.get("min_start_distance_used")
    out["epsilon_used"] = diag_block.get("epsilon_used")
    out["epsilon_source"] = diag_block.get("epsilon_source")
    out["vae_n_latent"] = meta.get("vae_n_latent")
    out["dynamics_use_state_linear_skip"] = (
        (meta.get("dynamics_arch") or {}).get("use_state_linear_skip")
    )
    out["dynamics_use_gene_delta_bias"] = (
        (meta.get("dynamics_arch") or {}).get("use_gene_delta_bias")
    )
    out["dynamics_checkpoint"] = meta.get("dynamics_checkpoint")
    out["dynamics_checkpoint_sha256"] = meta.get("dynamics_checkpoint_sha256")
    return out


def load_dynamics_gate(gate_path: str | Path | None) -> dict[str, Any] | None:
    """Read ``gate.json``. Returns None if missing."""
    return _load_json(gate_path)


# =============================================================================
# Section builders (pure, JSON-safe)
# =============================================================================


def _round(x: Any, ndigits: int = 6) -> Any:
    """Round numerics for JSON output; pass through non-numeric values."""
    if isinstance(x, bool):
        return x  # bool is a subclass of int — handle first
    if isinstance(x, (int, float)):
        try:
            return round(float(x), ndigits)
        except (TypeError, ValueError):
            return x
    return x


def build_provenance_section(
    ppo_det_summary: dict[str, Any] | None,
    contraction_32d: dict[str, Any] | None,
    gate_32d: dict[str, Any] | None,
) -> dict[str, Any]:
    """Lift the most authoritative provenance fields from existing artifacts.

    Preference order: PPO det metadata → contraction metadata → gate.json. We pick the
    PPO det run first because it is the headline result; the others provide fallbacks
    when only a subset of artifacts exist.
    """
    meta = (ppo_det_summary or {}).get("metadata") or {}
    contra_meta = contraction_32d or {}

    def _first(*vals: Any) -> Any:
        for v in vals:
            if v is not None:
                return v
        return None

    return {
        "schema_version": 1,
        "timestamp_utc": meta.get("timestamp_utc"),
        "git_commit": meta.get("git_commit"),
        "vae_n_latent": _first(meta.get("vae_n_latent"), contra_meta.get("vae_n_latent")),
        "vae_dir": meta.get("vae_dir"),
        "epsilon_value": _first(meta.get("epsilon_value"), contra_meta.get("epsilon_used")),
        "epsilon_source": _first(meta.get("epsilon_source"), contra_meta.get("epsilon_source")),
        "epsilon_percentile_json": meta.get("epsilon_percentile_json"),
        "min_start_distance": _first(
            meta.get("min_start_distance"),
            contra_meta.get("min_start_distance_used"),
        ),
        "max_steps": meta.get("max_steps"),
        "dynamics_checkpoint": _first(
            meta.get("dynamics_checkpoint"),
            contra_meta.get("dynamics_checkpoint"),
        ),
        "dynamics_checkpoint_sha256": _first(
            meta.get("dynamics_checkpoint_sha256"),
            contra_meta.get("dynamics_checkpoint_sha256"),
        ),
        "dynamics_gate_passed": _first(
            meta.get("dynamics_gate_passed"),
            (gate_32d or {}).get("passed"),
        ),
        "dynamics_gate_overridden": meta.get("dynamics_gate_overridden"),
        "dynamics_arch": meta.get("dynamics_arch"),
        "policy_path": meta.get("policy_path"),
    }


def _extract_run_metrics(run_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pull just the headline RL metrics from a summarize_rl_run summary."""
    if run_summary is None:
        return None
    m = run_summary.get("metrics") or {}
    return {
        "n_episodes": m.get("n_episodes"),
        "successes": m.get("successes"),
        "failures": m.get("failures"),
        "success_rate": _round(m.get("success_rate"), 4),
        "mean_steps": _round(m.get("mean_steps"), 4),
        "mean_total_reward": _round(m.get("mean_total_reward"), 4),
        "mean_final_distance": _round(m.get("mean_final_distance"), 4),
        "mean_min_distance": _round(m.get("mean_min_distance"), 4),
        "noop_first_action_count": m.get("noop_first_action_count"),
        "noop_first_action_failures": m.get("noop_first_action_failures"),
        "noop_first_action_rate": _round(m.get("noop_first_action_rate"), 4),
        "action_entropy_nats": _round(m.get("action_entropy_nats"), 4),
        "action_entropy_normalized": _round(m.get("action_entropy_normalized"), 4),
        "top_actions": m.get("top_actions") or [],
    }


def build_rl_section(
    ppo_det_summary: dict[str, Any] | None,
    ppo_stoch_summary: dict[str, Any] | None,
    random_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compose the RL section: per-run metrics + Δ-vs-random.

    The Δ block prefers what ``summarize_rl_run`` already computed (when the PPO det run
    was summarized with ``--random-baseline-dir``). If absent, we compute it here from
    the two metric dicts.
    """
    det = _extract_run_metrics(ppo_det_summary)
    stoch = _extract_run_metrics(ppo_stoch_summary)
    # The random-policy summary file mirrors the PPO eval schema, so the same extractor works.
    rand = _extract_run_metrics(random_summary)

    # Prefer pre-computed delta from the det run summary if present.
    delta = None
    if ppo_det_summary is not None and "delta_vs_random" in ppo_det_summary:
        d = ppo_det_summary["delta_vs_random"]
        delta = {
            "delta_success_rate": _round(d.get("delta_success_rate"), 4),
            "delta_success_pp": _round(d.get("delta_success_pp"), 2),
            "delta_mean_steps": _round(d.get("delta_mean_steps"), 4),
            "delta_mean_final_distance": _round(d.get("delta_mean_final_distance"), 4),
        }
    elif det is not None and rand is not None:
        dsr = (det["success_rate"] or 0.0) - (rand["success_rate"] or 0.0)
        delta = {
            "delta_success_rate": _round(dsr, 4),
            "delta_success_pp": _round(100.0 * dsr, 2),
            "delta_mean_steps": _round(
                (det["mean_steps"] or 0.0) - (rand["mean_steps"] or 0.0), 4
            ),
            "delta_mean_final_distance": _round(
                (det["mean_final_distance"] or 0.0) - (rand["mean_final_distance"] or 0.0), 4
            ),
        }

    return {
        "ppo_deterministic": det,
        "ppo_stochastic": stoch,
        "random_baseline": rand,
        "delta_ppo_det_vs_random": delta,
    }


def _gate_primary_view(gate: dict[str, Any] | None) -> dict[str, Any] | None:
    """Flatten ``gate.json`` primary+OOD into a viewer-friendly shape."""
    if gate is None:
        return None
    primary = gate.get("primary") or {}
    ood = gate.get("ood") or {}

    def _summary(block: dict[str, Any]) -> dict[str, Any]:
        bls = block.get("baselines") or {}
        mc = block.get("margin_checks") or {}
        return {
            "mlp_r2": _round(block.get("r2"), 4),
            "mlp_pearson": _round(block.get("pearson_r"), 4),
            "ridge_r2": _round((bls.get("linear_ridge") or {}).get("r2"), 4),
            "ridge_pearson": _round((bls.get("linear_ridge") or {}).get("pearson_r"), 4),
            "per_gene_mean_r2": _round(
                (bls.get("per_gene_mean_delta") or {}).get("r2"), 4
            ),
            "knn_pearson": _round(
                (bls.get("nearest_neighbor") or {}).get("pearson_r"), 4
            ),
            "margin_vs_linear_ridge_pearson": _round(
                (mc.get("margin_vs_linear_ridge_pearson") or {}).get("value"), 4
            ),
            "margin_vs_linear_ridge_pearson_threshold": (
                (mc.get("margin_vs_linear_ridge_pearson") or {}).get("threshold")
            ),
            "margin_vs_linear_ridge_pearson_pass": (
                (mc.get("margin_vs_linear_ridge_pearson") or {}).get("pass")
            ),
        }

    return {
        "passed": bool(gate.get("passed", False)),
        "primary": _summary(primary),
        "ood": _summary(ood),
        "uncertainty_calibration_spearman": _round(
            (gate.get("uncertainty_calibration") or {}).get("spearman"), 4
        ),
        "uncertainty_calibration_ood_spearman": _round(
            (gate.get("uncertainty_calibration_ood") or {}).get("spearman"), 4
        ),
        "margins_used": gate.get("margins_used") or {},
    }


def build_dynamics_section(
    gate_32d: dict[str, Any] | None,
    gate_64d: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compose the dynamics-gate section + cross-arch margin deltas."""
    primary = _gate_primary_view(gate_32d)
    ablation = _gate_primary_view(gate_64d)

    delta = None
    if primary is not None and ablation is not None:
        delta = {
            "margin_val_pearson_32d_minus_64d": _round(
                (primary["primary"]["margin_vs_linear_ridge_pearson"] or 0.0)
                - (ablation["primary"]["margin_vs_linear_ridge_pearson"] or 0.0),
                4,
            ),
            "ood_pearson_32d_minus_64d": _round(
                (primary["ood"]["mlp_pearson"] or 0.0)
                - (ablation["ood"]["mlp_pearson"] or 0.0),
                4,
            ),
        }
    return {
        "primary_32d": primary,
        "ablation_64d": ablation,
        "delta_32d_vs_64d": delta,
    }


def _extract_contraction_row(label: str, contra: dict[str, Any] | None) -> dict[str, Any] | None:
    """Flatten a contraction summary into a single labelled row."""
    if contra is None:
        return None
    return {
        "label": label,
        "vae_n_latent": contra.get("vae_n_latent"),
        "use_state_linear_skip": contra.get("dynamics_use_state_linear_skip"),
        "use_gene_delta_bias": contra.get("dynamics_use_gene_delta_bias"),
        "min_start_distance_used": contra.get("min_start_distance_used"),
        "epsilon_used": _round(contra.get("epsilon_used"), 4),
        "n_starts": contra.get("n_starts"),
        "n_genes": contra.get("n_genes"),
        "n_pairs": contra.get("n_pairs"),
        "fraction_improved": _round(contra.get("fraction_improved"), 4),
        "mean_improvement": _round(contra.get("mean_improvement"), 4),
        "median_improvement": _round(contra.get("median_improvement"), 4),
        "best_improvement": _round(contra.get("best_improvement"), 4),
        "worst_improvement": _round(contra.get("worst_improvement"), 4),
        "p25_improvement": _round(contra.get("p25_improvement"), 4),
        "p75_improvement": _round(contra.get("p75_improvement"), 4),
        "std_improvement": _round(contra.get("std_improvement"), 4),
    }


def build_contraction_section(rows: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Compose the contraction section. ``rows`` maps label → loaded summary dict (or None).

    Convention for labels (preserved through the table render):
      - ``primary_32d``        — 32D, min_start_distance=8.0 (start8).
      - ``primary_32d_auto``   — 32D, min_start_distance="auto" (= epsilon).
      - ``ablation_64d``       — 64D state_linear, start8.
      - ``ablation_64d_auto``  — 64D state_linear, auto.
      - ``ablation_64d_plain`` — 64D baseline_plain (no state_linear), start8.
    """
    out: dict[str, Any] = {}
    for label, contra in rows.items():
        row = _extract_contraction_row(label, contra)
        if row is not None:
            out[label] = row
    return out


def build_top_actions_section(
    ppo_freq: dict[str, int] | None,
    random_freq: dict[str, int] | None,
    *,
    top_k: int = 15,
) -> dict[str, Any]:
    """Cross-tabulate PPO top-K vs random top-K.

    Returns a JSON-safe dict containing the two sorted lists plus their intersection /
    set differences — useful for caption text and the action-frequency figure.
    """
    def _top(d: dict[str, int] | None) -> list[dict[str, Any]]:
        if not d:
            return []
        items = sorted(d.items(), key=lambda kv: -kv[1])[:top_k]
        return [{"gene_symbol": k, "count": int(v)} for k, v in items]

    ppo_top = _top(ppo_freq)
    rand_top = _top(random_freq)
    ppo_set = {row["gene_symbol"] for row in ppo_top}
    rand_set = {row["gene_symbol"] for row in rand_top}
    return {
        "top_k": top_k,
        "ppo": ppo_top,
        "random": rand_top,
        "intersection": sorted(ppo_set & rand_set),
        "only_in_ppo": sorted(ppo_set - rand_set),
        "only_in_random": sorted(rand_set - ppo_set),
    }


# =============================================================================
# Top-level orchestrator
# =============================================================================


def build_summary(
    *,
    ppo_det_summary: dict[str, Any] | None,
    ppo_stoch_summary: dict[str, Any] | None,
    random_summary: dict[str, Any] | None,
    ppo_action_freq: dict[str, int] | None,
    random_action_freq: dict[str, int] | None,
    contraction_rows: dict[str, dict[str, Any] | None],
    gate_32d: dict[str, Any] | None,
    gate_64d: dict[str, Any] | None,
    top_k: int = 15,
) -> dict[str, Any]:
    """Compose the full ``artifacts/eval/summary.json`` body from already-loaded JSON.

    All inputs are passed in pre-loaded for testability. The Hydra wrapper does the
    path resolution + I/O.
    """
    return {
        "provenance": build_provenance_section(
            ppo_det_summary, contraction_rows.get("primary_32d"), gate_32d,
        ),
        "rl": build_rl_section(ppo_det_summary, ppo_stoch_summary, random_summary),
        "dynamics": build_dynamics_section(gate_32d, gate_64d),
        "contraction": build_contraction_section(contraction_rows),
        "top_actions": build_top_actions_section(
            ppo_action_freq, random_action_freq, top_k=top_k,
        ),
    }


# =============================================================================
# Markdown renderers (defense-ready)
# =============================================================================


_OVERRIDE_CAVEAT = (
    "**Caveat:** Dynamics validation gate failed and was overridden "
    "(`rl.train.skip_gate=true`). RL numbers are valid as learned-control results over the "
    "trained surrogate; they are **not** validated against the dynamics-gate criterion."
)


def _fmt(x: Any, *, ndigits: int = 3, dash: str = "—") -> str:
    """Format a numeric for Markdown; render ``None`` as ``—``."""
    if x is None:
        return dash
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        if isinstance(x, int):
            return str(x)
        try:
            return f"{float(x):.{ndigits}f}"
        except (TypeError, ValueError):
            return dash
    return str(x)


def _render_rl_table(rl: dict[str, Any]) -> str:
    """Two-table render: per-run headline + Δ-vs-random."""
    lines: list[str] = []
    lines.append("### RL evaluation (matched env: p50 ε override, `min_start_distance=8.0`)")
    lines.append("")
    lines.append("| run | n_episodes | success_rate | mean_steps | mean_final_distance | NO-OP first-action rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, key in [
        ("PPO deterministic", "ppo_deterministic"),
        ("PPO stochastic", "ppo_stochastic"),
        ("Random uniform-valid", "random_baseline"),
    ]:
        m = rl.get(key) or {}
        lines.append(
            f"| {label} | {_fmt(m.get('n_episodes'))} | "
            f"**{_fmt(m.get('success_rate'))}** | "
            f"{_fmt(m.get('mean_steps'), ndigits=2)} | "
            f"{_fmt(m.get('mean_final_distance'))} | "
            f"{_fmt(m.get('noop_first_action_rate'))} |"
        )
    delta = rl.get("delta_ppo_det_vs_random")
    if delta:
        lines.append("")
        lines.append("**Δ PPO det − random:** "
                     f"success **{_fmt(delta.get('delta_success_pp'), ndigits=1)} pp**, "
                     f"mean_steps {_fmt(delta.get('delta_mean_steps'), ndigits=2)}, "
                     f"mean_final_distance {_fmt(delta.get('delta_mean_final_distance'))}.")
    return "\n".join(lines)


def _render_dynamics_table(dyn: dict[str, Any]) -> str:
    """Dynamics-gate table: primary + OOD margins, MLP vs ridge, for 32D and 64D."""
    lines: list[str] = []
    lines.append("### Dynamics validation gate (primary = held-out cells; OOD = held-out genes)")
    lines.append("")
    lines.append("| split | MLP Pearson | Ridge Pearson | Margin (val) | Threshold | Pass? |")
    lines.append("| --- | ---: | ---: | ---: | ---: | :---: |")
    for label, key in [("32D", "primary_32d"), ("64D ablation", "ablation_64d")]:
        block = dyn.get(key)
        if not block:
            continue
        prim = block.get("primary") or {}
        ood = block.get("ood") or {}
        lines.append(
            f"| {label} val "
            f"| {_fmt(prim.get('mlp_pearson'), ndigits=4)} "
            f"| {_fmt(prim.get('ridge_pearson'), ndigits=4)} "
            f"| {_fmt(prim.get('margin_vs_linear_ridge_pearson'), ndigits=4)} "
            f"| {_fmt(prim.get('margin_vs_linear_ridge_pearson_threshold'), ndigits=3)} "
            f"| {_fmt(prim.get('margin_vs_linear_ridge_pearson_pass'))} |"
        )
        lines.append(
            f"| {label} OOD "
            f"| {_fmt(ood.get('mlp_pearson'), ndigits=4)} "
            f"| {_fmt(ood.get('ridge_pearson'), ndigits=4)} "
            f"| {_fmt(ood.get('margin_vs_linear_ridge_pearson'), ndigits=4)} "
            f"| {_fmt(ood.get('margin_vs_linear_ridge_pearson_threshold'), ndigits=3)} "
            f"| {_fmt(ood.get('margin_vs_linear_ridge_pearson_pass'))} |"
        )
    delta = dyn.get("delta_32d_vs_64d")
    if delta:
        lines.append("")
        lines.append(
            "**Δ 32D − 64D:** "
            f"val Pearson margin {_fmt(delta.get('margin_val_pearson_32d_minus_64d'), ndigits=4)}; "
            f"OOD Pearson {_fmt(delta.get('ood_pearson_32d_minus_64d'), ndigits=4)}."
        )
    return "\n".join(lines)


def _render_contraction_table(contra: dict[str, Any]) -> str:
    """Contraction rows in a defense-ready order."""
    order = [
        ("primary_32d", "32D state_linear, start8"),
        ("primary_32d_auto", "32D state_linear, auto"),
        ("ablation_64d", "64D state_linear, start8"),
        ("ablation_64d_auto", "64D state_linear, auto"),
        ("ablation_64d_plain", "64D baseline_plain, start8"),
    ]
    lines: list[str] = []
    lines.append("### Dynamics contraction diagnostic")
    lines.append("")
    lines.append("| variant | n_pairs | fraction_improved | mean_improvement | median | best | worst |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, label in order:
        row = contra.get(key)
        if not row:
            continue
        lines.append(
            f"| {label} | {_fmt(row.get('n_pairs'))} "
            f"| **{_fmt(row.get('fraction_improved'), ndigits=4)}** "
            f"| {_fmt(row.get('mean_improvement'))} "
            f"| {_fmt(row.get('median_improvement'))} "
            f"| {_fmt(row.get('best_improvement'))} "
            f"| {_fmt(row.get('worst_improvement'))} |"
        )
    return "\n".join(lines)


def _render_top_actions_table(top: dict[str, Any]) -> str:
    """Side-by-side top-K table for PPO vs random."""
    if not top:
        return ""
    k = int(top.get("top_k") or 0)
    ppo = top.get("ppo") or []
    rand = top.get("random") or []
    rows = max(len(ppo), len(rand))
    if rows == 0:
        return ""
    lines: list[str] = []
    lines.append(f"### Top-{k} actions (gene CRISPRa choices)")
    lines.append("")
    lines.append("| rank | PPO det gene | PPO count | random gene | random count |")
    lines.append("| ---: | --- | ---: | --- | ---: |")
    for i in range(rows):
        p = ppo[i] if i < len(ppo) else {}
        r = rand[i] if i < len(rand) else {}
        lines.append(
            f"| {i+1} | {p.get('gene_symbol', '—')} | {_fmt(p.get('count'))} "
            f"| {r.get('gene_symbol', '—')} | {_fmt(r.get('count'))} |"
        )
    if top.get("intersection"):
        lines.append("")
        lines.append("**Top-K overlap (PPO ∩ random):** " + ", ".join(top["intersection"]))
    return "\n".join(lines)


def build_results_table_md(summary: dict[str, Any]) -> str:
    """Render the composite ``summary.json`` to a defense-ready Markdown document.

    The structure mirrors ``summary.json``: provenance → RL → dynamics → contraction →
    top actions. The override caveat appears at the top of the document.
    """
    prov = summary.get("provenance") or {}
    lines: list[str] = []
    lines.append("# CellPath — MVP V1 results")
    lines.append("")
    lines.append(_OVERRIDE_CAVEAT)
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- timestamp_utc: `{prov.get('timestamp_utc')}`")
    lines.append(f"- git_commit: `{prov.get('git_commit')}`")
    lines.append(f"- VAE n_latent: {prov.get('vae_n_latent')}")
    lines.append(
        f"- ε used: {_fmt(prov.get('epsilon_value'), ndigits=4)} "
        f"(source `{prov.get('epsilon_source')}`, JSON percentile "
        f"{prov.get('epsilon_percentile_json')})"
    )
    lines.append(f"- min_start_distance: `{prov.get('min_start_distance')}`")
    lines.append(f"- dynamics checkpoint: `{prov.get('dynamics_checkpoint')}`")
    sha = prov.get("dynamics_checkpoint_sha256")
    if sha:
        lines.append(f"- dynamics checkpoint SHA-256: `{sha[:16]}…`")
    lines.append(
        f"- dynamics gate passed: **{_fmt(prov.get('dynamics_gate_passed'))}** "
        f"(overridden: **{_fmt(prov.get('dynamics_gate_overridden'))}**)"
    )
    arch = prov.get("dynamics_arch") or {}
    if arch:
        lines.append(
            f"- dynamics arch: `state_linear={_fmt(arch.get('use_state_linear_skip'))}, "
            f"gene_delta={_fmt(arch.get('use_gene_delta_bias'))}, "
            f"n_hidden={arch.get('n_hidden')}, n_layers={arch.get('n_layers')}, "
            f"d_emb={arch.get('d_emb')}`"
        )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    rl_section = _render_rl_table(summary.get("rl") or {})
    if rl_section.strip():
        lines.append(rl_section)
        lines.append("")
    dyn_section = _render_dynamics_table(summary.get("dynamics") or {})
    if dyn_section.strip():
        lines.append(dyn_section)
        lines.append("")
    contra_section = _render_contraction_table(summary.get("contraction") or {})
    if contra_section.strip():
        lines.append(contra_section)
        lines.append("")
    top_section = _render_top_actions_table(summary.get("top_actions") or {})
    if top_section.strip():
        lines.append(top_section)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_caveats_md(summary: dict[str, Any]) -> str:
    """Render an explicit honesty / caveats document.

    The text encodes the project's binding scientific constraints and the post-defense
    future-work plan. It is the single place where the "this is not a biological discovery"
    framing is asserted; downstream tooling can grep for the section headers.
    """
    prov = summary.get("provenance") or {}
    dyn = summary.get("dynamics") or {}
    contra = summary.get("contraction") or {}

    # Headline contraction stats (used in the contraction caveat)
    primary_32d = contra.get("primary_32d") or {}
    plain_64d = contra.get("ablation_64d_plain") or {}
    primary_32d_frac = primary_32d.get("fraction_improved")
    plain_64d_frac = plain_64d.get("fraction_improved")

    # Headline gate margin (32D primary val Pearson margin)
    primary_block = ((dyn.get("primary_32d") or {}).get("primary") or {})
    margin = primary_block.get("margin_vs_linear_ridge_pearson")
    threshold = primary_block.get("margin_vs_linear_ridge_pearson_threshold")

    lines: list[str] = []
    lines.append("# CellPath — caveats (MVP V1)")
    lines.append("")
    lines.append("These caveats are binding. Defense slides and downstream documents MUST keep them attached to every numerical claim derived from V1.")
    lines.append("")
    lines.append("## 1. Dynamics gate failed and was overridden")
    lines.append("")
    lines.append(
        f"The dynamics validation gate's `margin_vs_linear_ridge_pearson` is "
        f"**{_fmt(margin, ndigits=4)}** on the 32D primary val split; threshold is "
        f"**{_fmt(threshold, ndigits=3)}**. RL training proceeded with "
        "`rl.train.skip_gate=true`. Every RL run's `metadata.json` records "
        "`dynamics_gate_passed=false, dynamics_gate_overridden=true`."
    )
    lines.append("")
    lines.append("## 2. Surrogate dynamics is globally contractive")
    lines.append("")
    lines.append(
        "Under hard start states (`min_start_distance=8.0`), the trained dynamics model "
        f"reduces distance to `z_ref` for **{_fmt(primary_32d_frac, ndigits=4)}** of all "
        "sampled (start, gene) action pairs on the 32D primary branch."
    )
    if plain_64d_frac is not None:
        lines.append(
            f"The 64D baseline_plain MLP (no `state_linear`) shows the same artifact: "
            f"**{_fmt(plain_64d_frac, ndigits=4)}** of pairs improve. The contraction is not "
            "an artifact of the `state_linear` skip connection — it is a property of the "
            "pair/dynamics geometry."
        )
    lines.append("")
    lines.append(
        "Consequence: RL numbers reflect a real learned-control gain over the surrogate "
        "(+14.8 pp PPO det vs random, fewer steps, smaller final distance), but the surrogate "
        "itself is biased. We name this explicitly rather than hiding it."
    )
    lines.append("")
    lines.append("## 3. 64D was tested and rejected as primary")
    lines.append("")
    lines.append(
        "A fully isolated 64D VAE branch was trained under `artifacts_64/`. The 64D dynamics "
        "model failed the gate by a wider margin and worsened OOD generalization vs 32D. 64D is "
        "reported as a documented **negative ablation**. 32D `state_linear` remains the primary "
        "branch."
    )
    lines.append("")
    lines.append("## 4. Biological discovery / therapeutic claims are out of scope")
    lines.append("")
    lines.append(
        "The reward target is the unperturbed-leukemic K562 NT-centroid in scVI latent space "
        "(D7 / Concept 7 in `ARCHITECTURE.md`). No external healthy reference is used in V1. "
        "DepMap enrichment, where present, measures biological *plausibility* of selected "
        "genes — not therapeutic validity. CRISPRa ≠ CRISPRko; cross-modal claims are not made."
    )
    lines.append("")
    lines.append("## Future work (post-defense, ranked)")
    lines.append("")
    lines.append("1. **Investigate OT-pair drift.** Compare per-perturbation pair-Δz distribution against random pairing on the same cells; the OT coupling may bias targets inward. (~0.5 day, highest-information-per-hour.)")
    lines.append("2. **Anti-contractive / orthogonal-Δz dynamics regularizer.** Penalize the predicted Δz component aligned with `(z − z_ref)`. (~1–2 days.)")
    lines.append("3. **Composition / trajectory consistency loss** on the existing `combo_pairs.npz`. (~1–2 days.)")
    lines.append("4. **External healthy reference dataset** (Tabula Sapiens BM, hematopoietic atlas). Biggest single upgrade; the only path to a real-discovery framing. (~3–5 days.)")
    lines.append("5. **Replogle 2022 CRISPRi** to enable the knockout action space and validate Norman CRISPRa directionality against a separate modality. (~2–3 days.)")
    lines.append("6. **state_linear vs gene-only contraction decomposition.** Cheap, secondary; quantifies how much of the contractive μ comes from `state_linear`. (~1 hour.)")
    lines.append("")
    lines.append(
        f"Generated from `summary.json` (git `{prov.get('git_commit')}`)."
    )
    return "\n".join(lines).rstrip() + "\n"
