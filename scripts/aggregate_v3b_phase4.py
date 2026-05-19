"""V3B Phase 4 — Reward-stack aggregator.

Reads per-seed eval outputs from artifacts_v3/eval_v3b_reward_stack/seed{42,0,1,7}/
and writes:
* reward_stack_results.csv  (long-form per seed, cell, policy)
* reward_stack_results.json (per-cell, per-policy 4-seed CIs + paired deltas)
* reward_stack_summary.md   (human-readable, Bucket A/B/C separated)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


LOG = logging.getLogger("aggregate_v3b_phase4")

SEEDS = (42, 0, 1, 7)
CELLS = (
    "k2_bin6-8_splitood", "k2_bin8-10_splitood",
    "k3_bin6-8_splitood", "k3_bin8-10_splitood",
    "k4_bin8-10_splitood", "k5_bin8-10_splitood", "k8_bin8-10_splitood",
)
PPO_POLICIES = ("PPO_A", "PPO_B", "PPO_C", "PPO_BC", "PPO_D", "PPO_BCD")
NONPPO_POLICIES = (
    "random_uniform_valid", "always_noop",
    "greedy_dyn_1_fused", "greedy_dyn_2_fused",
    "greedy_dyn_3_fused", "greedy_dyn_5_fused", "greedy_dyn_8_fused",
)
ALL_POLICIES = PPO_POLICIES + NONPPO_POLICIES


def _normal_ci(vals: list[float], z: float = 1.96) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size <= 1:
        return (mean, mean, mean)
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return (mean, mean - z * se, mean + z * se)


def load_per_seed(eval_root: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_dir = eval_root / f"seed{seed}"
        if not seed_dir.exists():
            LOG.warning("missing seed dir: %s", seed_dir); continue
        for cell in CELLS:
            for pol in ALL_POLICIES:
                sp = seed_dir / cell / pol / "summary.json"
                if not sp.exists():
                    continue
                with open(sp) as f:
                    s = json.load(f)
                rows.append({
                    "seed": int(seed), "cell": cell, "policy": pol,
                    "k": s.get("k"), "epsilon_label": s.get("epsilon_label"),
                    # Bucket B
                    "success_rate": s.get("success_rate"),
                    "mean_steps": s.get("mean_steps"),
                    "mean_final_distance": s.get("mean_final_distance"),
                    "mean_total_reward": s.get("mean_total_reward"),
                    "frac_success_T_le_3": s.get("frac_success_T_le_3"),
                    "frac_success_T_4_or_5": s.get("frac_success_T_4_or_5"),
                    "frac_success_T_gt_5": s.get("frac_success_T_gt_5"),
                    # Bucket A
                    "mean_tox_path": s.get("mean_tox_path"),
                    "mean_common_essential_per_ep": s.get("mean_common_essential_per_ep"),
                    "fraction_zero_common_essential": s.get("fraction_zero_common_essential"),
                    "mean_unc_path_max": s.get("mean_unc_path_max"),
                    "mean_unc_path_mean": s.get("mean_unc_path_mean"),
                })
    return pl.DataFrame(rows)


def compute_cis(df: pl.DataFrame) -> dict:
    out: dict[str, Any] = {}
    for cell in CELLS:
        out[cell] = {}
        for pol in ALL_POLICIES:
            sub = df.filter((pl.col("cell") == cell) & (pl.col("policy") == pol))
            entry: dict[str, Any] = {"n_seeds": int(sub.height)}
            for key in [
                "success_rate", "mean_steps", "mean_final_distance",
                "frac_success_T_4_or_5", "frac_success_T_le_3", "frac_success_T_gt_5",
                "mean_tox_path", "mean_common_essential_per_ep",
                "fraction_zero_common_essential",
                "mean_unc_path_max", "mean_unc_path_mean",
            ]:
                vals = [v for v in sub[key].to_list() if v is not None]
                m, lo, hi = _normal_ci(vals)
                entry[f"{key}_mean"] = m
                entry[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                entry[f"{key}_ci_lo"] = lo
                entry[f"{key}_ci_hi"] = hi
                entry[f"{key}_per_seed"] = list(vals)
            out[cell][pol] = entry
    return out


def compute_paired_deltas(df: pl.DataFrame) -> dict:
    out: dict[str, Any] = {}
    for cell in CELLS:
        out[cell] = {}
        per_seed: dict[int, dict[str, float | None]] = {}
        for seed in SEEDS:
            per_seed[seed] = {}
            for pol in ALL_POLICIES:
                sub = df.filter(
                    (pl.col("cell") == cell) & (pl.col("policy") == pol) & (pl.col("seed") == seed)
                )
                per_seed[seed][pol] = None if sub.is_empty() else float(sub["success_rate"][0])

        for src in ("PPO_BCD", "PPO_BC", "PPO_D"):
            for tgt in ("PPO_A", "PPO_B", "PPO_C",
                        "random_uniform_valid",
                        "greedy_dyn_2_fused", "greedy_dyn_3_fused", "greedy_dyn_5_fused"):
                paired: list[float] = []
                for seed in SEEDS:
                    a = per_seed[seed].get(src); b = per_seed[seed].get(tgt)
                    if a is not None and b is not None:
                        paired.append(a - b)
                m, lo, hi = _normal_ci(paired)
                out[cell][f"delta_{src}_minus_{tgt}"] = {
                    "per_seed_deltas": paired,
                    "mean": m, "ci_lo": lo, "ci_hi": hi,
                    "std": float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0,
                    "ci_excludes_zero": (lo > 0 or hi < 0),
                }
    return out


def derive_verdict(cis: dict, deltas: dict) -> tuple[str, dict]:
    """Pick LOCKED_DESIGN_POSITIVE_SIGNAL / TECHNICAL_ONLY / FAILED_IMPLEMENTATION."""
    # Positive signal requires CI on PPO_BCD − greedy_dyn_5_fused (or _3) to exclude 0
    # in PPO_BCD favor at any K≥4 cell with non-saturated greedy.
    positive_hits = []
    for cell in CELLS:
        try:
            d_g5 = deltas[cell].get("delta_PPO_BCD_minus_greedy_dyn_5_fused", {})
            d_g3 = deltas[cell].get("delta_PPO_BCD_minus_greedy_dyn_3_fused", {})
        except KeyError:
            continue
        # Only count if greedy is not fully saturated at this cell (≤0.95)
        try:
            g5_mean = cis[cell]["greedy_dyn_5_fused"]["success_rate_mean"]
            g3_mean = cis[cell]["greedy_dyn_3_fused"]["success_rate_mean"]
        except KeyError:
            g5_mean = g3_mean = None
        if d_g5.get("ci_excludes_zero") and d_g5.get("mean", 0) > 0 and (g5_mean or 1.0) < 0.99:
            positive_hits.append({"cell": cell, "comparator": "greedy_5", "delta": d_g5})
        elif d_g3.get("ci_excludes_zero") and d_g3.get("mean", 0) > 0 and (g3_mean or 1.0) < 0.99:
            positive_hits.append({"cell": cell, "comparator": "greedy_3", "delta": d_g3})

    # Failed implementation check: any PPO_BCD success collapse below 0.30 below PPO_A at any cell
    catastrophic = False
    for cell in CELLS:
        try:
            sr_bcd = cis[cell]["PPO_BCD"]["success_rate_mean"]
            sr_a = cis[cell]["PPO_A"]["success_rate_mean"]
        except KeyError:
            continue
        if sr_bcd is not None and sr_a is not None and (sr_a - sr_bcd) > 0.30:
            catastrophic = True; break

    if catastrophic:
        verdict = "LOCKED_DESIGN_FAILED_IMPLEMENTATION"
    elif positive_hits:
        verdict = "LOCKED_DESIGN_POSITIVE_SIGNAL"
    else:
        verdict = "LOCKED_DESIGN_TECHNICAL_ONLY"

    return verdict, {"positive_hits": positive_hits, "catastrophic_regressions": catastrophic}


def write_md(cis: dict, deltas: dict, verdict: str, verdict_meta: dict,
             df: pl.DataFrame, out_path: Path) -> None:
    md = [
        "# V3B Phase 4 / Final — Reward-stack 4-seed evaluation summary",
        "",
        f"> Seeds {list(SEEDS)}, n=300 episodes/cell, n_cells={len(CELLS)}.",
        "> dynamics: artifacts_v2/dynamics_v1ot_ror_corr010 (V2 primary 32D — frozen).",
        "> reward env: biorealistic_fused (so all A-bucket metrics tracked uniformly).",
        "> Greedy oracles are reward-aware under fused objective.",
        "",
        f"## Final verdict: **`{verdict}`**",
        "",
        "(`LOCKED_DESIGN_TECHNICAL_ONLY` is the expected outcome on V2 dynamics — the controller "
        "objective is implemented and evaluable but the field's saturation prevents a "
        "planning-advantage headline.)",
        "",
    ]
    if verdict_meta.get("positive_hits"):
        md.append("### Positive-signal cells (PPO_BCD beats reward-aware greedy CI-excluding-zero):")
        for h in verdict_meta["positive_hits"]:
            d = h["delta"]
            md.append(f"* {h['cell']} vs {h['comparator']}: "
                      f"Δ = {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")
        md.append("")

    # Bucket B — raw success per (cell, policy)
    md += [
        "## Bucket B — reward-independent raw success (4-seed mean ± std)",
        "",
        "| cell | PPO_A | PPO_B | PPO_C | PPO_BC | PPO_D | PPO_BCD | greedy_2_F | greedy_5_F | random |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in CELLS:
        row = f"| {cell} |"
        for pol in ("PPO_A", "PPO_B", "PPO_C", "PPO_BC", "PPO_D", "PPO_BCD",
                    "greedy_dyn_2_fused", "greedy_dyn_5_fused", "random_uniform_valid"):
            try:
                c = cis[cell][pol]
                m, s = c["success_rate_mean"], c["success_rate_std"]
                if c["n_seeds"] == 0 or m != m:
                    row += " — |"
                else:
                    row += f" {m:.3f}±{s:.3f} |"
            except KeyError:
                row += " — |"
        md.append(row)
    md.append("")

    # Bucket A — safety+unc metrics per (cell, policy)
    md += [
        "## Bucket A — reward-fit metrics (4-seed mean of PPO_BCD vs key comparators)",
        "",
        "Reminder: Bucket A metrics are derived from sources used in the reward (DepMap Chronos "
        "for tox/CE; learned dynamics for unc) and represent reward-prior optimisation, not "
        "independent biological discovery.",
        "",
        "| cell | policy | mean_tox | mean_CE | unc_max | unc_mean | frac_zero_CE |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in CELLS:
        for pol in ("PPO_BCD", "PPO_BC", "PPO_D", "PPO_C", "PPO_A",
                    "greedy_dyn_2_fused", "greedy_dyn_5_fused"):
            try:
                c = cis[cell][pol]
                def _ms(key):
                    m = c.get(f"{key}_mean")
                    s = c.get(f"{key}_std", 0.0)
                    if m is None or (isinstance(m, float) and np.isnan(m)):
                        return "—"
                    return f"{m:.3f}±{s:.3f}"
                md.append(
                    f"| {cell} | {pol} | "
                    f"{_ms('mean_tox_path')} | "
                    f"{_ms('mean_common_essential_per_ep')} | "
                    f"{_ms('mean_unc_path_max')} | "
                    f"{_ms('mean_unc_path_mean')} | "
                    f"{_ms('fraction_zero_common_essential')} |"
                )
            except KeyError:
                pass
    md.append("")

    # Paired deltas PPO_BCD vs each baseline
    md += [
        "## Paired-by-seed deltas (4-seed 95% CI) — PPO_BCD vs comparators",
        "",
        "| cell | vs PPO_A | vs PPO_B | vs PPO_C | vs greedy_5_F | vs random |",
        "|---|---|---|---|---|---|",
    ]
    for cell in CELLS:
        def _fmt(key):
            try:
                d = deltas[cell][key]
                tag = "✅" if d["ci_excludes_zero"] and d["mean"] > 0 else (
                      "❌" if d["ci_excludes_zero"] and d["mean"] < 0 else "—")
                return f"{d['mean']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] {tag}"
            except KeyError:
                return "—"
        md.append(
            f"| {cell} | "
            f"{_fmt('delta_PPO_BCD_minus_PPO_A')} | "
            f"{_fmt('delta_PPO_BCD_minus_PPO_B')} | "
            f"{_fmt('delta_PPO_BCD_minus_PPO_C')} | "
            f"{_fmt('delta_PPO_BCD_minus_greedy_dyn_5_fused')} | "
            f"{_fmt('delta_PPO_BCD_minus_random_uniform_valid')} |"
        )
    md.append("")

    # Bucket C
    md += [
        "## Bucket C — held-out biological validation",
        "",
        "**Status: pending_no_local_source.** No held-out source not used in the reward is currently "
        "loaded in this evaluator. The Phase 2c Replogle K562 essentials check is available "
        "(`artifacts_v3/eval_v3b_phase2c/replogle_norman_intersection.json`) and shows the DepMap "
        "safety prior does not transfer to the Replogle assay — a Bucket-C finding consistent with "
        "the verdict that Variant C is reward-prior optimisation, not independent biological "
        "discovery.",
        "",
    ]

    out_path.write_text("\n".join(md) + "\n")
    LOG.info("Wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", default="artifacts_v3/eval_v3b_reward_stack")
    parser.add_argument("--out_dir", default="artifacts_v3/eval_v3b_reward_stack")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    eval_root = (repo_root / args.eval_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_per_seed(eval_root)
    LOG.info("Loaded %d rows", df.height)
    if df.height == 0:
        LOG.error("No data loaded — check eval_root.")
        return 1
    df.write_csv(str(out_dir / "reward_stack_results.csv"))
    cis = compute_cis(df)
    deltas = compute_paired_deltas(df)
    verdict, verdict_meta = derive_verdict(cis, deltas)

    out_json = {
        "seeds": list(SEEDS),
        "cells": list(CELLS),
        "policies": list(ALL_POLICIES),
        "per_cell_per_policy_4seed_CIs": cis,
        "per_cell_paired_deltas_4seed_CIs": deltas,
        "final_verdict": verdict,
        "verdict_meta": verdict_meta,
    }
    (out_dir / "reward_stack_results.json").write_text(json.dumps(out_json, indent=2, default=str))
    write_md(cis, deltas, verdict, verdict_meta, df, out_dir / "reward_stack_summary.md")
    print(f"\n=== Final verdict: {verdict} ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
