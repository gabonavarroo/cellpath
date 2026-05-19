"""V3B Phase 3b aggregator — stricter-epsilon diagnostic across {p15, p10[, p5]}."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


LOG = logging.getLogger("aggregate_v3b_phase3b")

SEEDS = (42, 0, 1, 7)
CELLS = (
    "k2_bin6-8_splitood", "k2_bin8-10_splitood",
    "k3_bin6-8_splitood", "k3_bin8-10_splitood",
    "k4_bin6-8_splitood", "k4_bin8-10_splitood",
    "k5_bin6-8_splitood", "k5_bin8-10_splitood",
    "k8_bin8-10_splitood",
)
POLICIES = (
    "ppo_B", "ppo_A", "ppo_C",
    "random_uniform_valid", "always_noop",
    "greedy_dyn_1_B", "greedy_dyn_2_B", "greedy_dyn_3_B", "greedy_dyn_5_B", "greedy_dyn_8_B",
)
DEFAULT_EPSILONS = ("p15", "p10")


def _normal_ci(vals: list[float], z: float = 1.96) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size <= 1:
        return (mean, mean, mean)
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return (mean, mean - z * se, mean + z * se)


def load_per_seed(eval_root: Path, epsilons: tuple[str, ...]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for eps in epsilons:
        for seed in SEEDS:
            seed_dir = eval_root / f"eps_{eps}" / f"seed{seed}"
            if not seed_dir.exists():
                LOG.warning("missing seed dir: %s", seed_dir)
                continue
            for cell in CELLS:
                for pol in POLICIES:
                    sp = seed_dir / cell / pol / "summary.json"
                    if not sp.exists():
                        continue
                    with open(sp) as f:
                        s = json.load(f)
                    rows.append({
                        "epsilon": eps,
                        "epsilon_value": s.get("epsilon"),
                        "seed": int(seed),
                        "cell": cell,
                        "k": s.get("k"),
                        "policy": pol,
                        "success_rate": s.get("success_rate"),
                        "mean_steps": s.get("mean_steps"),
                        "mean_final_distance": s.get("mean_final_distance"),
                        "mean_total_reward": s.get("mean_total_reward"),
                        "frac_success_T_le_3": s.get("frac_success_in_freeband_T_le_3"),
                        "frac_success_T_4_or_5": s.get("frac_success_in_mild_T_4_or_5"),
                        "frac_success_T_gt_5": s.get("frac_success_in_heavy_T_gt_5"),
                        "n_successful_episodes": s.get("n_successful_episodes"),
                    })
    return pl.DataFrame(rows)


def compute_cis(df: pl.DataFrame, epsilons: tuple[str, ...]) -> dict:
    out: dict[str, Any] = {}
    for eps in epsilons:
        out[eps] = {}
        for cell in CELLS:
            out[eps][cell] = {}
            for pol in POLICIES:
                sub = df.filter(
                    (pl.col("epsilon") == eps) & (pl.col("cell") == cell) & (pl.col("policy") == pol)
                )
                entry: dict[str, Any] = {"n_seeds": int(sub.height)}
                for key in ("success_rate", "mean_steps", "mean_final_distance",
                            "frac_success_T_le_3", "frac_success_T_4_or_5", "frac_success_T_gt_5"):
                    vals = [v for v in sub[key].to_list() if v is not None]
                    m, lo, hi = _normal_ci(vals)
                    entry[f"{key}_mean"] = m
                    entry[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    entry[f"{key}_ci_lo"] = lo
                    entry[f"{key}_ci_hi"] = hi
                    entry[f"{key}_per_seed"] = list(vals)
                out[eps][cell][pol] = entry
    return out


def compute_paired_deltas(df: pl.DataFrame, epsilons: tuple[str, ...]) -> dict:
    out: dict[str, Any] = {}
    for eps in epsilons:
        out[eps] = {}
        for cell in CELLS:
            out[eps][cell] = {}
            per_seed: dict[int, dict[str, float | None]] = {}
            for seed in SEEDS:
                per_seed[seed] = {}
                for pol in POLICIES:
                    sub = df.filter(
                        (pl.col("epsilon") == eps) & (pl.col("cell") == cell)
                        & (pl.col("policy") == pol) & (pl.col("seed") == seed)
                    )
                    per_seed[seed][pol] = None if sub.is_empty() else float(sub["success_rate"][0])

            for ref_pol in ("ppo_A", "ppo_C", "random_uniform_valid",
                             "greedy_dyn_1_B", "greedy_dyn_2_B", "greedy_dyn_3_B",
                             "greedy_dyn_5_B", "greedy_dyn_8_B"):
                paired: list[float] = []
                for seed in SEEDS:
                    a = per_seed[seed].get("ppo_B")
                    b = per_seed[seed].get(ref_pol)
                    if a is not None and b is not None:
                        paired.append(a - b)
                m, lo, hi = _normal_ci(paired)
                out[eps][cell][f"delta_PPO_B_minus_{ref_pol}"] = {
                    "per_seed_deltas": paired,
                    "mean": m, "ci_lo": lo, "ci_hi": hi,
                    "std": float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0,
                    "ci_excludes_zero": (lo > 0 or hi < 0),
                }
    return out


def apply_decision_rules(cis: dict, deltas: dict, epsilons: tuple[str, ...]) -> dict:
    """Decision rules A/B/C from Phase 3b user spec."""
    diagnostic: dict[str, Any] = {}

    for eps in epsilons:
        K_GE_4 = [c for c in CELLS if int(c.split("_")[0][1:]) >= 4]
        K_ALL = list(CELLS)
        # Headroom analysis: cells where 0.30 ≤ greedy_dyn_5_B ≤ 0.95
        headroom_cells = []
        for cell in K_GE_4:
            try:
                v = cis[eps][cell]["greedy_dyn_5_B"]["success_rate_mean"]
            except KeyError:
                v = None
            if v is not None and 0.30 <= v <= 0.95:
                headroom_cells.append({"cell": cell, "greedy_dyn_5_B_success": v})
        # Saturation check
        all_saturated = True
        for cell in K_ALL:
            try:
                v = cis[eps][cell]["greedy_dyn_5_B"]["success_rate_mean"]
            except KeyError:
                v = None
            if v is None or v < 0.95:
                all_saturated = False; break
        # Collapse check
        all_collapsed = True
        for cell in K_ALL:
            try:
                vg = cis[eps][cell]["greedy_dyn_2_B"]["success_rate_mean"]
                vp = cis[eps][cell]["ppo_B"]["success_rate_mean"]
            except KeyError:
                vg = vp = None
            if vg is None or vp is None or vg > 0.10 or vp > 0.10:
                all_collapsed = False; break

        # PPO_B long-path usage at any headroom cell
        ppo_b_long_path_usage = []
        for hc in headroom_cells:
            try:
                v = cis[eps][hc["cell"]]["ppo_B"]["frac_success_T_4_or_5_mean"]
            except KeyError:
                v = None
            ppo_b_long_path_usage.append({"cell": hc["cell"], "frac_T_4_5_PPO_B": v})

        diagnostic[eps] = {
            "headroom_cells_K_ge_4": headroom_cells,
            "n_headroom_cells": len(headroom_cells),
            "all_saturated_greedy_5": all_saturated,
            "all_collapsed_greedy_2_and_ppo_B": all_collapsed,
            "ppo_B_long_path_usage_at_headroom_cells": ppo_b_long_path_usage,
        }

    # Pick best epsilon for B+C retraining recommendation
    best_eps = None
    best_n_headroom = 0
    for eps in epsilons:
        n = diagnostic[eps]["n_headroom_cells"]
        if n > best_n_headroom:
            best_n_headroom = n
            best_eps = eps
    diagnostic["recommended_epsilon_for_BC"] = best_eps

    # Final verdict per the user's spec rules A/B/C
    p10_data = diagnostic.get("p10", {})
    rule_B = bool(p10_data.get("all_saturated_greedy_5"))
    rule_C = bool(p10_data.get("all_collapsed_greedy_2_and_ppo_B"))
    rule_A_any = any(diagnostic[e]["n_headroom_cells"] > 0 for e in epsilons)

    if rule_B:
        verdict = "PHASE3B_FIELD_REMAINS_SATURATED_AT_P10_RECOMMEND_REPRESENTATION_REFORMULATION"
    elif rule_C:
        verdict = "PHASE3B_P10_TOO_STRICT_COLLAPSE_TEST_P15_INTERMEDIATE"
    elif rule_A_any:
        verdict = "PHASE3B_STRICTER_EPSILON_CREATES_HEADROOM_BC_RETRAIN_JUSTIFIED"
    else:
        verdict = "PHASE3B_NO_CLEAR_HEADROOM_REVIEW_DETAILS"

    diagnostic["final_verdict"] = verdict
    return diagnostic


def write_outputs(cis: dict, deltas: dict, diagnostic: dict, df: pl.DataFrame,
                  epsilons: tuple[str, ...], out_dir: Path) -> None:
    # CSV: long-form per-(eps, seed, cell, policy)
    df.write_csv(str(out_dir / "epsilon_sweep_results.csv"))
    LOG.info("Wrote epsilon_sweep_results.csv (%d rows)", df.height)

    # JSON: cis + deltas + diagnostic
    out_json = {
        "epsilons_tested": list(epsilons),
        "epsilon_values": {
            "p25": 3.1663, "p15": 2.9898, "p10": 2.8846, "p5": 2.7362,
        },
        "cells": list(CELLS),
        "seeds": list(SEEDS),
        "per_eps_per_cell_per_policy_4seed_CIs": cis,
        "per_eps_per_cell_paired_deltas_4seed_CIs": deltas,
        "diagnostic": diagnostic,
        "final_verdict": diagnostic["final_verdict"],
        "recommended_epsilon_for_BC": diagnostic["recommended_epsilon_for_BC"],
    }
    (out_dir / "epsilon_sweep_results.json").write_text(json.dumps(out_json, indent=2, default=str))
    LOG.info("Wrote epsilon_sweep_results.json")

    # Markdown summary
    md: list[str] = []
    md.append("# V3B Phase 3b — Stricter-epsilon diagnostic (p15, p10[, p5])")
    md.append("")
    md.append(f"> Seeds: {list(SEEDS)}, n=300 episodes/cell.")
    md.append(f"> dynamics: V2 primary 32D RoR_corr010.")
    md.append("> reward: path_length_freeband (free_steps=3, mild_until=5, mild_beta=0.02, heavy_beta=0.10, success_bonus=1.0).")
    md.append("> Greedy oracles are REWARD-AWARE under freeband.")
    md.append("")
    md.append(f"## Final verdict")
    md.append("")
    md.append(f"**`{diagnostic['final_verdict']}`**")
    md.append("")
    md.append(f"Recommended ε for B+C retraining: **{diagnostic['recommended_epsilon_for_BC'] or 'NONE'}**")
    md.append("")

    md.append("## Epsilon values")
    md.append("")
    md.append("| Label | Value (latent L2) | Notes |")
    md.append("|---|---:|---|")
    md.append("| p25 | 3.1663 | V2 reference (Phase 3) |")
    for eps in epsilons:
        v = {"p15": 2.9898, "p10": 2.8846, "p5": 2.7362}.get(eps, "?")
        md.append(f"| **{eps}** | **{v}** | Phase 3b test |")
    md.append("")

    # Headroom diagnostic
    md.append("## Headroom diagnostic per epsilon")
    md.append("")
    md.append("| ε | n_headroom_cells | all_saturated_greedy_5 | all_collapsed | PPO_B long-path usage at headroom |")
    md.append("|---|---:|:---:|:---:|---|")
    for eps in epsilons:
        d = diagnostic[eps]
        long_path = ", ".join(
            f"{x['cell']}:{(x['frac_T_4_5_PPO_B'] or 0):.3f}"
            for x in d["ppo_B_long_path_usage_at_headroom_cells"]
        ) or "—"
        md.append(
            f"| {eps} | {d['n_headroom_cells']} | "
            f"{'✅' if d['all_saturated_greedy_5'] else '❌'} | "
            f"{'✅' if d['all_collapsed_greedy_2_and_ppo_B'] else '❌'} | {long_path} |"
        )
    md.append("")

    # Per-epsilon × per-cell summary (4-seed mean only for clarity)
    for eps in epsilons:
        md.append(f"## ε = {eps} — per-cell × per-policy raw success (4-seed mean ± std)")
        md.append("")
        md.append("| cell | ppo_B | ppo_A | ppo_C | greedy_1 | greedy_2 | greedy_3 | greedy_5 | greedy_8 | random | noop |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for cell in CELLS:
            row = f"| {cell} |"
            for pol in ("ppo_B", "ppo_A", "ppo_C", "greedy_dyn_1_B", "greedy_dyn_2_B",
                        "greedy_dyn_3_B", "greedy_dyn_5_B", "greedy_dyn_8_B",
                        "random_uniform_valid", "always_noop"):
                try:
                    c = cis[eps][cell][pol]
                    m = c["success_rate_mean"]
                    s = c["success_rate_std"]
                    if c["n_seeds"] == 0 or m != m:  # NaN
                        row += " — |"
                    else:
                        row += f" {m:.3f}±{s:.3f} |"
                except KeyError:
                    row += " — |"
            md.append(row)
        md.append("")

        md.append(f"## ε = {eps} — PPO_B long-path usage")
        md.append("")
        md.append("| cell | frac_succ_T≤3 | frac_succ_T∈{4,5} | frac_succ_T>5 | mean_steps |")
        md.append("|---|---|---|---|---|")
        for cell in CELLS:
            try:
                c = cis[eps][cell]["ppo_B"]
                md.append(
                    f"| {cell} | "
                    f"{c['frac_success_T_le_3_mean']:.3f}±{c['frac_success_T_le_3_std']:.3f} | "
                    f"{c['frac_success_T_4_or_5_mean']:.3f}±{c['frac_success_T_4_or_5_std']:.3f} | "
                    f"{c['frac_success_T_gt_5_mean']:.3f}±{c['frac_success_T_gt_5_std']:.3f} | "
                    f"{c['mean_steps_mean']:.2f}±{c['mean_steps_std']:.2f} |"
                )
            except KeyError:
                md.append(f"| {cell} | — | — | — | — |")
        md.append("")

        md.append(f"## ε = {eps} — Paired-by-seed PPO_B deltas")
        md.append("")
        md.append("| cell | Δ(PPO_B − PPO_A) | Δ(PPO_B − PPO_C) | Δ(PPO_B − greedy_5_B) | Δ(PPO_B − random) |")
        md.append("|---|---|---|---|---|")
        for cell in CELLS:
            def _fmt(key):
                try:
                    d = deltas[eps][cell][key]
                    tag = "✅" if d["ci_excludes_zero"] and d["mean"] > 0 else (
                          "❌" if d["ci_excludes_zero"] and d["mean"] < 0 else "—")
                    return f"{d['mean']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] {tag}"
                except KeyError:
                    return "—"
            md.append(
                f"| {cell} | "
                f"{_fmt('delta_PPO_B_minus_ppo_A')} | "
                f"{_fmt('delta_PPO_B_minus_ppo_C')} | "
                f"{_fmt('delta_PPO_B_minus_greedy_dyn_5_B')} | "
                f"{_fmt('delta_PPO_B_minus_random_uniform_valid')} |"
            )
        md.append("")

    (out_dir / "epsilon_sweep_summary.md").write_text("\n".join(md) + "\n")
    LOG.info("Wrote epsilon_sweep_summary.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", default="artifacts_v3/eval_v3b_phase3b")
    parser.add_argument("--out_dir", default="artifacts_v3/eval_v3b_phase3b")
    parser.add_argument("--epsilons", nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    eval_root = (repo_root / args.eval_root) if not Path(args.eval_root).is_absolute() else Path(args.eval_root)
    out_dir = (repo_root / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epsilons = tuple(args.epsilons)

    df = load_per_seed(eval_root, epsilons)
    LOG.info("Loaded %d rows", df.height)
    cis = compute_cis(df, epsilons)
    deltas = compute_paired_deltas(df, epsilons)
    diagnostic = apply_decision_rules(cis, deltas, epsilons)
    write_outputs(cis, deltas, diagnostic, df, epsilons, out_dir)
    print(f"\n=== Phase 3b verdict: {diagnostic['final_verdict']} ===\n")
    for eps in epsilons:
        d = diagnostic[eps]
        print(f"  ε={eps}: headroom={d['n_headroom_cells']}, saturated_greedy_5={d['all_saturated_greedy_5']}, "
              f"all_collapsed={d['all_collapsed_greedy_2_and_ppo_B']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
