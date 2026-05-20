"""V3B Phase 3 aggregator — 4-seed CIs + Phase 3 acceptance check."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


LOG = logging.getLogger("aggregate_v3b_phase3")

SEEDS = (42, 0, 1, 7)
CELLS = (
    "k3_epsp25_bin8-10_splitood",
    "k4_epsp25_bin8-10_splitood",
    "k5_epsp25_bin8-10_splitood",
    "k4_epsp25_bin6-8_splitood",
    "k5_epsp25_bin6-8_splitood",
    "k8_epsp25_bin8-10_splitood",
)
POLICIES = (
    "ppo_B", "ppo_A",
    "random_uniform_valid", "always_noop",
    "greedy_dyn_1_B", "greedy_dyn_2_B", "greedy_dyn_3_B", "greedy_dyn_5_B", "greedy_dyn_8_B",
)


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
        for cell in CELLS:
            for pol in POLICIES:
                sp = seed_dir / cell / pol / "summary.json"
                if not sp.exists():
                    continue
                with open(sp) as f:
                    s = json.load(f)
                rows.append({
                    "seed": int(seed), "cell": cell, "policy": pol,
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


def compute_cis_and_deltas(df: pl.DataFrame) -> tuple[dict, dict]:
    cis: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        cis[cell] = {}
        for pol in POLICIES:
            sub = df.filter((pl.col("cell") == cell) & (pl.col("policy") == pol))
            entry: dict[str, Any] = {"n_seeds": int(sub.height)}
            for key in ("success_rate", "mean_steps", "mean_final_distance",
                        "frac_success_T_4_or_5", "frac_success_T_le_3", "frac_success_T_gt_5"):
                vals = [v for v in sub[key].to_list() if v is not None]
                m, lo, hi = _normal_ci(vals)
                entry[f"{key}_mean"] = m
                entry[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                entry[f"{key}_ci_lo"] = lo
                entry[f"{key}_ci_hi"] = hi
                entry[f"{key}_per_seed"] = list(vals)
            cis[cell][pol] = entry

    # Paired deltas: PPO_B − greedy_dyn_K_B for K ∈ {1, 2, 3, 5, 8}, and PPO_B − PPO_A,
    # PPO_B − random. Paired by seed.
    deltas: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        deltas[cell] = {}
        per_seed_sr: dict[int, dict[str, float | None]] = {}
        for seed in SEEDS:
            per_seed_sr[seed] = {}
            for pol in POLICIES:
                sub = df.filter(
                    (pl.col("cell") == cell)
                    & (pl.col("policy") == pol)
                    & (pl.col("seed") == seed)
                )
                if sub.is_empty():
                    per_seed_sr[seed][pol] = None
                else:
                    per_seed_sr[seed][pol] = float(sub["success_rate"][0])

        for ref_pol in ("ppo_A", "random_uniform_valid",
                        "greedy_dyn_1_B", "greedy_dyn_2_B", "greedy_dyn_3_B",
                        "greedy_dyn_5_B", "greedy_dyn_8_B"):
            paired: list[float] = []
            for seed in SEEDS:
                a = per_seed_sr[seed].get("ppo_B")
                b = per_seed_sr[seed].get(ref_pol)
                if a is not None and b is not None:
                    paired.append(a - b)
            m, lo, hi = _normal_ci(paired)
            deltas[cell][f"delta_PPO_B_minus_{ref_pol}_raw_paired"] = {
                "per_seed_deltas": paired,
                "mean": m, "ci_lo": lo, "ci_hi": hi,
                "std": float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0,
                "ci_excludes_zero": (lo > 0 or hi < 0),
            }
    return cis, deltas


def apply_acceptance(cis: dict, deltas: dict) -> dict[str, Any]:
    """Phase 3 acceptance rules per user spec.

    1. 4-seed CI on PPO_B − greedy_dyn_5_B raw success at any K≥4 cell strictly excludes
       zero in PPO_B's favor.
    2. PPO_B uses path lengths 4 or 5 in ≥30% of successful episodes at the winning cell.
    3. PPO_B does not simply exploit no-op or trivial starts (always_noop success ≤ 0.05).
    4. random remains clearly lower than PPO_B at the winning cell (Δ ≥ 0.10).
    5. The result is not only on a saturated K=3 cell.
    """
    K_GE_4_CELLS = [c for c in CELLS if int(c.split("_")[0][1:]) >= 4]

    # Rule 1: find any K≥4 cell with PPO_B − greedy_dyn_5_B CI > 0
    rule1 = {"threshold": 0.0, "per_cell": []}
    rule1_pass = False
    winning_cell = None
    best_delta = -float("inf")
    for cell in K_GE_4_CELLS:
        d = deltas[cell]["delta_PPO_B_minus_greedy_dyn_5_B_raw_paired"]
        rule1["per_cell"].append({
            "cell": cell, "mean": d["mean"], "ci_lo": d["ci_lo"],
            "ci_hi": d["ci_hi"], "ci_excludes_zero": d["ci_excludes_zero"],
        })
        if d["ci_excludes_zero"] and d["mean"] > 0:
            rule1_pass = True
            if d["mean"] > best_delta:
                best_delta = d["mean"]
                winning_cell = cell
    rule1["passed"] = rule1_pass
    rule1["winning_cell"] = winning_cell

    # Rule 2: at winning_cell (or strongest-delta cell), check frac_success_T_4_or_5 ≥ 0.30
    test_cell = winning_cell or max(
        K_GE_4_CELLS,
        key=lambda c: deltas[c]["delta_PPO_B_minus_greedy_dyn_5_B_raw_paired"]["mean"],
        default=None,
    )
    rule2 = {"threshold": 0.30, "test_cell": test_cell}
    if test_cell is not None:
        v = cis[test_cell]["ppo_B"]["frac_success_T_4_or_5_mean"]
        rule2["frac_success_T_4_or_5_mean"] = v
        rule2["per_seed"] = cis[test_cell]["ppo_B"]["frac_success_T_4_or_5_per_seed"]
        rule2["passed"] = bool(v is not None and v >= 0.30)
    else:
        rule2["passed"] = False

    # Rule 3: noop trivial check
    rule3 = {"threshold": 0.05, "per_cell": []}
    rule3_pass = True
    for cell in CELLS:
        v = cis[cell].get("always_noop", {}).get("success_rate_mean")
        rule3["per_cell"].append({"cell": cell, "noop_success": v})
        if v is not None and v > 0.05:
            rule3_pass = False
    rule3["passed"] = rule3_pass

    # Rule 4: at winning/test cell, PPO_B − random ≥ 0.10
    rule4 = {"threshold": 0.10, "test_cell": test_cell}
    if test_cell is not None:
        d = deltas[test_cell]["delta_PPO_B_minus_random_uniform_valid_raw_paired"]
        rule4["mean"] = d["mean"]
        rule4["ci_lo"] = d["ci_lo"]
        rule4["ci_hi"] = d["ci_hi"]
        rule4["passed"] = bool(d["mean"] >= 0.10)
    else:
        rule4["passed"] = False

    # Rule 5: winning cell is not K=3
    rule5 = {"K3_cells": [c for c in CELLS if c.startswith("k3_")]}
    rule5["winning_cell"] = winning_cell
    if winning_cell is None:
        rule5["passed"] = False  # nothing won
    else:
        rule5["passed"] = not winning_cell.startswith("k3_")

    overall = rule1["passed"] and rule2["passed"] and rule3["passed"] and rule4["passed"] and rule5["passed"]
    return {
        "rule_1_PPO_B_minus_greedy_dyn_5_B": rule1,
        "rule_2_path_length_4_or_5_usage": rule2,
        "rule_3_no_trivial_noop": rule3,
        "rule_4_PPO_B_minus_random": rule4,
        "rule_5_not_only_K3": rule5,
        "overall_passed": overall,
    }


def derive_verdict(acc: dict) -> str:
    if acc["overall_passed"]:
        return "PHASE3_PASS_PATH_LENGTH_FREEBAND_PRODUCES_PLANNING_ADVANTAGE"
    r1 = acc["rule_1_PPO_B_minus_greedy_dyn_5_B"]["passed"]
    r2 = acc["rule_2_path_length_4_or_5_usage"]["passed"]
    if not r1 and not r2:
        return "PHASE3_FAIL_NO_PATH_LENGTH_LEVERAGE_DETECTED_FIELD_SATURATED"
    if r1 and not r2:
        return "PHASE3_PARTIAL_PPO_B_BEATS_GREEDY_BUT_DOES_NOT_USE_LONG_PATHS"
    if r2 and not r1:
        return "PHASE3_PARTIAL_PPO_B_USES_LONG_PATHS_BUT_NO_GREEDY_ADVANTAGE"
    return "PHASE3_FAIL_REVIEW_DETAILS"


def write_md(cis, deltas, acceptance, df, out_path: Path) -> None:
    md: list[str] = []
    md.append("# V3B Phase 3 — Path-length free-band 4-seed escalation summary")
    md.append("")
    md.append(f"> Seeds: {list(SEEDS)}, n=300 episodes/cell, env.max_steps adapted per cell K.")
    md.append("> dynamics: artifacts_v2/dynamics_v1ot_ror_corr010 (V2 primary 32D).")
    md.append("> reward: path_length_freeband (free_steps=3, mild_until=5, mild_beta=0.02, heavy_beta=0.10, success_bonus=1.0).")
    md.append("> Greedy oracles are REWARD-AWARE under freeband.")
    md.append("")
    md.append(f"## Final verdict")
    md.append("")
    md.append(f"**`{derive_verdict(acceptance)}`**")
    md.append("")
    md.append("### Acceptance criteria")
    md.append("")
    md.append("| # | Rule | Result | Passed |")
    md.append("|---|---|---|:---:|")
    r1 = acceptance["rule_1_PPO_B_minus_greedy_dyn_5_B"]
    r1_max = max((c["mean"] for c in r1["per_cell"]), default=float("nan"))
    md.append(f"| 1 | PPO_B − greedy_dyn_5_B raw success CI excludes 0 at any K≥4 cell | "
              f"max Δ across K≥4 cells = {r1_max:+.4f}; winning_cell={r1['winning_cell']} | "
              f"{'✅' if r1['passed'] else '❌'} |")
    r2 = acceptance["rule_2_path_length_4_or_5_usage"]
    r2_val = r2.get("frac_success_T_4_or_5_mean")
    md.append(f"| 2 | PPO_B uses T∈{{4,5}} in ≥30% of successful episodes at winning/test cell | "
              f"frac = {r2_val:.4f} at {r2.get('test_cell')} | "
              f"{'✅' if r2['passed'] else '❌'} |")
    r3 = acceptance["rule_3_no_trivial_noop"]
    md.append(f"| 3 | always_noop success ≤ 0.05 at every cell | "
              f"max noop success = {max((c['noop_success'] or 0) for c in r3['per_cell']):.3f} | "
              f"{'✅' if r3['passed'] else '❌'} |")
    r4 = acceptance["rule_4_PPO_B_minus_random"]
    md.append(f"| 4 | PPO_B − random raw success ≥ 0.10 at winning/test cell | "
              f"mean = {r4.get('mean', float('nan')):+.4f} at {r4.get('test_cell')} | "
              f"{'✅' if r4['passed'] else '❌'} |")
    r5 = acceptance["rule_5_not_only_K3"]
    md.append(f"| 5 | Winning cell is not K=3 | "
              f"winning_cell = {r5['winning_cell']} | "
              f"{'✅' if r5['passed'] else '❌'} |")

    md.append("")
    md.append("## 4-seed mean ± std per (cell, policy)")
    md.append("")
    md.append("| cell | policy | success | mean_steps | frac_succ_T∈{4,5} | mean_final_dist |")
    md.append("|---|---|---|---|---|---|")
    for cell in CELLS:
        for pol in POLICIES:
            c = cis[cell][pol]
            row = (
                f"| {cell} | {pol} | "
                f"{c['success_rate_mean']:.3f} ± {c['success_rate_std']:.3f} | "
                f"{c['mean_steps_mean']:.2f} ± {c['mean_steps_std']:.2f} | "
                f"{c['frac_success_T_4_or_5_mean']:.3f} ± {c['frac_success_T_4_or_5_std']:.3f} | "
                f"{c['mean_final_distance_mean']:.3f} ± {c['mean_final_distance_std']:.3f} |"
            )
            md.append(row)
    md.append("")

    md.append("## Per-cell paired deltas (4-seed 95% CI)")
    md.append("")
    md.append("| Cell | Δ(PPO_B − PPO_A) | Δ(PPO_B − greedy_dyn_2_B) | Δ(PPO_B − greedy_dyn_5_B) | Δ(PPO_B − random) |")
    md.append("|---|---|---|---|---|")
    for cell in CELLS:
        d_a = deltas[cell]["delta_PPO_B_minus_ppo_A_raw_paired"]
        d_g2 = deltas[cell]["delta_PPO_B_minus_greedy_dyn_2_B_raw_paired"]
        d_g5 = deltas[cell]["delta_PPO_B_minus_greedy_dyn_5_B_raw_paired"]
        d_r = deltas[cell]["delta_PPO_B_minus_random_uniform_valid_raw_paired"]
        def _fmt(d):
            tag = "✅" if d["ci_excludes_zero"] and d["mean"] > 0 else ("❌" if d["ci_excludes_zero"] and d["mean"] < 0 else "—")
            return f"{d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] {tag}"
        md.append(f"| {cell} | {_fmt(d_a)} | {_fmt(d_g2)} | {_fmt(d_g5)} | {_fmt(d_r)} |")
    md.append("")

    md.append("## Per-seed PPO_B raw success (sanity)")
    md.append("")
    md.append("| Seed | " + " | ".join(CELLS) + " |")
    md.append("|---" + "|---" * len(CELLS) + "|")
    for seed in SEEDS:
        cells_str = []
        for cell in CELLS:
            sub = df.filter((pl.col("seed") == seed) & (pl.col("cell") == cell) & (pl.col("policy") == "ppo_B"))
            cells_str.append(f"{float(sub['success_rate'][0]):.3f}" if not sub.is_empty() else "—")
        md.append(f"| {seed} | " + " | ".join(cells_str) + " |")

    out_path.write_text("\n".join(md) + "\n")
    LOG.info("Wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", default="artifacts_v3/eval_v3b_phase3")
    parser.add_argument("--out_dir", default="artifacts_v3/eval_v3b_phase3")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    eval_root = (repo_root / args.eval_root) if not Path(args.eval_root).is_absolute() else Path(args.eval_root)
    out_dir = (repo_root / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_per_seed(eval_root)
    LOG.info("Loaded %d rows (expected %d)", df.height, len(SEEDS) * len(CELLS) * len(POLICIES))

    df.write_csv(str(out_dir / "phase3_results.csv"))
    cis, deltas = compute_cis_and_deltas(df)
    acceptance = apply_acceptance(cis, deltas)

    out_json = {
        "seeds": list(SEEDS),
        "cells": list(CELLS),
        "per_cell_per_policy_4seed_CIs": cis,
        "per_cell_paired_deltas_4seed_CIs": deltas,
        "acceptance_criteria": acceptance,
        "final_verdict": derive_verdict(acceptance),
        "greedy_is_reward_aware": True,
        "greedy_reward_objective": "path_length_freeband",
    }
    (out_dir / "phase3_results.json").write_text(json.dumps(out_json, indent=2, default=str))
    write_md(cis, deltas, acceptance, df, out_dir / "phase3_summary.md")

    print(f"\n=== Phase 3 verdict: {derive_verdict(acceptance)} ===\n")
    for k, r in acceptance.items():
        if k == "overall_passed":
            continue
        print(f"  {k}: passed={r['passed']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
