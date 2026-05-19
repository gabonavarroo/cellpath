"""V3B Phase 2c aggregator — 4-seed CI computation + acceptance check.

Reads per-seed eval outputs from artifacts_v3/eval_v3b_phase2c/seed{42,0,1,7}/
and writes:
* seed_escalation_results.csv  (raw per-seed × per-cell × per-policy table)
* seed_escalation_results.json (acceptance check + 4-seed CIs)
* seed_escalation_summary.md   (human-readable)
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


LOG = logging.getLogger("aggregate_v3b_phase2c")

SEEDS = (42, 0, 1, 7)
CELLS = (
    "k2_epsp25_bin6-8_splitood",
    "k2_epsp25_bin8-10_splitood",
    "k3_epsp25_bin6-8_splitood",
    "k3_epsp25_bin8-10_splitood",
)
PRIMARY_CELL = "k3_epsp25_bin8-10_splitood"
LEAKAGE_SAFE_HEADLINE_CELL = "k2_epsp25_bin8-10_splitood"
K2_BIN68_CELL = "k2_epsp25_bin6-8_splitood"
POLICIES_ALL = (
    "ppo_C", "ppo_A", "ppo_C_permuted",
    "greedy_dyn_2_C", "greedy_dyn_2_A",
    "random_uniform_valid", "always_noop",
)
N_SEEDS = len(SEEDS)


def _normal_ci(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """4-seed normal CI on the mean. Returns (mean, lo, hi)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size <= 1:
        return (mean, mean, mean)
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return (mean, mean - z * se, mean + z * se)


def load_per_seed(eval_root: Path) -> pl.DataFrame:
    """Load per-seed × per-cell × per-policy summary.json files into a long DF."""
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_dir = eval_root / f"seed{seed}"
        for cell in CELLS:
            for policy in POLICIES_ALL:
                sp = seed_dir / cell / policy / "summary.json"
                if not sp.exists():
                    LOG.warning("missing: %s", sp)
                    continue
                with open(sp) as f:
                    s = json.load(f)
                rows.append({
                    "seed": int(seed),
                    "cell": cell,
                    "policy": policy,
                    # Bucket B (reward-independent)
                    "raw_success_rate_B": s.get("success_rate"),
                    "mean_steps_B": s.get("mean_steps"),
                    "mean_final_distance_B": s.get("mean_final_distance"),
                    # Bucket A (reward-fit, Chronos-derived)
                    "safety_adjusted_SR_A": s.get("safety_adjusted_success_rate"),
                    "mean_tox_path_A": s.get("mean_tox_path"),
                    "mean_common_essential_per_ep_A": s.get("mean_common_essential_per_ep"),
                    "fraction_zero_common_essential_A": s.get("fraction_zero_common_essential"),
                    "weighted_mean_chronos_A": s.get("weighted_mean_chronos"),
                    "weighted_mean_tox_A": s.get("weighted_mean_tox"),
                    "fraction_actions_common_essential_A": s.get("fraction_actions_common_essential"),
                })
    return pl.DataFrame(rows)


def compute_cis(df: pl.DataFrame) -> dict[str, Any]:
    """Compute 4-seed CIs per (cell, policy) on raw success + key reward-fit metrics."""
    out: dict[str, Any] = {}
    for cell in CELLS:
        out[cell] = {}
        for policy in POLICIES_ALL:
            sub = df.filter((pl.col("cell") == cell) & (pl.col("policy") == policy))
            entry: dict[str, Any] = {"n_seeds": int(sub.height)}
            for key in [
                "raw_success_rate_B", "mean_steps_B", "mean_final_distance_B",
                "safety_adjusted_SR_A", "mean_tox_path_A",
                "mean_common_essential_per_ep_A",
                "fraction_zero_common_essential_A",
                "weighted_mean_chronos_A",
            ]:
                vals = [v for v in sub[key].to_list() if v is not None]
                m, lo, hi = _normal_ci(vals)
                entry[f"{key}_mean"] = m
                entry[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                entry[f"{key}_ci_lo"] = lo
                entry[f"{key}_ci_hi"] = hi
                entry[f"{key}_per_seed"] = list(vals)
            out[cell][policy] = entry
    return out


def compute_paired_deltas(df: pl.DataFrame) -> dict[str, Any]:
    """Paired-by-seed deltas: per-seed (PPO_C − PPO_A), (PPO_C − greedy_dyn_2_C), (real − permuted).

    Paired: subtract per seed first, then aggregate. Tighter CI than independent.
    """
    out: dict[str, Any] = {}
    for cell in CELLS:
        out[cell] = {}
        # Pull per-seed rows
        per_seed: dict[int, dict[str, float | None]] = {}
        for seed in SEEDS:
            per_seed[seed] = {}
            for policy in POLICIES_ALL:
                sub = df.filter(
                    (pl.col("cell") == cell)
                    & (pl.col("policy") == policy)
                    & (pl.col("seed") == seed)
                )
                if sub.is_empty():
                    per_seed[seed][policy] = None
                else:
                    per_seed[seed][policy] = float(sub["raw_success_rate_B"][0])

        # PPO_C − PPO_A
        deltas_C_minus_A = []
        for seed in SEEDS:
            c = per_seed[seed].get("ppo_C")
            a = per_seed[seed].get("ppo_A")
            if c is not None and a is not None:
                deltas_C_minus_A.append(c - a)
        mC_A, loC_A, hiC_A = _normal_ci(deltas_C_minus_A)
        out[cell]["delta_PPO_C_minus_PPO_A_raw_paired"] = {
            "per_seed_deltas": deltas_C_minus_A,
            "mean": mC_A, "ci_lo": loC_A, "ci_hi": hiC_A,
            "std": float(np.std(deltas_C_minus_A, ddof=1)) if len(deltas_C_minus_A) > 1 else 0.0,
            "ci_excludes_zero": (loC_A > 0 or hiC_A < 0),
        }

        # PPO_C − greedy_dyn_2_C
        deltas_C_minus_grdC = []
        for seed in SEEDS:
            c = per_seed[seed].get("ppo_C")
            g = per_seed[seed].get("greedy_dyn_2_C")
            if c is not None and g is not None:
                deltas_C_minus_grdC.append(c - g)
        mCG, loCG, hiCG = _normal_ci(deltas_C_minus_grdC)
        out[cell]["delta_PPO_C_minus_greedy_dyn_2_C_raw_paired"] = {
            "per_seed_deltas": deltas_C_minus_grdC,
            "mean": mCG, "ci_lo": loCG, "ci_hi": hiCG,
            "std": float(np.std(deltas_C_minus_grdC, ddof=1)) if len(deltas_C_minus_grdC) > 1 else 0.0,
            "ci_excludes_zero": (loCG > 0 or hiCG < 0),
        }

        # real − permuted (paired by seed)
        deltas_real_minus_perm = []
        for seed in SEEDS:
            c = per_seed[seed].get("ppo_C")
            p = per_seed[seed].get("ppo_C_permuted")
            if c is not None and p is not None:
                deltas_real_minus_perm.append(c - p)
        mRP, loRP, hiRP = _normal_ci(deltas_real_minus_perm)
        out[cell]["delta_real_minus_permuted_PPO_C_raw_paired"] = {
            "per_seed_deltas": deltas_real_minus_perm,
            "mean": mRP, "ci_lo": loRP, "ci_hi": hiRP,
            "std": float(np.std(deltas_real_minus_perm, ddof=1)) if len(deltas_real_minus_perm) > 1 else 0.0,
            "ci_excludes_zero": (loRP > 0 or hiRP < 0),
        }
    return out


def apply_acceptance(cis: dict[str, Any], deltas: dict[str, Any]) -> dict[str, Any]:
    """Apply Phase 2c acceptance criteria."""
    primary_delta = deltas[LEAKAGE_SAFE_HEADLINE_CELL]["delta_PPO_C_minus_PPO_A_raw_paired"]
    perm_delta = deltas[LEAKAGE_SAFE_HEADLINE_CELL]["delta_real_minus_permuted_PPO_C_raw_paired"]

    # Bounded regression at K=2/bin 6-8
    k2_68_PPOC = cis[K2_BIN68_CELL]["ppo_C"]["raw_success_rate_B_mean"]
    k2_68_PPOA = cis[K2_BIN68_CELL]["ppo_A"]["raw_success_rate_B_mean"]
    regression = (k2_68_PPOA - k2_68_PPOC) if (k2_68_PPOC is not None and k2_68_PPOA is not None) else None
    bounded_threshold = 0.05  # mean PPO_C >= PPO_A - 0.05
    regression_bounded = (regression is not None and regression <= bounded_threshold)

    return {
        "rule_1_PPO_C_minus_PPO_A_at_K2_bin8-10_OOD": {
            "cell": LEAKAGE_SAFE_HEADLINE_CELL,
            "mean": primary_delta["mean"],
            "ci_lo": primary_delta["ci_lo"],
            "ci_hi": primary_delta["ci_hi"],
            "per_seed_deltas": primary_delta["per_seed_deltas"],
            "ci_strictly_excludes_zero": primary_delta["ci_excludes_zero"],
            "passed": bool(primary_delta["ci_excludes_zero"] and primary_delta["mean"] > 0),
        },
        "rule_2_real_minus_permuted_at_K2_bin8-10_OOD": {
            "cell": LEAKAGE_SAFE_HEADLINE_CELL,
            "mean": perm_delta["mean"],
            "ci_lo": perm_delta["ci_lo"],
            "ci_hi": perm_delta["ci_hi"],
            "per_seed_deltas": perm_delta["per_seed_deltas"],
            "ci_strictly_excludes_zero": perm_delta["ci_excludes_zero"],
            "passed": bool(perm_delta["ci_excludes_zero"] and perm_delta["mean"] > 0),
        },
        "rule_3_bounded_regression_at_K2_bin6-8": {
            "cell": K2_BIN68_CELL,
            "mean_PPO_C_raw_4seed": k2_68_PPOC,
            "mean_PPO_A_raw_4seed": k2_68_PPOA,
            "regression": regression,
            "threshold": bounded_threshold,
            "passed": bool(regression_bounded),
        },
    }


def write_csv(df: pl.DataFrame, out_path: Path) -> None:
    df.write_csv(str(out_path))
    LOG.info("Wrote %s (%d rows)", out_path, df.height)


def write_results_json(cis, deltas, acceptance, out_path: Path) -> None:
    out = {
        "seeds": list(SEEDS),
        "cells": list(CELLS),
        "primary_cell_v2_protocol": PRIMARY_CELL,
        "leakage_safe_headline_cell": LEAKAGE_SAFE_HEADLINE_CELL,
        "per_cell_per_policy_4seed_CIs": cis,
        "per_cell_paired_deltas_4seed_CIs": deltas,
        "acceptance_criteria": acceptance,
        "final_verdict": derive_final_verdict(acceptance),
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("Wrote %s", out_path)


def derive_final_verdict(acceptance: dict[str, Any]) -> str:
    """Return the Phase 2c headline label."""
    r1 = acceptance["rule_1_PPO_C_minus_PPO_A_at_K2_bin8-10_OOD"]["passed"]
    r2 = acceptance["rule_2_real_minus_permuted_at_K2_bin8-10_OOD"]["passed"]
    r3 = acceptance["rule_3_bounded_regression_at_K2_bin6-8"]["passed"]
    if r1 and r2 and r3:
        return "PHASE2C_PASS_STATISTICALLY_SUPPORTED_REWARD_PRIOR_OPTIMIZATION"
    if r1 and r3 and not r2:
        return "PHASE2C_PARTIAL_PASS_PPO_BEATS_BASELINE_BUT_LABEL_STRUCTURE_INCONCLUSIVE"
    if r2 and not r1:
        return "PHASE2C_LABEL_STRUCTURE_USEFUL_BUT_NO_DECISIVE_BASELINE_ADVANTAGE"
    if not r1 and not r2 and r3:
        return "PHASE2C_PROMISING_BUT_UNSTABLE_SEED_VARIANCE_DOMINATES"
    return "PHASE2C_INCONCLUSIVE_OR_REGRESSION_REVIEW_DETAILS"


def write_summary_md(cis, deltas, acceptance, df, out_path: Path) -> None:
    md: list[str] = []
    md.append("# V3B Phase 2c — 4-seed escalation summary")
    md.append("")
    md.append(f"> Seeds: {list(SEEDS)}.  Cells: 4 V2 hardness-frontier × n=300 episodes each.")
    md.append(f"> dynamics: artifacts_v2/dynamics_v1ot_ror_corr010 (V2 primary 32D).")
    md.append(f"> reward: safety_aware (λ_tox=0.10, λ_ce=0.05).")
    md.append("")
    md.append(f"## Final verdict")
    md.append(f"")
    md.append(f"**`{derive_final_verdict(acceptance)}`**")
    md.append("")
    md.append("### Acceptance criteria (4-seed paired deltas)")
    md.append("")
    md.append("| # | Rule | Mean ± 95% CI | Passed |")
    md.append("|---|---|---|:---:|")
    for k, label in [
        ("rule_1_PPO_C_minus_PPO_A_at_K2_bin8-10_OOD",
            "PPO_C − PPO_A raw success at K=2/bin 8-10/OOD strictly > 0"),
        ("rule_2_real_minus_permuted_at_K2_bin8-10_OOD",
            "real − permuted PPO_C raw success at K=2/bin 8-10/OOD strictly > 0"),
        ("rule_3_bounded_regression_at_K2_bin6-8",
            "PPO_C raw success at K=2/bin 6-8/OOD ≥ PPO_A − 0.05"),
    ]:
        r = acceptance[k]
        if "mean" in r:
            md.append(f"| {k.split('_')[1]} | {label} | "
                      f"**{r['mean']:+.4f}** [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                      f"{'✅' if r['passed'] else '❌'} |")
        else:
            md.append(f"| 3 | {label} | "
                      f"PPO_C={r['mean_PPO_C_raw_4seed']:.4f} vs PPO_A={r['mean_PPO_A_raw_4seed']:.4f} "
                      f"(regression={r['regression']:+.4f} ≤ {r['threshold']:.2f}) | "
                      f"{'✅' if r['passed'] else '❌'} |")

    md.append("")
    md.append("## Bucket-B (reward-independent) raw-success per cell, 4-seed mean ± std")
    md.append("")
    md.append("| Cell | ppo_C | ppo_A | ppo_C_permuted | greedy_dyn_2_C | greedy_dyn_2_A |")
    md.append("|---|---|---|---|---|---|")
    for cell in CELLS:
        row = f"| {cell} |"
        for pol in ("ppo_C", "ppo_A", "ppo_C_permuted", "greedy_dyn_2_C", "greedy_dyn_2_A"):
            c = cis[cell][pol]
            m = c["raw_success_rate_B_mean"]
            s = c["raw_success_rate_B_std"]
            row += f" {m:.3f} ± {s:.3f} |"
        md.append(row)
    md.append("")

    md.append("## Bucket-B paired-by-seed deltas (4-seed 95% CI)")
    md.append("")
    md.append("| Cell | Δ(PPO_C − PPO_A) | Δ(PPO_C − greedy_dyn_2_C) | Δ(real − permuted) |")
    md.append("|---|---|---|---|")
    for cell in CELLS:
        d_ca = deltas[cell]["delta_PPO_C_minus_PPO_A_raw_paired"]
        d_cg = deltas[cell]["delta_PPO_C_minus_greedy_dyn_2_C_raw_paired"]
        d_rp = deltas[cell]["delta_real_minus_permuted_PPO_C_raw_paired"]
        def _fmt(d):
            tag = "✅" if d["ci_excludes_zero"] else "—"
            return f"{d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] {tag}"
        md.append(f"| {cell} | {_fmt(d_ca)} | {_fmt(d_cg)} | {_fmt(d_rp)} |")
    md.append("")

    md.append("## Bucket-A (reward-fit) metrics, 4-seed mean ± std")
    md.append("")
    md.append("Reminder: Bucket-A metrics are derived from DepMap Chronos which is in the reward; "
              "improving them is *reward-prior optimization*, not independent biological discovery.")
    md.append("")
    md.append("| Cell | Policy | safety_adj_SR | mean_tox | mean_CE/ep | wmean_chronos |")
    md.append("|---|---|---|---|---|---|")
    for cell in CELLS:
        for pol in ("ppo_C", "ppo_A", "ppo_C_permuted", "greedy_dyn_2_C", "greedy_dyn_2_A"):
            c = cis[cell][pol]
            def _ms(key):
                m = c[f"{key}_mean"]
                s = c[f"{key}_std"]
                if m is None or (isinstance(m, float) and np.isnan(m)):
                    return "—"
                return f"{m:.3f} ± {s:.3f}"
            md.append(f"| {cell} | {pol} | {_ms('safety_adjusted_SR_A')} | "
                      f"{_ms('mean_tox_path_A')} | "
                      f"{_ms('mean_common_essential_per_ep_A')} | "
                      f"{_ms('weighted_mean_chronos_A')} |")
    md.append("")

    md.append("## Per-seed raw success at K=2/bin 8-10/OOD (the leakage-safe headline cell)")
    md.append("")
    md.append("| Seed | ppo_C | ppo_A | ppo_C_permuted | greedy_dyn_2_C | greedy_dyn_2_A |")
    md.append("|---|---|---|---|---|---|")
    for seed in SEEDS:
        row = f"| {seed} |"
        for pol in ("ppo_C", "ppo_A", "ppo_C_permuted", "greedy_dyn_2_C", "greedy_dyn_2_A"):
            sub = df.filter(
                (pl.col("seed") == seed)
                & (pl.col("cell") == LEAKAGE_SAFE_HEADLINE_CELL)
                & (pl.col("policy") == pol)
            )
            if sub.is_empty():
                row += " — |"
            else:
                row += f" {float(sub['raw_success_rate_B'][0]):.3f} |"
        md.append(row)
    md.append("")

    md.append("## Held-out biological validation (Bucket C)")
    md.append("")
    md.append("**Status: pending_no_local_source.** No Bucket-C source (Replogle 2022 K562 essential "
              "CRISPRi Perturb-seq, OGEE v3, COSMIC CGC, Open Targets) is available locally for "
              "independent validation in this session. See `artifacts_v3/eval_v3b_phase2b/source_usage_table.md` "
              "for the full rationale.")
    md.append("")

    out_path.write_text("\n".join(md) + "\n")
    LOG.info("Wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", default="artifacts_v3/eval_v3b_phase2c")
    parser.add_argument("--out_dir", default="artifacts_v3/eval_v3b_phase2c")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    repo_root = Path(__file__).resolve().parents[1]
    eval_root = (repo_root / args.eval_root) if not Path(args.eval_root).is_absolute() else Path(args.eval_root)
    out_dir = (repo_root / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_per_seed(eval_root)
    LOG.info("Loaded %d rows (%d seeds × %d cells × %d policies expected = %d)",
             df.height, N_SEEDS, len(CELLS), len(POLICIES_ALL), N_SEEDS * len(CELLS) * len(POLICIES_ALL))

    # CSV
    write_csv(df, out_dir / "seed_escalation_results.csv")

    cis = compute_cis(df)
    deltas = compute_paired_deltas(df)
    acceptance = apply_acceptance(cis, deltas)

    write_results_json(cis, deltas, acceptance, out_dir / "seed_escalation_results.json")
    write_summary_md(cis, deltas, acceptance, df, out_dir / "seed_escalation_summary.md")

    print(f"\n=== Phase 2c verdict: {derive_final_verdict(acceptance)} ===\n")
    for k, r in acceptance.items():
        print(f"  {k}: passed={r['passed']}")
        if "mean" in r:
            print(f"    mean={r['mean']:+.4f}  CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
