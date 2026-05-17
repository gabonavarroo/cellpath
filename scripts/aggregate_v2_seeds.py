"""P0F Phase 3 — aggregate per-seed evaluation outputs into 95 % CIs.

Walks ``artifacts_v2/eval_p0f_<config>_seed<X>/<cell>/<policy>/summary.json`` for each
seed in {42, 0, 1, 7} and each config in {B5, C2} and emits:

* ``artifacts_v2/eval_p0f_seed_aggregate/seed_aggregate.json`` — per-(config, cell, policy)
  records with ``mean``, ``std``, ``min``, ``max``, ``n_seeds``, ``ci95_low``, ``ci95_high``.
* ``artifacts_v2/eval_p0f_seed_aggregate/seed_aggregate.md`` — human-readable table.

The CI is computed via the normal-approximation across seeds (mean ± 1.96 · std / sqrt(n))
which is the right thing when n=4 and we are bounded in [0, 1]. We also report Wilson-95
on the *pooled* success count (sum of successes / sum of episodes) as a secondary CI that
respects the binomial nature of the data — both are honest, with the seed-CI capturing
*model-training* variance and the Wilson-CI capturing *evaluation* variance.

Pure aggregation: reads only summary.json files; no model loading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect(run_dir: Path) -> dict[str, dict[str, dict]]:
    """{cell: {policy: summary}} for one PPO run's evaluation."""
    out: dict[str, dict[str, dict]] = {}
    if not run_dir.exists():
        return out
    for cell_dir in sorted(run_dir.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("k"):
            continue
        by_pol: dict[str, dict] = {}
        for pol_dir in cell_dir.iterdir():
            if not pol_dir.is_dir():
                continue
            blob = _safe_load(pol_dir / "summary.json")
            if blob is not None:
                by_pol[pol_dir.name] = blob
        if by_pol:
            out[cell_dir.name] = by_pol
    return out


def wilson_95(successes: int, n: int) -> tuple[float, float]:
    """Wilson 95 % CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_across_seeds(
    seed_dirs: list[Path],
    metric_key: str = "success_rate",
) -> dict[str, dict[str, dict[str, Any]]]:
    """{cell: {policy: stats}} aggregated across seeds.

    stats = {mean, std, min, max, n_seeds, ci95_low_normal, ci95_high_normal,
             ci95_low_wilson_pooled, ci95_high_wilson_pooled, values, n_per_seed}
    """
    # Collect all per-(cell, policy, seed) values.
    by_cell: dict[str, dict[str, list[tuple[float, dict]]]] = {}
    for sd in seed_dirs:
        for cell, by_pol in collect(sd).items():
            for pol, blob in by_pol.items():
                v = blob.get(metric_key)
                if v is None:
                    continue
                by_cell.setdefault(cell, {}).setdefault(pol, []).append(
                    (float(v), blob)
                )

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cell, by_pol in by_cell.items():
        out[cell] = {}
        for pol, recs in by_pol.items():
            vals = np.asarray([r[0] for r in recs], dtype=np.float64)
            n_per_seed = [int(r[1].get("n_episodes", 0)) for r in recs]
            successes = [int(r[1].get("successes", round(r[0] * n_per_seed[i])))
                         for i, r in enumerate(recs)]

            mean = float(vals.mean())
            std  = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
            n    = int(len(vals))

            if n > 1:
                z = 1.959963984540054
                half = z * std / np.sqrt(n)
                lo_norm, hi_norm = mean - half, mean + half
            else:
                lo_norm, hi_norm = mean, mean

            # Pooled Wilson over the sum across seeds (binomial on the combined evaluation).
            tot_succ = int(sum(successes))
            tot_eps  = int(sum(n_per_seed))
            lo_w, hi_w = wilson_95(tot_succ, tot_eps)

            out[cell][pol] = {
                "mean":   mean,
                "std":    std,
                "min":    float(vals.min()),
                "max":    float(vals.max()),
                "n_seeds": n,
                "ci95_low_normal":  float(max(0.0, min(1.0, lo_norm))),
                "ci95_high_normal": float(max(0.0, min(1.0, hi_norm))),
                "ci95_low_wilson_pooled":  float(lo_w),
                "ci95_high_wilson_pooled": float(hi_w),
                "values": vals.tolist(),
                "n_per_seed": n_per_seed,
                "successes_per_seed": successes,
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="P0F seed-sweep aggregator.")
    ap.add_argument(
        "--b5_eval_dirs",
        nargs="+",
        required=True,
        help="Per-seed eval dirs for the V1 OT × B5 config. Each must contain <cell>/<policy>/summary.json.",
    )
    ap.add_argument(
        "--c2_eval_dirs",
        nargs="+",
        required=True,
        help="Per-seed eval dirs for the RoR_corr010 × C2 config.",
    )
    ap.add_argument(
        "--metric",
        default="success_rate",
        help="Metric key to aggregate. Default 'success_rate'; also useful: 'mean_final_distance'.",
    )
    ap.add_argument(
        "--out",
        default="artifacts_v2/eval_p0f_seed_aggregate",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    b5 = summarize_across_seeds([Path(p) for p in args.b5_eval_dirs], metric_key=args.metric)
    c2 = summarize_across_seeds([Path(p) for p in args.c2_eval_dirs], metric_key=args.metric)

    record = {
        "metric":                    args.metric,
        "b5_v1ot_terminal_curric_1M": b5,
        "c2_ror_corr010_terminal_curric_1M": c2,
        "b5_eval_dirs":              [str(p) for p in args.b5_eval_dirs],
        "c2_eval_dirs":              [str(p) for p in args.c2_eval_dirs],
    }
    (out / f"seed_aggregate_{args.metric}.json").write_text(json.dumps(record, indent=2))

    # Build the comparison MD table.
    all_cells = sorted(set(b5.keys()) | set(c2.keys()))
    policies = ["ppo_deterministic", "random_uniform_valid", "always_noop",
                "greedy_dyn_1", "greedy_dyn_2"]

    lines: list[str] = []
    lines.append(f"# P0F Seed Aggregate — metric=`{args.metric}`\n")
    lines.append(f"B5 seeds: {[d.split('seed')[-1] for d in record['b5_eval_dirs']]}")
    lines.append(f"C2 seeds: {[d.split('seed')[-1] for d in record['c2_eval_dirs']]}\n")

    for cell in all_cells:
        lines.append(f"\n## `{cell}`\n")
        lines.append("| Config | Policy | mean | std | min | max | CI95 (normal) | Wilson95 (pooled) |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
        for cfg_name, blob in [("B5 (V1 OT)", b5.get(cell, {})),
                                ("C2 (RoR_corr010)", c2.get(cell, {}))]:
            for pol in policies:
                stats = blob.get(pol)
                if stats is None:
                    continue
                ci_n = f"[{stats['ci95_low_normal']:.3f}, {stats['ci95_high_normal']:.3f}]"
                ci_w = f"[{stats['ci95_low_wilson_pooled']:.3f}, {stats['ci95_high_wilson_pooled']:.3f}]"
                lines.append(
                    f"| {cfg_name} | `{pol}` | {stats['mean']:.3f} | {stats['std']:.3f} | "
                    f"{stats['min']:.3f} | {stats['max']:.3f} | {ci_n} | {ci_w} |"
                )

    # Aggregate planning-delta comparison (PPO - greedy_dyn_2 across seeds, per cell, per config).
    lines.append("\n## Aggregate: PPO − greedy_dyn_2 (per cell, per config, across seeds)\n")
    lines.append("| Cell | B5 PPO−grd2 mean | B5 PPO−grd2 95%CI | C2 PPO−grd2 mean | C2 PPO−grd2 95%CI | tied? |")
    lines.append("| --- | ---: | --- | ---: | --- | --- |")
    for cell in all_cells:
        b5_ppo = b5.get(cell, {}).get("ppo_deterministic")
        b5_g2  = b5.get(cell, {}).get("greedy_dyn_2")
        c2_ppo = c2.get(cell, {}).get("ppo_deterministic")
        c2_g2  = c2.get(cell, {}).get("greedy_dyn_2")
        if not (b5_ppo and b5_g2 and c2_ppo and c2_g2):
            continue

        # Pairwise PPO − greedy_dyn_2 per seed (assumes same seed ordering).
        def pairwise_diff(ppo: dict, g2: dict) -> tuple[float, float]:
            a = np.asarray(ppo["values"])
            b = np.asarray(g2["values"])
            n = min(len(a), len(b))
            if n == 0:
                return float("nan"), float("nan")
            d = a[:n] - b[:n]
            m = float(d.mean())
            s = float(d.std(ddof=1) if n > 1 else 0.0)
            z = 1.959963984540054
            half = z * s / np.sqrt(n)
            return m, half

        b5_m, b5_h = pairwise_diff(b5_ppo, b5_g2)
        c2_m, c2_h = pairwise_diff(c2_ppo, c2_g2)
        # Tied if the two configs' CIs overlap on PPO−grd2:
        b5_lo, b5_hi = b5_m - b5_h, b5_m + b5_h
        c2_lo, c2_hi = c2_m - c2_h, c2_m + c2_h
        tied = "yes" if not (b5_hi < c2_lo or c2_hi < b5_lo) else "no"
        lines.append(
            f"| `{cell}` | {b5_m:+.3f} | [{b5_lo:+.3f}, {b5_hi:+.3f}] | "
            f"{c2_m:+.3f} | [{c2_lo:+.3f}, {c2_hi:+.3f}] | {tied} |"
        )

    (out / f"seed_aggregate_{args.metric}.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}/seed_aggregate_{args.metric}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
