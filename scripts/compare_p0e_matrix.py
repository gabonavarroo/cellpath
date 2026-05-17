"""P0E Phase 5 — one-shot aggregator over the P0E evaluation matrix.

Walks `artifacts_v2/eval_p0e_matrix/<run>/<cell>/<policy>/summary.json` (and the in-tree
`artifacts_v2/eval_p0e_b5_extended_with_beam_baselines/` for the B5 reference) and emits
a single human-readable Markdown table at
`artifacts_v2/eval_p0e_matrix/comparison.md`.

Pure aggregation: no policy / dynamics is loaded. Reads only `summary.json` files.

Usage::

    python scripts/compare_p0e_matrix.py --root artifacts_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect_cells(run_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {cell: {policy: summary}} for one run's eval directory."""
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


def fmt(x, fmt_spec="{:.3f}", missing="  NA "):
    if x is None:
        return missing
    try:
        return fmt_spec.format(float(x))
    except (TypeError, ValueError):
        return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts_v2")
    ap.add_argument(
        "--matrix_dir",
        default="artifacts_v2/eval_p0e_matrix",
        help="Directory containing one subdir per <run> with the hard-bench output tree.",
    )
    ap.add_argument(
        "--b5_eval_dir",
        default="artifacts_v2/eval_p0e_b5_extended_with_beam_baselines",
        help="The reference B5 evaluation (Phase 0).",
    )
    ap.add_argument("--out", default="artifacts_v2/eval_p0e_matrix/comparison.md")
    args = ap.parse_args()

    matrix_dir = Path(args.matrix_dir)
    b5_dir = Path(args.b5_eval_dir)

    runs: dict[str, dict[str, dict[str, dict]]] = {}
    # B5 reference
    runs["B5 (V1 OT × terminal+curric K=3 1M)"] = collect_cells(b5_dir)
    # All matrix runs
    for run_dir in sorted(matrix_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name != "":
            runs[run_dir.name] = collect_cells(run_dir)

    # Union of cell names across all runs.
    all_cells: list[str] = sorted({c for r in runs.values() for c in r.keys()})

    lines: list[str] = []
    lines.append("# P0E Phase 5 — Combinatorial evaluation matrix\n")
    lines.append(f"Generated from `{matrix_dir}` + `{b5_dir}`.\n")
    lines.append("Columns: PPO_success, rnd, grd1, grd2, PPO−grd1, **PPO−grd2** (planning delta), mean_steps, mean_final_dist.\n")

    for cell in all_cells:
        lines.append(f"\n## Cell: `{cell}`\n")
        lines.append("| Run | PPO | rnd | grd1 | grd2 | PPO−grd1 | **PPO−grd2** | mean_steps | mean_d |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for run_name, run_cells in runs.items():
            policies = run_cells.get(cell, {})
            ppo = policies.get("ppo_deterministic", {})
            rnd = policies.get("random_uniform_valid", {})
            g1  = policies.get("greedy_dyn_1", {})
            g2  = policies.get("greedy_dyn_2", {})

            ppo_s = ppo.get("success_rate")
            rnd_s = rnd.get("success_rate")
            g1_s  = g1.get("success_rate")
            g2_s  = g2.get("success_rate")
            steps = ppo.get("mean_steps")
            mean_d = ppo.get("mean_final_distance")

            d_g1 = (ppo_s - g1_s) if (ppo_s is not None and g1_s is not None) else None
            d_g2 = (ppo_s - g2_s) if (ppo_s is not None and g2_s is not None) else None
            lines.append(
                f"| {run_name} | {fmt(ppo_s)} | {fmt(rnd_s)} | {fmt(g1_s)} | {fmt(g2_s)} | "
                f"{fmt(d_g1, '{:+.3f}')} | {fmt(d_g2, '{:+.3f}')} | "
                f"{fmt(steps, '{:.2f}', '  NA')} | {fmt(mean_d, '{:.3f}')} |"
            )

    # Aggregate: per-run best PPO − greedy_dyn_2 across cells
    lines.append("\n## Aggregate: best PPO − greedy_dyn_2 per run\n")
    lines.append("| Run | best cell | PPO success there | greedy_dyn_2 there | **PPO − grd2** |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for run_name, run_cells in runs.items():
        best_cell = None
        best_delta = -1e9
        best_ppo = None
        best_g2 = None
        for cell, policies in run_cells.items():
            ppo = policies.get("ppo_deterministic", {}).get("success_rate")
            g2 = policies.get("greedy_dyn_2", {}).get("success_rate")
            if ppo is None or g2 is None:
                continue
            d = ppo - g2
            if d > best_delta:
                best_delta = d
                best_cell = cell
                best_ppo = ppo
                best_g2 = g2
        if best_cell is None:
            lines.append(f"| {run_name} | (no cells with both metrics) | NA | NA | NA |")
        else:
            lines.append(
                f"| {run_name} | `{best_cell}` | {fmt(best_ppo)} | {fmt(best_g2)} | "
                f"{fmt(best_delta, '{:+.3f}')} |"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
