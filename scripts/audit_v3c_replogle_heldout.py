"""Final V3C Replogle held-out action-overlap audit.

Stage 4 of the V3C final-Replogle audit (see v3c_final_replogle_heldout_audit.md).

Computes per-policy, per-cell, per-seed action-overlap statistics for the V3C
champion (contraction_aware_v2_aggressive + PPO_BCD) against:
* same-field greedy_dyn_{1,2,3}_fused
* random_uniform_valid (random baseline on the same dynamics field)
* always_noop (control)
* Track L 4-seed (LOCKED_DEFAULT secondary reference)
* V2 anchor PPO_BCD (cross-field baseline)

Action overlap is measured against four gene sets, all intersected with the
Norman 105 single-gene CRISPRa action universe:

* Replogle-essential       (Replogle ∩ Norman 105)              — 6 genes
* Replogle-only essential  (Replogle ∩ Norman 105 \\ DepMap)     — 4 genes (FOXL2, KIF18B, NCL, SET)
* DepMap-essential          (DepMap ∩ Norman 105)                — 5 genes
* non-essential action universe (Norman 105 \\ Replogle \\ DepMap) — 95 genes (complement)

Outputs under `artifacts_v3/v3c/`:
* `replogle_heldout_action_overlap.csv`
* `interpretation/v3c_final_replogle_heldout_audit_metrics.json`

The script reads only existing per-policy `summary.json` files (action_freq
payloads). It does NOT retrain PPO, dynamics, or VAE; it does NOT re-run rollouts.

Usage:
    python scripts/audit_v3c_replogle_heldout.py
    python scripts/audit_v3c_replogle_heldout.py --primary-cell-only
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

LOG = logging.getLogger("audit_v3c_replogle_heldout")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Gene-set source — written by scripts/download_replogle_heldout.py
GENE_SET_DIR = REPO_ROOT / "data" / "processed" / "replogle"
INTERSECTION_JSON = GENE_SET_DIR / "replogle_norman_intersection.json"
FALLBACK_PHASE2C_JSON = (
    REPO_ROOT / "artifacts_v3" / "eval_v3b_phase2c" / "replogle_norman_intersection.json"
)

# Outputs
OUT_DIR = REPO_ROOT / "artifacts_v3" / "v3c"
OUT_CSV = OUT_DIR / "replogle_heldout_action_overlap.csv"
OUT_JSON = OUT_DIR / "interpretation" / "v3c_final_replogle_heldout_audit_metrics.json"

# Run configurations to audit
CHAMPION_RUNS = [
    # (run_label, dir, seed, policies_to_load, cells)
    (
        "v2_aggressive_champion_seed42",
        REPO_ROOT
        / "artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed42_500k/eval",
        42,
        ["PPO_BCD", "greedy_dyn_1_fused", "greedy_dyn_2_fused", "greedy_dyn_3_fused",
         "random_uniform_valid", "always_noop"],
        None,  # auto-detect available cells
    ),
    (
        "v2_aggressive_seed0",
        REPO_ROOT / "artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed0_500k/eval",
        0,
        ["PPO_BCD"],
        None,
    ),
    (
        "v2_aggressive_seed1",
        REPO_ROOT / "artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed1_500k/eval",
        1,
        ["PPO_BCD"],
        None,
    ),
    (
        "v2_aggressive_seed7",
        REPO_ROOT / "artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed7_500k/eval",
        7,
        ["PPO_BCD"],
        None,
    ),
]

# Secondary references (Track L 4-seed locked) — per-seed under eval/seed{N}/...
TRACK_L_SEEDS = [42, 0, 1, 7]
TRACK_L_EVAL_ROOT = (
    REPO_ROOT / "artifacts_v3/v3c/rl_final/track_l_4seed_locked/eval"
)

# V2 anchor (cross-field)
ANCHOR_RUN = (
    "v2_anchor_seed42",
    REPO_ROOT / "artifacts_v3/v3c/rl_smokes/anchor_v2_ror_corr010_1M_reused/eval",
    42,
    ["PPO_BCD", "PPO_A", "greedy_dyn_1_fused", "greedy_dyn_2_fused", "greedy_dyn_3_fused",
     "greedy_dyn_5_fused", "random_uniform_valid"],
)

PRIMARY_CELL = "k3_bin8-10_splitood"


def load_gene_sets() -> dict[str, Any]:
    """Load Replogle-essential, Replogle-only-essential, DepMap-essential
    Norman-105-intersected sets. Prefer fresh download artifacts; fall back to
    Phase 2c intersection JSON if absent."""
    if INTERSECTION_JSON.exists():
        src = INTERSECTION_JSON
        src_label = "fresh_download_processed"
    elif FALLBACK_PHASE2C_JSON.exists():
        src = FALLBACK_PHASE2C_JSON
        src_label = "phase2c_cached"
    else:
        raise SystemExit(
            "No Replogle intersection JSON found. Run scripts/download_replogle_heldout.py first."
        )
    payload = json.loads(src.read_text())
    return {
        "source_file": str(src.relative_to(REPO_ROOT)),
        "source_label": src_label,
        "replogle_in_norman": payload["norman_genes_in_replogle_essential"],
        "depmap_in_norman": payload["depmap_essential_in_norman"],
        "replogle_only_in_norman": payload["replogle_only_essential"],
        "agreement": payload["agreement_replogle_and_depmap_essential"],
        "n_norman_105": payload["n_norman_105"],
    }


def load_action_freq(summary_path: Path) -> tuple[dict[str, int], float | None, int | None]:
    """Load `action_freq`, `success_rate`, and `n_episodes` from a per-policy summary.json.

    Returns ({} , None, None) if the file is missing or malformed.
    """
    if not summary_path.exists():
        return {}, None, None
    try:
        d = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        LOG.warning("malformed JSON: %s", summary_path)
        return {}, None, None
    return (
        d.get("action_freq", {}) or {},
        d.get("success_rate"),
        d.get("n_episodes"),
    )


def overlap_metrics(
    action_freq: dict[str, int],
    replogle: set[str],
    replogle_only: set[str],
    depmap: set[str],
    norman_size: int,
) -> dict[str, Any]:
    """Compute action-overlap stats vs the four gene sets."""
    # NO_OP is not a gene action — exclude from denominators.
    gene_actions = {g: c for g, c in action_freq.items() if g != "NO_OP" and g}
    total = sum(gene_actions.values())
    if total == 0:
        return {
            "total_gene_actions": 0,
            "n_actions_replogle_essential": 0,
            "n_actions_replogle_only_essential": 0,
            "n_actions_depmap_essential": 0,
            "n_actions_non_essential": 0,
            "frac_actions_replogle_essential": None,
            "frac_actions_replogle_only_essential": None,
            "frac_actions_depmap_essential": None,
            "frac_actions_non_essential": None,
            "enrichment_replogle_only_vs_random": None,
            "enrichment_replogle_essential_vs_random": None,
            "unique_genes_selected": 0,
            "replogle_only_genes_selected": [],
            "replogle_only_gene_freqs": {},
        }
    n_rep = sum(c for g, c in gene_actions.items() if g in replogle)
    n_rep_only = sum(c for g, c in gene_actions.items() if g in replogle_only)
    n_dep = sum(c for g, c in gene_actions.items() if g in depmap)
    n_non = total - sum(c for g, c in gene_actions.items() if g in (replogle | depmap))

    # Random expected baseline = |set| / |Norman 105|
    exp_rep_only = len(replogle_only) / norman_size
    exp_rep = len(replogle) / norman_size

    frac_rep_only = n_rep_only / total
    frac_rep = n_rep / total

    return {
        "total_gene_actions": total,
        "n_actions_replogle_essential": n_rep,
        "n_actions_replogle_only_essential": n_rep_only,
        "n_actions_depmap_essential": n_dep,
        "n_actions_non_essential": n_non,
        "frac_actions_replogle_essential": frac_rep,
        "frac_actions_replogle_only_essential": frac_rep_only,
        "frac_actions_depmap_essential": n_dep / total,
        "frac_actions_non_essential": n_non / total,
        "expected_random_frac_replogle_essential": exp_rep,
        "expected_random_frac_replogle_only_essential": exp_rep_only,
        "enrichment_replogle_essential_vs_random": (
            frac_rep / exp_rep if exp_rep > 0 else None
        ),
        "enrichment_replogle_only_vs_random": (
            frac_rep_only / exp_rep_only if exp_rep_only > 0 else None
        ),
        "unique_genes_selected": len(gene_actions),
        "replogle_only_genes_selected": sorted(
            g for g in gene_actions if g in replogle_only
        ),
        "replogle_only_gene_freqs": {
            g: gene_actions[g] for g in sorted(gene_actions) if g in replogle_only
        },
    }


def collect_rows(gene_sets: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rep_set = set(gene_sets["replogle_in_norman"])
    rep_only_set = set(gene_sets["replogle_only_in_norman"])
    dep_set = set(gene_sets["depmap_in_norman"])
    norman_size = gene_sets["n_norman_105"]

    def _add(run_label, dynamics_field, seed, cell, policy, summary_path):
        af, sr, n_eps = load_action_freq(summary_path)
        if not af:
            return
        m = overlap_metrics(af, rep_set, rep_only_set, dep_set, norman_size)
        row = {
            "run": run_label,
            "dynamics_field": dynamics_field,
            "seed": seed,
            "cell": cell,
            "policy": policy,
            "success_rate": sr,
            "n_episodes": n_eps,
            **m,
        }
        rows.append(row)

    # Champion + sibling seeds on v2_aggressive
    for run_label, eval_root, seed, policies, cells in CHAMPION_RUNS:
        if not eval_root.exists():
            LOG.warning("missing eval root: %s", eval_root)
            continue
        if cells is None:
            cells = sorted(p.name for p in eval_root.iterdir() if p.is_dir())
        for cell in cells:
            for policy in policies:
                sp = eval_root / cell / policy / "summary.json"
                _add(run_label, "v2_aggressive", seed, cell, policy, sp)

    # Track L 4-seed
    for seed in TRACK_L_SEEDS:
        seed_dir = TRACK_L_EVAL_ROOT / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for cell_dir in sorted(seed_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            cell = cell_dir.name
            for pol_dir in sorted(cell_dir.iterdir()):
                if not pol_dir.is_dir():
                    continue
                sp = pol_dir / "summary.json"
                _add("track_l_4seed", "track_l_n64_legacy", seed, cell, pol_dir.name, sp)

    # V2 anchor
    run_label, eval_root, seed, policies = ANCHOR_RUN
    if eval_root.exists():
        for cell_dir in sorted(eval_root.iterdir()):
            if not cell_dir.is_dir():
                continue
            cell = cell_dir.name
            for policy in policies:
                sp = cell_dir / policy / "summary.json"
                _add(run_label, "v2_anchor", seed, cell, policy, sp)

    return rows


def aggregate_v2_aggressive_4seed(df: pl.DataFrame) -> dict[str, Any]:
    """Aggregate PPO_BCD across v2_aggressive seeds {42, 0, 1, 7} at the primary cell."""
    sub = df.filter(
        (pl.col("dynamics_field") == "v2_aggressive")
        & (pl.col("policy") == "PPO_BCD")
        & (pl.col("cell") == PRIMARY_CELL)
    )
    if sub.is_empty():
        return {"status": "no_data"}
    per_seed = sub.select(
        ["seed", "success_rate", "frac_actions_replogle_only_essential",
         "frac_actions_replogle_essential", "frac_actions_depmap_essential",
         "n_actions_replogle_only_essential", "total_gene_actions",
         "enrichment_replogle_only_vs_random"]
    ).sort("seed").to_dicts()
    arr = np.asarray(
        [r["frac_actions_replogle_only_essential"] for r in per_seed], dtype=np.float64
    )
    mean = float(arr.mean())
    if arr.size > 1:
        se = float(arr.std(ddof=1) / np.sqrt(arr.size))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
    else:
        lo, hi = mean, mean
    return {
        "policy": "PPO_BCD",
        "dynamics_field": "v2_aggressive",
        "cell": PRIMARY_CELL,
        "seeds": [r["seed"] for r in per_seed],
        "per_seed": per_seed,
        "frac_replogle_only_4seed_mean": mean,
        "frac_replogle_only_4seed_ci95": [lo, hi],
        "frac_replogle_only_4seed_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def write_outputs(rows: list[dict[str, Any]], gene_sets: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Drop list-valued fields from CSV so it stays tabular; keep them in JSON.
    csv_drop = {"replogle_only_genes_selected", "replogle_only_gene_freqs"}
    flat = [{k: v for k, v in r.items() if k not in csv_drop} for r in rows]
    pl.DataFrame(flat).write_csv(str(OUT_CSV))
    LOG.info("Wrote %s (%d rows)", OUT_CSV, len(flat))

    df = pl.DataFrame(rows)
    agg_4seed = aggregate_v2_aggressive_4seed(df)

    # Comparison block at the primary cell across v2_aggressive policies
    primary_v2agg = (
        df.filter(
            (pl.col("dynamics_field") == "v2_aggressive")
            & (pl.col("cell") == PRIMARY_CELL)
            & (pl.col("seed") == 42)
        )
        .select(
            [
                "policy",
                "success_rate",
                "total_gene_actions",
                "n_actions_replogle_only_essential",
                "frac_actions_replogle_only_essential",
                "frac_actions_replogle_essential",
                "frac_actions_depmap_essential",
                "enrichment_replogle_only_vs_random",
                "enrichment_replogle_essential_vs_random",
            ]
        )
        .sort("policy")
        .to_dicts()
    )

    track_l_primary = (
        df.filter(
            (pl.col("dynamics_field") == "track_l_n64_legacy")
            & (pl.col("cell") == "k2_bin8-10_splitood")
            & (pl.col("policy") == "PPO_BCD")
        )
        .select(["seed", "success_rate", "frac_actions_replogle_only_essential",
                 "n_actions_replogle_only_essential", "total_gene_actions"])
        .sort("seed")
        .to_dicts()
    )

    anchor_primary = (
        df.filter(pl.col("dynamics_field") == "v2_anchor")
        .select(["cell", "policy", "success_rate",
                 "frac_actions_replogle_only_essential",
                 "frac_actions_replogle_essential"])
        .sort(["cell", "policy"])
        .to_dicts()
    )

    payload = {
        "scope": "V3C final Replogle held-out action-overlap audit (Bucket-C post-hoc)",
        "champion": "contraction_aware_v2_aggressive + PPO_BCD seed42 500k (CHAMPION_TUNED_RESULT)",
        "primary_cell": PRIMARY_CELL,
        "gene_sets": gene_sets,
        "champion_4seed_aggregate_at_primary_cell": agg_4seed,
        "champion_field_primary_cell_policy_comparison_seed42": primary_v2agg,
        "track_l_4seed_at_k2_bin8-10_splitood": track_l_primary,
        "v2_anchor_seed42": anchor_primary,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    LOG.info("Wrote %s", OUT_JSON)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-cell-only", action="store_true",
                        help="Only audit the K=3/bin8-10/OOD primary cell.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    gene_sets = load_gene_sets()
    LOG.info("Replogle-only essentials (∩ Norman 105) [n=%d]: %s",
             len(gene_sets["replogle_only_in_norman"]), gene_sets["replogle_only_in_norman"])

    rows = collect_rows(gene_sets)
    if args.primary_cell_only:
        rows = [r for r in rows if r["cell"] == PRIMARY_CELL]

    if not rows:
        LOG.error("No rows produced — eval directories may be missing.")
        return 1

    write_outputs(rows, gene_sets)

    # Print a concise headline summary to stdout
    df = pl.DataFrame(rows)
    print("\n=== Champion-field primary-cell (K=3/bin8-10/OOD, seed42) ===")
    print(
        df.filter(
            (pl.col("dynamics_field") == "v2_aggressive")
            & (pl.col("cell") == PRIMARY_CELL)
            & (pl.col("seed") == 42)
        )
        .select(
            ["policy", "success_rate", "total_gene_actions",
             "n_actions_replogle_only_essential",
             "frac_actions_replogle_only_essential",
             "enrichment_replogle_only_vs_random"]
        )
        .sort("policy")
    )

    print("\n=== v2_aggressive PPO_BCD 4-seed mean at primary cell ===")
    agg = aggregate_v2_aggressive_4seed(df)
    if agg.get("status") == "no_data":
        print("  (no data)")
    else:
        print(f"  seeds: {agg['seeds']}")
        for r in agg["per_seed"]:
            print(
                f"   seed{r['seed']}:  success={r['success_rate']:.3f}  "
                f"frac_repl_only={r['frac_actions_replogle_only_essential']:.4f}  "
                f"(n_actions_repl_only={r['n_actions_replogle_only_essential']}/"
                f"{r['total_gene_actions']})  "
                f"enrichment×={r['enrichment_replogle_only_vs_random'] or 0:.2f}"
            )
        print(f"  4-seed mean frac_repl_only = {agg['frac_replogle_only_4seed_mean']:.4f}  "
              f"CI95 [{agg['frac_replogle_only_4seed_ci95'][0]:.4f}, "
              f"{agg['frac_replogle_only_4seed_ci95'][1]:.4f}]  "
              f"std={agg['frac_replogle_only_4seed_std']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
