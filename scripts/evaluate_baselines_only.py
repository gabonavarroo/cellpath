"""Baselines-only hard-bench evaluator (no PPO required).

V3A diagnostic: gets ``greedy_dyn_{1,2,3}``, ``random``, ``always_noop``
success rates at K-sweep × bin × OOD cells without needing a PPO checkpoint.
Mirrors the start-pool / env construction / metric computation in
``scripts/evaluate_rl_hard.py`` so V3 numbers are directly comparable to V2.

Usage:
    python scripts/evaluate_baselines_only.py \
        --vae_dir artifacts_v3/vae_n64_legacy \
        --dynamics_dir artifacts_v3/dynamics_n64_legacy_ror_corr010 \
        --pairs_dir artifacts_v3/pairs_n64_legacy \
        --out_dir artifacts_v3/eval_v3a_hardness/legacy_baselines_seed42 \
        --k_values 2 3 --distance_bins 6-8 8-10 \
        --epsilon 3.187233090400696 --n_episodes 300 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from scripts.evaluate_rl_hard import (
    _load_start_pool,
    _make_env,
    _parse_bins,
    run_policy_episodes,
    wilson_ci,
)
from src.analysis.gate_breakdown import load_dynamics_model
from src.rl.baselines import (
    AlwaysNoopPolicy,
    GreedyDynamicsBeamPolicy,
    GreedyDynamicsPolicy,
    NoopFreeGreedyPolicy,
    RandomUniformValidPolicy,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_dir", required=True)
    parser.add_argument("--dynamics_dir", required=True)
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k_values", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--distance_bins", nargs="+", default=["6-8", "8-10"])
    parser.add_argument("--epsilon", type=float, required=True,
                        help="Absolute epsilon (V3A: p25 in 64D, computed per track)")
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held_out_genes_only", action="store_true", default=True)
    parser.add_argument(
        "--baselines",
        default="random,always_noop,greedy_dyn_1,greedy_dyn_1_noop_free,greedy_dyn_2,greedy_dyn_3",
        help="comma-separated baseline names",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vae_dir = Path(args.vae_dir)
    pairs_dir = Path(args.pairs_dir)

    with open(vae_dir / "gene_vocab.json") as f:
        vocab = json.load(f)
    genes = [str(g) for g in vocab["genes"]]
    n_genes = int(vocab["n_genes"])
    noop_idx = int(vocab["noop_idx"])
    gene_lookup = {i: g for i, g in enumerate(genes)}
    gene_lookup[noop_idx] = "NO_OP"
    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)

    with open(pairs_dir / "metadata.json") as f:
        pairs_meta = json.load(f)
    held_out_genes = [str(g) for g in pairs_meta.get("held_out_genes", [])]

    dynamics_model = load_dynamics_model(args.dynamics_dir)
    baseline_names = set(name.strip() for name in args.baselines.split(",") if name.strip())

    metadata = {
        "stage": "v3a_baselines_only",
        "source_paths": {
            "vae_dir": str(args.vae_dir),
            "dynamics_dir": str(args.dynamics_dir),
            "pairs_dir": str(args.pairs_dir),
        },
        "matrix": {
            "k_values": args.k_values,
            "distance_bins": args.distance_bins,
            "epsilon": float(args.epsilon),
            "held_out_genes_only": bool(args.held_out_genes_only),
            "n_episodes": int(args.n_episodes),
            "baselines": sorted(baseline_names),
            "seed": int(args.seed),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    rows: list[dict] = []

    for k in args.k_values:
        for distance_bin in _parse_bins(args.distance_bins):
            cell = f"k{k}_bin{distance_bin[2]}_split{'ood' if args.held_out_genes_only else 'mixed'}"
            cell_dir = out_dir / cell
            cell_dir.mkdir(parents=True, exist_ok=True)
            cell_meta = {
                "k": int(k),
                "epsilon": float(args.epsilon),
                "distance_bin": distance_bin[2],
                "gene_split": "ood" if args.held_out_genes_only else "mixed",
            }
            try:
                start_pool = _load_start_pool(
                    vae_dir,
                    distance_bin=distance_bin,
                    held_out_genes_only=bool(args.held_out_genes_only),
                    held_out_genes=held_out_genes,
                )
            except ValueError as exc:
                print(f"{cell}: empty start pool ({exc})", file=sys.stderr)
                continue

            policies: dict = {}
            if "random" in baseline_names:
                policies["random"] = RandomUniformValidPolicy(seed=args.seed)
            if "always_noop" in baseline_names:
                policies["always_noop"] = AlwaysNoopPolicy(noop_idx)
            if "greedy_dyn_1" in baseline_names:
                policies["greedy_dyn_1"] = GreedyDynamicsPolicy(
                    dynamics_model, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx,
                )
            if "greedy_dyn_1_noop_free" in baseline_names:
                policies["greedy_dyn_1_noop_free"] = NoopFreeGreedyPolicy(
                    dynamics_model, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx,
                )
            if "greedy_dyn_2" in baseline_names:
                policies["greedy_dyn_2"] = GreedyDynamicsBeamPolicy(
                    dynamics_model, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx,
                    depth=2, beam_width=20,
                )
            if "greedy_dyn_3" in baseline_names:
                policies["greedy_dyn_3"] = GreedyDynamicsBeamPolicy(
                    dynamics_model, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx,
                    depth=3, beam_width=20,
                )

            for policy_name, policy in policies.items():
                t0 = time.time()
                env = _make_env(
                    dynamics_model=dynamics_model,
                    z_ref=z_ref,
                    epsilon=args.epsilon,
                    n_genes=n_genes,
                    max_steps=k,
                    start_pool=start_pool,
                    seed=args.seed,
                )
                summary = run_policy_episodes(
                    env,
                    policy,
                    n_episodes=int(args.n_episodes),
                    gene_lookup=gene_lookup,
                )
                n = int(summary.get("n_episodes", args.n_episodes))
                succ_rate = float(summary.get("success_rate", 0.0))
                successes = int(round(succ_rate * n))
                ci_lo, ci_hi = wilson_ci(successes, n)
                summary.update({
                    "policy": policy_name,
                    **cell_meta,
                    "n_start_pool": int(len(start_pool)),
                    "wilson_lo": ci_lo,
                    "wilson_hi": ci_hi,
                    "wall_seconds": time.time() - t0,
                })
                policy_dir = cell_dir / policy_name
                policy_dir.mkdir(parents=True, exist_ok=True)
                (policy_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, default=str)
                )
                rows.append({
                    "cell": cell,
                    "policy": policy_name,
                    "k": int(k),
                    "bin": distance_bin[2],
                    "success_rate": succ_rate,
                    "wilson_95": [ci_lo, ci_hi],
                    "mean_steps": float(summary.get("mean_steps", -1.0)),
                    "mean_final_distance": float(summary.get("mean_final_distance", -1.0)),
                    "n_episodes": n,
                    "n_start_pool": int(len(start_pool)),
                })
                print(
                    f"{cell} / {policy_name}: succ={succ_rate:.3f} "
                    f"CI=[{ci_lo:.3f},{ci_hi:.3f}] "
                    f"n_pool={len(start_pool)} ({time.time()-t0:.1f}s)"
                )

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str))

    # Markdown summary table
    lines = ["# V3A baselines-only summary", "",
             f"epsilon={args.epsilon}, n_episodes={args.n_episodes}, seed={args.seed}", "",
             "| cell | policy | success | Wilson 95 | mean_steps | mean_final_d |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = r["wilson_95"]
        lines.append(
            f"| {r['cell']} | {r['policy']} | {r['success_rate']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {r['mean_steps']:.2f} | {r['mean_final_distance']:.3f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
