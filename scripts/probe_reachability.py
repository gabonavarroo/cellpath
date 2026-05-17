"""P0C0 — multi-step beam-search reachability probe.

Tests whether any gene sequence of depth <= max_depth can bring OOD start
cells to distance < epsilon under a given dynamics model, without using the
RL environment (bypasses noop termination).

repeat_mask=True  (comparable to RL env): genes used in the current path
                   are excluded from subsequent steps.
repeat_mask=False (upper bound): any gene may be reused at every step.

Start pool is loaded via the same _load_start_pool() used in evaluate_rl_hard.py
to ensure identical distance-bin and held-out-gene semantics.

Outputs:
  probe_results.json   -- per-dynamics summary + per-start-cell best trajectory
  probe_summary.md     -- human-readable table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse start-pool loading from evaluate_rl_hard to guarantee identical semantics
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_rl_hard import _load_start_pool, _parse_bins  # noqa: E402


def _load_dynamics(dynamics_dir: Path) -> Any:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.analysis.gate_breakdown import load_dynamics_model
    return load_dynamics_model(dynamics_dir)


def beam_search(
    z_starts: np.ndarray,
    z_ref: np.ndarray,
    dynamics: Any,
    *,
    n_genes: int,
    max_depth: int,
    beam_width: int,
    epsilon: float,
    repeat_mask: bool,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run beam search from each start cell.

    gene_indices are 1-indexed to match the dynamics model convention
    (environment action a corresponds to gene_idx = a+1).
    Noop is never added to the beam.

    Returns one entry per start cell:
        {
            "start_distance": float,
            "best_final_distance": float,
            "best_gene_sequence": list[int],   # 1-indexed gene_idx
            "success": bool,
        }
    """
    all_gene_indices = np.arange(1, n_genes + 1, dtype=np.int64)  # 1-indexed
    results: list[dict[str, Any]] = []
    dynamics.eval()

    for z0 in z_starts:
        d_start = float(np.linalg.norm(z0 - z_ref))
        # Beam entry: (current_z, used_set_of_1idx, gene_sequence_list, distance)
        beam: list[tuple[np.ndarray, frozenset[int], list[int], float]] = [
            (z0, frozenset(), [], d_start)
        ]

        for _depth in range(max_depth):
            if not beam:
                break
            candidates: list[tuple[np.ndarray, frozenset[int], list[int], float]] = []

            for z_cur, used_set, seq, _ in beam:
                if repeat_mask:
                    avail = np.array(
                        [g for g in all_gene_indices if g not in used_set],
                        dtype=np.int64,
                    )
                else:
                    avail = all_gene_indices

                if len(avail) == 0:
                    continue

                z_batch = np.repeat(z_cur[None, :], len(avail), axis=0)
                with torch.no_grad():
                    z_t = torch.from_numpy(z_batch).float().to(device)
                    g_t = torch.from_numpy(avail).long().to(device)
                    out = dynamics(z_t, g_t)
                    z_next_all = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy()

                dists = np.linalg.norm(z_next_all - z_ref, axis=1)
                for g, z_next, d_next in zip(avail, z_next_all, dists, strict=True):
                    new_used = used_set | {int(g)} if repeat_mask else used_set
                    candidates.append((z_next, new_used, seq + [int(g)], float(d_next)))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[3])
            beam = candidates[:beam_width]

        if beam:
            _z_best, _used_best, seq_best, d_best = beam[0]
        else:
            seq_best, d_best = [], d_start

        results.append({
            "start_distance": float(d_start),
            "best_final_distance": float(d_best),
            "best_gene_sequence": seq_best,
            "success": bool(d_best < epsilon),
        })

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Beam-search reachability probe for dynamics models.")
    ap.add_argument("--dynamics_dirs", nargs="+", required=True,
                    help="One or more <label>:<path> pairs, e.g. soft_ot:artifacts_v2/dynamics_soft_ot_default")
    ap.add_argument("--vae_dir", required=True)
    ap.add_argument("--pairs_dir", default="artifacts/pairs",
                    help="Pairs dir for held-out gene metadata (default: V1 OT pairs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epsilon", type=float, default=3.166289806365967)
    ap.add_argument("--distance_bin", default="8-10",
                    help="Single bin label, e.g. '8-10'")
    ap.add_argument("--held_out_genes_only", action="store_true", default=True)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--beam_width", type=int, default=50)
    ap.add_argument("--n_genes", type=int, default=105)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae_dir = Path(args.vae_dir)
    pairs_dir = Path(args.pairs_dir)

    with open(pairs_dir / "metadata.json") as f:
        pairs_meta = json.load(f)
    held_out_genes = [str(g) for g in pairs_meta.get("held_out_genes", [])]

    # Parse bin using the same helper as evaluate_rl_hard
    bins = list(_parse_bins([args.distance_bin]))
    if len(bins) != 1:
        raise ValueError(f"Expected a single bin, got {bins}")
    distance_bin = bins[0]

    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)
    z_starts = _load_start_pool(
        vae_dir,
        distance_bin=distance_bin,
        held_out_genes_only=args.held_out_genes_only,
        held_out_genes=held_out_genes,
    )
    print(f"Start pool: {len(z_starts)} cells in bin {args.distance_bin} "
          f"(held_out={args.held_out_genes_only})")

    all_results: dict[str, Any] = {
        "epsilon": args.epsilon,
        "distance_bin": args.distance_bin,
        "held_out_genes_only": args.held_out_genes_only,
        "max_depth": args.max_depth,
        "beam_width": args.beam_width,
        "n_start_cells": int(len(z_starts)),
        "dynamics_runs": {},
    }

    for spec in args.dynamics_dirs:
        label, dyn_path = spec.split(":", 1)
        dynamics = _load_dynamics(Path(dyn_path))
        dynamics.to(args.device)
        for repeat_mask in (True, False):
            run_key = f"{label}_repeat{'on' if repeat_mask else 'off'}"
            print(f"\nBeam search: {run_key} ({dyn_path}), repeat_mask={repeat_mask}")
            per_cell = beam_search(
                z_starts, z_ref, dynamics,
                n_genes=args.n_genes,
                max_depth=args.max_depth,
                beam_width=args.beam_width,
                epsilon=args.epsilon,
                repeat_mask=repeat_mask,
                device=args.device,
            )
            n_success = sum(r["success"] for r in per_cell)
            best_dist = min(r["best_final_distance"] for r in per_cell) if per_cell else float("nan")
            mean_best = float(np.mean([r["best_final_distance"] for r in per_cell])) if per_cell else float("nan")
            print(f"  n_success={n_success}/{len(per_cell)} "
                  f"best_dist={best_dist:.4f} mean_best={mean_best:.4f}")
            all_results["dynamics_runs"][run_key] = {
                "dynamics_dir": dyn_path,
                "repeat_mask": repeat_mask,
                "n_successes": int(n_success),
                "success_rate": float(n_success / len(per_cell)) if per_cell else 0.0,
                "best_final_distance": float(best_dist),
                "mean_best_final_distance": float(mean_best),
                "per_cell": per_cell,
            }
        dynamics.to("cpu")

    (out_dir / "probe_results.json").write_text(json.dumps(all_results, indent=2))

    lines = ["# Reachability probe summary\n\n"]
    lines.append(
        f"epsilon_p25={args.epsilon:.4f}, depth={args.max_depth}, "
        f"beam={args.beam_width}, n_starts={len(z_starts)}\n\n"
    )
    lines.append("| run_key | repeat_mask | n_success | success_rate | best_final_distance | mean_best |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for run_key, r in all_results["dynamics_runs"].items():
        lines.append(
            f"| {run_key} | {r['repeat_mask']} | {r['n_successes']} | "
            f"{r['success_rate']:.3f} | {r['best_final_distance']:.4f} | "
            f"{r['mean_best_final_distance']:.4f} |\n"
        )
    (out_dir / "probe_summary.md").write_text("".join(lines))
    print(f"\nResults written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
