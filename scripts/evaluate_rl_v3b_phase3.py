"""V3B Phase 3 evaluator — path-length free-band reward (Variant B).

Evaluates PPO_B against PPO_A (V2 primary frozen baseline), random, noop, and
reward-aware multi-step greedy oracles (greedy_dyn_{1,2,3,5,8}_B) across the
Phase 3 hardness matrix:

  * K ∈ {3, 4, 5, 8} × bin 8-10 / OOD
  * K ∈ {4, 5}    × bin 6-8 / OOD

All policies are evaluated **under the freeband reward** with `env.max_steps`
set to the cell's K, so the comparison is reward-faithful per the Phase 2b
"greedy must be reward-aware" lesson.

Outputs (per --out_dir):
* `<cell>/<policy>/summary.json` — per-policy aggregate (also includes the per-episode
  step-count histogram so Phase 3's "K ∈ {4, 5} usage" criterion is computable).
* `aggregate.parquet` / `aggregate.csv` — long-form table.
* `phase3_summary.md` — human-readable per-cell × per-policy table.

This script does NOT aggregate 4-seed CIs; that's `scripts/aggregate_v3b_phase3.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


LOG = logging.getLogger("evaluate_rl_v3b_phase3")


# ---------------------------------------------------------------------------
# Cell matrix
# ---------------------------------------------------------------------------


CELL_DEFS = {
    # K=2 cells — V2 hardness-frontier informative cells (added in Phase 3b for stricter-ε diagnostic)
    "k2_bin6-8_splitood":  dict(k=2, bin=(6.0, 8.0, "6-8"),  held_out=True),
    "k2_bin8-10_splitood": dict(k=2, bin=(8.0, 10.0, "8-10"), held_out=True),
    # K=3 saturated control (same as Phase 2; expect saturation at p25)
    "k3_bin8-10_splitood": dict(k=3, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k3_bin6-8_splitood":  dict(k=3, bin=(6.0, 8.0, "6-8"),  held_out=True),
    # K=4 / K=5 cells — the freeband leverage band
    "k4_bin8-10_splitood": dict(k=4, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k5_bin8-10_splitood": dict(k=5, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k4_bin6-8_splitood":  dict(k=4, bin=(6.0, 8.0, "6-8"),  held_out=True),
    "k5_bin6-8_splitood":  dict(k=5, bin=(6.0, 8.0, "6-8"),  held_out=True),
    # K=8 speculative cell (heavy penalty band)
    "k8_bin8-10_splitood": dict(k=8, bin=(8.0, 10.0, "8-10"), held_out=True),

    # ---- Phase 3 legacy aliases (p25-tagged) for back-compat with the original eval ----
    "k3_epsp25_bin8-10_splitood": dict(k=3, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k4_epsp25_bin8-10_splitood": dict(k=4, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k5_epsp25_bin8-10_splitood": dict(k=5, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k4_epsp25_bin6-8_splitood":  dict(k=4, bin=(6.0, 8.0, "6-8"),  held_out=True),
    "k5_epsp25_bin6-8_splitood":  dict(k=5, bin=(6.0, 8.0, "6-8"),  held_out=True),
    "k8_epsp25_bin8-10_splitood": dict(k=8, bin=(8.0, 10.0, "8-10"), held_out=True),
}

# Default Phase 3 cell list (p25-tagged) when --cells is not given.
PHASE3_DEFAULT_CELLS = [
    "k3_epsp25_bin8-10_splitood",
    "k4_epsp25_bin8-10_splitood",
    "k5_epsp25_bin8-10_splitood",
    "k4_epsp25_bin6-8_splitood",
    "k5_epsp25_bin6-8_splitood",
    "k8_epsp25_bin8-10_splitood",
]

# Phase 3b cells (epsilon-agnostic names, paired with --epsilon_value)
PHASE3B_DEFAULT_CELLS = [
    "k2_bin6-8_splitood",
    "k2_bin8-10_splitood",
    "k3_bin6-8_splitood",
    "k3_bin8-10_splitood",
    "k4_bin6-8_splitood",
    "k4_bin8-10_splitood",
    "k5_bin6-8_splitood",
    "k5_bin8-10_splitood",
    "k8_bin8-10_splitood",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _make_freeband_env(
    *, dynamics_model, z_ref, epsilon, n_genes, max_steps, start_pool, seed,
    freeband: dict[str, float],
):
    """Construct an env in path_length_freeband mode, no safety layer."""
    from src.rl.environment import CellReprogrammingEnv
    return CellReprogrammingEnv(
        dynamics_model=dynamics_model,
        z_reference_centroid=z_ref,
        epsilon_success=float(epsilon),
        n_genes=int(n_genes),
        max_steps=int(max_steps),
        lambda_sparse=0.0,
        lambda_unc=0.0,
        repeat_mask=True,
        start_state_strategy="random_perturbation",
        distance_metric="l2",
        start_pool_latents=start_pool,
        success_bonus=0.0,
        failure_penalty=0.0,
        seed=seed,
        reward_mode="path_length_freeband",
        beta_step_cost=0.05,
        free_steps=int(freeband["free_steps"]),
        mild_until=int(freeband["mild_until"]),
        mild_beta=float(freeband["mild_beta"]),
        heavy_beta=float(freeband["heavy_beta"]),
        freeband_success_bonus=float(freeband["success_bonus"]),
    )


def _load_start_pool(vae_dir: Path, *, distance_bin, held_out_genes: list[str]):
    import anndata as ad
    lo, hi, _label = distance_bin
    adata = ad.read_h5ad(vae_dir / "latents.h5ad")
    z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)
    pert_idx = np.asarray(adata.obs["perturbation_idx"].values)
    mask = pert_idx != 0
    pert = np.asarray(adata.obs["perturbation"].astype(str).values)
    mask &= np.isin(pert, np.asarray(held_out_genes, dtype=object))
    d = np.linalg.norm(z - z_ref, axis=1)
    mask &= (d >= lo) & (d < hi)
    return z[mask].astype(np.float32)


def _run_policy(env, policy, *, n_episodes: int, gene_lookup: dict[int, str]) -> dict[str, Any]:
    """Per-policy aggregate; captures path-length distribution per episode."""
    successes = 0
    steps_list = []
    final_distances = []
    rewards = []
    action_freq: dict[str, int] = {}
    success_steps: list[int] = []     # path lengths of successful episodes only
    success_in_freeband: int = 0      # success episodes with T ≤ 3
    success_in_mild: int = 0          # success episodes with T ∈ {4, 5}
    success_in_heavy: int = 0         # success episodes with T > 5

    for ep in range(int(n_episodes)):
        obs, info = env.reset(seed=ep)
        terminated, truncated = False, False
        ep_reward = 0.0
        terminal_success = False
        last_info = info
        while not (terminated or truncated):
            action = int(policy.select_action(obs, info["action_mask"], info))
            sym = gene_lookup.get(action, f"gene_{action}")
            action_freq[sym] = action_freq.get(sym, 0) + 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            terminal_success = bool(info.get("success", False))
            last_info = info

        is_success = bool(terminal_success and terminated)
        successes += int(is_success)
        T_final = int(last_info.get("step", 0))
        steps_list.append(T_final)
        final_distances.append(float(last_info.get("distance", np.nan)))
        rewards.append(ep_reward)
        if is_success:
            success_steps.append(T_final)
            if T_final <= 3:
                success_in_freeband += 1
            elif T_final <= 5:
                success_in_mild += 1
            else:
                success_in_heavy += 1

    n = int(n_episodes)
    wlo, whi = _wilson_ci(successes, n)
    n_success = max(1, len(success_steps))
    return {
        "n_episodes": n,
        "successes": int(successes),
        "success_rate": float(successes / max(n, 1)),
        "success_rate_wilson95_low": wlo,
        "success_rate_wilson95_high": whi,
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list, ddof=1)) if n > 1 else 0.0,
        "mean_final_distance": float(np.nanmean(final_distances)),
        "mean_total_reward": float(np.mean(rewards)),
        # Path-length histogram across all episodes (success + fail/truncated)
        "step_distribution": _step_hist(steps_list),
        # Among successful episodes only:
        "n_successful_episodes": int(len(success_steps)),
        "success_step_distribution": _step_hist(success_steps) if success_steps else {},
        "frac_success_in_freeband_T_le_3": float(success_in_freeband / n_success),
        "frac_success_in_mild_T_4_or_5": float(success_in_mild / n_success),
        "frac_success_in_heavy_T_gt_5": float(success_in_heavy / n_success),
        "action_freq": action_freq,
        "top_actions": sorted(
            ({"gene_symbol": g, "count": int(c)} for g, c in action_freq.items()),
            key=lambda x: -x["count"],
        )[:10],
    }


def _step_hist(steps: list[int]) -> dict[str, int]:
    h: dict[str, int] = {}
    for s in steps:
        k = str(int(s))
        h[k] = h.get(k, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae_dir", default="artifacts/vae")
    parser.add_argument("--dynamics_dir", default="artifacts_v2/dynamics_v1ot_ror_corr010")
    parser.add_argument("--pairs_dir", default="artifacts/pairs")
    parser.add_argument("--ppo_zip_B", required=True,
                        help="Path to PPO_B (path-length freeband) checkpoint .zip")
    parser.add_argument(
        "--ppo_zip_A",
        default="artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed42/ppo.zip",
        help="Path to V2 primary PPO (frozen baseline).",
    )
    parser.add_argument(
        "--ppo_zip_C", default=None,
        help="Optional: V3B Phase 2 safety-aware PPO_C checkpoint (.zip).",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_episodes", type=int, default=300)
    # Freeband schedule (defaults match config/rl.yaml::reward.freeband)
    parser.add_argument("--free_steps", type=int, default=3)
    parser.add_argument("--mild_until", type=int, default=5)
    parser.add_argument("--mild_beta", type=float, default=0.02)
    parser.add_argument("--heavy_beta", type=float, default=0.10)
    parser.add_argument("--success_bonus", type=float, default=1.0)
    parser.add_argument(
        "--epsilon_value", type=float, default=3.1662898064,
        help="Success-distance threshold ε. Default: V2 p25=3.1663. Set "
             "to p15=2.9898 / p10=2.8846 / p5=2.7362 for Phase 3b stricter-ε diagnostic.",
    )
    parser.add_argument(
        "--max_greedy_depth", type=int, default=5,
        help="Cap on greedy beam depth — greedy_dyn_K is only run for K ≤ min(this, cell.K). "
             "Default 5; set to 8 to include greedy_dyn_8 at K=8 cells.",
    )
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Cells to evaluate. Default: Phase 3 legacy (--epsilon_value p25) "
                             "or Phase 3b matrix (--epsilon_value < p25).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_distance_greedy", action="store_true",
                        help="Also run V2-style distance-only greedy as a sanity contrast.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.analysis.gate_breakdown import load_dynamics_model
    from src.rl.baselines import (
        AlwaysNoopPolicy,
        GreedyDynamicsBeamPolicy,
        RandomUniformValidPolicy,
    )

    vae_dir = (repo_root / args.vae_dir) if not Path(args.vae_dir).is_absolute() else Path(args.vae_dir)
    pairs_dir = (repo_root / args.pairs_dir) if not Path(args.pairs_dir).is_absolute() else Path(args.pairs_dir)
    dyn_dir = (repo_root / args.dynamics_dir) if not Path(args.dynamics_dir).is_absolute() else Path(args.dynamics_dir)
    out_dir = (repo_root / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _abs(p): return (repo_root / p).resolve() if not Path(p).is_absolute() else Path(p)
    ppo_B = _abs(args.ppo_zip_B)
    ppo_A = _abs(args.ppo_zip_A)
    ppo_C = _abs(args.ppo_zip_C) if args.ppo_zip_C else None
    if not ppo_B.exists():
        raise FileNotFoundError(f"--ppo_zip_B does not exist: {ppo_B}")

    # Default cell list: pick Phase 3 (p25-tagged names) when epsilon ≈ p25, else Phase 3b.
    if args.cells is None:
        if abs(float(args.epsilon_value) - 3.1662898064) < 1e-6:
            cells_to_run = list(PHASE3_DEFAULT_CELLS)
        else:
            cells_to_run = list(PHASE3B_DEFAULT_CELLS)
    else:
        cells_to_run = list(args.cells)

    # Load assets
    with open(vae_dir / "gene_vocab.json") as f:
        vocab = json.load(f)
    genes = [str(g) for g in vocab["genes"]]
    n_genes = int(vocab["n_genes"])
    noop_idx = int(vocab["noop_idx"])
    gene_lookup = {i: g for i, g in enumerate(genes)}
    gene_lookup[noop_idx] = "NO_OP"
    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)

    # Phase 3b: epsilon is now a CLI argument. Default is V2 p25 = 3.1663 for back-compat.
    EPSILON_P25 = float(args.epsilon_value)
    LOG.info("epsilon = %.6f", EPSILON_P25)

    with open(pairs_dir / "metadata.json") as f:
        pairs_meta = json.load(f)
    held_out_genes = [str(g) for g in pairs_meta.get("held_out_genes", [])]

    # Dynamics
    dyn = load_dynamics_model(dyn_dir)

    from sb3_contrib import MaskablePPO

    class _SB3Policy:
        def __init__(self, model, *, deterministic=True, name="ppo"):
            self.model = model
            self.deterministic = bool(deterministic)
            self.name = name
        def select_action(self, z, mask, info):
            a, _ = self.model.predict(z, deterministic=self.deterministic, action_masks=mask)
            return int(np.asarray(a).item())

    # Build policy roster.
    # Greedy baselines are REWARD-AWARE under freeband (see GreedyDynamicsBeamPolicy
    # docstring): they score plans by `path_penalty(T) - success_bonus·1[d<ε] + d`,
    # making them fair comparators for PPO_B under the same reward objective.
    freeband_schedule = {
        "free_steps":   args.free_steps,
        "mild_until":   args.mild_until,
        "mild_beta":    args.mild_beta,
        "heavy_beta":   args.heavy_beta,
        "success_bonus": args.success_bonus,
    }
    LOG.info("Freeband schedule: %s", freeband_schedule)
    LOG.info("Greedy oracles are REWARD-AWARE (path_length_freeband objective).")

    def _make_greedy(depth: int) -> Any:
        return GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx,
            depth=depth, beam_width=20,
            freeband_schedule=freeband_schedule,
            success_epsilon=EPSILON_P25,
        )

    distance_greedies: dict[str, Any] = {}
    if args.include_distance_greedy:
        # Sanity contrast: V2-style distance-only greedy, NOT reward-aware.
        # Use a different name so the report distinguishes them clearly.
        for d in (2, 5):
            distance_greedies[f"greedy_dyn_{d}_distance_only"] = GreedyDynamicsBeamPolicy(
                dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx, depth=d, beam_width=20,
            )

    policies_base: dict[str, Any] = {
        "ppo_B":               _SB3Policy(MaskablePPO.load(str(ppo_B), device="cpu"), name="ppo_B"),
        "ppo_A":               _SB3Policy(MaskablePPO.load(str(ppo_A), device="cpu"), name="ppo_A") if ppo_A.exists() else None,
        "ppo_C":               (_SB3Policy(MaskablePPO.load(str(ppo_C), device="cpu"), name="ppo_C")
                                if (ppo_C is not None and ppo_C.exists()) else None),
        "random_uniform_valid": RandomUniformValidPolicy(seed=args.seed),
        "always_noop":         AlwaysNoopPolicy(noop_idx),
        "greedy_dyn_1_B":      _make_greedy(1),
        "greedy_dyn_2_B":      _make_greedy(2),
        "greedy_dyn_3_B":      _make_greedy(3),
        "greedy_dyn_5_B":      _make_greedy(5),
        "greedy_dyn_8_B":      _make_greedy(8),
        **distance_greedies,
    }
    policies = {k: v for k, v in policies_base.items() if v is not None}
    LOG.info("Policies to evaluate: %s", list(policies))
    LOG.info("Greedy depth cap = %d (only depths ≤ this AND ≤ cell.K are evaluated per cell).",
             args.max_greedy_depth)

    rows: list[dict[str, Any]] = []

    def _is_relevant_policy(pol_name: str, cell_k: int) -> bool:
        """Cap greedy depth: skip greedy_dyn_K policies where K > min(cap, cell_k).

        Reduces compute for shallow cells (e.g., K=2 cells don't need greedy_dyn_8).
        Behavior preserved: greedy_dyn_8 still runs at the K=8 cell when --max_greedy_depth=8.
        """
        if not pol_name.startswith("greedy_dyn_"):
            return True
        # parse depth from name "greedy_dyn_<D>_B" or "greedy_dyn_<D>_distance_only"
        try:
            d = int(pol_name.split("_")[2])
        except (IndexError, ValueError):
            return True
        return d <= min(args.max_greedy_depth, cell_k)

    for cell_name in cells_to_run:
        if cell_name not in CELL_DEFS:
            LOG.warning("Unknown cell %s — skipping", cell_name); continue
        defn = CELL_DEFS[cell_name]
        k_cell = int(defn["k"])
        bin_def = defn["bin"]
        try:
            start_pool = _load_start_pool(
                vae_dir, distance_bin=bin_def, held_out_genes=held_out_genes,
            )
        except ValueError as exc:
            LOG.warning("Cell %s: empty start pool (%s) — skipping", cell_name, exc); continue
        LOG.info(
            "Cell %s: K=%d, bin=%s, OOD=%s, |pool|=%d",
            cell_name, k_cell, bin_def[2], defn["held_out"], len(start_pool),
        )

        for pol_name, policy in policies.items():
            if not _is_relevant_policy(pol_name, k_cell):
                LOG.debug("  %-30s skipped (depth > cap or > cell.K)", pol_name)
                continue
            t0 = time.time()
            env = _make_freeband_env(
                dynamics_model=dyn, z_ref=z_ref, epsilon=EPSILON_P25,
                n_genes=n_genes, max_steps=k_cell, start_pool=start_pool, seed=args.seed,
                freeband=freeband_schedule,
            )
            result = _run_policy(env, policy, n_episodes=int(args.n_episodes), gene_lookup=gene_lookup)
            result.update({
                "cell": cell_name, "policy": pol_name,
                "k": k_cell, "epsilon": float(EPSILON_P25),
                "distance_bin": bin_def[2], "gene_split": "ood",
                "n_start_pool": int(len(start_pool)),
                "elapsed_sec": time.time() - t0,
            })
            cell_dir = out_dir / cell_name / pol_name
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))

            rows.append({
                "cell": cell_name, "policy": pol_name,
                "k": k_cell, "n_episodes": result["n_episodes"],
                "success_rate": result["success_rate"],
                "mean_steps": result["mean_steps"],
                "std_steps": result["std_steps"],
                "mean_final_distance": result["mean_final_distance"],
                "mean_total_reward": result["mean_total_reward"],
                "frac_success_T_le_3": result["frac_success_in_freeband_T_le_3"],
                "frac_success_T_4_or_5": result["frac_success_in_mild_T_4_or_5"],
                "frac_success_T_gt_5": result["frac_success_in_heavy_T_gt_5"],
                "n_successful_episodes": result["n_successful_episodes"],
            })
            LOG.info(
                "  %-30s success=%.3f mean_steps=%.2f frac_T={1-3:%.2f,4-5:%.2f,>5:%.2f}  (%.1fs)",
                pol_name, result["success_rate"], result["mean_steps"],
                result["frac_success_in_freeband_T_le_3"],
                result["frac_success_in_mild_T_4_or_5"],
                result["frac_success_in_heavy_T_gt_5"],
                result["elapsed_sec"],
            )

    df = pl.DataFrame(rows)
    df.write_parquet(str(out_dir / "aggregate.parquet"))
    df.write_csv(str(out_dir / "aggregate.csv"))

    # Markdown summary
    md = ["# V3B Phase 3 — Path-length free-band per-cell evaluation", "",
          f"> dynamics: {args.dynamics_dir}",
          f"> reward: path_length_freeband  (free_steps={args.free_steps}, mild_until={args.mild_until}, "
          f"mild_beta={args.mild_beta}, heavy_beta={args.heavy_beta}, success_bonus={args.success_bonus})",
          f"> n_episodes={args.n_episodes}, seed={args.seed}, epsilon={EPSILON_P25:.4f}",
          "> Greedy oracles are reward-aware (path_length_freeband objective).", "",
          "## Per-cell summary",
          "",
          "| cell | policy | success | mean_steps | frac_succ_T≤3 | frac_succ_T∈{4,5} | frac_succ_T>5 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for cell in cells_to_run:
        for pol_name in policies:
            sub = df.filter((pl.col("cell") == cell) & (pl.col("policy") == pol_name))
            if sub.is_empty(): continue
            r = sub.row(0, named=True)
            md.append(
                f"| {cell} | {pol_name} | {r['success_rate']:.3f} | {r['mean_steps']:.2f} | "
                f"{r['frac_success_T_le_3']:.3f} | {r['frac_success_T_4_or_5']:.3f} | "
                f"{r['frac_success_T_gt_5']:.3f} |"
            )
    (out_dir / "phase3_summary.md").write_text("\n".join(md) + "\n")
    LOG.info("Wrote %s", out_dir / "phase3_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
