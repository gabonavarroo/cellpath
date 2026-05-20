"""V3B Phase 4 — Reward-stack evaluator (B+C / D / B+C+D + variants).

Evaluates a roster of PPOs across the V3B hardness matrix under the BIOREALISTIC_FUSED
reward env (so all Bucket-A metrics — tox_path, common_essential_count, unc_path_max —
are populated consistently and comparably across all policies). Raw success rate is
the Bucket-B comparator and is invariant to the env's reward shape.

Greedy oracles are reward-aware under the full fused objective when the relevant λ are
nonzero. When λ=0, behavior collapses to the appropriate sub-mode.

Per (PPO label, cell):
* success_rate, mean_steps, mean_final_distance     (Bucket B)
* mean_tox_path, mean_common_essential_per_ep       (Bucket A)
* mean_unc_path_max, mean_unc_path_mean             (Bucket A)
* path-length histogram                             (B descriptor)
* action_freq                                       (descriptor)

Output: per-cell × per-policy summary.json + aggregate.{parquet,csv}.

This script does NOT do 4-seed aggregation; that's `scripts/aggregate_v3b_phase4.py`.
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


LOG = logging.getLogger("evaluate_rl_v3b_phase4")


CELL_DEFS = {
    "k2_bin6-8_splitood":  dict(k=2, bin=(6.0, 8.0, "6-8"),  held_out=True),
    "k2_bin8-10_splitood": dict(k=2, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k3_bin6-8_splitood":  dict(k=3, bin=(6.0, 8.0, "6-8"),  held_out=True),
    "k3_bin8-10_splitood": dict(k=3, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k4_bin6-8_splitood":  dict(k=4, bin=(6.0, 8.0, "6-8"),  held_out=True),
    "k4_bin8-10_splitood": dict(k=4, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k5_bin6-8_splitood":  dict(k=5, bin=(6.0, 8.0, "6-8"),  held_out=True),
    "k5_bin8-10_splitood": dict(k=5, bin=(8.0, 10.0, "8-10"), held_out=True),
    "k8_bin8-10_splitood": dict(k=8, bin=(8.0, 10.0, "8-10"), held_out=True),
}


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _make_fused_env(
    *, dynamics_model, z_ref, epsilon, n_genes, max_steps, start_pool, seed,
    safety_tox_arr, safety_ess_arr,
    freeband: dict[str, float], lambda_tox: float, lambda_ce: float, lambda_unc_path: float,
    uncertainty_clip_min: float, uncertainty_clip_max: float,
):
    from src.rl.environment import CellReprogrammingEnv
    return CellReprogrammingEnv(
        dynamics_model=dynamics_model,
        z_reference_centroid=z_ref,
        epsilon_success=float(epsilon),
        n_genes=int(n_genes),
        max_steps=int(max_steps),
        lambda_sparse=0.0,
        lambda_unc=0.0,                          # V1 path inactive (we use Phase-4 lambda_unc_path)
        repeat_mask=True,
        start_state_strategy="random_perturbation",
        distance_metric="l2",
        start_pool_latents=start_pool,
        success_bonus=0.0,
        failure_penalty=0.0,
        seed=seed,
        reward_mode="biorealistic_fused",        # env tracks all A-metrics under fused mode
        beta_step_cost=0.05,
        safety_tox_per_action=safety_tox_arr,
        safety_essential_per_action=safety_ess_arr,
        lambda_tox=float(lambda_tox),
        lambda_ce=float(lambda_ce),
        free_steps=int(freeband["free_steps"]),
        mild_until=int(freeband["mild_until"]),
        mild_beta=float(freeband["mild_beta"]),
        heavy_beta=float(freeband["heavy_beta"]),
        freeband_success_bonus=float(freeband["success_bonus"]),
        lambda_unc_path=float(lambda_unc_path),
        uncertainty_reduce="mean_sigma",
        uncertainty_clip_min=float(uncertainty_clip_min),
        uncertainty_clip_max=float(uncertainty_clip_max),
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


def _step_hist(steps: list[int]) -> dict[str, int]:
    h: dict[str, int] = {}
    for s in steps:
        k = str(int(s))
        h[k] = h.get(k, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


def _run_policy(env, policy, *, n_episodes: int, gene_lookup: dict[int, str]) -> dict[str, Any]:
    successes = 0
    steps_list, final_distances, rewards = [], [], []
    tox_paths, ce_counts, unc_maxes, unc_means = [], [], [], []
    success_steps: list[int] = []
    action_freq: dict[str, int] = {}

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
        tox_paths.append(float(last_info.get("tox_path", 0.0)))
        ce_counts.append(int(last_info.get("common_essential_count", 0)))
        unc_maxes.append(float(last_info.get("unc_path_max", 0.0)))
        unc_means.append(float(last_info.get("unc_path_mean", 0.0)))
        if is_success:
            success_steps.append(T_final)

    n = int(n_episodes)
    wlo, whi = _wilson_ci(successes, n)

    def _frac_band(seq: list[int], lo: int, hi: int) -> float:
        if not seq:
            return 0.0
        return float(np.mean([1.0 if (lo <= s <= hi) else 0.0 for s in seq]))

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
        "step_distribution_all": _step_hist(steps_list),
        "success_step_distribution": _step_hist(success_steps),
        "n_successful_episodes": int(len(success_steps)),
        "frac_success_T_le_3": _frac_band(success_steps, 0, 3),
        "frac_success_T_4_or_5": _frac_band(success_steps, 4, 5),
        "frac_success_T_gt_5": _frac_band(success_steps, 6, 999),
        # Bucket-A reward-fit metrics from env info
        "mean_tox_path": float(np.mean(tox_paths)),
        "std_tox_path": float(np.std(tox_paths, ddof=1)) if n > 1 else 0.0,
        "mean_common_essential_per_ep": float(np.mean(ce_counts)),
        "fraction_zero_common_essential": float(np.mean([1.0 if c == 0 else 0.0 for c in ce_counts])),
        "mean_unc_path_max": float(np.mean(unc_maxes)),
        "mean_unc_path_mean": float(np.mean(unc_means)),
        # Action frequency
        "action_freq": action_freq,
        "top_actions": sorted(
            ({"gene_symbol": g, "count": int(c)} for g, c in action_freq.items()),
            key=lambda x: -x["count"],
        )[:10],
    }


# ---------------------------------------------------------------------------
# PPO roster construction
# ---------------------------------------------------------------------------


def _parse_ppo_spec(spec: str) -> tuple[str, Path]:
    """Parse 'label:path' string into (label, Path)."""
    if ":" not in spec:
        raise ValueError(f"PPO spec must be 'label:path', got {spec!r}")
    label, path = spec.split(":", 1)
    return label.strip(), Path(path.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae_dir", default="artifacts/vae")
    parser.add_argument("--dynamics_dir", default="artifacts_v2/dynamics_v1ot_ror_corr010")
    parser.add_argument("--pairs_dir", default="artifacts/pairs")
    parser.add_argument(
        "--ppo", action="append", default=[],
        help="One or more 'label:path' specs (e.g. 'PPO_BCD:artifacts_v3/rl_v3b_biorealistic_fused_epsp10_seed42/ppo.zip'). "
             "Repeat for each PPO to evaluate.",
    )
    parser.add_argument(
        "--ppo_zip_A",
        default="artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed42/ppo.zip",
        help="Optional frozen V2 baseline (auto-added as PPO_A if present and not in --ppo).",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--epsilon_value", type=float, required=True,
                        help="Success-distance threshold ε (e.g. 2.9898 for p15).")
    parser.add_argument("--epsilon_label", default="custom")
    parser.add_argument("--biology_dir", default="artifacts_v3/v3b_biology")
    parser.add_argument("--free_steps", type=int, default=3)
    parser.add_argument("--mild_until", type=int, default=5)
    parser.add_argument("--mild_beta", type=float, default=0.02)
    parser.add_argument("--heavy_beta", type=float, default=0.10)
    parser.add_argument("--success_bonus", type=float, default=1.0)
    parser.add_argument("--lambda_tox", type=float, default=0.10)
    parser.add_argument("--lambda_ce", type=float, default=0.05)
    parser.add_argument("--lambda_unc_path", type=float, default=0.05)
    parser.add_argument("--uncertainty_clip_min", type=float, default=-5.0)
    parser.add_argument("--uncertainty_clip_max", type=float, default=3.0)
    parser.add_argument("--cells", nargs="+", default=list(CELL_DEFS.keys()))
    parser.add_argument("--max_greedy_depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.analysis.gate_breakdown import load_dynamics_model
    from src.analysis.path_feasibility import load_biology_layer
    from src.rl.baselines import (
        AlwaysNoopPolicy, GreedyDynamicsBeamPolicy, RandomUniformValidPolicy,
    )
    from src.rl.biology_rewards import build_safety_arrays

    def _abs(p): return (repo_root / p).resolve() if not Path(p).is_absolute() else Path(p)
    vae_dir = _abs(args.vae_dir)
    dyn_dir = _abs(args.dynamics_dir)
    pairs_dir = _abs(args.pairs_dir)
    biology_dir = _abs(args.biology_dir)
    out_dir = _abs(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load assets
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

    layer = load_biology_layer(biology_dir)
    tox_arr, ess_arr = build_safety_arrays(layer.gene_safety, n_genes=n_genes, permute_chronos=False)
    LOG.info("Biology layer: %d with Chronos, %d essential", int((~tox_arr.astype(bool) | True).sum()), int(ess_arr.sum()))

    dyn = load_dynamics_model(dyn_dir)

    # Build policy roster
    from sb3_contrib import MaskablePPO

    class _SB3Policy:
        def __init__(self, model, *, deterministic=True, name="ppo"):
            self.model = model
            self.deterministic = bool(deterministic)
            self.name = name
        def select_action(self, z, mask, info):
            a, _ = self.model.predict(z, deterministic=self.deterministic, action_masks=mask)
            return int(np.asarray(a).item())

    ppo_roster: dict[str, Any] = {}
    for spec in args.ppo:
        label, path = _parse_ppo_spec(spec)
        path_abs = _abs(path)
        if not path_abs.exists():
            LOG.warning("PPO %s missing at %s — skipping.", label, path_abs); continue
        ppo_roster[label] = _SB3Policy(MaskablePPO.load(str(path_abs), device="cpu"), name=label)

    # Auto-add PPO_A if present and not already labelled
    ppo_a_path = _abs(args.ppo_zip_A)
    if "PPO_A" not in ppo_roster and ppo_a_path.exists():
        ppo_roster["PPO_A"] = _SB3Policy(MaskablePPO.load(str(ppo_a_path), device="cpu"), name="PPO_A")

    freeband = {
        "free_steps": args.free_steps, "mild_until": args.mild_until,
        "mild_beta": args.mild_beta, "heavy_beta": args.heavy_beta,
        "success_bonus": args.success_bonus,
    }

    def _make_greedy(depth: int) -> Any:
        return GreedyDynamicsBeamPolicy(
            dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx,
            depth=depth, beam_width=20,
            freeband_schedule=freeband,
            success_epsilon=args.epsilon_value,
            safety_tox_per_action=tox_arr,
            safety_essential_per_action=ess_arr,
            lambda_tox=args.lambda_tox,
            lambda_ce=args.lambda_ce,
            lambda_unc_path=args.lambda_unc_path,
            uncertainty_clip_min=args.uncertainty_clip_min,
            uncertainty_clip_max=args.uncertainty_clip_max,
        )

    other_policies = {
        "random_uniform_valid": RandomUniformValidPolicy(seed=args.seed),
        "always_noop":          AlwaysNoopPolicy(noop_idx),
        "greedy_dyn_1_fused":   _make_greedy(1),
        "greedy_dyn_2_fused":   _make_greedy(2),
        "greedy_dyn_3_fused":   _make_greedy(3),
        "greedy_dyn_5_fused":   _make_greedy(5),
        "greedy_dyn_8_fused":   _make_greedy(8),
    }
    LOG.info("PPO roster: %s", list(ppo_roster))
    LOG.info("Baselines: %s", list(other_policies))
    LOG.info("epsilon = %.6f (label=%s)", args.epsilon_value, args.epsilon_label)

    def _is_relevant_policy(pol_name: str, cell_k: int) -> bool:
        if not pol_name.startswith("greedy_dyn_"):
            return True
        try:
            d = int(pol_name.split("_")[2])
        except (IndexError, ValueError):
            return True
        return d <= min(args.max_greedy_depth, cell_k)

    rows: list[dict[str, Any]] = []
    for cell_name in args.cells:
        if cell_name not in CELL_DEFS:
            LOG.warning("Unknown cell %s — skipping", cell_name); continue
        defn = CELL_DEFS[cell_name]
        k_cell = int(defn["k"])
        bin_def = defn["bin"]
        try:
            start_pool = _load_start_pool(vae_dir, distance_bin=bin_def, held_out_genes=held_out_genes)
        except ValueError as exc:
            LOG.warning("Cell %s: empty start pool (%s) — skipping", cell_name, exc); continue
        LOG.info("Cell %s: K=%d bin=%s |pool|=%d", cell_name, k_cell, bin_def[2], len(start_pool))

        policies_for_cell = {**ppo_roster, **other_policies}
        for pol_name, policy in policies_for_cell.items():
            if not _is_relevant_policy(pol_name, k_cell):
                LOG.debug("  %-30s skipped (depth > min(cap, K))", pol_name); continue
            t0 = time.time()
            env = _make_fused_env(
                dynamics_model=dyn, z_ref=z_ref, epsilon=args.epsilon_value,
                n_genes=n_genes, max_steps=k_cell, start_pool=start_pool, seed=args.seed,
                safety_tox_arr=tox_arr, safety_ess_arr=ess_arr,
                freeband=freeband, lambda_tox=args.lambda_tox, lambda_ce=args.lambda_ce,
                lambda_unc_path=args.lambda_unc_path,
                uncertainty_clip_min=args.uncertainty_clip_min,
                uncertainty_clip_max=args.uncertainty_clip_max,
            )
            result = _run_policy(env, policy, n_episodes=int(args.n_episodes), gene_lookup=gene_lookup)
            result.update({
                "cell": cell_name, "policy": pol_name, "k": k_cell,
                "epsilon": float(args.epsilon_value), "epsilon_label": args.epsilon_label,
                "distance_bin": bin_def[2], "gene_split": "ood",
                "n_start_pool": int(len(start_pool)),
                "elapsed_sec": time.time() - t0,
            })
            cell_dir = out_dir / cell_name / pol_name
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))

            rows.append({
                "epsilon_label": args.epsilon_label,
                "cell": cell_name, "policy": pol_name, "k": k_cell,
                "n_episodes": result["n_episodes"],
                "success_rate": result["success_rate"],
                "mean_steps": result["mean_steps"],
                "mean_final_distance": result["mean_final_distance"],
                "mean_total_reward": result["mean_total_reward"],
                "mean_tox_path": result["mean_tox_path"],
                "mean_common_essential_per_ep": result["mean_common_essential_per_ep"],
                "fraction_zero_common_essential": result["fraction_zero_common_essential"],
                "mean_unc_path_max": result["mean_unc_path_max"],
                "mean_unc_path_mean": result["mean_unc_path_mean"],
                "frac_success_T_le_3": result["frac_success_T_le_3"],
                "frac_success_T_4_or_5": result["frac_success_T_4_or_5"],
                "frac_success_T_gt_5": result["frac_success_T_gt_5"],
                "n_successful_episodes": result["n_successful_episodes"],
            })
            LOG.info(
                "  %-32s success=%.3f steps=%.2f tox=%.4f ce=%.3f unc=%.3f  (%.1fs)",
                pol_name, result["success_rate"], result["mean_steps"],
                result["mean_tox_path"], result["mean_common_essential_per_ep"],
                result["mean_unc_path_max"], result["elapsed_sec"],
            )

    df = pl.DataFrame(rows)
    df.write_parquet(str(out_dir / "aggregate.parquet"))
    df.write_csv(str(out_dir / "aggregate.csv"))
    LOG.info("Wrote aggregate.{parquet,csv} (%d rows)", df.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
