"""V3B Phase 2 — Safety-aware hardness-frontier evaluator with acceptance criteria.

Evaluates PPO_C (safety-aware) against:
* PPO_A: existing V2 primary PPO (frozen baseline, ``rl_v1ot_ror_corr010_*``).
* PPO_C_permuted_chronos: null-control PPO trained on Chronos-permuted safety table.
* greedy_dyn_2 under reward C: planner with the same safety penalty.
* random_uniform_valid, always_noop.

Across the V2 hardness frontier (K ∈ {2, 3} × bin ∈ {6-8, 8-10} × OOD), computing:
* raw success rate
* safety-adjusted success rate (V3B headline metric)
* mean per-episode tox_path / common_essential_count
* fraction of zero-essential episodes
* weighted action-freq Chronos / fraction-essential

And applying the 5-rule acceptance check (per the 2026-05-17 user directive):

1. Safety-adjusted PPO_C − greedy_dyn_2_C ≥ +0.03 at ≥ 1 frontier cell.
2. Raw success not catastrophically worse than PPO_A / greedy.
3. PPO_C reduces common-essential picks or weighted Chronos risk vs PPO_A.
4. Real-Chronos PPO_C beats permuted-Chronos PPO_C.
5. Strongest result at a non-saturated cell (e.g. K=2/bin 6-8/OOD).

Writes ``artifacts_v3/eval_v3b_phase2/`` with per-cell summaries, per-policy
metadata, an aggregate parquet, an acceptance-check JSON, and a Markdown report.
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


LOG = logging.getLogger("evaluate_rl_v3b")


# ---------------------------------------------------------------------------
# Helpers (reuse evaluate_rl_hard internals where possible)
# ---------------------------------------------------------------------------


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _make_safety_aware_env(
    *, dynamics_model, z_ref, epsilon, n_genes, max_steps, start_pool, seed,
    tox_arr, ess_arr, lambda_tox, lambda_ce,
):
    from src.rl.environment import CellReprogrammingEnv
    return CellReprogrammingEnv(
        dynamics_model=dynamics_model,
        z_reference_centroid=z_ref,
        epsilon_success=float(epsilon),
        n_genes=int(n_genes),
        max_steps=int(max_steps),
        lambda_sparse=0.0,  # safety_aware ignores sparsity by design
        lambda_unc=0.0,
        repeat_mask=True,
        start_state_strategy="random_perturbation",
        distance_metric="l2",
        start_pool_latents=start_pool,
        success_bonus=0.0,
        failure_penalty=0.0,
        seed=seed,
        reward_mode="safety_aware",
        beta_step_cost=0.05,
        safety_tox_per_action=tox_arr,
        safety_essential_per_action=ess_arr,
        lambda_tox=float(lambda_tox),
        lambda_ce=float(lambda_ce),
    )


def _run_policy(
    env, policy, *, n_episodes: int, gene_lookup: dict[int, str],
) -> dict[str, Any]:
    """Per-policy aggregate. Reads env's ``info`` safety accumulators per episode."""
    successes = 0
    steps_list = []
    final_distances = []
    rewards = []
    tox_paths = []
    ce_counts = []
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

        successes += int(terminal_success and terminated)
        steps_list.append(int(last_info.get("step", 0)))
        final_distances.append(float(last_info.get("distance", np.nan)))
        rewards.append(ep_reward)
        tox_paths.append(float(last_info.get("tox_path", 0.0)))
        ce_counts.append(int(last_info.get("common_essential_count", 0)))

    n = int(n_episodes)
    wlo, whi = _wilson_ci(successes, n)

    # Safety-adjusted success rate is computed by the caller using a second pass
    # to capture per-episode success flags (synchronised with ``ce_counts``).
    return {
        "n_episodes": n,
        "successes": int(successes),
        "success_rate": float(successes / max(n, 1)),
        "success_rate_wilson95_low": wlo,
        "success_rate_wilson95_high": whi,
        "mean_steps": float(np.mean(steps_list)),
        "mean_final_distance": float(np.nanmean(final_distances)),
        "mean_total_reward": float(np.mean(rewards)),
        "mean_tox_path": float(np.mean(tox_paths)),
        "std_tox_path": float(np.std(tox_paths, ddof=1)) if n > 1 else 0.0,
        "mean_common_essential_per_ep": float(np.mean(ce_counts)),
        "std_common_essential_per_ep": float(np.std(ce_counts, ddof=1)) if n > 1 else 0.0,
        "fraction_zero_common_essential": float(np.mean([1.0 if c == 0 else 0.0 for c in ce_counts])),
        "tox_paths": tox_paths,         # full lists for derived metrics downstream
        "ce_counts": ce_counts,
        "final_distances": final_distances,
        "action_freq": action_freq,
        "top_actions": sorted(
            ({"gene_symbol": g, "count": int(c)} for g, c in action_freq.items()),
            key=lambda x: -x["count"],
        )[:10],
    }


def _per_episode_success_flags(env, policy, *, n_episodes: int) -> list[bool]:
    """Lighter-weight re-run that only captures the success flag per episode."""
    flags: list[bool] = []
    for ep in range(int(n_episodes)):
        obs, info = env.reset(seed=ep)
        terminated, truncated = False, False
        terminal_success = False
        while not (terminated or truncated):
            action = int(policy.select_action(obs, info["action_mask"], info))
            obs, _reward, terminated, truncated, info = env.step(action)
            terminal_success = bool(info.get("success", False))
        flags.append(terminal_success and terminated)
    return flags


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------


def _load_start_pool(vae_dir: Path, *, distance_bin, held_out_genes):
    import anndata as ad
    lo, hi, label = distance_bin
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


# ---------------------------------------------------------------------------
# Acceptance-criteria check
# ---------------------------------------------------------------------------


def _safety_adjusted_success_rate(success_flags: list[bool], ce_counts: list[int]) -> float:
    """Strict definition: episode counts only if it succeeded AND ce_count == 0."""
    if not success_flags:
        return 0.0
    return float(np.mean(
        [1.0 if (s and c == 0) else 0.0 for s, c in zip(success_flags, ce_counts, strict=True)]
    ))


def evaluate_acceptance(
    by_cell: dict[str, dict[str, dict[str, Any]]],
    *,
    target_cells: list[str],
    catastrophic_threshold: float = 0.20,
) -> dict[str, Any]:
    """Apply the 5-rule acceptance check across cells.

    Parameters
    ----------
    by_cell
        ``{cell: {policy: per_policy_dict}}`` from the eval loop.
    target_cells
        Cells over which to look for ≥+0.03 safety-adjusted advantage. We expect
        ``k2_epsp25_bin6-8_splitood`` to be the leverage cell (per Phase 1).
    catastrophic_threshold
        Maximum acceptable raw-success regression vs PPO_A. Default 0.20.

    Returns
    -------
    dict
        ``{"verdict": "ACCEPT"|"REJECT", "rules": {1..5: {...}}, "by_cell": ...}``
    """
    rules: dict[str, dict[str, Any]] = {}

    # ---- Rule 1: safety-adjusted Δ(PPO_C, greedy_dyn_2_C) ≥ +0.03 at ≥ 1 cell ----
    rule1_hits = []
    for cell in target_cells:
        c = by_cell.get(cell, {})
        ppo_c = c.get("ppo_C")
        grd_c = c.get("greedy_dyn_2_C")
        if ppo_c is None or grd_c is None:
            continue
        ppo_safe = ppo_c.get("safety_adjusted_success_rate")
        grd_safe = grd_c.get("safety_adjusted_success_rate")
        if ppo_safe is None or grd_safe is None:
            continue
        delta = float(ppo_safe - grd_safe)
        rule1_hits.append({"cell": cell, "delta_safety_adjusted": delta})
    rule1_max = max([h["delta_safety_adjusted"] for h in rule1_hits], default=None)
    rules["1_safety_adjusted_vs_greedy"] = {
        "threshold": 0.03,
        "max_delta_across_cells": rule1_max,
        "per_cell": rule1_hits,
        "passed": (rule1_max is not None and rule1_max >= 0.03),
    }

    # ---- Rule 2: raw success not catastrophically worse than PPO_A / greedy_dyn_2 ----
    rule2_per_cell = []
    rule2_catastrophic = False
    for cell, policies in by_cell.items():
        ppo_c = policies.get("ppo_C", {})
        ppo_a = policies.get("ppo_A", {})
        grd_a = policies.get("greedy_dyn_2_A", {}) or policies.get("greedy_dyn_2", {})
        sr_c = ppo_c.get("success_rate")
        sr_a = ppo_a.get("success_rate")
        sr_g = grd_a.get("success_rate")
        if sr_c is None:
            continue
        worst_baseline = max([x for x in (sr_a, sr_g) if x is not None], default=None)
        regression = (worst_baseline - sr_c) if worst_baseline is not None else None
        is_catastrophic = bool(regression is not None and regression > catastrophic_threshold)
        rule2_per_cell.append({
            "cell": cell, "ppo_C": sr_c, "ppo_A": sr_a, "greedy_A": sr_g,
            "regression_vs_worst_baseline": regression,
            "is_catastrophic": is_catastrophic,
        })
        if is_catastrophic:
            rule2_catastrophic = True
    rules["2_raw_success_not_catastrophic"] = {
        "threshold": catastrophic_threshold,
        "any_catastrophic": rule2_catastrophic,
        "per_cell": rule2_per_cell,
        "passed": not rule2_catastrophic,
    }

    # ---- Rule 3: PPO_C reduces (a) common-essential picks OR (b) weighted Chronos vs PPO_A ----
    rule3_per_cell = []
    rule3_any_hit = False
    for cell, policies in by_cell.items():
        ppo_c = policies.get("ppo_C", {})
        ppo_a = policies.get("ppo_A", {})
        ce_c = ppo_c.get("mean_common_essential_per_ep")
        ce_a = ppo_a.get("mean_common_essential_per_ep")
        wmc_c = ppo_c.get("weighted_mean_chronos")
        wmc_a = ppo_a.get("weighted_mean_chronos")
        ce_reduced = (ce_c is not None and ce_a is not None and ce_c < ce_a - 1e-6)
        # "less Chronos-risk" = less negative weighted-mean Chronos (i.e. safer on avg).
        wmc_reduced = (wmc_c is not None and wmc_a is not None and wmc_c > wmc_a + 1e-3)
        any_hit = bool(ce_reduced or wmc_reduced)
        rule3_per_cell.append({
            "cell": cell,
            "ce_c": ce_c, "ce_a": ce_a, "ce_reduced": ce_reduced,
            "wmc_c": wmc_c, "wmc_a": wmc_a, "wmc_reduced": wmc_reduced,
            "any_axis_reduced": any_hit,
        })
        if any_hit:
            rule3_any_hit = True
    rules["3_safety_improvement_vs_ppo_A"] = {
        "per_cell": rule3_per_cell,
        "any_cell_with_reduction": rule3_any_hit,
        "passed": rule3_any_hit,
    }

    # ---- Rule 4: real-Chronos PPO_C beats permuted-Chronos PPO_C ----
    rule4_per_cell = []
    rule4_any_hit = False
    for cell, policies in by_cell.items():
        ppo_c = policies.get("ppo_C", {})
        ppo_cp = policies.get("ppo_C_permuted", {})
        sr_c = ppo_c.get("safety_adjusted_success_rate")
        sr_cp = ppo_cp.get("safety_adjusted_success_rate")
        if sr_c is None or sr_cp is None:
            continue
        delta = float(sr_c - sr_cp)
        rule4_per_cell.append({"cell": cell, "delta_real_minus_permuted": delta})
        if delta > 0.01:  # small margin so noise doesn't tilt
            rule4_any_hit = True
    rules["4_real_beats_permuted"] = {
        "per_cell": rule4_per_cell,
        "any_cell_real_strictly_better": rule4_any_hit,
        "passed": rule4_any_hit,
    }

    # ---- Rule 5: strongest result at a non-saturated cell ----
    non_saturated_cells = {"k2_epsp25_bin6-8_splitood", "k2_epsp25_bin8-10_splitood"}
    rule5_strongest_cell = None
    rule5_max_delta = -float("inf")
    for cell in by_cell:
        # Use the same per-cell deltas as Rule 1.
        c = by_cell.get(cell, {})
        ppo_c = c.get("ppo_C", {})
        grd_c = c.get("greedy_dyn_2_C", {})
        sr_c = ppo_c.get("safety_adjusted_success_rate")
        sr_g = grd_c.get("safety_adjusted_success_rate")
        if sr_c is None or sr_g is None:
            continue
        delta = float(sr_c - sr_g)
        if delta > rule5_max_delta:
            rule5_max_delta = delta
            rule5_strongest_cell = cell
    rules["5_strongest_at_non_saturated_cell"] = {
        "strongest_cell": rule5_strongest_cell,
        "is_non_saturated": (rule5_strongest_cell in non_saturated_cells),
        "max_delta": rule5_max_delta if rule5_strongest_cell is not None else None,
        "passed": (rule5_strongest_cell in non_saturated_cells),
    }

    # ---- Verdict ----
    passes = [int(rules[k]["passed"]) for k in rules]
    n_passed = int(sum(passes))
    # Plan-required: rules 1, 2, 4 are load-bearing; rule 5 is a tie-breaker; rule 3 is
    # secondary (a Pareto improvement). Headline ACCEPT if rules 1, 2, 4 all pass AND
    # ≥ 1 of rules 3, 5 passes.
    headline_passed = (
        rules["1_safety_adjusted_vs_greedy"]["passed"]
        and rules["2_raw_success_not_catastrophic"]["passed"]
        and rules["4_real_beats_permuted"]["passed"]
        and (rules["3_safety_improvement_vs_ppo_A"]["passed"]
             or rules["5_strongest_at_non_saturated_cell"]["passed"])
    )

    verdict = "ACCEPT" if headline_passed else "REJECT"
    return {
        "verdict": verdict,
        "n_passed": n_passed,
        "headline_pass_rule": "1 ∧ 2 ∧ 4 ∧ (3 ∨ 5)",
        "rules": rules,
    }


# ---------------------------------------------------------------------------
# Action-freq aggregate biology metrics (for Rule 3)
# ---------------------------------------------------------------------------


def _aggregate_action_freq_biology(action_freq: dict[str, int], gene_safety_df) -> dict[str, Any]:
    """Reproduces the Phase 1 aggregator on a single policy's action_freq."""
    safety_map = {row["gene_symbol"]: row for row in gene_safety_df.iter_rows(named=True)}
    cleaned = {g: c for g, c in action_freq.items() if g != "NO_OP" and c > 0}
    total = sum(cleaned.values())
    if total == 0:
        return {
            "weighted_mean_chronos": None,
            "weighted_mean_tox": None,
            "fraction_actions_common_essential": None,
        }
    wc_num, wc_den, wt_num, wess_num = 0.0, 0, 0.0, 0
    for gene, freq in cleaned.items():
        row = safety_map.get(gene)
        if row is None or row.get("missing_chronos") or row.get("chronos") is None:
            continue
        c = float(row["chronos"])
        wc_num += c * freq
        wc_den += freq
        wt_num += float(row["tox_raw"] or 0.0) * freq
        if row.get("is_essential"):
            wess_num += freq
    return {
        "weighted_mean_chronos": float(wc_num / wc_den) if wc_den > 0 else None,
        "weighted_mean_tox": float(wt_num / wc_den) if wc_den > 0 else None,
        "fraction_actions_common_essential": float(wess_num / total),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae_dir", default="artifacts/vae")
    parser.add_argument("--dynamics_dir", default="artifacts_v2/dynamics_v1ot_ror_corr010")
    parser.add_argument("--pairs_dir", default="artifacts/pairs")
    parser.add_argument(
        "--ppo_zip_C", required=True,
        help="Path to the safety-aware PPO checkpoint (.zip)."
    )
    parser.add_argument(
        "--ppo_zip_C_permuted", default=None,
        help="Optional path to the permuted-Chronos null-control PPO."
    )
    parser.add_argument(
        "--ppo_zip_A", default="artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed42/ppo.zip",
        help="Path to the V2 primary PPO (frozen baseline)."
    )
    parser.add_argument("--biology_dir", default="artifacts_v3/v3b_biology")
    parser.add_argument("--out_dir", default="artifacts_v3/eval_v3b_phase2")
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--lambda_tox", type=float, default=0.10)
    parser.add_argument("--lambda_ce", type=float, default=0.05)
    parser.add_argument(
        "--cells", nargs="+", default=[
            "k2_epsp25_bin6-8_splitood",
            "k2_epsp25_bin8-10_splitood",
            "k3_epsp25_bin6-8_splitood",
            "k3_epsp25_bin8-10_splitood",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.analysis.gate_breakdown import load_dynamics_model
    from src.analysis.path_feasibility import load_biology_layer
    from src.rl.baselines import (
        AlwaysNoopPolicy,
        GreedyDynamicsBeamPolicy,
        RandomUniformValidPolicy,
    )
    from src.rl.biology_rewards import build_safety_arrays

    vae_dir = (repo_root / args.vae_dir) if not Path(args.vae_dir).is_absolute() else Path(args.vae_dir)
    pairs_dir = (repo_root / args.pairs_dir) if not Path(args.pairs_dir).is_absolute() else Path(args.pairs_dir)
    dyn_dir = (repo_root / args.dynamics_dir) if not Path(args.dynamics_dir).is_absolute() else Path(args.dynamics_dir)
    out_dir = (repo_root / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    biology_dir = (repo_root / args.biology_dir) if not Path(args.biology_dir).is_absolute() else Path(args.biology_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve relative PPO zips
    def _abs(p): return (repo_root / p).resolve() if not Path(p).is_absolute() else Path(p)
    ppo_C = _abs(args.ppo_zip_C)
    ppo_A = _abs(args.ppo_zip_A)
    ppo_Cp = _abs(args.ppo_zip_C_permuted) if args.ppo_zip_C_permuted else None

    # --- Load assets ---
    with open(vae_dir / "gene_vocab.json") as f:
        vocab = json.load(f)
    genes = [str(g) for g in vocab["genes"]]
    n_genes = int(vocab["n_genes"])
    noop_idx = int(vocab["noop_idx"])
    gene_lookup = {i: g for i, g in enumerate(genes)}
    gene_lookup[noop_idx] = "NO_OP"
    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)

    # Eps p25
    with open(vae_dir / "epsilon_success.json") as f:
        eps_blob = json.load(f)
    # epsilon_success.json may not have p25; fall back to V2 primary value.
    if "p25" in eps_blob:
        epsilon = float(eps_blob["p25"])
    else:
        epsilon = 3.1662898064  # V2 primary p25; the V1 epsilon_success.json is p50.

    LOG.info("epsilon (p25) = %.6f", epsilon)

    # Held-out gene list
    with open(pairs_dir / "metadata.json") as f:
        pairs_meta = json.load(f)
    held_out_genes = [str(g) for g in pairs_meta.get("held_out_genes", [])]

    # Biology layer
    layer = load_biology_layer(biology_dir)
    tox_arr, ess_arr = build_safety_arrays(layer.gene_safety, n_genes=n_genes, permute_chronos=False)
    LOG.info("Biology layer: %d genes (%d with Chronos, %d essential)",
             layer.gene_safety.height,
             layer.coverage["gene_safety"]["n_with_chronos"],
             int(ess_arr.sum()))

    # Dynamics
    dyn = load_dynamics_model(dyn_dir)

    # PPO policies (lazy import sb3_contrib here to avoid import cost when not needed)
    from sb3_contrib import MaskablePPO

    class _SB3Policy:
        def __init__(self, model, deterministic=True, name="ppo"):
            self.model = model
            self.deterministic = bool(deterministic)
            self.name = name
        def select_action(self, z, mask, info):
            a, _ = self.model.predict(z, deterministic=self.deterministic, action_masks=mask)
            return int(np.asarray(a).item())

    policies_to_eval: dict[str, Any] = {}

    if ppo_C.exists():
        policies_to_eval["ppo_C"] = _SB3Policy(MaskablePPO.load(str(ppo_C), device="cpu"), name="ppo_C")
    else:
        raise FileNotFoundError(f"--ppo_zip_C does not exist: {ppo_C}")

    if ppo_A.exists():
        policies_to_eval["ppo_A"] = _SB3Policy(MaskablePPO.load(str(ppo_A), device="cpu"), name="ppo_A")
    else:
        LOG.warning("--ppo_zip_A does not exist: %s — skipping PPO_A baseline.", ppo_A)

    if ppo_Cp is not None and ppo_Cp.exists():
        policies_to_eval["ppo_C_permuted"] = _SB3Policy(MaskablePPO.load(str(ppo_Cp), device="cpu"), name="ppo_C_permuted")
    elif ppo_Cp is not None:
        LOG.warning("--ppo_zip_C_permuted does not exist: %s — Rule 4 will be inconclusive.", ppo_Cp)

    policies_to_eval["random_uniform_valid"] = RandomUniformValidPolicy(seed=args.seed)
    policies_to_eval["always_noop"] = AlwaysNoopPolicy(noop_idx)
    # greedy_dyn_2 under reward C (safety-aware planning) — the key fair comparator.
    policies_to_eval["greedy_dyn_2_C"] = GreedyDynamicsBeamPolicy(
        dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx, depth=2, beam_width=20,
        safety_tox_per_action=tox_arr, safety_essential_per_action=ess_arr,
        lambda_tox=args.lambda_tox, lambda_ce=args.lambda_ce,
    )
    # greedy_dyn_2 under reward A (V2 distance-only planning) — sanity reference.
    policies_to_eval["greedy_dyn_2_A"] = GreedyDynamicsBeamPolicy(
        dyn, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx, depth=2, beam_width=20,
    )

    LOG.info("Policies to evaluate: %s", list(policies_to_eval))

    # Parse cells
    bin_map = {
        "k2_epsp25_bin6-8_splitood":  (2, (6.0, 8.0, "6-8"), True),
        "k2_epsp25_bin8-10_splitood": (2, (8.0, 10.0, "8-10"), True),
        "k3_epsp25_bin6-8_splitood":  (3, (6.0, 8.0, "6-8"), True),
        "k3_epsp25_bin8-10_splitood": (3, (8.0, 10.0, "8-10"), True),
    }

    by_cell: dict[str, dict[str, dict[str, Any]]] = {}

    for cell_name in args.cells:
        if cell_name not in bin_map:
            LOG.warning("Unknown cell %s — skipping", cell_name)
            continue
        k, dist_bin, held_out = bin_map[cell_name]
        try:
            start_pool = _load_start_pool(vae_dir, distance_bin=dist_bin, held_out_genes=held_out_genes)
        except ValueError as exc:
            LOG.warning("Cell %s: empty start pool (%s) — skipping", cell_name, exc)
            continue
        LOG.info("Cell %s: k=%d, bin=%s, OOD=%s, |pool|=%d", cell_name, k, dist_bin[2], held_out, len(start_pool))

        cell_results: dict[str, dict[str, Any]] = {}
        for pol_name, policy in policies_to_eval.items():
            t0 = time.time()
            env = _make_safety_aware_env(
                dynamics_model=dyn, z_ref=z_ref, epsilon=epsilon, n_genes=n_genes,
                max_steps=k, start_pool=start_pool, seed=args.seed,
                tox_arr=tox_arr, ess_arr=ess_arr,
                lambda_tox=args.lambda_tox, lambda_ce=args.lambda_ce,
            )
            result = _run_policy(env, policy, n_episodes=int(args.n_episodes), gene_lookup=gene_lookup)
            elapsed = time.time() - t0

            # Per-episode success flags: we need the synchronised list for safety-adjusted SR.
            # Re-derive from result using a second lightweight pass (deterministic given seed).
            env2 = _make_safety_aware_env(
                dynamics_model=dyn, z_ref=z_ref, epsilon=epsilon, n_genes=n_genes,
                max_steps=k, start_pool=start_pool, seed=args.seed,
                tox_arr=tox_arr, ess_arr=ess_arr,
                lambda_tox=args.lambda_tox, lambda_ce=args.lambda_ce,
            )
            flags = _per_episode_success_flags(env2, policy, n_episodes=int(args.n_episodes))
            safe_sr = _safety_adjusted_success_rate(flags, result["ce_counts"])
            result["safety_adjusted_success_rate"] = safe_sr
            result["safety_adjusted_success_rate_wilson95"] = _wilson_ci(
                int(np.sum([f and c == 0 for f, c in zip(flags, result["ce_counts"], strict=True)])),
                len(flags),
            )

            # Action-freq aggregates (for Rule 3)
            agg = _aggregate_action_freq_biology(result["action_freq"], layer.gene_safety)
            result.update(agg)
            result["elapsed_sec"] = elapsed
            result.pop("tox_paths", None)  # too verbose for the summary; we have means+std
            result.pop("ce_counts", None)
            result.pop("final_distances", None)

            cell_results[pol_name] = result
            LOG.info(
                "  %s: success=%.3f  safe_adj=%.3f  mean_tox=%.4f  mean_ce=%.3f  (%.1fs)",
                pol_name, result["success_rate"], safe_sr,
                result["mean_tox_path"], result["mean_common_essential_per_ep"], elapsed,
            )

            # Write per-policy summary.json
            pol_dir = out_dir / cell_name / pol_name
            pol_dir.mkdir(parents=True, exist_ok=True)
            (pol_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))

        by_cell[cell_name] = cell_results

    # Apply acceptance criteria
    target_cells = [c for c in args.cells if c in by_cell]
    accept = evaluate_acceptance(by_cell, target_cells=target_cells)
    (out_dir / "acceptance.json").write_text(json.dumps(accept, indent=2, default=str))
    LOG.info("Acceptance verdict: %s", accept["verdict"])
    for k_, v in accept["rules"].items():
        LOG.info("  Rule %s passed=%s", k_, v.get("passed"))

    # Aggregate parquet
    rows: list[dict[str, Any]] = []
    for cell, policies in by_cell.items():
        for pol_name, r in policies.items():
            rows.append({
                "cell": cell, "policy": pol_name,
                "n_episodes": r["n_episodes"],
                "success_rate": r["success_rate"],
                "safety_adjusted_success_rate": r["safety_adjusted_success_rate"],
                "mean_steps": r["mean_steps"],
                "mean_final_distance": r["mean_final_distance"],
                "mean_tox_path": r["mean_tox_path"],
                "std_tox_path": r["std_tox_path"],
                "mean_common_essential_per_ep": r["mean_common_essential_per_ep"],
                "fraction_zero_common_essential": r["fraction_zero_common_essential"],
                "weighted_mean_chronos": r.get("weighted_mean_chronos"),
                "weighted_mean_tox": r.get("weighted_mean_tox"),
                "fraction_actions_common_essential": r.get("fraction_actions_common_essential"),
            })
    if rows:
        df = pl.DataFrame(rows)
        df.write_parquet(str(out_dir / "aggregate.parquet"))
        df.write_csv(str(out_dir / "aggregate.csv"))

    # Markdown report
    md = ["# V3B Phase 2 — Safety-aware PPO_C evaluation", "",
          f"> n_episodes={args.n_episodes}, seed={args.seed}, λ_tox={args.lambda_tox}, λ_ce={args.lambda_ce}",
          f"> dynamics: {args.dynamics_dir}", "",
          f"## Verdict: **{accept['verdict']}**", "",
          f"Headline rule: {accept['headline_pass_rule']}", ""]
    for k_, v in accept["rules"].items():
        md.append(f"* **Rule {k_}**: passed={v.get('passed')}")
    md.append("")
    md.append("## Per-cell × per-policy summary")
    md.append("")
    md.append("| cell | policy | success | safe_adj | mean_tox | mean_CE/ep | frac_zero_CE | wmean_chronos | frac_essential_actions |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for cell in by_cell:
        for pol_name, r in by_cell[cell].items():
            md.append(
                f"| {cell} | {pol_name} | "
                f"{r['success_rate']:.3f} | "
                f"{r['safety_adjusted_success_rate']:.3f} | "
                f"{r['mean_tox_path']:.4f} | "
                f"{r['mean_common_essential_per_ep']:.3f} | "
                f"{r['fraction_zero_common_essential']:.3f} | "
                f"{(r.get('weighted_mean_chronos') or 0):+.4f} | "
                f"{(r.get('fraction_actions_common_essential') or 0):+.4f} |"
            )
    (out_dir / "phase2_summary.md").write_text("\n".join(md) + "\n")
    LOG.info("Wrote %s", out_dir / "phase2_summary.md")

    return 0 if accept["verdict"] == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
