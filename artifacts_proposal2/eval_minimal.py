"""Minimal V3B evaluator — avoids the giant latents.h5ad reload.

The official scripts/evaluate_rl_v3b.py reads norman_hvg.h5ad + latents.h5ad
(~2 GB combined) which hangs under memory pressure on the 8 GB Mac being
used for this round. This script reproduces the v3b headline metrics
(success rate, mean common-essential per episode, mean weighted Chronos)
using only the val_pairs.npz start pool + the V3B biology layer parquet.

Outputs:
  - artifacts_proposal2/eval/per_policy_metrics.json
  - artifacts_proposal2/eval/per_policy_metrics.md
  - artifacts_proposal2/eval/rollouts_all.parquet

Coverage:
  * ppo_C            (artifacts_proposal2/rl_v3b_safety_aware_seed42/ppo.zip)
  * ppo_C_permuted   (artifacts_proposal2/rl_v3b_safety_aware_seed42_permuted_chronos/ppo.zip)
  * random_uniform_valid (sanity baseline; never picks no-op so success can be measured)
  * always_noop      (lower bound — pure inaction)

Honest caveats:
  - Uses val-cell start pool, NOT the V2 hardness-cell stratification
    (K=3, ε=p25=3.166, bin 8-10, OOD). Cross-policy comparisons are
    apples-to-apples within this start pool.
  - greedy_dyn_k baselines are not included; the safety claim only requires
    PPO_C > PPO_C_permuted on safety + parity-or-better on success.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch

from sb3_contrib import MaskablePPO

from src.rl.environment import CellReprogrammingEnv
from src.models.dynamics import PerturbationDynamicsModel


LOG = logging.getLogger("eval_minimal")


def load_dynamics(dyn_dir: Path) -> tuple[PerturbationDynamicsModel, dict]:
    """Load a trained dynamics model."""
    cfg = json.loads((dyn_dir / "config.json").read_text())
    sig = inspect.signature(PerturbationDynamicsModel.__init__)
    kwargs = {k: v for k, v in cfg.items() if k in sig.parameters}
    for k, default in [
        ("use_layernorm", True),
        ("use_state_linear_skip", False),
        ("use_residual_over_ridge", False),
        ("use_gene_delta_bias", False),
        ("log_var_init_bias", -2.0),
    ]:
        kwargs.setdefault(k, default)
    model = PerturbationDynamicsModel(**kwargs)
    state = torch.load(dyn_dir / "model.pt", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_biology_arrays(biology_dir: Path, n_genes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (chronos[n_genes], is_essential[n_genes], tox_norm[n_genes]) aligned by action_idx."""
    df = pl.read_parquet(biology_dir / "gene_safety.parquet").sort("action_idx")
    chr_arr = np.zeros(n_genes, dtype=np.float32)
    is_ess = np.zeros(n_genes, dtype=bool)
    tox = np.zeros(n_genes, dtype=np.float32)
    for row in df.iter_rows(named=True):
        i = int(row["action_idx"])
        if i < n_genes:
            chr_arr[i] = float(row["chronos"] or 0.0)
            is_ess[i] = bool(row["is_essential"]) if row["is_essential"] is not None else False
            tox[i] = float(row["tox_norm"] or 0.0)
    return chr_arr, is_ess, tox


def build_env(
    dyn_dir: Path,
    start_pool: np.ndarray,
    biology_dir: Path,
    z_ref: np.ndarray,
    eps: float,
    max_steps: int,
) -> tuple[CellReprogrammingEnv, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a CellReprogrammingEnv with the supplied start pool and biology arrays."""
    dyn_model, dyn_cfg = load_dynamics(dyn_dir)
    n_genes = int(dyn_cfg["n_genes"])
    chr_arr, is_ess, tox = load_biology_arrays(biology_dir, n_genes)

    env = CellReprogrammingEnv(
        dynamics_model=dyn_model,
        z_reference_centroid=z_ref.astype(np.float32),
        epsilon_success=float(eps),
        n_genes=n_genes,
        max_steps=max_steps,
        start_pool_latents=start_pool.astype(np.float32),
        reward_mode="terminal_only_step_cost",  # neutral reward for fair cross-policy comparison
        safety_tox_per_action=tox,
        safety_essential_per_action=is_ess,
        lambda_tox=0.0,   # not used; env still tracks tox_path
        lambda_ce=0.0,
    )
    return env, chr_arr, is_ess, tox


def rollout_policy(
    env: CellReprogrammingEnv,
    policy_name: str,
    policy,
    n_episodes: int,
    rng: np.random.Generator,
    chr_arr: np.ndarray,
    is_ess: np.ndarray,
    tox: np.ndarray,
) -> list[dict]:
    """Roll out a policy for n_episodes; return per-step rows."""
    rows: list[dict] = []
    noop_idx = env.noop_idx
    n_genes = env.n_genes
    for ep in range(n_episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=seed)
        step = 0
        while True:
            step += 1
            mask = info.get("action_mask")
            if mask is None:
                mask = np.ones(n_genes + 1, dtype=bool)
            if policy_name == "always_noop":
                action = noop_idx
            elif policy_name == "random_uniform_valid":
                valid = np.where(mask)[0]
                # exclude noop so this baseline doesn't trivially terminate on step 1
                valid_nonoop = valid[valid != noop_idx]
                action = int(rng.choice(valid_nonoop)) if len(valid_nonoop) > 0 else noop_idx
            else:  # PPO
                act, _ = policy.predict(obs, action_masks=mask, deterministic=True)
                action = int(act)
            obs, reward, terminated, truncated, info = env.step(action)
            is_noop_step = (action == noop_idx)
            row = {
                "policy": policy_name,
                "episode_id": ep,
                "step": step,
                "action": int(action),
                "is_noop": bool(is_noop_step),
                "gene_chronos": 0.0 if is_noop_step else float(chr_arr[action]),
                "gene_is_essential": False if is_noop_step else bool(is_ess[action]),
                "gene_tox": 0.0 if is_noop_step else float(tox[action]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(info.get("success", False)),
                "tox_path": float(info.get("tox_path", 0.0)),
                "ce_count": int(info.get("common_essential_count", 0)),
            }
            rows.append(row)
            if terminated or truncated:
                break
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per-step rows into per-policy summary."""
    df = pl.DataFrame(rows)
    # Per-episode aggregates (filter out noop steps for biology aggregation)
    gene_only = df.filter(~pl.col("is_noop"))
    # Use last step per episode for success / tox_path / ce_count (env-accumulated)
    last_per_ep = (
        df.sort(["episode_id", "step"])
        .group_by("episode_id", maintain_order=True)
        .agg([
            pl.col("success").last(),
            pl.col("tox_path").last(),
            pl.col("ce_count").last(),
            pl.col("step").max().alias("n_steps"),
        ])
    )
    gene_per_ep = (
        gene_only.group_by("episode_id")
        .agg([
            pl.col("gene_is_essential").sum().alias("n_ce_picks"),
            pl.col("gene_chronos").mean().alias("mean_chronos_per_pick"),
            pl.col("gene_tox").sum().alias("path_tox_picks"),
        ])
    )
    return {
        "n_episodes": int(last_per_ep.height),
        "success_rate": float(last_per_ep["success"].mean()),
        "mean_n_steps": float(last_per_ep["n_steps"].mean()),
        "mean_path_tox_env": float(last_per_ep["tox_path"].mean()),
        "mean_ce_count_env": float(last_per_ep["ce_count"].mean()),
        # Per-pick (gene-action only) breakdowns:
        "n_episodes_with_picks": int(gene_per_ep.height),
        "frac_zero_CE_picks": float((gene_per_ep["n_ce_picks"] == 0).mean()) if gene_per_ep.height else None,
        "mean_CE_picks_per_episode": float(gene_per_ep["n_ce_picks"].mean()) if gene_per_ep.height else 0.0,
        "wmean_chronos_picks": float(gene_per_ep["mean_chronos_per_pick"].drop_nulls().mean() or 0.0) if gene_per_ep.height else 0.0,
        "mean_path_tox_picks": float(gene_per_ep["path_tox_picks"].mean()) if gene_per_ep.height else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics_dir", default="artifacts_proposal1/dynamics_v1ot_ror")
    parser.add_argument("--pairs_val", default="artifacts/pairs/val_pairs.npz")
    parser.add_argument("--vae_centroid", default="artifacts/vae/z_reference_centroid.npy")
    parser.add_argument("--vae_eps", default="artifacts/vae/epsilon_success.json")
    parser.add_argument("--biology_dir", default="artifacts_v3/v3b_biology")
    parser.add_argument("--ppo_zip_C", required=True)
    parser.add_argument("--ppo_zip_C_permuted", required=True)
    parser.add_argument("--out_dir", default="artifacts_proposal2/eval")
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--eps_value", type=float, default=None, help="Override env ε; default reads from epsilon_success.json")
    parser.add_argument("--start_dist_min", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val = np.load(args.pairs_val)
    z_pert = val["z_pert"]
    z_ref = np.load(args.vae_centroid)
    dists = np.linalg.norm(z_pert - z_ref[None, :], axis=1)
    keep = dists > args.start_dist_min
    start_pool = z_pert[keep]
    LOG.info("Start pool: %d / %d val cells with dist > %.1f (median=%.3f, p25=%.3f, p75=%.3f)",
             keep.sum(), len(z_pert), args.start_dist_min,
             float(np.median(dists)), float(np.percentile(dists, 25)), float(np.percentile(dists, 75)))

    eps = args.eps_value
    if eps is None:
        eps_data = json.loads(Path(args.vae_eps).read_text())
        eps = float(eps_data.get("p90") or eps_data.get("value") or 4.5)
    LOG.info("Env ε = %.4f", eps)

    env, chr_arr, is_ess, tox = build_env(
        Path(args.dynamics_dir), start_pool, Path(args.biology_dir), z_ref, eps, args.max_steps
    )
    LOG.info("Env ready: n_genes=%d, max_steps=%d, ε=%.4f, n_essential=%d",
             env.n_genes, env.max_steps, env.epsilon, int(is_ess.sum()))

    ppo_C = MaskablePPO.load(args.ppo_zip_C, device="cpu")
    ppo_Cp = MaskablePPO.load(args.ppo_zip_C_permuted, device="cpu")
    LOG.info("Loaded PPO_C and PPO_C_permuted")

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for policy_name, policy in [
        ("ppo_C", ppo_C),
        ("ppo_C_permuted", ppo_Cp),
        ("random_uniform_valid", None),
        ("always_noop", None),
    ]:
        LOG.info("Evaluating %s for %d episodes ...", policy_name, args.n_episodes)
        rows = rollout_policy(env, policy_name, policy, args.n_episodes, rng, chr_arr, is_ess, tox)
        all_rows.extend(rows)
        s = aggregate(rows)
        summaries[policy_name] = s
        LOG.info("  → %s: SR=%.3f frac_zero_CE=%s mean_CE/ep=%.4f wmean_chronos=%+.4f n_steps=%.2f",
                 policy_name, s["success_rate"],
                 f"{s['frac_zero_CE_picks']:.3f}" if s["frac_zero_CE_picks"] is not None else "n/a",
                 s["mean_CE_picks_per_episode"], s["wmean_chronos_picks"], s["mean_n_steps"])

    pl.DataFrame(all_rows).write_parquet(out_dir / "rollouts_all.parquet")
    (out_dir / "per_policy_metrics.json").write_text(json.dumps(summaries, indent=2))

    md = [
        "# V3B Phase 2 — Minimal Evaluation (Proposal 2)",
        "",
        f"- Dynamics: `{args.dynamics_dir}` (V1 OT pairs + RoR + corr0.10)",
        f"- PPO_C: `{args.ppo_zip_C}`",
        f"- PPO_C_permuted: `{args.ppo_zip_C_permuted}`",
        f"- Start pool: {int(keep.sum())} held-out val cells with ‖z − z_ref‖ > {args.start_dist_min}",
        f"- ε = {eps:.4f} (p90 of control cell distances; epsilon_success.json)",
        f"- max_steps = {args.max_steps}; n_episodes per policy = {args.n_episodes}",
        "",
        "Reward used in env at eval time: `terminal_only_step_cost` (V2 reward, neutral cross-policy).",
        "Both PPOs were *trained* under `safety_aware` reward; differing terminal eval reward isolates",
        "the *policy* differences rather than reward-formula differences.",
        "",
        "| Policy | success_rate | frac_zero_CE | mean_CE/ep | mean_path_tox | wmean_chronos | mean_n_steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["ppo_C", "ppo_C_permuted", "random_uniform_valid", "always_noop"]:
        s = summaries[name]
        md.append(
            f"| {name} | {s['success_rate']:.3f} | "
            f"{(s['frac_zero_CE_picks'] or 0):.3f} | "
            f"{s['mean_CE_picks_per_episode']:.4f} | "
            f"{s['mean_path_tox_picks']:.4f} | "
            f"{s['wmean_chronos_picks']:+.4f} | "
            f"{s['mean_n_steps']:.2f} |"
        )
    md += [
        "",
        "**Legend**",
        "- *success_rate* — fraction of episodes that reached ‖z − z_ref‖ < ε within max_steps.",
        "- *frac_zero_CE* — fraction of episodes that picked **zero** common-essential genes (CBFA2T3, HK2, PLK4, PTPN1, STIL).",
        "- *mean_CE/ep* — mean count of common-essential genes picked per episode.",
        "- *mean_path_tox* — mean cumulative `tox_norm` of gene-action steps per episode.",
        "- *wmean_chronos* — mean of (per-episode mean Chronos of gene picks). More negative = more essential-leaning.",
        "- *mean_n_steps* — mean number of steps before termination.",
    ]
    (out_dir / "per_policy_metrics.md").write_text("\n".join(md) + "\n")
    LOG.info("Wrote outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
