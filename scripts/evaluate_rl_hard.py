"""P0A.3 V2 hard benchmark for frozen V1 PPO and read-only baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.gate_breakdown import load_dynamics_model
from src.analysis.metrics import _fit_ridge_baseline, action_freq_chronos_spearman
from src.rl.baselines import (
    AlwaysNoopPolicy,
    GreedyDynamicsPolicy,
    MeanDeltaGreedyPolicy,
    NoopFreeGreedyPolicy,
    RandomUniformValidPolicy,
    RidgeGreedyPolicy,
)
from src.rl.environment import CellReprogrammingEnv
from src.utils.epsilon_percentile import compute_epsilon_percentile


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial success rate."""
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _parse_bool_list(values: list[str]) -> list[bool]:
    out: list[bool] = []
    for item in values:
        for part in str(item).split(","):
            val = part.strip().lower()
            if val in {"true", "1", "yes", "ood"}:
                out.append(True)
            elif val in {"false", "0", "no", "mixed"}:
                out.append(False)
            elif val:
                raise ValueError(f"Cannot parse boolean value {part!r}")
    return out


def _parse_bins(values: list[str]) -> list[tuple[float, float, str]]:
    bins: list[tuple[float, float, str]] = []
    for item in values:
        for part in str(item).split(","):
            if not part:
                continue
            lo_s, hi_s = part.split("-", 1)
            bins.append((float(lo_s), float(hi_s), f"{lo_s}-{hi_s}"))
    return bins


def _parse_csv(values: str | None) -> list[str]:
    if not values:
        return []
    return [x.strip() for x in values.split(",") if x.strip()]


def _load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_pairs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _fit_and_save_ridge_buffers(pairs_dir: Path, out_dir: Path, n_genes: int) -> Path:
    train = _load_pairs(pairs_dir / "train_pairs.npz")
    z = np.asarray(train["z_ctrl"], dtype=np.float32)
    g = np.asarray(train["gene_idx"], dtype=np.int32)
    delta = np.asarray(train["z_pert"], dtype=np.float32) - z
    ridge = _fit_ridge_baseline(z, g, delta, n_genes)
    coef = np.asarray(ridge.coef_, dtype=np.float32)
    W_z = coef[:, : z.shape[1]].T.astype(np.float32)
    W_gene = coef[:, z.shape[1] :].T.astype(np.float32)
    b = np.asarray(ridge.intercept_, dtype=np.float32)
    path = out_dir / "ridge_buffers.npz"
    np.savez(path, W_z=W_z, W_gene=W_gene, b=b)
    return path


def _build_mean_delta_table(pairs_dir: Path, n_genes: int, n_latent: int) -> np.ndarray:
    train = _load_pairs(pairs_dir / "train_pairs.npz")
    z = np.asarray(train["z_ctrl"], dtype=np.float32)
    g = np.asarray(train["gene_idx"], dtype=np.int32)
    delta = np.asarray(train["z_pert"], dtype=np.float32) - z
    global_mean = delta.mean(axis=0)
    table = np.broadcast_to(global_mean, (n_genes, n_latent)).copy().astype(np.float32)
    for gene_idx in sorted(np.unique(g).astype(int).tolist()):
        if 1 <= gene_idx <= n_genes:
            table[gene_idx - 1] = delta[g == gene_idx].mean(axis=0)
    return table


def _load_start_pool(
    vae_dir: Path,
    *,
    distance_bin: tuple[float, float, str],
    held_out_genes_only: bool,
    held_out_genes: list[str],
) -> np.ndarray:
    import anndata as ad

    adata = ad.read_h5ad(vae_dir / "latents.h5ad")
    z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)
    pert_idx = np.asarray(adata.obs["perturbation_idx"].values)
    mask = pert_idx != 0
    if held_out_genes_only:
        if "perturbation" not in adata.obs:
            raise ValueError("held_out_genes_only requires obs['perturbation'] in latents.h5ad")
        pert = np.asarray(adata.obs["perturbation"].astype(str).values)
        mask &= np.isin(pert, np.asarray(held_out_genes, dtype=object))
    lo, hi, _label = distance_bin
    d = np.linalg.norm(z - z_ref, axis=1)
    mask &= (d >= lo) & (d < hi)
    pool = z[mask].astype(np.float32)
    if len(pool) == 0:
        split = "ood" if held_out_genes_only else "mixed"
        raise ValueError(f"Empty start pool for split={split}, distance_bin={_label}")
    return pool


def _top_actions(action_freq: dict[str, int], n: int = 10) -> list[dict[str, Any]]:
    return [
        {"gene_symbol": str(g), "count": int(c)}
        for g, c in sorted(action_freq.items(), key=lambda item: -item[1])[:n]
    ]


def run_policy_episodes(
    env: Any,
    policy: Any,
    *,
    n_episodes: int,
    gene_lookup: dict[int, str],
) -> dict[str, Any]:
    """Run a policy object with ``select_action(z, mask, info)`` and summarize episodes."""
    successes = 0
    steps: list[int] = []
    final_distances: list[float] = []
    rewards: list[float] = []
    action_freq: dict[str, int] = {}

    for ep in range(int(n_episodes)):
        obs, info = env.reset(seed=ep)
        terminated = False
        truncated = False
        ep_steps = 0
        ep_reward = 0.0
        terminal_success = False
        while not (terminated or truncated):
            action = int(policy.select_action(obs, info["action_mask"], info))
            sym = gene_lookup.get(action, f"gene_{action}")
            action_freq[sym] = action_freq.get(sym, 0) + 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1
            ep_reward += float(reward)
            terminal_success = bool(info.get("success", False))
        successes += int(terminal_success and terminated)
        steps.append(ep_steps)
        rewards.append(ep_reward)
        final_distances.append(float(info.get("distance", np.nan)))

    lo, hi = wilson_ci(successes, int(n_episodes))
    return {
        "n_episodes": int(n_episodes),
        "successes": int(successes),
        "success_rate": float(successes / max(int(n_episodes), 1)),
        "success_rate_wilson95_low": lo,
        "success_rate_wilson95_high": hi,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "se_steps": float(np.std(steps, ddof=1) / np.sqrt(len(steps))) if len(steps) > 1 else 0.0,
        "mean_final_distance": float(np.nanmean(final_distances)) if final_distances else 0.0,
        "mean_total_reward": float(np.mean(rewards)) if rewards else 0.0,
        "action_freq": action_freq,
        "top_actions": _top_actions(action_freq),
    }


def empty_start_pool_summaries(
    *,
    policy_names: list[str],
    cell: dict[str, Any],
    reason: str,
) -> dict[str, dict[str, Any]]:
    """Return per-policy skipped summaries for structurally empty benchmark cells."""
    out: dict[str, dict[str, Any]] = {}
    for policy_name in policy_names:
        out[policy_name] = {
            "status": "skipped_empty_start_pool",
            "skip_reason": str(reason),
            "policy": policy_name,
            "n_episodes": 0,
            "successes": 0,
            "success_rate": None,
            "success_rate_wilson95_low": None,
            "success_rate_wilson95_high": None,
            "mean_steps": None,
            "se_steps": None,
            "mean_final_distance": None,
            "mean_total_reward": None,
            "action_freq": {},
            "top_actions": [],
            "weighted_action_freq_chronos_spearman": None,
            "ppo_minus_random_delta_pp": None,
            "ppo_minus_greedy_dyn_1_delta_pp": None,
            "n_start_pool": 0,
            **cell,
        }
    return out


class _StableBaselinesPolicy:
    def __init__(self, model: Any, deterministic: bool) -> None:
        self.model = model
        self.deterministic = bool(deterministic)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        action, _ = self.model.predict(z, deterministic=self.deterministic, action_masks=mask)
        return int(np.asarray(action).item())


def _chronos_series() -> Any | None:
    path = Path("data/processed/depmap_k562_chronos.parquet")
    if not path.exists():
        return None
    try:
        import polars as pl
        import pandas as pd

        df = pl.read_parquet(str(path)).to_pandas()
        return pd.Series(df["chronos"].values, index=df["gene_symbol"].astype(str)).dropna()
    except Exception:
        return None


def _make_env(
    *,
    dynamics_model: Any,
    z_ref: np.ndarray,
    epsilon: float,
    n_genes: int,
    max_steps: int,
    start_pool: np.ndarray,
    seed: int,
) -> CellReprogrammingEnv:
    return CellReprogrammingEnv(
        dynamics_model=dynamics_model,
        z_reference_centroid=z_ref,
        epsilon_success=float(epsilon),
        n_genes=int(n_genes),
        max_steps=int(max_steps),
        lambda_sparse=0.05,
        lambda_unc=0.0,
        repeat_mask=True,
        start_state_strategy="random_perturbation",
        distance_metric="l2",
        start_pool_latents=start_pool,
        success_bonus=0.0,
        failure_penalty=0.0,
        seed=seed,
    )


def _policy_names(
    baseline_names: set[str],
    *,
    include_stochastic: bool,
) -> list[str]:
    names = ["ppo_deterministic"]
    if include_stochastic:
        names.append("ppo_stochastic")
    if "random" in baseline_names or "random_uniform_valid" in baseline_names:
        names.append("random_uniform_valid")
    if "always_noop" in baseline_names:
        names.append("always_noop")
    if "greedy_dyn_1" in baseline_names:
        names.append("greedy_dyn_1")
    if "greedy_dyn_1_noop_free" in baseline_names:
        names.append("greedy_dyn_1_noop_free")
    if "ridge_greedy" in baseline_names:
        names.append("ridge_greedy")
    if "mean_delta_greedy" in baseline_names:
        names.append("mean_delta_greedy")
    return names


def _cell_name(k: int, eps_label: str, bin_label: str, held_out: bool) -> str:
    split = "ood" if held_out else "mixed"
    return f"k{k}_eps{eps_label}_bin{bin_label}_split{split}".replace(".", "p")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--vae_dir", required=True)
    parser.add_argument("--dynamics_dir", required=True)
    parser.add_argument("--pairs_dir", default="artifacts/pairs")
    parser.add_argument("--ppo_zip", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 3, 8])
    parser.add_argument("--epsilon_values", nargs="+", default=["p25", "p50"])
    parser.add_argument("--distance_bins", nargs="+", default=["4-6", "6-8", "8-10", "10-12"])
    parser.add_argument("--held_out_genes_only", nargs="+", default=["true", "false"])
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--baselines", default="random,always_noop,greedy_dyn_1,ridge_greedy,mean_delta_greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_stochastic", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vae_dir = Path(args.vae_dir)
    pairs_dir = Path(args.pairs_dir)

    if args.smoke:
        args.k_values = [3]
        args.epsilon_values = ["p25"]
        args.distance_bins = ["8-10"]
        args.held_out_genes_only = ["true"]
        args.n_episodes = int(args.n_episodes or 20)

    with open(vae_dir / "gene_vocab.json") as f:
        vocab = json.load(f)
    genes = [str(g) for g in vocab["genes"]]
    n_genes = int(vocab["n_genes"])
    noop_idx = int(vocab["noop_idx"])
    gene_lookup = {i: g for i, g in enumerate(genes)}
    gene_lookup[noop_idx] = "NO_OP"
    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)
    n_latent = int(z_ref.shape[0])

    eps_cache: dict[str, float] = {}
    for eps_label in args.epsilon_values:
        if eps_label.startswith("p"):
            eps_cache[eps_label] = float(compute_epsilon_percentile(vae_dir, float(eps_label[1:]))["value"])
        else:
            eps_cache[eps_label] = float(eps_label)

    with open(pairs_dir / "metadata.json") as f:
        pairs_meta = json.load(f)
    held_out_genes = [str(g) for g in pairs_meta.get("held_out_genes", [])]

    dynamics_model = load_dynamics_model(args.dynamics_dir)
    ridge_path = _fit_and_save_ridge_buffers(pairs_dir, out_dir, n_genes)
    mean_delta_table = _build_mean_delta_table(pairs_dir, n_genes, n_latent)

    from sb3_contrib import MaskablePPO

    ppo_model = MaskablePPO.load(str(args.ppo_zip), device="cpu")
    baseline_names = set(_parse_csv(args.baselines))
    chronos = _chronos_series()
    rows_for_table: list[dict[str, Any]] = []

    metadata = {
        "stage": "p0a_rl_hard",
        "config_name": args.config_name,
        "git_commit": _git_commit(),
        "read_only_mode": True,
        "smoke": bool(args.smoke),
        "source_paths": {
            "vae_dir": str(args.vae_dir),
            "dynamics_dir": str(args.dynamics_dir),
            "pairs_dir": str(args.pairs_dir),
            "ppo_zip": str(args.ppo_zip),
        },
        "matrix": {
            "k_values": args.k_values,
            "epsilon_values": args.epsilon_values,
            "distance_bins": args.distance_bins,
            "held_out_genes_only": args.held_out_genes_only,
            "n_episodes": int(args.n_episodes),
            "baselines": sorted(baseline_names),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    for k in args.k_values:
        for eps_label in args.epsilon_values:
            epsilon = eps_cache[eps_label]
            for distance_bin in _parse_bins(args.distance_bins):
                for held_out in _parse_bool_list(args.held_out_genes_only):
                    cell = _cell_name(k, eps_label, distance_bin[2], held_out)
                    cell_dir = out_dir / cell
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    cell_meta = {
                        "k": int(k),
                        "epsilon_label": eps_label,
                        "epsilon_value": float(epsilon),
                        "distance_bin": distance_bin[2],
                        "gene_split": "ood" if held_out else "mixed",
                    }
                    policy_name_list = _policy_names(
                        baseline_names,
                        include_stochastic=bool(args.include_stochastic),
                    )
                    try:
                        start_pool = _load_start_pool(
                            vae_dir,
                            distance_bin=distance_bin,
                            held_out_genes_only=held_out,
                            held_out_genes=held_out_genes,
                        )
                    except ValueError as exc:
                        cell_summaries = empty_start_pool_summaries(
                            policy_names=policy_name_list,
                            cell=cell_meta,
                            reason=str(exc),
                        )
                        for policy_name, summary in cell_summaries.items():
                            policy_dir = cell_dir / policy_name
                            policy_dir.mkdir(parents=True, exist_ok=True)
                            (policy_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
                            rows_for_table.append({
                                "cell": cell,
                                "policy": policy_name,
                                "success_rate": None,
                                "mean_steps": None,
                                "mean_final_distance": None,
                                "ppo_minus_random_delta_pp": None,
                                "ppo_minus_greedy_dyn_1_delta_pp": None,
                            })
                        print(f"{cell}: skipped empty start pool ({exc})")
                        continue

                    policies: dict[str, Any] = {
                        "ppo_deterministic": _StableBaselinesPolicy(ppo_model, deterministic=True),
                    }
                    if args.include_stochastic:
                        policies["ppo_stochastic"] = _StableBaselinesPolicy(ppo_model, deterministic=False)
                    if "random" in baseline_names or "random_uniform_valid" in baseline_names:
                        policies["random_uniform_valid"] = RandomUniformValidPolicy(seed=args.seed)
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
                    if "ridge_greedy" in baseline_names:
                        policies["ridge_greedy"] = RidgeGreedyPolicy.from_npz(ridge_path, z_ref=z_ref, noop_idx=noop_idx)
                    if "mean_delta_greedy" in baseline_names:
                        policies["mean_delta_greedy"] = MeanDeltaGreedyPolicy(
                            mean_delta_table, z_ref=z_ref, noop_idx=noop_idx,
                        )

                    cell_summaries: dict[str, dict[str, Any]] = {}
                    for policy_name, policy in policies.items():
                        env = _make_env(
                            dynamics_model=dynamics_model,
                            z_ref=z_ref,
                            epsilon=epsilon,
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
                        summary.update({
                            "policy": policy_name,
                            **cell_meta,
                            "n_start_pool": int(len(start_pool)),
                            "status": "ok",
                        })
                        if chronos is not None:
                            summary["weighted_action_freq_chronos_spearman"] = action_freq_chronos_spearman(
                                summary["action_freq"], chronos, seed=args.seed, n_boot=1000,
                            )
                        else:
                            summary["weighted_action_freq_chronos_spearman"] = None
                        cell_summaries[policy_name] = summary

                    random_rate = cell_summaries.get("random_uniform_valid", {}).get("success_rate")
                    greedy_rate = cell_summaries.get("greedy_dyn_1", {}).get("success_rate")
                    for policy_name, summary in cell_summaries.items():
                        summary["ppo_minus_random_delta_pp"] = None
                        summary["ppo_minus_greedy_dyn_1_delta_pp"] = None
                        if policy_name.startswith("ppo"):
                            if random_rate is not None:
                                summary["ppo_minus_random_delta_pp"] = float(100.0 * (summary["success_rate"] - random_rate))
                            if greedy_rate is not None:
                                summary["ppo_minus_greedy_dyn_1_delta_pp"] = float(100.0 * (summary["success_rate"] - greedy_rate))
                        policy_dir = cell_dir / policy_name
                        policy_dir.mkdir(parents=True, exist_ok=True)
                        (policy_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
                        rows_for_table.append({
                            "cell": cell,
                            "policy": policy_name,
                            "success_rate": summary["success_rate"],
                            "mean_steps": summary["mean_steps"],
                            "mean_final_distance": summary["mean_final_distance"],
                            "ppo_minus_random_delta_pp": summary["ppo_minus_random_delta_pp"],
                            "ppo_minus_greedy_dyn_1_delta_pp": summary["ppo_minus_greedy_dyn_1_delta_pp"],
                        })
                    print(f"{cell}: wrote {len(cell_summaries)} policy summaries")

    table_lines = [
        "# V2 hard benchmark results",
        "",
        "| cell | policy | success_rate | mean_steps | mean_final_distance | PPO-random pp | PPO-greedy pp |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_for_table:
        dr = row["ppo_minus_random_delta_pp"]
        dg = row["ppo_minus_greedy_dyn_1_delta_pp"]
        sr = "-" if row["success_rate"] is None else f"{row['success_rate']:.3f}"
        ms = "-" if row["mean_steps"] is None else f"{row['mean_steps']:.2f}"
        fd = "-" if row["mean_final_distance"] is None else f"{row['mean_final_distance']:.3f}"
        drs = "-" if dr is None else f"{dr:.1f}"
        dgs = "-" if dg is None else f"{dg:.1f}"
        table_lines.append(
            f"| {row['cell']} | {row['policy']} | {sr} | {ms} | {fd} | {drs} | {dgs} |"
        )
    (out_dir / "results_table.md").write_text("\n".join(table_lines) + "\n")
    print(f"Wrote hard benchmark outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
