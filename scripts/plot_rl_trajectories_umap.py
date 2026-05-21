#!/usr/bin/env python3
"""
plot_rl_trajectories_umap.py — Final-model RL trajectories on VAE/scVI latent UMAP.

What this script does (in order):
  1. Reads `artifacts_v3/v3c/final_champion_manifest.json` to identify the FINAL MODEL
     (contraction_aware_v2_aggressive + PPO_BCD seed 42 500k).
  2. Validates that ALL required artifacts exist on disk. Prints exactly which is missing
     if any. Exits with a clear error.
  3. Loads the 64-D scVI latents (background cells).
  4. Loads the champion's rollouts.parquet. If it does NOT exist, generates it by running
     scripts/evaluate_rl.py against the champion PPO + champion dynamics + 64-D VAE.
  5. Fits ONE UMAP on the background latents (n=10 000 subsample for speed, fixed seed).
  6. Projects every trajectory's per-step latent z_vector into the SAME UMAP.
  7. Renders a publication-quality figure: grey background, green successful paths,
     red failed paths, yellow star for z_ref, start markers, arrows, title with version,
     legend, success rate.

Usage (the teammate runs this once after `make setup`):
    python scripts/plot_rl_trajectories_umap.py
    # or with options:
    python scripts/plot_rl_trajectories_umap.py --n-episodes 200 --max-traj 60

Outputs:
    artifacts_v3/v3c/figures/final_model_trajectories_umap.png
    artifacts_v3/v3c/figures/final_model_trajectories_umap.pdf
    artifacts_v3/v3c/figures/final_model_trajectories_umap_metadata.json
    artifacts_v3/v3c/figures/final_model_trajectories_rollouts.parquet   (if regenerated)
    artifacts_v3/v3c/figures/final_model_umap_coords.npz                 (cached UMAP)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Repo root: this script lives in scripts/, so root is its parent.
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artifacts_v3" / "v3c" / "final_champion_manifest.json"
FIG_DIR = REPO / "artifacts_v3" / "v3c" / "figures"


# ─── error helpers ─────────────────────────────────────────────────────────

def die(msg: str, code: int = 2) -> None:
    print(f"\n❌ ERROR: {msg}\n", file=sys.stderr)
    sys.exit(code)


def banner(msg: str) -> None:
    line = "─" * len(msg)
    print(f"\n{line}\n{msg}\n{line}")


# ─── champion identity ─────────────────────────────────────────────────────

@dataclass
class Champion:
    name: str
    type: str
    dynamics_dir: Path
    vae_dir: Path
    pairs_dir: Path
    ppo_checkpoint: Path
    reward_mode: str
    epsilon_value: float
    epsilon_label: str
    n_latent: int
    max_steps: int
    lambda_tox: float
    lambda_ce: float
    lambda_unc_path: float
    freeband: dict
    git_commit: str | None

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "Champion":
        if not manifest_path.exists():
            die(f"Champion manifest not found: {manifest_path}\n"
                f"This script requires the V3C closeout artifacts to be on disk.")
        m = json.loads(manifest_path.read_text())
        c = m["champion"]
        cfg = c["dynamics_training_config"]
        rc = c["reward_coefficients"]
        # ε for the champion's eval at the discriminating K=3/bin 8-10/OOD cell:
        # per manifest, Track L 64-D ε p15 = 3.0193 (also used for champion).
        # Override via --epsilon if needed.
        # We *prefer* an explicit field on the manifest if present; otherwise use 3.0193
        # (documented per-VAE p15 for Track L / champion in
        # artifacts_v3/v3c/interpretation/v3c_phase1_ppo_smoke_summary.md §2).
        eps_default = 3.0193
        eps = float(c.get("epsilon_value", eps_default))
        return cls(
            name=c["champion_name"],
            type=c["champion_type"],
            dynamics_dir=(REPO / c["dynamics_dir"]).resolve(),
            vae_dir=(REPO / c["vae_dir"]).resolve(),
            pairs_dir=(REPO / c["pairs_dir"]).resolve(),
            ppo_checkpoint=(REPO / c["ppo_checkpoint"]).resolve(),
            reward_mode=c.get("reward_mode", "biorealistic_fused"),
            epsilon_value=eps,
            epsilon_label="p15",
            n_latent=int(cfg.get("n_latent", 64)),
            max_steps=int(c.get("env_max_steps", 8)),
            lambda_tox=float(rc.get("lambda_tox", 0.10)),
            lambda_ce=float(rc.get("lambda_ce", 0.05)),
            lambda_unc_path=float(rc.get("lambda_unc_path", 0.05)),
            freeband=dict(rc.get("freeband", {})),
            git_commit=m.get("git_commit_at_generation"),
        )


def validate_artifacts(champ: Champion) -> None:
    """Hard validation: every required file/dir must exist on disk."""
    required = [
        ("Champion dynamics directory", champ.dynamics_dir),
        ("Champion dynamics weights",    champ.dynamics_dir / "model.pt"),
        ("Champion dynamics config",     champ.dynamics_dir / "config.json"),
        ("VAE directory (64-D, Track L)", champ.vae_dir),
        ("VAE latents",                  champ.vae_dir / "latents.h5ad"),
        ("VAE gene vocab",               champ.vae_dir / "gene_vocab.json"),
        ("VAE z_reference centroid",     champ.vae_dir / "z_reference_centroid.npy"),
        ("Pairs directory",              champ.pairs_dir),
        ("PPO checkpoint",               champ.ppo_checkpoint),
    ]
    missing = [(label, p) for label, p in required if not p.exists()]
    if missing:
        banner("MISSING ARTIFACTS")
        print("This machine does not have the full champion artifacts on disk.")
        print("The following files/directories are required and missing:\n")
        for label, p in missing:
            print(f"  ✗ {label}")
            print(f"      expected at: {p}")
        print("\nRecommended fix:")
        print("  • If you trained the champion locally, run `make pipeline` to regenerate.")
        print("  • If artifacts live on the cluster, transfer them locally first.")
        print(f"  • Or supply --vae-dir / --dynamics-dir / --ppo-zip overrides if your")
        print(f"    layout differs from the manifest.\n")
        die("artifact validation failed")
    banner("Artifact validation OK")
    for label, p in required:
        print(f"  ✓ {label}: {p.relative_to(REPO)}")


# ─── rollouts generation (only if not already on disk) ─────────────────────

def find_or_generate_rollouts(
    champ: Champion,
    out_dir: Path,
    n_episodes: int,
    min_start_distance: float,
    bin_max_distance: float,
    cell_label: str,
    force_regen: bool,
) -> Path:
    """Return the path to a rollouts.parquet with the champion's PPO + dynamics + VAE.

    Search order:
      1. <champion ppo dir>/eval/rollouts.parquet
      2. <champion ppo dir>/eval_deterministic/rollouts.parquet
      3. <champion ppo dir>/eval_*/rollouts.parquet  (any)
      4. Custom out_dir override
    If none exist (or --force-regen) → call scripts/evaluate_rl.py to produce them.
    """
    ppo_dir = champ.ppo_checkpoint.parent
    candidates = [
        ppo_dir / "eval" / "rollouts.parquet",
        ppo_dir / "eval_deterministic" / "rollouts.parquet",
        ppo_dir / "eval_stochastic" / "rollouts.parquet",
    ]
    # also any eval_* subdir
    for sub in ppo_dir.glob("eval*"):
        cand = sub / "rollouts.parquet"
        if cand not in candidates:
            candidates.append(cand)

    if not force_regen:
        for cand in candidates:
            if cand.exists():
                print(f"  ✓ Using existing rollouts: {cand.relative_to(REPO)}")
                return cand

    banner("Generating rollouts via scripts/evaluate_rl.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "rollouts.parquet"

    # Use evaluate_rl.py because it preserves z_vector in the parquet schema.
    # We point paths.vae_dir, paths.dynamics_dir, paths.pairs_dir to the champion's,
    # then override eval_rl.{ppo_path, out_dir, n_episodes, min_start_distance, epsilon_override}.
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "evaluate_rl.py"),
        "--config-name", "default",
        "rl.train.skip_gate=true",
        f"paths.vae_dir={champ.vae_dir}",
        f"paths.dynamics_dir={champ.dynamics_dir}",
        f"paths.pairs_dir={champ.pairs_dir}",
        f"+eval_rl.ppo_path={champ.ppo_checkpoint}",
        f"+eval_rl.out_dir={out_dir}",
        f"+eval_rl.n_episodes={n_episodes}",
        f"+eval_rl.min_start_distance={min_start_distance}",
        f"+eval_rl.epsilon_override={champ.epsilon_value}",
        f"rl.env.max_steps={champ.max_steps}",
        f"rl.reward.mode={champ.reward_mode}",
        f"rl.reward.lambda_tox={champ.lambda_tox}",
        f"rl.reward.lambda_ce={champ.lambda_ce}",
        f"rl.reward.lambda_unc_path={champ.lambda_unc_path}",
        f"+eval_rl.modes=[deterministic]",
    ]
    print("$ " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        die(f"evaluate_rl.py exited with code {rc}.\n"
            f"Hint: verify the dynamics ↔ VAE compatibility (n_latent must match), "
            f"and that pairs_dir matches the VAE used.")
    # evaluate_rl.py writes per-mode subdirs
    for sub in ["", "deterministic", "eval_deterministic"]:
        cand = out_dir / sub / "rollouts.parquet" if sub else out_dir / "rollouts.parquet"
        if cand.exists():
            return cand
    die(f"evaluate_rl.py finished but no rollouts.parquet found under {out_dir}.")


# ─── UMAP fitting + trajectory projection ──────────────────────────────────

def load_background_latents(vae_dir: Path, n_sample: int, seed: int):
    """Return (Z_bg, perturbation_labels, n_total). Subsamples if needed."""
    import anndata as ad
    print(f"  Loading latents from {vae_dir.relative_to(REPO)}/latents.h5ad ...")
    adata = ad.read_h5ad(vae_dir / "latents.h5ad")
    if "X_scVI" in adata.obsm:
        Z = np.asarray(adata.obsm["X_scVI"])
    elif "X_scVI_64" in adata.obsm:
        Z = np.asarray(adata.obsm["X_scVI_64"])
    else:
        # fallback: try adata.X if it looks like a latent (low dim, dense float)
        die("Could not find scVI latents in .obsm of latents.h5ad")
    print(f"  Total cells: {Z.shape[0]}  ·  latent dim: {Z.shape[1]}")
    # subsample for UMAP fit speed
    if Z.shape[0] > n_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(Z.shape[0], size=n_sample, replace=False)
        Z = Z[idx]
        adata_sub = adata[idx]
    else:
        adata_sub = adata
    pert = (adata_sub.obs["perturbation"].astype(str).values
            if "perturbation" in adata_sub.obs.columns
            else np.array(["?"] * Z.shape[0]))
    return Z, pert, adata.shape[0]


def fit_umap(Z: np.ndarray, seed: int):
    import umap
    print(f"  Fitting UMAP (n_neighbors=30, min_dist=0.30, seed={seed}) ...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.30, n_components=2,
                        metric="euclidean", random_state=seed)
    emb = reducer.fit_transform(Z)
    return reducer, emb


def project_trajectories(reducer, rollouts_path: Path, max_traj: int, seed: int):
    """Read champion rollouts.parquet → project per-step z_vector via the SAME UMAP."""
    import polars as pl
    df = pl.read_parquet(rollouts_path)
    print(f"  rollouts shape: {df.shape}, columns: {df.columns}")
    required_cols = {"episode_id", "step", "z_vector", "success"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        die(f"rollouts.parquet is missing columns: {sorted(missing_cols)}.\n"
            f"Expected the Contract-4 schema written by scripts/evaluate_rl.py / train_ppo.py.\n"
            f"Did this file come from evaluate_rl_v3b_phase4.py? That script does not save z_vector.")
    # success is per-step terminal; use the LAST step's success as the episode label
    last = (df.sort(["episode_id", "step"])
              .group_by("episode_id", maintain_order=True)
              .agg(pl.col("success").last().alias("ep_success"),
                   pl.col("step").max().alias("n_steps")))
    ep_ids_all = last["episode_id"].to_list()
    ep_success = dict(zip(ep_ids_all, last["ep_success"].to_list()))

    n_eps = len(ep_ids_all)
    n_succ = sum(ep_success.values())
    print(f"  Episodes: {n_eps}  ·  successes: {n_succ}  ·  rate: {n_succ/n_eps:.3f}")

    # pick a stratified random subset (some success, some fail) for visualization
    rng = np.random.default_rng(seed)
    succ_ids = [e for e, s in ep_success.items() if s]
    fail_ids = [e for e, s in ep_success.items() if not s]
    half = max_traj // 2
    pick_succ = rng.choice(succ_ids, size=min(half, len(succ_ids)), replace=False).tolist()
    pick_fail = rng.choice(fail_ids, size=min(max_traj - len(pick_succ), len(fail_ids)),
                           replace=False).tolist()
    keep = set(pick_succ) | set(pick_fail)
    sub = df.filter(pl.col("episode_id").is_in(list(keep))).sort(["episode_id", "step"])
    print(f"  Visualizing {len(keep)} trajectories  "
          f"({len(pick_succ)} success, {len(pick_fail)} fail).")

    # Project z_vector via the SAME reducer
    trajs = []  # list of dicts: {ep, success, n_steps, xs (np N×2)}
    for ep_id in keep:
        ep_rows = sub.filter(pl.col("episode_id") == ep_id)
        if ep_rows.height == 0:
            continue
        Z_traj = np.array(ep_rows["z_vector"].to_list(), dtype=np.float32)
        if Z_traj.ndim != 2 or Z_traj.shape[1] != reducer.embedding_.shape[1] * 0 + reducer._raw_data.shape[1]:
            # dimension mismatch
            die(f"Trajectory z_vector dim {Z_traj.shape[1]} ≠ VAE latent dim "
                f"{reducer._raw_data.shape[1]}. The PPO must be evaluated on the same VAE.")
        xy = reducer.transform(Z_traj)
        trajs.append({
            "ep": ep_id,
            "success": bool(ep_success[ep_id]),
            "n_steps": ep_rows.height,
            "xy": xy,
        })
    return trajs, n_eps, n_succ


def project_zref(reducer, vae_dir: Path) -> np.ndarray:
    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)
    xy = reducer.transform(z_ref.reshape(1, -1))
    return xy[0]


# ─── rendering ─────────────────────────────────────────────────────────────

def render(emb_bg, trajs, z_ref_xy, n_eps, n_succ, champ: Champion,
           out_png: Path, out_pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=200)

    # Background cloud
    ax.scatter(emb_bg[:, 0], emb_bg[:, 1],
               s=4, c="#BBBBBB", alpha=0.25, linewidths=0, zorder=1,
               rasterized=True)

    # Sort trajectories: failures first (drawn under), successes on top
    trajs_sorted = sorted(trajs, key=lambda t: t["success"])

    for t in trajs_sorted:
        xy = t["xy"]
        if xy.shape[0] < 1:
            continue
        col = "#1E823C" if t["success"] else "#B5371E"  # green vs red
        alpha = 0.85 if t["success"] else 0.7
        # line
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=col, alpha=alpha,
                    linewidth=1.4, zorder=3 if t["success"] else 2)
            # arrow on last segment
            x0, y0 = xy[-2]
            x1, y1 = xy[-1]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=col,
                                        lw=1.2, alpha=alpha),
                        zorder=4)
        # start marker
        ax.scatter(xy[0, 0], xy[0, 1], s=20, marker="o",
                   facecolor="white", edgecolor=col, linewidth=1.2, zorder=5)

    # z_ref star
    ax.scatter(z_ref_xy[0], z_ref_xy[1], s=380, marker="*",
               facecolor="#FFD12F", edgecolor="black", linewidth=1.4, zorder=10,
               label="target  z_ref  (unperturbed K562 centroid)")

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color="#1E823C", lw=2, label="successful trajectory"),
        plt.Line2D([0], [0], color="#B5371E", lw=2, label="failed trajectory"),
        plt.Line2D([0], [0], marker="o", lw=0, markeredgecolor="black",
                   markerfacecolor="white", markersize=7,
                   label="episode start"),
        plt.Line2D([0], [0], marker="*", lw=0, markeredgecolor="black",
                   markerfacecolor="#FFD12F", markersize=16,
                   label="target  z_ref"),
        plt.Line2D([0], [0], marker="o", lw=0, markeredgecolor="none",
                   markerfacecolor="#BBBBBB", markersize=6,
                   label="background cells (UMAP)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True,
              fontsize=10, framealpha=0.92)

    sr = n_succ / n_eps if n_eps else 0.0
    ax.set_title(
        f"Final model RL trajectories on scVI 64-D latent UMAP\n"
        f"contraction_aware_v2_aggressive  +  PPO_BCD seed 42 500k   ·   "
        f"N = {n_eps} episodes   ·   success rate = {sr*100:.1f}%",
        fontsize=12, pad=14,
    )
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Wrote figure  →  {out_png.relative_to(REPO)}")
    print(f"  ✓ Wrote figure  →  {out_pdf.relative_to(REPO)}")


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-episodes",         type=int,   default=300,
                    help="episodes to roll out IF rollouts.parquet must be regenerated (default 300)")
    ap.add_argument("--max-traj",           type=int,   default=80,
                    help="max trajectories to visually overlay (default 80)")
    ap.add_argument("--n-bg",               type=int,   default=10_000,
                    help="background cells to subsample for UMAP fitting (default 10 000)")
    ap.add_argument("--seed",               type=int,   default=42,
                    help="UMAP + sampling seed (default 42)")
    ap.add_argument("--min-start-distance", type=float, default=8.0,
                    help="min L2 distance from z_ref for start states (default 8.0 → bin 8-10)")
    ap.add_argument("--bin-max-distance",   type=float, default=10.0,
                    help="max L2 distance (default 10.0)")
    ap.add_argument("--cell-label",         type=str,   default="k3_bin8-10_OOD",
                    help="evaluation cell label (purely cosmetic in title)")
    ap.add_argument("--force-regen",        action="store_true",
                    help="ignore existing rollouts.parquet and rebuild from PPO checkpoint")
    ap.add_argument("--out-dir",            type=str,   default=str(FIG_DIR),
                    help=f"output figure directory (default {FIG_DIR.relative_to(REPO)})")
    args = ap.parse_args()

    banner("CellPath — final-model RL trajectories on VAE latent UMAP")

    # 1. champion identity from manifest
    champ = Champion.from_manifest(MANIFEST)
    print(f"  Champion: {champ.name}")
    print(f"    type   : {champ.type}")
    print(f"    vae    : {champ.vae_dir.relative_to(REPO)}  (n_latent={champ.n_latent})")
    print(f"    dynamics: {champ.dynamics_dir.relative_to(REPO)}")
    print(f"    ppo    : {champ.ppo_checkpoint.relative_to(REPO)}")
    print(f"    ε      : {champ.epsilon_value:.4f}  ({champ.epsilon_label})")
    print(f"    reward : {champ.reward_mode}  ·  max_steps={champ.max_steps}")

    # 2. validate
    validate_artifacts(champ)

    # 3. rollouts (find or regenerate)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh_eval_dir = out_dir.parent / "rl_smokes" / f"{champ.name}__final_traj_eval"
    rollouts_path = find_or_generate_rollouts(
        champ=champ,
        out_dir=fresh_eval_dir,
        n_episodes=args.n_episodes,
        min_start_distance=args.min_start_distance,
        bin_max_distance=args.bin_max_distance,
        cell_label=args.cell_label,
        force_regen=args.force_regen,
    )

    # 4. background latents
    banner("Loading background latents")
    Z_bg, pert_bg, n_total = load_background_latents(champ.vae_dir, args.n_bg, args.seed)

    # 5. UMAP fit
    banner("Fitting UMAP on background")
    t0 = time.time()
    reducer, emb_bg = fit_umap(Z_bg, args.seed)
    print(f"  UMAP fit in {time.time()-t0:.1f}s")

    # 6. project trajectories via the SAME reducer
    banner("Projecting trajectories")
    trajs, n_eps, n_succ = project_trajectories(reducer, rollouts_path, args.max_traj, args.seed)

    # 7. project z_ref
    z_ref_xy = project_zref(reducer, champ.vae_dir)
    print(f"  z_ref @ UMAP: ({z_ref_xy[0]:.2f}, {z_ref_xy[1]:.2f})")

    # 8. render
    banner("Rendering figure")
    out_png = Path(args.out_dir) / "final_model_trajectories_umap.png"
    out_pdf = Path(args.out_dir) / "final_model_trajectories_umap.pdf"
    render(emb_bg, trajs, z_ref_xy, n_eps, n_succ, champ, out_png, out_pdf)

    # 9. cache UMAP coords + metadata for reproducibility
    np.savez(Path(args.out_dir) / "final_model_umap_coords.npz",
             emb_bg=emb_bg, z_ref_xy=z_ref_xy)
    meta = {
        "champion_name": champ.name,
        "champion_type": champ.type,
        "vae_dir":       str(champ.vae_dir.relative_to(REPO)),
        "dynamics_dir":  str(champ.dynamics_dir.relative_to(REPO)),
        "ppo_checkpoint": str(champ.ppo_checkpoint.relative_to(REPO)),
        "rollouts_used": str(rollouts_path.relative_to(REPO)),
        "n_total_cells": int(n_total),
        "n_bg_subsample": int(emb_bg.shape[0]),
        "n_episodes": int(n_eps),
        "n_successes": int(n_succ),
        "success_rate": n_succ / n_eps if n_eps else 0.0,
        "n_trajectories_drawn": len(trajs),
        "umap_seed": int(args.seed),
        "epsilon_value": champ.epsilon_value,
        "epsilon_label": champ.epsilon_label,
        "max_steps": champ.max_steps,
        "reward_mode": champ.reward_mode,
        "lambda_tox": champ.lambda_tox,
        "lambda_ce": champ.lambda_ce,
        "lambda_unc_path": champ.lambda_unc_path,
        "min_start_distance": args.min_start_distance,
        "bin_max_distance": args.bin_max_distance,
        "git_commit_at_generation": champ.git_commit,
    }
    (Path(args.out_dir) / "final_model_trajectories_umap_metadata.json").write_text(
        json.dumps(meta, indent=2))

    # 10. summary
    banner("DONE")
    print(f"  Champion           : {champ.name}")
    print(f"  Episodes           : {n_eps}")
    print(f"  Successes          : {n_succ}  ({n_succ/n_eps*100:.1f}%)")
    print(f"  Failures           : {n_eps - n_succ}")
    print(f"  Trajectories drawn : {len(trajs)}")
    print(f"  Background cells   : {n_total} total  →  {emb_bg.shape[0]} subsampled for UMAP")
    print(f"")
    print(f"  Figure (PNG)       : {out_png}")
    print(f"  Figure (PDF)       : {out_pdf}")
    print(f"  Metadata JSON      : {Path(args.out_dir) / 'final_model_trajectories_umap_metadata.json'}")


if __name__ == "__main__":
    main()
