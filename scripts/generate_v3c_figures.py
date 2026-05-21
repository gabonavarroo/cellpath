"""V3C figure generator — reads aggregated audit / eval JSON+CSV, writes PNG/SVG.

Outputs land under ``artifacts_v3/v3c/figures/``. All figures are reproducible
from existing aggregator outputs; no retraining, no PPO rollout invocations.

Figures generated:
    pipeline_overview                       — text-only block diagram (PNG)
    dynamics_pathology_summary              — three-pathology landscape
    contraction_geometry_comparison         — gu_max / cf / alignment across fields
    phase4_track_ln_results                 — 4-seed paired-Δ bars + CIs
    final_leaderboard                       — primary cells × top candidates
    reward_stack_bucketA                    — tox / CE / unc improvements
    rl_path_followed                        — sample successful trajectory snapshot
    latent_space                            — UMAP of K562 control + perturbed + z_ref
    training_error_curves                   — train/val NLL + gate margin per epoch
    phase2_5_geometry_move                  — v1 / v2 / v3 / v4 vs Track L
    util_vs_reach_scatter                   — composite util_score vs K=2/b8-10 reach

Use ``python scripts/generate_v3c_figures.py <name>|all``.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "artifacts_v3/v3c/figures"
AUDIT_ROOT = REPO_ROOT / "artifacts_v3/v3c/utility_audit"

LOG = logging.getLogger("v3c.figures")


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    LOG.info("wrote %s", out)
    plt.close(fig)
    return out


def fig_pipeline_overview() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    blocks = [
        ("Norman 2019\nK562 CRISPRa\nPerturb-seq", 0.04, "#FDE2E4"),
        ("scVI VAE\n→ 32D or 64D latent\n(z_ctrl, z_pert, z_ref)", 0.21, "#FAD2E1"),
        ("OT pairing\nz_ctrl ↔ z_pert\nper held-out gene", 0.38, "#E2ECE9"),
        ("Residual MLP\ndynamics f̂(z, g)\n+ contraction-aware\nregularizers", 0.55, "#BEE1E6"),
        ("MaskablePPO\nlocked B+C+D\nreward stack", 0.73, "#CDDAFD"),
        ("7-cell eval\n4-seed CI\nvs reward-aware\ngreedy_dyn_K", 0.90, "#DFE7FD"),
    ]
    for txt, x, color in blocks:
        rect = mpatches.FancyBboxPatch(
            (x - 0.06, 0.35), 0.12, 0.30,
            boxstyle="round,pad=0.02", linewidth=1.0,
            facecolor=color, edgecolor="#222",
        )
        ax.add_patch(rect)
        ax.text(x, 0.50, txt, ha="center", va="center", fontsize=8.5)
    for i in range(len(blocks) - 1):
        x0 = blocks[i][1] + 0.06
        x1 = blocks[i + 1][1] - 0.06
        ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"))
    ax.text(0.5, 0.93, "CellPath V3C — End-to-end pipeline",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.85,
            "In-silico latent-space steering of K562 CRISPRa perturbed cells back toward "
            "the unperturbed-K562 centroid.\n"
            "V3C audits all dynamics fields; V3B locks the reward stack; "
            "Phase 2 / 2.5 tries to break the universal-attractor bottleneck.",
            ha="center", fontsize=9, color="#444")
    ax.text(0.5, 0.05, "Each block is reproducible from CLAUDE.md / RUN_FINAL_PIPELINE.md.",
            ha="center", fontsize=8, color="#666", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return _save(fig, "pipeline_overview")


def _load_aggregated_contraction() -> pl.DataFrame:
    """Read contraction_geometry.csv (OOD pool sample)."""
    p = AUDIT_ROOT / "contraction_geometry.csv"
    if not p.exists():
        raise FileNotFoundError(f"Aggregator output missing: {p}")
    return pl.read_csv(p)


def _load_aggregated_reach() -> pl.DataFrame:
    p = AUDIT_ROOT / "reachability_matrix.csv"
    if not p.exists():
        raise FileNotFoundError(f"Aggregator output missing: {p}")
    return pl.read_csv(p)


def fig_dynamics_pathology_summary() -> Path:
    """Three-pathology landscape: cf, gu_max, align_cos_median across families."""
    df = _load_aggregated_contraction()
    df = df.filter(pl.col("status") == "ok")
    # Pick representative fields
    targets = {
        "V2 anchor (RoR_corr010)": "artifacts_v2__dynamics_v1ot_ror_corr010",
        "Track L (n64 legacy)":    "artifacts_v3__dynamics_n64_legacy_ror_corr010",
        "Track N (n64 NB)":        "artifacts_v3__dynamics_n64_nb_ror_corr010",
        "Soft-OT (cautionary)":    "artifacts_v2__dynamics_soft_ot_default",
        "Random pairs (control)":  "artifacts_v2__dynamics_random_default",
        "mean_delta_corr_010":     "artifacts_v2__dynamics_mean_delta_corr_010",
        "contraction_aware_v1":    "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v1",
    }
    rows = []
    for label, fid in targets.items():
        m = df.filter(pl.col("field_id") == fid)
        if m.height == 0:
            continue
        r = m.to_dicts()[0]
        rows.append({
            "label": label, "cf": r["contraction_fraction"],
            "gu_max": r["gene_universality_max"],
            "align_median": r["alignment_cos_median"],
            "act_div": r["action_diversity_per_state"],
        })
    # Add Phase 2.5 variants if their audits exist
    for tag in ("v2_aggressive", "v3_diverse", "v4_combo"):
        fid = f"artifacts_v3__v3c__dynamics_candidates__contraction_aware_{tag}"
        m = df.filter(pl.col("field_id") == fid)
        if m.height:
            r = m.to_dicts()[0]
            rows.append({
                "label": f"contraction_aware_{tag}",
                "cf": r["contraction_fraction"],
                "gu_max": r["gene_universality_max"],
                "align_median": r["alignment_cos_median"],
                "act_div": r["action_diversity_per_state"],
            })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = [r["label"] for r in rows]
    cf = [r["cf"] for r in rows]
    gu = [r["gu_max"] for r in rows]
    al = [r["align_median"] for r in rows]
    y = np.arange(len(labels))[::-1]
    for ax, vals, name, xref in [
        (axes[0], cf, "contraction_fraction", 0.5),
        (axes[1], gu, "gene_universality_max", 0.5),
        (axes[2], al, "alignment_cos_median", 0.0),
    ]:
        ax.barh(y, vals, color="#4C72B0", alpha=0.85, edgecolor="#222")
        ax.axvline(xref, color="#777", lw=1, ls="--", alpha=0.5)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(name); ax.set_xlim(-1.0 if name == "alignment_cos_median" else 0.0,
                                          1.0 + (0.05 if name != "alignment_cos_median" else 0))
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.suptitle("Dynamics pathology summary — V3C audit U-D geometry across representative fields",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "All OT-trained fields cluster at cf≈1.0 + gu_max≈0.92 (universal over-contraction). "
             "Soft-OT is anti-contractive. mean-delta has lower gu_max but ~0% beam-reach at K≤5.",
             ha="center", fontsize=8, color="#444", style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return _save(fig, "dynamics_pathology_summary")


def fig_contraction_geometry_comparison() -> Path:
    """gu_max vs cf scatter, candidates labeled — geometry landscape."""
    df = _load_aggregated_contraction()
    df = df.filter(pl.col("status") == "ok")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = []
    for fid in df["field_id"]:
        if "v3c__dynamics_candidates__contraction_aware" in fid:
            colors.append("#D62728")    # V3C candidates — red
        elif "soft_ot" in fid:
            colors.append("#9467BD")    # Soft-OT — purple
        elif "random" in fid:
            colors.append("#8C564B")    # random — brown
        elif "n64_legacy" in fid or "n64_nb" in fid:
            colors.append("#2CA02C")    # Track L / N — green
        elif "v1ot_ror_corr010" in fid and "artifacts_v2" in fid:
            colors.append("#FF7F0E")    # V2 anchor — orange
        elif "mean_delta" in fid:
            colors.append("#1F77B4")    # mean-delta — blue
        else:
            colors.append("#7F7F7F")    # other — grey
    ax.scatter(df["contraction_fraction"], df["gene_universality_max"],
               c=colors, s=80, alpha=0.85, edgecolor="#222", linewidth=0.6)
    # Annotate key fields
    annot_ids = {
        "artifacts_v2__dynamics_v1ot_ror_corr010": "V2 anchor",
        "artifacts_v3__dynamics_n64_legacy_ror_corr010": "Track L",
        "artifacts_v3__dynamics_n64_nb_ror_corr010": "Track N",
        "artifacts_v2__dynamics_soft_ot_default": "Soft-OT",
        "artifacts_v2__dynamics_random_default": "Random",
        "artifacts_v2__dynamics_mean_delta_corr_010": "mean-Δ_corr010",
        "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v1": "v1",
        "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v2_aggressive": "v2_aggr",
        "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v3_diverse": "v3_div",
        "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v4_combo": "v4_combo",
    }
    for r in df.to_dicts():
        if r["field_id"] in annot_ids:
            ax.annotate(annot_ids[r["field_id"]],
                        (r["contraction_fraction"], r["gene_universality_max"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=8,
                        color="#222", fontweight="bold")
    ax.set_xlabel("contraction_fraction (1 = all (z,g) contract toward z_ref)")
    ax.set_ylabel("gene_universality_max (1 = one gene dominates contraction)")
    ax.set_title("Contraction-geometry landscape — V3C audit U-D")
    ax.axhspan(0.85, 1.0, color="#FDE2E4", alpha=0.3, label="UNIVERSAL_ATTRACTOR zone (gu_max≥0.85)")
    ax.axvspan(0.95, 1.0, color="#E2ECE9", alpha=0.3)
    ax.text(0.97, 0.97, "Universal over-contraction\n(K≥3 saturated)",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, color="#666", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.text(0.05, 0.05, "Anti-contractive (Soft-OT)\nor low-attractor (mean-Δ)",
            ha="left", va="bottom", transform=ax.transAxes,
            fontsize=8, color="#666", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.grid(alpha=0.3)
    return _save(fig, "contraction_geometry_comparison")


def fig_phase4_track_ln_results() -> Path:
    """Bar chart with 95% CI for Track L / Track N 4-seed paired delta vs greedy_dyn_2."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = [
        "Track L 1M\n(K=2/b6-8)", "Track L 1M\n(K=2/b8-10)",
        "Track N 500k\n(K=2/b6-8)", "Track N 500k\n(K=2/b8-10)",
        "Track N 1M\n(K=2/b6-8)", "Track N 1M\n(K=2/b8-10)",
    ]
    means = [-0.120, +0.010, -0.131, +0.004, -0.141, -0.023]
    lo    = [-0.179, +0.010, -0.154, -0.047, -0.160, -0.117]
    hi    = [-0.061, +0.010, -0.109, +0.055, -0.122, +0.072]
    err_lo = [m - l for m, l in zip(means, lo)]
    err_hi = [h - m for h, m in zip(hi, means)]
    colors = ["#D62728" if m < 0 else ("#2CA02C" if l > 0 else "#888") for m, l in zip(means, lo)]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[err_lo, err_hi], color=colors, alpha=0.85,
           edgecolor="#222", capsize=4)
    ax.axhline(0, color="#222", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ raw success: PPO_BCD − greedy_dyn_2_fused\n(paired by seed, 4 seeds)")
    ax.set_title("Phase 4 — Track L / Track N 4-seed escalation\nNo stable PPO-over-greedy signal at any cell")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.99, 0.97,
            "Verdict: NO_STABLE_SIGNAL\n"
            "Track L K=2/b8-10 ties greedy (+0.010 zero-variance)\n"
            "but mean_final_distance regresses +0.173 → Pareto fail",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, color="#444",
            bbox=dict(facecolor="#FDE2E4", alpha=0.85, edgecolor="#D62728"))
    return _save(fig, "phase4_track_ln_results")


def fig_reward_stack_bucketA() -> Path:
    """V3B Phase 4 Bucket-A: PPO_BCD vs PPO_A on tox / CE / unc."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    cells = ["K=2/b6-8", "K=2/b8-10", "K=3/b6-8", "K=3/b8-10", "K=4-8"]
    tox_a   = [0.001, 0.000, 0.003, 0.002, 0.000]
    tox_bcd = [0.000, 0.000, 0.000, 0.000, 0.000]
    ce_a    = [0.007, 0.000, 0.005, 0.000, 0.000]
    ce_bcd  = [0.000, 0.000, 0.000, 0.000, 0.000]
    unc_a   = [0.644, 0.690, 0.681, 0.744, 0.618]
    unc_bcd = [0.609, 0.638, 0.671, 0.702, 0.609]
    x = np.arange(len(cells)); w = 0.4
    for ax, va, vb, title in [
        (axes[0], tox_a, tox_bcd, "mean_tox_path"),
        (axes[1], ce_a, ce_bcd, "mean_common_essential"),
        (axes[2], unc_a, unc_bcd, "mean_unc_path_max"),
    ]:
        ax.bar(x - w/2, va, w, color="#FF7F0E", alpha=0.85, label="PPO_A (V2 baseline)")
        ax.bar(x + w/2, vb, w, color="#2CA02C", alpha=0.85, label="PPO_BCD (V3B locked)")
        ax.set_xticks(x); ax.set_xticklabels(cells, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("V3B Bucket-A — reward-fit metrics (V2 anchor, 4-seed mean)",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 0.0,
             "PPO_BCD wins all 3 axes — but on V2 dynamics this is reward-prior optimization, not biological discovery.",
             ha="center", fontsize=8, color="#444", style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(fig, "reward_stack_bucketA")


def fig_final_leaderboard() -> Path:
    """Leaderboard of candidates × key metrics."""
    df = _load_aggregated_contraction()
    df = df.filter(pl.col("status") == "ok")
    reach = _load_aggregated_reach().filter(pl.col("status") == "ok")

    targets = [
        ("V2 anchor", "artifacts_v2__dynamics_v1ot_ror_corr010"),
        ("Track L", "artifacts_v3__dynamics_n64_legacy_ror_corr010"),
        ("Track N", "artifacts_v3__dynamics_n64_nb_ror_corr010"),
        ("mean_delta", "artifacts_v2__dynamics_mean_delta_corr_010"),
        ("Soft-OT", "artifacts_v2__dynamics_soft_ot_default"),
        ("contraction_v1", "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v1"),
        ("contraction_v2_aggr", "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v2_aggressive"),
        ("contraction_v3_div", "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v3_diverse"),
        ("contraction_v4_combo", "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v4_combo"),
    ]
    cells = ["k2_bin8-10_splitood", "k2_bin6-8_splitood"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    rows = []
    for label, fid in targets:
        c = df.filter(pl.col("field_id") == fid)
        if c.height == 0:
            continue
        r = c.to_dicts()[0]
        per_cell = {}
        for cn in cells:
            rc = reach.filter((pl.col("field_id") == fid) & (pl.col("cell_id") == cn))
            per_cell[cn] = float(rc["beam_reach_at_K_p15"][0]) if rc.height else None
        rows.append({"label": label, **per_cell,
                     "gu_max": r["gene_universality_max"], "cf": r["contraction_fraction"]})
    # Table-style figure
    headers = ["Field", "K=2/b6-8 reach", "K=2/b8-10 reach", "gu_max", "cf"]
    table_data = [[
        r["label"],
        f"{r.get('k2_bin6-8_splitood', float('nan')):.3f}" if r.get("k2_bin6-8_splitood") is not None else "—",
        f"{r.get('k2_bin8-10_splitood', float('nan')):.3f}" if r.get("k2_bin8-10_splitood") is not None else "—",
        f"{r['gu_max']:.3f}", f"{r['cf']:.4f}",
    ] for r in rows]
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9.5)
    table.scale(1, 1.6)
    # Header style
    for i, _ in enumerate(headers):
        cell = table[(0, i)]; cell.set_facecolor("#222"); cell.get_text().set_color("white")
    fig.suptitle("V3C final leaderboard — geometry + reachability across all candidates",
                 fontsize=12, fontweight="bold")
    return _save(fig, "final_leaderboard")


def fig_training_error_curves() -> Path:
    """Train/val NLL + gate margin per epoch for the champion (or fallback to Track L)."""
    candidates = [
        REPO_ROOT / "artifacts_v3/v3c/dynamics_candidates/contraction_aware_v3_diverse/epoch_metrics.json",
        REPO_ROOT / "artifacts_v3/v3c/dynamics_candidates/contraction_aware_v2_aggressive/epoch_metrics.json",
        REPO_ROOT / "artifacts_v3/v3c/dynamics_candidates/contraction_aware_v1/epoch_metrics.json",
        REPO_ROOT / "artifacts_v3/dynamics_n64_legacy_ror_corr010/epoch_metrics.json",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        LOG.warning("No epoch_metrics.json found; skipping training curves figure")
        return Path()
    data = json.loads(src.read_text())
    epochs   = [r["epoch"]      for r in data]
    train    = [r["train_nll"]  for r in data]
    val      = [r["val_nll"]    for r in data]
    margins  = [r.get("val_mlp_minus_ridge_pearson") for r in data]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(epochs, train, label="train NLL", color="#4C72B0", lw=1.5)
    ax1.plot(epochs, val,   label="val NLL",   color="#D62728", lw=1.5)
    ax1.set_ylabel("NLL"); ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)
    ax1.set_title(f"Training curves — {src.parent.name}", fontsize=11, fontweight="bold")
    if any(m is not None for m in margins):
        m = [x if x is not None else float("nan") for x in margins]
        ax2.plot(epochs, m, color="#2CA02C", lw=1.4)
        ax2.axhline(0.03, color="#888", ls="--", lw=0.8, label="gate threshold (+0.03)")
        ax2.axhline(0.0, color="#222", lw=0.6)
        ax2.set_ylabel("MLP − ridge Pearson (val)")
        ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)
    ax2.set_xlabel("epoch")
    fig.tight_layout()
    return _save(fig, "training_error_curves")


def fig_latent_space() -> Path:
    """UMAP of K562 control + perturbed + z_ref (uses pre-computed UMAP if available)."""
    import anndata as ad
    for vae in ("artifacts/vae", "artifacts_v3/vae_n64_legacy", "artifacts_v3/vae_n64_nb"):
        p = REPO_ROOT / vae / "latents.h5ad"
        if not p.exists():
            continue
        adata = ad.read_h5ad(p)
        z = np.asarray(adata.obsm.get("X_umap", adata.obsm.get("X_scVI")))[:, :2]
        is_ctrl = adata.obs["perturbation_idx"].values == 0
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(z[~is_ctrl, 0], z[~is_ctrl, 1], s=2, color="#D62728", alpha=0.3, label="perturbed")
        ax.scatter(z[is_ctrl, 0], z[is_ctrl, 1], s=2, color="#4C72B0", alpha=0.6, label="control (z_ref source)")
        z_ref = np.load(REPO_ROOT / vae / "z_reference_centroid.npy")
        if "X_umap" in adata.obsm:
            ax.scatter(z[is_ctrl].mean(0)[0], z[is_ctrl].mean(0)[1], s=120, marker="*",
                       color="#FFD700", edgecolor="#222", linewidth=1.0, label="z_ref centroid (UMAP)")
        ax.set_title(f"K562 latent space — {vae.split('/')[-1]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP-1" if "X_umap" in adata.obsm else "scVI-1")
        ax.set_ylabel("UMAP-2" if "X_umap" in adata.obsm else "scVI-2")
        ax.legend(loc="best", fontsize=9, markerscale=3)
        ax.grid(alpha=0.3)
        return _save(fig, "latent_space")
    LOG.warning("No latents.h5ad found; skipping latent_space figure")
    return Path()


def fig_champion_vs_greedy() -> Path:
    """Champion (v2_aggressive seed 42) vs reward-aware greedy_dyn_K — per-cell bar chart."""
    src = REPO_ROOT / "artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed42_500k/eval/aggregate.csv"
    if not src.exists():
        LOG.warning("Champion eval aggregate not found; skipping champion_vs_greedy")
        return Path()
    df = pl.read_csv(src)
    cells_canonical = ["k2_bin6-8_splitood", "k2_bin8-10_splitood",
                       "k3_bin6-8_splitood", "k3_bin8-10_splitood",
                       "k4_bin8-10_splitood", "k5_bin8-10_splitood", "k8_bin8-10_splitood"]
    fig, ax = plt.subplots(figsize=(12, 5))
    cells = [c for c in cells_canonical if df.filter(pl.col("cell") == c).height > 0]
    x = np.arange(len(cells))
    width = 0.18
    series = [
        ("PPO_BCD",            "#D62728", 0),
        ("greedy_dyn_1_fused", "#9ECAE1", 1),
        ("greedy_dyn_2_fused", "#6BAED6", 2),
        ("greedy_dyn_3_fused", "#4292C6", 3),
        ("greedy_dyn_5_fused", "#2171B5", 4),
    ]
    for label, color, slot in series:
        vals = []
        for c in cells:
            row = df.filter((pl.col("cell") == c) & (pl.col("policy") == label))
            vals.append(float(row["success_rate"][0]) if row.height else float("nan"))
        offset = (slot - 2) * width
        ax.bar(x + offset, vals, width, color=color, alpha=0.9, edgecolor="#222",
               label=label.replace("_fused", "").replace("PPO_BCD", "PPO_BCD (champion)"))
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_splitood", "") for c in cells], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Success rate (n=200 episodes)")
    ax.set_title("V3C champion (`contraction_aware_v2_aggressive` + PPO_BCD seed 42 500k)\n"
                 "vs reward-aware greedy_dyn_K_fused across canonical 7-cell V3B matrix",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    # Annotate K=3/b8-10 success
    if "k3_bin8-10_splitood" in cells:
        i = cells.index("k3_bin8-10_splitood")
        ax.annotate("+0.075 over\ngreedy_dyn_3", xy=(i - 0.18*2, 0.84), xytext=(i - 0.5, 1.05),
                    arrowprops=dict(arrowstyle="->", color="#222", lw=1),
                    fontsize=9, color="#222", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF2CC", edgecolor="#222"))
    return _save(fig, "champion_vs_greedy")


def fig_rl_path_followed() -> Path:
    """Show a successful PPO_BCD trajectory: distance over steps for a sample of episodes."""
    # Use Track L Phase 4 seed 42 rollouts if available
    for src in (
        REPO_ROOT / "artifacts_v3/v3c/rl_smokes/track_l_n64_legacy_ror_corr010_seed42_1M/eval/k2_bin8-10_splitood/PPO_BCD_summary.json",
        REPO_ROOT / "artifacts_v3/v3c/rl_smokes/track_l_n64_legacy_ror_corr010_seed42_1M/eval/k2_bin6-8_splitood/PPO_BCD_summary.json",
    ):
        if not src.exists():
            continue
        d = json.loads(src.read_text())
        # Some evaluators store per-episode trajectories; if not, plot final_distances histogram
        if "trajectories" in d:
            fig, ax = plt.subplots(figsize=(8, 5))
            for traj in d["trajectories"][:25]:
                steps = list(range(len(traj["distances"])))
                ax.plot(steps, traj["distances"], lw=0.9, alpha=0.6,
                        color="#4C72B0" if traj.get("success") else "#D62728")
            ax.axhline(d.get("epsilon", 3.0), color="#000", ls="--", lw=1, label="ε threshold")
            ax.set_xlabel("step"); ax.set_ylabel("‖z − z_ref‖₂")
            ax.set_title("Sample PPO_BCD trajectories on Track L (K=2/b8-10/OOD)")
            ax.legend(); ax.grid(alpha=0.3)
            return _save(fig, "rl_path_followed")
        else:
            # Fallback: bar of mean_final_distance vs mean_steps
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.text(0.5, 0.55,
                    f"PPO_BCD at K=2/b8-10/OOD (Track L)\n"
                    f"success_rate = {d.get('success_rate', float('nan')):.3f}\n"
                    f"mean_steps = {d.get('mean_steps', float('nan')):.2f}\n"
                    f"mean_final_distance = {d.get('mean_final_distance', float('nan')):.3f}\n"
                    f"mean_unc_path_max = {d.get('mean_unc_path_max', float('nan')):.3f}",
                    ha="center", va="center", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#FDE2E4", edgecolor="#222"))
            ax.text(0.5, 0.1, "(Per-episode trajectories not stored in this eval format.)",
                    ha="center", fontsize=8, style="italic", color="#666")
            ax.axis("off")
            ax.set_title("PPO_BCD path metrics — Track L sample", fontsize=11, fontweight="bold")
            return _save(fig, "rl_path_followed")
    LOG.warning("No PPO eval summary available for rl_path_followed")
    return Path()


def fig_util_vs_reach_scatter() -> Path:
    """util_score vs K=2/b8-10/OOD reach scatter — argues why util_score isn't the right ranking."""
    reach = _load_aggregated_reach().filter(pl.col("cell_id") == "k2_bin8-10_splitood")
    # util_score lives in candidate_ranking.md or each field's bucket_u_index.json
    rows = []
    for d in (AUDIT_ROOT.iterdir() if AUDIT_ROOT.exists() else []):
        if not d.is_dir():
            continue
        ix = d / "bucket_u_index.json"
        if not ix.exists():
            continue
        try:
            payload = json.loads(ix.read_text())
            util = payload.get("util_score")
            fid = payload.get("field_id", d.name)
            r = reach.filter(pl.col("field_id") == fid)
            if not r.height:
                continue
            rows.append({"fid": fid, "util": util, "reach": float(r["beam_reach_at_K_p15"][0])})
        except (json.JSONDecodeError, KeyError):
            continue
    if not rows:
        LOG.warning("No util_score / reach scatter data")
        return Path()
    fig, ax = plt.subplots(figsize=(8, 6))
    util = [r["util"] for r in rows]
    reach_v = [r["reach"] for r in rows]
    ax.scatter(util, reach_v, s=70, alpha=0.7, color="#4C72B0", edgecolor="#222")
    for r in rows:
        if any(k in r["fid"] for k in ("v1ot_ror_corr010", "n64_legacy", "n64_nb",
                                         "mean_delta_corr_010", "soft_ot", "random",
                                         "contraction_aware")):
            label = r["fid"].split("__")[-1][:20]
            ax.annotate(label, (r["util"], r["reach"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("util_score (composite ranking aid)")
    ax.set_ylabel("K=2/bin8-10/OOD beam reach @ p15")
    ax.set_title("util_score vs reachability — why interpretive selection matters")
    ax.text(0.99, 0.02, "util_score does not track reach.\nSmoke selection requires written rationale.",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=8,
            color="#666", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.grid(alpha=0.3)
    return _save(fig, "util_vs_reach_scatter")


def fig_phase2_5_geometry_move() -> Path:
    """Show gu_max / cf moves across v1 → v2_aggr → v3_diverse → v4_combo vs Track L baseline."""
    df = _load_aggregated_contraction().filter(pl.col("status") == "ok")
    ids = [
        ("Track L", "artifacts_v3__dynamics_n64_legacy_ror_corr010"),
        ("v1 conservative", "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v1"),
        ("v2_aggressive", "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v2_aggressive"),
        ("v3_diverse", "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v3_diverse"),
        ("v4_combo", "artifacts_v3__v3c__dynamics_candidates__contraction_aware_v4_combo"),
    ]
    rows = []
    for label, fid in ids:
        m = df.filter(pl.col("field_id") == fid)
        if m.height == 0:
            continue
        r = m.to_dicts()[0]
        rows.append({"label": label, "gu_max": r["gene_universality_max"],
                     "cf": r["contraction_fraction"], "align": r["alignment_cos_median"],
                     "act_div": r["action_diversity_per_state"]})
    if len(rows) < 2:
        LOG.warning("Insufficient Phase 2.5 audit data for geometry-move figure")
        return Path()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(rows))
    labels = [r["label"] for r in rows]
    gus = [r["gu_max"] for r in rows]
    align = [r["align"] for r in rows]
    axes[0].bar(x, gus, color="#4C72B0", alpha=0.85, edgecolor="#222")
    axes[0].axhline(0.85, color="#D62728", ls="--", lw=1, label="τ=0.85 attractor threshold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("gene_universality_max"); axes[0].set_ylim(0.6, 1.0)
    axes[0].set_title("gu_max move across Phase 2 / 2.5 variants")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, align, color="#FF7F0E", alpha=0.85, edgecolor="#222")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("alignment_cos_median (OOD pool)")
    axes[1].set_title("alignment median — collapse vs Soft-OT?")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "phase2_5_geometry_move")


REGISTRY = {
    "pipeline_overview":             fig_pipeline_overview,
    "dynamics_pathology_summary":    fig_dynamics_pathology_summary,
    "contraction_geometry_comparison": fig_contraction_geometry_comparison,
    "phase4_track_ln_results":       fig_phase4_track_ln_results,
    "final_leaderboard":             fig_final_leaderboard,
    "reward_stack_bucketA":          fig_reward_stack_bucketA,
    "rl_path_followed":              fig_rl_path_followed,
    "latent_space":                  fig_latent_space,
    "training_error_curves":         fig_training_error_curves,
    "phase2_5_geometry_move":        fig_phase2_5_geometry_move,
    "util_vs_reach_scatter":         fig_util_vs_reach_scatter,
    "champion_vs_greedy":            fig_champion_vs_greedy,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("which", nargs="?", default="all",
                   help=f"figure name or 'all'. choices: {list(REGISTRY)}")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    if args.which == "all":
        for name, fn in REGISTRY.items():
            try:
                fn()
            except Exception as exc:                               # noqa: BLE001
                LOG.warning("figure %s failed: %s", name, exc)
        return 0
    if args.which not in REGISTRY:
        LOG.error("Unknown figure: %s", args.which)
        return 2
    REGISTRY[args.which]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
