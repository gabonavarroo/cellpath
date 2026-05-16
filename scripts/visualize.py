"""scripts/visualize.py — produce every defense figure from artifacts/eval/.

Single source of truth is ``artifacts/eval/summary.json`` (P1A aggregator output).
The composite ``evaluate_report.json`` is read to locate the DepMap enrichment CSV
when present.

Figures (under ``cfg.paths.eval_figures_dir``):

Required (always emitted when source data is present):
  1. ``fig_rl_ppo_vs_random.png``
  2. ``fig_contraction_comparison.png``
  3. ``fig_dynamics_gate.png``
  4. ``fig_rl_action_freq.png``
  5. ``fig_depmap_enrichment.png``    (skipped with a warning if no DepMap CSV)

Optional (only if a fitted UMAP is on disk):
  6. ``fig_rl_trajectories.png``
  7. ``fig_umap_perturbations.png``

The optional figures need a fitted UMAP reducer; we look for it under
``${paths.vae_dir}/umap_reducer.joblib`` (the path some notebooks cache) and skip
gracefully if missing. They are stretch goals, not required for V1.

Usage
-----
::

    python scripts/visualize.py --config-name default rl.train.skip_gate=true
    python scripts/visualize.py --config-name default rl.train.skip_gate=true \\
        +visualize.only=fig_rl_action_freq
    python scripts/visualize.py --config-name default rl.train.skip_gate=true \\
        +visualize.compute_umap_if_missing=true   # fit + cache UMAP, then render optional figs
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
from omegaconf import DictConfig


log = logging.getLogger(__name__)

# Use a non-interactive backend so the script never blocks waiting on a window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# =============================================================================
# Figure 1 — PPO vs random comparison
# =============================================================================


def fig_rl_ppo_vs_random(summary: dict[str, Any], out_path: Path) -> bool:
    """Grouped-bar comparison of success_rate, mean_steps, mean_final_distance."""
    rl = summary.get("rl") or {}
    runs = [
        ("PPO det", rl.get("ppo_deterministic")),
        ("PPO stoch", rl.get("ppo_stochastic")),
        ("Random", rl.get("random_baseline")),
    ]
    runs = [(name, m) for name, m in runs if m is not None]
    if len(runs) < 2:
        log.warning("Not enough RL runs to render PPO-vs-random; skipping.")
        return False

    metrics = [
        ("success_rate", "success rate", 1.0),
        ("mean_steps", "mean steps", None),
        ("mean_final_distance", "mean final distance", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"PPO det": "#2563eb", "PPO stoch": "#0ea5e9", "Random": "#d4d4d8"}
    for ax, (key, ylabel, ymax) in zip(axes, metrics):
        names = [name for name, _ in runs]
        vals = [(m.get(key) or 0.0) for _, m in runs]
        bars = ax.bar(names, vals, color=[colors.get(n, "#888") for n in names], edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel)
        if ymax is not None:
            ax.set_ylim(0, ymax * 1.10)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.3)
    delta = rl.get("delta_ppo_det_vs_random") or {}
    suptitle = "PPO vs random (matched env)"
    if delta.get("delta_success_pp") is not None:
        suptitle += f" — Δ PPO det − random = {delta['delta_success_pp']:+.1f} pp"
    suptitle += "  •  gate failed/overridden"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# =============================================================================
# Figure 2 — contraction comparison
# =============================================================================


def fig_contraction_comparison(summary: dict[str, Any], out_path: Path) -> bool:
    """Two-panel bar plot: fraction_improved + mean_improvement across variants."""
    contra = summary.get("contraction") or {}
    order = [
        ("primary_32d", "32D state_linear\nstart8"),
        ("primary_32d_auto", "32D state_linear\nauto"),
        ("ablation_64d", "64D state_linear\nstart8"),
        ("ablation_64d_auto", "64D state_linear\nauto"),
        ("ablation_64d_plain", "64D baseline_plain\nstart8"),
    ]
    rows = [(label, contra.get(key)) for key, label in order]
    rows = [(lbl, r) for lbl, r in rows if r is not None]
    if not rows:
        log.warning("No contraction rows in summary; skipping.")
        return False

    labels = [lbl for lbl, _ in rows]
    frac = [(r.get("fraction_improved") or 0.0) for _, r in rows]
    mean_imp = [(r.get("mean_improvement") or 0.0) for _, r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#16a34a" if "32D" in lbl else "#dc2626" for lbl in labels]

    # Left: fraction_improved
    axes[0].bar(labels, frac, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(frac):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(1.0, color="black", linestyle=":", alpha=0.5, linewidth=0.8)
    axes[0].set_ylabel("fraction of (start, gene) actions improving distance")
    axes[0].set_title("Fraction improved (strict, > 0)")
    axes[0].yaxis.grid(True, alpha=0.3)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="x", labelsize=8)

    # Right: mean_improvement (latent units)
    axes[1].bar(labels, mean_imp, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(mean_imp):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].set_ylabel("mean ||z − z_ref|| reduction (latent units)")
    axes[1].set_title("Mean improvement per action")
    axes[1].yaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)
    axes[1].tick_params(axis="x", labelsize=8)

    fig.suptitle("Dynamics contraction — 32D vs 64D × start8 vs auto", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# =============================================================================
# Figure 3 — dynamics gate (MLP vs ridge, primary + OOD)
# =============================================================================


def fig_dynamics_gate(summary: dict[str, Any], out_path: Path) -> bool:
    """Two-panel grouped bars: primary val Pearson and OOD Pearson for 32D + 64D."""
    dyn = summary.get("dynamics") or {}
    primary = dyn.get("primary_32d")
    ablation = dyn.get("ablation_64d")
    if primary is None and ablation is None:
        log.warning("No dynamics-gate data; skipping fig_dynamics_gate.")
        return False

    def _vals(block: dict[str, Any], split: str) -> tuple[float, float, float, float]:
        b = (block or {}).get(split) or {}
        return (
            float(b.get("mlp_pearson") or 0.0),
            float(b.get("ridge_pearson") or 0.0),
            float(b.get("margin_vs_linear_ridge_pearson") or 0.0),
            float(b.get("margin_vs_linear_ridge_pearson_threshold") or 0.03),
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, split, title in [(axes[0], "primary", "Primary (val)"),
                              (axes[1], "ood", "OOD (held-out genes)")]:
        labels = []
        mlp_vals = []
        ridge_vals = []
        threshold = 0.03
        for label, block in [("32D", primary), ("64D", ablation)]:
            if block is None:
                continue
            mlp, ridge, _, th = _vals(block, split)
            labels.append(label)
            mlp_vals.append(mlp)
            ridge_vals.append(ridge)
            threshold = th  # same across rows in practice

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, mlp_vals, width=w, label="MLP", color="#2563eb", edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, ridge_vals, width=w, label="Ridge", color="#9ca3af", edgecolor="black", linewidth=0.5)
        for xi, (mv, rv) in enumerate(zip(mlp_vals, ridge_vals)):
            ax.text(xi - w / 2, mv, f"{mv:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + w / 2, rv, f"{rv:.3f}", ha="center", va="bottom", fontsize=8)
            margin = mv - rv
            ax.annotate(f"Δ={margin:+.4f}",
                        xy=(xi, max(mv, rv)),
                        xytext=(xi, max(mv, rv) + 0.05),
                        ha="center", fontsize=8, color="#0f172a")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Pearson correlation")
        ax.set_title(f"{title} — MLP vs ridge")
        ax.set_ylim(0, max(mlp_vals + ridge_vals + [0.7]) * 1.20)
        ax.legend(loc="lower right", fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(f"Dynamics gate — MLP-ridge margin threshold = +{threshold:.3f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# =============================================================================
# Figure 4 — top-K action frequency, PPO vs random
# =============================================================================


def fig_rl_action_freq(summary: dict[str, Any], out_path: Path) -> bool:
    """Side-by-side top-K bars for PPO det and the random baseline."""
    top = summary.get("top_actions") or {}
    ppo = top.get("ppo") or []
    rand = top.get("random") or []
    if not ppo and not rand:
        log.warning("No top-action data; skipping fig_rl_action_freq.")
        return False
    k = int(top.get("top_k") or max(len(ppo), len(rand)))

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.32 * k)))

    for ax, rows, label, color in [
        (axes[0], ppo, "PPO deterministic", "#2563eb"),
        (axes[1], rand, "Random uniform-valid", "#9ca3af"),
    ]:
        if not rows:
            ax.axis("off")
            ax.set_title(f"{label} (no data)")
            continue
        genes = [r["gene_symbol"] for r in rows]
        counts = [int(r["count"]) for r in rows]
        y = np.arange(len(genes))
        ax.barh(y, counts, color=color, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(genes, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("action count over 500 episodes")
        ax.set_title(f"{label} — top {len(genes)}")
        for yi, c in zip(y, counts):
            ax.text(c, yi, f" {c}", va="center", fontsize=8)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    overlap = top.get("intersection") or []
    suptitle = f"Top-{k} action frequency"
    if overlap:
        suptitle += f"   •   Overlap PPO ∩ random: {', '.join(overlap)}"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# =============================================================================
# Figure 5 — DepMap enrichment heatmap
# =============================================================================


def fig_depmap_enrichment(depmap_csv: Path | None, out_path: Path) -> bool:
    """Heatmap of -log10(q_value) per (panel, test) with effect-size annotations."""
    if depmap_csv is None or not depmap_csv.exists():
        log.warning("DepMap CSV not present; skipping fig_depmap_enrichment.")
        return False

    import polars as pl

    df = pl.read_csv(str(depmap_csv))
    if df.height == 0:
        log.warning("DepMap CSV is empty; skipping fig_depmap_enrichment.")
        return False

    # Pivot to (panel, test) → q_value
    panels = sorted(df["panel"].unique().to_list())
    tests = sorted(df["test"].unique().to_list())
    q_grid = np.full((len(panels), len(tests)), np.nan, dtype=np.float64)
    eff_grid = np.full((len(panels), len(tests)), np.nan, dtype=np.float64)
    for row in df.iter_rows(named=True):
        i = panels.index(row["panel"])
        j = tests.index(row["test"])
        q_grid[i, j] = float(row["q_value"])
        eff_grid[i, j] = float(row["effect_size"])

    # -log10(q) for the colormap. Guard against q==0 → -log10(0)=inf.
    with np.errstate(divide="ignore"):
        cells = -np.log10(np.clip(q_grid, 1e-12, 1.0))

    # Scale figure so each cell is at least 2.5 × 1.8 inches with generous margins.
    cell_w = max(2.5, 1.8 * len(tests))
    cell_h = max(1.8, 1.4 * len(panels))
    fig_w = cell_w + 2.4   # room for y-labels + colorbar
    fig_h = cell_h + 2.0   # room for title + x-labels
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cells, aspect="auto", cmap="viridis", vmin=0,
                   vmax=max(1.301, float(np.nanmax(cells)) if np.isfinite(np.nanmax(cells)) else 1.301))
    # 1.301 ≈ -log10(0.05) — the significance threshold visible on the colorbar
    ax.set_xticks(np.arange(len(tests)))
    ax.set_xticklabels(tests, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(panels)))
    ax.set_yticklabels(panels, fontsize=11)
    for i in range(len(panels)):
        for j in range(len(tests)):
            q = q_grid[i, j]
            e = eff_grid[i, j]
            if np.isnan(q):
                continue
            text_color = "white" if cells[i, j] > 0.7 else "black"
            ax.text(j, i, f"q = {q:.3g}\nES = {e:.2f}",
                    ha="center", va="center", fontsize=10, color=text_color,
                    linespacing=1.5)
            if q < 0.05:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                           fill=False, edgecolor="red", linewidth=2.5))
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03, shrink=0.8)
    cbar.set_label("-log10(q-value)  (BH-FDR)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title("DepMap enrichment — RL top-K genes vs K562 essentials", fontsize=12, pad=14)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# =============================================================================
# Figure 6 — DepMap gene-score comparison (violin/strip plot)
# =============================================================================


def fig_depmap_gene_score_comparison(
    comparison_summary_path: Path | None,
    gene_scores_csv_path: Path | None,
    out_path: Path,
) -> bool:
    """Violin + jitter plot comparing Chronos score distributions across policy groups.

    Reads from depmap_comparison_summary.json (for group-level stats) and optionally
    depmap_gene_level_scores.csv (for individual points). Falls back to summary-only
    rendering when the CSV is absent.
    """
    if comparison_summary_path is None or not comparison_summary_path.exists():
        log.warning("DepMap comparison summary not found — skipping gene-score figure.")
        return False

    with open(comparison_summary_path) as f:
        comp = json.load(f)

    top_k = comp.get("top_k", 20)
    stats = comp.get("chronos_stats") or {}
    mw = (comp.get("comparisons") or {}).get("ppo_det_vs_random_top_k") or {}
    perm = (comp.get("comparisons") or {}).get("ppo_det_permutation_vs_action_universe") or {}

    # Groups to plot (label, color, stats_key)
    groups = [
        (f"PPO det\ntop-{top_k}", "#2563eb", "ppo_det_top_k"),
        (f"PPO stoch\ntop-{top_k}", "#0ea5e9", "ppo_stoch_top_k"),
        (f"Random\ntop-{top_k}", "#9ca3af", "random_top_k"),
        ("Action\nuniverse", "#d97706", "action_universe"),
    ]

    # Load individual gene points if available
    gene_data: dict[str, list[float]] = {}
    if gene_scores_csv_path is not None and gene_scores_csv_path.exists():
        import polars as pl
        gdf = pl.read_csv(str(gene_scores_csv_path))
        for _, col_key, stats_key in [
            (f"PPO det top-{top_k}", "in_ppo_det_top_k", "ppo_det_top_k"),
            (f"PPO stoch top-{top_k}", "in_ppo_stoch_top_k", "ppo_stoch_top_k"),
            (f"Random top-{top_k}", "in_random_top_k", "random_top_k"),
            ("Action universe", "in_action_universe", "action_universe"),
        ]:
            col = col_key if col_key in gdf.columns else None
            if col:
                subset = gdf.filter(
                    (pl.col(col) == True) & pl.col("chronos_score").is_not_null()  # noqa: E712
                )["chronos_score"].to_list()
                gene_data[stats_key] = [float(v) for v in subset]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    positions = np.arange(len(groups))
    rng = np.random.default_rng(42)

    for xi, (label, color, stats_key) in zip(positions, groups):
        pts = gene_data.get(stats_key, [])
        s = stats.get(stats_key) or {}
        mean_c = s.get("mean_chronos")
        if pts:
            # Violin patch
            parts = ax.violinplot([pts], positions=[xi], widths=0.55,
                                  showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.35)
                pc.set_edgecolor(color)
            # Jittered scatter
            jitter = rng.uniform(-0.12, 0.12, size=len(pts))
            ax.scatter([xi + j for j in jitter], pts, s=22, c=color,
                       alpha=0.75, zorder=3, edgecolors="none")
        # Mean marker
        if mean_c is not None:
            ax.plot([xi - 0.22, xi + 0.22], [mean_c, mean_c],
                    color="black", linewidth=2.0, zorder=4, solid_capstyle="round")
            ax.text(xi + 0.28, mean_c, f"{mean_c:.3f}",
                    va="center", ha="left", fontsize=8, color="black")

    # Essential threshold
    ax.axhline(-0.5, color="crimson", linestyle="--", linewidth=1.2, alpha=0.8,
               label="essential threshold (Chronos < −0.5)")

    ax.set_xticks(positions)
    ax.set_xticklabels([lbl for lbl, _, _ in groups], fontsize=10)
    ax.set_ylabel("Chronos dependency score  (↓ = more essential)", fontsize=10)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    # Annotation: p-values
    p_mw = mw.get("p_value")
    q_mw = mw.get("q_value_bh")
    p_perm = perm.get("empirical_p")
    annot_parts = []
    if p_mw is not None:
        q_str = f", q={q_mw:.3g}" if q_mw is not None else ""
        annot_parts.append(f"MWU PPO det vs random: p={p_mw:.3g}{q_str}")
    if p_perm is not None:
        annot_parts.append(f"Permutation PPO vs action universe: p={p_perm:.3g}")
    if annot_parts:
        ax.text(0.98, 0.02, "\n".join(annot_parts),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#374151",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"DepMap K562 Chronos scores — PPO vs random (top-{top_k})\n"
        "Plausibility check only • lower = more K562-essential",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# =============================================================================
# Optional figures — UMAP-based; require a fitted reducer on disk
# =============================================================================


def _try_load_umap_reducer(cfg: DictConfig) -> Any | None:
    """Best-effort look-up of a fitted UMAP reducer; return ``None`` if not found.

    Search order:
    1. ``${paths.eval_dir}/umap_cache/umap_reducer.joblib``  (written by ``_fit_and_cache_umap``)
    2. ``${paths.vae_dir}/umap_reducer.joblib``               (optional notebook cache)
    """
    try:
        from joblib import load
    except Exception:
        return None
    candidates = [
        Path(cfg.paths.eval_dir) / "umap_cache" / "umap_reducer.joblib",
        Path(cfg.paths.vae_dir) / "umap_reducer.joblib",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return load(candidate)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load %s: %s", candidate, exc)
    return None


def _fit_and_cache_umap(cfg: DictConfig, *, max_cells: int = 50_000) -> Any | None:
    """Fit a UMAP reducer on the VAE latents and cache it under ``eval/umap_cache/``.

    Only called when ``+visualize.compute_umap_if_missing=true``. Returns the fitted
    reducer on success, ``None`` on any failure (missing latents, import error, …).
    The caller always wraps this in try/except so UMAP failures never abort the main run.
    """
    try:
        import anndata as ad
        import joblib
        import umap as umap_lib  # umap-learn
    except ImportError as exc:
        log.warning("UMAP fit skipped — missing dependency: %s", exc)
        return None

    latents_path = Path(cfg.paths.vae_latents_h5ad)
    if not latents_path.exists():
        log.warning("Latents not found at %s — cannot fit UMAP.", latents_path)
        return None

    log.info("Loading latents for UMAP fit from %s", latents_path)
    adata = ad.read_h5ad(str(latents_path))
    if "X_scVI" not in adata.obsm:
        log.warning("X_scVI not in adata.obsm — cannot fit UMAP.")
        return None

    Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    if len(Z) > max_cells:
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        idx = rng.choice(len(Z), size=max_cells, replace=False)
        Z_fit = Z[idx]
        log.info("Subsampled %d → %d cells for UMAP fit.", len(Z), max_cells)
    else:
        Z_fit = Z

    log.info("Fitting UMAP on %d × %d (this may take a minute) …", *Z_fit.shape)
    reducer = umap_lib.UMAP(n_components=2, min_dist=0.1, random_state=int(cfg.get("seed", 42)))
    reducer.fit(Z_fit)

    cache_dir = Path(cfg.paths.eval_dir) / "umap_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    reducer_path = cache_dir / "umap_reducer.joblib"
    joblib.dump(reducer, reducer_path)
    log.info("Saved UMAP reducer → %s", reducer_path)

    # Also save the full embedding for inspection.
    embedding = reducer.transform(Z).astype(np.float32)
    np.save(str(cache_dir / "umap_embedding.npy"), embedding)
    log.info("Saved UMAP embedding (%d × 2) → %s", len(Z), cache_dir / "umap_embedding.npy")

    return reducer


def _get_visualize_int(cfg: DictConfig, key: str, default: int) -> int:
    """Read an optional integer from ``+visualize.*`` Hydra overrides."""
    try:
        if hasattr(cfg, "visualize") and cfg.visualize is not None:
            value = cfg.visualize.get(key, default)
            return max(1, int(value))
    except Exception:
        pass
    return default


def _get_visualize_bool(cfg: DictConfig, key: str, default: bool = False) -> bool:
    """Read an optional boolean from ``+visualize.*`` Hydra overrides."""
    try:
        if hasattr(cfg, "visualize") and cfg.visualize is not None:
            return bool(cfg.visualize.get(key, default))
    except Exception:
        pass
    return default


_DISTANCE_TRAJECTORY_COLS = {
    "episode_id",
    "step",
    "umap_x",
    "umap_y",
    "z_norm",
    "terminated",
    "success",
}


def _trajectory_distance_segments(
    projected_rollouts: Any,
    episode_ids: list[int],
) -> tuple[list[dict[str, float]], dict[int, dict[str, float]]]:
    """Return per-step true-distance improvements and per-episode d0/dT summaries."""
    import polars as pl

    segments: list[dict[str, float]] = []
    summaries: dict[int, dict[str, float]] = {}
    for ep_id in episode_ids:
        ep = projected_rollouts.filter(pl.col("episode_id") == ep_id).sort("step")
        if ep.height == 0:
            continue

        xs = [float(x) for x in ep["umap_x"].to_list()]
        ys = [float(y) for y in ep["umap_y"].to_list()]
        distances = [float(d) for d in ep["z_norm"].to_list()]
        terminal_success = ep.filter(pl.col("terminated") & pl.col("success")).height > 0
        summaries[int(ep_id)] = {
            "d0": distances[0],
            "dT": distances[-1],
            "end_x": xs[-1],
            "end_y": ys[-1],
            "success": float(terminal_success),
        }

        for i in range(len(xs) - 1):
            segments.append(
                {
                    "episode_id": float(ep_id),
                    "x0": xs[i],
                    "y0": ys[i],
                    "x1": xs[i + 1],
                    "y1": ys[i + 1],
                    "d_before": distances[i],
                    "d_after": distances[i + 1],
                    "improvement": distances[i] - distances[i + 1],
                    "success": float(terminal_success),
                }
            )
    return segments, summaries


def _select_trajectory_label_episodes(
    summaries: dict[int, dict[str, float]],
) -> dict[int, str]:
    """Pick representative trajectories for sparse d0 -> dT labels."""
    if not summaries:
        return {}

    ordered = sorted(
        summaries.items(),
        key=lambda item: item[1]["d0"] - item[1]["dT"],
    )
    best_id = ordered[-1][0]
    median_id = ordered[len(ordered) // 2][0]

    labels = {
        best_id: "best improvement",
        median_id: "median improvement",
    }

    failure_rows = [
        (ep_id, values)
        for ep_id, values in summaries.items()
        if not bool(values.get("success", 0.0))
    ]
    if failure_rows:
        failure_id = min(
            failure_rows,
            key=lambda item: item[1]["d0"] - item[1]["dT"],
        )[0]
        labels[failure_id] = "failure"

    return labels


def _plot_distance_aware_trajectories(
    projected_rollouts: Any,
    background_umap: Any,
    background_labels: Any,
    z_reference_centroid_xy: Any,
    n_episodes_to_plot: int = 30,
    verbose: bool = False,
    out_path: str | Path | None = None,
) -> Any:
    """Render trajectories with arrows colored by true 32D latent improvement."""
    import matplotlib.patheffects as pe
    import polars as pl
    from matplotlib.lines import Line2D

    _ = background_labels  # Reserved for future faceting; background is intentionally grey.
    df = projected_rollouts
    all_eps = df["episode_id"].unique().sort().to_list()
    terminal = df.filter(pl.col("terminated"))
    success_eps = set(terminal.filter(pl.col("success"))["episode_id"].to_list())
    failure_eps = [ep for ep in all_eps if ep not in success_eps]

    rng = np.random.default_rng(42)
    chosen = rng.choice(
        all_eps, size=min(n_episodes_to_plot, len(all_eps)), replace=False
    ).tolist()
    if failure_eps and not (set(chosen) & set(failure_eps)):
        chosen[-1] = failure_eps[0]
    df_plot = df.filter(pl.col("episode_id").is_in(chosen)).sort(["episode_id", "step"])

    segments, summaries = _trajectory_distance_segments(df_plot, [int(ep) for ep in chosen])
    label_episodes = summaries.keys() if verbose else _select_trajectory_label_episodes(summaries)

    fig, ax = plt.subplots(figsize=(10.5, 8.2))

    bg = np.asarray(background_umap)
    ax.scatter(bg[:, 0], bg[:, 1], c="lightgray", s=1, alpha=0.16, rasterized=True)

    improvements = np.asarray([s["improvement"] for s in segments], dtype=np.float32)
    max_abs = float(np.nanmax(np.abs(improvements))) if len(improvements) else 1.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    cmap = plt.get_cmap("RdYlGn")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    for ep_id in chosen:
        ep = df_plot.filter(pl.col("episode_id") == ep_id)
        xs = [float(x) for x in ep["umap_x"].to_list()]
        ys = [float(y) for y in ep["umap_y"].to_list()]
        if not xs:
            continue

        succeeded = ep_id in success_eps
        line_color = "#16a34a" if succeeded else "#dc2626"
        line_alpha = 0.28 if succeeded else 0.20
        line_style = "-" if succeeded else "--"
        marker_alpha = 0.92 if succeeded else 0.55

        if len(xs) > 1:
            ax.plot(
                xs,
                ys,
                color=line_color,
                linewidth=0.9,
                alpha=line_alpha,
                linestyle=line_style,
                zorder=2,
            )
        ax.scatter(
            xs[0],
            ys[0],
            s=55,
            marker="o",
            facecolors="white",
            edgecolors=line_color,
            linewidths=1.2,
            alpha=marker_alpha,
            zorder=5,
        )

        summary = summaries.get(int(ep_id))
        end_color = line_color
        if verbose and summary is not None:
            total_improvement = summary["d0"] - summary["dT"]
            end_color = cmap(norm(total_improvement))
        ax.scatter(
            xs[-1],
            ys[-1],
            s=70,
            marker="X",
            c=[end_color],
            edgecolors="black",
            linewidths=0.7,
            alpha=marker_alpha,
            zorder=6,
        )

        if summary is not None and int(ep_id) in label_episodes:
            label = label_episodes[int(ep_id)] if isinstance(label_episodes, dict) else ""
            prefix = f"{label}\n" if label else ""
            ax.annotate(
                f"{prefix}d {summary['d0']:.2f} → {summary['dT']:.2f}",
                xy=(summary["end_x"], summary["end_y"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color="#111827",
                alpha=0.9 if succeeded else 0.65,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=7,
            )

    for seg in segments:
        succeeded = bool(seg["success"])
        color = cmap(norm(seg["improvement"])) if verbose else ("#16a34a" if succeeded else "#dc2626")
        alpha = (0.78 if succeeded else 0.45) if verbose else (0.48 if succeeded else 0.36)
        ax.annotate(
            "",
            xy=(seg["x1"], seg["y1"]),
            xytext=(seg["x0"], seg["y0"]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.45,
                "alpha": alpha,
                "shrinkA": 1,
                "shrinkB": 1,
                "mutation_scale": 8,
            },
            zorder=4,
        )

    z_ref_xy = np.asarray(z_reference_centroid_xy)
    ax.scatter(
        z_ref_xy[0], z_ref_xy[1],
        c="gold", s=320, marker="*", zorder=10,
        edgecolors="black", linewidths=0.8,
        label="z_ref centroid",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    )

    if verbose:
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("true improvement per step: d_before - d_after", fontsize=9)

    n_success_chosen = len(success_eps & set(chosen))
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#16a34a",
            linewidth=2.0,
            label=f"success ({n_success_chosen})",
        ),
        Line2D(
            [0],
            [0],
            color="#dc2626",
            linewidth=2.0,
            linestyle="--",
            label=f"failure ({len(set(chosen) - success_eps)})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#0f172a",
            linestyle="None",
            markersize=7,
            label="start",
        ),
        Line2D([0], [0], marker="X", color="black", linestyle="None", markersize=8, label="end"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="gold",
            markeredgecolor="black",
            linestyle="None",
            markersize=13,
            label="z_ref centroid",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(
        f"RL trajectories on VAE latent UMAP  |  {len(chosen)} episodes  "
        f"|  overall success rate={len(success_eps) / max(len(all_eps), 1):.1%}"
    )
    ax.text(
        0.015,
        0.025,
        "Distances shown are true 32D latent distances, not UMAP distances.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.88,
        },
    )
    fig.text(
        0.5,
        0.022,
        "Qualitative 2D UMAP projection. Success is measured in original latent space; UMAP can distort directions.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#94a3b8",
            "alpha": 0.95,
        },
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        log.info("Distance-aware trajectory plot saved → %s", out_path)

    return fig


def fig_rl_trajectories(cfg: DictConfig, summary: dict[str, Any], out_path: Path) -> bool:
    """Project PPO rollouts to UMAP and plot a small set of trajectories.

    Skipped when:
    - no fitted UMAP reducer is cached on disk;
    - PPO rollouts.parquet is not present at the path the aggregator used;
    - latents.h5ad is not present (needed for the background).
    """
    reducer = _try_load_umap_reducer(cfg)
    if reducer is None:
        log.info("No UMAP reducer cached — skipping fig_rl_trajectories.")
        return False

    from src.analysis.trajectory import (
        load_rollouts,
        plot_trajectories,
        project_rollouts_to_umap,
    )

    # Locate rollouts via the aggregator's resolved ppo_det_dir (if present in summary).
    # Fall back to canonical V1 layout.
    repo_root = Path(cfg.paths.root)
    ppo_det_dir = repo_root / "artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic"
    rollouts_path = ppo_det_dir / "rollouts.parquet"
    if not rollouts_path.exists():
        log.warning("Rollouts not found at %s — skipping fig_rl_trajectories.", rollouts_path)
        return False

    latents_path = Path(cfg.paths.vae_latents_h5ad)
    if not latents_path.exists():
        log.warning("Latents not found at %s — skipping fig_rl_trajectories.", latents_path)
        return False

    import anndata as ad
    adata = ad.read_h5ad(str(latents_path))
    Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    bg = reducer.transform(Z).astype(np.float32)
    labels = np.asarray(adata.obs["perturbation_idx"].values)
    z_ref = np.load(str(cfg.paths.vae_z_reference_centroid)).astype(np.float32)
    z_ref_xy = reducer.transform(z_ref[None, :]).astype(np.float32)[0]

    rollouts = load_rollouts(rollouts_path)
    projected = project_rollouts_to_umap(rollouts, reducer)
    n_episodes = _get_visualize_int(cfg, "n_trajectory_episodes", 30)
    trajectory_verbose = _get_visualize_bool(cfg, "trajectory_verbose", False)
    missing = _DISTANCE_TRAJECTORY_COLS - set(getattr(projected, "columns", []))
    if missing:
        log.warning(
            "Distance-aware trajectory plot missing required fields %s; "
            "falling back to the basic UMAP trajectory plot.",
            sorted(missing),
        )
        fig = plot_trajectories(
            projected, background_umap=bg, background_labels=labels,
            z_reference_centroid_xy=z_ref_xy,
            n_episodes_to_plot=n_episodes, out_path=out_path,
        )
    else:
        try:
            fig = _plot_distance_aware_trajectories(
                projected, background_umap=bg, background_labels=labels,
                z_reference_centroid_xy=z_ref_xy,
                n_episodes_to_plot=n_episodes,
                verbose=trajectory_verbose,
                out_path=out_path,
            )
        except Exception as exc:
            log.warning(
                "Distance-aware trajectory plot failed (%s); "
                "falling back to the basic UMAP trajectory plot.",
                exc,
            )
            fig = plot_trajectories(
                projected, background_umap=bg, background_labels=labels,
                z_reference_centroid_xy=z_ref_xy,
                n_episodes_to_plot=n_episodes, out_path=out_path,
            )
    if fig is not None:
        plt.close(fig)
    return out_path.exists()


def fig_umap_perturbations(cfg: DictConfig, out_path: Path) -> bool:
    """UMAP scatter of cells colored by perturbation_idx, with the reference centroid marked."""
    reducer = _try_load_umap_reducer(cfg)
    if reducer is None:
        log.info("No UMAP reducer cached — skipping fig_umap_perturbations.")
        return False

    latents_path = Path(cfg.paths.vae_latents_h5ad)
    if not latents_path.exists():
        log.warning("Latents not found at %s — skipping fig_umap_perturbations.", latents_path)
        return False

    import anndata as ad
    adata = ad.read_h5ad(str(latents_path))
    Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    bg = reducer.transform(Z).astype(np.float32)
    labels = np.asarray(adata.obs["perturbation_idx"].values).astype(int)
    z_ref = np.load(str(cfg.paths.vae_z_reference_centroid)).astype(np.float32)
    z_ref_xy = reducer.transform(z_ref[None, :]).astype(np.float32)[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ctrl_mask = (labels == 0)
    ax.scatter(bg[~ctrl_mask, 0], bg[~ctrl_mask, 1], s=2, c=labels[~ctrl_mask],
               cmap="tab20", alpha=0.4, label="perturbed")
    ax.scatter(bg[ctrl_mask, 0], bg[ctrl_mask, 1], s=2, c="black", alpha=0.4, label="control")
    ax.scatter([z_ref_xy[0]], [z_ref_xy[1]], marker="*", s=300, c="gold",
               edgecolor="black", linewidth=1.2, zorder=5, label="z_ref")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("scVI latent UMAP — perturbations + reference centroid")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# =============================================================================
# Orchestrator
# =============================================================================


_REQUIRED_FIGURES: dict[str, Callable[..., bool]] = {
    "fig_rl_ppo_vs_random": fig_rl_ppo_vs_random,
    "fig_contraction_comparison": fig_contraction_comparison,
    "fig_dynamics_gate": fig_dynamics_gate,
    "fig_rl_action_freq": fig_rl_action_freq,
    # fig_depmap_enrichment needs a path, not summary — handled separately.
}


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra entry point."""
    from src.utils.seeding import set_seed

    set_seed(int(cfg.get("seed", 42)))

    eval_dir = Path(cfg.paths.eval_dir)
    figures_dir = Path(cfg.paths.eval_figures_dir)
    summary_path = eval_dir / "summary.json"
    report_path = eval_dir / "evaluate_report.json"

    if cfg.get("dry_run", False):
        _compute_umap_dry = False
        try:
            if hasattr(cfg, "visualize") and cfg.visualize is not None:
                _compute_umap_dry = bool(cfg.visualize.get("compute_umap_if_missing", False))
        except Exception:
            pass
        print(f"DRY RUN — would render figures into {figures_dir}")
        print(f"  required: {list(_REQUIRED_FIGURES) + ['fig_depmap_enrichment']}")
        print(f"  optional: ['fig_rl_trajectories', 'fig_umap_perturbations']")
        print(f"  compute_umap_if_missing: {_compute_umap_dry}")
        return 0

    if not summary_path.exists():
        log.error(
            "summary.json not found at %s — run `make evaluate` first.", summary_path,
        )
        return 2

    with open(summary_path) as f:
        summary = json.load(f)

    figures_dir.mkdir(parents=True, exist_ok=True)

    only = (cfg.get("visualize", {}) or {}).get("only") if isinstance(cfg.get("visualize", None), dict) else None
    if hasattr(cfg, "visualize") and cfg.visualize is not None:
        try:
            only = cfg.visualize.get("only", None)
        except Exception:
            pass

    written: dict[str, str] = {}
    skipped: list[str] = []

    # Required, summary-driven figures
    for name, fn in _REQUIRED_FIGURES.items():
        if only and name != only:
            continue
        out_path = figures_dir / f"{name}.png"
        log.info("Rendering %s → %s", name, out_path)
        ok = fn(summary, out_path)
        if ok and out_path.exists():
            written[name] = str(out_path)
        else:
            skipped.append(name)

    # Required, separate-source figure: DepMap enrichment heatmap
    if not only or only == "fig_depmap_enrichment":
        out_path = figures_dir / "fig_depmap_enrichment.png"
        csv_path: Path | None = None
        if report_path.exists():
            report = json.loads(report_path.read_text())
            cp = report.get("depmap_enrichment_csv")
            if cp:
                csv_path = Path(cp)
        if csv_path is None:
            csv_path = eval_dir / "depmap_enrichment.csv"
        log.info("Rendering fig_depmap_enrichment from %s", csv_path)
        ok = fig_depmap_enrichment(csv_path, out_path)
        if ok and out_path.exists():
            written["fig_depmap_enrichment"] = str(out_path)
        else:
            skipped.append("fig_depmap_enrichment")

    # DepMap gene-score comparison figure
    if not only or only == "fig_depmap_gene_score_comparison":
        out_path = figures_dir / "fig_depmap_gene_score_comparison.png"
        comp_summary: Path | None = None
        gene_scores: Path | None = None
        if report_path.exists():
            _report = json.loads(report_path.read_text())
            csp = _report.get("depmap_comparison_summary_json")
            gsp = _report.get("depmap_gene_level_scores_csv")
            if csp:
                comp_summary = Path(csp)
            if gsp:
                gene_scores = Path(gsp)
        if comp_summary is None:
            comp_summary = eval_dir / "depmap_comparison_summary.json"
        if gene_scores is None:
            gene_scores = eval_dir / "depmap_gene_level_scores.csv"
        log.info("Rendering fig_depmap_gene_score_comparison from %s", comp_summary)
        try:
            ok = fig_depmap_gene_score_comparison(comp_summary, gene_scores, out_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("fig_depmap_gene_score_comparison failed (%s) — skipping.", exc)
            ok = False
        if ok and out_path.exists():
            written["fig_depmap_gene_score_comparison"] = str(out_path)
        else:
            skipped.append("fig_depmap_gene_score_comparison")

    # Optional: UMAP-based figures (require a cached reducer)
    # Check for compute_umap_if_missing flag before attempting optional figures.
    _compute_umap = False
    try:
        if hasattr(cfg, "visualize") and cfg.visualize is not None:
            _compute_umap = bool(cfg.visualize.get("compute_umap_if_missing", False))
    except Exception:
        pass

    _umap_optional_names = ["fig_rl_trajectories", "fig_umap_perturbations"]
    _any_umap_requested = not only or only in _umap_optional_names
    if _compute_umap and _any_umap_requested:
        # Only fit if no reducer is cached yet.
        if _try_load_umap_reducer(cfg) is None:
            log.info("compute_umap_if_missing=true — fitting UMAP reducer …")
            try:
                _fit_and_cache_umap(cfg)
            except Exception as exc:  # noqa: BLE001
                log.warning("UMAP fit failed (%s) — optional figures will be skipped.", exc)

    for name, fn in [
        ("fig_rl_trajectories", lambda s, p: fig_rl_trajectories(cfg, s, p)),
        ("fig_umap_perturbations", lambda s, p: fig_umap_perturbations(cfg, p)),
    ]:
        if only and name != only:
            continue
        out_path = figures_dir / f"{name}.png"
        log.info("Trying optional %s …", name)
        try:
            ok = fn(summary, out_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed (%s) — skipping.", name, exc)
            ok = False
        if ok and out_path.exists():
            written[name] = str(out_path)
        else:
            skipped.append(name)

    print("\nFigures written:")
    for k, v in written.items():
        print(f"  ✓ {k}: {v}")
    if skipped:
        print("\nFigures skipped:")
        for k in skipped:
            print(f"  – {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
