"""Trajectory rendering: project PPO rollouts onto UMAP, plot paths.

Owner: Agent A owns plumbing; Agent B contributes the rollout data structure (Contract 4).

Used in ``notebooks/03_rl_trajectory_viz.ipynb`` and ``scripts/visualize.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Contract 4 required columns (AGENTS.md §4)
_CONTRACT4_COLS = {
    "episode_id", "step", "action", "gene_symbol",
    "z_norm", "reward", "terminated", "success", "z_vector",
}


def load_rollouts(path: str | Path) -> Any:
    """Load ``rollouts.parquet`` produced by RL evaluation.

    Validates the Contract 4 schema (AGENTS.md §4). Missing columns raise
    ``ValueError`` so downstream callers get a clear error rather than a
    confusing KeyError.

    Parameters
    ----------
    path
        Path to ``artifacts/rl/rollouts.parquet``.

    Returns
    -------
    polars.DataFrame
        Columns: episode_id, step, action, gene_symbol, z_norm,
        reward, terminated, success, z_vector.
    """
    import polars as pl

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"rollouts.parquet not found at {path}. "
            "Run `make rl` first, or generate untrained-policy rollouts "
            "for a smoke baseline (label clearly as 'untrained')."
        )
    df = pl.read_parquet(str(path))
    missing = _CONTRACT4_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"rollouts.parquet missing Contract 4 columns: {missing}. "
            f"Present: {df.columns}"
        )
    n_eps = df["episode_id"].n_unique()
    n_success = int(df.filter(pl.col("terminated") & pl.col("success"))["success"].sum())
    log.info(
        "Loaded rollouts: %d episodes, %d steps, %d successes",
        n_eps, len(df), n_success,
    )
    return df


def project_rollouts_to_umap(
    rollouts: Any,
    umap_reducer: Any,
) -> Any:
    """Project each rollout step's ``z_vector`` into the fitted UMAP 2-D space.

    Reuses the SAME fitted reducer from :func:`src.analysis.latent_space.compute_umap`
    so trajectories appear on the same 2-D canvas as the cell cloud.

    Parameters
    ----------
    rollouts
        Polars DataFrame from :func:`load_rollouts`.
    umap_reducer
        Fitted ``umap.UMAP`` instance (call ``.transform()``, not ``.fit_transform()``).

    Returns
    -------
    polars.DataFrame
        Same as input plus float32 columns ``umap_x`` and ``umap_y``.
    """
    import polars as pl

    # Unpack list-of-list z_vector column → (N_steps, n_latent) float32 array
    z_matrix = np.array(rollouts["z_vector"].to_list(), dtype=np.float32)
    log.info("Projecting %d rollout steps into UMAP space...", len(z_matrix))

    # transform() applies the fitted UMAP — does NOT re-fit
    umap_coords = umap_reducer.transform(z_matrix).astype(np.float32)

    return rollouts.with_columns([
        pl.Series("umap_x", umap_coords[:, 0].tolist()),
        pl.Series("umap_y", umap_coords[:, 1].tolist()),
    ])


def plot_trajectories(
    projected_rollouts: Any,
    background_umap: Any,
    background_labels: Any,
    z_reference_centroid_xy: Any,
    n_episodes_to_plot: int = 20,
    out_path: str | Path | None = None,
) -> Any:
    """Render RL trajectories on a UMAP background scatter.

    Plots up to ``n_episodes_to_plot`` episodes as line paths on the cell-cloud
    UMAP. Successful episodes are drawn in green, failed ones in red.
    Arrows show the direction of travel. The reference centroid is marked with
    a gold star.

    Parameters
    ----------
    projected_rollouts
        Output of :func:`project_rollouts_to_umap` — must have ``umap_x``,
        ``umap_y``, ``episode_id``, ``success``, ``terminated`` columns.
    background_umap
        Array shape ``(N_cells, 2)`` — UMAP coordinates of the cell cloud.
    background_labels
        String labels for the background (used for grey scatter; no legend per cell).
    z_reference_centroid_xy
        2-element array: UMAP coordinates of ``z_reference_centroid``.
    n_episodes_to_plot
        Number of episodes to render (sampled uniformly from all episodes).
    out_path
        Where to save the figure. If ``None``, returned without saving.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import polars as pl

    df = projected_rollouts

    # ------------------------------------------------------------------
    # Sample episodes to plot
    # ------------------------------------------------------------------
    all_eps = df["episode_id"].unique().sort().to_list()
    rng = np.random.default_rng(42)
    chosen = rng.choice(
        all_eps, size=min(n_episodes_to_plot, len(all_eps)), replace=False
    ).tolist()
    df_plot = df.filter(pl.col("episode_id").is_in(chosen)).sort(["episode_id", "step"])

    # Determine which episodes succeeded
    terminal = df.filter(pl.col("terminated"))
    success_eps = set(
        terminal.filter(pl.col("success"))["episode_id"].to_list()
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background scatter — all cells, light grey
    bg = np.asarray(background_umap)
    ax.scatter(bg[:, 0], bg[:, 1], c="lightgray", s=1, alpha=0.2, rasterized=True)

    # Trajectories
    for ep_id in chosen:
        ep = df_plot.filter(pl.col("episode_id") == ep_id)
        xs = ep["umap_x"].to_list()
        ys = ep["umap_y"].to_list()
        if len(xs) < 2:
            continue
        color = "#2ecc71" if ep_id in success_eps else "#e74c3c"  # green / red
        alpha = 0.85 if ep_id in success_eps else 0.5

        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=alpha)
        # Start marker
        ax.scatter(xs[0], ys[0], c=color, s=30, marker="o", zorder=5, alpha=alpha)
        # Arrow at the end showing direction of last step
        if len(xs) >= 2:
            ax.annotate(
                "", xy=(xs[-1], ys[-1]),
                xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

    # Reference centroid
    z_ref_xy = np.asarray(z_reference_centroid_xy)
    ax.scatter(
        z_ref_xy[0], z_ref_xy[1],
        c="gold", s=300, marker="*", zorder=10,
        edgecolors="black", linewidths=0.8,
        label="z_ref centroid",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    )

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#2ecc71", linewidth=2, label=f"success ({len(success_eps & set(chosen))})"),
        Line2D([0], [0], color="#e74c3c", linewidth=2, label=f"failure ({len(set(chosen) - success_eps)})"),
        Line2D([0], [0], marker="*", color="gold", markersize=12, linestyle="None",
               markeredgecolor="black", label="z_ref centroid"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(
        f"RL trajectories on VAE latent UMAP  |  {n_episodes_to_plot} episodes  "
        f"|  success rate={len(success_eps) / max(len(all_eps), 1):.1%}"
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        log.info("Trajectory plot saved → %s", out_path)

    return fig
