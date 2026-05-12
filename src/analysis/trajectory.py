"""Trajectory rendering: project PPO rollouts onto UMAP, plot paths.

Owner: Agent A owns plumbing; Agent B contributes the rollout data structure (Contract 4).

Used in ``notebooks/03_rl_trajectory_viz.ipynb`` and ``scripts/visualize.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_rollouts(path: str | Path) -> Any:
    """Load ``rollouts.parquet`` produced by RL evaluation.

    Parameters
    ----------
    path
        Path to ``artifacts/rl/rollouts.parquet``.

    Returns
    -------
    polars.DataFrame
        Columns per AGENTS.md §4 Contract 4.

    Raises
    ------
    NotImplementedError
        Agent A: ``polars.read_parquet(path)``; schema validation.
    """
    raise NotImplementedError("Agent A: polars.read_parquet + schema check on Contract 4.")


def project_rollouts_to_umap(
    rollouts: Any,
    umap_reducer: Any,
) -> Any:
    """Project each rollout step's ``z_vector`` into the fitted UMAP 2-D space.

    Parameters
    ----------
    rollouts
        DataFrame from :func:`load_rollouts`.
    umap_reducer
        Fitted UMAP reducer (re-using the one from latent-space analysis).

    Returns
    -------
    polars.DataFrame
        Same as input plus columns ``umap_x`` and ``umap_y``.

    Raises
    ------
    NotImplementedError
        Agent A: implement (stack z_vectors, transform, attach as cols).
    """
    raise NotImplementedError("Agent A: stack z_vector list-of-list → np.array; transform.")


def plot_trajectories(
    projected_rollouts: Any,
    background_umap: Any,
    background_labels: Any,
    z_reference_centroid_xy: Any,
    n_episodes_to_plot: int = 20,
    out_path: str | Path | None = None,
) -> Any:
    """Render trajectories on a UMAP background.

    Parameters
    ----------
    projected_rollouts
        Output of :func:`project_rollouts_to_umap`.
    background_umap, background_labels
        Latent UMAP + perturbation labels for the background scatter.
    z_reference_centroid_xy
        UMAP coordinates of the reference centroid.
    n_episodes_to_plot
        Number of representative episodes (faceted or overplotted).
    out_path
        Where to save the figure.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    NotImplementedError
        Agent A: implement with matplotlib paths + alpha shading. Highlight successes vs failures.
    """
    raise NotImplementedError(
        "Agent A: matplotlib LineCollection-style trajectories + background scatter."
    )
