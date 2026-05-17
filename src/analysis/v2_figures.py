"""V2 wrap-up figures.

Plot helpers that read existing ``summary.json`` / ``action_freq.json`` / ``rollouts.parquet``
files and emit PNGs under ``artifacts_v2/figures/``. No new metrics are defined here — this
module is figure generation only (per CLAUDE.md §3 rule 8).

All six figures from the V2_WRAP_OR_V3_PIVOT_PLAN are produced by ``generate_all_figures()``.
Each function gracefully skips a missing input file with a clear warning.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in CI
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _collect_eval_grid(eval_dir: Path) -> dict[str, dict[str, dict]]:
    """{cell: {policy: summary}} for a single PPO run's eval dir."""
    out: dict[str, dict[str, dict]] = {}
    if not eval_dir.exists():
        return out
    for cell_dir in sorted(eval_dir.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("k"):
            continue
        per_pol: dict[str, dict] = {}
        for pol_dir in cell_dir.iterdir():
            if pol_dir.is_dir():
                blob = _safe_load(pol_dir / "summary.json")
                if blob is not None:
                    per_pol[pol_dir.name] = blob
        if per_pol:
            out[cell_dir.name] = per_pol
    return out


def _cell_to_k_bin(cell: str) -> tuple[int, str]:
    """Parse 'k3_epsp25_bin8-10_splitood' → (3, '8-10')."""
    k = int(cell.split("_", 1)[0][1:])
    bin_str = cell.split("bin", 1)[1].split("_split")[0]
    return k, bin_str


# ---------------------------------------------------------------------------
# Figure 1 — success rate vs K, faceted by distance bin
# ---------------------------------------------------------------------------

def plot_success_vs_K(
    eval_dirs: dict[str, Path],
    out_path: Path,
    *,
    bins: tuple[str, ...] = ("6-8", "8-10"),
    policies: tuple[str, ...] = (
        "ppo_deterministic", "random_uniform_valid",
        "greedy_dyn_1", "greedy_dyn_2",
    ),
) -> None:
    """X = K, Y = success rate, lines per policy, one panel per bin.

    Plots ALL eval dirs on the same axes with different markers per config (so multiple
    PPOs can be overlaid). Greedy/random are read from the first dir only.
    """
    fig, axes = plt.subplots(1, len(bins), figsize=(6 * len(bins), 5), sharey=True)
    axes = np.atleast_1d(axes)

    # Color map: PPO configs get distinct colours; baselines share a palette.
    cfg_names = list(eval_dirs.keys())
    cfg_colors = plt.cm.tab10(np.linspace(0, 1, max(len(cfg_names), 3)))
    baseline_colors = {
        "random_uniform_valid": "0.55",
        "greedy_dyn_1": "darkblue",
        "greedy_dyn_2": "crimson",
        "always_noop": "olive",
    }

    for ax_i, bin_str in enumerate(bins):
        ax = axes[ax_i]
        # PPO curves per config
        for c_i, (cfg_name, eval_dir) in enumerate(eval_dirs.items()):
            grid = _collect_eval_grid(eval_dir)
            xs, ys = [], []
            for cell, per_pol in grid.items():
                k, b = _cell_to_k_bin(cell)
                if b != bin_str:
                    continue
                if "ppo_deterministic" in per_pol:
                    xs.append(k)
                    ys.append(per_pol["ppo_deterministic"]["success_rate"])
            order = np.argsort(xs)
            ax.plot([xs[i] for i in order], [ys[i] for i in order],
                    marker="o", linewidth=2, color=cfg_colors[c_i], label=f"PPO: {cfg_name}")

        # Baselines from the first dir (they don't depend on PPO)
        first_dir = next(iter(eval_dirs.values()))
        grid = _collect_eval_grid(first_dir)
        for pol in policies:
            if pol == "ppo_deterministic":
                continue
            xs, ys = [], []
            for cell, per_pol in grid.items():
                k, b = _cell_to_k_bin(cell)
                if b != bin_str or pol not in per_pol:
                    continue
                xs.append(k)
                ys.append(per_pol[pol]["success_rate"])
            order = np.argsort(xs)
            ax.plot([xs[i] for i in order], [ys[i] for i in order],
                    marker="s", linestyle="--", linewidth=1.5,
                    color=baseline_colors.get(pol, "gray"),
                    label=pol)

        ax.set_xlabel("K (episode budget)")
        if ax_i == 0:
            ax.set_ylabel("Success rate (OOD held-out genes)")
        ax.set_title(f"Distance bin {bin_str}")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Success rate vs K (V2 hard benchmark, OOD)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Figure 2 — hardness frontier scatter
# ---------------------------------------------------------------------------

def plot_hardness_frontier(
    eval_dirs: dict[str, Path],
    out_path: Path,
) -> None:
    """Scatter: x = PPO − random, y = PPO − greedy_dyn_2. One point per (cell, config)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cfg_names = list(eval_dirs.keys())
    cfg_colors = plt.cm.tab10(np.linspace(0, 1, max(len(cfg_names), 3)))
    marker_by_k = {1: "v", 2: "o", 3: "s", 8: "D"}

    for c_i, (cfg_name, eval_dir) in enumerate(eval_dirs.items()):
        for cell, per_pol in _collect_eval_grid(eval_dir).items():
            ppo = per_pol.get("ppo_deterministic", {}).get("success_rate")
            rnd = per_pol.get("random_uniform_valid", {}).get("success_rate")
            g2  = per_pol.get("greedy_dyn_2", {}).get("success_rate")
            if ppo is None or rnd is None or g2 is None:
                continue
            k, b = _cell_to_k_bin(cell)
            ax.scatter(
                ppo - rnd, ppo - g2,
                marker=marker_by_k.get(k, "x"),
                s=90, color=cfg_colors[c_i],
                edgecolor="black", linewidth=0.7,
                label=f"{cfg_name} K={k} bin {b}",
            )

    ax.axhline(0, color="0.7", linewidth=1)
    ax.axvline(0, color="0.7", linewidth=1)
    ax.axhline(0.05, color="crimson", linewidth=1, linestyle="--", alpha=0.5,
               label="planning evidence threshold (+0.05)")
    ax.set_xlabel("PPO − random  (learning signal)")
    ax.set_ylabel("PPO − greedy_dyn_2  (planning signal)")
    ax.set_title("Hardness frontier: per-cell (learning, planning) deltas")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Figure 3 — action diversity (entropy of action_freq) for PPO vs random
# ---------------------------------------------------------------------------

def _entropy_from_action_freq(action_freq: dict) -> float:
    counts = np.asarray([int(v) for v in action_freq.values() if isinstance(v, (int, float))],
                        dtype=np.float64)
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def plot_action_diversity(
    run_dirs: dict[str, Path],
    out_path: Path,
) -> None:
    """Horizontal bar chart of action_freq entropies per PPO config."""
    fig, ax = plt.subplots(figsize=(7, 4))
    cfgs: list[str] = []
    entropies: list[float] = []
    n_unique: list[int] = []
    for name, run_dir in run_dirs.items():
        f = run_dir / "action_freq.json"
        if not f.exists():
            log.warning("action_freq.json missing in %s, skipping", run_dir)
            continue
        af = json.loads(f.read_text())
        cfgs.append(name)
        entropies.append(_entropy_from_action_freq(af))
        n_unique.append(sum(1 for v in af.values() if isinstance(v, (int, float)) and v > 0))

    y_pos = np.arange(len(cfgs))
    bars = ax.barh(y_pos, entropies, color=plt.cm.tab10(np.linspace(0, 1, max(len(cfgs), 3))))
    for i, (b, n) in enumerate(zip(bars, n_unique, strict=True)):
        ax.text(b.get_width() + 0.02, b.get_y() + b.get_height() / 2,
                f"{n} unique", va="center", fontsize=8)
    ax.set_yticks(y_pos, cfgs, fontsize=9)
    ax.set_xlabel("Shannon entropy of action_freq (higher → more diverse)")
    ax.set_title("Action diversity per PPO config (training-side rollouts)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Figure 4 — seed variance violin plot
# ---------------------------------------------------------------------------

def plot_seed_variance(
    seed_aggregate_json: Path,
    out_path: Path,
    *,
    cells: tuple[str, ...] = (
        "k2_epsp25_bin6-8_splitood",
        "k2_epsp25_bin8-10_splitood",
        "k3_epsp25_bin6-8_splitood",
        "k3_epsp25_bin8-10_splitood",
    ),
) -> None:
    """One panel per cell. For each cell, plot per-seed PPO success points for B5 vs C2."""
    blob = _safe_load(seed_aggregate_json)
    if blob is None:
        log.warning("Seed aggregate JSON missing: %s — skipping", seed_aggregate_json)
        return
    b5 = blob.get("b5_v1ot_terminal_curric_1M", {})
    c2 = blob.get("c2_ror_corr010_terminal_curric_1M", {})

    fig, axes = plt.subplots(1, len(cells), figsize=(4 * len(cells), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for i, cell in enumerate(cells):
        ax = axes[i]
        b5_vals = b5.get(cell, {}).get("ppo_deterministic", {}).get("values", [])
        c2_vals = c2.get(cell, {}).get("ppo_deterministic", {}).get("values", [])
        if not b5_vals or not c2_vals:
            ax.set_title(f"{cell}\n(missing seeds)")
            continue
        ax.scatter([0] * len(b5_vals), b5_vals, color="tab:blue", s=60, alpha=0.7, label="B5 (V1 OT)")
        ax.scatter([1] * len(c2_vals), c2_vals, color="tab:orange", s=60, alpha=0.7, label="C2 (RoR)")
        ax.boxplot([b5_vals, c2_vals], positions=[0, 1], widths=0.4, showmeans=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["B5", "C2"])
        ax.set_ylim(-0.05, 1.05)
        # Compact cell title
        k, b = _cell_to_k_bin(cell)
        ax.set_title(f"K={k}, bin {b}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("PPO success rate")
    fig.suptitle("PPO success rate across seeds (B5 vs C2)", fontsize=12)
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Figure 5 — dynamics taxonomy (4-panel)
# ---------------------------------------------------------------------------

def plot_dynamics_taxonomy(
    rows: Iterable[dict],
    out_path: Path,
) -> None:
    """4-panel summary of dynamics fields: each panel shows one axis.

    Expected ``rows`` = iterable of dicts with keys:
       label, gate_val_margin, ood_pearson, beam_success, ppo_primary
    """
    rows = list(rows)
    if not rows:
        log.warning("plot_dynamics_taxonomy: no rows — skipping")
        return
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    panels = [
        ("Gate val margin",     "gate_val_margin",  0.030, "≥+0.030 = gate PASS"),
        ("OOD Pearson",         "ood_pearson",      0.40,  "≥0.40 = healthy"),
        ("Beam k=3 success",    "beam_success",     0.5,   "≥0.5 = controllable"),
        ("PPO at primary cell", "ppo_primary",      0.7,   "≥0.7 = strong"),
    ]
    for ax, (title, key, threshold, hint) in zip(axes, panels, strict=True):
        vals = [r.get(key, np.nan) for r in rows]
        bars = ax.bar(x, vals, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
        if threshold is not None:
            ax.axhline(threshold, color="crimson", linestyle="--", alpha=0.6, label=hint)
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("V2 Dynamics taxonomy (gate / OOD / reachability / RL)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Figure 6 — mean_final_distance distribution at primary cell
# ---------------------------------------------------------------------------

def plot_mean_d_distribution(
    rollouts_paths: dict[str, Path],
    out_path: Path,
) -> None:
    """Histogram of final distances per policy at primary cell. Reads rollouts.parquet."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rollouts_paths), 3)))
    any_data = False
    # Load z_ref once
    z_ref_path = Path("artifacts/vae/z_reference_centroid.npy")
    z_ref = np.load(z_ref_path).astype(np.float32) if z_ref_path.exists() else None
    for c_i, (name, p) in enumerate(rollouts_paths.items()):
        if not p.exists():
            log.warning("rollouts.parquet missing: %s — skipping policy", p)
            continue
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(p, columns=["episode_id", "step", "z_vector", "terminated"])
            df = table.to_pandas()
        except Exception as exc:
            log.warning("Could not read %s: %s", p, exc)
            continue
        if z_ref is None:
            continue
        # Compute distance per row from z_vector
        z_arr = np.asarray([np.asarray(zv, dtype=np.float32) for zv in df["z_vector"]],
                            dtype=np.float32)
        dists = np.linalg.norm(z_arr - z_ref[None, :], axis=1)
        df["dist_to_ref"] = dists
        # Take the last distance per episode (terminal)
        terminal = df.sort_values(["episode_id", "step"]).groupby("episode_id").tail(1)
        ax.hist(terminal["dist_to_ref"], bins=40, alpha=0.55, color=colors[c_i], label=name)
        any_data = True

    if not any_data:
        log.warning("plot_mean_d_distribution: no usable rollouts — skipping")
        plt.close(fig)
        return

    ax.axvline(3.166, color="crimson", linestyle="--", label="ε_p25 = 3.17")
    ax.set_xlabel("Final distance ‖z − z_ref‖")
    ax.set_ylabel("Episode count")
    ax.set_title("Final-distance distribution per policy (training-side rollouts)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Entry point — generate all six figures
# ---------------------------------------------------------------------------

def generate_all_figures(
    *,
    eval_dirs: dict[str, Path],
    run_dirs: dict[str, Path],
    seed_aggregate_json: Path,
    dynamics_taxonomy_rows: list[dict],
    rollouts_paths: dict[str, Path],
    out_dir: Path,
) -> list[Path]:
    """Generate the 6 V2 wrap-up figures. Returns the list of written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    paths = {
        "success_vs_K":         out_dir / "success_vs_K.png",
        "hardness_frontier":    out_dir / "hardness_frontier.png",
        "action_diversity":     out_dir / "action_diversity.png",
        "seed_variance":        out_dir / "seed_variance.png",
        "dynamics_taxonomy":    out_dir / "dynamics_taxonomy.png",
        "mean_d_distribution":  out_dir / "mean_d_distribution.png",
    }
    plot_success_vs_K(eval_dirs, paths["success_vs_K"])
    written.append(paths["success_vs_K"])
    plot_hardness_frontier(eval_dirs, paths["hardness_frontier"])
    written.append(paths["hardness_frontier"])
    plot_action_diversity(run_dirs, paths["action_diversity"])
    written.append(paths["action_diversity"])
    plot_seed_variance(seed_aggregate_json, paths["seed_variance"])
    written.append(paths["seed_variance"])
    plot_dynamics_taxonomy(dynamics_taxonomy_rows, paths["dynamics_taxonomy"])
    written.append(paths["dynamics_taxonomy"])
    plot_mean_d_distribution(rollouts_paths, paths["mean_d_distribution"])
    written.append(paths["mean_d_distribution"])
    return [p for p in written if p.exists()]
