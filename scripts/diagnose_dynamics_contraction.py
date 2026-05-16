"""scripts/diagnose_dynamics_contraction.py — quantify dynamics contraction toward z_ref.

For each sampled start state and each gene action, evaluate the trained dynamics model
once, compute the change in distance to the reference centroid, and aggregate:

- **fraction_improved** (strict, ``improvement > 0``): if this is ≈ 1.0, the model is
  globally contractive and the RL task is partly shaped by a dynamics artifact rather than
  policy quality. The user observed this informally on the 32D ``state_linear`` model.
- **per-gene** improvement table: which genes consistently move cells closer to z_ref.
- **random-action baseline**: the expected improvement if a uniform-random gene is chosen.
  Matches the user's informally observed +2.8 latent units on the 32D model.
- **improvement distribution**: histogram across all (start, gene) pairs.

This script **does not** modify dynamics training, RL, or any artifact under
``artifacts/<existing>/``. It only writes into ``cfg.diagnostic.out_dir``.

Usage
-----
::

    # 32D (default config — writes to artifacts/dynamics/contraction by default)
    python scripts/diagnose_dynamics_contraction.py --config-name default

    # 64D branch
    python scripts/diagnose_dynamics_contraction.py --config-name default \\
        paths.artifacts=$(pwd)/artifacts_64 vae.n_latent=64 \\
        +diagnostic.out_dir=artifacts_64/contraction \\
        +diagnostic.n_starts=1000

    # Custom min_start_distance + dedicated output dir
    python scripts/diagnose_dynamics_contraction.py --config-name default \\
        +diagnostic.out_dir=artifacts/contraction \\
        +diagnostic.n_starts=2000 \\
        +diagnostic.min_start_distance=8.0
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict


log = logging.getLogger(__name__)


# =============================================================================
# Pure-function core (testable without Hydra / real artifacts)
# =============================================================================


def _l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise L2 distance between ``a`` and ``b``; ``a`` shape (N, D), ``b`` shape (D,) or (N, D)."""
    diff = a - b
    return np.linalg.norm(diff, axis=-1)


def evaluate_contraction(
    starts: np.ndarray,
    z_ref: np.ndarray,
    dynamics_callable: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_genes: int,
    *,
    chunk_starts: int = 256,
) -> dict[str, np.ndarray]:
    """Evaluate every gene action on every start state and return per-(start, gene) arrays.

    Parameters
    ----------
    starts
        Shape ``(N, D)`` float32 — start latents.
    z_ref
        Shape ``(D,)`` float32 — reference centroid.
    dynamics_callable
        A callable that accepts ``(z_batch: (B, D), gene_idx: (B,))`` and returns ``mu: (B, D)``.
        The script wraps :class:`PerturbationDynamicsModel.forward` to expose this contract;
        tests pass a NumPy stub directly.
    n_genes
        Number of distinct gene actions. ``gene_idx`` runs over ``1..n_genes`` (1-indexed,
        per Contract 2).
    chunk_starts
        Number of start states processed per forward batch. Keeps peak memory bounded.

    Returns
    -------
    dict
        ``{"d_before": (N, n_genes), "d_after": (N, n_genes), "improvement": (N, n_genes)}``
        where ``improvement = d_before - d_after`` (positive = action moves cell closer to ref).
    """
    starts = np.asarray(starts, dtype=np.float32)
    z_ref = np.asarray(z_ref, dtype=np.float32)
    if starts.ndim != 2:
        raise ValueError(f"starts must be (N, D); got shape {starts.shape}")
    if z_ref.shape != (starts.shape[1],):
        raise ValueError(
            f"z_ref shape {z_ref.shape} incompatible with starts.shape[1]={starts.shape[1]}"
        )
    if n_genes < 1:
        raise ValueError(f"n_genes must be ≥ 1; got {n_genes}")

    n, d = starts.shape
    d_before = np.zeros((n, n_genes), dtype=np.float32)
    d_after = np.zeros((n, n_genes), dtype=np.float32)

    # Precompute d_before per start (same for all genes)
    per_start_d_before = _l2(starts, z_ref)
    d_before[:] = per_start_d_before[:, None]

    gene_indices = np.arange(1, n_genes + 1, dtype=np.int64)  # Contract-2 1-indexed
    for start_idx in range(0, n, chunk_starts):
        chunk = starts[start_idx : start_idx + chunk_starts]  # (B, D)
        b = chunk.shape[0]
        # Tile: each start replicated n_genes times; each gene_idx tiled across starts in chunk.
        z_batch = np.repeat(chunk, n_genes, axis=0)            # (B * n_genes, D)
        g_batch = np.tile(gene_indices, b)                      # (B * n_genes,)
        mu = dynamics_callable(z_batch, g_batch)                # (B * n_genes, D)
        z_next = z_batch + mu
        d_after_chunk = _l2(z_next, z_ref).reshape(b, n_genes)  # (B, n_genes)
        d_after[start_idx : start_idx + b] = d_after_chunk

    improvement = d_before - d_after
    return {"d_before": d_before, "d_after": d_after, "improvement": improvement}


def aggregate_contraction(
    improvement: np.ndarray,
    *,
    gene_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Roll up a ``(N, n_genes)`` improvement matrix into JSON-safe summary stats.

    A "strict" definition of improvement is used (``improvement > 0``). A zero-mu dynamics
    model produces a matrix of zeros and therefore yields ``fraction_improved == 0.0`` —
    this is the conservative, plan-locked definition.
    """
    if improvement.ndim != 2:
        raise ValueError(f"improvement must be (N, n_genes); got {improvement.shape}")
    n, n_genes = improvement.shape

    flat = improvement.reshape(-1).astype(np.float64)
    improved_mask = flat > 0.0
    summary = {
        "n_starts": int(n),
        "n_genes": int(n_genes),
        "n_pairs": int(flat.size),
        "fraction_improved": float(improved_mask.mean()) if flat.size else 0.0,
        "mean_improvement": float(flat.mean()) if flat.size else 0.0,
        "median_improvement": float(np.median(flat)) if flat.size else 0.0,
        "std_improvement": float(flat.std()) if flat.size else 0.0,
        "best_improvement": float(flat.max()) if flat.size else 0.0,
        "worst_improvement": float(flat.min()) if flat.size else 0.0,
        "p25_improvement": float(np.percentile(flat, 25)) if flat.size else 0.0,
        "p75_improvement": float(np.percentile(flat, 75)) if flat.size else 0.0,
    }

    # Per-gene aggregation (columns of the matrix)
    per_gene_rows: list[dict[str, Any]] = []
    if n > 0:
        improved_col = improvement > 0.0
        col_mean = improvement.mean(axis=0).astype(np.float64)
        col_median = np.median(improvement, axis=0).astype(np.float64)
        col_std = improvement.std(axis=0).astype(np.float64)
        col_frac = improved_col.mean(axis=0).astype(np.float64)
        for g in range(n_genes):
            sym = gene_symbols[g] if (gene_symbols is not None and g < len(gene_symbols)) else f"gene_{g+1}"
            per_gene_rows.append({
                "gene_symbol": sym,
                "gene_idx": int(g + 1),  # 1-indexed per Contract 2
                "fraction_improved": float(col_frac[g]),
                "mean_improvement": float(col_mean[g]),
                "median_improvement": float(col_median[g]),
                "std_improvement": float(col_std[g]),
                "n_starts": int(n),
            })
        # Sort: most contractive first
        per_gene_rows.sort(key=lambda r: -r["mean_improvement"])

    # Random-action baseline: under uniform random gene choice per start.
    # E[improvement | start s] = mean over genes of improvement(s, g)
    if n > 0 and n_genes > 0:
        per_start_mean = improvement.mean(axis=1).astype(np.float64)
        per_start_frac = (improvement > 0).mean(axis=1).astype(np.float64)
        random_baseline = {
            "mean_improvement_uniform_random": float(flat.mean()),
            "per_start_mean_stats": {
                "mean": float(per_start_mean.mean()),
                "median": float(np.median(per_start_mean)),
                "std": float(per_start_mean.std()),
                "p25": float(np.percentile(per_start_mean, 25)),
                "p75": float(np.percentile(per_start_mean, 75)),
                "min": float(per_start_mean.min()),
                "max": float(per_start_mean.max()),
            },
            "per_start_fraction_improved_stats": {
                "mean": float(per_start_frac.mean()),
                "median": float(np.median(per_start_frac)),
                "std": float(per_start_frac.std()),
            },
        }
    else:
        random_baseline = {
            "mean_improvement_uniform_random": 0.0,
            "per_start_mean_stats": {},
            "per_start_fraction_improved_stats": {},
        }

    return {
        "summary": summary,
        "per_gene": per_gene_rows,
        "random_action_baseline": random_baseline,
    }


# =============================================================================
# Hydra wrapper
# =============================================================================


def _torch_dynamics_callable(model: Any) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Wrap a :class:`PerturbationDynamicsModel` into a ``(z, gene_idx) -> mu`` NumPy callable.

    The model is invoked in ``eval()`` mode with ``torch.no_grad()`` on CPU. Only ``mu`` is
    returned; ``log_var`` is discarded for the diagnostic.
    """
    import torch

    def _call(z_np: np.ndarray, gene_idx_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.from_numpy(np.asarray(z_np, dtype=np.float32))
            g = torch.from_numpy(np.asarray(gene_idx_np, dtype=np.int64))
            out = model(z, g)
            if isinstance(out, tuple):
                # PerturbationDynamicsModel returns (z_next, mu, log_var). Prefer mu directly
                # so we don't accumulate floating-point error from (z_next - z).
                if len(out) >= 2:
                    mu = out[1]
                else:
                    mu = out[0] - z  # fall back to (z_next - z)
            else:
                mu = out - z
            return mu.detach().cpu().numpy().astype(np.float32)

    return _call


def _load_gene_symbols(cfg: Any) -> list[str] | None:
    """Read gene names from ``gene_vocab.json`` in Contract-1 order; ``None`` on failure."""
    try:
        with open(cfg.paths.vae_gene_vocab_json) as f:
            vocab = json.load(f)
        genes = list(vocab["genes"])
        return [str(g) for g in genes]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        log.warning("Could not load gene_vocab.json (%s); per_gene.csv will use gene_<idx>.", exc)
        return None


def _resolve_diagnostic_settings(cfg: DictConfig) -> dict[str, Any]:
    """Read ``cfg.diagnostic.*`` with code defaults; return a plain dict."""
    blob = cfg.get("diagnostic", None)
    d: dict[str, Any] = (
        OmegaConf.to_container(blob, resolve=True) if blob is not None else {}
    )
    # Defaults:
    out_dir = d.get("out_dir", None) or str(Path(cfg.paths.dynamics_dir) / "contraction")
    n_starts = int(d.get("n_starts", 1000))
    min_start_distance = d.get("min_start_distance", "auto")
    seed = int(d.get("seed", cfg.get("seed", 42)))
    chunk_starts = int(d.get("chunk_starts", 256))

    if n_starts < 1:
        raise ValueError(f"diagnostic.n_starts must be ≥ 1; got {n_starts}")

    return {
        "out_dir": out_dir,
        "n_starts": n_starts,
        "min_start_distance": min_start_distance,
        "seed": seed,
        "chunk_starts": chunk_starts,
    }


def _resolve_min_distance(min_start_distance: Any, epsilon: float) -> float | None:
    """Mirror the env factory's ``"auto" | "none" | float`` resolution."""
    if isinstance(min_start_distance, str):
        low = min_start_distance.lower()
        if low == "auto":
            return float(epsilon)
        if low == "none":
            return None
        return float(min_start_distance)
    if min_start_distance is None:
        return None
    return float(min_start_distance)


def _write_outputs(
    out_dir: Path,
    agg: dict[str, Any],
    improvement: np.ndarray,
    *,
    title_suffix: str,
) -> None:
    """Write summary.json, per_gene.csv, random_action_baseline.json, improvement_hist.png."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(agg["summary"], f, indent=2)

    with open(out_dir / "random_action_baseline.json", "w") as f:
        json.dump(agg["random_action_baseline"], f, indent=2)

    per_gene_rows = agg["per_gene"]
    fields = (
        ["gene_symbol", "gene_idx", "fraction_improved", "mean_improvement",
         "median_improvement", "std_improvement", "n_starts"]
    )
    with open(out_dir / "per_gene.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in per_gene_rows:
            writer.writerow(row)

    # Histogram (matplotlib is already a project dep)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        flat = improvement.reshape(-1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(flat, bins=80, alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("improvement = ||z - z_ref|| - ||z + mu - z_ref||")
        ax.set_ylabel("count")
        ax.set_title(f"Dynamics contraction — {title_suffix}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "improvement_hist.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write improvement_hist.png: %s", exc)


def _write_diagnostic_metadata(
    cfg: DictConfig,
    out_dir: Path,
    *,
    n_starts: int,
    n_genes: int,
    min_start_distance_used: float | None,
    epsilon_used: float,
    epsilon_source: str,
    summary: dict[str, Any],
    diagnostic_seed: int,
) -> None:
    """Write a focused metadata.json. Reuses _write_run_metadata for shared provenance fields."""
    from src.rl.train_ppo import _write_run_metadata

    _write_run_metadata(
        cfg,
        out_dir,
        deterministic=None,
        n_episodes=None,
        extras={
            "stage": "diagnose_dynamics_contraction",
            "diagnostic": {
                "n_starts": int(n_starts),
                "n_genes": int(n_genes),
                "min_start_distance_used": (
                    None if min_start_distance_used is None else float(min_start_distance_used)
                ),
                "epsilon_used": float(epsilon_used),
                "epsilon_source": epsilon_source,
                "seed": int(diagnostic_seed),
            },
            "results": {
                "fraction_improved": summary.get("fraction_improved"),
                "mean_improvement": summary.get("mean_improvement"),
                "median_improvement": summary.get("median_improvement"),
                "best_improvement": summary.get("best_improvement"),
                "worst_improvement": summary.get("worst_improvement"),
            },
        },
    )


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra entry point."""
    from src.rl.environment import _build_start_pool, _load_dynamics_model, resolve_epsilon
    from src.utils.device import device_summary
    from src.utils.seeding import set_seed

    settings = _resolve_diagnostic_settings(cfg)
    set_seed(int(settings["seed"]))
    print(device_summary())

    # Propagate min_start_distance so that subsequent metadata.json reports the value used,
    # and so _build_start_pool sees the diagnostic's choice (not the RL config's).
    with open_dict(cfg):
        cfg.rl.env.min_start_distance = settings["min_start_distance"]

    out_dir = Path(settings["out_dir"])
    n_starts = settings["n_starts"]
    seed = settings["seed"]

    if cfg.get("dry_run", False):
        print("DRY RUN — would run contraction diagnostic:")
        print(f"  out_dir            = {out_dir}")
        print(f"  n_starts           = {n_starts}")
        print(f"  min_start_distance = {settings['min_start_distance']}")
        print(f"  seed               = {seed}")
        print(f"  dynamics_model     = {cfg.paths.dynamics_model}")
        print(f"  vae_latents        = {cfg.paths.vae_latents_h5ad}")
        return 0

    # Resolve artifacts
    z_ref = np.load(str(cfg.paths.vae_z_reference_centroid)).astype(np.float32)
    epsilon_used, epsilon_source = resolve_epsilon(cfg)
    log.info("Diagnostic ε = %.4f (source: %s)", epsilon_used, epsilon_source)

    gene_symbols = _load_gene_symbols(cfg)
    n_genes = len(gene_symbols) if gene_symbols is not None else int(
        json.load(open(cfg.paths.vae_gene_vocab_json))["n_genes"]
    )

    # Build start pool (epsilon-aware filter — same semantics as the env)
    min_distance = _resolve_min_distance(settings["min_start_distance"], epsilon_used)
    pool = _build_start_pool(cfg, min_distance=min_distance)
    if len(pool) == 0:
        log.error("Start pool is empty after filtering; aborting.")
        return 2

    rng = np.random.default_rng(seed)
    if len(pool) > n_starts:
        idx = rng.choice(len(pool), size=n_starts, replace=False)
        starts = pool[idx].astype(np.float32)
    else:
        starts = pool.astype(np.float32)
        n_starts = len(starts)
        log.info("Pool smaller than n_starts; using all %d cells.", n_starts)

    log.info(
        "Loading dynamics model from %s (n_genes=%d, vae_n_latent=%d)",
        cfg.paths.dynamics_model, n_genes, int(cfg.vae.n_latent),
    )
    model = _load_dynamics_model(cfg, allow_untrained=False)

    log.info("Evaluating contraction: %d starts × %d genes = %d pairs",
             n_starts, n_genes, n_starts * n_genes)
    dyn_call = _torch_dynamics_callable(model)
    arrays = evaluate_contraction(
        starts=starts,
        z_ref=z_ref,
        dynamics_callable=dyn_call,
        n_genes=n_genes,
        chunk_starts=settings["chunk_starts"],
    )

    agg = aggregate_contraction(arrays["improvement"], gene_symbols=gene_symbols)

    title = (
        f"n_latent={int(cfg.vae.n_latent)}, n_starts={n_starts}, n_genes={n_genes}, "
        f"min_dist={min_distance if min_distance is not None else 'none'}"
    )
    _write_outputs(out_dir, agg, arrays["improvement"], title_suffix=title)

    _write_diagnostic_metadata(
        cfg, out_dir,
        n_starts=n_starts,
        n_genes=n_genes,
        min_start_distance_used=min_distance,
        epsilon_used=epsilon_used,
        epsilon_source=epsilon_source,
        summary=agg["summary"],
        diagnostic_seed=seed,
    )

    s = agg["summary"]
    print(
        f"Done. fraction_improved={s['fraction_improved']:.4f}  "
        f"mean_improvement={s['mean_improvement']:.4f}  "
        f"best={s['best_improvement']:.4f}  worst={s['worst_improvement']:.4f}  "
        f"→ {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
