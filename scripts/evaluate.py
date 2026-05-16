"""scripts/evaluate.py — full Phase-5 evaluation suite.

End-to-end orchestrator that produces every reporting artifact under ``artifacts/eval/``.
The script does NOT retrain or modify any artifact under ``artifacts/<component>/`` —
it only *reads* the JSON / parquet / npy outputs of the training pipeline and computes
report-layer metrics.

Pipeline (in order):

1. **Aggregator** (:mod:`src.analysis.aggregate`) → ``summary.json``, ``results_table.md``,
   ``caveats.md``.
2. **Latent quality** (:func:`src.analysis.latent_space.analyze_latent_quality` over
   ``artifacts/vae/latents.h5ad``) → ``latent_quality.json``.
3. **DepMap enrichment** (:func:`src.analysis.depmap_validation.run_depmap_enrichment`
   over the PPO deterministic ``action_freq.json``) → ``depmap_enrichment.csv``.
4. **Composite index** → ``evaluate_report.json`` listing the above for the visualizer.

This script never auto-edits ``README.md``. ``results_table.md`` is the human-paste source.

Usage
-----
::

    python scripts/evaluate.py --config-name default rl.train.skip_gate=true
    python scripts/evaluate.py --config-name default rl.train.skip_gate=true \\
        +evaluate.skip_depmap=true     # skip DepMap if data/processed parquet missing
    python scripts/evaluate.py --config-name default rl.train.skip_gate=true \\
        +evaluate.skip_latent_quality=true
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _resolve_evaluate_settings(cfg: DictConfig) -> dict[str, Any]:
    """Read ``cfg.get("evaluate", {})`` with code defaults; return a plain dict.

    Knobs (all override via ``+evaluate.*``):
      - ``skip_aggregate``       (default False): skip P1A aggregator stage.
      - ``skip_latent_quality``  (default False): skip silhouette/ARI on latents.h5ad.
      - ``skip_depmap``          (default False): skip DepMap enrichment.
      - ``top_k``                (default 20): top-K genes fed to ``run_depmap_enrichment``.
      - ``n_null``               (default 1000): permutations for the DepMap null comparison.
      - ``panel_dir``            (default ``data/panels``): curated gene panels root.
    """
    blob = cfg.get("evaluate", None)
    from omegaconf import OmegaConf
    d: dict[str, Any] = (
        OmegaConf.to_container(blob, resolve=True) if blob is not None else {}
    )
    return {
        "skip_aggregate": bool(d.get("skip_aggregate", False)),
        "skip_latent_quality": bool(d.get("skip_latent_quality", False)),
        "skip_depmap": bool(d.get("skip_depmap", False)),
        "skip_depmap_compare": bool(d.get("skip_depmap_compare", False)),
        "top_k": int(d.get("top_k", 20)),
        "n_null": int(d.get("n_null", 1000)),
        "n_permutations": int(d.get("n_permutations", 10_000)),
        "panel_dir": d.get("panel_dir", "data/panels"),
    }


def _resolve_path(p: str | None, repo_root: Path) -> Path | None:
    if not p:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (repo_root / pp)


# =============================================================================
# Stages
# =============================================================================


def run_aggregator(cfg: DictConfig, out_dir: Path) -> Path | None:
    """Run the aggregator stage in-process; return the path to summary.json (or None)."""
    from scripts.aggregate_eval import run_aggregate

    rc = run_aggregate(cfg)
    if rc != 0:
        log.error("Aggregator returned non-zero exit code %d", rc)
        return None
    summary_path = out_dir / "summary.json"
    return summary_path if summary_path.exists() else None


def run_latent_quality(cfg: DictConfig, out_dir: Path) -> Path | None:
    """Compute silhouette + ARI on the VAE latents; write ``latent_quality.json``."""
    from src.analysis.latent_space import analyze_latent_quality

    latents_path = _resolve_path(
        str(cfg.paths.vae_latents_h5ad), Path(cfg.paths.root),
    )
    if latents_path is None or not latents_path.exists():
        log.warning("Latents not found at %s — skipping latent_quality stage.", latents_path)
        return None

    import anndata as ad

    log.info("Loading latents from %s", latents_path)
    adata = ad.read_h5ad(str(latents_path))
    if "X_scVI" not in adata.obsm:
        log.warning("X_scVI not in adata.obsm — skipping latent_quality stage.")
        return None
    if "perturbation_idx" not in adata.obs.columns:
        log.warning("perturbation_idx not in adata.obs — skipping latent_quality stage.")
        return None

    result = analyze_latent_quality(adata, latent_key="X_scVI", save_dir=None)
    # JSON-safe
    safe = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in result.items()}
    safe["n_cells"] = int(adata.n_obs)
    safe["n_latent"] = int(adata.obsm["X_scVI"].shape[1])
    safe["source_h5ad"] = str(latents_path)

    out_path = out_dir / "latent_quality.json"
    with open(out_path, "w") as f:
        json.dump(safe, f, indent=2, default=str)
    log.info("Wrote latent_quality → %s (silhouette=%.4f ARI=%.4f)",
             out_path, safe.get("silhouette", float("nan")), safe.get("ari", float("nan")))
    return out_path


def run_depmap_enrichment_stage(
    cfg: DictConfig,
    settings: dict[str, Any],
    out_dir: Path,
    summary_blob: dict[str, Any] | None,
) -> Path | None:
    """Run DepMap enrichment on the PPO deterministic action_freq.json.

    Uses the path layout the aggregator already resolved (``summary.json[ppo_det_dir]``
    is captured via the aggregator's ``aggregate`` settings). Falls back to the canonical
    default if the aggregator wasn't run.
    """
    from src.analysis.depmap_validation import (
        load_depmap_k562,
        load_gene_panel_manifest,
        load_gene_panels,
        run_depmap_enrichment,
    )

    repo_root = Path(cfg.paths.root)

    # Resolve PPO det dir — prefer the aggregator's resolved path; else use the default.
    ppo_det_dir = None
    if summary_blob is not None:
        # The aggregator embeds the run_dir via the run summary's metadata.policy_path
        # but the simpler approach is to read it from the aggregator's settings on cfg.
        agg_blob = cfg.get("aggregate", None)
        if agg_blob is not None:
            from omegaconf import OmegaConf
            agg_d = OmegaConf.to_container(agg_blob, resolve=True) or {}
            if "ppo_det_dir" in agg_d:
                ppo_det_dir = _resolve_path(agg_d["ppo_det_dir"], repo_root)
    if ppo_det_dir is None:
        ppo_det_dir = _resolve_path(
            "artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic",
            repo_root,
        )

    action_freq_path = ppo_det_dir / "action_freq.json" if ppo_det_dir else None
    if action_freq_path is None or not action_freq_path.exists():
        log.warning("action_freq.json not found at %s — skipping DepMap stage.", action_freq_path)
        return None

    depmap_path = _resolve_path(str(cfg.paths.depmap_k562_parquet), repo_root)
    if depmap_path is None or not depmap_path.exists():
        log.warning("DepMap parquet not found at %s — skipping DepMap stage.", depmap_path)
        return None

    latents_path = _resolve_path(str(cfg.paths.vae_latents_h5ad), repo_root)
    if latents_path is None or not latents_path.exists():
        log.warning("Latents not found at %s — skipping DepMap stage.", latents_path)
        return None

    with open(action_freq_path) as f:
        rl_action_freq = {str(k): int(v) for k, v in json.load(f).items()}
    rl_action_freq.pop("NO_OP", None)  # NO_OP is not a gene; drop before enrichment

    # HVG universe from the latents file's var_names
    import anndata as ad
    log.info("Reading background gene list from %s", latents_path)
    adata = ad.read_h5ad(str(latents_path))
    background_genes = [str(g) for g in adata.var_names]
    log.info("Background HVG universe: %d genes", len(background_genes))
    expression_mean_per_gene: dict[str, float] | None = None
    try:
        x_mean = adata.X.mean(axis=0)
        means = np.asarray(x_mean).ravel()
        if means.shape[0] == len(background_genes):
            expression_mean_per_gene = {
                gene: float(value) for gene, value in zip(background_genes, means, strict=True)
                if np.isfinite(float(value))
            }
            log.info(
                "Expression-matched null available for %d/%d background genes.",
                len(expression_mean_per_gene),
                len(background_genes),
            )
        else:
            log.warning(
                "Expression mean shape mismatch (%s vs %d genes); using size-matched null.",
                means.shape,
                len(background_genes),
            )
    except Exception as exc:  # pragma: no cover - depends on backed/sparse AnnData variants
        log.warning("Could not derive expression means from latents.h5ad: %s", exc)

    log.info("Loading DepMap K562 chronos from %s", depmap_path)
    chronos_df = load_depmap_k562(depmap_path)

    panel_dir = _resolve_path(settings["panel_dir"], repo_root)
    panels = load_gene_panels(panel_dir) if panel_dir is not None else {}
    panel_sources = (
        load_gene_panel_manifest(panel_dir, required_collections=("hallmark", "lineage"))
        if panel_dir is not None
        else {}
    )
    log.info("Loaded %d curated gene panels from %s",
             len(panels), panel_dir if panel_dir else "(none)")

    out_csv = out_dir / "depmap_enrichment.csv"
    log.info("Running DepMap enrichment (top_k=%d, n_null=%d) → %s",
             settings["top_k"], settings["n_null"], out_csv)
    df = run_depmap_enrichment(
        rl_action_freq=rl_action_freq,
        background_genes=background_genes,
        panels=panels,
        chronos_df=chronos_df,
        top_k=settings["top_k"],
        n_null=settings["n_null"],
        expression_mean_per_gene=expression_mean_per_gene,
        panel_sources=panel_sources,
        out_path=out_csv,
    )
    if df is None or len(df) == 0:
        log.warning("DepMap enrichment produced no rows.")
    else:
        q_vals = [q for q in df["q_value"].to_list() if q is not None]
        n_sig = int(sum(float(q) < 0.05 for q in q_vals))
        log.info("DepMap enrichment: %d rows, %d with q < 0.05", len(df), n_sig)
    return out_csv if out_csv.exists() else None


def run_depmap_comparison_stage(
    cfg: DictConfig,
    settings: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path | None]:
    """Compare Chronos score distributions: PPO det/stoch, random, action universe.

    Reads the three canonical action_freq.json files from the V1 eval layout.
    Writes depmap_gene_level_scores.csv, depmap_comparison_summary.json,
    depmap_comparison_table.md into ``out_dir``.
    Returns a dict of output paths (value=None if not written).
    """
    from src.analysis.depmap_validation import (
        load_depmap_k562,
        run_depmap_comparison,
    )

    repo_root = Path(cfg.paths.root)

    # Canonical V1 eval dirs (same logic as the enrichment stage)
    agg_blob = cfg.get("aggregate", None)
    ppo_det_dir: Path | None = None
    if agg_blob is not None:
        from omegaconf import OmegaConf
        agg_d = OmegaConf.to_container(agg_blob, resolve=True) or {}
        if "ppo_det_dir" in agg_d:
            ppo_det_dir = _resolve_path(agg_d["ppo_det_dir"], repo_root)
    if ppo_det_dir is None:
        ppo_det_dir = _resolve_path(
            "artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic",
            repo_root,
        )

    ppo_stoch_dir = _resolve_path(
        "artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_stochastic",
        repo_root,
    )
    random_dir = _resolve_path(
        "artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/random_baseline",
        repo_root,
    )

    def _load_freq(d: Path | None) -> dict[str, int] | None:
        p = d / "action_freq.json" if d else None
        if p is None or not p.exists():
            return None
        with open(p) as f:
            return {str(k): int(v) for k, v in json.load(f).items()}

    ppo_det_freq = _load_freq(ppo_det_dir)
    ppo_stoch_freq = _load_freq(ppo_stoch_dir)
    random_freq = _load_freq(random_dir)

    if ppo_det_freq is None or random_freq is None:
        log.warning("PPO det or random action_freq.json missing — skipping DepMap comparison.")
        return {"gene_level_csv": None, "summary_json": None, "table_md": None}
    if ppo_stoch_freq is None:
        log.warning("PPO stoch action_freq.json missing — using det freq as fallback.")
        ppo_stoch_freq = ppo_det_freq

    # DepMap parquet
    depmap_path = _resolve_path(str(cfg.paths.depmap_k562_parquet), repo_root)
    if depmap_path is None or not depmap_path.exists():
        log.warning("DepMap parquet missing — skipping DepMap comparison.")
        return {"gene_level_csv": None, "summary_json": None, "table_md": None}

    chronos_df = load_depmap_k562(depmap_path)

    # Background genes from latents.h5ad
    latents_path = _resolve_path(str(cfg.paths.vae_latents_h5ad), repo_root)
    background_genes: list[str] = []
    if latents_path is not None and latents_path.exists():
        import anndata as ad
        adata = ad.read_h5ad(str(latents_path))
        background_genes = [str(g) for g in adata.var_names]

    # Action universe from gene_vocab.json
    vocab_path = _resolve_path(str(cfg.paths.get("vae_gene_vocab_json", "artifacts/vae/gene_vocab.json")), repo_root)
    if vocab_path is None or not vocab_path.exists():
        vocab_path = repo_root / "artifacts" / "vae" / "gene_vocab.json"
    action_universe: list[str] | None = None
    if vocab_path.exists():
        with open(vocab_path) as f:
            vocab = json.load(f)
        action_universe = vocab.get("genes")

    seed = int(cfg.get("seed", 42))
    log.info(
        "Running DepMap comparison (top_k=%d, n_permutations=%d) …",
        settings["top_k"], settings["n_permutations"],
    )
    _, _, _ = run_depmap_comparison(
        ppo_det_freq=ppo_det_freq,
        ppo_stoch_freq=ppo_stoch_freq,
        random_freq=random_freq,
        background_genes=background_genes,
        action_universe=action_universe,
        chronos_df=chronos_df,
        top_k=settings["top_k"],
        n_permutations=settings["n_permutations"],
        seed=seed,
        out_dir=out_dir,
    )

    return {
        "gene_level_csv": out_dir / "depmap_gene_level_scores.csv",
        "summary_json": out_dir / "depmap_comparison_summary.json",
        "table_md": out_dir / "depmap_comparison_table.md",
    }


# =============================================================================
# Entry point
# =============================================================================


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra entry point."""
    settings = _resolve_evaluate_settings(cfg)
    repo_root = Path(cfg.paths.root) if hasattr(cfg.paths, "root") else Path.cwd()
    out_dir = Path(cfg.paths.eval_dir) if hasattr(cfg.paths, "eval_dir") else (repo_root / "artifacts" / "eval")

    if cfg.get("dry_run", False):
        print("DRY RUN — would run the following stages:")
        print(f"  out_dir              = {out_dir}")
        print(f"  skip_aggregate       = {settings['skip_aggregate']}")
        print(f"  skip_latent_quality  = {settings['skip_latent_quality']}")
        print(f"  skip_depmap          = {settings['skip_depmap']}")
        print(f"  skip_depmap_compare  = {settings['skip_depmap_compare']}")
        print(f"  top_k                = {settings['top_k']}")
        print(f"  n_null               = {settings['n_null']}")
        print(f"  n_permutations       = {settings['n_permutations']}")
        print(f"  panel_dir            = {settings['panel_dir']}")
        return 0

    from src.utils.device import device_summary
    from src.utils.seeding import set_seed

    set_seed(int(cfg.get("seed", 42)))
    print(device_summary())

    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path: Path | None = None
    summary_blob: dict[str, Any] | None = None
    if not settings["skip_aggregate"]:
        log.info("=== Stage 1/4: aggregator ===")
        summary_path = run_aggregator(cfg, out_dir)
        if summary_path is not None and summary_path.exists():
            with open(summary_path) as f:
                summary_blob = json.load(f)

    latent_quality_path: Path | None = None
    if not settings["skip_latent_quality"]:
        log.info("=== Stage 2/4: latent quality ===")
        try:
            latent_quality_path = run_latent_quality(cfg, out_dir)
        except Exception as exc:
            log.warning("Latent quality stage failed (%s) — continuing.", exc)

    depmap_csv_path: Path | None = None
    if not settings["skip_depmap"]:
        log.info("=== Stage 3/4: DepMap enrichment ===")
        try:
            depmap_csv_path = run_depmap_enrichment_stage(cfg, settings, out_dir, summary_blob)
        except Exception as exc:
            log.warning("DepMap stage failed (%s) — continuing.", exc)

    depmap_compare_paths: dict[str, Path | None] = {}
    if not settings["skip_depmap_compare"]:
        log.info("=== Stage 4/4: DepMap gene-score comparison ===")
        try:
            depmap_compare_paths = run_depmap_comparison_stage(cfg, settings, out_dir)
        except Exception as exc:
            log.warning("DepMap comparison stage failed (%s) — continuing.", exc)

    # Composite index — points to everything the visualizer will consume.
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "summary_json": str(summary_path) if summary_path is not None else None,
        "results_table_md": str(out_dir / "results_table.md") if (out_dir / "results_table.md").exists() else None,
        "caveats_md": str(out_dir / "caveats.md") if (out_dir / "caveats.md").exists() else None,
        "latent_quality_json": str(latent_quality_path) if latent_quality_path is not None else None,
        "depmap_enrichment_csv": str(depmap_csv_path) if depmap_csv_path is not None else None,
        "depmap_gene_level_scores_csv": str(depmap_compare_paths.get("gene_level_csv"))
            if depmap_compare_paths.get("gene_level_csv") else None,
        "depmap_comparison_summary_json": str(depmap_compare_paths.get("summary_json"))
            if depmap_compare_paths.get("summary_json") else None,
        "depmap_comparison_table_md": str(depmap_compare_paths.get("table_md"))
            if depmap_compare_paths.get("table_md") else None,
        "stages": {
            "aggregate": (not settings["skip_aggregate"]),
            "latent_quality": (not settings["skip_latent_quality"]),
            "depmap_enrichment": (not settings["skip_depmap"]),
            "depmap_compare": (not settings["skip_depmap_compare"]),
        },
    }
    report_path = out_dir / "evaluate_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote composite report → %s", report_path)

    # Headline to stdout so the user can see what landed.
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {out_dir / 'results_table.md'}")
    print(f"Wrote: {out_dir / 'caveats.md'}")
    if latent_quality_path:
        print(f"Wrote: {latent_quality_path}")
    if depmap_csv_path:
        print(f"Wrote: {depmap_csv_path}")
    for key in ("gene_level_csv", "summary_json", "table_md"):
        p = depmap_compare_paths.get(key)
        if p and Path(p).exists():
            print(f"Wrote: {p}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
