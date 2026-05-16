"""P0A.2 per-gene action-contraction diagnostic for frozen V1 dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.diagnose_dynamics_contraction import _torch_dynamics_callable, evaluate_contraction
from src.analysis.gate_breakdown import load_dynamics_model
from src.analysis.metrics import gini_coefficient, shannon_entropy


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _load_action_freq(path: str | Path | None) -> dict[str, int]:
    if path is None or not Path(path).exists():
        return {}
    with open(path) as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def _load_start_pool(vae_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    import anndata as ad

    with open(vae_dir / "gene_vocab.json") as f:
        vocab = json.load(f)
    genes = [str(g) for g in vocab["genes"]]
    adata = ad.read_h5ad(vae_dir / "latents.h5ad")
    z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    pert_idx = np.asarray(adata.obs["perturbation_idx"].values)
    return z[pert_idx != 0], np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32), genes, int(vocab["noop_idx"])


def summarize_per_gene_contraction(
    improvement: np.ndarray,
    genes: list[str],
    ppo_action_freq: dict[str, int] | None = None,
    top_n: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate the ``(n_starts, n_genes)`` improvement matrix by gene."""
    rows: list[dict[str, Any]] = []
    for j, gene in enumerate(genes):
        vals = improvement[:, j].astype(np.float64)
        rows.append({
            "gene_symbol": gene,
            "gene_idx": int(j + 1),
            "mean_improvement": float(np.mean(vals)),
            "median_improvement": float(np.median(vals)),
            "std_improvement": float(np.std(vals)),
            "fraction_positive": float(np.mean(vals > 0.0)),
            "n_starts": int(improvement.shape[0]),
            "ppo_action_count": int((ppo_action_freq or {}).get(gene, 0)),
        })
    rows.sort(key=lambda r: r["mean_improvement"], reverse=True)
    means = np.asarray([r["mean_improvement"] for r in rows], dtype=np.float64)
    entropy = shannon_entropy(np.abs(means))
    top_contract = rows[:top_n]
    bottom_contract = sorted(rows, key=lambda r: r["mean_improvement"])[:top_n]
    ppo_top = [
        g for g, _ in sorted(
            ((g, c) for g, c in (ppo_action_freq or {}).items() if g != "NO_OP"),
            key=lambda x: -x[1],
        )[:top_n]
    ]
    top_contract_genes = {r["gene_symbol"] for r in top_contract}
    overlap = sorted(top_contract_genes & set(ppo_top))
    summary = {
        "n_starts": int(improvement.shape[0]),
        "n_genes": int(improvement.shape[1]),
        "gini_mean_improvement": gini_coefficient(means),
        "shannon_entropy_abs_mean": entropy,
        "max_entropy": float(np.log(max(len(means), 1))),
        "entropy_fraction_of_max": float(entropy / np.log(len(means))) if len(means) > 1 else 0.0,
        "top_contracting_genes": top_contract,
        "bottom_contracting_genes": bottom_contract,
        "ppo_top_genes": ppo_top,
        "top_contracting_ppo_top_overlap": overlap,
        "top_contracting_ppo_top_overlap_count": int(len(overlap)),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "gene_symbol", "gene_idx", "mean_improvement", "median_improvement",
        "std_improvement", "fraction_positive", "n_starts", "ppo_action_count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--dynamics_dir", required=True)
    parser.add_argument("--vae_dir", required=True)
    parser.add_argument("--n_starts", type=int, default=500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_starts", type=int, default=128)
    parser.add_argument(
        "--ppo_action_freq",
        default="artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic/action_freq.json",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    starts_pool, z_ref, genes, _noop_idx = _load_start_pool(Path(args.vae_dir))
    rng = np.random.default_rng(args.seed)
    if len(starts_pool) > args.n_starts:
        idx = rng.choice(len(starts_pool), size=args.n_starts, replace=False)
        starts = starts_pool[idx]
    else:
        starts = starts_pool

    model = load_dynamics_model(args.dynamics_dir)
    arrays = evaluate_contraction(
        starts=starts,
        z_ref=z_ref,
        dynamics_callable=_torch_dynamics_callable(model),
        n_genes=len(genes),
        chunk_starts=int(args.chunk_starts),
    )
    rows, summary = summarize_per_gene_contraction(
        arrays["improvement"],
        genes,
        ppo_action_freq=_load_action_freq(args.ppo_action_freq),
        top_n=10,
    )
    csv_path = out_dir / "per_gene_contraction.csv"
    summary_path = out_dir / "per_gene_contraction_summary.json"
    _write_csv(csv_path, rows)
    payload = {
        "stage": "p0a_action_contraction",
        "config_name": args.config_name,
        "git_commit": _git_commit(),
        "read_only_mode": True,
        "source_paths": {
            "dynamics_dir": str(args.dynamics_dir),
            "vae_dir": str(args.vae_dir),
            "ppo_action_freq": str(args.ppo_action_freq),
        },
        **summary,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "gini_mean_improvement": summary["gini_mean_improvement"],
        "entropy_fraction_of_max": summary["entropy_fraction_of_max"],
        "top_contracting_genes": [r["gene_symbol"] for r in summary["top_contracting_genes"]],
        "bottom_contracting_genes": [r["gene_symbol"] for r in summary["bottom_contracting_genes"]],
        "ppo_overlap": summary["top_contracting_ppo_top_overlap"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
