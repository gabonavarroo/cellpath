"""P0A.4 read-only OT pairing-noise scoping."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def compute_pairing_noise(pairs: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute per-gene residual Delta-z variance and the residual/total ratio."""
    z_ctrl = np.asarray(pairs["z_ctrl"], dtype=np.float32)
    gene_idx = np.asarray(pairs["gene_idx"], dtype=np.int32)
    z_pert = np.asarray(pairs["z_pert"], dtype=np.float32)
    delta = z_pert - z_ctrl
    global_mean = delta.mean(axis=0)

    rows: list[dict[str, Any]] = []
    for g in sorted(np.unique(gene_idx).astype(int).tolist()):
        d = delta[gene_idx == g]
        if len(d) == 0:
            continue
        gene_mean = d.mean(axis=0)
        residual = d - gene_mean
        around_global = d - global_mean
        residual_var = float(np.mean(np.sum(residual * residual, axis=1)))
        total_var = float(np.mean(np.sum(around_global * around_global, axis=1)))
        mean_delta_signal = float(np.sum((gene_mean - global_mean) ** 2))
        denom = total_var if total_var > 0.0 else residual_var + mean_delta_signal
        ratio = float(residual_var / denom) if denom > 0.0 else 0.0
        rows.append({
            "gene_idx": int(g),
            "n_pairs": int(len(d)),
            "within_delta_variance": residual_var,
            "mean_delta_residual_variance": residual_var,
            "total_delta_variance": total_var,
            "mean_delta_signal": mean_delta_signal,
            "noise_ratio": max(0.0, min(1.0, ratio)),
        })

    ratios = np.asarray([r["noise_ratio"] for r in rows], dtype=np.float64)
    summary = {
        "n_genes": int(len(rows)),
        "median_noise_ratio": float(np.median(ratios)) if len(ratios) else 0.0,
        "mean_noise_ratio": float(np.mean(ratios)) if len(ratios) else 0.0,
        "p25_noise_ratio": float(np.percentile(ratios, 25)) if len(ratios) else 0.0,
        "p75_noise_ratio": float(np.percentile(ratios, 75)) if len(ratios) else 0.0,
        "max_noise_ratio": float(np.max(ratios)) if len(ratios) else 0.0,
    }
    return rows, summary


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _write_md(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    worst = sorted(rows, key=lambda r: r["noise_ratio"], reverse=True)[:10]
    lines = [
        "# P0A.4 Pairing-Noise Scoping",
        "",
        f"- Genes measured: {summary['n_genes']}",
        f"- Median residual/total Delta-z variance ratio: {summary['median_noise_ratio']:.4f}",
        f"- Mean residual/total Delta-z variance ratio: {summary['mean_noise_ratio']:.4f}",
        "",
        "High ratios mean the OT pseudo-pair target is dominated by within-gene residual variation, ",
        "which limits what any deterministic dynamics model can predict beyond a per-gene mean.",
        "",
        "## Highest Noise-Ratio Genes",
        "",
        "| gene_idx | n_pairs | noise_ratio | within_delta_variance | total_delta_variance |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in worst:
        lines.append(
            f"| {row['gene_idx']} | {row['n_pairs']} | {row['noise_ratio']:.4f} | "
            f"{row['within_delta_variance']:.4f} | {row['total_delta_variance']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train = _load_npz(Path(args.pairs_dir) / "train_pairs.npz")
    rows, summary = compute_pairing_noise(train)
    payload = {
        "stage": "p0a_pairing_noise",
        "config_name": args.config_name,
        "git_commit": _git_commit(),
        "read_only_mode": True,
        "summary": summary,
        "per_gene": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    md_path = out_path.with_suffix(".md")
    _write_md(md_path, rows, summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
