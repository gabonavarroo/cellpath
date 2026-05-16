"""Compute V2 epsilon percentiles from frozen VAE control latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_epsilon_percentile(vae_dir: str | Path, p: float) -> dict[str, Any]:
    """Return the ``p``th percentile of control-cell L2 distances to ``z_ref``."""
    import anndata as ad

    vae_dir = Path(vae_dir)
    adata = ad.read_h5ad(vae_dir / "latents.h5ad")
    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)
    z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    if "perturbation_idx" in adata.obs:
        ctrl = np.asarray(adata.obs["perturbation_idx"].values) == 0
    elif "perturbation" in adata.obs:
        ctrl = np.asarray(adata.obs["perturbation"].astype(str).values) == "ctrl"
    else:
        raise ValueError("latents.h5ad must contain obs['perturbation_idx'] or obs['perturbation']")
    dists = np.linalg.norm(z[ctrl] - z_ref, axis=1)
    if len(dists) == 0:
        raise ValueError("No control cells found in latents.h5ad")
    return {
        "percentile": float(p),
        "value": float(np.percentile(dists, float(p))),
        "n_ctrl_cells": int(len(dists)),
        "method": "L2_distance",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--vae_dir", required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of only the value.")
    args = parser.parse_args(argv)

    result = compute_epsilon_percentile(args.vae_dir, args.p)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['value']:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
