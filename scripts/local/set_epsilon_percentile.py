import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--percentile", type=float, required=True)
parser.add_argument("--latents", default="artifacts/vae/latents.h5ad")
parser.add_argument("--z-ref", default="artifacts/vae/z_reference_centroid.npy")
parser.add_argument("--out", default="artifacts/vae/epsilon_success.json")
args = parser.parse_args()

lat = ad.read_h5ad(args.latents)
Z = np.asarray(lat.obsm["X_scVI"], dtype=np.float32)
z_ref = np.load(args.z_ref).astype(np.float32)
pert_idx = np.asarray(lat.obs["perturbation_idx"].values)

ctrl = pert_idx == 0
dists = np.linalg.norm(Z[ctrl] - z_ref, axis=1)

eps = {
    "percentile": float(args.percentile),
    "value": float(np.percentile(dists, args.percentile)),
    "n_ctrl_cells": int(ctrl.sum()),
    "method": "L2_distance",
}

Path(args.out).write_text(json.dumps(eps, indent=2))
print(json.dumps(eps, indent=2))
