"""P0A gate-breakdown diagnostics for the frozen V1 dynamics artifacts."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.metrics import (
    _fit_ridge_baseline,
    _predict_ridge_baseline,
    pearson_r_per_dim,
    predictive_r2,
)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def load_dynamics_model(dynamics_dir: str | Path) -> Any:
    """Load the frozen V1 dynamics model from a run directory without retraining."""
    import torch

    from src.models.dynamics import PerturbationDynamicsModel

    dynamics_dir = Path(dynamics_dir)
    with open(dynamics_dir / "config.json") as f:
        cfg = json.load(f)

    accepted = set(inspect.signature(PerturbationDynamicsModel.__init__).parameters)
    kwargs = {
        "n_latent": int(cfg["n_latent"]),
        "n_genes": int(cfg["n_genes"]),
        "d_emb": int(cfg.get("d_emb", 64)),
        "n_hidden": int(cfg.get("n_hidden", 256)),
        "n_layers": int(cfg.get("n_layers", 3)),
        "dropout": float(cfg.get("dropout", 0.1)),
    }
    for key, default in [
        ("activation", "silu"),
        ("use_layernorm", True),
        ("log_var_min", -5.0),
        ("log_var_max", 3.0),
        ("log_var_init_bias", -2.0),
        ("use_state_linear_skip", False),
        ("use_gene_delta_bias", False),
    ]:
        if key in accepted:
            kwargs[key] = cfg.get(key, default)

    model = PerturbationDynamicsModel(**kwargs)
    state = torch.load(dynamics_dir / "model.pt", map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    model.to("cpu")
    return model


def predict_z_next(model: Any, z_ctrl: np.ndarray, gene_idx: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Predict post-perturbation latents for Contract-2 pair arrays."""
    import torch

    z_ctrl = np.asarray(z_ctrl, dtype=np.float32)
    gene_idx = np.asarray(gene_idx, dtype=np.int64)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(z_ctrl), batch_size):
            z = torch.from_numpy(z_ctrl[start : start + batch_size]).float()
            g = torch.from_numpy(gene_idx[start : start + batch_size]).long()
            out = model(z, g)
            z_next = out[0] if isinstance(out, tuple) else out
            preds.append(z_next.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.zeros_like(z_ctrl, dtype=np.float32)


def _split_arrays(split: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(split["z_ctrl"], dtype=np.float32),
        np.asarray(split["gene_idx"], dtype=np.int32),
        np.asarray(split["z_pert"], dtype=np.float32),
        np.asarray(split["z_pred"], dtype=np.float32),
    )


def build_gate_breakdown_tables(
    *,
    train: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    per_gene_min_for_pearson: int = 30,
    gene_symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-dimension and per-gene MLP-vs-ridge margin tables.

    Ridge is fit once on train pairs as ``Ridge(z_ctrl + one_hot(gene), delta)``.
    Each split reports MLP Pearson, ridge Pearson, and MLP-minus-ridge margins.
    """
    z_train = np.asarray(train["z_ctrl"], dtype=np.float32)
    gene_train = np.asarray(train["gene_idx"], dtype=np.int32)
    z_pert_train = np.asarray(train["z_pert"], dtype=np.float32)
    delta_train = z_pert_train - z_train
    n_genes = int(gene_train.max()) if len(gene_train) else 0
    ridge = _fit_ridge_baseline(z_train, gene_train, delta_train, n_genes)

    per_dim_rows: list[dict[str, Any]] = []
    per_gene_rows: list[dict[str, Any]] = []

    for split_name, split in splits.items():
        z_ctrl, gene_idx, z_pert, z_pred = _split_arrays(split)
        delta_true = z_pert - z_ctrl
        delta_mlp = z_pred - z_ctrl
        delta_ridge = _predict_ridge_baseline(ridge, z_ctrl, gene_idx, n_genes)

        mlp_dim = pearson_r_per_dim(delta_true, delta_mlp)
        ridge_dim = pearson_r_per_dim(delta_true, delta_ridge)
        for dim, (m, r) in enumerate(zip(mlp_dim, ridge_dim, strict=True)):
            per_dim_rows.append({
                "split": split_name,
                "dim": int(dim),
                "mlp_pearson": float(m),
                "ridge_pearson": float(r),
                "mlp_minus_ridge_pearson": float(m - r),
            })

        for g in sorted(np.unique(gene_idx).astype(int).tolist()):
            mask = gene_idx == g
            dt = delta_true[mask]
            dm = delta_mlp[mask]
            dr = delta_ridge[mask]
            mlp_r2 = predictive_r2(dt, dm)
            ridge_r2 = predictive_r2(dt, dr)
            row: dict[str, Any] = {
                "split": split_name,
                "gene_idx": int(g),
                "gene_symbol": (
                    gene_symbols[g - 1]
                    if gene_symbols is not None and 1 <= g <= len(gene_symbols)
                    else ""
                ),
                "n": int(mask.sum()),
                "mlp_r2": float(mlp_r2),
                "ridge_r2": float(ridge_r2),
                "mlp_minus_ridge_r2": float(mlp_r2 - ridge_r2),
                "mlp_pearson": np.nan,
                "ridge_pearson": np.nan,
                "mlp_minus_ridge_pearson": np.nan,
            }
            if int(mask.sum()) >= int(per_gene_min_for_pearson):
                mlp_p = float(np.nanmean(pearson_r_per_dim(dt, dm)))
                ridge_p = float(np.nanmean(pearson_r_per_dim(dt, dr)))
                row.update({
                    "mlp_pearson": mlp_p,
                    "ridge_pearson": ridge_p,
                    "mlp_minus_ridge_pearson": float(mlp_p - ridge_p),
                })
            per_gene_rows.append(row)

    per_dim = pd.DataFrame(per_dim_rows).sort_values(
        ["split", "mlp_minus_ridge_pearson", "dim"],
        ascending=[True, True, True],
    )
    per_gene = pd.DataFrame(per_gene_rows).sort_values(
        ["split", "mlp_minus_ridge_r2", "gene_idx"],
        ascending=[True, True, True],
    )
    return per_dim, per_gene


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _load_gene_symbols() -> list[str] | None:
    path = Path("artifacts/vae/gene_vocab.json")
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return [str(g) for g in json.load(f)["genes"]]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--dynamics_dir", required=True)
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args(argv)

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _load_npz(pairs_dir / "train_pairs.npz")
    val = _load_npz(pairs_dir / "val_pairs.npz")
    ood = _load_npz(pairs_dir / "ood_pairs.npz")

    model = load_dynamics_model(args.dynamics_dir)
    val["z_pred"] = predict_z_next(model, val["z_ctrl"], val["gene_idx"], batch_size=args.batch_size)
    ood["z_pred"] = predict_z_next(model, ood["z_ctrl"], ood["gene_idx"], batch_size=args.batch_size)

    per_dim, per_gene = build_gate_breakdown_tables(
        train=train,
        splits={"val": val, "ood": ood},
        gene_symbols=_load_gene_symbols(),
    )
    per_dim_path = out_dir / "per_dim_margin.csv"
    per_gene_path = out_dir / "per_gene_margin.csv"
    per_dim.to_csv(per_dim_path, index=False)
    per_gene.to_csv(per_gene_path, index=False)

    metadata = {
        "stage": "p0a_gate_breakdown",
        "config_name": args.config_name,
        "git_commit": _git_commit(),
        "read_only_mode": True,
        "source_paths": {"dynamics_dir": str(args.dynamics_dir), "pairs_dir": str(args.pairs_dir)},
        "outputs": {"per_dim_margin": str(per_dim_path), "per_gene_margin": str(per_gene_path)},
    }
    (out_dir / "gate_breakdown_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Worst per-dim margins:")
    print(per_dim.head(5).to_string(index=False))
    print("Worst per-gene margins:")
    print(per_gene.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
