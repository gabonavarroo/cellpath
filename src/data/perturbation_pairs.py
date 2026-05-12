"""OT / random / mean-delta pseudo-pairing for the dynamics model.

Owner: Agent A. See ARCHITECTURE.md Concept 7 + DATA.md §4.

Why pseudo-pairing
------------------
Perturb-seq is cross-sectional: we never observe the same cell pre- and post-perturbation.
For each perturbation ``p`` we have a control population and a perturbed population, and we
*construct* training triples ``(z_ctrl_i, p, z_pert_i)`` via a pseudo-pairing strategy.

Three strategies are supported:
- ``ot``         : entropic optimal transport (CellOT, Bunne et al. 2023). Default.
- ``random``     : random within-perturbation pairing. Fallback when OT is too slow.
- ``mean_delta`` : pair ``z_pert_j`` with the control cell closest to ``z_pert_j − Δ̄p``.

A mock generator (:func:`generate_mock_pairs`) produces synthetic pairs matching the contract
schema so Agent B can train dynamics on Day 0 without waiting for the real data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np


def build_pairs(
    cfg: Any,
    adata: Any | None = None,
    latents: Any | None = None,
    method: Literal["ot", "random", "mean_delta"] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Build all four pair files (train / val / ood / combo) and metadata.

    Parameters
    ----------
    cfg
        Hydra config (must have ``pairing``, ``paths``).
    adata
        Preprocessed AnnData (output of :func:`src.data.preprocess.run_preprocessing`).
        If ``None``, loaded from ``cfg.paths.norman_processed_h5ad``.
    latents
        scVI latent matrix ``Z`` of shape ``(N_cells, n_latent)``. If ``None``, loaded from
        ``cfg.paths.vae_latents_h5ad`` (``adata.obsm["X_scVI"]``).
    method
        Pairing strategy. Defaults to ``cfg.pairing.method``.
    out_dir
        Destination dir. Defaults to ``cfg.paths.pairs_dir``.

    Returns
    -------
    dict[str, Path]
        ``{"train": train_pairs.npz, "val": val_pairs.npz, "ood": ood_pairs.npz,
           "combo": combo_pairs.npz, "metadata": metadata.json}``.

    Raises
    ------
    NotImplementedError
        Agent A: implement pairing logic per DATA.md §4 and AGENTS.md Contract 2.

    Examples
    --------
    Expected npz schemas (Contract 2)::

        train_pairs.npz:
            z_ctrl    (M, 32) float32
            gene_idx  (M,)    int32   # 1..N (0 = ctrl is excluded from training)
            z_pert    (M, 32) float32

        combo_pairs.npz:
            z_ctrl    (M, 32) float32
            gene_idx_a(M,)    int32
            gene_idx_b(M,)    int32
            z_pert_ab (M, 32) float32
    """
    raise NotImplementedError(
        "Agent A: build pairs per DATA.md §4. OT default, random + mean_delta fallbacks. "
        "Splits: 90/10 within-perturbation (train/val); 80/20 across perturbations (train/ood); "
        "80/20 combo split."
    )


def pair_ot(z_ctrl: Any, z_pert: Any, epsilon: float = 0.05, max_iter: int = 500) -> Any:
    """Entropic optimal transport pairing for a single perturbation.

    Parameters
    ----------
    z_ctrl
        Latent vectors of control cells, shape ``(N_ctrl, d)``.
    z_pert
        Latent vectors of perturbed cells for one perturbation, shape ``(N_pert, d)``.
    epsilon
        Sinkhorn entropic regularization.
    max_iter
        Max Sinkhorn iterations.

    Returns
    -------
    np.ndarray
        Hard pairing: for each row of ``z_pert``, the index of the matched control cell.
        Shape ``(N_pert,) int64``.

    Raises
    ------
    NotImplementedError
        Agent A: use ``ot.sinkhorn`` (POT). Cost matrix = pairwise L2², normalize by median.
    """
    raise NotImplementedError(
        "Agent A: Sinkhorn pairing via POT. See DATA.md §4.1 for the recipe."
    )


def pair_random(z_ctrl: Any, z_pert: Any, rng: Any) -> Any:
    """Random within-perturbation pairing.

    Parameters
    ----------
    z_ctrl, z_pert
        See :func:`pair_ot`.
    rng
        NumPy ``Generator``.

    Returns
    -------
    np.ndarray
        ``(N_pert,) int64`` random indices into ``z_ctrl``.

    Raises
    ------
    NotImplementedError
        Agent A: trivially ``rng.integers(0, N_ctrl, size=N_pert)``.
    """
    raise NotImplementedError("Agent A: rng.integers based random pairing.")


def pair_mean_delta(z_ctrl: Any, z_pert: Any) -> Any:
    """Mean-delta pseudo-pairing.

    For each ``z_pert_j``, find the closest ``z_ctrl_i`` to ``z_pert_j − mean(z_pert) + mean(z_ctrl)``.

    Parameters
    ----------
    z_ctrl, z_pert
        See :func:`pair_ot`.

    Returns
    -------
    np.ndarray
        ``(N_pert,) int64`` nearest-neighbor indices into ``z_ctrl``.

    Raises
    ------
    NotImplementedError
        Agent A: compute Δp = mean(z_pert) − mean(z_ctrl), then kNN (k=1) on adjusted targets.
    """
    raise NotImplementedError("Agent A: Δp adjustment + kNN search via sklearn or numpy.")


def generate_mock_pairs(
    n: int = 10_000,
    n_genes: int = 100,
    n_latent: int = 32,
    n_combo: int = 1_000,
    seed: int = 42,
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Synthetic pairs matching the Contract 2 schema.

    Used by Agent B to start dynamics training before real Norman data is processed.
    Each perturbation gets a fixed Δz signature (sampled once from rng), so the dynamics model
    has a non-trivial learning target. The data does NOT come from biology. Metrics computed on
    mock runs are meaningless — log them as ``mock_*`` if logged at all.

    Parameters
    ----------
    n
        Total number of train+val pairs (across all train-gene perturbations).
    n_genes
        Action space size (excluding NO-OP). Genes are 1-indexed: [1, n_genes].
    n_latent
        Latent dimension.
    n_combo
        Number of combo pairs.
    seed
        RNG seed.
    out_dir
        Destination dir. If ``None``, writes to ``artifacts/pairs/``.

    Returns
    -------
    dict[str, Path]
        ``{"train", "val", "ood", "combo", "metadata"}``.
    """
    rng = np.random.default_rng(seed)

    if out_dir is None:
        out_dir = Path("artifacts/pairs")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-gene constant displacement vectors — index 0 unused (ctrl placeholder)
    gene_deltas = rng.normal(scale=0.5, size=(n_genes + 1, n_latent)).astype("float32")

    # Gene split: 80% train, 20% OOD (by gene identity)
    all_gene_ids = np.arange(1, n_genes + 1, dtype=np.int32)
    shuffled = rng.permutation(all_gene_ids)
    n_ood_genes = max(1, int(n_genes * 0.2))
    ood_gene_set = set(shuffled[:n_ood_genes].tolist())
    train_gene_set = set(shuffled[n_ood_genes:].tolist())

    # Generate n pairs, assigning genes uniformly over all genes
    z_ctrl_all = rng.standard_normal((n, n_latent)).astype("float32")
    gene_idx_all = rng.choice(all_gene_ids, size=n).astype("int32")
    noise = rng.normal(scale=0.1, size=(n, n_latent)).astype("float32")
    z_pert_all = (z_ctrl_all + gene_deltas[gene_idx_all] + noise).astype("float32")

    # Partition by gene type
    is_ood = np.array([int(g) in ood_gene_set for g in gene_idx_all.tolist()], dtype=bool)
    is_train_gene = ~is_ood

    z_ctrl_tg = z_ctrl_all[is_train_gene]
    gene_idx_tg = gene_idx_all[is_train_gene]
    z_pert_tg = z_pert_all[is_train_gene]

    # Within-train-gene: 90% → train, 10% → val (cell-level split)
    n_tg = len(z_ctrl_tg)
    perm = rng.permutation(n_tg)
    n_val = max(1, int(n_tg * 0.1))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Combo pairs: two sequential gene activations from train genes
    train_gene_arr = np.array(sorted(train_gene_set), dtype=np.int32)
    combo_a = rng.choice(train_gene_arr, size=n_combo).astype("int32")
    combo_b = rng.choice(train_gene_arr, size=n_combo).astype("int32")
    z_ctrl_combo = rng.standard_normal((n_combo, n_latent)).astype("float32")
    noise_combo = rng.normal(scale=0.1, size=(n_combo, n_latent)).astype("float32")
    z_pert_ab = (
        z_ctrl_combo + gene_deltas[combo_a] + gene_deltas[combo_b] + noise_combo
    ).astype("float32")

    # Write all four npz files
    paths: dict[str, Path] = {}

    train_path = out_dir / "train_pairs.npz"
    np.savez(
        train_path,
        z_ctrl=z_ctrl_tg[train_idx],
        gene_idx=gene_idx_tg[train_idx],
        z_pert=z_pert_tg[train_idx],
    )
    paths["train"] = train_path

    val_path = out_dir / "val_pairs.npz"
    np.savez(
        val_path,
        z_ctrl=z_ctrl_tg[val_idx],
        gene_idx=gene_idx_tg[val_idx],
        z_pert=z_pert_tg[val_idx],
    )
    paths["val"] = val_path

    ood_path = out_dir / "ood_pairs.npz"
    np.savez(
        ood_path,
        z_ctrl=z_ctrl_all[is_ood],
        gene_idx=gene_idx_all[is_ood],
        z_pert=z_pert_all[is_ood],
    )
    paths["ood"] = ood_path

    combo_path = out_dir / "combo_pairs.npz"
    np.savez(
        combo_path,
        z_ctrl=z_ctrl_combo,
        gene_idx_a=combo_a,
        gene_idx_b=combo_b,
        z_pert_ab=z_pert_ab,
    )
    paths["combo"] = combo_path

    # Metadata matching Contract 2
    n_per_pert: dict[str, int] = {}
    train_gene_idx_written = gene_idx_tg[train_idx]
    for g in sorted(train_gene_set):
        n_per_pert[str(g)] = int((train_gene_idx_written == g).sum())

    metadata = {
        "pairing_method": "mock",
        "n_train": int(len(train_idx)),
        "n_val": int(n_val),
        "n_ood": int(is_ood.sum()),
        "n_combo": n_combo,
        "held_out_genes": sorted(ood_gene_set),
        "ot_epsilon": None,
        "n_per_perturbation": n_per_pert,
        "mock": True,
        "n_latent": n_latent,
        "n_genes": n_genes,
        "seed": seed,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    paths["metadata"] = meta_path

    return paths
