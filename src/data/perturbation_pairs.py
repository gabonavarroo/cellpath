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
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

log = logging.getLogger(__name__)


def _pair_with_fallback(
    z_ctrl: np.ndarray,
    z_pert: np.ndarray,
    method: str,
    ot_epsilon: float,
    ot_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply pairing method; fall back to mean_delta on OT failure (DATA.md §4.1)."""
    if method == "ot":
        try:
            return pair_ot(z_ctrl, z_pert, epsilon=ot_epsilon, max_iter=ot_iter)
        except RuntimeError as exc:
            log.warning("OT failed — falling back to mean_delta. Reason: %s", exc)
            return pair_mean_delta(z_ctrl, z_pert)
    if method == "mean_delta":
        return pair_mean_delta(z_ctrl, z_pert)
    return pair_random(z_ctrl, z_pert, rng)


def _split_combo_name(combo_name: str, single_gene_names: set[str]) -> tuple[str, str]:
    """Split a combo perturbation name into two single-gene names.

    Tries all binary split points so names with internal underscores work correctly.
    Example: ``"KLF1_BAK1"`` → ``("KLF1", "BAK1")``.

    Raises
    ------
    ValueError
        If no valid binary split can be found.
    """
    parts = combo_name.split("_")
    for i in range(1, len(parts)):
        gene_a = "_".join(parts[:i])
        gene_b = "_".join(parts[i:])
        if gene_a in single_gene_names and gene_b in single_gene_names:
            return gene_a, gene_b
    raise ValueError(
        f"Cannot split combo '{combo_name}' into two valid single genes. "
        f"Single-gene set (sample): {sorted(single_gene_names)[:10]}"
    )


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
    import anndata as ad

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    if adata is None:
        adata = ad.read_h5ad(str(cfg.paths.norman_processed_h5ad))
        log.info("Loaded adata from %s (%d cells)", cfg.paths.norman_processed_h5ad, adata.n_obs)

    if latents is not None:
        Z = np.asarray(latents, dtype=np.float32)
    else:
        lat = ad.read_h5ad(str(cfg.paths.vae_latents_h5ad))
        Z = np.asarray(lat.obsm["X_scVI"], dtype=np.float32)
        log.info("Loaded latents from %s — shape %s", cfg.paths.vae_latents_h5ad, Z.shape)

    if len(Z) != adata.n_obs:
        raise ValueError(
            f"Latent matrix rows ({len(Z)}) ≠ adata cells ({adata.n_obs}). "
            "Ensure latents.h5ad was produced from the same preprocessed h5ad."
        )

    if method is None:
        method = str(cfg.pairing.method)

    out_dir = Path(out_dir) if out_dir is not None else Path(cfg.paths.pairs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ot_epsilon   = float(cfg.pairing.ot_epsilon)
    ot_iter      = int(cfg.pairing.ot_iter)
    val_frac     = float(cfg.pairing.val_cell_fraction)
    ood_frac     = float(cfg.pairing.ood_gene_fraction)
    combo_ho_frac = float(cfg.pairing.combo_held_out_fraction)
    min_cells    = int(cfg.pairing.min_cells_per_perturbation)
    pair_seed    = int(cfg.pairing.pair_seed)

    rng = np.random.default_rng(pair_seed)

    log.info("build_pairs: method=%s  ot_epsilon=%.3f  val_frac=%.2f  ood_frac=%.2f",
             method, ot_epsilon, val_frac, ood_frac)

    # ------------------------------------------------------------------
    # 2. Perturbation metadata
    # ------------------------------------------------------------------
    pert_idx  = np.asarray(adata.obs["perturbation_idx"].values, dtype=np.int32)
    encoder   = dict(adata.uns["perturbation_encoder"])   # name → int
    ctrl_label = str(adata.uns["ctrl_label"])
    n_single  = int(adata.uns["noop_idx"])                # number of single-gene perturbations

    inv_encoder = {v: k for k, v in encoder.items()}     # int → name
    single_gene_names = {
        k for k, v in encoder.items()
        if k != ctrl_label and v <= n_single
    }

    # Control cells — source distribution for ALL pairings
    ctrl_mask = pert_idx == 0
    Z_ctrl = Z[ctrl_mask]
    log.info("Control cells: %d", int(ctrl_mask.sum()))

    # Categorise cells using nperts column (most reliable for Norman scperturb)
    if "nperts" in adata.obs.columns:
        nperts = np.asarray(adata.obs["nperts"].values, dtype=np.int32)
        single_mask = nperts == 1
        combo_mask  = nperts == 2
    else:
        single_mask = (pert_idx >= 1) & (pert_idx <= n_single)
        combo_mask  = pert_idx > n_single

    # ------------------------------------------------------------------
    # 3. Filter single-gene perturbations by min_cells
    # ------------------------------------------------------------------
    all_single_ids = np.unique(pert_idx[single_mask]).tolist()
    valid_single_ids = [
        pid for pid in all_single_ids
        if int((pert_idx == pid).sum()) >= min_cells
    ]
    log.info("Single-gene perturbations: %d total, %d with ≥%d cells",
             len(all_single_ids), len(valid_single_ids), min_cells)

    # ------------------------------------------------------------------
    # 4. 80/20 perturbation-level split (train genes vs OOD genes)
    # ------------------------------------------------------------------
    arr      = np.array(sorted(valid_single_ids), dtype=np.int32)
    shuffled = rng.permutation(arr)
    n_ood    = max(1, round(len(arr) * ood_frac))
    ood_gene_ids   = set(int(x) for x in shuffled[:n_ood])
    train_gene_ids = set(int(x) for x in shuffled[n_ood:])
    log.info("Gene split: %d train genes, %d OOD genes", len(train_gene_ids), len(ood_gene_ids))

    # ------------------------------------------------------------------
    # 5. Build train + val pairs (90/10 within-perturbation cell split)
    # ------------------------------------------------------------------
    tr_z_ctrl, tr_gene, tr_z_pert = [], [], []
    va_z_ctrl, va_gene, va_z_pert = [], [], []
    n_per_pert: dict[str, int] = {}

    sorted_train = sorted(train_gene_ids)
    for i, pid in enumerate(sorted_train, start=1):
        gene_name = inv_encoder.get(pid, str(pid))
        log.info("[%d/%d] Pairing %s ...", i, len(sorted_train), gene_name)

        pert_cell_mask = pert_idx == pid
        Z_pert = Z[pert_cell_mask]
        n_pert = len(Z_pert)

        # 90/10 cell split (AGENTS.md Contract 2 — primary gate uses val cells)
        perm  = rng.permutation(n_pert)
        n_val = max(1, round(n_pert * val_frac))
        val_perm, train_perm = perm[:n_val], perm[n_val:]

        ci_tr = _pair_with_fallback(Z_ctrl, Z_pert[train_perm], method, ot_epsilon, ot_iter, rng)
        ci_va = _pair_with_fallback(Z_ctrl, Z_pert[val_perm],   method, ot_epsilon, ot_iter, rng)

        tr_z_ctrl.append(Z_ctrl[ci_tr]);  tr_gene.append(np.full(len(ci_tr), pid, dtype=np.int32))
        tr_z_pert.append(Z_pert[train_perm])

        va_z_ctrl.append(Z_ctrl[ci_va]);  va_gene.append(np.full(len(ci_va), pid, dtype=np.int32))
        va_z_pert.append(Z_pert[val_perm])

        n_per_pert[gene_name] = int(len(train_perm))

    # ------------------------------------------------------------------
    # 6. Build OOD pairs (ALL cells of OOD genes)
    # ------------------------------------------------------------------
    oo_z_ctrl, oo_gene, oo_z_pert = [], [], []
    for pid in sorted(ood_gene_ids):
        gene_name = inv_encoder.get(pid, str(pid))
        log.info("OOD: pairing %s ...", gene_name)
        pert_cell_mask = pert_idx == pid
        Z_pert = Z[pert_cell_mask]
        ci = _pair_with_fallback(Z_ctrl, Z_pert, method, ot_epsilon, ot_iter, rng)
        oo_z_ctrl.append(Z_ctrl[ci]); oo_gene.append(np.full(len(ci), pid, dtype=np.int32))
        oo_z_pert.append(Z_pert)

    # ------------------------------------------------------------------
    # 7. Build combo pairs (80/20 combo split)
    # ------------------------------------------------------------------
    co_z_ctrl, co_gene_a, co_gene_b, co_z_pert = [], [], [], []
    held_out_combo_names: list[str] = []

    all_combo_ids = np.unique(pert_idx[combo_mask]).tolist()
    valid_combo_ids = [
        pid for pid in all_combo_ids
        if int((pert_idx == pid).sum()) >= min_cells
    ]
    log.info("Combo perturbations: %d total, %d with ≥%d cells",
             len(all_combo_ids), len(valid_combo_ids), min_cells)

    if valid_combo_ids:
        combo_arr     = np.array(sorted(valid_combo_ids), dtype=np.int32)
        combo_shuf    = rng.permutation(combo_arr)
        n_combo_ho    = max(1, round(len(combo_arr) * combo_ho_frac))
        combo_ho_ids  = set(int(x) for x in combo_shuf[:n_combo_ho])
        combo_tr_ids  = set(int(x) for x in combo_shuf[n_combo_ho:])
        held_out_combo_names = [inv_encoder.get(pid, str(pid)) for pid in combo_ho_ids]

        for pid in sorted(combo_tr_ids):
            combo_name = inv_encoder.get(pid, str(pid))
            try:
                g_a_name, g_b_name = _split_combo_name(combo_name, single_gene_names)
                g_a_idx = int(encoder[g_a_name])
                g_b_idx = int(encoder[g_b_name])
            except (ValueError, KeyError) as exc:
                log.warning("Skipping combo '%s': %s", combo_name, exc)
                continue

            log.info("Combo: pairing %s (%d+%d) ...", combo_name, g_a_idx, g_b_idx)
            pert_cell_mask = pert_idx == pid
            Z_pert = Z[pert_cell_mask]
            ci = _pair_with_fallback(Z_ctrl, Z_pert, method, ot_epsilon, ot_iter, rng)

            co_z_ctrl.append(Z_ctrl[ci])
            co_gene_a.append(np.full(len(ci), g_a_idx, dtype=np.int32))
            co_gene_b.append(np.full(len(ci), g_b_idx, dtype=np.int32))
            co_z_pert.append(Z_pert)

    # ------------------------------------------------------------------
    # 8. Concatenate arrays and write npz files (Contract 2 schema)
    # ------------------------------------------------------------------
    n_lat = Z.shape[1]

    def _cat2d(lst: list) -> np.ndarray:
        return np.concatenate(lst, axis=0).astype(np.float32) if lst else np.empty((0, n_lat), dtype=np.float32)

    def _cat1d(lst: list) -> np.ndarray:
        return np.concatenate(lst, axis=0).astype(np.int32) if lst else np.empty(0, dtype=np.int32)

    paths: dict[str, Path] = {}

    train_path = out_dir / "train_pairs.npz"
    np.savez(train_path, z_ctrl=_cat2d(tr_z_ctrl), gene_idx=_cat1d(tr_gene), z_pert=_cat2d(tr_z_pert))
    paths["train"] = train_path
    log.info("train_pairs.npz: %d pairs across %d genes", len(_cat1d(tr_gene)), len(train_gene_ids))

    val_path = out_dir / "val_pairs.npz"
    np.savez(val_path, z_ctrl=_cat2d(va_z_ctrl), gene_idx=_cat1d(va_gene), z_pert=_cat2d(va_z_pert))
    paths["val"] = val_path
    log.info("val_pairs.npz:   %d pairs", len(_cat1d(va_gene)))

    ood_path = out_dir / "ood_pairs.npz"
    np.savez(ood_path, z_ctrl=_cat2d(oo_z_ctrl), gene_idx=_cat1d(oo_gene), z_pert=_cat2d(oo_z_pert))
    paths["ood"] = ood_path
    log.info("ood_pairs.npz:   %d pairs across %d genes", len(_cat1d(oo_gene)), len(ood_gene_ids))

    combo_path = out_dir / "combo_pairs.npz"
    np.savez(
        combo_path,
        z_ctrl=_cat2d(co_z_ctrl),
        gene_idx_a=_cat1d(co_gene_a),
        gene_idx_b=_cat1d(co_gene_b),
        z_pert_ab=_cat2d(co_z_pert),
    )
    paths["combo"] = combo_path
    log.info("combo_pairs.npz: %d pairs", len(_cat1d(co_gene_a)))

    # ------------------------------------------------------------------
    # 9. Write metadata.json (Contract 2)
    # ------------------------------------------------------------------
    held_out_gene_names = [inv_encoder.get(pid, str(pid)) for pid in ood_gene_ids]

    metadata = {
        "pairing_method": method,
        "n_train": int(len(_cat1d(tr_gene))),
        "n_val":   int(len(_cat1d(va_gene))),
        "n_ood":   int(len(_cat1d(oo_gene))),
        "n_combo": int(len(_cat1d(co_gene_a))),
        "held_out_genes": sorted(held_out_gene_names),
        "held_out_combos": sorted(held_out_combo_names),
        "ot_epsilon": ot_epsilon if method == "ot" else None,
        "n_per_perturbation": n_per_pert,
        "n_train_genes": len(train_gene_ids),
        "n_ood_genes": len(ood_gene_ids),
        "val_cell_fraction": val_frac,
        "ood_gene_fraction": ood_frac,
        "pair_seed": pair_seed,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    paths["metadata"] = meta_path

    log.info("build_pairs complete → %s", out_dir)
    return paths


def pair_ot(z_ctrl: Any, z_pert: Any, epsilon: float = 0.05, max_iter: int = 500) -> Any:
    """Entropic optimal transport pairing for a single perturbation.

    Recipe (DATA.md §4.1):
    1. Cost matrix C[i,j] = ||z_ctrl[i] - z_pert[j]||_2, median-normalized.
    2. Sinkhorn divergence (POT) with uniform marginals → transport plan T (n_ctrl × n_pert).
    3. Hard pairing: for each pert cell j, ctrl_idx[j] = argmax_i T[:,j].
    4. On NaN/degenerate T, retry up to 3× with doubling epsilon; raise RuntimeError on failure
       so ``build_pairs`` can fall back to ``mean_delta`` (DATA.md §4.1 fallback rule).

    Parameters
    ----------
    z_ctrl
        Latent vectors of control cells, shape ``(N_ctrl, d)``.
    z_pert
        Latent vectors of perturbed cells for one perturbation, shape ``(N_pert, d)``.
    epsilon
        Sinkhorn entropic regularization (DATA.md default = 0.05).
    max_iter
        Max Sinkhorn iterations.

    Returns
    -------
    np.ndarray
        Hard pairing: for each row of ``z_pert``, the index of the matched control cell.
        Shape ``(N_pert,) int64``.

    Raises
    ------
    RuntimeError
        After 3 failed retries; caller should fall back to ``pair_mean_delta``.
    """
    import ot as pot
    from scipy.spatial.distance import cdist

    z_ctrl = np.asarray(z_ctrl, dtype=np.float64)
    z_pert = np.asarray(z_pert, dtype=np.float64)

    # Pairwise L2 cost matrix, median-normalized for numerical stability
    C = cdist(z_ctrl, z_pert, metric="euclidean")
    c_median = np.median(C)
    if c_median > 0:
        C = C / c_median

    # Uniform marginals: each ctrl / pert cell has equal weight
    a = np.ones(len(z_ctrl)) / len(z_ctrl)
    b = np.ones(len(z_pert)) / len(z_pert)

    _eps = float(epsilon)
    for attempt in range(3):
        T = pot.sinkhorn(a, b, C, reg=_eps, numItermax=max_iter, warn=False)

        if np.isnan(T).any() or np.isinf(T).any():
            _eps *= 2.0
            log.debug("OT NaN/Inf on attempt %d — epsilon → %.4f", attempt + 1, _eps)
            continue

        # Hard pairing: for each pert cell j, pick ctrl i = argmax T[:, j]
        return np.argmax(T, axis=0).astype(np.int64)

    raise RuntimeError(
        f"OT failed after 3 retries (final epsilon={_eps:.4f}). "
        "build_pairs will fall back to mean_delta."
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
    return rng.integers(0, len(z_ctrl), size=len(z_pert)).astype(np.int64)


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
    from scipy.spatial import cKDTree

    z_ctrl = np.asarray(z_ctrl, dtype=np.float32)
    z_pert = np.asarray(z_pert, dtype=np.float32)

    # Reverse the population-level shift: project pert cells back into ctrl space
    delta_p = z_pert.mean(axis=0) - z_ctrl.mean(axis=0)
    z_adjusted = z_pert - delta_p  # (N_pert, d)

    # Nearest-neighbor match in ctrl space (k=1)
    _, ctrl_indices = cKDTree(z_ctrl).query(z_adjusted, k=1)
    return ctrl_indices.astype(np.int64)


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
