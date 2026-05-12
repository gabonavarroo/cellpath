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

from pathlib import Path
from typing import Any, Literal


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
    """Synthetic pairs matching the Contract 2 schema. **Day 0 deliverable for Agent A.**

    Used by Agent B to start dynamics training before real Norman data is processed.
    Each perturbation gets a learned-on-rng constant ``Δz`` signature, so the dynamics model has
    a non-trivial learning target; the data does NOT come from biology. Metrics computed on
    mock runs are meaningless — log them as ``mock_*`` if logged at all.

    Parameters
    ----------
    n
        Total number of train pairs.
    n_genes
        Action space size (excluding NO-OP).
    n_latent
        Latent dimension.
    n_combo
        Number of combo pairs.
    seed
        RNG seed.
    out_dir
        Destination dir. If ``None``, writes to ``artifacts/pairs/`` (mock files).

    Returns
    -------
    dict[str, Path]
        ``{"train", "val", "ood", "combo", "metadata"}``.

    Raises
    ------
    NotImplementedError
        Agent A: implement on Day 0 so Agent B is unblocked. Generate per-gene Δz from a fixed
        rng + small per-cell noise.
    """
    raise NotImplementedError(
        "Agent A (Day 0 deliverable): implement synthetic pairs. "
        "Per-gene constant Δz + per-cell N(0, 0.1) noise. "
        "Write the same npz schema as the real pipeline."
    )
