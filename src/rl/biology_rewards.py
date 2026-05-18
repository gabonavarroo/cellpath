"""V3B biology-aware reward extensions.

Pure-function rewards that compose with V2's ``terminal_only_step_cost`` base:

* :func:`safety_aware_reward`  — Variant C (V3B Phase 2): adds toxicity + common-essential
  penalty at the terminal step. **No** per-step shaping (preserves V2 hard-bench protocol).

The signatures intentionally mirror :func:`src.rl.reward.compute_reward` so that the
environment can dispatch by ``reward_mode`` without bloating the V2 reward function.

Math (per V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §4 Variant C, updated per
Phase 1 acceptance refinements 2026-05-17):

::

    R_t = 0                                                            (t < T)
    R_T = 1 · is_success
          − β · t
          − λ_tox · tox_path
          − λ_ce  · common_essential_count

where ``tox_path`` and ``common_essential_count`` are episode-cumulative quantities
accumulated by :class:`src.rl.environment.CellReprogrammingEnv` over non-NOOP actions.

Defaults: ``β = 0.05`` (V2 primary, unchanged), ``λ_tox = 0.10``, ``λ_ce = 0.05``.

Honesty caveat: DepMap Chronos is CRISPR-Cas9 knockout; Norman 2019 is CRISPRa. The
``tox`` term is a prior on K562 experimental disturbance, not direct toxicity. See
``artifacts_v3/v3b_biology/README.md`` §2.5.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)


def safety_aware_reward(
    *,
    is_success: bool,
    terminated: bool,
    truncated: bool,
    step_idx: int,
    tox_path: float,
    common_essential_count: int,
    beta_step_cost: float = 0.05,
    lambda_tox: float = 0.10,
    lambda_ce: float = 0.05,
    distance_scale: float = 1.0,
) -> float:
    """Compute Variant C safety-aware reward.

    Mid-episode reward is 0 (matches ``terminal_only_step_cost``); only the terminal
    step yields a non-zero value.

    Parameters
    ----------
    is_success
        True if terminal state is within ε of z_ref.
    terminated, truncated
        Episode end flags.
    step_idx
        Number of steps taken in the episode (1-indexed at terminal).
    tox_path
        Episode-cumulative ``max(0, -chronos(a) - 0.5)`` over non-NOOP actions.
    common_essential_count
        Episode-cumulative ``1[chronos(a) < -0.5]`` over non-NOOP actions.
    beta_step_cost
        Per-step cost (V2 primary default 0.05). Same as V2 ``terminal_only_step_cost``.
    lambda_tox
        Weight on cumulative toxicity penalty (default 0.10).
    lambda_ce
        Weight on common-essential picks (default 0.05).
    distance_scale
        Multiplier on the success bonus (default 1.0). Mirrors compute_reward's API.

    Returns
    -------
    float
        Scalar reward for this transition. Always 0 at non-terminal steps.

    Notes
    -----
    Setting ``lambda_tox = lambda_ce = 0`` recovers ``terminal_only_step_cost`` exactly.
    This is the V2-mode-regression invariant tested in ``tests/test_biology_rewards.py``.
    """
    if not (terminated or truncated):
        return 0.0

    base = float(distance_scale) * (1.0 if is_success else 0.0)
    step = float(beta_step_cost) * float(step_idx)
    safety = float(lambda_tox) * float(tox_path) + float(lambda_ce) * float(common_essential_count)
    return base - step - safety


def build_safety_arrays(
    gene_safety_df: Any,
    *,
    n_genes: int,
    permute_chronos: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize per-gene safety arrays indexed by action_idx ∈ [0, n_genes).

    Parameters
    ----------
    gene_safety_df
        Polars or pandas-like DataFrame from ``artifacts_v3/v3b_biology/gene_safety.parquet``
        with columns ``action_idx``, ``tox_raw``, ``is_essential``.
    n_genes
        Action-space width excluding NOOP (= ``gene_vocab.json::n_genes``).
    permute_chronos
        If True, randomly permute the gene order before extracting tox/essential
        arrays. Used for the V3B Phase 2 null control (preserves the marginal
        distribution but destroys per-gene identity).
    seed
        RNG seed for the permutation.

    Returns
    -------
    (tox_array, essential_array) :
        ``tox_array``       shape ``(n_genes,)`` float32, with 0.0 for missing genes.
        ``essential_array`` shape ``(n_genes,)`` bool,    with False for missing genes.

    Notes
    -----
    Genes missing from DepMap (Chronos = None) get tox=0, is_essential=False — a
    NEUTRAL prior, by design. Coverage is recorded in ``coverage.json``.
    """
    # Build by-action-idx arrays.
    tox = np.zeros(int(n_genes), dtype=np.float32)
    ess = np.zeros(int(n_genes), dtype=bool)

    # Sort by action_idx so we get deterministic ordering.
    if hasattr(gene_safety_df, "iter_rows"):
        rows = list(gene_safety_df.iter_rows(named=True))
    else:
        rows = gene_safety_df.to_dict(orient="records")  # type: ignore[union-attr]

    for row in rows:
        idx = int(row["action_idx"])
        if not (0 <= idx < n_genes):
            continue
        t = row.get("tox_raw")
        if t is not None:
            tox[idx] = float(t)
        e = row.get("is_essential")
        if e is not None:
            ess[idx] = bool(e)

    if permute_chronos:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_genes)
        tox = tox[perm].astype(np.float32, copy=True)
        ess = ess[perm].copy()
        LOG.info(
            "permute_chronos=True (seed=%d) — gene_safety arrays permuted. "
            "n_essential preserved: %d. THIS IS A NULL CONTROL — do not use as "
            "the production safety table.",
            seed, int(ess.sum()),
        )

    return tox, ess
