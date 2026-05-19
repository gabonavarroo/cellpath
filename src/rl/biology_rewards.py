"""V3B reward extensions.

Pure-function rewards that compose with V2's ``terminal_only_step_cost`` base:

* :func:`safety_aware_reward`     — Variant C (V3B Phase 2): adds toxicity + common-essential
  penalty at the terminal step. **No** per-step shaping (preserves V2 hard-bench protocol).
* :func:`path_length_freeband_reward` — Variant B (V3B Phase 3): global, nonlinear
  path-length penalty with a free band. No biology terms. The reward is terminal-only
  (mid-episode = 0). The penalty is a three-piece schedule of the final path length T:
  zero up to ``free_steps``, mild slope until ``mild_until``, then a heavier slope.

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


def path_length_freeband_reward(
    *,
    is_success: bool,
    terminated: bool,
    truncated: bool,
    step_idx: int,
    free_steps: int = 3,
    mild_until: int = 5,
    mild_beta: float = 0.02,
    heavy_beta: float = 0.10,
    success_bonus: float = 1.0,
) -> float:
    """Compute Variant B (path-length free-band) reward.

    Terminal-only; mid-episode reward is 0 (matches ``terminal_only_step_cost``).
    The terminal reward is::

        R_T = success_bonus · 1[is_success] − path_penalty(T)

    where ``T = step_idx`` (number of non-NOOP gene actions taken). The penalty is a
    piecewise-linear-with-corner schedule of three regions::

        T ≤ free_steps                 →  0                                 (free band)
        free_steps < T ≤ mild_until    →  mild_beta · (T − free_steps)      (plausible band)
        T > mild_until                 →  mild_beta · (mild_until − free_steps)
                                          + heavy_beta · (T − mild_until)   (speculative band)

    Defaults (per V3B Phase 3 plan): ``free_steps=3, mild_until=5, mild_beta=0.02,
    heavy_beta=0.10, success_bonus=1.0``. With these:

    +--------+---------+-----------+---------+---------+----------+
    |   T    |   1     |     3     |    4    |    5    |     8    |
    +--------+---------+-----------+---------+---------+----------+
    | penalty|   0     |     0     |  0.02   |  0.04   |   0.34   |
    +--------+---------+-----------+---------+---------+----------+

    Design rationale (see V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §6 + Phase 3 brief):

    * **Free up to K=3**: Norman 2019 directly measures dual-perturbation cells
      (K=2 dynamics composability). K=3 is one step beyond empirical anchoring but
      still in the local composability regime.
    * **Mild penalty for K=4,5**: makes these depths *available* when needed but not
      *attractive* unless K≤3 fails to reach ε. Respects the dynamics' Markov-composition
      extrapolation risk.
    * **Heavy penalty for K>5**: hard-discourages speculative depths where the dynamics
      composes 6+ single-step predictions without empirical anchor — the user's "model
      mathematical hallucination" concern.

    Setting ``free_steps = mild_until = max_steps - 1`` and ``mild_beta = heavy_beta = 0``
    recovers a pure success-bonus reward (no path penalty). Setting ``free_steps = 0,
    mild_until = max_steps, mild_beta = heavy_beta = beta`` recovers V2's
    ``terminal_only_step_cost`` exactly.

    Parameters
    ----------
    is_success
        True if terminal latent is within ε of z_ref.
    terminated, truncated
        Episode-end flags. Mid-episode (neither) returns 0.0.
    step_idx
        Number of non-NOOP gene actions taken at terminal/truncation. The env
        increments ``self._step_idx`` after each gene action and passes it here.
    free_steps, mild_until, mild_beta, heavy_beta, success_bonus
        Schedule hyperparameters; all non-negative.

    Returns
    -------
    float
        Scalar reward for this transition. 0.0 at non-terminal steps.

    Notes
    -----
    Truncated-without-success: R_T = -path_penalty(max_steps), bounded and finite.
    NOOP-at-step-0 success: R_T = success_bonus - 0 = success_bonus (penalty zero).
    """
    if not (terminated or truncated):
        return 0.0
    T = int(step_idx)
    fs = int(free_steps)
    mu = int(mild_until)
    mb = float(mild_beta)
    hb = float(heavy_beta)
    if T <= fs:
        penalty = 0.0
    elif T <= mu:
        penalty = mb * float(T - fs)
    else:
        penalty = mb * float(mu - fs) + hb * float(T - mu)
    base = float(success_bonus) * (1.0 if is_success else 0.0)
    return base - penalty


def per_step_uncertainty_scalar(
    log_var: np.ndarray,
    *,
    clip_min: float = -5.0,
    clip_max: float = 3.0,
    reduce: str = "mean_sigma",
) -> float:
    """Convert a single-step dynamics ``log σ²`` vector to a bounded scalar.

    The dynamics model's heteroscedastic head emits ``log σ²`` per latent dim per
    (state, gene) pair (see ``src/models/dynamics.py``). This function reduces it
    to one number so the reward can use a single, comparable uncertainty term per
    step.

    Parameters
    ----------
    log_var
        Shape ``(n_latent,)``; output of dynamics for one (z, gene).
    clip_min, clip_max
        Per-element clamp on log_var (matches the model's own internal clamp).
    reduce
        How to reduce the per-dim vector:
        * ``"mean_sigma"``  (default): ``mean(exp(0.5·log_var))`` — mean
          per-dim standard deviation. Bounded in roughly [0.08, 4.48] for
          default clamps; safe to sum across a K=8 trajectory.
        * ``"max_sigma"``: ``max(exp(0.5·log_var))``.
        * ``"mean_log_var"``: ``mean(log_var)``; signed but unbounded below.

    Returns
    -------
    float, finite. Caller should pre-clip if log_var was already clamped.

    Notes
    -----
    This is an **epistemic+aleatoric proxy from the learned dynamics**, not a
    biological ground-truth uncertainty. The reward penalises trajectories with
    high model uncertainty under the assumption that uncertain transitions are
    more likely to be off-distribution or compositionally extrapolated.
    """
    arr = np.asarray(log_var, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    clipped = np.clip(arr, float(clip_min), float(clip_max))
    if reduce == "mean_sigma":
        return float(np.mean(np.exp(0.5 * clipped)))
    if reduce == "max_sigma":
        return float(np.max(np.exp(0.5 * clipped)))
    if reduce == "mean_log_var":
        return float(np.mean(clipped))
    raise ValueError(f"Unknown reduce={reduce!r}; choose mean_sigma|max_sigma|mean_log_var")


def uncertainty_aware_reward(
    *,
    is_success: bool,
    terminated: bool,
    truncated: bool,
    step_idx: int,
    unc_path_max: float,
    beta_step_cost: float = 0.05,
    lambda_unc: float = 0.05,
    success_bonus: float = 1.0,
) -> float:
    """Variant D — uncertainty-aware reward (V3B Phase 4).

    Terminal-only::

        R_T = success_bonus · 1[is_success] − β·step_idx − λ_unc · unc_path_max

    Per-step ``unc`` is computed by :func:`per_step_uncertainty_scalar` from the
    dynamics' ``log σ²`` output; the env aggregates ``unc_path_max`` over non-NOOP
    steps. This term proxies trajectory-level worst-case predictive uncertainty
    from the learned dynamics — **not** biological ground truth.

    Setting ``lambda_unc=0`` recovers V2's ``terminal_only_step_cost``.

    Caveats
    -------
    * ``unc_path_max`` is meaningful only when the heteroscedastic head is
      well-calibrated (see V3A Track L OOD-σ Spearman = 0.738).
    * High λ_unc can push the policy toward NOOP (low-uncertainty terminate) —
      keep success_bonus dominant.
    """
    if not (terminated or truncated):
        return 0.0
    base = float(success_bonus) * (1.0 if is_success else 0.0)
    step = float(beta_step_cost) * float(step_idx)
    unc = float(lambda_unc) * float(unc_path_max)
    return base - step - unc


def safety_path_freeband_reward(
    *,
    is_success: bool,
    terminated: bool,
    truncated: bool,
    step_idx: int,
    tox_path: float,
    common_essential_count: int,
    free_steps: int = 3,
    mild_until: int = 5,
    mild_beta: float = 0.02,
    heavy_beta: float = 0.10,
    lambda_tox: float = 0.10,
    lambda_ce: float = 0.05,
    success_bonus: float = 1.0,
) -> float:
    """Variant B+C — path-length free-band + safety prior (V3B Phase 4).

    Terminal-only::

        R_T = success_bonus · 1[is_success]
              − path_penalty(step_idx)            (freeband schedule, Variant B)
              − λ_tox · tox_path                  (Chronos prior, Variant C)
              − λ_ce  · common_essential_count    (Chronos prior, Variant C)

    ``path_penalty`` is the 3-region freeband schedule from
    :func:`path_length_freeband_reward`.

    Reductions:
    * λ_tox = λ_ce = 0  → reduces to ``path_length_freeband`` (Variant B).
    * free_steps = max_steps, mild_until = max_steps, mild_beta = 0, heavy_beta = 0
      → reduces to ``safety_aware`` with β=0 (pure safety, no step cost).
    """
    if not (terminated or truncated):
        return 0.0
    T = int(step_idx)
    fs = int(free_steps); mu = int(mild_until)
    mb = float(mild_beta); hb = float(heavy_beta)
    if T <= fs:
        path_pen = 0.0
    elif T <= mu:
        path_pen = mb * float(T - fs)
    else:
        path_pen = mb * float(mu - fs) + hb * float(T - mu)
    base = float(success_bonus) * (1.0 if is_success else 0.0)
    safety = float(lambda_tox) * float(tox_path) + float(lambda_ce) * float(common_essential_count)
    return base - path_pen - safety


def biorealistic_fused_reward(
    *,
    is_success: bool,
    terminated: bool,
    truncated: bool,
    step_idx: int,
    tox_path: float,
    common_essential_count: int,
    unc_path_max: float,
    free_steps: int = 3,
    mild_until: int = 5,
    mild_beta: float = 0.02,
    heavy_beta: float = 0.10,
    lambda_tox: float = 0.10,
    lambda_ce: float = 0.05,
    lambda_unc: float = 0.05,
    success_bonus: float = 1.0,
) -> float:
    """Variant B+C+D — full biorealistic fused reward (V3B Phase 4).

    Terminal-only::

        R_T = success_bonus · 1[is_success]
              − path_penalty(step_idx)            (freeband, Variant B)
              − λ_tox · tox_path                  (safety, Variant C)
              − λ_ce  · common_essential_count    (safety, Variant C)
              − λ_unc · unc_path_max              (uncertainty, Variant D)

    All four axes additively composed. The synthetic-lethal (SL) term from the
    V3B plan §4 Variant E is **structurally inactive** because Horlbeck 2018
    K562 SL pairs have zero overlap with the Norman 105 cell-fate-gene action
    universe (see ``artifacts_v3/v3b_biology/coverage.json``). Variant E
    therefore reduces to B+C+D on this action space.

    Reductions:
    * λ_tox = λ_ce = λ_unc = 0  →  ``path_length_freeband`` (Variant B).
    * λ_unc = 0 + freeband = (T·β-equivalent linear schedule)  →  ``safety_path_freeband`` (B+C).
    * λ_tox = λ_ce = 0  →  freeband + uncertainty.
    * All four axes off (terminal_only_step_cost equivalents)  → V2 baseline.

    Numerical stability: caller supplies bounded ``unc_path_max`` via
    :func:`per_step_uncertainty_scalar`. ``tox_path`` is sum of clamped
    ``max(0, -Chronos - 0.5)`` over actions; bounded above by K·max_tox. Total
    reward bounded in roughly [−K·(β_total + λ_tox·max_tox + λ_unc·max_unc),
    success_bonus].
    """
    if not (terminated or truncated):
        return 0.0
    T = int(step_idx)
    fs = int(free_steps); mu = int(mild_until)
    mb = float(mild_beta); hb = float(heavy_beta)
    if T <= fs:
        path_pen = 0.0
    elif T <= mu:
        path_pen = mb * float(T - fs)
    else:
        path_pen = mb * float(mu - fs) + hb * float(T - mu)
    base = float(success_bonus) * (1.0 if is_success else 0.0)
    safety = float(lambda_tox) * float(tox_path) + float(lambda_ce) * float(common_essential_count)
    unc = float(lambda_unc) * float(unc_path_max)
    return base - path_pen - safety - unc


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
