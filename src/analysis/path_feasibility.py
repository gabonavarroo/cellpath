"""V3B biology-aware path feasibility scoring.

Loads the per-gene safety table and the K562 SL pair table produced by
``scripts/build_v3b_biology_layer.py`` and provides ``score_episode`` for
per-episode safety / SL / uncertainty / realism metrics.

Single source of truth for V3B biology scoring (CLAUDE.md §3 rule 4).

Mathematical definitions (matching V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §4):

* ``tox(g) = max(0, -Chronos(g) - 0.5)``     (raw toxicity score, ≥ 0)
* ``tox_norm(g) = min(1.0, tox(g))``         (bounded toxicity, ∈ [0, 1])
* ``tox_path = Σ_{a ≠ NOOP} tox(a)``         (path-aggregate raw toxicity)
* ``common_essential_count = #{a ≠ NOOP : Chronos(a) < -0.5}``
* ``sl_violations(path) = #{(i, j) : i < j, (a_i, a_j) ∈ K562_SL ∪ (a_j, a_i) ∈ K562_SL}``
* ``peak_unc(path) = max_{a ≠ NOOP} ‖log σ²(z, a)‖_2``  (heteroscedastic head output)
* ``mean_unc(path) = mean_{a ≠ NOOP} ‖log σ²(z, a)‖_2``

Realism composite (V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §9.3):

    realism = w_chr·(1 − norm_tox) + w_sl·1[no SL violation]
            + w_ess·(1 − frac_common_essential) + w_nor·1[final on Norman manifold]
    where w_chr=0.4, w_sl=0.3, w_ess=0.2, w_nor=0.1

The Norman-manifold check requires an empirical reference set; the API exposes
``on_norman_manifold`` as a caller-supplied bool. If not given, ``w_nor=0`` is
absorbed into ``w_chr`` (renormalisation).

Honesty caveats:
* DepMap Chronos is CRISPR-Cas9 knockout; Norman 2019 is CRISPRa. ``tox`` is a
  prior on "experimental disturbance", not therapeutic toxicity.
* The Horlbeck-2018 K562 SL pair set has **zero overlap** with the Norman 105
  cell-fate-gene action universe (build_v3b_biology_layer.py coverage.json).
  Therefore ``sl_violations`` is structurally 0 for all V3B episodes until a
  different SL source (e.g. DepMap co-essentiality) provides coverage on the
  Norman action space. The function still computes it correctly so the table
  is non-trivial if SL coverage appears later.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiologyLayer:
    """Lightweight container for the biology annotation layer.

    Attributes
    ----------
    gene_safety
        Polars DataFrame from gene_safety.parquet (105 rows × {gene_symbol,
        action_idx, chronos, is_essential, tox_raw, tox_norm, missing_chronos}).
    sl_pairs
        Polars DataFrame from k562_sl_pairs.parquet (may be empty).
    coverage
        Loaded coverage.json content for provenance access.

    Derived lookup tables (computed in __post_init__):

    * ``tox_by_action``: dict[int, float] mapping action_idx → tox_raw
      (0.0 for missing).
    * ``tox_norm_by_action``: dict[int, float] mapping action_idx → tox_norm.
    * ``is_essential_by_action``: dict[int, bool].
    * ``sl_pair_set``: frozenset of (action_idx_a, action_idx_b) canonical
      ordered pairs (a < b).
    """

    gene_safety: pl.DataFrame
    sl_pairs: pl.DataFrame
    coverage: Mapping[str, Any]

    tox_by_action: Mapping[int, float] = field(init=False, default_factory=dict)
    tox_norm_by_action: Mapping[int, float] = field(init=False, default_factory=dict)
    is_essential_by_action: Mapping[int, bool] = field(init=False, default_factory=dict)
    sl_pair_set: frozenset = field(init=False, default_factory=frozenset)

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass setattr restriction via object.__setattr__.
        tox = {}
        tox_n = {}
        ess = {}
        for row in self.gene_safety.iter_rows(named=True):
            idx = int(row["action_idx"])
            t = row.get("tox_raw")
            tn = row.get("tox_norm")
            e = row.get("is_essential")
            tox[idx] = float(t) if t is not None else 0.0
            tox_n[idx] = float(tn) if tn is not None else 0.0
            ess[idx] = bool(e) if e is not None else False
        object.__setattr__(self, "tox_by_action", tox)
        object.__setattr__(self, "tox_norm_by_action", tox_n)
        object.__setattr__(self, "is_essential_by_action", ess)

        pairs: set[tuple[int, int]] = set()
        if self.sl_pairs.height > 0:
            for row in self.sl_pairs.iter_rows(named=True):
                a = int(row["action_idx_a"])
                b = int(row["action_idx_b"])
                lo, hi = (a, b) if a < b else (b, a)
                pairs.add((lo, hi))
        object.__setattr__(self, "sl_pair_set", frozenset(pairs))


def load_biology_layer(biology_dir: str | Path) -> BiologyLayer:
    """Load the V3B biology layer artifacts.

    Parameters
    ----------
    biology_dir
        Directory containing ``gene_safety.parquet``, ``k562_sl_pairs.parquet``,
        and ``coverage.json`` (produced by ``scripts/build_v3b_biology_layer.py``).

    Returns
    -------
    BiologyLayer

    Raises
    ------
    FileNotFoundError
        If any of the three artifacts is missing.
    """
    biology_dir = Path(biology_dir)
    safety_path = biology_dir / "gene_safety.parquet"
    sl_path = biology_dir / "k562_sl_pairs.parquet"
    cov_path = biology_dir / "coverage.json"

    for p in (safety_path, sl_path, cov_path):
        if not p.exists():
            raise FileNotFoundError(
                f"V3B biology artifact missing: {p}. Run "
                f"scripts/build_v3b_biology_layer.py first."
            )

    safety = pl.read_parquet(str(safety_path))
    sl = pl.read_parquet(str(sl_path))
    with open(cov_path) as f:
        coverage = json.load(f)

    return BiologyLayer(gene_safety=safety, sl_pairs=sl, coverage=coverage)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


# Default realism composite weights (V3B plan §9.3)
DEFAULT_REALISM_WEIGHTS = {"chr": 0.4, "sl": 0.3, "ess": 0.2, "nor": 0.1}


def score_episode(
    actions: Sequence[int],
    layer: BiologyLayer,
    *,
    noop_idx: int,
    log_var_per_step: np.ndarray | Sequence[np.ndarray] | None = None,
    on_norman_manifold: bool | None = None,
    realism_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Score a single episode under the V3B biology layer.

    Parameters
    ----------
    actions
        Sequence of action ids taken in the episode. NOOP entries are skipped
        for safety/SL aggregation but counted in ``n_steps``.
    layer
        Output of :func:`load_biology_layer`.
    noop_idx
        Index of the NO-OP action (= ``gene_vocab.json::noop_idx``).
    log_var_per_step
        Optional shape-``(n_steps, n_latent)`` array (or sequence of arrays)
        of the dynamics' per-step ``log σ²`` output. If provided, ``peak_unc``
        and ``mean_unc`` are computed.
    on_norman_manifold
        Optional caller-supplied bool indicating whether the terminal state lies
        on the empirically-observed Norman-perturbation latent manifold. If None,
        the realism composite renormalises the remaining weights.
    realism_weights
        Mapping with keys ``chr, sl, ess, nor``. Defaults to
        :data:`DEFAULT_REALISM_WEIGHTS`.

    Returns
    -------
    dict with keys:
        ``n_steps``                 — total steps (incl. NOOP).
        ``n_gene_steps``            — non-NOOP actions.
        ``tox_path``                — Σ tox_raw across non-NOOP actions.
        ``tox_path_norm``           — Σ tox_norm across non-NOOP actions.
        ``mean_tox``                — tox_path / max(1, n_gene_steps).
        ``common_essential_count``  — Σ is_essential across non-NOOP actions.
        ``frac_common_essential``   — common_essential_count / max(1, n_gene_steps).
        ``frac_safe_actions``       — fraction of non-NOOP actions with
                                       chronos > -0.5 (i.e. NOT common-essential).
        ``sl_violations``           — count of distinct SL pairs visited within the
                                       path (canonical-ordered pair check).
        ``peak_unc``                — max log-σ² L2 norm, or None.
        ``mean_unc``                — mean log-σ² L2 norm, or None.
        ``realism``                 — composite score in [0, 1].
        ``realism_weights``         — the weights used (after renormalisation if
                                       on_norman_manifold is None).
    """
    weights = dict(realism_weights or DEFAULT_REALISM_WEIGHTS)

    gene_actions = [int(a) for a in actions if int(a) != int(noop_idx)]
    n_steps = len(actions)
    n_gene = len(gene_actions)

    tox_path = sum(layer.tox_by_action.get(a, 0.0) for a in gene_actions)
    tox_path_norm = sum(layer.tox_norm_by_action.get(a, 0.0) for a in gene_actions)
    ce_count = sum(int(layer.is_essential_by_action.get(a, False)) for a in gene_actions)

    sl_violations = 0
    # All unordered pairs of distinct (gene) actions; check membership in canonical pair set.
    for i in range(n_gene):
        for j in range(i + 1, n_gene):
            a, b = gene_actions[i], gene_actions[j]
            lo, hi = (a, b) if a < b else (b, a)
            if (lo, hi) in layer.sl_pair_set:
                sl_violations += 1

    peak_unc: float | None = None
    mean_unc: float | None = None
    if log_var_per_step is not None:
        arr = np.asarray(log_var_per_step, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]  # single-step case
        # arr shape: (n_steps_recorded, n_latent)
        per_step_norm = np.linalg.norm(arr, axis=1)
        if per_step_norm.size > 0:
            peak_unc = float(per_step_norm.max())
            mean_unc = float(per_step_norm.mean())

    # Realism composite
    chr_term = 1.0 - min(1.0, tox_path_norm / max(1, n_gene))  # mean normalised tox in [0,1]
    sl_term = 1.0 if sl_violations == 0 else 0.0
    ess_term = 1.0 - (ce_count / max(1, n_gene))
    if on_norman_manifold is None:
        # Renormalise the remaining three terms to sum to 1.
        total = weights["chr"] + weights["sl"] + weights["ess"]
        wc, wsl, we = weights["chr"] / total, weights["sl"] / total, weights["ess"] / total
        realism = wc * chr_term + wsl * sl_term + we * ess_term
        w_used = {"chr": wc, "sl": wsl, "ess": we, "nor": 0.0, "_renormalised": True}
    else:
        nor_term = 1.0 if bool(on_norman_manifold) else 0.0
        realism = (
            weights["chr"] * chr_term
            + weights["sl"] * sl_term
            + weights["ess"] * ess_term
            + weights["nor"] * nor_term
        )
        w_used = {**weights, "_renormalised": False}

    return {
        "n_steps": int(n_steps),
        "n_gene_steps": int(n_gene),
        "tox_path": float(tox_path),
        "tox_path_norm": float(tox_path_norm),
        "mean_tox": float(tox_path / max(1, n_gene)),
        "common_essential_count": int(ce_count),
        "frac_common_essential": float(ce_count / max(1, n_gene)),
        "frac_safe_actions": float(1.0 - (ce_count / max(1, n_gene))),
        "sl_violations": int(sl_violations),
        "peak_unc": peak_unc,
        "mean_unc": mean_unc,
        "realism": float(realism),
        "realism_weights": w_used,
    }


def aggregate_episode_scores(per_episode: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate :func:`score_episode` outputs across a batch of episodes.

    Returns mean ± std + count of successes-on-safety-axis if available.
    """
    per_episode = list(per_episode)
    if not per_episode:
        return {"n_episodes": 0}

    def _mean_std(key: str) -> tuple[float | None, float | None]:
        vals = [float(e[key]) for e in per_episode if e.get(key) is not None]
        if not vals:
            return None, None
        arr = np.asarray(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    out: dict[str, Any] = {"n_episodes": len(per_episode)}
    for key in (
        "n_steps", "n_gene_steps", "tox_path", "tox_path_norm", "mean_tox",
        "common_essential_count", "frac_common_essential", "frac_safe_actions",
        "sl_violations", "peak_unc", "mean_unc", "realism",
    ):
        m, s = _mean_std(key)
        out[f"{key}_mean"] = m
        out[f"{key}_std"] = s
    out["fraction_zero_sl_violations"] = float(
        np.mean([1.0 if e["sl_violations"] == 0 else 0.0 for e in per_episode])
    )
    out["fraction_zero_common_essential"] = float(
        np.mean([1.0 if e["common_essential_count"] == 0 else 0.0 for e in per_episode])
    )
    return out
