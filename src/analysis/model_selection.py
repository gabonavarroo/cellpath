"""Checkpoint selection logic for the dynamics model.

Owner: Agent B.

Contains :func:`recommend_checkpoint`, which compares the best-NLL and best-gate-margin
checkpoints produced by ``scripts/train_dynamics.py`` and returns a conservative
recommendation for which one should become the final ``model.pt``.

This module is intentionally small and pure — no I/O, no torch, no sklearn.  All metric
computation lives in ``src.analysis.metrics``; this module only implements the decision
tree for checkpoint selection.

CLAUDE.md sacred rule #4: the recommendation *decision* lives here; the metric values that
feed the decision are computed in ``src.analysis.metrics``.
"""

from __future__ import annotations

from typing import Any


def recommend_checkpoint(
    best_nll_eval: dict[str, Any] | None,
    best_gate_eval: dict[str, Any] | None,
    *,
    ood_tolerance: float = 0.02,
    min_uncertainty: float = 0.20,
) -> tuple[str, str]:
    """Compare best-NLL and best-gate-margin checkpoints; return a conservative recommendation.

    Decision tree (evaluated in order — first matching rule wins):

    1. ``best_gate_eval`` is ``None``
       → ``"keep_best_nll"``  (no gate checkpoint was ever saved).
    2. ``best_nll_eval`` is ``None``
       → ``"consider_best_gate"``  (no NLL reference; gate is the only evaluated option).
    3. ``best_gate_eval["val"]["mlp_minus_ridge_pearson"]
       ≤ best_nll_eval["val"]["mlp_minus_ridge_pearson"]``
       → ``"keep_best_nll"``  (gate checkpoint did not improve the target val metric).
    4. ``best_gate_eval["val"]["uncertainty_spearman"] < min_uncertainty``
       → ``"reject_best_gate"``  (uncertainty calibration is broken).
    5. OOD data available **and**
       (``best_gate["ood"]["mlp_pearson"] < best_nll["ood"]["mlp_pearson"] − ood_tolerance``
       **or** ``best_gate["ood"]["mlp_r2"] < best_nll["ood"]["mlp_r2"] − ood_tolerance``)
       → ``"reject_best_gate"``  (OOD performance collapsed).
    6. Otherwise → ``"consider_best_gate"``.

    The rule is conservative by design: the default falls back to ``keep_best_nll`` when
    evidence is absent or ambiguous.  ``consider_best_gate`` means the gate checkpoint
    *looks* better on the measured criteria; the engineer should still inspect
    ``checkpoint_comparison.json`` before committing to it.

    Parameters
    ----------
    best_nll_eval
        Dict produced by ``_eval_ckpt(best_nll_path, ...)`` in
        ``scripts/train_dynamics.py``.  Schema::

            {
              "val": {
                "mlp_pearson": float,
                "mlp_r2": float,
                "mlp_minus_ridge_pearson": float,
                "uncertainty_spearman": float,
                ...
              },
              "ood": {        # None when OOD pairs are absent
                "mlp_pearson": float,
                "mlp_r2": float,
                ...
              } | None,
            }

    best_gate_eval
        Same schema as ``best_nll_eval``.  ``None`` when no gate checkpoint was saved
        during training (e.g. no epoch satisfied the uncertainty filter).
    ood_tolerance
        Maximum allowed degradation in OOD Pearson/R² for the gate checkpoint relative to
        the NLL checkpoint before it is rejected.  Default 0.02 (2 pp).
    min_uncertainty
        Minimum accepted ``uncertainty_spearman`` for the gate checkpoint.  Default 0.20
        matches the dynamics gate threshold in ``config/dynamics.yaml``.

    Returns
    -------
    tuple[str, str]
        ``(recommendation, rationale)`` where *recommendation* is one of:

        - ``"keep_best_nll"`` — use the best-NLL checkpoint as ``model.pt``.
        - ``"consider_best_gate"`` — the gate checkpoint looks better; worth trying,
          but verify ``checkpoint_comparison.json`` before relying on it.
        - ``"reject_best_gate"`` — the gate checkpoint degrades calibration or OOD;
          do not use it.
    """
    # Rule 1: no gate checkpoint saved
    if best_gate_eval is None:
        return (
            "keep_best_nll",
            "No gate-margin checkpoint was saved during training.",
        )

    # Rule 2: no NLL reference (edge case — should not occur in practice)
    if best_nll_eval is None:
        return (
            "consider_best_gate",
            "No NLL checkpoint evaluation available; best-gate is the only evaluated option.",
        )

    gate_val = best_gate_eval.get("val") or {}
    nll_val  = best_nll_eval.get("val") or {}

    gate_margin = float(gate_val.get("mlp_minus_ridge_pearson", float("-inf")))
    nll_margin  = float(nll_val.get("mlp_minus_ridge_pearson", float("-inf")))

    # Rule 3: gate did not improve the target val metric
    if gate_margin <= nll_margin:
        return (
            "keep_best_nll",
            f"best_gate val mlp_minus_ridge_pearson ({gate_margin:+.4f}) did not improve "
            f"over best_nll ({nll_margin:+.4f}).",
        )

    # Rule 4: uncertainty calibration broken
    gate_unc = float(gate_val.get("uncertainty_spearman", 0.0))
    if gate_unc < min_uncertainty:
        return (
            "reject_best_gate",
            f"best_gate uncertainty_spearman ({gate_unc:.4f}) is below the minimum "
            f"required threshold ({min_uncertainty:.2f}).",
        )

    # Rule 5: OOD collapse
    gate_ood = best_gate_eval.get("ood")
    nll_ood  = best_nll_eval.get("ood")
    if gate_ood is not None and nll_ood is not None:
        gate_ood_pearson = float(gate_ood.get("mlp_pearson", 0.0))
        nll_ood_pearson  = float(nll_ood.get("mlp_pearson", 0.0))
        gate_ood_r2      = float(gate_ood.get("mlp_r2", 0.0))
        nll_ood_r2       = float(nll_ood.get("mlp_r2", 0.0))

        if gate_ood_pearson < nll_ood_pearson - ood_tolerance:
            return (
                "reject_best_gate",
                f"best_gate OOD Pearson ({gate_ood_pearson:.4f}) dropped more than "
                f"ood_tolerance={ood_tolerance} below best_nll ({nll_ood_pearson:.4f}).",
            )
        if gate_ood_r2 < nll_ood_r2 - ood_tolerance:
            return (
                "reject_best_gate",
                f"best_gate OOD R² ({gate_ood_r2:.4f}) dropped more than "
                f"ood_tolerance={ood_tolerance} below best_nll ({nll_ood_r2:.4f}).",
            )

    # Rule 6: all checks passed → recommend considering the gate checkpoint
    return (
        "consider_best_gate",
        f"best_gate improves val mlp_minus_ridge_pearson "
        f"({nll_margin:+.4f} → {gate_margin:+.4f}), "
        f"uncertainty_spearman={gate_unc:.4f} ≥ {min_uncertainty:.2f}, "
        f"and OOD is within tolerance.",
    )
