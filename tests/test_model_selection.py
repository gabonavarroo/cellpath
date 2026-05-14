"""Tests for src.analysis.model_selection.recommend_checkpoint.

All tests use synthetic eval dicts; no I/O, no torch.
"""

from __future__ import annotations


def _nll_eval(**overrides) -> dict:
    """Baseline eval dict representing the best-NLL checkpoint."""
    base: dict = {
        "val": {
            "mlp_pearson":             0.6031,
            "mlp_r2":                  0.380,
            "ridge_pearson":           0.6011,
            "ridge_r2":                0.383,
            "mlp_minus_ridge_pearson": 0.002,
            "uncertainty_spearman":    0.247,
            "passed":                  False,
        },
        "ood": {
            "mlp_pearson":             0.4854,
            "mlp_r2":                  0.050,
            "ridge_pearson":           0.470,
            "ridge_r2":                0.040,
            "mlp_minus_ridge_pearson": 0.015,
            "uncertainty_spearman":    0.220,
        },
    }
    for k, v in overrides.items():
        if "." in k:
            block, key = k.split(".", 1)
            base[block][key] = v
        else:
            base[k] = v
    return base


def _gate_eval_improved(**overrides) -> dict:
    """Gate checkpoint with improved val margin (val mlp_minus_ridge=0.025 > nll's 0.002)."""
    base = _nll_eval()
    base["val"]["mlp_minus_ridge_pearson"] = 0.025
    for k, v in overrides.items():
        if "." in k:
            block, key = k.split(".", 1)
            base[block][key] = v
        else:
            base[k] = v
    return base


class TestRecommendCheckpoint:
    """Decision-tree tests for recommend_checkpoint."""

    def test_gate_improves_val_and_preserves_ood(self) -> None:
        """Gate improves val margin and OOD unchanged → consider_best_gate."""
        from src.analysis.model_selection import recommend_checkpoint

        rec, rationale = recommend_checkpoint(_nll_eval(), _gate_eval_improved())
        assert rec == "consider_best_gate"
        assert len(rationale) > 0

    def test_gate_does_not_improve_val(self) -> None:
        """Gate margin ≤ NLL margin → keep_best_nll."""
        from src.analysis.model_selection import recommend_checkpoint

        gate = _gate_eval_improved()
        gate["val"]["mlp_minus_ridge_pearson"] = 0.001   # worse than nll's 0.002
        rec, rationale = recommend_checkpoint(_nll_eval(), gate)
        assert rec == "keep_best_nll"
        assert "0.001" in rationale or "did not improve" in rationale.lower()

    def test_gate_equal_val_margin_uses_keep_best_nll(self) -> None:
        """Gate margin exactly equal to NLL margin → keep_best_nll (≤ not <)."""
        from src.analysis.model_selection import recommend_checkpoint

        gate = _gate_eval_improved()
        gate["val"]["mlp_minus_ridge_pearson"] = 0.002   # equal to nll baseline
        rec, _ = recommend_checkpoint(_nll_eval(), gate)
        assert rec == "keep_best_nll"

    def test_gate_breaks_uncertainty(self) -> None:
        """Gate improves val but uncertainty_spearman < 0.20 → reject_best_gate."""
        from src.analysis.model_selection import recommend_checkpoint

        gate = _gate_eval_improved()
        gate["val"]["uncertainty_spearman"] = 0.10
        rec, rationale = recommend_checkpoint(_nll_eval(), gate)
        assert rec == "reject_best_gate"
        assert "uncertainty" in rationale.lower()

    def test_gate_improves_val_but_ood_pearson_collapses(self) -> None:
        """Gate improves val but OOD Pearson drops > ood_tolerance → reject_best_gate."""
        from src.analysis.model_selection import recommend_checkpoint

        gate = _gate_eval_improved()
        # nll ood Pearson = 0.4854; drop by >0.02 → reject
        gate["ood"]["mlp_pearson"] = 0.20
        rec, rationale = recommend_checkpoint(_nll_eval(), gate)
        assert rec == "reject_best_gate"
        assert "OOD" in rationale or "ood" in rationale.lower()

    def test_gate_improves_val_but_ood_r2_collapses(self) -> None:
        """Gate improves val but OOD R² drops > ood_tolerance → reject_best_gate."""
        from src.analysis.model_selection import recommend_checkpoint

        gate = _gate_eval_improved()
        # nll ood R² = 0.050; set well below 0.05 - 0.02 = 0.03
        gate["ood"]["mlp_r2"] = -0.10
        rec, rationale = recommend_checkpoint(_nll_eval(), gate)
        assert rec == "reject_best_gate"
        assert "R²" in rationale or "r2" in rationale.lower() or "ood" in rationale.lower()

    def test_missing_gate_eval_returns_keep_best_nll(self) -> None:
        """best_gate_eval=None → keep_best_nll (no gate checkpoint was saved)."""
        from src.analysis.model_selection import recommend_checkpoint

        rec, rationale = recommend_checkpoint(_nll_eval(), None)
        assert rec == "keep_best_nll"
        assert len(rationale) > 0

    def test_missing_nll_eval_returns_consider_best_gate(self) -> None:
        """best_nll_eval=None (edge case) → consider_best_gate."""
        from src.analysis.model_selection import recommend_checkpoint

        rec, _ = recommend_checkpoint(None, _gate_eval_improved())
        assert rec == "consider_best_gate"

    def test_both_none_returns_keep_best_nll(self) -> None:
        """Both None → keep_best_nll (rule 1: no gate checkpoint)."""
        from src.analysis.model_selection import recommend_checkpoint

        rec, _ = recommend_checkpoint(None, None)
        assert rec == "keep_best_nll"

    def test_ood_absent_from_gate_still_considers(self) -> None:
        """If gate checkpoint has no OOD block, OOD veto is skipped → consider_best_gate."""
        from src.analysis.model_selection import recommend_checkpoint

        gate = _gate_eval_improved()
        gate["ood"] = None   # no OOD data in gate eval
        rec, _ = recommend_checkpoint(_nll_eval(), gate)
        assert rec == "consider_best_gate"

    def test_ood_absent_from_nll_still_considers(self) -> None:
        """If NLL checkpoint has no OOD block, OOD veto is also skipped → consider_best_gate."""
        from src.analysis.model_selection import recommend_checkpoint

        nll  = _nll_eval()
        nll["ood"] = None
        gate = _gate_eval_improved()
        gate["ood"] = None
        rec, _ = recommend_checkpoint(nll, gate)
        assert rec == "consider_best_gate"

    def test_custom_ood_tolerance(self) -> None:
        """Relaxed ood_tolerance lets a slightly OOD-dropping gate pass."""
        from src.analysis.model_selection import recommend_checkpoint

        gate = _gate_eval_improved()
        # Drop OOD Pearson by 0.03 — just over the default tolerance of 0.02.
        gate["ood"]["mlp_pearson"] = 0.4854 - 0.03
        rec_default, _  = recommend_checkpoint(_nll_eval(), gate)
        rec_relaxed, _  = recommend_checkpoint(_nll_eval(), gate, ood_tolerance=0.05)
        assert rec_default == "reject_best_gate"
        assert rec_relaxed == "consider_best_gate"
