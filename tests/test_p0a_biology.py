from __future__ import annotations

import pandas as pd
import pytest


def test_action_freq_chronos_spearman_uses_full_overlap() -> None:
    from src.analysis.metrics import action_freq_chronos_spearman

    chronos = pd.Series({"A": -1.0, "B": -0.5, "C": 0.0, "D": 0.5})
    result = action_freq_chronos_spearman(
        {"A": 40, "B": 30, "C": 20, "D": 10, "NO_OP": 999},
        chronos,
        n_boot=200,
        seed=0,
    )

    assert result["n_overlap"] == 4
    assert result["rho"] == pytest.approx(-1.0)
    assert result["ci95_low"] <= result["rho"] <= result["ci95_high"]


def test_preranked_gsea_returns_bh_corrected_panel_rows() -> None:
    from src.analysis.depmap_validation import preranked_gsea

    chronos = pd.Series({"A": -1.0, "B": -0.8, "C": 0.0, "D": 0.3, "E": 0.5})
    rows = preranked_gsea(
        {"A": 50, "B": 40, "C": 5, "D": 2, "E": 1},
        chronos,
        {"essential_panel": ["A", "B"], "irrelevant_panel": ["D", "E"]},
        n_perm=200,
        seed=1,
    )

    by_panel = {row["panel"]: row for row in rows}
    assert set(by_panel) == {"essential_panel", "irrelevant_panel"}
    assert by_panel["essential_panel"]["nes"] > 0
    assert 0.0 <= by_panel["essential_panel"]["q_value"] <= 1.0
