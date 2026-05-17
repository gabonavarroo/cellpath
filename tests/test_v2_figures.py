"""P0F Phase 5 — smoke test for `src/analysis/v2_figures.py`.

Builds a tiny mock evaluation tree and verifies the figure module emits non-empty PNGs
without raising.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _torch_or_skip():
    return pytest.importorskip("torch")


@pytest.fixture()
def mock_eval_tree(tmp_path: Path) -> dict[str, Path]:
    """Build two PPO eval dirs with a few cells each."""
    cells = [
        ("k2_epsp25_bin6-8_splitood", "ppo_deterministic", 0.65),
        ("k2_epsp25_bin6-8_splitood", "random_uniform_valid", 0.05),
        ("k2_epsp25_bin6-8_splitood", "greedy_dyn_1", 0.70),
        ("k2_epsp25_bin6-8_splitood", "greedy_dyn_2", 0.74),
        ("k3_epsp25_bin8-10_splitood", "ppo_deterministic", 1.0),
        ("k3_epsp25_bin8-10_splitood", "random_uniform_valid", 0.14),
        ("k3_epsp25_bin8-10_splitood", "greedy_dyn_1", 1.0),
        ("k3_epsp25_bin8-10_splitood", "greedy_dyn_2", 1.0),
    ]
    eval_dirs = {}
    for cfg in ("B5", "C2"):
        cfg_root = tmp_path / f"eval_{cfg}"
        for cell, pol, rate in cells:
            d = cfg_root / cell / pol
            d.mkdir(parents=True, exist_ok=True)
            (d / "summary.json").write_text(json.dumps({
                "success_rate": rate,
                "mean_steps": 2.3,
                "mean_final_distance": 2.5,
                "n_episodes": 200,
                "successes": int(rate * 200),
            }))
        eval_dirs[cfg] = cfg_root
    return eval_dirs


def test_generate_all_figures_smoke(tmp_path: Path, mock_eval_tree: dict[str, Path]) -> None:
    """Smoke test: all six plotting functions complete without exception and emit PNGs.

    Some inputs (rollouts.parquet, seed-aggregate JSON) are intentionally absent — the
    figure module must skip those gracefully and emit the remaining figures.
    """
    _torch_or_skip()
    from src.analysis.v2_figures import generate_all_figures

    out_dir = tmp_path / "figs"

    # Build a tiny dynamics-taxonomy table.
    dynamics_rows = [
        {"label": "V1 OT",       "gate_val_margin": 0.0074, "ood_pearson": 0.479,
         "beam_success": 1.000, "ppo_primary": 1.000},
        {"label": "RoR_corr010", "gate_val_margin": 0.0136, "ood_pearson": 0.516,
         "beam_success": 1.000, "ppo_primary": 1.000},
        {"label": "mean_delta",  "gate_val_margin": 0.0232, "ood_pearson": 0.385,
         "beam_success": 0.000, "ppo_primary": 0.000},
        {"label": "soft_ot",     "gate_val_margin": 0.0413, "ood_pearson": 0.743,
         "beam_success": 0.000, "ppo_primary": 0.000},
    ]

    written = generate_all_figures(
        eval_dirs=mock_eval_tree,
        run_dirs={},                       # no action_freq.json — that figure should skip
        seed_aggregate_json=tmp_path / "missing.json",  # missing — should skip
        dynamics_taxonomy_rows=dynamics_rows,
        rollouts_paths={},                 # no rollouts — skip
        out_dir=out_dir,
    )
    # At least the three figures that don't depend on missing inputs should exist:
    assert any(p.name == "success_vs_K.png"      for p in written)
    assert any(p.name == "hardness_frontier.png" for p in written)
    assert any(p.name == "dynamics_taxonomy.png" for p in written)
    for p in written:
        assert p.stat().st_size > 0, f"{p} is empty"
