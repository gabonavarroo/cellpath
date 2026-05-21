"""Lightweight test: champion manifest parses and key referenced paths exist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "artifacts_v3/v3c/final_champion_manifest.json"


@pytest.mark.skipif(not MANIFEST_PATH.exists(),
                    reason="Champion manifest not yet generated (Stage 4.5 of V3C session).")
def test_manifest_parses_and_required_keys_present():
    payload = json.loads(MANIFEST_PATH.read_text())

    # Top-level structure
    assert "champion" in payload, "manifest missing 'champion' block"
    assert "baselines" in payload, "manifest missing 'baselines' block"
    assert "reproduction" in payload, "manifest missing 'reproduction' block"

    champ = payload["champion"]
    required_keys = {
        "champion_name", "champion_type", "dynamics_dir", "dynamics_field_id",
        "ppo_checkpoint", "reward_mode", "reward_coefficients",
        "epsilon_label", "epsilon_scalar",
        "vae_dir", "pairs_dir",
        "training_horizon", "seeds",
        "evaluation_matrix", "best_metrics", "limitations",
    }
    missing = required_keys - set(champ)
    assert not missing, f"champion missing required keys: {sorted(missing)}"

    # Type/value sanity
    assert champ["champion_type"] in {
        "LOCKED_DEFAULT_RESULT", "CHAMPION_TUNED_RESULT", "DIAGNOSTIC_ONLY",
    }
    assert isinstance(champ["epsilon_scalar"], (int, float))
    assert isinstance(champ["seeds"], list) and champ["seeds"]
    assert isinstance(champ["limitations"], list) and champ["limitations"]


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Champion manifest not yet generated.")
def test_manifest_referenced_paths_exist():
    payload = json.loads(MANIFEST_PATH.read_text())
    champ = payload["champion"]
    for key in ("dynamics_dir", "vae_dir", "pairs_dir"):
        p = REPO_ROOT / champ[key]
        assert p.exists(), f"champion {key} → {p} does not exist"
    ppo = REPO_ROOT / champ["ppo_checkpoint"]
    assert ppo.exists(), f"champion ppo_checkpoint → {ppo} does not exist"


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Champion manifest not yet generated.")
def test_reproduction_commands_present():
    payload = json.loads(MANIFEST_PATH.read_text())
    rep = payload["reproduction"]
    assert "evaluate_command" in rep
    assert "full_pipeline_command" in rep
    # Commands should be strings starting with a python invocation or make
    for k in ("evaluate_command", "full_pipeline_command"):
        cmd = rep[k]
        assert isinstance(cmd, str) and len(cmd) > 0
        assert any(tok in cmd for tok in ("python", "make"))
