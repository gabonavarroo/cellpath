"""Tests for P0 RL evaluation infrastructure.

Covers:
- ``resolve_epsilon`` (src.rl.environment) — override vs JSON precedence.
- ``_write_run_metadata`` (src.rl.train_ppo) — robust to missing files; key presence.
- ``scripts/summarize_rl_run`` — Shannon entropy + summary on a tiny synthetic parquet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


# =============================================================================
# resolve_epsilon — override vs JSON precedence
# =============================================================================


def _make_eps_cfg(
    tmp_path: Path,
    *,
    json_value: float = 4.52,
    json_percentile: int = 90,
    override: Any = None,
) -> Any:
    """Build a minimal SimpleNamespace cfg with the keys ``resolve_epsilon`` reads."""
    eps_path = tmp_path / "epsilon_success.json"
    eps_path.write_text(json.dumps({
        "percentile": json_percentile,
        "value": float(json_value),
        "n_ctrl_cells": 11_855,
        "method": "p90",
    }))

    # rl.env supports dict-style .get() — SimpleNamespace doesn't, so use a small dict shim.
    class _Env(dict):
        # Allow both attr and .get access (mirrors OmegaConf DictConfig surface)
        def __getattr__(self, k: str) -> Any:
            try:
                return self[k]
            except KeyError as exc:
                raise AttributeError(k) from exc

    env = _Env()
    env["epsilon_override"] = override

    rl = SimpleNamespace(env=env)
    paths = SimpleNamespace(vae_epsilon_success_json=str(eps_path))
    return SimpleNamespace(rl=rl, paths=paths)


class TestResolveEpsilon:
    def test_no_override_reads_json(self, tmp_path: Path) -> None:
        from src.rl.environment import resolve_epsilon

        cfg = _make_eps_cfg(tmp_path, json_value=4.52, json_percentile=90, override=None)
        eps, source = resolve_epsilon(cfg)
        assert eps == pytest.approx(4.52)
        assert source == "json(p90)"

    def test_override_wins(self, tmp_path: Path) -> None:
        from src.rl.environment import resolve_epsilon

        cfg = _make_eps_cfg(tmp_path, json_value=4.52, json_percentile=90, override=2.8)
        eps, source = resolve_epsilon(cfg)
        assert eps == pytest.approx(2.8)
        assert source.startswith("override(")
        assert "2.8" in source

    def test_override_float_string_safe(self, tmp_path: Path) -> None:
        """Hydra passes ``+rl.env.epsilon_override=2.8`` as a string in some flows."""
        from src.rl.environment import resolve_epsilon

        cfg = _make_eps_cfg(tmp_path, override="3.14")
        eps, source = resolve_epsilon(cfg)
        assert eps == pytest.approx(3.14)
        assert "3.14" in source

    def test_default_behavior_unchanged_when_override_null(self, tmp_path: Path) -> None:
        """Acceptance criterion: existing behavior is bit-for-bit identical when override is null."""
        from src.rl.environment import resolve_epsilon

        cfg = _make_eps_cfg(tmp_path, json_value=4.52, json_percentile=90, override=None)
        eps1, src1 = resolve_epsilon(cfg)

        # Even when the env block lacks the key entirely
        class _Bare(dict):
            def __getattr__(self, k: str) -> Any:
                try:
                    return self[k]
                except KeyError as exc:
                    raise AttributeError(k) from exc

        bare_env = _Bare()  # no epsilon_override key at all
        cfg_bare = SimpleNamespace(
            rl=SimpleNamespace(env=bare_env),
            paths=cfg.paths,
        )
        eps2, src2 = resolve_epsilon(cfg_bare)
        assert eps1 == eps2 == pytest.approx(4.52)
        assert src1 == src2 == "json(p90)"


# =============================================================================
# _write_run_metadata — robustness when files / cfg fields are missing
# =============================================================================


def _minimal_meta_cfg(tmp_path: Path) -> Any:
    """Build a minimal cfg accepted by ``_write_run_metadata`` even when most files are absent."""
    eps_path = tmp_path / "epsilon_success.json"
    eps_path.write_text(json.dumps({"percentile": 50, "value": 2.5}))

    class _D(dict):
        def __getattr__(self, k: str) -> Any:
            try:
                return self[k]
            except KeyError as exc:
                raise AttributeError(k) from exc

    paths = _D({
        "vae_epsilon_success_json": str(eps_path),
        "vae_dir": str(tmp_path),
        "dynamics_gate": str(tmp_path / "nonexistent_gate.json"),
        "dynamics_config": str(tmp_path / "nonexistent_dyn_config.json"),
        "dynamics_model": str(tmp_path / "nonexistent_model.pt"),
        "rl_ppo_zip": str(tmp_path / "ppo.zip"),
    })
    rl_env = _D({
        "max_steps": 10,
        "min_start_distance": "auto",
        "epsilon_override": None,
    })
    rl_train = _D({"skip_gate": True})
    rl_eval = _D({"deterministic": True, "n_rollout_episodes": 100})
    ppo = _D({"lr": 3e-4, "ent_coef": 0.02, "total_timesteps": 200_000})
    reward = _D({"lambda_sparse": 0.01, "success_bonus": 5.0, "failure_penalty": 5.0})
    rl = SimpleNamespace(env=rl_env, train=rl_train, eval=rl_eval, ppo=ppo, reward=reward)
    dynamics = _D({"use_state_linear_skip": True, "use_gene_delta_bias": False})
    vae = _D({"n_latent": 32})
    cfg = _D({"seed": 42, "rl": rl, "paths": paths, "dynamics": dynamics, "vae": vae})
    # rl is SimpleNamespace so .ppo etc. work; cfg.rl access works via _D.__getattr__
    return cfg


class TestWriteRunMetadata:
    def test_missing_files_does_not_crash(self, tmp_path: Path) -> None:
        from src.rl.train_ppo import _write_run_metadata

        cfg = _minimal_meta_cfg(tmp_path)
        out = _write_run_metadata(
            cfg, tmp_path / "out", deterministic=True, n_episodes=42,
            extras={"stage": "test"},
        )
        assert out.exists()
        blob = json.loads(out.read_text())

        # Schema basics
        assert blob["schema_version"] == 1
        assert blob["seed"] == 42
        assert blob["n_episodes"] == 42
        assert blob["deterministic_eval"] is True
        assert blob["stage"] == "test"

        # Epsilon provenance — JSON path because override is null
        assert blob["epsilon_value"] == pytest.approx(2.5)
        assert blob["epsilon_source"] == "json(p50)"
        assert blob["epsilon_percentile_json"] == 50

        # Missing files surface as None, not crash
        assert blob["dynamics_gate_passed"] is None
        assert blob["dynamics_checkpoint_sha256"] is None
        assert blob["dynamics_gate_overridden"] is True   # cfg.rl.train.skip_gate=True

        # Snapshots present
        assert blob["ppo_hparams"]["lr"] == pytest.approx(3e-4)
        assert blob["reward_hparams"]["success_bonus"] == pytest.approx(5.0)
        assert blob["dynamics_arch"]["use_state_linear_skip"] is True
        assert blob["vae_n_latent"] == 32

    def test_extras_override_auto_fields(self, tmp_path: Path) -> None:
        from src.rl.train_ppo import _write_run_metadata

        cfg = _minimal_meta_cfg(tmp_path)
        _write_run_metadata(
            cfg, tmp_path / "out2", extras={"policy_path": "random_policy://uniform_valid"},
        )
        blob = json.loads((tmp_path / "out2" / "metadata.json").read_text())
        assert blob["policy_path"] == "random_policy://uniform_valid"


# =============================================================================
# scripts/summarize_rl_run — pure-Python tests on a synthetic parquet
# =============================================================================


def _write_synthetic_rollouts(out_dir: Path, *, n_episodes: int, success_per_ep: list[bool]) -> None:
    """Create a Contract-4 parquet + matching action_freq.json under ``out_dir``."""
    import polars as pl

    assert len(success_per_ep) == n_episodes
    rows: list[dict[str, Any]] = []
    action_freq: dict[str, int] = {"NO_OP": 0, "GENE_A": 0, "GENE_B": 0}
    noop_idx = 2
    for ep in range(n_episodes):
        is_success = success_per_ep[ep]
        # 2 steps per episode: first GENE_A, then NO_OP (success conditional on ep parity here)
        rows.append({
            "episode_id": int(ep), "step": 1, "action": 0,
            "gene_symbol": "GENE_A",
            "z_norm": 5.0, "reward": -5.05, "terminated": False, "success": False,
            "z_vector": [0.0] * 32,
        })
        rows.append({
            "episode_id": int(ep), "step": 2, "action": noop_idx,
            "gene_symbol": "NO_OP",
            "z_norm": 2.0 if is_success else 6.0,
            "reward": (5.0 if is_success else -5.0),
            "terminated": True, "success": bool(is_success),
            "z_vector": [0.0] * 32,
        })
        action_freq["GENE_A"] += 1
        action_freq["NO_OP"] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(str(out_dir / "rollouts.parquet"))
    (out_dir / "action_freq.json").write_text(json.dumps(action_freq))


class TestSummarizeRlRun:
    def test_summary_metrics_basic(self, tmp_path: Path) -> None:
        # Insert the scripts/ folder onto sys.path so we can import the module directly.
        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        import summarize_rl_run as srl

        run_dir = tmp_path / "run"
        _write_synthetic_rollouts(
            run_dir,
            n_episodes=10,
            success_per_ep=[True] * 6 + [False] * 4,  # 60% success
        )

        s = srl._summarize_rollouts(
            run_dir / "rollouts.parquet", run_dir / "action_freq.json",
        )
        assert s["n_episodes"] == 10
        assert s["successes"] == 6
        assert s["failures"] == 4
        assert s["success_rate"] == pytest.approx(0.6)
        # Each episode is 2 steps
        assert s["mean_steps"] == pytest.approx(2.0)
        # First action is always GENE_A, so NO-OP first-action count == 0
        assert s["noop_first_action_count"] == 0
        # action_freq has 2 distinct actions used equally → entropy = log(2)
        assert s["action_entropy_nats"] == pytest.approx(np.log(2.0), rel=1e-6)
        assert s["action_entropy_normalized"] == pytest.approx(1.0, rel=1e-6)
        # Top actions sorted by count
        assert {a["gene_symbol"] for a in s["top_actions"][:2]} == {"GENE_A", "NO_OP"}

    def test_delta_vs_random_block(self, tmp_path: Path) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        import summarize_rl_run as srl

        run_dir = tmp_path / "ppo"
        rand_dir = tmp_path / "rand"
        _write_synthetic_rollouts(run_dir, n_episodes=10, success_per_ep=[True] * 9 + [False])
        _write_synthetic_rollouts(rand_dir, n_episodes=10, success_per_ep=[True] * 8 + [False] * 2)

        rc = srl.main([
            "--run-dir", str(run_dir),
            "--random-baseline-dir", str(rand_dir),
            "--no-markdown",
        ])
        assert rc == 0

        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["metrics"]["success_rate"] == pytest.approx(0.9)
        assert summary["random_baseline"]["metrics"]["success_rate"] == pytest.approx(0.8)
        assert summary["delta_vs_random"]["delta_success_rate"] == pytest.approx(0.1)
        assert summary["delta_vs_random"]["delta_success_pp"] == pytest.approx(10.0)
