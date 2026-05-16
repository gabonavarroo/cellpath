"""Tests for ``src.analysis.aggregate`` — the V1 result aggregator.

The aggregator is pure-function so all tests build small synthetic JSON dicts that mimic
the on-disk artifacts. No file globbing or Hydra here; the Hydra wrapper is exercised by
the end-to-end pipeline smoke test.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.analysis import aggregate as agg


# =============================================================================
# Helpers — build synthetic JSON shaped like the real artifacts
# =============================================================================


def _ppo_summary(*, success_rate: float, mean_steps: float,
                 final_d: float, n_episodes: int = 500,
                 top_actions: list[tuple[str, int]] | None = None,
                 stage: str = "evaluate_rl",
                 with_random_baseline: bool = False) -> dict[str, Any]:
    """A summarize_rl_run-shaped dict — same keys this script emits in real life."""
    if top_actions is None:
        top_actions = [("CKS1B", 200), ("TSC22D1", 100), ("CELF2", 80)]
    metrics = {
        "n_episodes": n_episodes,
        "successes": int(success_rate * n_episodes),
        "failures": n_episodes - int(success_rate * n_episodes),
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_total_reward": -9.0,
        "mean_final_distance": final_d,
        "mean_min_distance": final_d,
        "noop_first_action_count": 6,
        "noop_first_action_failures": 6,
        "noop_first_action_rate": 0.012,
        "action_entropy_nats": 2.5,
        "action_entropy_normalized": 0.8,
        "top_actions": [{"gene_symbol": g, "count": c} for g, c in top_actions],
    }
    metadata = {
        "timestamp_utc": "2026-05-15T18:47:00+00:00",
        "git_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "seed": 42,
        "epsilon_value": 3.5310792922973633,
        "epsilon_source": "override(3.53108)",
        "epsilon_percentile_json": 50.0,
        "min_start_distance": 8.0,
        "max_steps": 10,
        "vae_n_latent": 32,
        "vae_dir": "/tmp/artifacts/vae",
        "dynamics_checkpoint": "/tmp/artifacts/dynamics/model.pt",
        "dynamics_checkpoint_sha256": "278028682ee5030f6d03393a78bb3234fe531e43763c0e1f519ad2f805b03510",
        "dynamics_gate_passed": False,
        "dynamics_gate_overridden": True,
        "dynamics_arch": {
            "use_state_linear_skip": True,
            "use_gene_delta_bias": False,
            "n_latent": 32, "n_hidden": 256, "n_layers": 3, "d_emb": 64,
        },
        "ppo_hparams": {"lr": 3e-4, "ent_coef": 0.01, "total_timesteps": 2_000_000},
        "reward_hparams": {"lambda_sparse": 0.05, "success_bonus": 0.0},
        "reward_mode": "absolute_distance",
        "deterministic_eval": True,
        "n_episodes": n_episodes,
        "policy_path": "/tmp/ppo.zip",
        "stage": stage,
    }
    out = {"run_dir": "/tmp/run", "metadata": metadata, "metrics": metrics}
    if with_random_baseline:
        out["random_baseline"] = {
            "run_dir": "/tmp/rand",
            "metadata": {**metadata, "stage": "random_policy",
                         "policy_path": "random_policy://uniform_valid"},
            "metrics": {**metrics, "success_rate": 0.84,
                        "successes": int(0.84 * n_episodes),
                        "failures": n_episodes - int(0.84 * n_episodes),
                        "mean_steps": 5.53, "mean_final_distance": 3.41},
        }
        out["delta_vs_random"] = {
            "delta_success_rate": success_rate - 0.84,
            "delta_success_pp": 100.0 * (success_rate - 0.84),
            "delta_mean_steps": mean_steps - 5.53,
            "delta_mean_final_distance": final_d - 3.41,
        }
    return out


def _contra(*, fraction_improved: float, mean_improvement: float,
            n_latent: int = 32, min_start: Any = 8.0,
            use_state_linear: bool = True) -> dict[str, Any]:
    """A contraction summary with the merged metadata fields the aggregator expects."""
    return {
        "n_starts": 100, "n_genes": 105, "n_pairs": 100 * 105,
        "fraction_improved": fraction_improved,
        "mean_improvement": mean_improvement,
        "median_improvement": mean_improvement - 0.05,
        "std_improvement": 0.7,
        "best_improvement": mean_improvement + 2.0,
        "worst_improvement": mean_improvement - 2.0,
        "p25_improvement": mean_improvement - 0.5,
        "p75_improvement": mean_improvement + 0.5,
        # Merged from metadata.json by load_contraction_summary
        "min_start_distance_used": min_start,
        "epsilon_used": 3.531,
        "epsilon_source": "json(p50.0)",
        "vae_n_latent": n_latent,
        "dynamics_use_state_linear_skip": use_state_linear,
        "dynamics_use_gene_delta_bias": False,
        "dynamics_checkpoint": "/tmp/dyn/model.pt",
        "dynamics_checkpoint_sha256": "abcd" * 16,
    }


def _gate(*, val_pearson: float, ridge_pearson: float, ood_pearson: float = 0.4,
          ood_ridge: float = 0.44, passed: bool = False) -> dict[str, Any]:
    """A gate.json-shaped dict with primary + OOD blocks."""
    margin_val = val_pearson - ridge_pearson
    margin_ood = ood_pearson - ood_ridge
    return {
        "passed": passed,
        "primary": {
            "r2": 0.4, "pearson_r": val_pearson,
            "baselines": {
                "no_op": {"r2": 0.0, "pearson_r": 0.0},
                "global_mean_delta": {"r2": 0.01, "pearson_r": 0.0},
                "per_gene_mean_delta": {"r2": 0.15, "pearson_r": 0.33},
                "linear_ridge": {"r2": 0.38, "pearson_r": ridge_pearson},
                "nearest_neighbor": {"r2": 0.08, "pearson_r": 0.36},
            },
            "margin_checks": {
                "margin_vs_linear_ridge_pearson": {
                    "value": margin_val, "threshold": 0.03, "pass": margin_val >= 0.03,
                },
            },
        },
        "ood": {
            "r2": 0.26, "pearson_r": ood_pearson,
            "baselines": {
                "linear_ridge": {"r2": 0.17, "pearson_r": ood_ridge},
                "nearest_neighbor": {"r2": 0.02, "pearson_r": 0.33},
            },
            "margin_checks": {
                "margin_vs_linear_ridge_pearson": {
                    "value": margin_ood, "threshold": 0.03, "pass": margin_ood >= 0.03,
                },
            },
        },
        "uncertainty_calibration": {"spearman": 0.25, "pass": True},
        "uncertainty_calibration_ood": {"spearman": 0.20, "pass": True},
        "margins_used": {"margin_vs_linear_ridge_pearson": 0.03},
    }


# =============================================================================
# load_rl_run_summary — flat vs wrapped normalization
# =============================================================================


class TestLoadRlRunSummary:
    def test_wrapped_form_passes_through(self, tmp_path: Any) -> None:
        import json as _json
        d = tmp_path / "wrapped"
        d.mkdir()
        wrapped = _ppo_summary(success_rate=0.9, mean_steps=2.0, final_d=3.0)
        (d / "summary.json").write_text(_json.dumps(wrapped))
        out = agg.load_rl_run_summary(d)
        assert out is not None
        assert "metrics" in out and out["metrics"]["success_rate"] == pytest.approx(0.9)

    def test_flat_form_is_upgraded(self, tmp_path: Any) -> None:
        """The random-policy script emits a flat summary; the loader must wrap it."""
        import json as _json
        d = tmp_path / "flat"
        d.mkdir()
        flat = {
            "policy_kind": "uniform_valid",
            "n_episodes": 500,
            "success_rate": 0.84,
            "successes": 420,
            "failures": 80,
            "mean_steps": 5.53,
            "mean_total_reward": -25.2,
            "mean_final_distance": 3.41,
            "mean_min_distance": 3.36,
            "noop_first_action_failures": 4,
            "noop_first_action_rate": 0.008,
        }
        (d / "summary.json").write_text(_json.dumps(flat))
        # Also write a minimal metadata.json so the loader includes provenance.
        (d / "metadata.json").write_text(_json.dumps({"stage": "random_policy"}))
        out = agg.load_rl_run_summary(d)
        assert out is not None
        # The wrapped form is the canonical schema
        assert "metrics" in out and "metadata" in out
        assert out["metrics"]["success_rate"] == pytest.approx(0.84)
        assert out["metrics"]["n_episodes"] == 500
        # Fields the flat form doesn't carry are filled with None, not crashed
        assert out["metrics"]["action_entropy_nats"] is None
        # Metadata round-trips when present
        assert out["metadata"]["stage"] == "random_policy"

    def test_missing_summary_returns_none(self, tmp_path: Any) -> None:
        assert agg.load_rl_run_summary(tmp_path / "does_not_exist") is None


# =============================================================================
# build_summary — composite shape and section presence
# =============================================================================


class TestBuildSummary:
    def test_full_inputs_compose_all_sections(self) -> None:
        ppo_det = _ppo_summary(success_rate=0.988, mean_steps=2.28, final_d=3.03,
                               with_random_baseline=True)
        ppo_stoch = _ppo_summary(success_rate=0.988, mean_steps=2.29, final_d=3.04)
        rand = _ppo_summary(success_rate=0.84, mean_steps=5.53, final_d=3.41,
                            stage="random_policy",
                            top_actions=[("PTPN13", 38), ("FOSB", 38)])

        summary = agg.build_summary(
            ppo_det_summary=ppo_det,
            ppo_stoch_summary=ppo_stoch,
            random_summary=rand,
            ppo_action_freq={"CKS1B": 200, "TSC22D1": 100, "CELF2": 80},
            random_action_freq={"PTPN13": 38, "FOSB": 38, "CKS1B": 5},
            contraction_rows={
                "primary_32d": _contra(fraction_improved=1.0, mean_improvement=2.74),
                "primary_32d_auto": _contra(fraction_improved=0.955, mean_improvement=1.0,
                                            min_start="auto"),
                "ablation_64d": _contra(fraction_improved=1.0, mean_improvement=3.28,
                                        n_latent=64),
                "ablation_64d_plain": _contra(fraction_improved=1.0, mean_improvement=3.10,
                                              n_latent=64, use_state_linear=False),
            },
            gate_32d=_gate(val_pearson=0.6085, ridge_pearson=0.6010),
            gate_64d=_gate(val_pearson=0.5965, ridge_pearson=0.6156, ood_pearson=0.37, ood_ridge=0.47),
        )

        # Top-level sections
        assert set(summary.keys()) >= {"provenance", "rl", "dynamics", "contraction", "top_actions"}

        # Provenance picked up the deterministic-run metadata
        prov = summary["provenance"]
        assert prov["dynamics_gate_passed"] is False
        assert prov["dynamics_gate_overridden"] is True
        assert prov["vae_n_latent"] == 32
        assert prov["epsilon_source"] == "override(3.53108)"

        # RL: all three runs present, with Δ block from delta_vs_random
        rl = summary["rl"]
        assert rl["ppo_deterministic"]["success_rate"] == pytest.approx(0.988)
        assert rl["ppo_stochastic"]["success_rate"] == pytest.approx(0.988)
        assert rl["random_baseline"]["success_rate"] == pytest.approx(0.84)
        assert rl["delta_ppo_det_vs_random"]["delta_success_pp"] == pytest.approx(14.8)

        # Dynamics: both branches plus Δ
        dyn = summary["dynamics"]
        assert dyn["primary_32d"]["primary"]["margin_vs_linear_ridge_pearson"] == pytest.approx(0.0075, rel=1e-3)
        assert dyn["primary_32d"]["primary"]["margin_vs_linear_ridge_pearson_pass"] is False
        assert dyn["ablation_64d"]["primary"]["margin_vs_linear_ridge_pearson_pass"] is False
        assert dyn["delta_32d_vs_64d"] is not None

        # Contraction: rows are labeled and rounded
        contra = summary["contraction"]
        assert contra["primary_32d"]["label"] == "primary_32d"
        assert contra["primary_32d"]["fraction_improved"] == pytest.approx(1.0)
        # 64D plain stripped state_linear — flag exposed
        assert contra["ablation_64d_plain"]["use_state_linear_skip"] is False

        # Top actions: PPO and random side-by-side + set ops
        top = summary["top_actions"]
        assert top["top_k"] == 15
        assert any(row["gene_symbol"] == "CKS1B" for row in top["ppo"])
        assert any(row["gene_symbol"] == "PTPN13" for row in top["random"])
        assert "CKS1B" in top["intersection"] or "CKS1B" in top["only_in_ppo"]

    def test_missing_64d_gate_gracefully_skipped(self) -> None:
        """No 64D gate → ablation_64d in dynamics is None and delta is None; no crash."""
        ppo = _ppo_summary(success_rate=0.9, mean_steps=2.0, final_d=3.0,
                           with_random_baseline=True)
        summary = agg.build_summary(
            ppo_det_summary=ppo,
            ppo_stoch_summary=None,
            random_summary=None,
            ppo_action_freq=None,
            random_action_freq=None,
            contraction_rows={
                "primary_32d": _contra(fraction_improved=1.0, mean_improvement=2.74),
            },
            gate_32d=_gate(val_pearson=0.6, ridge_pearson=0.59),
            gate_64d=None,
        )
        assert summary["dynamics"]["primary_32d"] is not None
        assert summary["dynamics"]["ablation_64d"] is None
        assert summary["dynamics"]["delta_32d_vs_64d"] is None
        # Missing 64D contraction labels are skipped, not present as None
        assert "ablation_64d" not in summary["contraction"]

    def test_missing_stochastic_falls_back_to_det_only(self) -> None:
        ppo_det = _ppo_summary(success_rate=0.95, mean_steps=2.0, final_d=3.0,
                               with_random_baseline=True)
        summary = agg.build_summary(
            ppo_det_summary=ppo_det,
            ppo_stoch_summary=None,
            random_summary=_ppo_summary(success_rate=0.8, mean_steps=5.0, final_d=3.5,
                                        stage="random_policy"),
            ppo_action_freq=None, random_action_freq=None,
            contraction_rows={},
            gate_32d=None, gate_64d=None,
        )
        assert summary["rl"]["ppo_deterministic"] is not None
        assert summary["rl"]["ppo_stochastic"] is None
        assert summary["rl"]["random_baseline"]["success_rate"] == pytest.approx(0.8)

    def test_delta_recomputed_when_not_in_det_summary(self) -> None:
        """If summarize_rl_run wasn't given --random-baseline-dir, we still compute the delta."""
        det = _ppo_summary(success_rate=0.9, mean_steps=2.0, final_d=3.0)  # no random_baseline
        rand = _ppo_summary(success_rate=0.7, mean_steps=5.0, final_d=3.5, stage="random_policy")
        summary = agg.build_summary(
            ppo_det_summary=det, ppo_stoch_summary=None, random_summary=rand,
            ppo_action_freq=None, random_action_freq=None,
            contraction_rows={}, gate_32d=None, gate_64d=None,
        )
        delta = summary["rl"]["delta_ppo_det_vs_random"]
        assert delta["delta_success_rate"] == pytest.approx(0.2, rel=1e-3)
        assert delta["delta_success_pp"] == pytest.approx(20.0, rel=1e-3)


# =============================================================================
# Markdown renderers
# =============================================================================


class TestMarkdownRenderers:
    def test_results_table_md_contains_override_caveat(self) -> None:
        ppo_det = _ppo_summary(success_rate=0.988, mean_steps=2.28, final_d=3.03,
                               with_random_baseline=True)
        summary = agg.build_summary(
            ppo_det_summary=ppo_det, ppo_stoch_summary=None,
            random_summary=_ppo_summary(success_rate=0.84, mean_steps=5.53, final_d=3.41,
                                        stage="random_policy"),
            ppo_action_freq={"CKS1B": 200, "TSC22D1": 100},
            random_action_freq={"PTPN13": 38, "FOSB": 38},
            contraction_rows={
                "primary_32d": _contra(fraction_improved=1.0, mean_improvement=2.74),
                "ablation_64d_plain": _contra(fraction_improved=1.0, mean_improvement=3.10,
                                              n_latent=64, use_state_linear=False),
            },
            gate_32d=_gate(val_pearson=0.6085, ridge_pearson=0.6010),
            gate_64d=_gate(val_pearson=0.5965, ridge_pearson=0.6156),
        )
        md = agg.build_results_table_md(summary)
        # The override caveat block must appear and be visible at the top.
        assert "gate failed and was overridden" in md.lower() or "skip_gate" in md.lower()
        # Headline numbers
        assert "0.988" in md
        # 14.8 pp Δ — printed by _render_rl_table
        assert "14.8" in md
        # Contraction rows
        assert "32D state_linear, start8" in md
        assert "64D baseline_plain, start8" in md
        # Top-actions section
        assert "Top-15 actions" in md or "Top-" in md

    def test_caveats_md_asserts_all_four_constraints(self) -> None:
        """caveats.md must mention each binding constraint by name."""
        ppo_det = _ppo_summary(success_rate=0.988, mean_steps=2.28, final_d=3.03)
        summary = agg.build_summary(
            ppo_det_summary=ppo_det, ppo_stoch_summary=None, random_summary=None,
            ppo_action_freq=None, random_action_freq=None,
            contraction_rows={
                "primary_32d": _contra(fraction_improved=1.0, mean_improvement=2.74),
                "ablation_64d_plain": _contra(fraction_improved=1.0, mean_improvement=3.10,
                                              n_latent=64, use_state_linear=False),
            },
            gate_32d=_gate(val_pearson=0.6085, ridge_pearson=0.6010),
            gate_64d=_gate(val_pearson=0.5965, ridge_pearson=0.6156),
        )
        text = agg.build_caveats_md(summary).lower()
        # 1. gate failed and overridden
        assert "gate" in text and "overridden" in text
        # 2. globally contractive
        assert "contractive" in text
        # 3. 64d rejected as primary
        assert "64d" in text and ("rejected" in text or "negative" in text or "ablation" in text)
        # 4. biological-discovery / therapeutic claim out of scope
        assert "therapeutic" in text or "biological discovery" in text or "out of scope" in text
        # Future work section present
        assert "future work" in text
        assert "ot-pair" in text or "ot pair" in text  # ranked direction #1
