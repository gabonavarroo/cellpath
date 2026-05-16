"""scripts/aggregate_eval.py — produce artifacts/eval/{summary,results_table,caveats}.

Hydra wrapper around :mod:`src.analysis.aggregate`. Does NOT compute any new metrics; it
discovers the JSON artifacts already produced by ``summarize_rl_run``, ``diagnose_dynamics_contraction``,
and ``train_dynamics``, composes a single ``summary.json``, and renders the Markdown report
+ caveats document used for the thesis defense.

Inputs (all overridable; defaults match the canonical V1 layout):

::

    +aggregate.ppo_det_dir=<path>           # PPO deterministic eval folder (Contract-4)
    +aggregate.ppo_stoch_dir=<path>         # PPO stochastic eval folder (optional)
    +aggregate.random_dir=<path>            # matched random-policy baseline folder
    +aggregate.contraction_dirs.primary_32d=<path>
    +aggregate.contraction_dirs.primary_32d_auto=<path>
    +aggregate.contraction_dirs.ablation_64d=<path>
    +aggregate.contraction_dirs.ablation_64d_auto=<path>
    +aggregate.contraction_dirs.ablation_64d_plain=<path>
    +aggregate.dynamics_gate.primary_32d=<path-to-gate.json>
    +aggregate.dynamics_gate.ablation_64d=<path-to-gate.json>
    +aggregate.out_dir=<path>               # defaults to ${paths.eval_dir}
    +aggregate.top_k=15                     # rank cutoff for the action-frequency table

Usage
-----
::

    # Default V1 layout
    python scripts/aggregate_eval.py --config-name default rl.train.skip_gate=true

    # Override a specific run
    python scripts/aggregate_eval.py --config-name default rl.train.skip_gate=true \\
        +aggregate.ppo_det_dir=artifacts/rl_sweeps/<run>/eval_deterministic
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


log = logging.getLogger(__name__)


_DEFAULT_LAYOUT = {
    "ppo_det_dir": "artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic",
    "ppo_stoch_dir": "artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_stochastic",
    "random_dir": "artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/random_baseline",
    "contraction_dirs": {
        "primary_32d": "artifacts/contraction",
        "primary_32d_auto": "artifacts/contraction_auto",
        "ablation_64d": "artifacts_64/contraction",
        "ablation_64d_auto": "artifacts_64/contraction_auto",
        "ablation_64d_plain": "artifacts_64/contraction_baseline_plain",
    },
    "dynamics_gate": {
        "primary_32d": "artifacts/dynamics/gate.json",
        "ablation_64d": "artifacts_64/dynamics/gate.json",
    },
    "top_k": 15,
}


def _resolve_settings(cfg: DictConfig) -> dict[str, Any]:
    """Merge ``cfg.get("aggregate", {})`` over the defaults; return a plain dict."""
    blob = cfg.get("aggregate", None)
    user = OmegaConf.to_container(blob, resolve=True) if blob is not None else {}

    def _merge(default: Any, override: Any) -> Any:
        if isinstance(default, dict) and isinstance(override, dict):
            out = {k: _merge(default.get(k), override.get(k, default.get(k))) for k in default}
            # Carry through any new override-only keys (e.g. extra contraction labels).
            for k, v in override.items():
                if k not in out:
                    out[k] = v
            return out
        return override if override is not None else default

    resolved = _merge(_DEFAULT_LAYOUT, user)
    resolved["out_dir"] = user.get("out_dir") or str(cfg.paths.eval_dir)
    return resolved


def _resolve_path(p: str | None, repo_root: Path) -> Path | None:
    """Resolve a possibly-relative path against the repo root; return None if blank."""
    if not p:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (repo_root / pp)


def run_aggregate(cfg: DictConfig) -> int:
    """Body of the entry point; returns POSIX exit code."""
    from src.analysis import aggregate as agg

    settings = _resolve_settings(cfg)
    repo_root = Path(cfg.paths.root) if hasattr(cfg.paths, "root") else Path.cwd()

    out_dir = _resolve_path(settings["out_dir"], repo_root)
    if out_dir is None:
        out_dir = repo_root / "artifacts" / "eval"

    if cfg.get("dry_run", False):
        print("DRY RUN — would write composite summary to:")
        print(f"  out_dir = {out_dir}")
        print(f"  ppo_det_dir = {settings['ppo_det_dir']}")
        print(f"  ppo_stoch_dir = {settings['ppo_stoch_dir']}")
        print(f"  random_dir = {settings['random_dir']}")
        print(f"  contraction_dirs = {settings['contraction_dirs']}")
        print(f"  dynamics_gate = {settings['dynamics_gate']}")
        print(f"  top_k = {settings['top_k']}")
        return 0

    # ---- Load everything (tolerant) ---------------------------------------------------
    ppo_det_dir = _resolve_path(settings["ppo_det_dir"], repo_root)
    ppo_stoch_dir = _resolve_path(settings["ppo_stoch_dir"], repo_root)
    random_dir = _resolve_path(settings["random_dir"], repo_root)

    ppo_det_summary = agg.load_rl_run_summary(ppo_det_dir)
    ppo_stoch_summary = agg.load_rl_run_summary(ppo_stoch_dir)
    random_summary = agg.load_rl_run_summary(random_dir)
    ppo_action_freq = agg.load_rl_action_freq(ppo_det_dir)
    random_action_freq = agg.load_rl_action_freq(random_dir)

    if ppo_det_summary is None:
        log.error("PPO deterministic summary not found at %s — aborting.", ppo_det_dir)
        return 2
    if ppo_stoch_summary is None:
        log.warning("PPO stochastic summary not found at %s — proceeding without it.",
                    ppo_stoch_dir)
    if random_summary is None:
        log.warning("Random-policy summary not found at %s — proceeding without it.",
                    random_dir)

    contraction_rows: dict[str, Any] = {}
    for label, p in (settings["contraction_dirs"] or {}).items():
        path = _resolve_path(p, repo_root)
        loaded = agg.load_contraction_summary(path) if path is not None else None
        contraction_rows[label] = loaded
        if loaded is None and path is not None:
            log.warning("Contraction summary missing for label=%s at %s", label, path)

    gate_32d = agg.load_dynamics_gate(
        _resolve_path((settings["dynamics_gate"] or {}).get("primary_32d"), repo_root)
    )
    gate_64d = agg.load_dynamics_gate(
        _resolve_path((settings["dynamics_gate"] or {}).get("ablation_64d"), repo_root)
    )
    if gate_32d is None:
        log.warning("32D gate.json not found — provenance fields will be partial.")
    if gate_64d is None:
        log.warning("64D gate.json not found — ablation section will be omitted.")

    # ---- Compose summary --------------------------------------------------------------
    summary = agg.build_summary(
        ppo_det_summary=ppo_det_summary,
        ppo_stoch_summary=ppo_stoch_summary,
        random_summary=random_summary,
        ppo_action_freq=ppo_action_freq,
        random_action_freq=random_action_freq,
        contraction_rows=contraction_rows,
        gate_32d=gate_32d,
        gate_64d=gate_64d,
        top_k=int(settings.get("top_k") or 15),
    )

    # ---- Write outputs ----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    results_path = out_dir / "results_table.md"
    caveats_path = out_dir / "caveats.md"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    results_path.write_text(agg.build_results_table_md(summary))
    caveats_path.write_text(agg.build_caveats_md(summary))

    log.info("Wrote %s", summary_path)
    log.info("Wrote %s", results_path)
    log.info("Wrote %s", caveats_path)

    # Console headline so the user sees the key Δ value immediately.
    delta = (summary.get("rl") or {}).get("delta_ppo_det_vs_random") or {}
    if delta:
        print(
            f"PPO det − random = {delta.get('delta_success_pp')} pp success, "
            f"Δsteps={delta.get('delta_mean_steps')}, "
            f"Δfinal_d={delta.get('delta_mean_final_distance')}."
        )

    return 0


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra entry point."""
    from src.utils.device import device_summary
    from src.utils.seeding import set_seed

    set_seed(int(cfg.get("seed", 42)))
    print(device_summary())
    return run_aggregate(cfg)


if __name__ == "__main__":
    sys.exit(main() or 0)
