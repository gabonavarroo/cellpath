"""End-to-end pipeline orchestrator.

Owner: shared (Agents A + B coordinate on changes — this is a conflict zone).

CLI
---
::

    python -m src.pipeline run --config-name default
    python -m src.pipeline run --config-name default --dry-run
    python -m src.pipeline run --config-name default --skip data --skip vae \\
        --skip pairs --skip dynamics --skip rl     # evaluate-only on existing artifacts
    python -m src.pipeline run --config-name default --from evaluate
    make pipeline                                   # convenience wrapper

Workflow (each step is idempotent given existing artifacts; ``--force <step>`` re-runs)::

    data       → preprocesses Norman to data/processed/norman_hvg.h5ad
    vae        → trains scVI; writes artifacts/vae/*
    pairs      → builds OT pseudo-pairs; writes artifacts/pairs/*
    dynamics   → trains MLP; writes artifacts/dynamics/* + gate.json
    rl         → trains MaskablePPO (refuses if gate.json.passed=False unless skip_gate)
    evaluate   → aggregator + DepMap enrichment + latent quality + 5 figures

Each ``step_*`` function reads ``cfg.paths.*`` to check whether its primary output already
exists (CLAUDE.md rule #1) and skips silently when present and ``force`` is False. Step
bodies dispatch to the standalone script entry points via ``hydra.main``-compatible
subprocesses, so each step gets a fresh Hydra context and identical CLI semantics.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)

log = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FORCE_OPTION = typer.Option(None, "--force", help="Step name(s) to force re-run")
_SKIP_OPTION = typer.Option(None, "--skip", help="Step name(s) to skip")


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def _run_script(script: str, cfg_overrides: list[str]) -> int:
    """Invoke a standalone Hydra script with the supplied CLI overrides.

    Each step gets its own subprocess so Hydra can re-initialize cleanly. The repo root
    is added to ``PYTHONPATH`` so the script's ``from src...`` imports resolve.
    """
    script_path = _REPO_ROOT / "scripts" / script
    if not script_path.exists():
        log.error("Script %s not found.", script_path)
        return 2
    env = os.environ.copy()
    env.setdefault("CELLPATH_ROOT", str(_REPO_ROOT))
    # Make `from src.*` importable inside the subprocess
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(script_path), *cfg_overrides]
    log.info("[pipeline] $ %s", " ".join(cmd))
    return subprocess.run(cmd, env=env, check=False).returncode


def _artifact_exists(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and (p.is_file() or any(p.iterdir()))


def step_data(cfg: Any, force: bool = False) -> Path:
    """Preprocess Norman → ``data/processed/norman_hvg.h5ad``.

    Only runs when missing. Delegates to ``scripts/train_vae.py`` is not appropriate
    because preprocessing happens inside that script when its input is missing; for an
    explicit ``data`` step we re-use ``scripts/train_vae.py`` with ``+dry_run=true`` is
    not appropriate either — the canonical entry point for the data step is the
    ``make data`` shell script. We invoke it directly.
    """
    target = Path(cfg.paths.norman_processed_h5ad)
    if not force and _artifact_exists(target):
        log.info("[pipeline] skip data (exists: %s)", target)
        return target
    script = _REPO_ROOT / "scripts" / "download_data.sh"
    if not script.exists():
        log.warning("[pipeline] %s missing; skipping data step.", script)
        return target
    rc = subprocess.run(["bash", str(script)], cwd=str(_REPO_ROOT), check=False).returncode
    if rc != 0:
        raise RuntimeError(f"download_data.sh exited {rc}")
    return target


def _hydra_overrides_from_cfg(cfg: Any) -> list[str]:
    """Pass-through the config_name through ``--config-name`` plus paths.root."""
    try:
        config_name = str(cfg.get("_config_name", "default"))
    except AttributeError:
        config_name = "default"
    return ["--config-name", config_name, f"paths.root={_REPO_ROOT!s}"]


def step_vae(cfg: Any, force: bool = False) -> Path:
    """Train scVI VAE → ``artifacts/vae/``."""
    target = Path(cfg.paths.vae_latents_h5ad)
    if not force and _artifact_exists(target):
        log.info("[pipeline] skip vae (exists: %s)", target)
        return target
    rc = _run_script("train_vae.py", _hydra_overrides_from_cfg(cfg))
    if rc != 0:
        raise RuntimeError(f"train_vae.py exited {rc}")
    return target


def step_pairs(cfg: Any, force: bool = False) -> Path:
    """Build OT pseudo-pairs → ``artifacts/pairs/``."""
    target = Path(cfg.paths.pairs_train)
    if not force and _artifact_exists(target):
        log.info("[pipeline] skip pairs (exists: %s)", target)
        return target
    rc = _run_script("build_pairs.py", _hydra_overrides_from_cfg(cfg))
    if rc != 0:
        raise RuntimeError(f"build_pairs.py exited {rc}")
    return target


def step_dynamics(cfg: Any, force: bool = False) -> Path:
    """Train dynamics → ``artifacts/dynamics/{model.pt, gate.json}``."""
    target = Path(cfg.paths.dynamics_model)
    if not force and _artifact_exists(target):
        log.info("[pipeline] skip dynamics (exists: %s)", target)
        return target
    rc = _run_script("train_dynamics.py", _hydra_overrides_from_cfg(cfg))
    if rc != 0:
        raise RuntimeError(f"train_dynamics.py exited {rc}")
    return target


def step_rl(cfg: Any, force: bool = False) -> Path:
    """Train MaskablePPO → ``artifacts/rl/{ppo.zip, rollouts.parquet, action_freq.json}``.

    The script itself enforces the dynamics-gate refusal (honoring ``rl.train.skip_gate``).
    """
    target = Path(cfg.paths.rl_ppo_zip)
    if not force and _artifact_exists(target):
        log.info("[pipeline] skip rl (exists: %s)", target)
        return target
    rc = _run_script("train_rl.py", _hydra_overrides_from_cfg(cfg))
    if rc != 0:
        raise RuntimeError(f"train_rl.py exited {rc}")
    return target


def step_evaluate(cfg: Any, force: bool = False) -> Path:
    """Run the full Phase-5 evaluation: aggregator + DepMap + latent quality + figures.

    Idempotent only at the directory level — re-runs always refresh the reports because
    the underlying RL/dynamics/contraction inputs may have changed without bumping a
    canonical filename. To skip entirely, pass ``--skip evaluate``.
    """
    eval_dir = Path(cfg.paths.eval_dir)
    rc = _run_script(
        "evaluate.py",
        [*_hydra_overrides_from_cfg(cfg), "rl.train.skip_gate=true"],
    )
    if rc != 0:
        raise RuntimeError(f"evaluate.py exited {rc}")
    rc = _run_script(
        "visualize.py",
        [*_hydra_overrides_from_cfg(cfg), "rl.train.skip_gate=true"],
    )
    if rc != 0:
        raise RuntimeError(f"visualize.py exited {rc}")
    return eval_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_STEPS = ["data", "vae", "pairs", "dynamics", "rl", "evaluate"]
_STEP_FNS = {
    "data": step_data,
    "vae": step_vae,
    "pairs": step_pairs,
    "dynamics": step_dynamics,
    "rl": step_rl,
    "evaluate": step_evaluate,
}


@app.callback()
def _callback() -> None:
    """CellPath — RL-based in-silico cell-state steering pipeline.

    Run ``python -m src.pipeline run --help`` for pipeline options.
    """


@app.command()
def run(
    config_name: str = typer.Option("default", "--config-name", help="Hydra config name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config, exit before compute"),
    force: list[str] | None = _FORCE_OPTION,
    skip: list[str] | None = _SKIP_OPTION,
    from_step: str = typer.Option("", "--from", help="Resume from this step (skip earlier steps)"),
) -> None:
    """Run the full pipeline.

    Steps in order: ``data → vae → pairs → dynamics → rl → evaluate``.

    Examples
    --------
    ::

        python -m src.pipeline run --config-name default
        python -m src.pipeline run --config-name default --dry-run
        python -m src.pipeline run --config-name default --force vae
        python -m src.pipeline run --config-name default --skip evaluate
        python -m src.pipeline run --config-name default --from evaluate
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_dir = _REPO_ROOT / "config"
    os.environ.setdefault("CELLPATH_ROOT", str(_REPO_ROOT))
    force = force or []
    skip = skip or []

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name=config_name,
            overrides=[f"paths.root={_REPO_ROOT!s}"],
        )
    with open_dict(cfg):
        cfg._config_name = config_name

    # Resolve which steps to run after honoring --from / --skip
    if from_step:
        if from_step not in _STEPS:
            print(f"ERROR: unknown --from step '{from_step}'. Valid: {_STEPS}",
                  file=sys.stderr)
            raise typer.Exit(code=2)
        start_idx = _STEPS.index(from_step)
    else:
        start_idx = 0

    plan = [s for s in _STEPS[start_idx:] if s not in skip]

    if dry_run:
        print(f"DRY RUN — config={config_name}, seed={cfg.seed}")
        print(
            f"  vae.n_latent={cfg.vae.n_latent}, "
            f"dynamics.n_hidden={cfg.dynamics.n_hidden}, "
            f"rl.ppo.total_timesteps={cfg.rl.ppo.total_timesteps}"
        )
        for step in _STEPS:
            label = (
                "SKIP " if step in skip else (
                    "FORCE" if step in force else (
                        "RUN  " if step in plan else "SKIP "
                    )
                )
            )
            print(f"  {label} {step}")
        print("DRY RUN — config validated OK. Exiting.")
        return

    from src.utils.device import device_summary
    from src.utils.seeding import set_seed

    set_seed(int(cfg.seed))
    print(device_summary())

    for step_name in plan:
        log.info("[pipeline] running %s...", step_name)
        try:
            _STEP_FNS[step_name](cfg, force=(step_name in force))
        except RuntimeError as exc:
            log.error("[pipeline] %s failed: %s", step_name, exc)
            raise typer.Exit(code=1) from exc

    print("\n[pipeline] all steps complete.")


def main() -> None:
    """Entry point for ``python -m src.pipeline``."""
    app()


if __name__ == "__main__":
    main()
