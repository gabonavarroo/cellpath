"""End-to-end pipeline orchestrator.

Owner: shared (Agents A + B coordinate on changes — this is a conflict zone).

CLI
---
::

    python -m src.pipeline run --config-name default
    python -m src.pipeline run --config-name default --dry-run
    make pipeline                                   # convenience wrapper

Workflow (each step is idempotent given existing artifacts; ``--force <step>`` re-runs)::

    data       → preprocesses Norman to data/processed/norman_hvg.h5ad
    vae        → trains scVI; writes artifacts/vae/*
    pairs      → builds OT pseudo-pairs; writes artifacts/pairs/*
    dynamics   → trains MLP; writes artifacts/dynamics/* + gate.json
    rl         → trains MaskablePPO (refuses if gate.json.passed=False)
    evaluate   → DepMap enrichment + trajectory rendering

The runner checks for existing artifacts before each step and skips if present (per
CLAUDE.md sacred rule #1). Use ``--force <step>`` or delete the artifacts dir to re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


# ---------------------------------------------------------------------------
# Step implementations (each calls the relevant component module)
# ---------------------------------------------------------------------------


def step_data(cfg: Any, force: bool = False) -> Path:
    """Step 1: download + preprocess Norman.

    Parameters
    ----------
    cfg
        Hydra config.
    force
        If True, re-run even if processed h5ad exists.

    Returns
    -------
    Path
        Path to processed h5ad.

    Raises
    ------
    NotImplementedError
        Shared: orchestrate :func:`src.data.download.download_norman` +
        :func:`src.data.preprocess.run_preprocessing`.
    """
    raise NotImplementedError(
        "Shared: implement step_data. Check cfg.paths.norman_processed_h5ad; if exists and "
        "not force, skip. Else call download + preprocess."
    )


def step_vae(cfg: Any, force: bool = False) -> Path:
    """Step 2: train scVI VAE."""
    raise NotImplementedError(
        "Shared: implement step_vae. Check artifacts/vae/model exists; skip if not force."
    )


def step_pairs(cfg: Any, force: bool = False) -> Path:
    """Step 3: build OT pseudo-pairs."""
    raise NotImplementedError(
        "Shared: implement step_pairs. Check artifacts/pairs/*.npz; skip if not force."
    )


def step_dynamics(cfg: Any, force: bool = False) -> Path:
    """Step 4: train dynamics model + run validation gate."""
    raise NotImplementedError(
        "Shared: implement step_dynamics. Train + gate. Write gate.json with passed=bool."
    )


def step_rl(cfg: Any, force: bool = False) -> Path:
    """Step 5: train MaskablePPO (refuses unless gate passed)."""
    raise NotImplementedError(
        "Shared: implement step_rl. Refuse to start unless gate.json.passed=True."
    )


def step_evaluate(cfg: Any, force: bool = False) -> Path:
    """Step 6: DepMap enrichment + trajectory rendering."""
    raise NotImplementedError(
        "Shared: implement step_evaluate. Aggregate all final metrics into artifacts/eval/."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def run(
    config_name: str = typer.Option("default", "--config-name", help="Hydra config name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config, exit before compute"),
    force: list[str] = typer.Option([], "--force", help="Step name(s) to force re-run"),
    skip: list[str] = typer.Option([], "--skip", help="Step name(s) to skip"),
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
    """
    # NOTE: implementation is shared. Compose Hydra config, set seed, log device, run steps in
    # order. For each step, skip if artifact exists and step not in `force`. Always honor
    # `--dry-run` which prints planned actions and exits.
    raise NotImplementedError(
        "Shared: implement Hydra compose + sequential step runner. "
        "Honor --dry-run (print plan and exit). Use src.utils.seeding.set_seed + device logging."
    )


def main() -> None:
    """Entry point for ``python -m src.pipeline``."""
    app()


if __name__ == "__main__":
    main()
