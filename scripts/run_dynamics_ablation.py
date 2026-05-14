"""scripts/run_dynamics_ablation.py — controlled architecture ablation for the dynamics model.

Owner: Agent B. See AGENTS.md §2 Phase 2 + PHASES.md Phase 2.

Runs four setups of ``PerturbationDynamicsModel`` back-to-back, each into its own
``artifacts/dynamics_ablation/<name>/`` directory, then aggregates the resulting
``gate.json`` / ``val_metrics.json`` / ``ood_metrics.json`` / ``gate_diagnostics.json``
artifacts into a single ``summary.json`` + ``summary.csv`` + a conservative recommendation.

The four setups (all other dynamics hyperparameters at default):

    baseline               use_state_linear_skip=False  use_gene_delta_bias=False
    state_linear           use_state_linear_skip=True   use_gene_delta_bias=False
    gene_bias              use_state_linear_skip=False  use_gene_delta_bias=True
    state_linear_gene_bias use_state_linear_skip=True   use_gene_delta_bias=True

The recommendation logic is **conservative**: a non-baseline setup is only accepted when
it (a) passes the gate or strictly improves the failed ridge margin, (b) does not collapse
OOD R² or Pearson relative to baseline, and (c) holds uncertainty Spearman ≥ 0.20. If no
setup is accepted, the recommendation falls back to ``keep_baseline`` and the script logs
that RL must remain blocked (per PHASES.md Phase 2 fallback rules).

This script does NOT mutate ``config/dynamics.yaml`` and does NOT start RL.

Usage
-----
::

    python scripts/run_dynamics_ablation.py             # full four-way run
    python scripts/run_dynamics_ablation.py --dry-run   # print planned commands, no exec
    python scripts/run_dynamics_ablation.py --smoke     # tiny epoch cap; wiring smoke
    python scripts/run_dynamics_ablation.py --only state_linear   # just one setup
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DYNAMICS = REPO_ROOT / "scripts" / "train_dynamics.py"

SMOKE_EPOCHS = 3
SMOKE_PATIENCE = 2


@dataclass(frozen=True)
class Setup:
    """One ablation cell: a name + the two architecture flags."""
    name: str
    use_state_linear_skip: bool
    use_gene_delta_bias: bool


SETUPS: list[Setup] = [
    Setup("baseline",               False, False),
    Setup("state_linear",           True,  False),
    Setup("gene_bias",              False, True),
    Setup("state_linear_gene_bias", True,  True),
]


# ---------------------------------------------------------------------------
# Path resolution via Hydra
# ---------------------------------------------------------------------------


def _resolve_paths() -> dict[str, str]:
    """Compose the default Hydra config and return the resolved ``paths`` block as a dict."""
    config_dir = REPO_ROOT / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="default",
            overrides=[f"paths.root={REPO_ROOT}"],
        )
    paths = OmegaConf.to_container(cfg.paths, resolve=True)
    assert isinstance(paths, dict)
    return paths  # str -> str


# ---------------------------------------------------------------------------
# Subprocess driver
# ---------------------------------------------------------------------------


def _build_command(
    setup: Setup,
    *,
    ablation_dir: Path,
    smoke: bool,
) -> list[str]:
    """Return the python+Hydra CLI for ``train_dynamics.py`` for one setup."""
    cmd = [
        sys.executable,
        str(TRAIN_DYNAMICS),
        f"paths.dynamics_dir={ablation_dir / setup.name}",
        f"dynamics.use_state_linear_skip={str(setup.use_state_linear_skip).lower()}",
        f"dynamics.use_gene_delta_bias={str(setup.use_gene_delta_bias).lower()}",
        "+force=true",
    ]
    if smoke:
        cmd += [
            f"dynamics.max_epochs={SMOKE_EPOCHS}",
            f"dynamics.early_stop_patience={SMOKE_PATIENCE}",
        ]
    return cmd


def _run_one(
    setup: Setup,
    *,
    ablation_dir: Path,
    smoke: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Invoke ``train_dynamics.py`` for one setup. Continue-on-error: never raises."""
    cmd = _build_command(setup, ablation_dir=ablation_dir, smoke=smoke)
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    log.info("[%s] %s", setup.name, cmd_str)

    if dry_run:
        return {
            "name": setup.name,
            "exit_code": 0,
            "duration_s": 0.0,
            "command": cmd_str,
            "dry_run": True,
            "error_tail": "",
        }

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    duration = time.time() - t0

    # Print stdout/stderr live-ish to the engineer so failures are visible.
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    error_tail = ""
    if proc.returncode != 0:
        tail_src = proc.stderr or proc.stdout or ""
        error_tail = "\n".join(tail_src.splitlines()[-20:])

    return {
        "name": setup.name,
        "exit_code": proc.returncode,
        "duration_s": duration,
        "command": cmd_str,
        "dry_run": False,
        "error_tail": error_tail,
    }


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def _safe_load_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _row_from_run(
    setup: Setup,
    run_meta: dict[str, Any],
    setup_dir: Path,
) -> dict[str, Any]:
    """Build a single summary row from a setup's artifact dir."""
    gate    = _safe_load_json(setup_dir / "gate.json")    or {}
    val     = _safe_load_json(setup_dir / "val_metrics.json") or {}
    ood     = _safe_load_json(setup_dir / "ood_metrics.json") or {}
    diag    = _safe_load_json(setup_dir / "gate_diagnostics.json") or {}
    config  = _safe_load_json(setup_dir / "config.json")  or {}

    def _g(d: dict | None, *keys: str, default: Any = None) -> Any:
        cur: Any = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    val_overall = _g(diag, "overall", "val") or {}
    ood_overall = _g(diag, "overall", "ood") or {}

    row: dict[str, Any] = {
        "name":                    setup.name,
        "use_state_linear_skip":   setup.use_state_linear_skip,
        "use_gene_delta_bias":     setup.use_gene_delta_bias,
        "exit_code":               run_meta.get("exit_code"),
        "duration_s":              round(float(run_meta.get("duration_s", 0.0)), 1),
        "error_tail":              run_meta.get("error_tail", ""),
        "dry_run":                 bool(run_meta.get("dry_run", False)),

        # gate
        "passed":                  bool(_g(gate, "passed", default=False)),

        # val (from diagnostics — single source of ridge truth)
        "val_mlp_r2":              _g(val_overall, "mlp_r2"),
        "val_mlp_pearson":         _g(val_overall, "mlp_pearson"),
        "val_ridge_r2":            _g(val_overall, "ridge_r2"),
        "val_ridge_pearson":       _g(val_overall, "ridge_pearson"),
        "val_mlp_minus_ridge_pearson": _g(val_overall, "mlp_minus_ridge_pearson"),

        # uncertainty
        "uncertainty_spearman":    _g(gate, "uncertainty_calibration", "spearman"),
        "uncertainty_pass":        _g(gate, "uncertainty_calibration", "pass"),

        # ood (from diagnostics)
        "ood_mlp_r2":              _g(ood_overall, "mlp_r2"),
        "ood_mlp_pearson":         _g(ood_overall, "mlp_pearson"),
        "ood_ridge_r2":            _g(ood_overall, "ridge_r2"),
        "ood_ridge_pearson":       _g(ood_overall, "ridge_pearson"),
        "ood_mlp_minus_ridge_pearson": _g(ood_overall, "mlp_minus_ridge_pearson"),

        # training state
        "best_epoch":              _g(config, "best_epoch"),
        "best_val_nll":            _g(config, "best_val_nll"),
        "epochs_run":              _g(config, "epochs_run"),
    }

    artifacts_present = all(
        (setup_dir / fname).exists()
        for fname in ("gate.json", "val_metrics.json", "gate_diagnostics.json")
    )
    row["status"] = "complete" if artifacts_present else "incomplete"
    return row


# ---------------------------------------------------------------------------
# Conservative recommendation
# ---------------------------------------------------------------------------


def recommend(
    rows: list[dict[str, Any]],
    *,
    ood_collapse_tolerance: float = 0.02,
    uncertainty_floor: float = 0.20,
) -> dict[str, Any]:
    """Pick the smallest architectural change that wins without breaking OOD.

    Acceptance for a non-baseline setup:
      1. ``passed`` is True, OR ``val_mlp_minus_ridge_pearson`` strictly improved
         over baseline (i.e. less negative / more positive).
      2. ``ood_mlp_r2 >= baseline.ood_mlp_r2 - ood_collapse_tolerance`` (no R² collapse).
      3. ``ood_mlp_pearson >= baseline.ood_mlp_pearson - ood_collapse_tolerance``.
      4. ``uncertainty_spearman >= uncertainty_floor`` (Phase 2 floor; sacred).
      5. Memorization guard: val improvement <= 2x OOD improvement. Penalises a setup
         that gains a lot on val while OOD does not move — the gene_bias-overfit
         signature.

    Among accepted setups, prefer (in order):
      - a setup whose ``passed`` is True,
      - the largest improvement in ``val_mlp_minus_ridge_pearson`` over baseline,
      - tie-break by largest ``ood_mlp_pearson``,
      - tie-break by simpler architecture (state_linear < gene_bias < both).

    If no setup is accepted, returns ``setup="keep_baseline"`` with a rationale string
    explaining the fallback. The caller logs this; do NOT auto-mutate
    ``config/dynamics.yaml``.

    Parameters
    ----------
    rows
        List of row dicts from :func:`_row_from_run`.
    ood_collapse_tolerance, uncertainty_floor
        See above.

    Returns
    -------
    dict
        ``{"setup": str, "passed": bool, "rationale": str, "fallback_invoked": bool}``.
    """
    by_name = {r["name"]: r for r in rows}
    baseline = by_name.get("baseline")
    if baseline is None or baseline.get("status") != "complete":
        return {
            "setup": "keep_baseline",
            "passed": False,
            "rationale": "baseline run did not complete; cannot evaluate alternatives",
            "fallback_invoked": True,
        }

    def _num(v: Any, default: float = float("nan")) -> float:
        return float(v) if isinstance(v, (int, float)) else default

    b_val_margin = _num(baseline.get("val_mlp_minus_ridge_pearson"))
    b_ood_r2     = _num(baseline.get("ood_mlp_r2"), default=0.0)
    b_ood_p      = _num(baseline.get("ood_mlp_pearson"), default=0.0)

    simplicity_rank = {
        "baseline": 0, "state_linear": 1, "gene_bias": 2, "state_linear_gene_bias": 3,
    }

    accepted: list[dict[str, Any]] = []
    rejection_reasons: dict[str, str] = {}
    for r in rows:
        if r["name"] == "baseline":
            continue
        if r.get("status") != "complete":
            rejection_reasons[r["name"]] = "run incomplete"
            continue

        val_margin = _num(r.get("val_mlp_minus_ridge_pearson"))
        ood_r2     = _num(r.get("ood_mlp_r2"), default=0.0)
        ood_p      = _num(r.get("ood_mlp_pearson"), default=0.0)
        unc        = _num(r.get("uncertainty_spearman"), default=0.0)
        passed     = bool(r.get("passed"))

        # Criterion 1
        improves_margin = val_margin > b_val_margin
        if not (passed or improves_margin):
            rejection_reasons[r["name"]] = (
                f"did not pass gate and did not improve val MLP-ridge margin "
                f"({val_margin:+.4f} vs baseline {b_val_margin:+.4f})"
            )
            continue
        # Criterion 2
        if ood_r2 < b_ood_r2 - ood_collapse_tolerance:
            rejection_reasons[r["name"]] = (
                f"OOD R² collapsed: {ood_r2:.4f} < baseline {b_ood_r2:.4f} − {ood_collapse_tolerance}"
            )
            continue
        # Criterion 3
        if ood_p < b_ood_p - ood_collapse_tolerance:
            rejection_reasons[r["name"]] = (
                f"OOD Pearson collapsed: {ood_p:.4f} < baseline {b_ood_p:.4f} − {ood_collapse_tolerance}"
            )
            continue
        # Criterion 4
        if unc < uncertainty_floor:
            rejection_reasons[r["name"]] = (
                f"uncertainty Spearman {unc:.4f} < floor {uncertainty_floor}"
            )
            continue
        # Criterion 5 — memorization guard
        val_gain = val_margin - b_val_margin
        ood_gain = (ood_p - b_ood_p)
        if val_gain > 0.02 and ood_gain <= 0 and val_gain > 2 * max(ood_gain, 0):
            rejection_reasons[r["name"]] = (
                f"memorization signature: val gain {val_gain:+.4f} >> OOD gain {ood_gain:+.4f}"
            )
            continue

        accepted.append(r)

    if not accepted:
        return {
            "setup": "keep_baseline",
            "passed": bool(baseline.get("passed")),
            "rationale": (
                "no setup met the conservative acceptance criteria; ridge remains the "
                "stronger held-out-cell baseline. RL stays blocked. Per-setup rejection "
                "reasons: " + json.dumps(rejection_reasons)
            ),
            "fallback_invoked": True,
        }

    # Prefer passed; then biggest val margin improvement; tie-break by OOD Pearson;
    # final tie-break by simplicity.
    def _key(r: dict[str, Any]) -> tuple:
        return (
            0 if bool(r.get("passed")) else 1,  # passed first
            -(_num(r.get("val_mlp_minus_ridge_pearson")) - b_val_margin),
            -_num(r.get("ood_mlp_pearson"), default=0.0),
            simplicity_rank.get(r["name"], 99),
        )

    accepted.sort(key=_key)
    pick = accepted[0]
    return {
        "setup": pick["name"],
        "passed": bool(pick.get("passed")),
        "rationale": (
            f"selected {pick['name']}: val MLP-ridge Pearson "
            f"{_num(pick.get('val_mlp_minus_ridge_pearson')):+.4f} vs baseline "
            f"{b_val_margin:+.4f}; OOD Pearson "
            f"{_num(pick.get('ood_mlp_pearson')):.4f} vs baseline {b_ood_p:.4f}; "
            f"passed={pick.get('passed')}"
        ),
        "fallback_invoked": False,
    }


# ---------------------------------------------------------------------------
# Summary I/O
# ---------------------------------------------------------------------------


_CSV_FIELDS = [
    "name",
    "use_state_linear_skip", "use_gene_delta_bias",
    "status", "exit_code", "passed",
    "val_mlp_r2", "val_mlp_pearson", "val_ridge_pearson",
    "val_mlp_minus_ridge_pearson",
    "uncertainty_spearman", "uncertainty_pass",
    "ood_mlp_r2", "ood_mlp_pearson", "ood_ridge_pearson",
    "ood_mlp_minus_ridge_pearson",
    "best_epoch", "best_val_nll", "epochs_run",
    "duration_s",
]


def _write_summary(rows: list[dict[str, Any]], rec: dict[str, Any], paths: dict[str, str]) -> None:
    summary_json = Path(paths["dynamics_ablation_summary_json"])
    summary_csv  = Path(paths["dynamics_ablation_summary_csv"])
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    summary_json.write_text(json.dumps(
        {"setups": rows, "recommendation": rec},
        indent=2,
    ))

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in _CSV_FIELDS})

    log.info("summary.json → %s", summary_json)
    log.info("summary.csv  → %s", summary_csv)


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'setup':<24} {'pass':<5} "
        f"{'val_mlp_P':>10} {'val_ridge_P':>12} {'val_diff':>9} "
        f"{'ood_mlp_P':>10} {'ood_ridge_P':>12} {'ood_diff':>9} "
        f"{'unc':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        def _f(k: str, w: int) -> str:
            v = r.get(k)
            return f"{v:>{w}.4f}" if isinstance(v, (int, float)) else f"{'n/a':>{w}}"
        print(
            f"{r['name']:<24} "
            f"{str(r.get('passed')):<5} "
            f"{_f('val_mlp_pearson', 10)} {_f('val_ridge_pearson', 12)} "
            f"{_f('val_mlp_minus_ridge_pearson', 9)} "
            f"{_f('ood_mlp_pearson', 10)} {_f('ood_ridge_pearson', 12)} "
            f"{_f('ood_mlp_minus_ridge_pearson', 9)} "
            f"{_f('uncertainty_spearman', 6)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dynamics architecture ablation runner")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned commands and the resolved output directory; do not invoke training.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"Tiny-epoch run for wiring smoke tests "
             f"(max_epochs={SMOKE_EPOCHS}, patience={SMOKE_PATIENCE}). "
             "Numbers are NOT scientifically meaningful.",
    )
    parser.add_argument(
        "--only", choices=[s.name for s in SETUPS], default=None,
        help="Run a single setup instead of all four.",
    )
    args = parser.parse_args(argv)

    _setup_logging()
    paths = _resolve_paths()
    ablation_dir = Path(paths["dynamics_ablation_dir"])
    ablation_dir.mkdir(parents=True, exist_ok=True)

    setups = [s for s in SETUPS if (args.only is None or s.name == args.only)]
    log.info(
        "Running %d setup(s) into %s | dry_run=%s | smoke=%s",
        len(setups), ablation_dir, args.dry_run, args.smoke,
    )

    rows: list[dict[str, Any]] = []
    for s in setups:
        run_meta = _run_one(
            s, ablation_dir=ablation_dir, smoke=args.smoke, dry_run=args.dry_run,
        )
        if args.dry_run:
            # No artifacts; record a placeholder row with the planned flags.
            rows.append({
                "name": s.name,
                "use_state_linear_skip": s.use_state_linear_skip,
                "use_gene_delta_bias":   s.use_gene_delta_bias,
                "status": "dry_run",
                "exit_code": 0,
                "passed": False,
                "duration_s": 0.0,
                "error_tail": "",
                "command": run_meta["command"],
            })
            continue
        rows.append(_row_from_run(s, run_meta, ablation_dir / s.name))

    if args.dry_run:
        print("\nPlanned commands:")
        for r in rows:
            print(f"  [{r['name']}] {r['command']}")
        return 0

    rec = recommend(rows)
    if args.smoke:
        rec["rationale"] = "[smoke run — numbers not authoritative] " + rec["rationale"]
    _write_summary(rows, rec, paths)
    print()
    _print_table(rows)
    print()
    print(f"RECOMMENDATION: {rec['setup']}  (passed={rec['passed']}, "
          f"fallback_invoked={rec['fallback_invoked']})")
    print(f"  {rec['rationale']}")

    if rec["fallback_invoked"]:
        log.warning(
            "Conservative fallback invoked: no setup accepted. "
            "config/dynamics.yaml is NOT modified; RL remains blocked per PHASES.md."
        )
    # Always exit 0 — failed gates inside individual runs are EXPECTED ablation data.
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
