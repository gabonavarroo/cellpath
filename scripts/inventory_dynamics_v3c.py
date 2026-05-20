"""V3C dynamics inventory — Phase 0A.

Walks the four artifact tiers (artifacts/, artifacts_64/, artifacts_v2/,
artifacts_v3/) to a max depth of 5, identifies directories containing both
model.pt and config.json as candidate dynamics fields, extracts metadata,
classifies audit eligibility, and writes
artifacts_v3/v3c/utility_audit/dynamics_inventory.csv.

Read-only on frozen tiers; writes only under artifacts_v3/v3c/. Idempotent:
re-running with unchanged artifacts yields the same CSV (field order, row
order, scalar values).

See V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md §2, §11 Phase 0A.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_TIERS = ("artifacts", "artifacts_64", "artifacts_v2", "artifacts_v3")
MAX_DEPTH = 5
ANCHOR_REL_PATH = "artifacts_v2/dynamics_v1ot_ror_corr010"
OUTPUT_REL_PATH = "artifacts_v3/v3c/utility_audit/dynamics_inventory.csv"

CONDITIONAL_TOKENS = ("dynamics_ablation", "dynamics_sweeps", "dynamics_variants")

CSV_COLUMNS = (
    "field_id",
    "path",
    "n_latent",
    "pair_source",
    "ror_flag",
    "lambda_corr",
    "val_pearson",
    "val_r2",
    "ood_pearson",
    "ood_r2",
    "ridge_margin",
    "knn_margin",
    "uncertainty_spearman",
    "uncertainty_spearman_ood",
    "gate_passed",
    "best_gate_epoch",
    "selection_metric",
    "mtime",
    "eligible",
    "audit_class",
    "notes",
)


def walk_dirs(start: Path, max_depth: int):
    """Yield directories under `start` inclusive, capped at `max_depth` extra levels."""
    if not start.exists():
        return
    yield start
    if max_depth <= 0:
        return
    try:
        children = sorted(p for p in start.iterdir() if p.is_dir())
    except (PermissionError, OSError):
        return
    for child in children:
        yield from walk_dirs(child, max_depth - 1)


def find_dynamics_dirs(root: Path) -> list[Path]:
    """Find all directories under the search tiers that contain model.pt + config.json."""
    found: list[Path] = []
    for tier in SEARCH_TIERS:
        tier_root = root / tier
        for path in walk_dirs(tier_root, MAX_DEPTH):
            if (path / "model.pt").is_file() and (path / "config.json").is_file():
                found.append(path)
    found.sort()
    return found


def load_json_safe(path: Path) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


_LAMBDA_CORR_RE = re.compile(r"(?:^|[/_])corr_?0*(\d{1,3})(?:[/_]|$)", re.IGNORECASE)


def infer_lambda_corr(path_rel: str) -> float:
    """Parse a λ_corr value out of the directory name (e.g. corr010 → 0.10, corr_030 → 0.30)."""
    match = _LAMBDA_CORR_RE.search(path_rel)
    if not match:
        return 0.0
    digits = match.group(1)
    # corr005 → 5 → 0.05, corr010 → 10 → 0.10, corr030 → 30 → 0.30
    # corr1 → 1 → 0.01 (defensive)
    return int(digits) / 1000.0 if len(digits) >= 3 else int(digits) / 100.0


def infer_pair_source(path_rel: str) -> str:
    """Infer pair source from directory naming convention."""
    lower = path_rel.lower()
    if "soft_ot" in lower or "softot" in lower:
        return "soft_ot"
    if "mean_delta" in lower or "meandelta" in lower:
        return "mean_delta"
    if "random" in lower:
        return "random"
    if re.search(r"_n64_nb_|/dynamics_n64_nb", lower):
        return "ot_nb"
    return "ot"


def extract_gate_metrics(gate: dict | None) -> dict[str, Any]:
    """Pull validated metrics from gate.json (preferred) including margins."""
    if not gate:
        return {}
    primary = gate.get("primary") or {}
    ood = gate.get("ood") or {}
    margin_checks = primary.get("margin_checks") or {}
    out: dict[str, Any] = {
        "val_pearson": primary.get("pearson_r"),
        "val_r2": primary.get("r2"),
        "ood_pearson": (ood.get("pearson_r") if ood else None),
        "ood_r2": (ood.get("r2") if ood else None),
        "gate_passed": bool(gate.get("passed", False)),
    }
    ridge_check = margin_checks.get("margin_vs_linear_ridge_pearson")
    if isinstance(ridge_check, dict):
        out["ridge_margin"] = ridge_check.get("value")
    knn_check = margin_checks.get("margin_vs_knn_r2")
    if isinstance(knn_check, dict):
        out["knn_margin"] = knn_check.get("value")
    unc = gate.get("uncertainty_calibration") or {}
    out["uncertainty_spearman"] = unc.get("spearman")
    unc_ood = gate.get("uncertainty_calibration_ood") or {}
    out["uncertainty_spearman_ood"] = unc_ood.get("spearman")
    return out


def extract_fallback_metrics(val_metrics: dict | None, ood_metrics: dict | None) -> dict[str, Any]:
    """Fill in val/OOD Pearson/R² from per-split JSONs if gate.json is missing them."""
    out: dict[str, Any] = {}
    if val_metrics:
        out.setdefault("val_pearson", val_metrics.get("pearson_r") or val_metrics.get("pearson"))
        out.setdefault("val_r2", val_metrics.get("r2"))
    if ood_metrics:
        out.setdefault("ood_pearson", ood_metrics.get("pearson_r") or ood_metrics.get("pearson"))
        out.setdefault("ood_r2", ood_metrics.get("r2"))
    return out


def newest_mtime(path: Path) -> float | None:
    """Most recent mtime among gate.json, model.pt, config.json for stability."""
    candidates: list[float] = []
    for name in ("gate.json", "model.pt", "config.json"):
        p = path / name
        if p.is_file():
            candidates.append(p.stat().st_mtime)
    return max(candidates) if candidates else None


def field_id_from_path(rel_path: str) -> str:
    """Stable filesystem-safe identifier (slashes → double underscore)."""
    return rel_path.replace("/", "__")


def classify(rel_path: str, n_latent: Any) -> tuple[bool, str, str]:
    """Decide eligibility + audit_class + free-text notes from path + metadata."""
    rel_lower = rel_path.lower()
    notes: list[str] = []

    if n_latent is None:
        return False, "rejected_invalid", "config.json missing n_latent"

    if rel_path == ANCHOR_REL_PATH:
        return True, "Anchor", "V2 primary RoR_corr010 — Phase 1 fixed reference"

    if any(token in rel_lower for token in CONDITIONAL_TOKENS):
        return True, "Eligible-conditional", "architecture/lr/loss ablation subtree"

    return True, "Eligible", ""


def build_row(path: Path, root: Path) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    config = load_json_safe(path / "config.json") or {}
    gate = load_json_safe(path / "gate.json")
    val_metrics = load_json_safe(path / "val_metrics.json")
    ood_metrics = load_json_safe(path / "ood_metrics.json")

    n_latent = config.get("n_latent")
    ror_flag = bool(config.get("use_residual_over_ridge", False))
    lambda_corr = infer_lambda_corr(rel)
    pair_source = infer_pair_source(rel)
    best_gate_epoch = config.get("best_gate_epoch")
    selection_metric = config.get("selection_metric")

    metrics: dict[str, Any] = {
        "val_pearson": None,
        "val_r2": None,
        "ood_pearson": None,
        "ood_r2": None,
        "ridge_margin": None,
        "knn_margin": None,
        "uncertainty_spearman": None,
        "uncertainty_spearman_ood": None,
        "gate_passed": False,
    }
    metrics.update(extract_gate_metrics(gate))
    fallback = extract_fallback_metrics(val_metrics, ood_metrics)
    for key, val in fallback.items():
        if metrics.get(key) is None:
            metrics[key] = val

    # If gate.json is absent, fall back to config's best_gate_margin for ridge_margin
    if metrics["ridge_margin"] is None and "best_gate_margin" in config:
        metrics["ridge_margin"] = config["best_gate_margin"]

    eligible, audit_class, notes = classify(rel, n_latent)

    mtime = newest_mtime(path)
    mtime_iso = (
        datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(timespec="seconds")
        if mtime is not None else None
    )

    return {
        "field_id": field_id_from_path(rel),
        "path": rel,
        "n_latent": n_latent,
        "pair_source": pair_source,
        "ror_flag": ror_flag,
        "lambda_corr": lambda_corr,
        "val_pearson": metrics["val_pearson"],
        "val_r2": metrics["val_r2"],
        "ood_pearson": metrics["ood_pearson"],
        "ood_r2": metrics["ood_r2"],
        "ridge_margin": metrics["ridge_margin"],
        "knn_margin": metrics["knn_margin"],
        "uncertainty_spearman": metrics["uncertainty_spearman"],
        "uncertainty_spearman_ood": metrics["uncertainty_spearman_ood"],
        "gate_passed": metrics["gate_passed"],
        "best_gate_epoch": best_gate_epoch,
        "selection_metric": selection_metric,
        "mtime": mtime_iso,
        "eligible": eligible,
        "audit_class": audit_class,
        "notes": notes,
    }


def to_dataframe(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Force a stable, fully-specified schema (so empty cells round-trip as nulls, not nan)."""
    schema: dict[str, pl.DataType] = {
        "field_id": pl.Utf8,
        "path": pl.Utf8,
        "n_latent": pl.Int64,
        "pair_source": pl.Utf8,
        "ror_flag": pl.Boolean,
        "lambda_corr": pl.Float64,
        "val_pearson": pl.Float64,
        "val_r2": pl.Float64,
        "ood_pearson": pl.Float64,
        "ood_r2": pl.Float64,
        "ridge_margin": pl.Float64,
        "knn_margin": pl.Float64,
        "uncertainty_spearman": pl.Float64,
        "uncertainty_spearman_ood": pl.Float64,
        "gate_passed": pl.Boolean,
        "best_gate_epoch": pl.Int64,
        "selection_metric": pl.Utf8,
        "mtime": pl.Utf8,
        "eligible": pl.Boolean,
        "audit_class": pl.Utf8,
        "notes": pl.Utf8,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, strict=False).select(list(CSV_COLUMNS))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V3C dynamics inventory (Phase 0A)")
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="Repository root (default: parent of scripts/)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Output CSV path (default: {OUTPUT_REL_PATH})")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    out_path = args.out if args.out is not None else (root / OUTPUT_REL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dyn_dirs = find_dynamics_dirs(root)
    if args.verbose:
        print(f"[inventory] root={root}", file=sys.stderr)
        print(f"[inventory] found {len(dyn_dirs)} dynamics dirs", file=sys.stderr)

    rows = [build_row(p, root) for p in dyn_dirs]
    rows.sort(key=lambda r: r["path"])
    df = to_dataframe(rows)
    df.write_csv(str(out_path))

    print(f"[inventory] wrote {len(rows)} rows → {out_path}")

    anchor_rows = [r for r in rows if r["audit_class"] == "Anchor"]
    if not anchor_rows:
        print(
            f"[inventory] ERROR: V2 anchor ({ANCHOR_REL_PATH}) not found in inventory",
            file=sys.stderr,
        )
        return 1
    print(f"[inventory] anchor present: {anchor_rows[0]['path']}", file=sys.stderr)

    eligible = sum(1 for r in rows if r["eligible"])
    classes = {}
    for r in rows:
        classes[r["audit_class"]] = classes.get(r["audit_class"], 0) + 1
    print(f"[inventory] eligible={eligible} classes={classes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
