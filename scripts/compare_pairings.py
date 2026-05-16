"""P0B' — side-by-side comparator across pairing-method runs.

Reads {gate.json, metadata.json, pairing_noise.json} from each run dir and emits
a tidy {json, md} comparison. No model is loaded; no torch import.

Usage::

    python scripts/compare_pairings.py \\
        --runs ot:artifacts/pairs:artifacts/dynamics \\
               mean_delta:artifacts_v2/pairs_mean_delta:artifacts_v2/dynamics_mean_delta_default \\
               random:artifacts_v2/pairs_random:artifacts_v2/dynamics_random_default \\
        --noise_json ot=artifacts_v2/diagnostics/pairing_noise.json \\
                     mean_delta=artifacts_v2/diagnostics/pairing_noise_mean_delta.json \\
                     random=artifacts_v2/diagnostics/pairing_noise_random.json \\
        --out artifacts_v2/diagnostics/pairing_comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _row(
    method: str,
    pairs_dir: str,
    dynamics_dir: str,
    noise_json: Path | None,
) -> dict[str, Any]:
    pmeta = json.loads((Path(pairs_dir) / "metadata.json").read_text())
    gate = json.loads((Path(dynamics_dir) / "gate.json").read_text())
    prim = gate["primary"]
    ridge_p = float(prim["baselines"]["linear_ridge"]["pearson_r"])
    ood = gate.get("ood") or {}
    if ood:
        ood_ridge = float(
            (ood.get("baselines") or {}).get("linear_ridge", {}).get(
                "pearson_r", float("nan")
            )
        )
        ood_mlp = float(ood.get("pearson_r", float("nan")))
    else:
        ood_ridge = float("nan")
        ood_mlp = float("nan")
    noise = float("nan")
    cand: Path | None = noise_json
    if cand is None:
        for c in (
            Path(dynamics_dir) / "pairing_noise.json",
            Path(pairs_dir) / "pairing_noise.json",
        ):
            if c.exists():
                cand = c
                break
    if cand is not None and cand.exists():
        noise = float(json.loads(cand.read_text())["summary"]["median_noise_ratio"])
    return {
        "pairing_method": method,
        "n_train": int(pmeta.get("n_train", 0)),
        "pairing_noise_median": noise,
        "val_mlp_pearson": float(prim["pearson_r"]),
        "val_ridge_pearson": ridge_p,
        "val_mlp_minus_ridge_pearson": float(prim["pearson_r"]) - ridge_p,
        "ood_mlp_pearson": ood_mlp,
        "ood_ridge_pearson": ood_ridge,
        "ood_mlp_minus_ridge_pearson": ood_mlp - ood_ridge,
        "uncertainty_spearman": float(gate["uncertainty_calibration"]["spearman"]),
        "gate_passed": bool(gate["passed"]),
    }


def _format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(
    *,
    runs: Iterable[tuple[str, str, str]] | None = None,
    out: str | None = None,
    noise_map: dict[str, str] | None = None,
) -> int:
    if runs is None or out is None:
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--runs",
            nargs="+",
            required=True,
            help="One or more <method>:<pairs_dir>:<dynamics_dir>",
        )
        ap.add_argument(
            "--noise_json",
            nargs="*",
            default=[],
            help="Optional <method>=<path> overrides for pairing_noise.json",
        )
        ap.add_argument("--out", required=True)
        args = ap.parse_args()
        runs_parsed: list[tuple[str, str, str]] = []
        for r in args.runs:
            parts = r.split(":", 2)
            if len(parts) != 3:
                raise SystemExit(
                    f"--runs spec '{r}' must be <method>:<pairs_dir>:<dynamics_dir>"
                )
            runs_parsed.append((parts[0], parts[1], parts[2]))
        noise_overrides = dict(s.split("=", 1) for s in args.noise_json)
        out_path = args.out
    else:
        runs_parsed = list(runs)
        noise_overrides = dict(noise_map) if noise_map else {}
        out_path = out

    rows = [
        _row(
            method,
            pdir,
            ddir,
            Path(noise_overrides[method]) if method in noise_overrides else None,
        )
        for (method, pdir, ddir) in runs_parsed
    ]

    out_json = Path(out_path + ".json")
    out_md = Path(out_path + ".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, indent=2))

    cols = [
        "pairing_method",
        "n_train",
        "pairing_noise_median",
        "val_mlp_minus_ridge_pearson",
        "ood_mlp_minus_ridge_pearson",
        "uncertainty_spearman",
        "gate_passed",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(_format_cell(r[c]) for c in cols) + " |" for r in rows
    )
    out_md.write_text(f"# P0B' pairing comparison\n\n{header}\n{sep}\n{body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
