"""Aggregate pairing-noise + dynamics-gate results across the Proposal 1 sweep.

Reads:
  - artifacts_proposal1/diagnostics/00_baselines_existing.json (V1/soft_ot/mean_delta/random at eps=0.05)
  - artifacts_proposal1/diagnostics/noise_<name>.json          (new sweeps)
  - artifacts_proposal1/dynamics_<name>_ror/gate.json          (new dynamics)
  - artifacts_proposal1/dynamics_<name>_ror/val_metrics.json
  - artifacts_proposal1/dynamics_<name>_ror/ood_metrics.json

Writes:
  - artifacts_proposal1/reports/comparison.json
  - artifacts_proposal1/reports/comparison.md (human-readable)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
DIAG = ROOT / "diagnostics"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _noise_summary(name: str) -> dict | None:
    p = DIAG / f"noise_{name}.json"
    d = _load(p)
    return d.get("summary") if d else None


def _dyn_metrics(name: str) -> dict | None:
    d = ROOT / f"dynamics_{name}_ror"
    if not d.exists():
        return None
    gate = _load(d / "gate.json") or {}
    val_raw = _load(d / "val_metrics.json") or {}
    ood_raw = _load(d / "ood_metrics.json") or {}
    cfg = _load(d / "config.json") or {}
    # val/ood metric files nest the actual numbers under "primary"
    val = val_raw.get("primary", val_raw)
    ood = ood_raw.get("primary", ood_raw)
    out: dict = {
        "dir": str(d),
        "gate_passed": gate.get("passed"),
        "val_pearson": val.get("pearson_r") or val.get("pearson"),
        "val_r2": val.get("r2"),
        "ood_pearson": ood.get("pearson_r") or ood.get("pearson"),
        "ood_r2": ood.get("r2"),
        "epochs_run": cfg.get("epochs_run"),
        "best_gate_epoch": cfg.get("best_gate_epoch"),
        "best_gate_margin": cfg.get("best_gate_margin"),
        "use_residual_over_ridge": cfg.get("use_residual_over_ridge"),
        "use_state_linear_skip": cfg.get("use_state_linear_skip"),
        "lambda_corr": cfg.get("lambda_corr"),
    }
    # Pull MLP-vs-ridge margins straight from gate.json if present
    primary = gate.get("primary", {})
    out["val_ridge_pearson"] = (
        primary.get("baselines", {}).get("linear_ridge", {}).get("pearson_r")
    )
    out["val_margin_vs_ridge_pearson"] = (
        primary.get("margin_checks", {}).get("margin_vs_linear_ridge_pearson", {}).get("value")
    )
    ood_p = gate.get("ood", {})
    out["ood_ridge_pearson"] = (
        ood_p.get("baselines", {}).get("linear_ridge", {}).get("pearson_r")
    )
    out["ood_margin_vs_ridge_pearson"] = (
        ood_p.get("margin_checks", {}).get("margin_vs_linear_ridge_pearson", {}).get("value")
    )
    return out


def main() -> None:
    # Existing baselines
    baselines = _load(DIAG / "00_baselines_existing.json") or {}

    # New noise measurements
    new_noise_names = ["ot_eps001", "ot_eps002", "soft_ot_eps001", "soft_ot_eps005"]
    noise_new: dict = {}
    for n in new_noise_names:
        s = _noise_summary(n)
        if s:
            noise_new[n] = s

    # Dynamics metrics (new + reference snapshots we control)
    dyn_names = ["v1ot", "soft_ot", "mean_delta", "ot_eps001", "ot_eps002", "soft_ot_eps001"]
    dyn = {n: _dyn_metrics(n) for n in dyn_names if _dyn_metrics(n) is not None}

    # External references from frozen artifacts
    references: dict = {}
    ref_specs = {
        "artifacts/dynamics  (V1, state_linear_skip, OT eps=0.05)": Path(
            "artifacts/dynamics"
        ),
        "artifacts_v2_experiments/dynamics_soft_ot_default  (V2 soft_ot, state_linear_skip)": Path(
            "artifacts_v2_experiments/dynamics_soft_ot_default"
        ),
        "artifacts_v2_experiments/dynamics_mean_delta_corr_010  (V2 mean_delta, RoR+corr0.10)": Path(
            "artifacts_v2_experiments/dynamics_mean_delta_corr_010"
        ),
        "artifacts_64/dynamics  (64D legacy, state_linear_skip)": Path(
            "artifacts_64/dynamics"
        ),
    }
    for label, d in ref_specs.items():
        if not d.exists():
            continue
        gate = _load(d / "gate.json") or {}
        val_raw = _load(d / "val_metrics.json") or {}
        ood_raw = _load(d / "ood_metrics.json") or {}
        cfg = _load(d / "config.json") or {}
        val = val_raw.get("primary", val_raw)
        ood = ood_raw.get("primary", ood_raw)
        references[label] = {
            "gate_passed": gate.get("passed"),
            "val_pearson": val.get("pearson_r") or val.get("pearson"),
            "ood_pearson": ood.get("pearson_r") or ood.get("pearson"),
            "use_residual_over_ridge": cfg.get("use_residual_over_ridge"),
            "use_state_linear_skip": cfg.get("use_state_linear_skip"),
            "lambda_corr": cfg.get("lambda_corr"),
            "n_latent": cfg.get("n_latent"),
            "val_margin_vs_ridge_pearson": (
                gate.get("primary", {})
                .get("margin_checks", {})
                .get("margin_vs_linear_ridge_pearson", {})
                .get("value")
            ),
        }

    out = {
        "baselines_existing": baselines,
        "noise_new_sweeps": noise_new,
        "dynamics_new_runs": dyn,
        "dynamics_references_on_disk": references,
    }
    (REPORTS / "comparison.json").write_text(json.dumps(out, indent=2))

    # ----- markdown -----
    lines: list[str] = []
    lines.append("# Proposal 1 — Aggregated Comparison")
    lines.append("")
    lines.append("## 1. Pairing noise (per-gene residual var / total var of Δz; lower = better)")
    lines.append("")
    lines.append("| Method | median | mean | p25 | p75 | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")

    def _row(label: str, s: dict | None) -> str:
        if s is None:
            return f"| {label} | – | – | – | – | – |"
        return (
            f"| {label} | {s['median_noise_ratio']:.4f} | {s['mean_noise_ratio']:.4f} | "
            f"{s['p25_noise_ratio']:.4f} | {s['p75_noise_ratio']:.4f} | {s['max_noise_ratio']:.4f} |"
        )

    lines.append(_row("V1 OT eps=0.05 (existing)", baselines.get("v1_ot_eps005")))
    lines.append(_row("OT eps=0.02 (NEW)",        noise_new.get("ot_eps002")))
    lines.append(_row("OT eps=0.01 (NEW)",        noise_new.get("ot_eps001")))
    lines.append(_row("soft_ot eps=0.05 (existing)", baselines.get("soft_ot_eps005")))
    lines.append(_row("soft_ot eps=0.05 (verify)",   noise_new.get("soft_ot_eps005")))
    lines.append(_row("soft_ot eps=0.01 (NEW)",  noise_new.get("soft_ot_eps001")))
    lines.append(_row("mean_delta (existing)",   baselines.get("mean_delta")))
    lines.append(_row("random (existing)",       baselines.get("random")))
    lines.append("")

    lines.append("## 2. Dynamics (RoR + corr0.10 on each pair set; gate passes if val MLP−ridge Pearson margin ≥ 0.030)")
    lines.append("")
    lines.append("| Pair set | gate | val Pearson | ridge | margin | OOD Pearson | OOD margin | epochs | arch |")
    lines.append("| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    def _drow(label: str, m: dict | None) -> str:
        if m is None:
            return f"| {label} | – | – | – | – | – | – | – | – |"
        arch = "RoR" if m.get("use_residual_over_ridge") else ("state_linear" if m.get("use_state_linear_skip") else "?")
        return (
            f"| {label} | {'✓' if m.get('gate_passed') else '✗'} | "
            f"{(m.get('val_pearson') or 0):.4f} | "
            f"{(m.get('val_ridge_pearson') or 0):.4f} | "
            f"{(m.get('val_margin_vs_ridge_pearson') or 0):+.4f} | "
            f"{(m.get('ood_pearson') or 0):.4f} | "
            f"{(m.get('ood_margin_vs_ridge_pearson') or 0):+.4f} | "
            f"{m.get('epochs_run', '?')} | {arch} |"
        )

    for n, label in [
        ("v1ot",        "V1 OT eps=0.05 + RoR (NEW, baseline)"),
        ("soft_ot",     "V2 soft_ot eps=0.05 + RoR (NEW, ⭐ winner)"),
        ("mean_delta",  "V2 mean_delta + RoR (NEW, abandoned — memory)"),
    ]:
        lines.append(_drow(label, dyn.get(n)))
    lines.append("")

    lines.append("## 3. Reference dynamics already on disk (read-only, for comparison)")
    lines.append("")
    lines.append("| Reference | gate | val Pearson | OOD Pearson | val margin | n_latent | arch |")
    lines.append("| --- | :---: | ---: | ---: | ---: | ---: | --- |")
    for label, m in references.items():
        arch = "RoR" if m.get("use_residual_over_ridge") else ("state_linear" if m.get("use_state_linear_skip") else "?")
        lines.append(
            f"| {label} | {'✓' if m.get('gate_passed') else '✗'} | "
            f"{(m.get('val_pearson') or 0):.4f} | "
            f"{(m.get('ood_pearson') or 0):.4f} | "
            f"{(m.get('val_margin_vs_ridge_pearson') or 0):+.4f} | "
            f"{m.get('n_latent', '?')} | {arch} |"
        )
    lines.append("")

    (REPORTS / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORTS / 'comparison.json'} and {REPORTS / 'comparison.md'}")


if __name__ == "__main__":
    main()
