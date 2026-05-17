"""P0F Phase 5 — driver that calls `src.analysis.v2_figures.generate_all_figures()`.

Reads P0F evaluation outputs and emits PNGs under `artifacts_v2/figures/`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_dynamics_taxonomy(root: Path) -> list[dict]:
    """Pull gate/OOD/beam/PPO numbers for the four V2 dynamics fields."""
    rows: list[dict] = []
    sources = [
        ("V1 OT",       Path("artifacts/dynamics/gate.json"),
         Path("artifacts_v2/reachability_probe/probe_results.json"), "v1_ot_repeaton",
         Path("artifacts_v2/eval_p0f_seed_aggregate/seed_aggregate_success_rate.json"),
         "b5_v1ot_terminal_curric_1M"),
        ("RoR_corr010", Path("artifacts_v2/dynamics_v1ot_ror_corr010/gate.json"),
         Path("artifacts_v2/reachability_probe_p0d_trackA/probe_results.json"),
         "v1ot_ror_corr010_repeaton",
         Path("artifacts_v2/eval_p0f_seed_aggregate/seed_aggregate_success_rate.json"),
         "c2_ror_corr010_terminal_curric_1M"),
        ("mean_delta",  Path("artifacts_v2/dynamics_mean_delta_corr_030/gate.json"),
         Path("artifacts_v2/reachability_probe/probe_results.json"), "mean_delta_repeaton",
         None, None),
        ("soft_ot",     Path("artifacts_v2/dynamics_soft_ot_default/gate.json"),
         Path("artifacts_v2/reachability_probe/probe_results.json"), "soft_ot_repeaton",
         None, None),
    ]
    for label, gate_path, beam_path, beam_key, ppo_agg_path, ppo_cfg in sources:
        gate_val = float("nan")
        ood_p = float("nan")
        beam_s = float("nan")
        ppo_p = float("nan")
        if gate_path.exists():
            g = json.loads(gate_path.read_text())
            mlp = g.get("primary", {}).get("pearson_r")
            ridge = g.get("primary", {}).get("baselines", {}).get("linear_ridge", {}).get("pearson_r")
            if mlp is not None and ridge is not None:
                gate_val = float(mlp - ridge)
            ood_p = float(g.get("ood", {}).get("pearson_r", float("nan")))
        if beam_path.exists():
            b = json.loads(beam_path.read_text())
            run = b.get("dynamics_runs", {}).get(beam_key, {})
            beam_s = float(run.get("success_rate", float("nan")))
        if ppo_agg_path and ppo_agg_path.exists() and ppo_cfg:
            agg = json.loads(ppo_agg_path.read_text())
            primary = agg.get(ppo_cfg, {}).get("k3_epsp25_bin8-10_splitood", {}).get("ppo_deterministic", {})
            ppo_p = float(primary.get("mean", float("nan")))
        else:
            ppo_p = 0.0  # mean_delta and soft_ot are RL-dead
        rows.append({
            "label": label,
            "gate_val_margin": gate_val,
            "ood_pearson": ood_p,
            "beam_success": beam_s,
            "ppo_primary": ppo_p,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts_v2/figures")
    args = ap.parse_args()
    from src.analysis.v2_figures import generate_all_figures

    eval_dirs = {
        "B5 (V1 OT) seed42":     Path("artifacts_v2/eval_p0f_b5_seed42"),
        "C2 (RoR_corr010) seed42": Path("artifacts_v2/eval_p0f_c2_seed42"),
    }
    run_dirs = {
        "B5 seed42": Path("artifacts_v2/rl_v1ot_terminal_curric_k3_1M_seed42"),
        "B5 seed0":  Path("artifacts_v2/rl_v1ot_terminal_curric_k3_1M_seed0"),
        "C2 seed42": Path("artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed42"),
        "C2 seed0":  Path("artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed0"),
    }
    seed_aggregate_json = Path("artifacts_v2/eval_p0f_seed_aggregate/seed_aggregate_success_rate.json")
    rollouts_paths = {
        "B5 (V1 OT) seed42": Path("artifacts_v2/rl_v1ot_terminal_curric_k3_1M_seed42/rollouts.parquet"),
        "C2 (RoR_corr010) seed42": Path("artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed42/rollouts.parquet"),
    }
    dynamics_rows = collect_dynamics_taxonomy(Path("."))
    out_dir = Path(args.out_dir)
    written = generate_all_figures(
        eval_dirs=eval_dirs,
        run_dirs=run_dirs,
        seed_aggregate_json=seed_aggregate_json,
        dynamics_taxonomy_rows=dynamics_rows,
        rollouts_paths=rollouts_paths,
        out_dir=out_dir,
    )
    for p in written:
        print(f"  wrote {p}")
    print(f"{len(written)} figures emitted under {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
