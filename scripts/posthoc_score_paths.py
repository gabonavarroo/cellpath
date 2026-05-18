"""V3B Phase 1 — post-hoc biology scoring of V2 primary rollouts.

Loads the V3B biology layer (gene_safety.parquet + k562_sl_pairs.parquet) and
scores the V2 primary `eval_p0f_c2_seed{42,0,1,7}` × 4 hardness-frontier cells ×
{ppo_deterministic, greedy_dyn_1, greedy_dyn_2, random_uniform_valid, always_noop}
under the new biology layer. Also processes per-episode training-rollouts.parquet
where available for richer per-episode scoring.

Writes `artifacts_v3/eval_v3b_posthoc/posthoc_summary.md` + JSON per-cell tables.

The decisive Phase 1 verdict: does greedy_dyn_2 already have a strictly better
biology profile than PPO under V2's reward? If yes, V3B's premise is suspect —
halt before Phase 2. If no, V3B retraining under safety-aware reward is justified.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


LOG = logging.getLogger("posthoc_score_paths")


# Aggregate-level scoring -----------------------------------------------------


def score_action_freq_aggregate(
    action_freq: dict[str, int],
    gene_safety: pl.DataFrame,
) -> dict[str, Any]:
    """Aggregate biology score from a policy's action frequency table.

    Computes action-frequency-weighted statistics:
    * ``weighted_mean_chronos``: Σ_g freq(g) * chronos(g) / Σ_g freq(g)
    * ``weighted_mean_tox``: Σ_g freq(g) * tox_raw(g) / Σ_g freq(g)
    * ``fraction_actions_common_essential``: Σ_g freq(g) * 1[is_essential] / Σ_g freq(g)
    * ``fraction_actions_safe``: 1 − fraction_actions_common_essential
    * ``n_unique_genes_used``
    * ``top10_genes_mean_chronos``: mean Chronos of the 10 most-frequent genes
    """
    safety_map = {row["gene_symbol"]: row for row in gene_safety.iter_rows(named=True)}

    # Drop NO_OP entry; only score gene actions.
    cleaned = {g: c for g, c in action_freq.items() if g != "NO_OP" and c > 0}
    total = sum(cleaned.values())
    if total == 0:
        return {
            "total_gene_actions": 0,
            "n_unique_genes_used": 0,
            "weighted_mean_chronos": None,
            "weighted_mean_tox": None,
            "weighted_mean_tox_norm": None,
            "fraction_actions_common_essential": None,
            "fraction_actions_safe": None,
            "top10_genes_mean_chronos": None,
            "top10_genes_fraction_essential": None,
            "n_actions_missing_chronos": 0,
        }

    wc_num = 0.0
    wc_den = 0
    wt_num = 0.0
    wtn_num = 0.0
    wess_num = 0
    missing = 0
    for gene, freq in cleaned.items():
        row = safety_map.get(gene)
        if row is None or row.get("missing_chronos") or row.get("chronos") is None:
            missing += freq
            continue
        c = float(row["chronos"])
        wc_num += c * freq
        wc_den += freq
        wt_num += float(row["tox_raw"] or 0.0) * freq
        wtn_num += float(row["tox_norm"] or 0.0) * freq
        if row.get("is_essential"):
            wess_num += freq

    top10 = sorted(cleaned.items(), key=lambda kv: -kv[1])[:10]
    top10_chronos = []
    top10_ess = 0
    for gene, _ in top10:
        row = safety_map.get(gene)
        if row is None or row.get("chronos") is None:
            continue
        top10_chronos.append(float(row["chronos"]))
        if row.get("is_essential"):
            top10_ess += 1

    return {
        "total_gene_actions": int(total),
        "n_unique_genes_used": int(len(cleaned)),
        "weighted_mean_chronos": float(wc_num / wc_den) if wc_den > 0 else None,
        "weighted_mean_tox": float(wt_num / wc_den) if wc_den > 0 else None,
        "weighted_mean_tox_norm": float(wtn_num / wc_den) if wc_den > 0 else None,
        "fraction_actions_common_essential": float(wess_num / total),
        "fraction_actions_safe": float(1.0 - wess_num / total),
        "top10_genes_mean_chronos": float(np.mean(top10_chronos)) if top10_chronos else None,
        "top10_genes_fraction_essential": float(top10_ess / max(1, len(top10))),
        "top10_genes": [g for g, _ in top10],
        "n_actions_missing_chronos": int(missing),
    }


# Per-episode scoring from training rollouts.parquet --------------------------


def score_rollouts_per_episode(
    rollouts: pl.DataFrame,
    gene_safety: pl.DataFrame,
    sl_pair_set: frozenset[tuple[int, int]],
    gene_to_action_idx: dict[str, int],
) -> dict[str, Any]:
    """Score per-episode metrics from a rollouts.parquet table.

    Expected columns: ``episode_id, step, action, gene_symbol, success, terminated, z_norm``.
    Note: in the V2 rollouts schema, ``action`` is the 0-indexed RL action id
    (NOT the dynamics 1-indexed gene_idx). ``gene_symbol`` is the resolved name
    or ``"NO_OP"`` for terminate.
    """
    safety_map = {row["gene_symbol"]: row for row in gene_safety.iter_rows(named=True)}

    per_ep: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts.iter_rows(named=True):
        per_ep[int(row["episode_id"])].append(row)

    n_ep = len(per_ep)
    if n_ep == 0:
        return {"n_episodes": 0}

    tox_paths = []
    tox_norm_paths = []
    ce_counts = []
    sl_viols = []
    n_gene_steps_list = []
    successes = []
    final_distances = []

    for ep_id, steps in per_ep.items():
        steps.sort(key=lambda r: r["step"])
        gene_actions = []
        tox = 0.0
        tox_norm = 0.0
        ce = 0
        for s in steps:
            sym = s["gene_symbol"]
            if sym == "NO_OP":
                continue
            row = safety_map.get(sym)
            if row is None or row.get("missing_chronos"):
                continue
            tox += float(row["tox_raw"] or 0.0)
            tox_norm += float(row["tox_norm"] or 0.0)
            if row.get("is_essential"):
                ce += 1
            idx = gene_to_action_idx.get(sym)
            if idx is not None:
                gene_actions.append(idx)
        # SL pair check
        sl = 0
        for i in range(len(gene_actions)):
            for j in range(i + 1, len(gene_actions)):
                a, b = gene_actions[i], gene_actions[j]
                lo, hi = (a, b) if a < b else (b, a)
                if (lo, hi) in sl_pair_set:
                    sl += 1
        tox_paths.append(tox)
        tox_norm_paths.append(tox_norm)
        ce_counts.append(ce)
        sl_viols.append(sl)
        n_gene_steps_list.append(len(gene_actions))
        successes.append(bool(steps[-1].get("success", False)))
        final_distances.append(float(steps[-1].get("z_norm", float("nan"))))

    def _stat(vals):
        if not vals:
            return None, None
        arr = np.asarray(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    tox_m, tox_s = _stat(tox_paths)
    txn_m, txn_s = _stat(tox_norm_paths)
    ce_m, ce_s = _stat(ce_counts)
    sl_m, sl_s = _stat(sl_viols)
    ngs_m, ngs_s = _stat(n_gene_steps_list)
    sr = float(np.mean(successes))
    fd_m, fd_s = _stat([d for d in final_distances if np.isfinite(d)])

    return {
        "n_episodes": int(n_ep),
        "mean_tox_path": tox_m, "std_tox_path": tox_s,
        "mean_tox_path_norm": txn_m, "std_tox_path_norm": txn_s,
        "mean_common_essential_per_ep": ce_m, "std_common_essential_per_ep": ce_s,
        "mean_sl_violations_per_ep": sl_m, "std_sl_violations_per_ep": sl_s,
        "mean_n_gene_steps": ngs_m, "std_n_gene_steps": ngs_s,
        "success_rate": sr,
        "mean_final_distance": fd_m, "std_final_distance": fd_s,
        "fraction_zero_ce": float(np.mean([1.0 if c == 0 else 0.0 for c in ce_counts])),
        "fraction_zero_sl": float(np.mean([1.0 if c == 0 else 0.0 for c in sl_viols])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biology_dir", default="artifacts_v3/v3b_biology")
    parser.add_argument(
        "--eval_dirs", nargs="+",
        default=[
            "artifacts_v2/eval_p0f_c2_seed42",
            "artifacts_v2/eval_p0f_c2_seed0",
            "artifacts_v2/eval_p0f_c2_seed1",
            "artifacts_v2/eval_p0f_c2_seed7",
        ],
        help="V2 hardness-frontier eval directories to score.",
    )
    parser.add_argument(
        "--rl_dirs", nargs="+",
        default=[
            "artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed42",
            "artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed0",
            "artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed1",
            "artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed7",
        ],
        help="V2 PPO training-rollout directories (have rollouts.parquet).",
    )
    parser.add_argument("--gene_vocab", default="artifacts/vae/gene_vocab.json")
    parser.add_argument("--out_dir", default="artifacts_v3/eval_v3b_posthoc")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    biology_dir = (repo_root / args.biology_dir).resolve() if not Path(args.biology_dir).is_absolute() else Path(args.biology_dir)
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load biology layer
    sys.path.insert(0, str(repo_root))
    from src.analysis.path_feasibility import load_biology_layer

    layer = load_biology_layer(biology_dir)
    gene_safety = layer.gene_safety
    sl_pair_set = layer.sl_pair_set
    LOG.info(
        "Biology layer: %d genes (%d with Chronos, %d essential); SL pair set: %d",
        gene_safety.height,
        layer.coverage["gene_safety"]["n_with_chronos"],
        layer.coverage["gene_safety"]["n_essential_chronos_lt_minus_0_5"],
        len(sl_pair_set),
    )

    # Build gene→action_idx lookup
    with open(repo_root / args.gene_vocab) as f:
        vocab = json.load(f)
    gene_to_idx = {g: i for i, g in enumerate(vocab["genes"])}

    # ----- Aggregate-level scoring across V2 eval cells × policies -----
    aggregate_rows: list[dict[str, Any]] = []
    cells = ["k2_epsp25_bin6-8_splitood", "k2_epsp25_bin8-10_splitood",
             "k3_epsp25_bin6-8_splitood", "k3_epsp25_bin8-10_splitood"]
    policies = ["ppo_deterministic", "greedy_dyn_1", "greedy_dyn_2",
                "random_uniform_valid", "always_noop"]

    for eval_dir_str in args.eval_dirs:
        eval_dir = (repo_root / eval_dir_str).resolve() if not Path(eval_dir_str).is_absolute() else Path(eval_dir_str)
        if not eval_dir.exists():
            LOG.warning("Skipping missing eval dir: %s", eval_dir)
            continue
        seed = eval_dir.name.split("seed")[-1]
        for cell in cells:
            for policy in policies:
                summary_path = eval_dir / cell / policy / "summary.json"
                if not summary_path.exists():
                    continue
                with open(summary_path) as f:
                    summ = json.load(f)
                action_freq = summ.get("action_freq", {})
                agg = score_action_freq_aggregate(action_freq, gene_safety)
                aggregate_rows.append({
                    "seed": seed,
                    "cell": cell,
                    "policy": policy,
                    "success_rate": summ.get("success_rate"),
                    "mean_steps": summ.get("mean_steps"),
                    "mean_final_distance": summ.get("mean_final_distance"),
                    **agg,
                })

    if not aggregate_rows:
        LOG.error("No V2 eval summaries found. Did the eval_p0f_c2_seed* dirs move?")
        return 1

    agg_df = pl.DataFrame(aggregate_rows)
    agg_path = out_dir / "aggregate_per_cell_per_policy.parquet"
    agg_df.write_parquet(str(agg_path))
    # CSV: drop list columns (top10_genes) — CSV can't represent nested data.
    csv_cols = [c for c in agg_df.columns if not isinstance(agg_df.schema[c], pl.List)]
    agg_df.select(csv_cols).write_csv(str(out_dir / "aggregate_per_cell_per_policy.csv"))
    LOG.info("Wrote aggregate scores: %s (%d rows)", agg_path, agg_df.height)

    # ----- Per-episode scoring across PPO training rollouts -----
    per_ep_rows: list[dict[str, Any]] = []
    for rl_dir_str in args.rl_dirs:
        rl_dir_path = (repo_root / rl_dir_str) if not Path(rl_dir_str).is_absolute() else Path(rl_dir_str)
        rl_dir = rl_dir_path.resolve()
        rp = rl_dir / "rollouts.parquet"
        if not rp.exists():
            LOG.warning("Skipping missing rollouts: %s", rp)
            continue
        # Use the original (non-resolved) path name to preserve _seedXX suffix even when
        # the directory is a symlink to a name without the suffix.
        link_name = rl_dir_path.name
        seed = link_name.split("seed")[-1] if "seed" in link_name else link_name
        df = pl.read_parquet(str(rp))
        result = score_rollouts_per_episode(df, gene_safety, sl_pair_set, gene_to_idx)
        per_ep_rows.append({"seed": seed, "source": link_name, **result})

    per_ep_df = pl.DataFrame(per_ep_rows) if per_ep_rows else None
    if per_ep_df is not None:
        per_ep_path = out_dir / "per_episode_training_rollouts.parquet"
        per_ep_df.write_parquet(str(per_ep_path))
        per_ep_df.write_csv(str(out_dir / "per_episode_training_rollouts.csv"))
        LOG.info("Wrote per-episode training-rollout scores: %s (%d rows)", per_ep_path, per_ep_df.height)

    # ----- Build the verdict -----
    primary_cell = "k3_epsp25_bin8-10_splitood"
    primary = agg_df.filter(pl.col("cell") == primary_cell)

    def _mean_by_policy(df: pl.DataFrame, policy: str, col: str) -> float | None:
        sub = df.filter(pl.col("policy") == policy)[col]
        if sub.is_empty():
            return None
        vals = [v for v in sub.to_list() if v is not None]
        return float(np.mean(vals)) if vals else None

    ppo_chr = _mean_by_policy(primary, "ppo_deterministic", "weighted_mean_chronos")
    g2_chr = _mean_by_policy(primary, "greedy_dyn_2", "weighted_mean_chronos")
    ppo_ess = _mean_by_policy(primary, "ppo_deterministic", "fraction_actions_common_essential")
    g2_ess = _mean_by_policy(primary, "greedy_dyn_2", "fraction_actions_common_essential")
    ppo_tox = _mean_by_policy(primary, "ppo_deterministic", "weighted_mean_tox")
    g2_tox = _mean_by_policy(primary, "greedy_dyn_2", "weighted_mean_tox")

    LOG.info("Primary cell %s (mean across seeds):", primary_cell)
    LOG.info("  PPO weighted-mean Chronos: %.4f  | greedy_dyn_2: %.4f", ppo_chr or 0, g2_chr or 0)
    LOG.info("  PPO fraction common-essential: %.4f | greedy_dyn_2: %.4f", ppo_ess or 0, g2_ess or 0)
    LOG.info("  PPO weighted-mean tox: %.4f | greedy_dyn_2: %.4f", ppo_tox or 0, g2_tox or 0)

    # The Phase 1 verdict:
    # PROCEED if PPO's biology profile is comparable to or worse than greedy_dyn_2 —
    # then a biorealistic-reward retrain can find safer paths.
    # HALT if greedy_dyn_2 already has strictly safer biology than PPO under V2's
    # reward; then the V3B premise is moot at primary.
    if ppo_chr is None or g2_chr is None:
        verdict = "INCONCLUSIVE — insufficient Chronos coverage to compare"
    elif (g2_chr > ppo_chr + 0.02) and (g2_ess < ppo_ess - 0.02):
        # greedy_dyn_2 has higher (less negative, safer) Chronos AND lower essential rate
        verdict = "HALT — greedy_dyn_2 already has strictly safer biology profile than PPO at primary"
    else:
        verdict = "PROCEED — greedy_dyn_2 is NOT strictly safer than PPO; V3B retraining under safety-aware reward is justified"

    LOG.info("Verdict: %s", verdict)

    # ----- Write the summary markdown -----
    md_lines = [
        "# V3B Phase 1 — Post-hoc biology scoring of V2 primary",
        "",
        "> Built by `scripts/posthoc_score_paths.py` against `artifacts_v3/v3b_biology/`",
        "> + `artifacts_v2/eval_p0f_c2_seed{42,0,1,7}/` + `artifacts_v2/rl_v1ot_ror_corr010_*` rollouts.",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Why this verdict",
        "",
        f"At the V2 primary cell (`{primary_cell}`, mean across 4 seeds):",
        "",
        "| Policy | Weighted-mean Chronos | Weighted-mean tox_raw | Fraction common-essential actions |",
        "|---|---:|---:|---:|",
    ]
    for policy in policies:
        chr_ = _mean_by_policy(primary, policy, "weighted_mean_chronos")
        tox_ = _mean_by_policy(primary, policy, "weighted_mean_tox")
        ess_ = _mean_by_policy(primary, policy, "fraction_actions_common_essential")
        chr_s = f"{chr_:+.4f}" if chr_ is not None else "—"
        tox_s = f"{tox_:.4f}" if tox_ is not None else "—"
        ess_s = f"{ess_:.4f}" if ess_ is not None else "—"
        md_lines.append(f"| {policy} | {chr_s} | {tox_s} | {ess_s} |")
    md_lines.append("")
    md_lines.append("More negative Chronos = stronger K562 essentiality (likely riskier action).")
    md_lines.append("Higher tox = stronger experimental disturbance prior.")
    md_lines.append("Higher fraction_essential = more often picks `is_essential` (Chronos < −0.5) genes.")
    md_lines.append("")

    # All cells × policies table (compact)
    md_lines += [
        "## All hardness-frontier cells × policies",
        "",
        "(One row per (seed, cell, policy). See `aggregate_per_cell_per_policy.csv` for the full table.)",
        "",
        "Mean across seeds, per (cell, policy):",
        "",
        "| Cell | Policy | success | n_unique_genes | wmean_chronos | wmean_tox | frac_essential | top10_mean_chronos |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        for policy in policies:
            sub = agg_df.filter((pl.col("cell") == cell) & (pl.col("policy") == policy))
            if sub.is_empty():
                continue
            def _mean(col):
                vals = [v for v in sub[col].to_list() if v is not None]
                return float(np.mean(vals)) if vals else None
            sr = _mean("success_rate"); ng = _mean("n_unique_genes_used")
            wmc = _mean("weighted_mean_chronos"); wmt = _mean("weighted_mean_tox")
            fe = _mean("fraction_actions_common_essential")
            t10c = _mean("top10_genes_mean_chronos")
            fmt = lambda x, d=4: (f"{x:+.{d}f}" if x is not None else "—")
            md_lines.append(
                f"| {cell} | {policy} | "
                f"{(sr or 0):.3f} | {(ng or 0):.1f} | "
                f"{fmt(wmc)} | {fmt(wmt)} | {fmt(fe, 3)} | {fmt(t10c)} |"
            )

    # Per-episode training rollout aggregates
    if per_ep_df is not None and per_ep_df.height > 0:
        md_lines += [
            "",
            "## Per-episode metrics (V2 PPO training-side rollouts)",
            "",
            "**Note:** training-rollout episodes use the curriculum's start-pool distribution",
            "(distance bins 4–10 over time), not the V2-hard-bench cells. Use this table for",
            "magnitudes; the cell-level comparisons above use V2's true OOD eval.",
            "",
            "| Seed | n_ep | success | mean_n_gene_steps | mean_tox_path | mean_CE/ep | mean_SL/ep | frac_zero_CE | frac_zero_SL |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in per_ep_df.iter_rows(named=True):
            md_lines.append(
                f"| {row['seed']} | {row['n_episodes']} | "
                f"{(row.get('success_rate') or 0):.3f} | "
                f"{(row.get('mean_n_gene_steps') or 0):.2f} | "
                f"{(row.get('mean_tox_path') or 0):.4f} | "
                f"{(row.get('mean_common_essential_per_ep') or 0):.3f} | "
                f"{(row.get('mean_sl_violations_per_ep') or 0):.3f} | "
                f"{(row.get('fraction_zero_ce') or 0):.3f} | "
                f"{(row.get('fraction_zero_sl') or 0):.3f} |"
            )

    md_lines += [
        "",
        "## Honesty caveats",
        "",
        "* DepMap Chronos is CRISPR-Cas9 knockout; Norman 2019 is CRISPRa. Treat",
        "  Chronos as a prior on experimental disturbance, not therapeutic toxicity.",
        "  (See V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §3.5.)",
        "* The Horlbeck K562 SL pair set has **zero overlap** with the Norman 105-gene",
        "  action universe (`artifacts_v3/v3b_biology/coverage.json::k562_sl_pairs`).",
        "  All `sl_violations` columns are structurally 0 here. The Phase 5b reward",
        "  Variant E reduces to B + C + D.",
        "* Per-episode metrics from the training-rollout table use the curriculum's",
        "  bin-4–10 start pool, not the V2 hard-bench cells. Magnitudes are indicative.",
    ]

    summary_path = out_dir / "posthoc_summary.md"
    summary_path.write_text("\n".join(md_lines) + "\n")
    LOG.info("Wrote %s", summary_path)

    # Also dump JSON for downstream
    verdict_json = out_dir / "verdict.json"
    verdict_json.write_text(json.dumps({
        "verdict": verdict,
        "primary_cell": primary_cell,
        "ppo_weighted_mean_chronos": ppo_chr,
        "greedy_dyn_2_weighted_mean_chronos": g2_chr,
        "ppo_fraction_common_essential": ppo_ess,
        "greedy_dyn_2_fraction_common_essential": g2_ess,
        "ppo_weighted_mean_tox": ppo_tox,
        "greedy_dyn_2_weighted_mean_tox": g2_tox,
        "n_seed_dirs_loaded": len(args.eval_dirs),
    }, indent=2))
    LOG.info("Wrote %s", verdict_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
