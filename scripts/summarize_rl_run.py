"""scripts/summarize_rl_run.py — aggregate metrics for a single RL run.

Reads ``rollouts.parquet`` + ``action_freq.json`` (+ optional ``metadata.json``) and emits
``summary.json`` plus an optional Markdown report (``summary.md``) suitable for pasting into
the thesis defense or PROGRESS.md. If a random-policy baseline directory is supplied,
computes the PPO-over-random success delta and includes the comparison block.

Designed to work on any folder that follows the Contract-4 schema (PPO eval folder, random
policy folder, sweep subfolders under ``artifacts/rl_sweeps/``).

Usage
-----
::

    python scripts/summarize_rl_run.py \\
        --run-dir artifacts/rl_sweeps/p50_start8_noopfix_500k_detfinal/eval_deterministic \\
        --random-baseline-dir artifacts/rl_sweeps/p50_start8_noopfix_500k_detfinal/random_baseline \\
        --out summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: could not parse {path}: {exc}", file=sys.stderr)
        return None


def _shannon_entropy(counts: dict[str, int]) -> float:
    """Shannon entropy of an action-frequency histogram, in nats (natural log)."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return float(h)


def _entropy_normalized(counts: dict[str, int]) -> float:
    """Entropy normalized by ``log(K)`` where K = number of *observed* distinct actions."""
    k = sum(1 for c in counts.values() if c > 0)
    if k <= 1:
        return 0.0
    return _shannon_entropy(counts) / math.log(k)


def _summarize_rollouts(rollouts_path: Path, action_freq_path: Path) -> dict[str, Any]:
    import polars as pl

    df = pl.read_parquet(str(rollouts_path))
    if df.height == 0:
        return {
            "n_episodes": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "mean_total_reward": 0.0,
            "mean_final_distance": 0.0,
            "mean_min_distance": 0.0,
            "noop_first_action_count": 0,
            "noop_first_action_rate": 0.0,
            "action_entropy_nats": 0.0,
            "action_entropy_normalized": 0.0,
            "top_actions": [],
        }

    by_ep = df.group_by("episode_id").agg(
        pl.col("step").max().alias("steps"),
        pl.col("success").max().alias("success"),
        pl.col("reward").sum().alias("total_reward"),
        pl.col("z_norm").min().alias("min_z_norm"),
        pl.col("z_norm").last().alias("final_z_norm"),
        pl.col("gene_symbol").first().alias("first_action"),
    )

    n_episodes = int(by_ep.height)
    successes = int(by_ep["success"].sum())
    failures = n_episodes - successes
    success_rate = successes / max(n_episodes, 1)
    mean_steps = float(by_ep["steps"].mean())
    mean_total_reward = float(by_ep["total_reward"].mean())
    mean_final_distance = float(by_ep["final_z_norm"].mean())
    mean_min_distance = float(by_ep["min_z_norm"].mean())

    # NO-OP first-action failures: first_action == "NO_OP" AND success == False
    noop_first_failures = int(
        by_ep.filter(
            (pl.col("first_action") == "NO_OP") & (~pl.col("success"))
        ).height
    )
    noop_first_count = int(
        by_ep.filter(pl.col("first_action") == "NO_OP").height
    )

    # Action frequency (use the canonical action_freq.json if present, else compute)
    af = _safe_load_json(action_freq_path)
    if af is None:
        af_polars = df.group_by("gene_symbol").len().sort("len", descending=True)
        af = {row["gene_symbol"]: int(row["len"]) for row in af_polars.iter_rows(named=True)}

    top_actions = sorted(af.items(), key=lambda kv: -kv[1])[:10]
    top_actions_list = [{"gene_symbol": k, "count": int(v)} for k, v in top_actions]

    return {
        "n_episodes": n_episodes,
        "successes": successes,
        "failures": failures,
        "success_rate": float(success_rate),
        "mean_steps": mean_steps,
        "mean_total_reward": mean_total_reward,
        "mean_final_distance": mean_final_distance,
        "mean_min_distance": mean_min_distance,
        "noop_first_action_count": noop_first_count,
        "noop_first_action_failures": noop_first_failures,
        "noop_first_action_rate": float(noop_first_count / max(n_episodes, 1)),
        "action_entropy_nats": _shannon_entropy(af),
        "action_entropy_normalized": _entropy_normalized(af),
        "top_actions": top_actions_list,
    }


def _format_markdown(
    run_summary: dict[str, Any],
    run_meta: dict[str, Any] | None,
    random_summary: dict[str, Any] | None,
    random_meta: dict[str, Any] | None,
    delta: dict[str, Any] | None,
    run_dir: Path,
    random_dir: Path | None,
) -> str:
    """Render a small, defense-ready Markdown block."""
    lines: list[str] = []
    lines.append(f"# RL run summary — `{run_dir.name}`")
    lines.append("")
    if run_meta is not None:
        lines.append("## Provenance")
        lines.append("")
        eps_src = run_meta.get("epsilon_source")
        eps_val = run_meta.get("epsilon_value")
        gate_passed = run_meta.get("dynamics_gate_passed")
        gate_overridden = run_meta.get("dynamics_gate_overridden")
        msd = run_meta.get("min_start_distance")
        nl = run_meta.get("vae_n_latent")
        ckpt = run_meta.get("dynamics_checkpoint")
        ckpt_sha = run_meta.get("dynamics_checkpoint_sha256")
        det = run_meta.get("deterministic_eval")
        git = run_meta.get("git_commit")
        ts = run_meta.get("timestamp_utc")
        lines.append(f"- timestamp: `{ts}`")
        lines.append(f"- git_commit: `{git}`")
        lines.append(f"- vae_n_latent: {nl}")
        lines.append(f"- epsilon: {eps_val}  (source: `{eps_src}`)")
        lines.append(f"- min_start_distance: `{msd}`")
        lines.append(f"- dynamics_checkpoint: `{ckpt}`")
        if ckpt_sha:
            lines.append(f"- dynamics_checkpoint_sha256: `{ckpt_sha[:16]}…`")
        lines.append(
            f"- dynamics_gate_passed: **{gate_passed}**  (overridden: **{gate_overridden}**)"
        )
        lines.append(f"- deterministic_eval: {det}")
        lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| n_episodes | {run_summary['n_episodes']} |")
    lines.append(f"| success_rate | **{run_summary['success_rate']:.3f}** |")
    lines.append(f"| successes / failures | {run_summary['successes']} / {run_summary['failures']} |")
    lines.append(f"| mean_steps | {run_summary['mean_steps']:.2f} |")
    lines.append(f"| mean_total_reward | {run_summary['mean_total_reward']:.3f} |")
    lines.append(f"| mean_final_distance | {run_summary['mean_final_distance']:.3f} |")
    lines.append(f"| mean_min_distance | {run_summary['mean_min_distance']:.3f} |")
    lines.append(
        f"| NO-OP first-action (count / failures / rate) | "
        f"{run_summary['noop_first_action_count']} / "
        f"{run_summary['noop_first_action_failures']} / "
        f"{run_summary['noop_first_action_rate']:.3f} |"
    )
    lines.append(
        f"| action_entropy (nats / normalized) | "
        f"{run_summary['action_entropy_nats']:.3f} / "
        f"{run_summary['action_entropy_normalized']:.3f} |"
    )
    lines.append("")

    if run_summary["top_actions"]:
        lines.append("### Top actions")
        lines.append("")
        lines.append("| rank | gene_symbol | count |")
        lines.append("| --- | --- | --- |")
        for i, item in enumerate(run_summary["top_actions"], 1):
            lines.append(f"| {i} | {item['gene_symbol']} | {item['count']} |")
        lines.append("")

    if random_summary is not None and delta is not None:
        lines.append("## Comparison to random baseline")
        lines.append("")
        if random_dir is not None:
            lines.append(f"- random run: `{random_dir}`")
        if random_meta is not None:
            kind = random_meta.get("policy_kind")
            if kind:
                lines.append(f"- random policy_kind: `{kind}`")
        lines.append("")
        lines.append("| metric | run | random | Δ (run − random) |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(
            f"| success_rate | {run_summary['success_rate']:.3f} | "
            f"{random_summary['success_rate']:.3f} | "
            f"**{delta['delta_success_rate']:+.3f}** "
            f"({delta['delta_success_pp']:+.1f} pp) |"
        )
        lines.append(
            f"| mean_steps | {run_summary['mean_steps']:.2f} | "
            f"{random_summary['mean_steps']:.2f} | "
            f"{delta['delta_mean_steps']:+.2f} |"
        )
        lines.append(
            f"| mean_final_distance | {run_summary['mean_final_distance']:.3f} | "
            f"{random_summary['mean_final_distance']:.3f} | "
            f"{delta['delta_mean_final_distance']:+.3f} |"
        )
        lines.append("")
        lines.append(
            f"**Headline:** policy success rate is "
            f"{delta['delta_success_pp']:+.1f} percentage points relative to the random baseline."
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize an RL evaluation run (PPO eval or random-policy baseline)."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing rollouts.parquet (+ action_freq.json, metadata.json).",
    )
    parser.add_argument(
        "--random-baseline-dir",
        default=None,
        help="Optional directory of a random-policy baseline run with the same env settings.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to a Markdown summary (default: <run-dir>/summary.md).",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip Markdown output (only write summary.json).",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    rollouts = run_dir / "rollouts.parquet"
    action_freq = run_dir / "action_freq.json"
    metadata = run_dir / "metadata.json"

    if not rollouts.exists():
        print(f"ERROR: {rollouts} not found.", file=sys.stderr)
        return 2

    run_summary = _summarize_rollouts(rollouts, action_freq)
    run_meta = _safe_load_json(metadata)

    random_summary = None
    random_meta = None
    random_dir = None
    delta = None
    if args.random_baseline_dir is not None:
        random_dir = Path(args.random_baseline_dir)
        rand_rollouts = random_dir / "rollouts.parquet"
        rand_action_freq = random_dir / "action_freq.json"
        rand_metadata = random_dir / "metadata.json"
        if not rand_rollouts.exists():
            print(
                f"WARN: --random-baseline-dir given but {rand_rollouts} missing; "
                "skipping comparison.",
                file=sys.stderr,
            )
        else:
            random_summary = _summarize_rollouts(rand_rollouts, rand_action_freq)
            random_meta = _safe_load_json(rand_metadata)
            delta = {
                "delta_success_rate": run_summary["success_rate"] - random_summary["success_rate"],
                "delta_success_pp": 100.0
                * (run_summary["success_rate"] - random_summary["success_rate"]),
                "delta_mean_steps": run_summary["mean_steps"] - random_summary["mean_steps"],
                "delta_mean_final_distance": (
                    run_summary["mean_final_distance"] - random_summary["mean_final_distance"]
                ),
            }

    out_json = run_dir / "summary.json"
    summary_blob: dict[str, Any] = {
        "run_dir": str(run_dir),
        "metadata": run_meta,
        "metrics": run_summary,
    }
    if random_summary is not None:
        summary_blob["random_baseline"] = {
            "run_dir": str(random_dir),
            "metadata": random_meta,
            "metrics": random_summary,
        }
        summary_blob["delta_vs_random"] = delta
    with open(out_json, "w") as f:
        json.dump(summary_blob, f, indent=2, default=str)
    print(f"Wrote {out_json}")

    if not args.no_markdown:
        md_path = Path(args.out) if args.out is not None else (run_dir / "summary.md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_text = _format_markdown(
            run_summary, run_meta, random_summary, random_meta, delta, run_dir, random_dir,
        )
        md_path.write_text(md_text)
        print(f"Wrote {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
