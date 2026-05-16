"""P0A.5 V2 biology rerank on existing frozen rollout action frequencies."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.depmap_validation import load_depmap_k562, load_gene_panels, preranked_gsea
from src.analysis.metrics import action_freq_chronos_spearman


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _read_freq(path: str | Path) -> dict[str, int]:
    with open(path) as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def _load_action_universe(path: str | Path) -> list[str]:
    with open(path) as f:
        return [str(g) for g in json.load(f)["genes"]]


def _complete_freq(freq: dict[str, int], genes: list[str]) -> dict[str, int]:
    return {g: int(freq.get(g, 0)) for g in genes}


def _chronos_series(depmap_df: Any) -> pd.Series:
    pdf = depmap_df.to_pandas() if hasattr(depmap_df, "to_pandas") else pd.DataFrame(depmap_df)
    return pd.Series(pdf["chronos"].values, index=pdf["gene_symbol"].astype(str)).dropna()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--rl_dir", required=True)
    parser.add_argument("--action_freq_ppo_det", required=True)
    parser.add_argument("--action_freq_random", required=True)
    parser.add_argument("--action_freq_ppo_stoch", default=None)
    parser.add_argument("--depmap_csv", required=True)
    parser.add_argument("--gene_vocab", default="artifacts/vae/gene_vocab.json")
    parser.add_argument("--panel_dir", default="data/panels")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--n_perm", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    summary_path = out_prefix.parent / f"{out_prefix.name}_summary.json"
    metadata_path = out_prefix.parent / f"{out_prefix.name}_metadata.json"

    depmap = load_depmap_k562(args.depmap_csv)
    chronos = _chronos_series(depmap)
    action_universe = _load_action_universe(args.gene_vocab)
    panels = load_gene_panels(args.panel_dir)

    stoch_path = args.action_freq_ppo_stoch
    if stoch_path is None:
        candidate = Path(args.rl_dir) / "eval_stochastic" / "action_freq.json"
        stoch_path = str(candidate) if candidate.exists() else None

    policies = {
        "ppo_det": _complete_freq(_read_freq(args.action_freq_ppo_det), action_universe),
        "random": _complete_freq(_read_freq(args.action_freq_random), action_universe),
    }
    if stoch_path is not None and Path(stoch_path).exists():
        policies["ppo_stoch"] = _complete_freq(_read_freq(stoch_path), action_universe)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "stage": "p0a_biology_v2",
        "note": (
            "DepMap V2 rerank is a plausibility check only. Norman is CRISPRa and DepMap "
            "Chronos is loss-of-function; negative evidence is reported directly."
        ),
        "spearman": {},
        "gsea": {},
    }
    for policy, freq in policies.items():
        spearman = action_freq_chronos_spearman(freq, chronos, seed=args.seed, n_boot=args.n_boot)
        summary["spearman"][policy] = spearman
        rows.append({"policy": policy, "test": "action_freq_chronos_spearman", "panel": "", **spearman})

        gsea_rows = preranked_gsea(freq, chronos, panels, n_perm=args.n_perm, seed=args.seed)
        summary["gsea"][policy] = gsea_rows
        for row in gsea_rows:
            rows.append({"policy": policy, "test": "preranked_gsea", **row})

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    metadata = {
        "config_name": args.config_name,
        "git_commit": _git_commit(),
        "read_only_mode": True,
        "source_paths": {
            "rl_dir": args.rl_dir,
            "action_freq_ppo_det": args.action_freq_ppo_det,
            "action_freq_ppo_stoch": stoch_path,
            "action_freq_random": args.action_freq_random,
            "depmap_csv": args.depmap_csv,
            "gene_vocab": args.gene_vocab,
            "panel_dir": args.panel_dir,
        },
        "outputs": {"csv": str(csv_path), "summary": str(summary_path)},
        "n_boot": int(args.n_boot),
        "n_perm": int(args.n_perm),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    det = summary["spearman"].get("ppo_det", {})
    sig_panels = [
        row for row in summary["gsea"].get("ppo_det", [])
        if row.get("q_value") is not None and float(row["q_value"]) <= 0.10
    ]
    print(f"PPO det Spearman rho={det.get('rho'):.4f} p={det.get('p_value'):.4g} n={det.get('n_overlap')}")
    if sig_panels:
        print("GSEA panels q<=0.10:")
        for row in sorted(sig_panels, key=lambda r: r["q_value"]):
            print(f"  {row['panel']}: NES={row['nes']:.3f} p={row['p_value']:.4g} q={row['q_value']:.4g}")
    else:
        print("No PPO det GSEA panels reached q<=0.10.")
    print(f"Wrote {csv_path}, {summary_path}, {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
