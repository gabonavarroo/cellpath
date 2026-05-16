import argparse
import json
from pathlib import Path

import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("--rl-dir", required=True)
args = parser.parse_args()

rl_dir = Path(args.rl_dir)
df = pl.read_parquet(rl_dir / "rollouts.parquet")
action_freq = json.loads((rl_dir / "action_freq.json").read_text())

by_ep = df.group_by("episode_id").agg(
    pl.col("step").max().alias("steps"),
    pl.col("success").max().alias("success"),
    pl.col("reward").sum().alias("total_reward"),
    pl.col("z_norm").min().alias("min_z_norm"),
    pl.col("z_norm").last().alias("final_z_norm"),
    pl.col("gene_symbol").first().alias("first_action"),
)

print("\nEpisode summary:")
print({
    "episodes": by_ep.height,
    "success_rate": by_ep["success"].mean(),
    "mean_steps": by_ep["steps"].mean(),
    "mean_total_reward": by_ep["total_reward"].mean(),
    "mean_final_z_norm": by_ep["final_z_norm"].mean(),
    "mean_min_z_norm": by_ep["min_z_norm"].mean(),
})

print("\nSteps distribution:")
print(by_ep.group_by("steps").agg(
    pl.len().alias("episodes"),
    pl.col("success").mean().alias("success_rate"),
    pl.col("final_z_norm").mean().alias("mean_final_z_norm"),
).sort("steps"))

print("\nTop first actions:")
print(by_ep.group_by("first_action").agg(
    pl.len().alias("episodes"),
    pl.col("success").mean().alias("success_rate"),
    pl.col("final_z_norm").mean().alias("mean_final_z_norm"),
).sort("episodes", descending=True).head(20))

print("\nTop action_freq:")
for gene, count in sorted(action_freq.items(), key=lambda x: -x[1])[:20]:
    print(f"{gene:20s} {count}")
