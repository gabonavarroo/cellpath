import argparse
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.rl.environment import make_env_factory

parser = argparse.ArgumentParser()
parser.add_argument("--min-start-distance", required=True)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()

repo_root = Path.cwd()
rng = np.random.default_rng(args.seed)

with initialize_config_dir(version_base=None, config_dir=str(repo_root / "config")):
    cfg = compose(
        config_name="default",
        overrides=[
            f"paths.root={repo_root}",
            "rl.train.skip_gate=true",
            f"rl.env.min_start_distance={args.min_start_distance}",
        ],
    )

env = make_env_factory(cfg)()

success = 0
steps = []
final_dists = []

for ep in range(args.episodes):
    obs, info = env.reset(seed=ep)
    terminated = False
    truncated = False
    ep_steps = 0

    while not (terminated or truncated):
        mask = info["action_mask"]
        valid = np.where(mask)[0]
        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        ep_steps += 1

    success += int(info.get("success", False))
    steps.append(ep_steps)
    final_dists.append(info["distance"])

out = {
    "min_start_distance": args.min_start_distance,
    "episodes": args.episodes,
    "success_rate": success / args.episodes,
    "mean_steps": float(np.mean(steps)),
    "mean_final_distance": float(np.mean(final_dists)),
}

print(json.dumps(out, indent=2))
