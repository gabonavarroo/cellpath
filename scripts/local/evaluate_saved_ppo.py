import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from sb3_contrib import MaskablePPO

from src.rl.environment import make_env_factory
from src.rl.train_ppo import evaluate_policy

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--min-start-distance", default="8.0")
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--deterministic", action="store_true")
args = parser.parse_args()

repo_root = Path.cwd()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

with initialize_config_dir(version_base=None, config_dir=str(repo_root / "config")):
    cfg = compose(
        config_name="default",
        overrides=[
            f"paths.root={repo_root}",
            "rl.train.skip_gate=true",
            f"rl.env.min_start_distance={args.min_start_distance}",
        ],
    )

with open_dict(cfg):
    cfg.paths.rl_dir = str(out_dir)
    cfg.paths.rl_ppo_zip = str(out_dir / "ppo.zip")
    cfg.paths.rl_rollouts_parquet = str(out_dir / "rollouts.parquet")
    cfg.paths.rl_action_freq_json = str(out_dir / "action_freq.json")
    cfg.paths.rl_success_curves_png = str(out_dir / "success_curves.png")
    cfg.rl.eval.deterministic = bool(args.deterministic)
    cfg.rl.eval.n_rollout_episodes = int(args.episodes)

env = make_env_factory(cfg)()
model = MaskablePPO.load(args.model, device="cpu")

metrics = evaluate_policy(
    model,
    env,
    n_episodes=args.episodes,
    deterministic=bool(args.deterministic),
    cfg=cfg,
)

print(metrics)
print(f"Saved deterministic={args.deterministic} rollouts to {out_dir}")
