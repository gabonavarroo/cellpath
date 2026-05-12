# EXPERIMENTS.md — hyperparameters, naming, logging, and the ablation matrix

This file documents every tunable hyperparameter across CellPath's four components, how to run
experiments with Hydra, and the ablation matrix planned for the 14-day budget.

---

## 1. Tunable hyperparameters

### 1.1 Data / Preprocessing (`config/default.yaml::data`)

| Key | Default | Range | Effect |
|---|---|---|---|
| `data.min_counts` | 500 | 100–2000 | Min UMIs per cell — defensive QC |
| `data.min_cells` | 10 | 5–50 | Min cells expressing a gene |
| `data.n_hvg` | 2000 | 1000–5000 | HVG count fed into scVI |
| `data.hvg_flavor` | `seurat_v3` | `seurat_v3 \| seurat \| cell_ranger` | HVG algorithm (only `seurat_v3` uses raw counts) |
| `data.normalize_total` | 10000 | 5000–20000 | Library-size target for visualization branch |

### 1.2 scVI VAE (`config/vae.yaml`)

| Key | Default | Range | Effect |
|---|---|---|---|
| `vae.n_latent` | 32 | 16 / 32 / 64 | Latent dim — ablation primary axis |
| `vae.n_layers` | 2 | 1–3 | Encoder/decoder depth |
| `vae.n_hidden` | 128 | 64–256 | Hidden width |
| `vae.dropout_rate` | 0.1 | 0.0–0.3 | Dropout in encoder/decoder |
| `vae.gene_likelihood` | `nb` | `nb \| zinb \| poisson` | NB recommended for 10x v3 |
| `vae.dispersion` | `gene` | `gene \| gene-cell` | Per-gene θ (default) is sufficient |
| `vae.max_epochs` | 400 | 100–800 | Cap on training |
| `vae.early_stopping` | true | bool | Monitor val ELBO |
| `vae.batch_size` | 128 | 64–512 | scvi default |
| `vae.lr` | 1e-3 | 1e-4–5e-3 | scvi default |

### 1.3 Pairing (`config/default.yaml::pairing`)

| Key | Default | Range | Effect |
|---|---|---|---|
| `pairing.method` | `ot` | `ot \| random \| mean_delta` | Pairing algorithm |
| `pairing.ot_epsilon` | 0.05 | 0.01–0.5 | Sinkhorn regularization |
| `pairing.ot_iter` | 500 | 100–2000 | Sinkhorn max iters |
| `pairing.normalize_cost` | true | bool | Divide cost matrix by median |
| `pairing.val_cell_fraction` | 0.10 | 0.05–0.20 | Within-perturbation held-out cells |
| `pairing.ood_gene_fraction` | 0.20 | 0.10–0.30 | Held-out genes (OOD report) |
| `pairing.combo_held_out_fraction` | 0.20 | 0.10–0.30 | Held-out combinations |
| `pairing.seed` | 42 | int | RNG seed for splits |

### 1.4 Dynamics (`config/dynamics.yaml`)

| Key | Default | Range | Effect |
|---|---|---|---|
| `dynamics.d_emb` | 64 | 16–128 | Gene-embedding dim |
| `dynamics.n_hidden` | 256 | 128–512 | MLP width |
| `dynamics.n_layers` | 3 | 2–5 | MLP depth |
| `dynamics.activation` | `silu` | `silu \| relu \| gelu` | Activation function |
| `dynamics.dropout` | 0.1 | 0.0–0.3 | MLP dropout |
| `dynamics.log_var_min` | -5.0 | -10–0 | Clamp on log σ² |
| `dynamics.log_var_max` | 3.0 | 0–6 | Clamp on log σ² |
| `dynamics.lambda_combo` | 0.5 | 0.0–1.0 | Weight on composition loss |
| `dynamics.batch_size` | 256 | 64–1024 | Mini-batch size |
| `dynamics.lr` | 1e-3 | 1e-4–3e-3 | AdamW learning rate |
| `dynamics.weight_decay` | 1e-5 | 0–1e-3 | AdamW weight decay |
| `dynamics.max_epochs` | 100 | 20–300 | Cap |
| `dynamics.early_stop_patience` | 10 | 5–30 | Early stop on val NLL |

### 1.5 RL environment + PPO (`config/rl.yaml`)

| Key | Default | Range | Effect |
|---|---|---|---|
| `rl.action_space.enable_knockout` | false | bool | If true, requires CRISPRi dataset (errors otherwise) |
| `rl.reference.source` | `unperturbed_k562` | `unperturbed_k562 \| external_healthy` | Reward target (only first implemented) |
| `rl.reference.epsilon_percentile` | 90 | 80 / 90 / 95 / 99 | Percentile of ‖z_ctrl − z_ref‖ used for ε_success |
| `rl.env.max_steps` | 10 | 5 / 10 / 20 | Episode budget K |
| `rl.env.start_state` | `random_perturbation` | `random_perturbation \| specific \| uniform_latent` | Starting state strategy |
| `rl.env.repeat_mask` | true | bool | Mask already-used genes |
| `rl.env.distance_metric` | `l2` | `l2 \| cosine` | Distance to z_ref |
| `rl.reward.lambda_sparse` | 0.05 | 0.0 / 0.01 / 0.05 / 0.1 / 0.2 | Per-action sparsity penalty |
| `rl.reward.lambda_unc` | 0.0 | 0.0 / 0.05 / 0.1 | Per-step uncertainty penalty |
| `rl.reward.success_bonus` | 0.0 | 0.0–5.0 | Optional terminal bonus on success |
| `rl.ppo.total_timesteps` | 2_000_000 | 5e5–5e6 | Total env steps |
| `rl.ppo.n_envs` | 16 | 4–64 | Parallel envs |
| `rl.ppo.n_steps` | 1024 | 256–4096 | Steps per rollout |
| `rl.ppo.batch_size` | 256 | 64–1024 | PPO minibatch |
| `rl.ppo.n_epochs` | 10 | 4–20 | PPO inner epochs |
| `rl.ppo.lr` | 3e-4 | 1e-4–1e-3 | PPO learning rate |
| `rl.ppo.gamma` | 0.99 | 0.95–0.999 | Discount |
| `rl.ppo.gae_lambda` | 0.95 | 0.9–0.99 | GAE λ |
| `rl.ppo.clip_range` | 0.2 | 0.1–0.3 | PPO clip ε |
| `rl.ppo.ent_coef` | 0.01 | 0.001–0.1 | Entropy coefficient |
| `rl.ppo.vf_coef` | 0.5 | 0.25–1.0 | Value function coefficient |

### 1.6 Global

| Key | Default | Effect |
|---|---|---|
| `seed` | 42 | Global RNG seed (Numpy/Torch/Python/Gym) |
| `device.force` | null | If set, overrides `get_device()` (e.g. "cpu" for CI) |
| `log.level` | `INFO` | rich logger level |
| `log.tensorboard_dir` | `${paths.artifacts}/tensorboard` | tb logdir |
| `log.use_wandb` | false | Opt-in W&B |

---

## 2. Experiment naming convention

Hydra automatically writes each run under `outputs/<date>/<time>/`. We also enforce a stable
human-readable experiment ID for cross-run comparison:

```
exp_<component>_<key>=<value>_<key>=<value>_..._YYYYMMDD
```

Examples:
```
exp_vae_n_latent=16_layers=2_likelihood=nb_20260101
exp_dynamics_pairing=ot_lambda_combo=0.5_20260103
exp_rl_lambda_sparse=0.05_K=10_seed=42_20260108
```

This ID is stored in every output artifact's metadata.json. Engineers MUST NOT manually rename
output directories — Hydra's `outputs/<date>/<time>/` structure is preserved; the human-readable
ID lives inside the artifacts.

---

## 3. What to log

Every training script logs the following to TensorBoard from **epoch 1**:

### VAE
- `vae/elbo_train`, `vae/elbo_val` — reconstruction quality
- `vae/kl_local`, `vae/kl_global`
- `vae/reconstruction_loss`
- Histograms: `vae/z_mean`, `vae/z_var` (post-warmup)
- Scalar: `vae/library_size_mean` (sanity)
- Scalar: `vae/epsilon_success` (computed once at end of training)

### Dynamics
- `dyn/loss_total`, `dyn/loss_nll`, `dyn/loss_combo`
- `dyn/val_r2_primary`, `dyn/val_pearson_primary`
- `dyn/val_r2_ood`, `dyn/val_pearson_ood`
- `dyn/log_var_mean`, `dyn/log_var_std`
- `dyn/spearman_uncertainty_calibration`
- Gate snapshot (logged at end): `dyn/gate_passed` (0/1), per-baseline metrics

### RL
- `rl/ep_rew_mean`, `rl/ep_rew_min`, `rl/ep_rew_max`
- `rl/success_rate`
- `rl/episode_length_mean`
- `rl/action_entropy`
- `rl/value_loss`, `rl/policy_loss`, `rl/approx_kl`
- Histograms: `rl/action_distribution` (top-K genes used)

### Artifacts written
- All hyperparameters as `<artifact_dir>/config.yaml` (Hydra dump).
- Model weights (VAE via `model.save()`, dynamics via state dict, PPO via `model.save()`).
- Metric dumps as JSON for easy aggregation across runs.

---

## 4. Hydra usage

We use Hydra's compose API everywhere. The master config is `config/default.yaml`:

```yaml
defaults:
  - paths
  - vae
  - dynamics
  - rl
  - _self_

seed: 42
device:
  force: null
log:
  level: INFO
  tensorboard_dir: ${paths.artifacts}/tensorboard
  use_wandb: false
```

### 4.1 CLI overrides

```bash
# Single override
python scripts/train_vae.py vae.n_latent=64

# Composition with experiment-specific config
python scripts/train_vae.py --config-name vae_ablation

# Multiple overrides + custom output dir
python scripts/train_vae.py vae.n_latent=64 vae.max_epochs=200 \
    hydra.run.dir=outputs/manual_run
```

### 4.2 Experiment configs (`config/experiments/`)

These are pre-baked override stacks:
- `baseline.yaml` — all defaults; the first run on Day 0.
- `vae_ablation.yaml` — runs n_latent ∈ {16, 32, 64} via Hydra multirun.
- `rl_sparse.yaml` — runs λ_sparse ∈ {0.01, 0.05, 0.1}.

Multirun:
```bash
python scripts/train_vae.py --multirun --config-name vae_ablation
```

---

## 5. Baselines (Day 0–1 sanity runs)

Before any ablation, run these baselines exactly once and commit the metric values to
`PROGRESS.md`:

| Baseline | Purpose | Expected outcome |
|---|---|---|
| `make pipeline CONFIG=baseline --dry-run` | Pipeline wiring sanity | exit 0 |
| Train mock-data dynamics for 100 steps | Optimizer + loss sanity | loss decreases monotonically |
| `tests/test_integration.py` | Full mock pipeline | passes |
| VAE on mock data (100 cells × 200 genes) | scVI setup_anndata + train smoke | ELBO finite, no NaN |
| Untrained PPO over mock env | Gym compliance | env_checker passes, random policy runs 1000 steps |

If any of these fail, halt and fix before starting Phase 1 experiments.

---

## 6. Ablation matrix (14-day budget)

We have time for ~10–15 distinct runs given the cluster availability. The matrix below is
**ordered by priority** — top entries must be run; bottom entries are stretch.

### P0 — required runs

| ID | Component | Variable | Values | Total runs |
|---|---|---|---|---|
| AB-01 | VAE | `n_latent` | 16 / 32 / 64 | 3 |
| AB-02 | Pairing | `pairing.method` | ot / random / mean_delta | 3 |
| AB-03 | RL | `lambda_sparse` | 0.0 / 0.01 / 0.05 / 0.1 / 0.2 | 5 |

Total P0: **11 runs**.

### P1 — strongly recommended

| ID | Component | Variable | Values | Total runs |
|---|---|---|---|---|
| AB-04 | RL | `K` (max steps) | 5 / 10 / 20 | 3 |
| AB-05 | RL | `epsilon_percentile` | 80 / 90 / 95 | 3 |
| AB-06 | Dynamics | `lambda_combo` | 0.0 / 0.5 / 1.0 | 3 |

Total P0 + P1: **20 runs**.

### P2 — stretch (only if time permits in Phase 5)

| ID | Component | Variable | Values | Total runs |
|---|---|---|---|---|
| AB-07 | VAE | `gene_likelihood` | nb / zinb / poisson | 3 |
| AB-08 | RL | algorithm | MaskablePPO vs SAC-Discrete | 2 |
| AB-09 | Dynamics | architecture | Residual MLP vs vanilla MLP | 2 |
| AB-10 | RL | `lambda_unc` | 0.0 / 0.05 / 0.1 (uncertainty penalty) | 3 |

### Ablation hygiene

- Each ablation varies exactly one axis; all other axes are at default.
- Each run is repeated with **3 seeds** for the final P0 results (so P0 = 11 × 3 = 33 wall-clock
  runs minimum). P1/P2 may be single-seed if compute is tight.
- All ablation results are stored in `artifacts/eval/ablations/<id>/<seed>/metrics.json` and
  aggregated by `scripts/evaluate.py --aggregate-ablations`.

---

## 7. What to report in the thesis

The thesis defense slides and PDF must include, at minimum:

1. **VAE quality.** ELBO curve, UMAP of latent with perturbation labels, silhouette score.
2. **Dynamics gate.** A table showing MLP vs every baseline, primary gate metrics, OOD report.
3. **RL success.** Success rate curve over training. Final success rate on held-out starting
   states. Mean number of interventions per successful episode.
4. **DepMap enrichment.** Hypergeometric p-values for each gene-set panel, null comparison
   z-scores.
5. **Ablation table.** P0 ablations as a single table with each row = ablation cell, each
   column = (final RL success rate, final dynamics R², ε_success used).
6. **Honest limitations.** Explicit statement of the cross-sectional / Markov caveats per
   ARCHITECTURE.md Concept 7.

Any plot in the thesis must be reproducible via `scripts/visualize.py --figure <name>`.

---

## 8. Pointers

- `config/` — every hyperparameter listed in §1 has a corresponding YAML entry.
- `src/analysis/metrics.py` — all metrics referenced in §3.
- `scripts/evaluate.py` — runs ablation aggregation.
- `scripts/visualize.py` — generates the thesis figures listed in §7.
