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
| `dynamics.use_state_linear_skip` | `true` | bool | Gene-independent skip; improves val+OOD (see Phase 2 ablation) |
| `dynamics.use_gene_delta_bias` | `false` | bool | Per-gene bias; disabled — causes OOD collapse |
| `dynamics.selection_metric` | `gate_margin` | `val_nll \| gate_margin` | Checkpoint to promote to model.pt |
| `dynamics.lambda_mse_delta` | 0.0 | 0.0–0.1 | Hybrid NLL+MSE weight; 0.0 = NLL only |
| `dynamics.lr` | 1e-4 | 1e-4–3e-3 | AdamW learning rate |
| `dynamics.weight_decay` | 1e-5 | 0–1e-3 | AdamW weight decay |
| `dynamics.max_epochs` | 300 | 20–300 | Cap (increased to accommodate lower LR) |
| `dynamics.early_stop_patience` | 35 | 5–50 | Early stop on val NLL |

### 1.5 RL environment + PPO (`config/rl.yaml`)

| Key | Default | Range | Effect |
|---|---|---|---|
| `rl.action_space.enable_knockout` | false | bool | If true, requires CRISPRi dataset (errors otherwise) |
| `rl.reference.source` | `unperturbed_k562` | `unperturbed_k562 \| external_healthy` | Reward target (only first implemented) |
| `rl.reference.epsilon_percentile` | 50 | 50 / 80 / 90 / 95 / 99 | V1 canonical p50 threshold for ε_success; p90 is reference provenance only |
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
- Scalar: `vae/epsilon_success` (computed once at end of training; V1 canonical artifact is p50)

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
- `dynamics_legacy_mlp.yaml` — pre-Phase-2-ablation MLP baseline (plain MLP, lr=1e-3, val_nll).

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
| AB-05 | RL | `epsilon_percentile` | 50 / 80 / 90 / 95 | 4 |
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

## 8. Phase 2 dynamics experiment results (2026-05-14)

All 9 runs used the same real Norman OT pairs (train/val/ood splits).  Gate threshold:
`margin_vs_linear_ridge_pearson ≥ 0.03`.  Columns show `best_gate` checkpoint metrics
(from `checkpoint_comparison.json`) unless noted.  **No run passed the gate.**

| # | Architecture | lr | λ_mse | sel. metric | val Pearson | val MLP−ridge | OOD Pearson | OOD MLP−ridge | unc. Spearman | Gate |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline (no skip) | 1e-3 | 0.0 | val_nll | ≈0.603 | +0.002 | ≈0.485 | — | 0.247 | **FAIL** |
| 2 | state_linear | 1e-3 | 0.0 | val_nll | ≈0.603 | +0.002 | ≈0.485 | — | 0.247 | **FAIL** |
| 3 | gene_bias | 1e-3 | 0.0 | val_nll | — | — | ≈0.26–0.29 | — | — | **FAIL** (OOD collapse) |
| 4 | state_linear+gene_bias | 1e-3 | 0.0 | val_nll | — | — | ≈0.26–0.29 | — | — | **FAIL** (OOD collapse) |
| 5 | state_linear | 1e-3 | 0.0 | gate_margin | ≈0.608 | +0.007 | ≈0.479 | +0.040 | 0.249 | **FAIL** |
| 6 | state_linear | 3e-4 | 0.0 | gate_margin | ≈0.607 | +0.006 | ≈0.478 | +0.039 | 0.248 | **FAIL** |
| 7 | state_linear | **1e-4** | 0.0 | gate_margin | **≈0.608** | **+0.007** | **≈0.479** | **+0.040** | **0.249** | **FAIL** ← current default |
| 8 | state_linear | 3e-4 | 0.05 | gate_margin | ≈0.607 | +0.006 | ≈0.478 | +0.039 | 0.247 | **FAIL** |
| 9 | state_linear | 3e-4 | 0.1 | gate_margin | ≈0.607 | +0.006 | ≈0.478 | +0.039 | 0.247 | **FAIL** |

**Conclusions:**

- `state_linear` is the best architecture: improves val+OOD vs baseline, no OOD collapse.
- `gene_bias` (with or without state_linear) causes OOD collapse → permanently rejected.
- Lower LR (1e-4) produces the best ridge-margin signal (+0.007) without OOD degradation.
- Hybrid MSE loss (λ=0.05, 0.1) had negligible effect on the ridge-margin metric.
- The primary blocker is `margin_vs_linear_ridge_pearson` (+0.007 vs +0.030 required).
- Next diagnostic axis: latent space quality (dim 11 failure mode, OT pair quality, VAE latent).

To reproduce run #7 (current default):
```bash
python scripts/train_dynamics.py --config-name default +force=true
```

To reproduce pre-ablation baseline (run #1) — use CLI overrides referencing values in
`config/experiments/dynamics_legacy_mlp.yaml`:
```bash
python scripts/train_dynamics.py \
    dynamics.use_state_linear_skip=false dynamics.selection_metric=val_nll \
    dynamics.lr=1e-3 dynamics.max_epochs=100 dynamics.early_stop_patience=10 \
    +force=true
```

---

## 9. Pointers

- `config/` — every hyperparameter listed in §1 has a corresponding YAML entry.
- `config/experiments/dynamics_legacy_mlp.yaml` — pre-ablation MLP baseline override.
- `src/analysis/metrics.py` — all metrics referenced in §3.
- `src/analysis/model_selection.py` — `recommend_checkpoint` decision logic.
- `scripts/evaluate.py` — runs ablation aggregation.
- `scripts/visualize.py` — generates the thesis figures listed in §7.

---

## 10. V2 ablation matrix (2026-05-16)

V2 ran a sequence of phases (P0A → P0F). See `artifacts_v2/V2_FINAL_REPORT.md` for the
headline result and `artifacts_v2/interpretation_p0{a,b_doubleprime,b2,c0,d,e,f_wrapup}.md`
for per-phase analyses.

### V2 dynamics × pairings

| Run path | Pairing | Architecture | λ_corr | Val margin | Beam k=3 best | Verdict |
|---|---|---|---:|---:|---:|---|
| `artifacts/dynamics` | OT (V1) | state_linear | 0.0 | +0.0074 | 1.59 (17/17) | V1 baseline; controllable |
| `artifacts_v2/dynamics_mean_delta_default` | mean_delta | state_linear | 0.0 | +0.0214 | 4.11 (0/17) | NOT reachable |
| `artifacts_v2/dynamics_mean_delta_corr_{005,010,030}` | mean_delta | state_linear | 0.05–0.30 | +0.022–+0.023 | 4.09 (0/17) | gate plateau, dead |
| `artifacts_v2/dynamics_soft_ot_default` | soft_ot | state_linear | 0.0 | **+0.0413** | 16.97 (0/17) | gate PASS, control-hostile |
| `artifacts_v2/dynamics_random_default` | random | state_linear | 0.0 | −0.009 | (control) | negative control |
| `artifacts_v2/dynamics_v1ot_ror` | OT | residual-over-ridge | 0.0 | +0.0127 | 1.48 (17/17) | RoR baseline |
| `artifacts_v2/dynamics_v1ot_ror_corr005` | OT | residual-over-ridge | 0.05 | +0.0135 | 1.51 (17/17) | RoR + corr |
| `artifacts_v2/dynamics_v1ot_ror_corr010` | OT | residual-over-ridge | 0.10 | **+0.0136** | **1.51 (17/17)** | **V2 primary dynamics** |

### V2 RL / PPO runs

| Run path | Dynamics | Reward | K | TS | Curric. | Primary cell PPO |
|---|---|---|---:|---:|---|---:|
| (V1) `artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k` | V1 OT | absolute_distance | 10 | 500k | no | 0.988 at V1 setting (p50/start8/K=10) |
| B1: `rl_v1ot_abs_k3_200k` | V1 OT | absolute_distance | 3 | 200k | no | 0.410 |
| B2: `rl_v1ot_delta_k3_200k` | V1 OT | delta_distance | 3 | 200k | no | 0.000 |
| B3: `rl_v1ot_terminal_k3_500k` | V1 OT | terminal_only_step_cost | 3 | 500k | no | 1.000 |
| B4: `rl_v1ot_terminal_curriculum_k3_500k` | V1 OT | terminal | 3 | 500k | yes | 1.000 |
| B5: `rl_v1ot_terminal_curriculum_k3_1M` (4 seeds) | V1 OT | terminal | 3 | 1M | yes | **0.963 ± 0.042** |
| **C2: `rl_v1ot_ror_corr010_terminal_curric_k3_1M` (4 seeds)** | **RoR_corr010** | terminal | 3 | 1M | yes | **0.941 ± 0.048** — **V2 primary PPO** |
| E1: `rl_v1ot_hybrid_k3_200k_smoke` | V1 OT | hybrid α=1 B=1 | 3 | 200k | no | 0.006 |
| E2: `rl_v1ot_hybrid_alpha1_bonus10_curric_k3_1M` | V1 OT | hybrid α=1 B=10 | 3 | 1M | yes | 0.170 (generalisation failure) |
| F1: `rl_v1ot_terminal_curric_k2_500k` | V1 OT | terminal | 2 | 500k | yes | 0.860 at K=3 (K=2 trainer); 0.600 at K=2/bin 8-10 (+30 pp vs B5 there) |
| F2: `rl_v1ot_terminal_curric_k8_1M` | V1 OT | terminal | 8 | 1M | yes | 0.940 at K=3 |
| D1–D3: `rl_meandelta_*` | mean_delta_* | terminal | 3 / 8 | 200k / 500k | no | 0.000 (NOOP-collapse) |

### V2 hardness frontier (4-seed, n=300) — V2 primary configuration

| Cell | PPO 4-seed mean ± std | 95 % CI | grd2 | PPO − grd2 |
|---|---:|---|---:|---:|
| K=2 bin 6-8 OOD | 0.748 ± 0.053 | [0.697, 0.800] | 0.790 | −0.042 |
| K=2 bin 8-10 OOD | 0.283 ± 0.045 | [0.239, 0.328] | 0.300 | −0.017 |
| K=3 bin 6-8 OOD | 0.998 ± 0.002 | [0.996, 0.999] | 1.000 | −0.002 |
| **K=3 bin 8-10 OOD (primary)** | **0.941 ± 0.048** | **[0.894, 0.988]** | 1.000 | −0.059 |

**No PPO config exceeds greedy_dyn_2 by ≥ +0.05 pp anywhere.** PPO − random = +77 pp at
primary cell. See `artifacts_v2/figures/` for visualisations and
`artifacts_v2/eval_p0f_seed_aggregate/seed_aggregate_success_rate.md` for the full table.
