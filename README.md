# CellPath

> **In-silico cell-state steering** via a learned scVI latent geometry, a residual
> perturbation-dynamics surrogate, and a MaskablePPO reinforcement-learning agent that
> sequences CRISPRa interventions to drive K562 leukemia cells toward an unperturbed
> reference centroid.

This is **not** a demonstration of therapeutic cancer reprogramming. CellPath learns to
reverse perturbation drift in scVI latent space over a CRISPRa surrogate environment built
from Norman et al. 2019 (GSE133344) Perturb-seq data. The reward target is the
unperturbed-K562 NT-guide centroid; "normal" / "healthy" cell states are explicitly out
of scope (see [`ARCHITECTURE.md`](ARCHITECTURE.md), Concept 7 and [`DATA.md`](DATA.md) §7).

---

## What this project does (today)

CellPath frames cancer cell-state steering as a Markov Decision Process:

| Element | Choice |
|---|---|
| **State** | 32-dim scVI latent of K562 CRISPRa Perturb-seq |
| **Action** | `Discrete(N_genes + 1)`, one per single-gene CRISPRa target (≈106) + NO-OP |
| **Transition** | Learned residual MLP `f_θ(z, gene) → (μ_Δz, log σ²_Δz)` with heteroscedastic NLL |
| **Reward** | distance-to-centroid term + sparsity penalty (+ optional safety / freeband / uncertainty terms, see V3B) |
| **Termination** | NO-OP, success (`‖z − z_ref‖₂ < ε`), or step budget `K` |
| **Policy** | MaskablePPO (sb3-contrib) with per-step gene mask |

The project has gone through four ≈ shippable iterations (V1 → V2 → V3B → "Proposals 1-3").
The full version log is in [`§ Version history`](#version-history). The **current default
Hydra composition** (`config/default.yaml`) reproduces the **V2 primary** result, which is
the last result with multi-seed CIs and an honest written wrap-up
([`artifacts_v2/V2_FINAL_REPORT.md`](artifacts_v2/V2_FINAL_REPORT.md)).

Work after V2 (V3A/B/C and the "Proposal" series) is **research in progress** — there is
no new multi-seed headline beyond V2 yet. See [`PROGRESS.md`](PROGRESS.md) for the latest
session log (2026-05-20: V3C Phase 1).

---

## System diagram

```
  Norman 2019 Perturb-seq (K562 + CRISPRa)
            │
            ▼
   preprocess  ──► HVG selection, raw counts preserved in adata.layers["counts"]
            │
            ▼
        scVI VAE (NB likelihood, 32-dim latent)
            │              │
   z_reference_centroid    ε_success
            │              │
            └───►  OT pseudo-pairing (Sinkhorn)  ◄───  Norman perturbed populations
                           │
                           ▼
       Residual heteroscedastic dynamics MLP (μ, log σ²)
       (RoR_corr010 = residual-over-ridge + correlation loss λ = 0.10)
                           │
                  validation gate + beam reachability probe
                           │
                           ▼
         MaskablePPO over CellReprogrammingEnv
           (terminal-only step-cost reward + distance-bin curriculum, K = 3)
                           │
                           ▼
                    rollouts.parquet
                           │
                           ▼
       DepMap enrichment + trajectory analysis
```

For the full pipeline and 7-concept walk-through, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Quick start (3 commands)

```bash
make setup           # uv venv (.venv) + install all dependencies
make data            # download Norman 2019 + DepMap K562 Chronos
make pipeline        # data → vae → pairs → dynamics → rl → evaluate
```

The pipeline is idempotent — each step skips if its artifact already exists. Pass
`--force <step>` or run `make nuke` to start from scratch.

To validate config + resolved paths without running any compute:

```bash
PYTHONPATH=. python -m src.pipeline run --config-name default --dry-run
```

---

## Installation

Two supported environments:

### 1. Native venv on Mac / Linux (recommended for dev — MPS / CPU)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Why native venv on Apple Silicon: MPS does **not** pass through to Linux containers, so
running inside Docker on a Mac is 20-40× slower than native MPS.

### 2. Cluster GPU (CUDA 12.1, linux/amd64)

```bash
make docker-cuda                                    # build cellpath:cuda
docker compose --profile cuda up training           # runs `python -m src.pipeline run --config-name default`
```

### 3. CI / smoke (CPU)

```bash
make docker-cpu                                     # build cellpath:cpu
docker compose --profile cpu up smoke               # runs the pipeline in --dry-run mode
```

See [`§ Docker`](#docker) below for the full container contract and known caveats.

---

## Datasets

| Dataset | Used for | Source | Local target | Notes |
|---|---|---|---|---|
| Norman et al. 2019 (GSE133344) | scVI training, perturbation pairing, dynamics, RL env | `pertpy.dt.norman_2019()` primary; Zenodo scperturb mirror fallback | `data/raw/norman_2019.h5ad` → `data/processed/norman_hvg.h5ad` | K562 CRISPRa Perturb-seq; ≈111k cells × 19k genes; ≈106 single-gene targets + ≈131 dual-gene combos |
| DepMap Chronos | Biological-plausibility enrichment of RL-selected genes | DepMap public release (default `24Q2`) | `data/raw/depmap_chronos.csv` → `data/processed/depmap_k562_chronos.parquet` | CRISPR knockout fitness, K562 line. Used for enrichment only, **not** therapeutic validation. |

`scripts/download_data.sh` (invoked by `make data`) handles both. Full preprocessing
biology, OT-pairing methodology and the cross-sectional honesty section are in
[`DATA.md`](DATA.md).

### Citations

- Norman TM et al. "Exploring genetic interaction manifolds constructed from rich single-cell
  phenotypes." *Science* 365, 786–793 (2019). doi:10.1126/science.aax4438
- Lopez R et al. "Deep generative modeling for single-cell transcriptomics." *Nature Methods*
  15, 1053–1058 (2018). doi:10.1038/s41592-018-0229-2
- Bunne C et al. "Learning single-cell perturbation responses using neural optimal transport."
  *Nature Methods* 20, 1759–1768 (2023). doi:10.1038/s41592-023-01969-x
- Dempster JM et al. "Chronos: a cell population dynamics model of CRISPR experiments."
  *Genome Biology* 22, 343 (2021). doi:10.1186/s13059-021-02540-7

---

## Current "official" version — V2 primary

| Component | Choice (V2 primary) |
|---|---|
| VAE | 32-dim scVI, NB likelihood (`artifacts/vae/`) |
| Pairing | Entropic OT (Sinkhorn, V1 OT pairs, `artifacts/pairs/`) |
| Dynamics | `RoR_corr010` — residual-over-ridge + correlation loss λ = 0.10 (V2 frozen) |
| RL reward | `terminal_only_step_cost` (mid-step 0; terminal = `1·success − β·t`) |
| Curriculum | distance-bin 4 → 10 over the first 70 % of training |
| Horizon | K = 3 |
| ε success threshold | p25 = 3.166 latent units |
| PPO budget | 1 000 000 timesteps |
| Seeds reported | {42, 0, 1, 7} |

### Headline numbers (V2 hard benchmark, 4-seed mean ± std, n = 300 episodes/cell)

| Cell | PPO (C2 / RoR_corr010) | 95 % CI | random | greedy_dyn_2 | PPO − grd2 |
|---|---:|---|---:|---:|---:|
| **K = 3, bin 8-10, OOD (primary)** | **0.941 ± 0.048** | [0.894, 0.988] | 0.170 | 1.000 | −0.059 |
| K = 3, bin 6-8, OOD | 0.998 ± 0.002 | [0.996, 0.999] | 0.177 | 1.000 | −0.002 |
| K = 2, bin 6-8, OOD (frontier) | **0.748 ± 0.053** | [0.697, 0.800] | 0.070 | 0.790 | −0.042 |
| K = 2, bin 8-10, OOD | 0.283 ± 0.045 | [0.239, 0.328] | 0.020 | 0.300 | −0.017 |

**PPO − random = +77 pp** at the primary cell; mean steps per success ≈ 2.5–2.7 (random
uses ≈ 5.5).

**Honest framing.** PPO matches but does **not** exceed `greedy_dyn_2` (a depth-2
model-based oracle) anywhere on this benchmark. The V2 result is *"PPO compressed a
2-step lookahead into a feedforward controller without runtime model access"*, not *"PPO
discovers a superior strategy"*. The +0.05-pp planning-advantage threshold is not reached
at any cell.

### V2 methodological contribution — gate ⊥ controllability

| Dynamics | Supervised gate (val margin ≥ +0.030) | Beam k = 3 reachability | PPO at primary |
|---|---|---|---|
| V1 OT | FAIL (+0.0074) | PASS (17/17) | PASS (0.963 ± 0.042) |
| **RoR_corr010** (V2 primary) | FAIL (+0.0136) | PASS (17/17) | PASS (0.941 ± 0.048) |
| soft-OT | **PASS (+0.0413)** | **FAIL (0/17)** | FAIL (0.000) |
| mean_delta_corr_030 | FAIL (+0.0232) | FAIL (0/17) | FAIL (0.000) |

The supervised gate is **necessary but not sufficient** for RL controllability: soft-OT
passes the gate yet is control-hostile (every CRISPRa action moves *away* from `z_ref`).
This is V2's main methodological contribution; see
[`artifacts_v2/figures/dynamics_taxonomy.png`](artifacts_v2/figures/dynamics_taxonomy.png).

Full V2 report: [`artifacts_v2/V2_FINAL_REPORT.md`](artifacts_v2/V2_FINAL_REPORT.md).

---

## Required artifacts and where they live

```
artifacts/                # V1 baseline (FROZEN per CLAUDE.md §3)
├── vae/                  # 32-dim scVI checkpoint, latents, centroid, ε
├── pairs/                # OT pairs: train / val / ood / combo .npz
├── dynamics/             # V1 dynamics MLP + gate.json (passed = false; OT noise ceiling)
├── rl/                   # V1 PPO checkpoint (K = 10, p50)
├── rl_hard/              # V1 hard-bench PPO
└── tensorboard/

artifacts_64/             # V1 64-D latent ablation (FROZEN)
artifacts_v2/             # V2 primary + ablations
artifacts_v3/             # V3A/B/C working dir (interpretation md + figures tracked)
artifacts_v2_experiments/ # legacy V2-era experiments (pairs / dynamics ablations)
artifacts_proposal*/      # Proposal 1, 2, 3+5c, and Fusion runs (see Version history)

data/
├── raw/norman_2019.h5ad
├── raw/depmap_chronos.csv
└── processed/{norman_hvg.h5ad, depmap_k562_chronos.parquet}
```

**Important — what is actually on disk right now (audit 2026-05-20):**

| Tier | On disk locally | Source |
|---|---|---|
| `artifacts/` (V1) | ✅ full (vae, pairs, dynamics, rl, rl_hard) | local training |
| `artifacts_64/` | ✅ frozen 64-D ablation | local training |
| `artifacts_v2/` | ⚠️ **docs + figures only** — the V2 primary dynamics and PPO checkpoints (`artifacts_v2/dynamics_v1ot_ror_corr010/`, `artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M`) are **referenced by `config/paths.yaml` but not present on this machine**. They must be regenerated by re-running the V2 pipeline (or copied in from a backup). | reproduced |
| `artifacts_v3/` | ⚠️ interpretation markdown only — no V3 model checkpoints on disk | reproduced |
| `artifacts_proposal2/rl_v3b_safety_aware_seed42/` | ✅ PPO_C `ppo.zip` + `rollouts.parquet` + `action_freq.json` | committed |
| `artifacts_proposal_fusion/rl_v3b_softot_safety_aware_seed42/` | ✅ PPO_C trained on soft-OT-RoR dynamics; **action collapses to NO-OP, 0 % success — confirms V2 gate-vs-control finding** | local run |
| `data/processed/{norman_hvg.h5ad, depmap_k562_chronos.parquet}` | ✅ on disk | `make data` |

To reproduce V2 primary from a clean checkout, run `make pipeline` after `make data` —
the default config composes the V2 pipeline end to end. **Expect ~2 h on Apple Silicon
MPS and ~30-45 min on a CUDA GPU.**

---

## How to run

### End-to-end (V2 primary, default)

```bash
make pipeline                  # python -m src.pipeline run --config-name default
```

The pipeline runs, in order:

```
data → vae → pairs → dynamics → rl → evaluate
```

Each step is idempotent. Force-rerun a single step:

```bash
python -m src.pipeline run --config-name default --force vae
python -m src.pipeline run --config-name default --skip evaluate
python -m src.pipeline run --config-name default --from rl
```

### Step-by-step

```bash
make vae         # python scripts/train_vae.py --config-name default
make pairs       # python scripts/build_pairs.py --config-name default
make dynamics    # python scripts/train_dynamics.py --config-name default
make rl          # python scripts/train_rl.py --config-name default  (refuses unless gate passes or rl.train.skip_gate=true)
make evaluate    # python scripts/evaluate.py --config-name default + python scripts/visualize.py
```

### Reproduce V1 (not V2) from the same defaults

```bash
python scripts/train_rl.py \
    paths.dynamics_dir=artifacts/dynamics \
    paths.rl_dir=artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k \
    rl.env.max_steps=10 rl.env.epsilon_override=null \
    rl.reward.mode=absolute_distance rl.train.curriculum.enabled=false \
    dynamics.use_state_linear_skip=true dynamics.use_residual_over_ridge=false \
    dynamics.lambda_corr=0.0
```

### Evaluate-only / visualize-only (existing checkpoints)

```bash
make rl-eval     RUN_DIR=<existing-dir>          # PPO eval helper
make rl-random   RUN_DIR=<existing-dir>          # random-policy baseline
make rl-summary  RUN_DIR=<existing-dir>          # summary of a run dir
make aggregate                                    # build artifacts/eval/{summary,results,caveats}
make visualize                                    # render all defense figures
make depmap-compare                               # DepMap gene-score comparison
```

### Hydra overrides

```bash
# Bigger latent
python scripts/train_vae.py vae.n_latent=64 vae.max_epochs=200

# Multi-run / experiment composition
python scripts/train_vae.py --config-name vae_ablation     # composes experiments/vae_ablation.yaml
python scripts/train_dynamics.py --config-name dynamics_legacy_mlp
```

### Tests

```bash
make test          # mock-data only, fast (CI default; runs without real GEO data)
make test-all      # includes integration tests (requires data/processed/)
```

### Notebooks

```bash
make notebooks     # jupyter lab notebooks/
```

`notebooks/{01_data_exploration, 02_vae_latent_inspection, 03_rl_trajectory_viz}.ipynb`
are visualization-only templates (never define metrics there — see CLAUDE.md §3 rule 8).

---

## Docker

### Files

| File | Purpose |
|---|---|
| `Dockerfile.cuda` | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` + Python 3.11 + uv + repo |
| `Dockerfile.cpu` | `python:3.11-slim-bookworm` + CPU torch wheel; sets `CELLPATH_FORCE_DEVICE=cpu` |
| `docker-compose.yml` | three profiles: `cuda` (training), `cpu` (smoke), and always-on `tensorboard` |
| `.dockerignore` | excludes `.git`, `.venv`, caches, `artifacts/**/*.pt|*.h5ad|*.npz`, raw/processed data |

### Build

```bash
make docker-cuda                       # docker build -f Dockerfile.cuda -t cellpath:cuda .
make docker-cpu                        # docker build -f Dockerfile.cpu  -t cellpath:cpu  .
make docker-build                      # build both
```

### Run

```bash
# CUDA cluster — runs the full pipeline against mounted artifacts
docker compose --profile cuda up training

# CPU smoke — validates the pipeline config via `--dry-run`
docker compose --profile cpu  up smoke

# TensorBoard on the named artifacts volume (any time)
docker compose up tensorboard          # http://localhost:6006
```

### Volumes and ports

| Service | Mounts | Ports |
|---|---|---|
| `training` (cuda) | `.:/workspace` + named `cellpath-data:/workspace/data` + `cellpath-artifacts:/workspace/artifacts` | — |
| `smoke` (cpu) | `.:/workspace` | — |
| `tensorboard` | `cellpath-artifacts:/workspace/artifacts:ro` | `6006:6006` |

### What was statically verified

- ✅ `Dockerfile.cpu` and `Dockerfile.cuda` install the same `pyproject.toml` deps via
  `uv pip install --system`; both set `PYTHONPATH=/workspace`.
- ✅ `Dockerfile.cpu` pins `torch==2.4.1+cpu`; `Dockerfile.cuda` pins `torch==2.4.1` with
  the `cu121` wheel index. Both satisfy `pyproject.toml`'s `torch>=2.2,<2.5`.
- ✅ `docker-compose.yml` references both Dockerfiles and the right profiles.
- ✅ `.dockerignore` correctly excludes large binary artifacts (`*.pt`, `*.h5ad`,
  `*.npz`) and raw data, so the build context is small.
- ✅ The default container `CMD` for `cellpath:cpu` is `python -m src.pipeline run --dry-run`,
  which I confirmed works end-to-end on the host.
- ✅ The default container `CMD` for `cellpath:cuda` is
  `python -m src.pipeline run --config-name default` (full training).

### What was *not* verified in this audit

- ❌ **Docker daemon was not running on this Mac at audit time**, so I could not actually
  `docker build` or `docker compose up`. Treat the above as a static review.
- ❌ The `training` service uses both `.:/workspace` and the named volumes
  `cellpath-data:/workspace/data` and `cellpath-artifacts:/workspace/artifacts`. The
  bind mount **shadows** the named volumes on those subpaths in some Docker versions.
  If you intend to persist `artifacts/` across runs on the cluster, prefer **one** of
  the two strategies (named volume *or* bind mount), not both. Verify on first cluster
  run with `docker compose --profile cuda exec training mount | grep workspace`.
- ❌ The CUDA `runtime: nvidia` Compose key is the legacy spelling; modern Docker uses
  `deploy.resources.reservations.devices`. The legacy form still works with the
  `nvidia-container-toolkit` but emits a warning.

### Local verification commands

```bash
# Static lint of Dockerfiles
docker buildx debug --invoke /bin/sh build .                # ad-hoc inspection
hadolint Dockerfile.cuda Dockerfile.cpu                     # if you have hadolint

# Confirm the daemon is up and build both images
docker info
make docker-build
docker run --rm cellpath:cpu  python -m src.pipeline run --dry-run
docker run --rm --gpus all cellpath:cuda nvidia-smi
```

---

## Configuration (Hydra)

All configuration lives in [`config/`](config/) and is composed by `config/default.yaml`:

```
config/
├── default.yaml      # master — composes paths + vae + dynamics + rl
├── paths.yaml        # @package paths — single source of truth for every path
├── vae.yaml          # scVI hyperparameters
├── dynamics.yaml     # MLP + RoR + correlation-loss + gate thresholds
├── rl.yaml           # env + reward + PPO + curriculum + safety / freeband / uncertainty knobs
└── experiments/
    ├── baseline.yaml
    ├── dynamics_legacy_mlp.yaml
    ├── rl_sparse.yaml
    └── vae_ablation.yaml
```

**Never hardcode a path.** Add it to `config/paths.yaml` and reference the new key. The
default composition reproduces V2 primary; see the top of `config/default.yaml` for the
V1 override snippet.

---

## Version history

This project is a master's thesis in progress. Below is the honest, chronological log of
every model iteration. "Current default Hydra composition" = V2 primary; everything after
V2 is research-in-progress with single-seed or no-headline results.

### V1 — original baseline (32-D scVI, V1 OT pairs, K = 10, p50 ε)

- **Config:** 32-dim scVI NB latent; OT pairs; Residual MLP with `state_linear_skip` and
  heteroscedastic head; MaskablePPO with `absolute_distance` reward (no curriculum);
  horizon K = 10; success threshold ε = `epsilon_success.json` p50 = 3.531.
- **Artifacts:** `artifacts/{vae, pairs, dynamics, rl, rl_hard}/` — all on disk, frozen.
- **Result:** dynamics gate **fails** (`passed = false` in `artifacts/dynamics/gate.json`
  — fails the `margin_vs_linear_ridge_pearson` ≥ 0.03 threshold and uncertainty
  calibration). PPO trains and produces successful trajectories at K = 10 but with no
  multi-seed CIs and no honest hard benchmark.
- **Lesson:** the OT-pair construction has a noise floor (median per-perturbation
  correlation ≈ 0.89) — supervised gate cannot be closed with the V1 architecture.

### V1 64-D ablation

- **Config:** same as V1 with `vae.n_latent = 64`.
- **Artifacts:** `artifacts_64/` — frozen.
- **Result:** comparable behaviour to 32-D, no advantage on V1 benchmark. Documented in
  `V3A_LATENT_AUDIT_AND_64D_PLAN.md` and the contraction diagnostics.

### V2 primary — RoR_corr010 × C2 PPO  (2026-05-16, **current default**)

- **Config:**
  - Dynamics: `use_residual_over_ridge = true`, `lambda_corr = 0.10` — the MLP learns the
    residual on top of a frozen ridge baseline; correlation loss penalises per-dim Pearson
    decoupling.
  - RL: `reward.mode = terminal_only_step_cost` (mid-episode reward = 0, terminal = `1·success − β·t`).
  - Curriculum: distance-bin schedule `start_d = 4.0 → end_d = 10.0` over the first 70 %
    of training.
  - Horizon K = 3; ε = p25 = 3.166; 1 M timesteps; 4 seeds.
- **Artifacts (intended):** `artifacts_v2/dynamics_v1ot_ror_corr010/`,
  `artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed{42,0,1,7}/`. On this
  machine only the docs + figures + reachability summaries are present
  (`artifacts_v2/V2_FINAL_REPORT.md`, `artifacts_v2/figures/`, `artifacts_v2/reachability_probe/`).
- **Headline:** PPO 0.941 ± 0.048 at the primary cell; +77 pp over random; +16 pp over
  the V1-OT alternative at K = 2 / bin 6-8 with non-overlapping seed CIs.
- **Honest framing:** PPO matches `greedy_dyn_2` everywhere; does not exceed it.
- **Methodological contribution:** gate-vs-controllability decoupling table.
- **Full report:** [`artifacts_v2/V2_FINAL_REPORT.md`](artifacts_v2/V2_FINAL_REPORT.md).

### V3A — latent / 64-D NB scVI ablation

- **Goal:** test whether a 64-D or alternative-likelihood latent space exposes a field
  where multi-step planning is genuinely required (V2 left this open).
- **Config:** Track L (64-D legacy scVI reuse) and Track N (fresh 64-D NB scVI). Same
  RoR + correlation-loss dynamics.
- **Result:** dynamics + reachability check ran; greedy oracles saturate at K ≥ 3 on
  both tracks, just like V2. Track N completed late 2026-05-17.
- **Plan:** [`V3A_LATENT_AUDIT_AND_64D_PLAN.md`](V3A_LATENT_AUDIT_AND_64D_PLAN.md);
  decision log [`artifacts_v3/interpretation/v3a_final.md`](artifacts_v3/interpretation/v3a_final.md).

### V3B — biorealistic control objective (reward stack)

- **Goal:** investigate whether richer biology-aware reward terms (safety / path-length /
  uncertainty) yield a planning advantage that V2's distance-only reward could not.
- **Phases:**
  - **Phase 0–1.** Built `artifacts_v3/v3b_biology/{gene_safety.parquet, k562_sl_pairs.parquet}`
    from DepMap K562 Chronos + Horlbeck 2018; post-hoc scored V2 PPO + greedy paths.
  - **Phase 2 / 2b / 2c.** Variant C — safety-aware reward (`λ_tox = 0.10`, `λ_ce = 0.05`).
    Single-seed (PPO_C) showed +4 pp at K = 2 / bin 8-10 / OOD; 4-seed escalation
    collapsed the signal back into noise (mean +0.9 pp, CI straddles zero).
  - **Phase 3 / 3b.** Variant B — `path_length_freeband` reward. PPO_B respects the
    freeband schedule but finds no leverage: the V2 dynamics field is locally
    well-conditioned at every K ≥ 4 cell even at ε = p5.
  - **Phase 4 (final V3B).** Variant D (uncertainty-aware) + fused B+C+D. Outcome:
    **`LOCKED_DESIGN_TECHNICAL_ONLY`** — the full reward stack implements, dispatches,
    trains and evaluates without numerical issues, but produces no headline gain over
    V2 on V2 primary dynamics. The +0.05-pp planning-advantage criterion is not met.
- **Plan:** [`V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md`](V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md).
- **Spec lock:** [`V3_CONTROLLER_OBJECTIVE_SPEC.md`](V3_CONTROLLER_OBJECTIVE_SPEC.md).

### V3C — dynamics-utility audit (in progress)

- **Goal:** identify a dynamics field where the locked V3B reward stack can produce a
  *real* planning advantage. Hypothesis: every existing field is either
  over-contracting (OT-trained), low-reachability (mean-delta) or anti-contractive
  (soft-OT). Need a contraction-aware regularizer.
- **Phase 0–1 (2026-05-19/20).** Audited 29 candidate fields; ran PPO_BCD smokes on
  Anchor (V2 RoR), Track L, Track N, and a mean-delta wildcard.
- **Most recent finding (Track N, single seed, 500 k):** PPO_BCD at K = 2 / bin 8-10 /
  OOD = 0.570 vs `greedy_dyn_2_fused = 0.495`, i.e. **+0.075 over greedy** — the first
  V3-era same-field same-reward planning-advantage signal. Did **not** survive doubling
  to 1 M steps (regressed to 0.445). Multi-seed Phase 4 escalation pending.
- **Plan:** [`V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md`](V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md).
- **Status:** **single-seed only**. Not a headline. See
  [`artifacts_v3/v3c/interpretation/v3c_phase1_ppo_smoke_summary.md`](artifacts_v3/v3c/interpretation/v3c_phase1_ppo_smoke_summary.md).

### Proposal series — experimental side-branch (2026-05-18 → 19)

The "Proposal" commits on the `retrainVae` branch (Proposal 1 b57d051, Proposal 2 51b90ce)
are a parallel exploration that does **not** appear in `PROGRESS.md`. They are
documented inline as commit messages and `docs/proposal*_results.md`.

- **Proposal 1 — soft-OT + RoR_corr010 dynamics (gate-passing).** Trained the V2 RoR
  architecture on **soft-OT pairs** rather than V1 OT pairs. Result: gate **passes** at
  val Pearson 0.94 / OOD 0.83. Code+models were committed to
  `artifacts_proposal1/dynamics_soft_ot_ror/` but **the binaries are not in the current
  working tree**; only `diagnostics/*.log` and `docs/proposal3_5c_results.md` survive on
  disk locally. Re-extract from the commit if you need them.
- **Proposal 2 — safety-aware PPO_C on Proposal 1 dynamics.** Trained PPO_C with
  `mode = safety_aware`, `λ_tox = 0.10`, `λ_ce = 0.05`. Checkpoint at
  `artifacts_proposal2/rl_v3b_safety_aware_seed42/{ppo.zip, rollouts.parquet, action_freq.json}`.
- **Proposals 3 + 5c — hardness sweep + DepMap functional enrichment.** Read-only
  evaluations on existing PPO checkpoints. Headline: PPO_C's top-20 picks are enriched
  for `hallmark_apoptosis` (q = 3.5×10⁻⁵), `leukemia_stem_cell_program` (q = 3.6×10⁻⁴),
  `k562_erythroid_lineage` (q = 1.3×10⁻³); PPO_C is the only policy whose top-5 picks
  contain **zero K562 essentials**; hardness sweep ε ∈ {p25, p10, p5} shows PPO_C
  maintains 100 % success AND 0 essential picks across all ε. Full results:
  [`docs/proposal3_5c_results.md`](docs/proposal3_5c_results.md).
- **Proposal Fusion — soft-OT-RoR dynamics × safety-aware PPO_C.** Tried to compose
  Proposal 1 (gate-passing dynamics) with Proposal 2 (safety reward). Result:
  **policy collapses to NO-OP** (`final_eval_metrics.success_rate = 0.0`, NO-OP picked
  495/500 episodes). Reproduces V2's gate-vs-controllability finding: gate-passing
  soft-OT field is **control-hostile** even with a richer reward. Artifacts at
  [`artifacts_proposal_fusion/rl_v3b_softot_safety_aware_seed42/`](artifacts_proposal_fusion/rl_v3b_softot_safety_aware_seed42/).

**Bottom line on the Proposal series:** Proposal 1 produced a gate-passing dynamics
field but Proposal Fusion confirmed that gate-passing alone is not enough — the soft-OT
field is structurally control-hostile, exactly as V2 §6 documented. Proposal 5c is the
only Proposal-series result that adds real new evidence (biological coherence of PPO_C's
gene picks via DepMap enrichment) and survives on disk locally.

---

## Limitations (honest)

1. **Not therapeutic.** The reward target is the **unperturbed K562 NT-guide centroid**,
   not a non-leukemic / healthy reference. CellPath learns to reverse perturbation drift
   in latent space; it does **not** demonstrate cancer reprogramming.
2. **Cross-sectional data.** Perturb-seq is single-snapshot; the `(z_ctrl, gene, z_pert)`
   triples are pseudo-pairs constructed via Sinkhorn OT, not real before/after observations.
   The dynamics model inherits the OT pairing noise (median per-perturbation correlation
   ≈ 0.89 on Norman 32-D), which appears to be the supervised-gate ceiling on V1 OT pairs.
3. **PPO matches but does not exceed `greedy_dyn_2`.** Under V2 32-D geometry, the
   field is locally well-conditioned and a depth-2 model-based oracle is already
   near-optimal at every reported cell. The +0.05 pp planning-advantage threshold is
   not reached anywhere. V3A/B confirmed this on 64-D and on richer reward stacks. V3C
   has a single-seed candidate signal that did not survive longer training.
4. **No cross-dynamics generalisation.** The V2 PPO drops ≥ 14 pp when evaluated on a
   sibling dynamics field. PPO is a *dynamics-specific* controller.
5. **CRISPRa only.** The action space is CRISPRa over-expression; no knock-out / CRISPRi
   axis until a CRISPRi dataset (e.g. Replogle 2022 K562 essentials) is registered in
   `config/paths.yaml::replogle_crispri_h5ad`. The flag is wired
   (`rl.action_space.enable_knockout`) but raises `NotImplementedError` until then.
6. **DepMap is enrichment, not validation.** Hypergeometric / GSEA tests against
   DepMap K562 essentials and MSigDB Hallmark panels demonstrate *biological plausibility*
   of the gene picks, not *therapeutic validity* of any sequence.
7. **V2 primary checkpoints are not currently on this machine.** The default Hydra
   composition references `artifacts_v2/dynamics_v1ot_ror_corr010/` and
   `artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_*`. Only docs + figures are
   present. You must re-run `make pipeline` (~30 min on CUDA, ~2 h on Apple MPS) to
   regenerate them.
8. **`requirements.txt` is out of sync with `pyproject.toml`.** The pyproject is the
   source of truth (e.g. `anndata>=0.11`, `scvi-tools>=1.4`); `requirements.txt` still
   lists `anndata<0.11` and `scvi-tools<1.2`. Prefer `uv pip install -e ".[dev]"`.

---

## Roadmap / next steps

Realistic next moves, in order of expected payoff:

1. **V3C Phase 4 — 4-seed escalation on Tracks L and N** at 1 M timesteps with the
   locked B+C+D reward stack. If the K = 2 / bin 8-10 / OOD planning advantage survives
   multi-seed CIs, that becomes the first V3 `LOCKED_DESIGN_POSITIVE_SIGNAL`.
2. **V3C Phase 2 — contraction-aware dynamics regulariser** (already specced in
   `artifacts_v3/v3c/interpretation/v3c_phase2_contraction_aware_spec.md`). Adds
   `λ_excessive_alignment` and `λ_universal_attractor` regularisers to the dynamics
   loss. Default-off so V2 behaviour is byte-identical.
3. **CRISPRi axis (knockout).** Register Replogle 2022 K562 essential CRISPRi in
   `config/paths.yaml`, lift the `NotImplementedError` guard, retrain dynamics over
   the union action space.
4. **External healthy reference (ethics-reviewed only).** Add a non-leukemic K562 sibling
   reference (e.g. Granja 2019 CD34+ HSPC); make `reference.source = external_healthy`
   operational; re-do the gate-vs-controllability taxonomy.
5. **Pipeline hardening.** Reconcile `requirements.txt` with `pyproject.toml`; make
   `make docker-cuda` deterministic against a pinned base image SHA.

---

## Repository layout

```
src/{data,models,rl,analysis,utils}    # implementation
config/                                 # Hydra configs (single source of truth)
tests/                                  # pytest (mock-data only by default)
scripts/                                # entry points (--config-name, --dry-run)
notebooks/                              # visualization-only templates
artifacts/                              # V1 baseline (frozen, gitignored binaries)
artifacts_v2/                           # V2 primary (frozen)
artifacts_v3/                           # V3A/B/C working dir
artifacts_proposal{1,2,3_5c,_fusion}/   # Proposal-series experimental runs
```

Sacred rules (no hardcoded paths, no inline metrics, no VAE retrain without checkpoint
check, etc.) are in [`CLAUDE.md`](CLAUDE.md) §3.

Companion docs:

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system diagram + 7 concept explanations + decision log
- [`CLAUDE.md`](CLAUDE.md) — agent context, sacred rules, env setup
- [`AGENTS.md`](AGENTS.md) — Agent A + B missions, interface contracts, conflict zones
- [`PHASES.md`](PHASES.md) — 14-day master plan
- [`DATA.md`](DATA.md) — preprocessing biology, OT pairing methodology, DepMap schema
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — hyperparameters + ablation matrix
- [`PROGRESS.md`](PROGRESS.md) — living state file (latest session 2026-05-20, V3C Phase 1)

Plans on disk (versioned alongside the work that produced them):
`PLAN-1.md`, `V2_RESEARCH_PLAN.md`, `V2_STRATEGY_REASSESSMENT_PLAN.md`,
`V2_STRATEGY_P0E_PLAN.md`, `V2_WRAP_OR_V3_PIVOT_PLAN.md`,
`P0B_PRIME_PAIRING_CORRECTION_PLAN.md`, `P0C0_REACHABILITY_PLAN.md`,
`V3_RESEARCH_PLAN.md`, `V3A_LATENT_AUDIT_AND_64D_PLAN.md`,
`V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md`,
`V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md`,
`V3_CONTROLLER_OBJECTIVE_SPEC.md`.

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## How to cite

```bibtex
@misc{cellpath2026,
  title  = {CellPath: in-silico cell-state steering over a CRISPRa surrogate environment},
  author = {CellPath Team},
  year   = {2026},
  note   = {Master's thesis project. Code: https://github.com/gabonavarroo/cellpath},
}
```
