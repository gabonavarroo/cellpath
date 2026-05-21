# CellPath

> **In-silico cell-state steering** via a learned scVI latent geometry, a residual
> perturbation-dynamics surrogate, and a MaskablePPO agent that sequences CRISPRa
> interventions to drive K562 leukemia cells toward an unperturbed reference centroid.

This is **not** therapeutic cancer reprogramming. CellPath learns to reverse perturbation
drift in scVI latent space over a CRISPRa surrogate environment built from Norman et al.
2019 (GSE133344). The reward target is the unperturbed-K562 NT-guide centroid; "normal"
/ "healthy" cell states are out of scope (see [`ARCHITECTURE.md`](ARCHITECTURE.md)
Concept 7 and [`DATA.md`](DATA.md) §7).

---

## What this project does

CellPath frames cancer cell-state steering as an MDP:

| Element | Choice |
|---|---|
| **State** | 64-dim scVI latent of K562 CRISPRa Perturb-seq |
| **Action** | `Discrete(N_genes + 1)` — one per single-gene CRISPRa target (≈106) + NO-OP |
| **Transition** | Residual MLP `f_θ(z, gene) → (μ_Δz, log σ²_Δz)` with heteroscedastic NLL |
| **Reward** | distance-to-centroid + sparsity penalty (plus optional safety / freeband / uncertainty terms) |
| **Termination** | NO-OP, success (`‖z − z_ref‖₂ < ε`), or step budget `K` |
| **Policy** | MaskablePPO (sb3-contrib) with per-step gene mask |

The **default Hydra composition** (`config/default.yaml`) reproduces the **V2 primary**
result — the last result with multi-seed CIs and a written wrap-up
([`artifacts_v2/V2_FINAL_REPORT.md`](artifacts_v2/V2_FINAL_REPORT.md)). Subsequent V3A/B/C
work is research-in-progress (see [`PROGRESS.md`](PROGRESS.md) and the V3 interpretation
docs under `artifacts_v3/`).

---

## System diagram

```
  Norman 2019 Perturb-seq (K562 + CRISPRa)
            │
            ▼
   preprocess  ──► HVG selection, raw counts in adata.layers["counts"]
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
       (V2 primary: RoR_corr010 = residual-over-ridge + corr-loss λ = 0.10)
                           │
                  validation gate + beam reachability probe
                           │
                           ▼
         MaskablePPO over CellReprogrammingEnv
           (terminal-only step-cost reward + distance-bin curriculum, K = 3)
                           │
                           ▼
                    rollouts.parquet → DepMap enrichment + trajectories
```

Full diagram and 7-concept walk-through: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Quick start

```bash
make setup            # uv venv (.venv) + install all dependencies
make data             # download Norman 2019 + DepMap K562 Chronos
make pipeline-final   # FINAL MODEL — V3C champion (contraction_aware_v2_aggressive + PPO_BCD)
```

Two pipelines coexist:

| Target | What it composes | When to use |
|---|---|---|
| `make pipeline-final` | `config/experiments/final.yaml` → V3C **champion** (64-D scVI + contraction-aware dynamics + PPO_BCD seed 42 500k). The latest model on the repo. | Default for new runs — this is the final delivery. |
| `make pipeline` | `config/default.yaml` → V2 primary (`RoR_corr010 × C2 PPO`, 32-D scVI). Multi-seed validated headline. | Reproduce the V2 publishable result with 4-seed CIs. |
| `make eval-final` | `experiments/final` from the `evaluate` step only — no retraining. | Re-run only eval + figures of the V3C champion using on-disk checkpoints. |

Both pipelines are idempotent — each step skips if its target artifact already exists.
Force-rerun a step with `--force <step>` or `make nuke` to start from scratch.

Validate any config without compute:

```bash
PYTHONPATH=. python -m src.pipeline run --config-name experiments/final --dry-run
PYTHONPATH=. python -m src.pipeline run --config-name default          --dry-run
```

---

## Installation

Native venv (Mac MPS / Linux dev — recommended):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

MPS does **not** pass through to Linux containers, so on Apple Silicon prefer the native
venv over Docker for development. CUDA cluster and CI use the Docker images below.

---

## Datasets

| Dataset | Used for | Source | Local target |
|---|---|---|---|
| Norman 2019 (GSE133344) | scVI, pairs, dynamics, RL env | `pertpy.dt.norman_2019()` primary; Zenodo scperturb fallback | `data/processed/norman_hvg.h5ad` |
| DepMap Chronos | Enrichment of RL-selected genes | DepMap public release (`24Q2` default) | `data/processed/depmap_k562_chronos.parquet` |

`make data` (= `scripts/download_data.sh`) handles both. Preprocessing biology and the
cross-sectional honesty section: [`DATA.md`](DATA.md).

### Citations

- Norman TM et al. *Science* 365, 786–793 (2019). doi:10.1126/science.aax4438
- Lopez R et al. *Nature Methods* 15, 1053–1058 (2018). doi:10.1038/s41592-018-0229-2
- Bunne C et al. *Nature Methods* 20, 1759–1768 (2023). doi:10.1038/s41592-023-01969-x
- Dempster JM et al. *Genome Biology* 22, 343 (2021). doi:10.1186/s13059-021-02540-7

---

## Current "official" version — V2 primary

| Component | Choice |
|---|---|
| VAE | 32-dim scVI, NB likelihood |
| Pairing | Entropic OT (Sinkhorn, V1 OT pairs) |
| Dynamics | `RoR_corr010` — residual-over-ridge + correlation loss λ = 0.10 |
| RL reward | `terminal_only_step_cost` (mid-step 0; terminal = `1·success − β·t`) |
| Curriculum | distance-bin 4 → 10 over the first 70 % of training |
| Horizon | K = 3 |
| ε success threshold | p25 = 3.166 latent units |
| PPO budget | 1 000 000 timesteps |
| Seeds reported | {42, 0, 1, 7} |

### Headline numbers (V2 hard benchmark, 4-seed mean ± std, n = 300 ep/cell)

| Cell | PPO (C2 / RoR_corr010) | 95 % CI | random | greedy_dyn_2 | PPO − grd2 |
|---|---:|---|---:|---:|---:|
| **K = 3, bin 8-10, OOD (primary)** | **0.941 ± 0.048** | [0.894, 0.988] | 0.170 | 1.000 | −0.059 |
| K = 3, bin 6-8, OOD | 0.998 ± 0.002 | [0.996, 0.999] | 0.177 | 1.000 | −0.002 |
| K = 2, bin 6-8, OOD (frontier) | **0.748 ± 0.053** | [0.697, 0.800] | 0.070 | 0.790 | −0.042 |
| K = 2, bin 8-10, OOD | 0.283 ± 0.045 | [0.239, 0.328] | 0.020 | 0.300 | −0.017 |

**PPO − random = +77 pp** at the primary cell; mean steps per success ≈ 2.5–2.7
(random uses ≈ 5.5).

**Honest framing.** PPO matches but does not exceed `greedy_dyn_2` (a depth-2
model-based oracle) anywhere on this benchmark. The V2 result is *"PPO compressed a
2-step lookahead into a feedforward controller"*, not *"PPO discovers a superior
strategy"*.

### V2 methodological contribution — gate ⊥ controllability

| Dynamics | Supervised gate (val margin ≥ +0.030) | Beam k = 3 reachability | PPO at primary |
|---|---|---|---|
| V1 OT | FAIL (+0.0074) | PASS (17/17) | PASS (0.963 ± 0.042) |
| **RoR_corr010** (V2 primary) | FAIL (+0.0136) | PASS (17/17) | PASS (0.941 ± 0.048) |
| soft-OT | **PASS (+0.0413)** | **FAIL (0/17)** | FAIL (0.000) |
| mean_delta_corr_030 | FAIL (+0.0232) | FAIL (0/17) | FAIL (0.000) |

The supervised gate is necessary but not sufficient for RL controllability — soft-OT
passes the gate and is control-hostile. See
[`artifacts_v2/figures/dynamics_taxonomy.png`](artifacts_v2/figures/dynamics_taxonomy.png).

Full report: [`artifacts_v2/V2_FINAL_REPORT.md`](artifacts_v2/V2_FINAL_REPORT.md).

---

## Artifact layout

```
artifacts/                # V1 baseline (FROZEN per CLAUDE.md §3)
├── vae/                  # 32-dim scVI checkpoint, latents, centroid, ε
├── pairs/                # OT pairs: train / val / ood / combo .npz
├── dynamics/             # V1 dynamics MLP + gate.json
├── rl/                   # V1 PPO checkpoint (K = 10, p50)
└── rl_hard/              # V1 hard-bench PPO

artifacts_64/             # V1 64-D latent ablation (FROZEN)
artifacts_v2/             # V2 primary + ablations (frozen)
artifacts_v3/             # V3A/B/C interpretation md (model checkpoints gitignored)

data/
├── raw/{norman_2019.h5ad, depmap_chronos.csv}
└── processed/{norman_hvg.h5ad, depmap_k562_chronos.parquet}
```

**What is on this machine right now (audit 2026-05-20):**

| Tier | Status |
|---|---|
| `artifacts/` (V1) | ✅ full (vae, pairs, dynamics, rl, rl_hard) |
| `artifacts_64/` | ✅ frozen 64-D ablation |
| `artifacts_v2/` | ⚠️ **docs + figures only** — the V2 primary dynamics and PPO checkpoints referenced by `config/paths.yaml` are not present locally. Regenerate with `make pipeline` (~30 min CUDA, ~2 h Apple MPS). |
| `artifacts_v3/` | ⚠️ interpretation markdown only — no V3 model checkpoints on disk |
| `data/processed/` | ✅ Norman HVG + DepMap K562 |

---

## How to run

### End-to-end (V2 primary, default)

```bash
make pipeline                  # python -m src.pipeline run --config-name default
```

Steps: `data → vae → pairs → dynamics → rl → evaluate`. Each step is idempotent.

```bash
# V3C champion (final model)
python -m src.pipeline run --config-name experiments/final --force vae
python -m src.pipeline run --config-name experiments/final --skip evaluate
python -m src.pipeline run --config-name experiments/final --from rl

# V2 primary (multi-seed publishable headline)
python -m src.pipeline run --config-name default --force vae
```

### Step-by-step

```bash
# Per-component targets honor CONFIG=<name>. Default is the V2 composition.
# For the final V3C champion, set CONFIG=experiments/final.
make vae         CONFIG=experiments/final   # python scripts/train_vae.py
make pairs       CONFIG=experiments/final
make dynamics    CONFIG=experiments/final
make rl          CONFIG=experiments/final   # refuses unless rl.train.skip_gate=true; experiments/final sets it.
make evaluate    CONFIG=experiments/final
```

### Reproduce V1 (not V2) from defaults

```bash
python scripts/train_rl.py \
    paths.dynamics_dir=artifacts/dynamics \
    paths.rl_dir=artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k \
    rl.env.max_steps=10 rl.env.epsilon_override=null \
    rl.reward.mode=absolute_distance rl.train.curriculum.enabled=false \
    dynamics.use_state_linear_skip=true dynamics.use_residual_over_ridge=false \
    dynamics.lambda_corr=0.0
```

### Eval / visualize / tests

```bash
make rl-eval     RUN_DIR=<existing-dir>          # PPO eval helper
make aggregate                                    # build artifacts/eval/{summary,results,caveats}
make visualize                                    # render all defense figures
make depmap-compare                               # DepMap gene-score comparison
make test                                         # pytest, mock-data only
make notebooks                                    # jupyter lab notebooks/
```

### Hydra overrides

```bash
python scripts/train_vae.py vae.n_latent=64 vae.max_epochs=200
python scripts/train_vae.py --config-name vae_ablation         # composes experiments/vae_ablation.yaml
```

---

## Docker

| File | Purpose |
|---|---|
| `Dockerfile.cuda` | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` + Python 3.11 + uv + repo |
| `Dockerfile.cpu`  | `python:3.11-slim-bookworm` + CPU torch wheel; sets `CELLPATH_FORCE_DEVICE=cpu` |
| `docker-compose.yml` | profiles: `cuda` (training), `cpu` (smoke), tensorboard (always-on) |

### Build and run

```bash
make docker-cuda                                    # build cellpath:cuda
make docker-cpu                                     # build cellpath:cpu  (forces --platform linux/amd64 on Apple Silicon)

docker compose --profile cuda up training           # full pipeline on GPU
docker compose --profile cpu  up smoke              # dry-run validation
docker compose up tensorboard                       # http://localhost:6006
```

On Apple Silicon you must pass `--platform linux/amd64` because the `torch==2.4.1+cpu`
wheel only ships for `linux_x86_64`. The Makefile target does this for you; if you call
`docker build` directly, use:

```bash
docker build --platform linux/amd64 -f Dockerfile.cpu -t cellpath:cpu .
```

### Verified

Verified end-to-end on this machine (2026-05-21):
- ✓ `docker build --platform linux/amd64 -f Dockerfile.cpu -t cellpath:cpu .`  → 1.37 GB image.
- ✓ `docker run --rm --platform linux/amd64 cellpath:cpu python -m src.pipeline run --config-name experiments/final --dry-run`  → composes V3C champion cleanly.
- ✓ `docker run --rm --platform linux/amd64 cellpath:cpu python -m src.pipeline run --config-name default --dry-run`  → composes V2 primary cleanly.
- ✓ `docker run --rm --platform linux/amd64 cellpath:cpu pytest --version`  → `pytest 9.0.3` (test runner available).

### Caveats not verified on this machine

- The CUDA image (`Dockerfile.cuda`) requires NVIDIA hardware; this is an Apple Silicon
  laptop, so `cellpath:cuda` build + `docker compose --profile cuda up training` were
  not executed locally. Static review of the Dockerfile is unchanged from prior audit.
- The `training` service in `docker-compose.yml` mixes a bind mount (`.:/workspace`) with
  named volumes (`cellpath-data:/workspace/data`, `cellpath-artifacts:/workspace/artifacts`).
  On some Docker versions the bind mount shadows the named volumes on those subpaths.
  Pick one strategy or verify on first cluster run with:
  ```bash
  docker compose --profile cuda exec training mount | grep workspace
  ```
- `runtime: nvidia` in `docker-compose.yml` is the legacy spelling; still works with
  `nvidia-container-toolkit` but emits a warning.
- If you mount the host repo as a volume (`-v $(pwd):/workspace`) the local `.venv/`
  (built natively for the host OS) can collide with the container's Python. Either
  (a) skip the volume mount (the container has everything baked in), or (b) bind-mount
  only specific subdirs, e.g. `-v $(pwd)/artifacts:/workspace/artifacts`.

---

## Configuration (Hydra)

```
config/
├── default.yaml      # composes paths + vae + dynamics + rl
├── paths.yaml        # @package paths — single source of truth for every path
├── vae.yaml          # scVI hyperparameters
├── dynamics.yaml     # MLP + RoR + correlation-loss + gate thresholds
├── rl.yaml           # env + reward + PPO + curriculum + safety / freeband / uncertainty
└── experiments/{baseline,dynamics_legacy_mlp,rl_sparse,vae_ablation}.yaml
```

Never hardcode a path — add it to `config/paths.yaml` and reference the new key. See
`config/default.yaml` for the V1 override snippet.

---

## Version history

### V1 — original baseline (32-D scVI, V1 OT, K = 10, p50 ε)

- 32-dim scVI NB latent; OT pairs; Residual MLP with `state_linear_skip` + heteroscedastic
  head; MaskablePPO with `absolute_distance` reward (no curriculum); horizon K = 10;
  ε = p50 = 3.531.
- Artifacts: `artifacts/{vae, pairs, dynamics, rl, rl_hard}/`, frozen.
- Dynamics gate fails (`passed = false` — `margin_vs_linear_ridge_pearson < 0.03`). PPO
  trains and produces trajectories at K = 10 but with no multi-seed CIs and no honest
  hard benchmark.
- OT-pair noise floor (median per-perturbation correlation ≈ 0.89) is the gate ceiling.

### V1 64-D ablation

- Same as V1 with `vae.n_latent = 64`. Artifacts: `artifacts_64/`. No advantage on V1
  benchmark.

### V2 primary — RoR_corr010 × C2 PPO  (current default)

- Dynamics: `use_residual_over_ridge = true`, `lambda_corr = 0.10` — MLP learns the
  residual over a frozen ridge baseline; correlation loss penalises per-dim Pearson
  decoupling.
- RL: `reward.mode = terminal_only_step_cost`; distance-bin curriculum 4 → 10 over 70 %;
  K = 3; ε = p25 = 3.166; 1 M timesteps; 4 seeds.
- Headline: PPO 0.941 ± 0.048 at primary cell; +77 pp over random; +16 pp over the V1-OT
  alternative at K = 2 / bin 6-8 (non-overlapping seed CIs).
- Honest framing: PPO matches `greedy_dyn_2` everywhere; does not exceed it.
- Methodological contribution: gate-vs-controllability decoupling table (above).
- Full report: [`artifacts_v2/V2_FINAL_REPORT.md`](artifacts_v2/V2_FINAL_REPORT.md).

### V3A — latent / 64-D NB scVI ablation

- Tests whether a 64-D or alternative-likelihood latent exposes a field where multi-step
  planning is genuinely required. Track L (64-D legacy scVI reuse) and Track N (fresh
  64-D NB scVI) with the same RoR + correlation-loss dynamics. Greedy oracles still
  saturate at K ≥ 3 on both tracks.
- Plan: [`V3A_LATENT_AUDIT_AND_64D_PLAN.md`](docs/plans/V3A_LATENT_AUDIT_AND_64D_PLAN.md).
  Final: [`artifacts_v3/interpretation/v3a_final.md`](artifacts_v3/interpretation/v3a_final.md).

### V3B — biorealistic reward stack

- Investigates safety / path-length / uncertainty reward terms. Phases 2c (safety
  Variant C), 3 (freeband Variant B), 4 (uncertainty Variant D and fused B+C+D) all
  trained and evaluated. Outcome on V2 primary dynamics: **`LOCKED_DESIGN_TECHNICAL_ONLY`**
  — the full stack implements and trains, but the +0.05 pp planning-advantage criterion
  is not met on V2 dynamics.
- Plan: [`V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md`](docs/plans/V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md).
- Lock: [`V3_CONTROLLER_OBJECTIVE_SPEC.md`](docs/plans/V3_CONTROLLER_OBJECTIVE_SPEC.md).
- Interpretations: `artifacts_v3/interpretation/v3b_phase{01,2,2b,2c,3,3b,reward_stack_lock}.md`.

### V3C — dynamics-utility audit (in progress)

- Identifies whether a different dynamics field can unlock real planning advantage. Phase
  0–1 audited 29 candidate fields; Phase 1 PPO_BCD smokes on Anchor, Track L, Track N,
  mean-delta wildcard. Track N single-seed at 500 k showed +0.075 over `greedy_dyn_2`
  at K = 2 / bin 8-10 / OOD — first V3-era same-field same-reward signal; did not
  survive 1 M training. Phase 2 (contraction-aware dynamics regulariser) specced and
  partially executed.
- Status: single-seed / partial multi-seed only. Not a headline.
- Plan: [`V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md`](docs/plans/V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md).
- Latest: `artifacts_v3/v3c/interpretation/v3c_phase{0,0b,1,2_spec,2_summary,4_track_ln_escalation}.md`.

---

## Limitations

1. **Not therapeutic.** The reward target is the unperturbed K562 NT-guide centroid, not
   a healthy reference. CellPath reverses perturbation drift in latent space; it does
   not demonstrate cancer reprogramming.
2. **Cross-sectional data.** Perturb-seq is single-snapshot; `(z_ctrl, gene, z_pert)`
   triples are Sinkhorn-OT pseudo-pairs, not real before/after observations. The
   dynamics inherits the OT pairing-noise ceiling (median per-perturbation correlation
   ≈ 0.89 on 32-D Norman) — this is the supervised-gate floor on V1 OT pairs.
3. **PPO matches but does not exceed `greedy_dyn_2`** at any cell on V2 32-D geometry.
   V3A/V3B confirmed this on 64-D and on richer reward stacks. V3C has a single-seed
   candidate that did not survive longer training.
4. **No cross-dynamics generalisation.** V2 PPO drops ≥ 14 pp when evaluated on a
   sibling dynamics field. PPO is a dynamics-specific controller.
5. **CRISPRa only.** No knock-out / CRISPRi axis until a CRISPRi dataset is registered
   in `config/paths.yaml`. The flag exists (`rl.action_space.enable_knockout`) but
   raises `NotImplementedError`.
6. **DepMap is enrichment, not validation.** Hypergeometric / GSEA against DepMap K562
   essentials and MSigDB Hallmark panels demonstrate biological plausibility of the gene
   picks, not therapeutic validity of any sequence.
7. **V2 primary checkpoints are not currently on this machine.** The default Hydra
   composition references `artifacts_v2/dynamics_v1ot_ror_corr010/` and
   `artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_*`. Only docs + figures are
   present. Regenerate with `make pipeline`.

---

## Roadmap

1. **V3C Phase 4 finalisation** — 4-seed escalation on Tracks L and N at 1 M with the
   locked B+C+D reward stack. First multi-seed V3 headline candidate.
2. **V3C Phase 2 — contraction-aware dynamics regulariser** (specced in
   `artifacts_v3/v3c/interpretation/v3c_phase2_contraction_aware_spec.md`). Default-off
   so V2 behaviour stays byte-identical.
3. **CRISPRi axis.** Register Replogle 2022 K562 essential CRISPRi in `config/paths.yaml`,
   lift the `NotImplementedError` guard, retrain dynamics over the union action space.
4. **External healthy reference (ethics-reviewed).** Non-leukemic K562 sibling reference;
   `reference.source = external_healthy` operational; redo gate/control taxonomy.
5. **Pipeline hardening.** Pin Docker base image SHAs; ensure `requirements.txt` and
   `pyproject.toml` stay in sync via CI.

---

## Repository layout

```
src/{data,models,rl,analysis,utils}    # implementation
config/                                 # Hydra configs (single source of truth)
tests/                                  # pytest (mock-data only by default)
scripts/                                # entry points (--config-name, --dry-run)
notebooks/                              # visualization-only templates
artifacts/                              # V1 baseline (frozen; binaries gitignored)
artifacts_v2/                           # V2 primary (frozen)
artifacts_v3/                           # V3A/B/C interpretation md (checkpoints gitignored)
```

Sacred rules (no hardcoded paths, no inline metrics, no VAE retrain without checkpoint
check, etc.): [`CLAUDE.md`](CLAUDE.md) §3.

Companion docs: [`ARCHITECTURE.md`](ARCHITECTURE.md), [`CLAUDE.md`](CLAUDE.md),
[`AGENTS.md`](AGENTS.md), [`PHASES.md`](PHASES.md), [`DATA.md`](DATA.md),
[`EXPERIMENTS.md`](EXPERIMENTS.md), [`PROGRESS.md`](PROGRESS.md).

Historical plans and specs live under [`docs/plans/`](docs/plans/):
`PLAN-1.md`, `P0B_PRIME_PAIRING_CORRECTION_PLAN.md`, `P0C0_REACHABILITY_PLAN.md`,
`V2_RESEARCH_PLAN.md`, `V2_STRATEGY_P0E_PLAN.md`, `V2_STRATEGY_REASSESSMENT_PLAN.md`,
`V2_WRAP_OR_V3_PIVOT_PLAN.md`, `V3_RESEARCH_PLAN.md`, `V3_CONTROLLER_OBJECTIVE_SPEC.md`,
`V3A_LATENT_AUDIT_AND_64D_PLAN.md`, `V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md`,
`V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md`.

---

## License

MIT. See [`LICENSE`](LICENSE).

## How to cite

```bibtex
@misc{cellpath2026,
  title  = {CellPath: in-silico cell-state steering over a CRISPRa surrogate environment},
  author = {CellPath Team},
  year   = {2026},
  note   = {Master's thesis project. Code: https://github.com/gabonavarroo/cellpath},
}
```
