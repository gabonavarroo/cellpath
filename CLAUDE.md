# CLAUDE.md — context for every CC agent on this repo

> **Read this file first.** Then read `ARCHITECTURE.md` and `AGENTS.md`. Everything else
> derives from those three documents.

## 1. Project purpose (two sentences)

CellPath frames cancer cell-state steering as a Markov Decision Process whose state space is a
32-dim scVI latent of K562 CRISPRa Perturb-seq data, whose transitions are a learned residual
dynamics model, and whose policy (MaskablePPO) selects sequences of CRISPRa gene activations
that drive cells toward an unperturbed-K562 reference centroid. **It is in-silico latent-space
steering, not therapeutic reprogramming**; the v1 target is the unperturbed-leukemic baseline,
not a healthy cell state.

## 2. Directory layout

```
cellpath/
├── ARCHITECTURE.md          # System diagram + 7 concept explanations + decision log + failure modes
├── CLAUDE.md                # This file — context for every CC agent
├── AGENTS.md                # Agent A + B mission briefs, interface contracts, conflict zones
├── PHASES.md                # 14-day plan (Day 0 + 5 phases, deliverables, gates, fallbacks)
├── PROGRESS.md              # Living state file — update at end of every work session
├── DATA.md                  # Norman 2019 download, preprocessing biology, OT pairing, DepMap
├── EXPERIMENTS.md           # Hyperparams, naming, Hydra usage, ablation matrix
├── README.md                # User-facing intro (quick start, install, dataset, cite)
├── LICENSE                  # MIT
├── pyproject.toml           # uv-managed deps for Python 3.11
├── requirements.txt         # Mirror of pyproject deps for non-uv users
├── Dockerfile.cuda          # Cluster GPU image (linux/amd64, CUDA 12.1)
├── Dockerfile.cpu           # CI / smoke image
├── docker-compose.yml       # training (cuda profile) + smoke (cpu profile) + tensorboard
├── .dockerignore
├── Makefile                 # `make help` lists targets
│
├── config/                  # Hydra configs — never hardcode paths anywhere
│   ├── default.yaml         # Master config (composes the others)
│   ├── paths.yaml           # All file paths — SINGLE SOURCE OF TRUTH
│   ├── vae.yaml             # scVI hyperparameters
│   ├── dynamics.yaml        # Dynamics MLP hyperparameters
│   ├── rl.yaml              # Env + PPO hyperparameters
│   └── experiments/
│       ├── baseline.yaml    # First run — all defaults
│       ├── vae_ablation.yaml# n_latent ∈ {16, 32, 64}
│       └── rl_sparse.yaml   # λ_sparse ∈ {0.01, 0.05, 0.1}
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py            # GEO + pertpy download (Agent A)
│   │   ├── preprocess.py          # scanpy pipeline; raw counts in adata.layers["counts"] (Agent A)
│   │   └── perturbation_pairs.py  # OT / random / mean-delta pairer (Agent A)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vae.py                 # scVI wrapper, save/load, centroid + ε helpers (Agent A)
│   │   ├── dynamics.py            # Residual heteroscedastic MLP (Agent B)
│   │   └── embeddings.py          # Gene-embedding utilities (Agent B)
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── environment.py         # CellReprogrammingEnv (gym) — NO-OP terminates (Agent B)
│   │   ├── train_ppo.py           # MaskablePPO trainer (Agent B)
│   │   └── reward.py              # Distance + sparsity + uncertainty terms (Agent B)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── latent_space.py        # UMAP, silhouette, ARI (Agent A)
│   │   ├── trajectory.py          # Rollout projection + viz (Agent A/B shared, Agent A owns)
│   │   ├── depmap_validation.py   # Hypergeometric + GSEA + null comparison (Agent A)
│   │   └── metrics.py             # SINGLE SOURCE OF TRUTH for all metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── device.py              # MPS/CUDA/CPU detection — only this file calls torch.device()
│   │   ├── seeding.py             # set_seed(42) — called by every entry point
│   │   ├── checkpointing.py       # save_dict / load_dict + scVI save/load wrappers
│   │   └── logging.py             # TensorBoard + rich logger setup
│   └── pipeline.py                # End-to-end runner (data → vae → pairs → dynamics → rl → eval)
│
├── tests/
│   ├── conftest.py                # Mock 200-cell anndata, mock latent, mock env fixtures
│   ├── test_data.py               # Preprocessing tests
│   ├── test_dynamics.py           # Forward pass, shapes, gradient flow
│   ├── test_environment.py        # Gym compliance, reset/step, repeat-mask, NO-OP semantics
│   └── test_integration.py        # End-to-end pipeline on mock data
│
├── scripts/                       # Entry points — all accept --config-name and --dry-run
│   ├── setup_env.sh               # uv venv + install
│   ├── download_data.sh           # pertpy + checksum + DepMap
│   ├── train_vae.py
│   ├── train_dynamics.py
│   ├── train_rl.py                # Refuses to start unless artifacts/dynamics/gate.json.passed
│   ├── evaluate.py
│   └── visualize.py               # All presentation figures
│
├── notebooks/                     # Visualization only — never define metrics here
│   ├── 01_data_exploration.ipynb
│   ├── 02_vae_latent_inspection.ipynb
│   └── 03_rl_trajectory_viz.ipynb
│
├── artifacts/                     # gitignored, populated at runtime
│   ├── vae/                       # scVI model dir + latents.h5ad + z_reference_centroid.npy + epsilon_success.json
│   ├── pairs/                     # train_pairs.npz + val_pairs.npz + ood_pairs.npz + combo_pairs.npz
│   ├── dynamics/                  # model.pt + gate.json + val_metrics.json + ood_metrics.json
│   ├── rl/                        # ppo.zip + rollouts.parquet + curves.png
│   └── eval/                      # depmap_enrichment.csv + figures/
│
└── data/                          # gitignored, populated by scripts/download_data.sh
    ├── raw/
    └── processed/
```

## 3. Sacred rules — agents MUST NOT

1. **Never retrain the VAE without first checking for an existing checkpoint.** Loading must
   be a one-liner via `scvi.model.SCVI.load(path, adata)`. If a checkpoint is present *and*
   the AnnData schema matches, reuse it. Manual `state_dict` saves are forbidden — use the
   official `model.save() / SCVI.load()` API.
2. **Never modify shared interfaces without updating ARCHITECTURE.md §2 and AGENTS.md §
   "Interface Contract"** in the same commit. Shared interfaces are: VAE→Pairs handshake
   (`artifacts/vae/{latents.h5ad, gene_vocab.json, z_reference_centroid.npy, epsilon_success.json}`),
   Pairs→Dynamics handshake (`artifacts/pairs/*.npz` schema), Dynamics→RL handshake
   (`artifacts/dynamics/{model.pt, gate.json}`), and the centralized `src/utils/device.py`,
   `src/utils/seeding.py`, `src/analysis/metrics.py` modules.
3. **Never hardcode a path.** All paths live in `config/paths.yaml`. Read them through Hydra.
   If a path is missing from the config, add it to `config/paths.yaml` and reference the new
   key — never inline a literal string.
4. **Never compute a metric inline.** Every metric (loss, distance, success rate, enrichment
   p-value, calibration score) goes in `src/analysis/metrics.py` with a docstring containing
   its mathematical definition. Training and evaluation scripts call those functions.
5. **Never call `torch.device()` outside `src/utils/device.py`.** Use `get_device()` everywhere.
6. **Never call `random.seed()`, `np.random.seed()`, or `torch.manual_seed()` outside
   `src/utils/seeding.py`.** Entry points call `set_seed(cfg.seed)` once at start; downstream
   code reads from it.
7. **Never claim the system performs therapeutic reprogramming.** The reward target is the
   unperturbed-K562 NT centroid; the system reverses perturbation drift in a CRISPRa surrogate
   environment. Documentation, docstrings, and figures must respect this. The word "normal",
   "healthy", or "non-leukemic" appears only when explicitly discussing future external-
   healthy-reference work.
8. **Never define a metric or transformation inside a Jupyter notebook.** Notebooks visualize
   results computed by `src/analysis/*`. Add new metrics to `metrics.py` and import.
9. **Never start RL training before the dynamics validation gate passes.** `train_rl.py` reads
   `artifacts/dynamics/gate.json` and exits with a clear error if `passed=False`.
10. **Never enable the knockout / CRISPRi action space without first registering a CRISPRi
    dataset in `config/paths.yaml`.** Setting `rl.action_space.enable_knockout=true` without
    the dataset raises `NotImplementedError`.

## 4. Running components in isolation

All entry-point scripts accept `--config-name <name>` (Hydra) and `--dry-run` (validate config
+ resolve paths, exit before any heavy computation).

```bash
# Foundation
make setup                                         # one-time uv venv + install
make data                                          # download Norman + DepMap

# Training (each step writes to artifacts/<component>/)
make vae                                           # python scripts/train_vae.py --config-name default
make dynamics                                      # python scripts/train_dynamics.py --config-name default
make rl                                            # python scripts/train_rl.py --config-name default   # refuses unless gate passed

# Evaluation
make evaluate                                      # python scripts/evaluate.py --config-name default
python scripts/visualize.py --config-name default # all presentation figures

# Hydra overrides
python scripts/train_vae.py --config-name default vae.n_latent=64 vae.max_epochs=200

# Hydra experiment composition
python scripts/train_vae.py --config-name vae_ablation   # composes experiments/vae_ablation.yaml
```

## 5. End-to-end pipeline

```bash
make pipeline                                      # data → vae → pairs → dynamics → rl → evaluate
# Equivalent:
PYTHONPATH=. python -m src.pipeline run --config-name default
PYTHONPATH=. python -m src.pipeline run --config-name default --dry-run   # validate only
```

## 6. Environment setup

**Mac (Apple Silicon, MPS, dev only):**
```bash
make setup                # creates .venv with uv, installs all deps in editable mode
source .venv/bin/activate
```
Why native venv on Mac: MPS does not pass through to Linux containers. The Docker images are
amd64 and CPU-only on Mac, which would be 20-40× slower than MPS.

**Linux GPU cluster (CUDA):**
```bash
make docker-cuda
docker run --gpus all -v $(pwd):/workspace -v $(pwd)/artifacts:/workspace/artifacts \
    cellpath:cuda python -m src.pipeline run --config-name default
# Or:
docker compose --profile cuda up training
```

**CI / smoke:**
```bash
make docker-cpu
docker compose --profile cpu up smoke
```

## 7. Device detection (canonical snippet)

```python
# src/utils/device.py — the only place this is allowed
from __future__ import annotations
import os
import torch

def get_device() -> torch.device:
    """Resolve the best available accelerator, with env-var override.

    Precedence: CELLPATH_FORCE_DEVICE > CUDA > MPS > CPU.
    """
    forced = os.environ.get("CELLPATH_FORCE_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
```

Every other module uses:
```python
from src.utils.device import get_device
device = get_device()
model.to(device)
```

## 8. Reading and updating PROGRESS.md

At the **end** of every work session, an agent updates `PROGRESS.md` with:

- The phase number and whether the agent's deliverables for that phase are complete.
- Any new metric values (Component → Target → Current → Status table).
- Any blockers encountered (with severity: P0 / P1 / P2).
- Next session's first three priorities.

`PROGRESS.md` is **append-then-edit**: leave previous timestamped sections in place when adding
a new session. The current state is always the top section.

Format for new entry:
```markdown
## Session YYYY-MM-DD-HHMM  (agent: A | B | both)

**Phase:** N — <name>
**Status:** <what changed>
**Metrics:**
| Component | Target | Current | Status |
| --- | --- | --- | --- |
**Blockers:** (P0/P1/P2 / none)
**Next:** (3 bullets)
```

## 9. `.venv` advisory

For ANY dependency installation or trial run on a developer's machine, the canonical workflow
is `uv venv .venv --python 3.11 && uv pip install -e ".[dev]"`. The repo's `.gitignore` already
excludes `.venv/`. Do not commit it. Do not use `conda` — it has not been tested with the
scvi-tools / torch wheel matrix we pin.

## 10. Pointers

- ARCHITECTURE.md — system diagram, 7 concepts, decision log, failure modes
- AGENTS.md — Agent A and Agent B missions + interface contracts + conflict zones
- PHASES.md — 14-day plan
- DATA.md — preprocessing biology + OT pairing details + DepMap schema
- EXPERIMENTS.md — hyperparameters + ablation matrix + Hydra usage
- PROGRESS.md — current state
