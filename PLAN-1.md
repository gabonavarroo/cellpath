# CellPath — Lead-Architect Scaffold Plan

## Context

The repository at `/Users/gabo/Developer/ITAM/IA/cellpath` currently contains only `README.md`, `LICENSE`, and `.gitignore`. The user wants the **complete living project scaffold** (12 deliverables: ARCHITECTURE.md, CLAUDE.md, AGENTS.md, PHASES.md, PROGRESS.md, DATA.md, EXPERIMENTS.md, full `config/`, `src/`, `tests/`, `scripts/`, README.md) for a 14-day, 2-engineer master's thesis project. The scaffold must let two Claude Code agents work in parallel from Day 1 without dependency conflicts.

CellPath frames cancer cell reprogramming as an MDP whose state space is a learned scVI latent of K562 Perturb-seq data (Norman 2019, GSE133344), whose transitions are a learned neural dynamics model, and whose policy (PPO) selects sequences of gene perturbations that drive cells toward a reference (non-leukemic-target) latent centroid. The user explicitly demanded a **scientific realism audit first**: the original spec contains several biological overclaims that must be corrected before any code is scaffolded.

This plan file is the contract for the scaffold. After approval, all 12 deliverables will be written verbatim in the order listed in §6. **No file outside `src/`, `config/`, `tests/`, `scripts/`, and the 7 root-level Markdown docs will be created.**

---

## 1. Scientific Realism Audit — Findings & Decisions

The original spec's MDP must be revised on five points before any file is written. These corrections propagate into ARCHITECTURE.md, DATA.md, PHASES.md, AGENTS.md, `config/`, and `src/rl/`.

### 1.1 Action space: CRISPRa-only by default

**Finding.** Norman et al. 2019 (GSE133344) is a CRISPR**a** (activation) Perturb-seq dataset using dCas9-SunTag-VP64 in K562. It contains gain-of-function (overexpression) perturbations only — ~106 single-gene targets and ~131 dual-gene combinations (≈237 unique guide identities total, of which ~287 distinct guide constructs exist when counting both directions). There are no loss-of-function (knockout) perturbations.

**Decision.**
- Default action space: `Discrete(N_genes + 1)` where `N_genes` = set of single-gene CRISPRa targets in Norman (~106), and `+1` is a NO-OP / terminate action.
- The original spec's `action = gene_idx * 2 + direction, direction ∈ {KO, OE}` is **replaced** with `action = gene_idx`, with `direction` fixed to `OE` (overexpression).
- `config/rl.yaml::action_space.enable_knockout` is a flag, **default `false`**, that errors out unless a CRISPRi/KO dataset (e.g. Replogle 2022 K562 essential-genes Perturb-seq) is registered in `config/paths.yaml`. KO support is documented as future work in PHASES.md Phase 5+ stretch.

### 1.2 Reference state: not "normal", explicitly `z_reference_centroid`

**Finding.** "Control" cells in Norman are K562 leukemia cells expressing non-targeting (NT) sgRNAs. They are still leukemic. Calling their centroid the "normal cell state" is biologically wrong and would invalidate the thesis claim.

**Decision.**
- Rename throughout: `normal_centroid` → `z_reference_centroid` (default) or `z_unperturbed_k562_centroid` (more specific). Code, configs, docs, and metric names all change.
- The reward target is the **NT-guide K562 centroid** by default. This means the agent learns to drive perturbed K562 cells *back* toward their unperturbed signature — i.e. the system learns to **reverse perturbation drift**, not "cure leukemia."
- ARCHITECTURE.md and AGENTS.md include a "scope of claim" section: results are about *in-silico latent-space steering over a CRISPRa surrogate environment*, not therapeutic reprogramming.
- `config/rl.yaml::reference.source` is an enum {`unperturbed_k562`, `external_healthy`} with `unperturbed_k562` as default. External-healthy mode (e.g. Granja 2019 CD34+ HSPC) is a stretch flag documented in DATA.md but unimplemented in v1.

### 1.3 Cross-sectional data: OT pseudo-pairs + uncertainty-aware dynamics

**Finding.** Perturb-seq is single-snapshot: we never observe the same cell pre- and post-perturbation. The original spec's `(z_ctrl, gene, direction, z_pert)` triple is therefore impossible to construct from raw observations — there is no `z_ctrl` for any specific `z_pert`.

**Decision.**
- `src/data/perturbation_pairs.py` builds pseudo-pairs by **entropic optimal transport** (Sinkhorn divergence) between the control population and each perturbation population in latent space. This is the CellOT (Bunne et al. 2023, *Nature Methods*) approach. Each control cell is matched to a soft distribution over perturbed cells under the same perturbation label; we sample a hard pairing once per epoch.
- Fallback: random sampling of (ctrl, pert) pairs within the same perturbation, used for ablation and when OT is too slow during dev.
- The dynamics model `f_θ(z, gene) → (μ_Δz, log σ²_Δz)` predicts **mean and per-dimension log-variance**, trained with heteroscedastic Gaussian NLL. This delivers the required uncertainty estimates for the RL reward shaping and validation gate.
- The cross-sectional assumption is documented explicitly in ARCHITECTURE.md §"Concept 7: Markov assumption" and DATA.md §"Pairing methodology."

### 1.4 RL gated by dynamics-validation threshold

**Finding.** Training PPO over a learned dynamics model is meaningless if the dynamics model does not generalize. The user demanded baseline comparisons before RL.

**Decision.** PHASES.md Phase 2 (Days 4–6) closes with a **validation gate**. RL training (Phase 3) cannot start until the dynamics model satisfies all of:

| Baseline                            | Metric         | Pass criterion vs MLP dynamics                         |
| ----------------------------------- | -------------- | ------------------------------------------------------ |
| No-op (Δz = 0)                      | held-out R²    | MLP > no-op by ≥ 0.10                                  |
| Global mean-Δ                       | held-out R²    | MLP > by ≥ 0.05                                        |
| Per-gene mean-Δ                     | held-out R²    | MLP ≥ this baseline (tie acceptable, must not regress) |
| Linear ridge per gene               | held-out Pearson R on top genes | MLP > by ≥ 0.03                       |
| Nearest-neighbor perturbation (k=5) | held-out R²    | MLP > by ≥ 0.03                                        |

Plus uncertainty calibration: Spearman correlation between predicted variance and squared error ≥ 0.2 on held-out perturbations.

Held-out set: 20% of unique perturbations entirely held out (so the gate measures OOD generalization, not in-distribution memorization). Implemented in `src/analysis/metrics.py::dynamics_validation_gate()`.

### 1.5 DepMap: enrichment, not validation

**Finding.** Norman uses gain-of-function CRISPRa; DepMap is loss-of-function CRISPR/RNAi essentiality. They are not the same modality. Overlap is weak-signal plausibility, not validation. Furthermore, the user demanded statistical tests, not anecdotal overlap.

**Decision.** `src/analysis/depmap_validation.py` runs three tests:

1. **Hypergeometric enrichment** of RL-selected genes against (a) DepMap K562 essential genes (Chronos < −0.5), (b) MSigDB Hallmark gene sets, (c) hematopoietic-lineage and leukemia gene panels.
2. **GSEA-style preranked test** of RL action frequencies against the K562 dependency score distribution.
3. **Null comparison**: same tests on (i) random gene sets of matched size, (ii) frequency-matched random genes (matched to HVG expression distribution). Reports z-scores and FDR-adjusted p-values.

ARCHITECTURE.md "Concept 6" explicitly states this is *biological plausibility of selected genes*, not *therapeutic validation of reprogramming sequences*.

---

## 2. Architectural Decisions Log (recorded in ARCHITECTURE.md)

| # | Decision | Alternatives considered | Chosen | Why |
|---|---|---|---|---|
| D1 | VAE likelihood | Gaussian, Poisson, NB, ZINB | **NB** (scVI default) | NB handles overdispersion natively; ZINB unnecessary for 10x v3 droplet K562 data (low zero-inflation beyond NB after normalization). Documented per Svensson 2020, Lopez 2018. |
| D2 | Latent dim | 16 / 32 / 64 / 128 | **32** baseline, ablation over {16, 32, 64} | 32 is scVI's published default for K562-scale data; gives ≥99% reconstruction fidelity in Norman's published analyses. |
| D3 | Dynamics architecture | Direct (z→z'), residual (z→z+Δz), conditional VAE | **Residual MLP with heteroscedastic head** | Residual inductive bias matches the small effect-size of single CRISPRa perturbations (most cells move by ‖Δz‖ < 1.0 in scVI latent). Heteroscedastic head gives per-step uncertainty. |
| D4 | Pairing strategy | Random within-perturbation, k-NN in PCA, **entropic OT**, distributional NLL | **Entropic OT (Sinkhorn) + random fallback** | Best practice for cross-sectional perturbation modeling (CellOT, scGen, CPA). Validated on Norman in Bunne et al. 2023. |
| D5 | RL algorithm | DQN, SAC-Discrete, PPO, MaskablePPO, MCTS | **MaskablePPO (sb3-contrib)** baseline; SAC-Discrete as ablation | Discrete ≤200 actions + cheap surrogate env + sparse-ish reward + need for entropy-driven exploration ⇒ on-policy PPO. MaskablePPO allows per-step gene masks (no repeats, action-budget constraints). PPO is also the most-published on similar molecule/cell design tasks (e.g. MolDQN→ChemRL pivot). SAC-Discrete kept as ablation in EXPERIMENTS.md. |
| D6 | Action repetition policy | Allow repeats, mask repeats, soft penalty | **Mask repeats by default**, configurable | A second activation of the same gene has no biological meaning; masking avoids reward-hacking via repeat spam. |
| D7 | Sparsity penalty form | Per-step constant λ, decaying λ, terminal-only count | **Per-step constant λ + small step cost** | Matches the clinical-translation argument (fewer interventions = lower toxicity/risk). EXPERIMENTS.md ablates λ ∈ {0.0, 0.01, 0.05, 0.1, 0.2}. |
| D8 | Combinatorial perturbations | Ignore, use as training, use as held-out test, both | **Both** — 80% of combos used in dynamics composition loss; 20% held out as composition-generalization test | Norman's dual perturbations are the only direct signal we have that compositions of single-gene effects map to known joint effects. Holding out 20% lets us measure how well the agent's *sequential* policy approximates *true* combinatorial biology. |
| D9 | Polars vs pandas | pandas, polars | **Polars for tabular metadata, DepMap, gene lists; pandas only inside scanpy/anndata I/O boundary** | User preference + Polars is materially faster on the DepMap join; scanpy forces pandas at the anndata boundary. |
| D10 | Docker | Single image, dual image, native-only | **Dual: `Dockerfile.cuda` (linux/amd64, cluster) + `Dockerfile.cpu` (linux/amd64, CI) + native `uv venv` for Mac/MPS development** | MPS does not pass through to Linux containers on Apple Silicon. Native venv for dev gives MPS speedup; Docker images for reproducible cluster runs and CI. `docker-compose.yml` orchestrates a training service + tensorboard service. |
| D11 | Config | argparse, OmegaConf-only, Hydra | **Hydra** | User-specified. Composition via `config/experiments/*.yaml`. |
| D12 | Test data | Use small real subset, full mock | **Synthetic mock fixtures + 200-cell real subset** | CI runs on mock only (fast, no GEO dependency). The 200-cell real subset (committed via Git LFS or download cache) is used for integration tests. |
| D13 | Reproducibility | Per-script seeds, central seed util | **Central `utils/seeding.py` called from every entry point** | Required by user quality standard #6. |

---

## 3. Revised MDP Specification (replaces original spec)

```
State space S        : R^32     (scVI latent of K562 unperturbed and CRISPRa-perturbed cells)
Action space A       : Discrete(N_genes + 1)
                       N_genes ≈ 106 single-gene CRISPRa targets in Norman
                       +1 = NO-OP / terminate
Action semantics     : Gain-of-function activation only (matches Norman modality)
Transition T_θ       : (μ_Δz, log σ²_Δz) = f_θ(z, gene_embedding)
                       z_next = z + μ_Δz + ε,  ε ~ N(0, σ²) at training; deterministic μ at inference
Reward R(z, a, z')   : −‖z' − z_ref‖₂  −  λ_sparse · 1[a ≠ NO-OP]  −  λ_unc · ‖σ(z, a)‖
                       where z_ref = z_reference_centroid (default: unperturbed-K562 NT centroid)
Discount γ           : 0.99
Episode budget K     : 10 steps (baseline), ablate {5, 10, 20}
Success criterion    : ‖z − z_ref‖₂ < 0.5  OR  policy emits NO-OP
Action mask          : repeat-mask (genes used earlier in episode are masked)
Termination          : Success ∨ NO-OP ∨ steps == K
```

---

## 4. Deliverable Inventory (12 files / dirs + notebooks)

> User confirmations applied: (a) action space = **CRISPRa-only**, knockout disabled with future-work flag; (b) reference state = **unperturbed K562 NT centroid only**, no external healthy stub; (c) **`notebooks/` scaffolded with templates** (see §4a below).

Each row notes (a) the file's purpose, (b) the corrections from the original spec that propagate into it, and (c) ownership.

| # | Path | Purpose | Key corrections | Owner agent |
|---|---|---|---|---|
| 1 | `ARCHITECTURE.md` | System diagram + 7 concept explanations (≥150 words each) + design decisions log + failure modes + integration diagram + Docker topology. | All 5 audit findings, all 13 D-decisions. Concept 4 (PPO) explicitly weighs alternatives. | Lead (this plan) |
| 2 | `CLAUDE.md` | Master context for all CC agents. Directory tree, sacred rules, run instructions, env setup, device snippet, `.venv` advisory. | Reference to z_reference_centroid naming. CRISPRa-only sacred rule. | Lead |
| 3 | `AGENTS.md` | Agent A + B mission briefs, interface contract, conflict zones, blocked-dependency mocks, daily sync protocol, definitions of done. | Mock OT-pair generator for Agent B unblocking. Contract: dynamics model receives `gene_idx`, not `(gene_idx, direction)`. | Lead |
| 4 | `PHASES.md` | 14-day plan, Day 0 + 5 phases, per-phase deliverables / dependencies / success criteria / fallback. | Phase 2 includes validation gate. CRISPRa-only scope. | Lead |
| 5 | `PROGRESS.md` | Living state file. Checklist, metrics table, blockers, next-session priorities. | Initial state = "Phase 0 in progress." Metrics table has gate rows. | Lead |
| 6 | `DATA.md` | Norman download (pertpy primary, GEO fallback), preprocessing pipeline (per-step biology), pairing methodology (OT), DepMap schema. | Section on cross-sectional honesty. CRISPRa explanation. Pseudo-pair section. | Lead |
| 7 | `EXPERIMENTS.md` | Hyperparams, naming convention, what to log, Hydra usage, baselines, ablation matrix. | Validation-gate experiment as P0. λ_sparse / latent-dim / pairing-strategy ablations. SAC-Discrete ablation. | Lead |
| 8 | `config/` | `default.yaml`, `paths.yaml`, `vae.yaml`, `dynamics.yaml`, `rl.yaml`, `experiments/{baseline,vae_ablation,rl_sparse}.yaml` | `rl.yaml.action_space.enable_knockout: false`. `rl.yaml.reference.source: unperturbed_k562`. | Lead |
| 9 | `src/` (15 .py + 5 __init__.py) | Full directory tree per spec, every stub with complete docstring + type hints + `NotImplementedError("Agent X: implement <thing>")`. | `dynamics.py` signature reflects `(z, gene_idx) → (μ, log_var)`. `environment.py` enforces repeat-mask and NO-OP action. `perturbation_pairs.py` exposes both OT and random pairers. | Lead |
| 10 | `tests/` | `test_data.py`, `test_dynamics.py`, `test_environment.py`, `test_integration.py`, `conftest.py` | Conftest fixtures: synthetic 200-cell anndata, 32-dim mock latent, 16-action mock env. Integration test runs the full mock pipeline. | Lead |
| 11 | `scripts/` | `setup_env.sh`, `download_data.sh`, `train_vae.py`, `train_dynamics.py`, `train_rl.py`, `evaluate.py`, `visualize.py` — every Python script accepts `--config` (Hydra override) and `--dry-run`. | `download_data.sh` checksums pertpy cache. `train_rl.py` refuses to start unless validation gate is recorded as passed in `artifacts/dynamics_gate.json`. | Lead |
| 12 | `README.md` | Title + abstract + ASCII diagram + 3-command quick start + install + dataset + results placeholder + cite + MIT | Honest abstract: in-silico latent-space steering. | Lead |

Plus root-level support:
- `pyproject.toml` (uv-managed) with pinned dependencies for Python 3.11.
- `Dockerfile.cuda`, `Dockerfile.cpu`, `docker-compose.yml`, `.dockerignore`.
- `requirements.txt` (mirror of pyproject for non-uv users).
- `Makefile` with targets: `setup`, `data`, `vae`, `dynamics`, `rl`, `evaluate`, `test`, `lint`, `docker-build`, `tensorboard`.

### 4a. `notebooks/` (scaffolded per user request)

Three template notebooks live under `notebooks/` and **always import their analysis logic from `src/analysis/`**, never duplicate it. CLAUDE.md adds a sacred rule: *"Notebooks may visualize results but may not define new metrics or transformations — those live in `src/analysis/metrics.py` or `src/analysis/*.py`."* This prevents the metric-drift risk the user accepted.

| Notebook | Owner | Purpose |
|---|---|---|
| `notebooks/01_data_exploration.ipynb` | Agent A | Loads preprocessed `adata`, plots HVG distributions, perturbation counts per gene, control vs perturbed cell counts, basic QC stats. Imports from `src.data.preprocess` and `src.analysis.metrics`. |
| `notebooks/02_vae_latent_inspection.ipynb` | Agent A | Loads `artifacts/vae/latents.h5ad`, plots UMAP colored by perturbation, computes silhouette scores, visualizes the `z_reference_centroid` location, ELBO curves. Imports `src.analysis.latent_space`. |
| `notebooks/03_rl_trajectory_viz.ipynb` | Agent B | Loads PPO rollouts from `artifacts/rl/`, renders trajectory traces in UMAP, shows discovered intervention sequences, reward curves, action-frequency bar charts. Imports `src.analysis.trajectory`. |

Each notebook starts with the same boilerplate cell (autoreload, sys.path append for `src/`, Hydra compose of the `default` config) plus a header markdown cell explicitly stating the owner agent and the artifacts it depends on. Notebooks are committed but their outputs are stripped via `nbstripout` (configured in `pyproject.toml`).

`.gitignore` adds `notebooks/*-scratch.ipynb` and `notebooks/.ipynb_checkpoints/` to prevent ad-hoc throwaway notebooks polluting the repo while keeping the three templates tracked.

---

## 5. Interface Contract (Agent A ↔ Agent B handshake)

This contract is locked in AGENTS.md and re-stated in `src/*/__init__.py` docstrings. Neither agent may change these without updating both.

```python
# Contract 1: VAE → Dynamics
# Agent A produces these artifacts; Agent B consumes them.
artifacts/vae/
  model.pt                       # scVI model state dict
  latents.h5ad                   # adata with adata.obsm["X_scVI"] of shape (N_cells, 32)
                                 # adata.obs["perturbation"] : str, "ctrl" or gene symbol
                                 # adata.obs["perturbation_idx"] : int, 0 = ctrl, 1..N for genes
  gene_vocab.json                # {"genes": [gene_symbol_0, ...], "ctrl_idx": 0}
  z_reference_centroid.npy       # shape (32,)

# Contract 2: Pairs → Dynamics training
# Agent A's perturbation_pairs.py produces this; Agent B's dynamics trainer consumes it.
artifacts/pairs/
  train_pairs.npz                # arrays: z_ctrl (N, 32), gene_idx (N,), z_pert (N, 32)
  val_pairs.npz                  # held-out perturbations
  combo_pairs.npz                # for composition loss: z_ctrl, gene_idx_a, gene_idx_b, z_pert_ab
  metadata.json                  # pairing method, n_train, n_val, held_out_genes, OT epsilon

# Contract 3: Dynamics → RL
# Agent B produces; Agent B's RL env consumes (single-agent boundary, but documented).
artifacts/dynamics/
  model.pt
  gate.json                      # {"passed": bool, "metrics": {...}}  — train_rl.py reads this
  val_metrics.json
```

Both agents import the same `src.utils.device.get_device()`, the same `src.utils.seeding.set_seed()`, and the same `config/paths.yaml`. No file outside `src/utils/` may call `torch.device()` directly.

---

## 6. Implementation Order (after ExitPlanMode)

I execute the 12 deliverables in this dependency-aware order. Steps within a group can be written in any order; later groups read earlier groups.

1. **Foundations**: `pyproject.toml`, `requirements.txt`, `Dockerfile.cuda`, `Dockerfile.cpu`, `docker-compose.yml`, `Makefile`, `.dockerignore`.
2. **ARCHITECTURE.md** — written first because every other doc references its diagrams, concepts, and decision log.
3. **CLAUDE.md** — references ARCHITECTURE.md.
4. **AGENTS.md** — references the interface contract in §5.
5. **PHASES.md** — references the validation gate (§1.4) and architecture decisions.
6. **DATA.md** — references the pairing methodology (§1.3) and gene-vocab contract.
7. **EXPERIMENTS.md** — references all hyperparameters in `config/`.
8. **`config/`** all 6 + 3 YAML files, populated with defaults and inline comments.
9. **`src/`**: utils → data → models → rl → analysis → pipeline. Every stub has full docstring + type hints + `NotImplementedError("Agent A|B: ...")`.
10. **`tests/`**: `conftest.py` first (fixtures), then unit tests, then integration test.
11. **`scripts/`**: every script importable but raises `NotImplementedError` until Agent A/B fills it in (except `setup_env.sh` and `download_data.sh`, which are fully functional shell scripts).
12. **PROGRESS.md** — written last so it can accurately enumerate the just-created checklist.
13. **README.md** — quick-start references the now-existing Makefile targets.

After all files are written, I run:
- `find . -name "*.py" | xargs python -m py_compile`  — every module imports cleanly
- `pytest tests/ -k "not slow"`  — fixtures + import smoke tests pass on mock data
- `python -c "from src.pipeline import main; main(dry_run=True)"`  — dry-run end-to-end

If any of these fail, I fix the scaffold before declaring done.

---

## 7. Verification Plan (post-scaffold)

For the user to confirm the scaffold is correct **before** Agent A/B start filling implementations:

| Check | Command | Expected |
|---|---|---|
| Every module imports | `python -m compileall src/` | exit 0, no syntax errors |
| Pytest smoke | `pytest tests/ -x --no-cov` | All NotImplementedError-guarded stubs pass via mocking; conftest fixtures load |
| Hydra config loads | `python -c "import hydra; from hydra import compose, initialize; initialize(config_path='config'); cfg = compose('default'); print(cfg)"` | Prints full composed config; no missing keys |
| Device detection | `python -c "from src.utils.device import get_device; print(get_device())"` | Prints `mps`, `cuda`, or `cpu` |
| Docker CPU image builds | `docker build -f Dockerfile.cpu -t cellpath:cpu .` | Build succeeds; entrypoint `python -m src.pipeline --dry-run` exits 0 |
| `make help` | `make help` | Lists all targets |

Engineers verify their agent missions are clearly scoped by reading AGENTS.md and confirming the interface contract.

---

## 8. Out of Scope (explicit non-goals for the scaffold step)

- **No implementation code.** All `src/` modules are stubs that `raise NotImplementedError`. Agent A and Agent B will fill them in Phases 1–5.
- **No downloaded data.** `scripts/download_data.sh` is functional but is not executed by the scaffold.
- **No trained models.** No `.pt` files committed.
- **No CI workflow file.** Github Actions / similar is left for the engineers if they want it (mentioned as optional in PHASES.md Phase 0).
- **Notebook outputs are not stored.** `nbstripout` strips cell outputs on commit (configured in `pyproject.toml`). The three templates from §4a are scaffolded; ad-hoc scratch notebooks are gitignored.

---

## 9. Resolved Decisions (confirmed with user)

- **Action space.** CRISPRa-only confirmed. `config/rl.yaml::action_space.enable_knockout: false` is the default; setting it to `true` raises `NotImplementedError("Register a CRISPRi dataset in config/paths.yaml first")`. KO is documented as Phase 5+ stretch.
- **Reference state.** Unperturbed K562 NT centroid confirmed. `config/rl.yaml::reference.source` enum has values `{unperturbed_k562, external_healthy}`; default is `unperturbed_k562`. Selecting `external_healthy` raises `NotImplementedError`.
- **Notebooks.** Three template notebooks scaffolded under `notebooks/` with the sacred rule that they may not define metrics — only `src/analysis/*` may. `nbstripout` strips outputs on commit.

---

## 10. Critical Files to Read When Executing

When I exit plan mode, I will reference these files repeatedly. Listed here so the execution phase doesn't drift:

- This plan file — source of truth for all decisions.
- `/Users/gabo/Developer/ITAM/IA/cellpath/.gitignore` — already excludes `.venv`, `__pycache__`, `*.log`, etc. New scaffold files won't be ignored.
- `/Users/gabo/Developer/ITAM/IA/cellpath/README.md` — will be overwritten in step 13 of §6.

No other files in the repo are load-bearing.
