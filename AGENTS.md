# AGENTS.md — operating instructions for parallel Claude Code / Codex agents

> Two human engineers (Agent A and Agent B) work in parallel for 14 days. Each engineer drives
> one or more Claude Code / Codex agent sessions per work block. This file is the contract
> that prevents collisions. Read it before starting any work.

## 0. Workflow primer

Each engineer:
1. Creates a `.venv` via `make setup` (one-time).
2. At the start of each session, reads `PROGRESS.md` to know the current phase and outstanding
   blockers.
3. Picks the next task from `PHASES.md` matching their role (A or B).
4. Implements **only** within their owned files (see §3). Never touches the other agent's files.
5. At the end of the session, appends to `PROGRESS.md` per the format in CLAUDE.md §8.

When in doubt, ask the *other* engineer rather than guessing. The interface contract (§4) is
sacred — both agents depend on it.

## 1. Agent A — Mission Brief: Data + Representation

**Role.** Owns the data pipeline, the scVI VAE, all latent-space analysis, the OT pseudo-pairing
generator (handed off to Agent B), and DepMap validation. End deliverable: a trained VAE whose
latent space distinguishes perturbation clusters, plus tested pairing data for the dynamics
model, plus a DepMap enrichment report.

**Owned files.** Agent A is the sole author of:
```
src/data/download.py
src/data/preprocess.py
src/data/perturbation_pairs.py
src/models/vae.py
src/analysis/latent_space.py
src/analysis/depmap_validation.py
notebooks/01_data_exploration.ipynb
notebooks/02_vae_latent_inspection.ipynb
scripts/download_data.sh
scripts/train_vae.py
```

**Phase-by-phase tasks (see PHASES.md for dates).**

- **Phase 0 (Day 0):** Verify `make setup` works on Mac. Confirm `make data` downloads Norman
  via pertpy and the DepMap K562 Chronos table. Stub `notebooks/01_data_exploration.ipynb`
  with the boilerplate cell.

- **Phase 1 (Days 1–3):** Implement `src/data/preprocess.py` end-to-end:
  - Load Norman AnnData.
  - **Preserve raw counts in `adata.layers["counts"]`** before any normalization.
  - Run `sc.pp.filter_cells(min_counts=...)` and `sc.pp.filter_genes(min_cells=10)`.
  - Compute `sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000, layer="counts")`.
  - Normalize to library size 10k via `sc.pp.normalize_total` and `sc.pp.log1p` for visualization /
    HVG selection only. The integer counts in `layers["counts"]` are NOT mutated.
  - Subset to HVG: `adata = adata[:, adata.var["highly_variable"]].copy()`.
  - Build `adata.obs["perturbation_idx"]` integer encoding (0 = ctrl).
  - Save to `data/processed/norman_hvg.h5ad`.
  - Train `src/models/vae.py`:
    - `scvi.model.SCVI.setup_anndata(adata, layer="counts")` — the **counts** layer, not `adata.X`.
    - `SCVI(adata, n_latent=32, gene_likelihood="nb", n_layers=2, dropout_rate=0.1)`.
    - Train, save with `model.save("artifacts/vae/")`.
    - Compute `Z = model.get_latent_representation()`, write `latents.h5ad`.
    - Compute `z_reference_centroid = Z[ctrl].mean(0)` → `artifacts/vae/z_reference_centroid.npy`.
    - Compute `epsilon_success = np.percentile(||Z[ctrl] - z_ref||_2, 90)` → `artifacts/vae/epsilon_success.json`.
    - Save `gene_vocab.json` with the integer-encoded vocabulary.

- **Phase 2 (Days 4–6):** Implement `src/data/perturbation_pairs.py`:
  - `build_pairs(adata, latents, method="ot")` returns `train_pairs.npz`, `val_pairs.npz`,
    `ood_pairs.npz` (gene-held-out), `combo_pairs.npz`.
  - OT path: `ot.sinkhorn` with `reg=0.05` (config), normalize cost matrix by median.
  - Random and mean-delta fallbacks must be available; selectable via `pairing.method`.
  - Latent-space analysis: `src/analysis/latent_space.py` UMAP, silhouette on perturbation,
    Adjusted Rand Index, and **the `epsilon_success` derivation function** (used by `vae.py` and
    by the RL config loader).

- **Phase 3 (Days 7–9):** Implement `src/analysis/depmap_validation.py`:
  - Hypergeometric enrichment vs (a) DepMap K562 essentials (Chronos < −0.5), (b) MSigDB Hallmarks,
    (c) hematopoietic-lineage panels.
  - GSEA preranked test on RL action frequency vector vs Chronos score distribution.
  - Null comparison: matched-size random gene sets + matched-expression-distribution sets.
  - Output `artifacts/eval/depmap_enrichment.csv` with FDR-adjusted p-values.

- **Phase 4 (Days 10–12):** Integration with Agent B. Run end-to-end pipeline, verify all
  artifacts match the contract in §4. Update notebooks 01 and 02 with final figures.

- **Phase 5 (Days 13–14):** Presentation figures via `scripts/visualize.py`. Final DepMap
  enrichment table for thesis defense.

**Definition of "done"** for Agent A's milestones:
- VAE: ELBO converged + UMAP shows visible perturbation clustering + `silhouette_score ≥ 0.05`
  + `epsilon_success` written to disk.
- Pairing: `train_pairs.npz` exists with shape contract verified by `tests/test_data.py` + 
  metadata.json contains pairing method and held-out gene list.
- DepMap: at least one FDR-adjusted p < 0.05 enrichment finding documented.

## 2. Agent B — Mission Brief: Models + Agent

**Role.** Owns the dynamics model architecture, training, and validation gate; the RL environment;
PPO training; reward shaping. End deliverable: a dynamics model that passes the primary
validation gate, an RL environment compliant with gymnasium, and a trained PPO policy whose
rollouts can be projected to UMAP for analysis.

**Owned files.** Agent B is the sole author of:
```
src/models/dynamics.py
src/models/embeddings.py
src/rl/environment.py
src/rl/train_ppo.py
src/rl/reward.py
src/analysis/trajectory.py
notebooks/03_rl_trajectory_viz.ipynb
scripts/train_dynamics.py
scripts/train_rl.py
```

**Phase-by-phase tasks.**

- **Phase 0 (Day 0):** Verify `make setup` works on Mac. Stub `src/models/dynamics.py` with the
  class signature matching ARCHITECTURE.md §1 and the interface contract in §4. Build a *mock
  pairs generator* (see §5) so dynamics training can run without waiting for Agent A's OT
  output.

- **Phase 1 (Days 1–3, in parallel with Agent A):** Implement `src/models/dynamics.py`:
  - `PerturbationDynamicsModel(n_latent=32, n_genes=N, d_emb=64, n_hidden=256, n_layers=3)`.
  - Residual head + heteroscedastic head (`mu`, `log_var ∈ [-5, 3]`).
  - `forward(z, gene_idx) → (z_next, mu, log_var)`.
  - Heteroscedastic Gaussian NLL loss in `train_dynamics.py`.
  - Train on **mock pairs** initially; switch to Agent A's real pairs at Phase 2 hand-off.

- **Phase 2 (Days 4–6):** Implement the validation gate in
  `src/analysis/metrics.py::dynamics_validation_gate()` (shared file, but Agent B writes the
  gate function; metric utilities used by it are shared with Agent A — coordinate on PR).
  - **Primary gate:** held-out cells within seen perturbations (90/10 split *within* each
    perturbation). Compare MLP vs no-op, global-mean-Δ, per-gene-mean-Δ, linear ridge,
    nearest-neighbor (k=5).
  - **OOD report:** fully held-out genes (20% of perturbations). Same baselines. Reported but
    **not gating** (Norman has no external gene-feature side info; OOD generalization across
    genes is a known-hard problem, not a hard requirement for v1).
  - Write `artifacts/dynamics/gate.json` with `{"passed": bool, "metrics": {...}, "ood": {...}}`.

- **Phase 3 (Days 7–9):** Implement `src/rl/environment.py`:
  - `CellReprogrammingEnv(dynamics_model, z_reference_centroid, epsilon_success, n_genes, K, λ_sparse, λ_unc=0)`.
  - `observation_space = Box(-inf, inf, (32,))`, `action_space = Discrete(n_genes + 1)`.
  - `reset(options=None)`: sample `z₀` from a random perturbation cluster (off-target), reset
    repeat-mask.
  - `step(action)`:
    - If `action == NO_OP_IDX`: terminate; `success = (||z - z_ref|| < ε)`, reward includes any
      pending shaping.
    - Else: pass `(z, gene_idx)` through dynamics, update repeat-mask, recompute success.
    - Always populate `info["action_mask"]` as required by `sb3-contrib.MaskablePPO`.
  - Implement `src/rl/reward.py` with distance term, sparsity term, optional uncertainty term.
  - Implement `src/rl/train_ppo.py` with `MaskablePPO` from `sb3-contrib`.
  - `scripts/train_rl.py` **MUST** read `artifacts/dynamics/gate.json` and exit with code 2 if
    `passed=False`. No exceptions, including dev runs (override via `train_rl.skip_gate=true`
    Hydra flag with a loud warning).

- **Phase 4 (Days 10–12):** Integration testing with Agent A. End-to-end run.

- **Phase 5 (Days 13–14):** Roll-out rendering in `src/analysis/trajectory.py` (Agent A owns
  this file's UMAP plumbing; Agent B contributes the trajectory data structure). Final
  trajectory figures via `scripts/visualize.py`.

**Definition of "done"** for Agent B's milestones:
- Dynamics: validation gate `passed=true`, with primary-gate metrics in `val_metrics.json`.
- Env: `tests/test_environment.py` passes including gymnasium API compliance, NO-OP semantics,
  and repeat-mask behaviour.
- PPO: training runs to completion, `rollouts.parquet` populated, ≥30% success rate on
  in-distribution starting points (a stretch threshold; the floor is "training stable").

## 3. Conflict zones

Files that neither agent may modify without coordinating with the other in advance. Coordinate
means: a brief sync (Slack / DM) + a co-authored commit with both initials in the message.

```
src/analysis/metrics.py        # both agents add metrics here — coordinate
src/utils/device.py            # changes affect both agents' training
src/utils/seeding.py           # changes affect reproducibility for both
src/utils/checkpointing.py     # both agents save/load through these helpers
src/utils/logging.py           # both agents log through these helpers
src/pipeline.py                # the orchestrator — coordinated changes only
config/paths.yaml              # adding paths is fine; removing/renaming requires coordination
config/default.yaml            # adding keys is fine; removing/renaming requires coordination
ARCHITECTURE.md                # never edit without commit-level rationale
AGENTS.md (this file)          # changes require both agents' approval
```

When two agents do need to touch the same file, the **second** agent writes a `// COORDINATION:`
comment at the top of the diff describing the change and gets a 👍 from the first before
merging.

## 4. Interface contract — the handshake

This is the **single source of truth** for cross-agent data exchange. Both agents may
*assume* these schemas exist and may *not* unilaterally change them. If a schema change is
proposed, both AGENTS.md and ARCHITECTURE.md must be updated in the same commit.

### Contract 1: VAE output (Agent A → Agent B + Agent A)

```
artifacts/vae/
    model/                          ← scVI's own directory (model.pt, attr.pkl, var_names.csv)
    latents.h5ad                    ← AnnData with:
                                       adata.obsm["X_scVI"]    : (N, 32) float32
                                       adata.obs["perturbation"] : str   (= "ctrl" | gene)
                                       adata.obs["perturbation_idx"] : int (0 = ctrl)
                                       adata.var unchanged from preprocess output
    gene_vocab.json                 ← {
                                         "genes": [<perturbed gene symbols, ordered>],
                                         "ctrl_idx": 0,
                                         "n_genes": <int>,
                                         "noop_idx": <n_genes>   # NO-OP action index
                                       }
    z_reference_centroid.npy        ← (32,) float32
    epsilon_success.json            ← {
                                         "percentile": 90,
                                         "value": <float>,
                                         "n_ctrl_cells": <int>,
                                         "method": "L2_distance"
                                       }
```

### Contract 2: Pseudo-pairs (Agent A → Agent B)

```
artifacts/pairs/
    train_pairs.npz                 ← arrays:
                                       z_ctrl    (M, 32) float32
                                       gene_idx  (M,)    int32   # range [1, n_genes]
                                       z_pert    (M, 32) float32
    val_pairs.npz                   ← same schema (held-out CELLS within seen perturbations,
                                                    used for the primary validation gate)
    ood_pairs.npz                   ← same schema (held-out GENES / perturbations, OOD report)
    combo_pairs.npz                 ← arrays:
                                       z_ctrl    (M, 32) float32
                                       gene_idx_a(M,)    int32
                                       gene_idx_b(M,)    int32
                                       z_pert_ab (M, 32) float32
    metadata.json                   ← {
                                         "pairing_method": "ot" | "random" | "mean_delta",
                                         "n_train": <int>, "n_val": <int>,
                                         "n_ood": <int>, "n_combo": <int>,
                                         "held_out_genes": [<list>],
                                         "ot_epsilon": <float | null>,
                                         "n_per_perturbation": {<gene>: <count>, ...}
                                       }
```

### Contract 3: Dynamics output (Agent B → RL + Agent A)

```
artifacts/dynamics/
    model.pt                        ← torch state dict for PerturbationDynamicsModel
    config.json                     ← {n_latent, n_genes, d_emb, n_hidden, n_layers, ...}
    gate.json                       ← {
                                         "passed": bool,
                                         "primary": {
                                             "r2": <float>,
                                             "pearson_r": <float>,
                                             "baselines": {<name>: {<metric>: <float>}}
                                         },
                                         "ood": {
                                             "r2": <float>, "pearson_r": <float>,
                                             "baselines": {...}
                                         },
                                         "uncertainty_calibration": {
                                             "spearman": <float>, "pass": bool
                                         }
                                       }
    val_metrics.json                ← richer dump of primary-gate metrics
    ood_metrics.json                ← richer dump of OOD metrics
```

### Contract 4: RL output (Agent B → Agent A's analysis)

```
artifacts/rl/
    ppo.zip                         ← MaskablePPO.save() format
    rollouts.parquet                ← columns:
                                       episode_id  int64
                                       step        int32
                                       action      int32
                                       gene_symbol str
                                       z_norm      float32     # ||z - z_ref||_2
                                       reward      float32
                                       terminated  bool
                                       success     bool
                                       z_vector    list[float32]  # length 32
    success_curves.png              ← matplotlib output
    action_freq.json                ← {<gene_symbol>: <count>, ...}
```

## 5. Blocked-dependency unblocking — mock data contracts

If Agent B needs Agent A's pairs file before it's ready, Agent B uses
`src/data/perturbation_pairs.py::generate_mock_pairs(n=10_000, n_genes=100, n_latent=32, seed=42)`
which produces a synthetic dataset matching Contract 2's schema:

```python
# Defined in src/data/perturbation_pairs.py — Agent A writes it on Day 0.
def generate_mock_pairs(
    n: int = 10_000,
    n_genes: int = 100,
    n_latent: int = 32,
    n_combo: int = 1_000,
    seed: int = 42,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate synthetic pairs matching the real Contract 2 schema.

    Used by Agent B (or anyone) to unblock dynamics training before real
    Norman data is processed. Each perturbation has a learned-on-rng
    constant Δz signature so the dynamics model has a non-trivial learning
    target, but the data does NOT come from biology. Do not report metrics
    from mock runs.
    """
```

Agent A guarantees this function is implemented by **end of Day 0**. Agent B may then start
dynamics development immediately without waiting for Phase 1 to complete.

Similarly, if Agent A needs a dynamics model checkpoint for visualization before Agent B has
trained one, Agent A may instantiate the architecture directly from `config/dynamics.yaml`
with random weights and run rollouts — clearly labelled in figures as "untrained baseline."

## 6. Daily sync protocol

At the end of every work session, each agent appends an entry to `PROGRESS.md` using the
format in CLAUDE.md §8. The entry must contain:

- Phase number + status (e.g. "Phase 2 — 60% complete").
- Concrete metric values (new ELBO, new R², new success rate, etc.).
- Any blockers, tagged P0 (blocking the other agent) / P1 (blocking yourself) / P2 (annoyance).
- Top 3 priorities for the next session.

When Agent A or Agent B is *blocked by a P0 from the other*, the blocked agent writes a short
note to the other agent (Slack DM or co-authored issue) and pivots to a non-blocked task —
which is almost always available because of the mock data contracts in §5.

## 7. Quality bar — what to refuse

An agent should refuse to merge code that:

- Hardcodes a path. Add the path to `config/paths.yaml` first.
- Computes a metric inline. Move it to `src/analysis/metrics.py`.
- Calls `torch.device()` outside `src/utils/device.py`. Use `get_device()`.
- Trains a model without first checking for an existing checkpoint.
- Trains RL without verifying `artifacts/dynamics/gate.json`.
- Mutates the integer counts in `adata.layers["counts"]`. Those are sacred input to scVI.
- Uses the word "normal" / "healthy" / "non-leukemic" in a context that is not explicitly
  flagged as future work. The v1 target is the unperturbed-K562 NT reference centroid.
- Saves the scVI model via raw `torch.save(model.module.state_dict(), ...)`. Use the official
  `model.save(path)`. Loading must be tested in `tests/test_integration.py`.
- Defines new metrics inside a Jupyter notebook.

## 8. Cross-references

- ARCHITECTURE.md — system diagram, concepts, decision log, failure modes.
- CLAUDE.md — sacred rules, environment setup, device snippet, PROGRESS.md format.
- PHASES.md — 14-day plan.
- DATA.md — preprocessing biology + OT pairing details.
- EXPERIMENTS.md — hyperparameters + ablation matrix.
