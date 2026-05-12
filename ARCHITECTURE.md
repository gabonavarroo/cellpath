# CellPath — Architecture

> **Scope of claim.** CellPath performs *in-silico latent-space steering over a CRISPRa surrogate
> environment learned from Norman et al. 2019 Perturb-seq data*. It is **not** a demonstration of
> therapeutic cancer reprogramming. The v1 reward target is the unperturbed K562 NT-guide
> centroid in scVI latent space, i.e. the system learns to reverse perturbation drift back to
> the unperturbed-leukemic baseline. The word "normal," "healthy," or "non-leukemic" appears
> in this codebase only when discussing future external healthy-reference work (see DATA.md).

---

## 1. System Diagram

```
                             CellPath system (4 components)

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                                  DATA PIPELINE  (Agent A)                        │
  │                                                                                  │
  │  GEO GSE133344 ──(pertpy.dt.norman_2019)──► raw AnnData ──┐                      │
  │                                                            │                     │
  │   raw counts in adata.X  (UMI integers, 19,018 genes)      │                     │
  │                                                            ▼                     │
  │      filter_cells / filter_genes ── normalize_total ── log1p ── HVG (top 2,000)  │
  │                          (used for HVG selection & visualization only)           │
  │                                                            │                     │
  │      ┌─────────────────────────────────────────────────────┘                     │
  │      ▼                                                                           │
  │   adata.layers["counts"]   ← RAW integer counts, persists across preprocessing   │
  │   adata.X                  ← normalized log1p (for HVG selection)                │
  │   adata.var["highly_variable"] = True for 2,000 HVGs                             │
  │   adata.obs["perturbation"]    = "ctrl" | "<GENE>" | "<GENE_A>+<GENE_B>"         │
  │   adata.obs["perturbation_idx"]= int (0 = ctrl, 1..N = single, encoded combos)   │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                                  scVI VAE  (Agent A)                             │
  │                                                                                  │
  │     scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=None)         │
  │     model = SCVI(adata, n_latent=32, gene_likelihood="nb",                       │
  │                  encode_covariates=False, dropout_rate=0.1, n_layers=2)          │
  │     model.train(max_epochs=400, early_stopping=True)                             │
  │     model.save("artifacts/vae/")                  # official scVI save API       │
  │                                                                                  │
  │     latent encode: Z = model.get_latent_representation()  →  shape (N, 32)       │
  │     adata.obsm["X_scVI"] = Z                                                     │
  │     z_reference_centroid = Z[adata.obs["perturbation"] == "ctrl"].mean(axis=0)   │
  │     epsilon_success      = np.percentile(                                        │
  │                                ||Z_ctrl - z_reference_centroid||_2, 90)          │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │ artifacts/vae/{model dir, latents.h5ad,
                                      │   gene_vocab.json, z_reference_centroid.npy,
                                      │   epsilon_success.json}
                                      ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                        OT PSEUDO-PAIRING  (Agent A → Agent B handshake)          │
  │                                                                                  │
  │     For each perturbation p ∈ {1..N_genes}:                                      │
  │        z_ctrl_pool   = Z[ctrl mask]                                              │
  │        z_pert_pool_p = Z[adata.obs.perturbation_idx == p]                        │
  │        T_p = ot.sinkhorn(uniform, uniform, C=||z_ctrl - z_pert||²,                │
  │                          reg=0.05)                                               │
  │        sample hard pairing once per epoch  →  (z_ctrl_i, p, z_pert_i)            │
  │                                                                                  │
  │     Fallbacks (config: pairing.method ∈ {ot, random, mean_delta}):               │
  │        random      :  random within-perturbation pairing                         │
  │        mean_delta  :  z_pert := z_ctrl + mean(Δp);    use as baseline pairer     │
  │                                                                                  │
  │     Splits: 80% perturbations train / 20% held-out for OOD gene generalization.  │
  │     Within each training perturbation: 90% pair-cells train / 10% val (primary). │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │ artifacts/pairs/{train_pairs.npz,
                                      │   val_pairs.npz, ood_pairs.npz,
                                      │   combo_pairs.npz, metadata.json}
                                      ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                           DYNAMICS MODEL  (Agent B)                              │
  │                                                                                  │
  │     gene_emb = nn.Embedding(N_genes + 1, d_emb=64)   # +1 = NO-OP                │
  │     trunk    = MLP([32 + 64] -> 256 -> 256 -> 256, residual blocks, SiLU)        │
  │     head_mu  = Linear(256, 32)                                                   │
  │     head_lv  = Linear(256, 32)                       # log σ², clamped to [-5,3] │
  │                                                                                  │
  │     forward(z, g):                                                               │
  │       h = trunk(cat(z, gene_emb(g)))                                             │
  │       mu, log_var = head_mu(h), head_lv(h)                                       │
  │       z_next      = z + mu                            # residual                 │
  │       return z_next, mu, log_var                                                 │
  │                                                                                  │
  │     loss = heteroscedastic Gaussian NLL on Δz_pred vs Δz_true                    │
  │            + λ_combo · composition_loss(z_ctrl, g_a, g_b, z_pert_ab)             │
  │                                                                                  │
  │     Primary validation gate: held-out cells within seen perturbations.           │
  │     OOD report (not gating): held-out genes (perturbations).                     │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │ artifacts/dynamics/{model.pt, gate.json,
                                      │   val_metrics.json, ood_metrics.json}
                                      ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                    RL ENVIRONMENT + PPO POLICY  (Agent B)                        │
  │                                                                                  │
  │     gym.Env "CellReprogrammingEnv-v0"                                            │
  │       obs space   : Box(R^32)                                                    │
  │       action space: Discrete(N_genes + 1)            # +1 = NO-OP / terminate    │
  │       action_mask : zero out genes already used (repeat-mask)                    │
  │       reset()     : sample z₀ from a random perturbation cluster (off-target)    │
  │       step(a)     :                                                              │
  │         if a == NO_OP:                                                           │
  │             terminated = True                                                    │
  │             success = (||z - z_ref|| < ε_success)   # NO-OP ≠ success!           │
  │         else:                                                                    │
  │             z, _, log_var = dynamics(z, a)                                       │
  │             update repeat-mask                                                   │
  │             success = (||z - z_ref|| < ε_success)                                │
  │             terminated = success or steps == K                                   │
  │         reward = −||z' − z_ref||  − λ_sparse·1[a≠NO_OP]                          │
  │                       − λ_unc·||σ(z, a)||  (optional)                            │
  │                                                                                  │
  │     PPO   : MaskablePPO from sb3-contrib (action mask via gymnasium info)        │
  │     vec   : N_ENV parallel envs (vectorized via SB3's VecEnv)                    │
  │     train : 2e6 timesteps default; ablation over (λ_sparse, K, n_envs)           │
  │                                                                                  │
  │     train_rl.py refuses to start unless artifacts/dynamics/gate.json.passed.     │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │ artifacts/rl/{ppo.zip, rollouts.parquet,
                                      │   success_curves.png, action_freq.json}
                                      ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                          ANALYSIS + DEPMAP  (Agent A)                            │
  │                                                                                  │
  │   - UMAP of latent space (ctrl vs each perturbation cluster, z_ref overlay)      │
  │   - Trajectory rendering (rollouts projected to UMAP)                            │
  │   - Latent-space metrics: silhouette, ARI on perturbation labels                 │
  │   - DepMap enrichment of RL action set vs K562 essentials (hypergeometric,       │
  │     GSEA-style preranked, null comparison: matched-size & matched-expression)    │
  │   - Per-component metrics in src/analysis/metrics.py (single source of truth)    │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications (shapes & contracts)

| Component | Input | Output | Tensor shapes |
|---|---|---|---|
| Data pipeline | GEO MTX or pertpy cached `.h5ad` | preprocessed `AnnData` | `adata.X: (N_cells, N_HVG=2000)` log1p; `adata.layers["counts"]: (N_cells, 19,018)` int counts; `adata.obs.perturbation_idx: (N_cells,)` int |
| scVI VAE | `adata` with raw `counts` layer | `Z: (N_cells, 32)` | `model.save()` API → `artifacts/vae/{model.pt,attr.pkl,var_names.csv}`; `latents.h5ad`; `z_reference_centroid: (32,)` |
| Pseudo-pairing | `Z`, perturbation labels | `(z_ctrl, gene_idx, z_pert)` triples | `train_pairs.npz`: `z_ctrl (M,32) int8/float32`, `gene_idx (M,) int32`, `z_pert (M,32) float32` |
| Dynamics MLP | `z: (B,32), gene_idx: (B,)` | `(z_next, mu, log_var)` each `(B,32)` | `model.pt` PyTorch state dict; sigmoid-clamped `log_var ∈ [-5, 3]` |
| RL env (single) | `z: (32,)`, action `int` | `obs, reward, terminated, truncated, info` | obs `(32,)`, `info["action_mask"]: (N_genes+1,) bool` |
| PPO | obs batches | discrete action distribution | sb3-contrib `MaskablePPO`; policy net `(32) → 128 → 128 → N_genes+1` |
| DepMap enrichment | RL action frequencies | enrichment table | hypergeometric / GSEA preranked / null comparisons; FDR-adjusted |

---

## 3. Concept Explanations

These seven explanations are required reading before contributing. Each is written for an
advanced undergraduate who knows ML but not computational biology. Each explanation justifies
*why* the design choice exists, grounded in both biology and ML simultaneously.

### Concept 1 — Why Negative Binomial likelihood (not Gaussian) for scRNA-seq

Single-cell RNA-seq counts arise from a stochastic capture process: each cell contains tens
of thousands of mRNA molecules, but droplet-based platforms (10x Chromium v3, used by Norman
2019) only sequence a small fraction of them. The result for each gene in each cell is an
integer count drawn from a distribution with two distinguishing features. First, the variance
of the count exceeds its mean — this *overdispersion* is biological (gene expression is
bursty: a gene flips between "on" and "off" transcriptional states) and technical (some cells
have systematically more captured molecules than others, called library size variation).
Second, the counts have very many zeros, not because the gene is unexpressed, but because the
capture efficiency is well below 100% (typically 10–30%); this is sometimes called *dropout*,
though the term is imprecise. A Gaussian likelihood is wrong on both counts: it has equal
mean and variance only at fixed σ², it places probability mass on negative values, and it
cannot model the integer-count zero inflation. The Negative Binomial distribution, parameterised
as `NB(μ, θ)` with mean `μ` and inverse-dispersion `θ`, has variance `μ + μ²/θ` — it natively
handles overdispersion. scVI (Lopez et al., 2018) trains an encoder to produce a 32-dimensional
latent z, then a decoder produces gene-wise μ and θ; the likelihood is `NB(x | μ(z), θ)`.
This is why we use `gene_likelihood="nb"` and why we **preserve raw counts in
`adata.layers["counts"]`** — the NB likelihood requires integer counts as targets, not normalized
or log-transformed values. Note: ZINB (zero-inflated NB) is available in scVI but adds an extra
parameter and is *not* needed for 10x v3 data, where the residual zeros after NB are negligible
(Svensson 2020). We default to NB and leave ZINB as an ablation in EXPERIMENTS.md.

### Concept 2 — Why 32 latent dimensions

Gene expression vectors live in a roughly 20,000-dimensional space (one dimension per gene),
but the *effective* dimensionality is far lower. K562 cells, being a clonal leukemia line, have
nearly identical genomes; their transcriptomes differ along a small number of *biological axes*:
cell cycle phase, metabolic state, differentiation pseudotime, and stress response. These axes
plus a few hundred CRISPRa perturbation directions form a manifold of intrinsic dimension on
the order of 20–50. Picking 32 is empirically motivated: scVI's published K562 analyses use
`n_latent=10..30`; Norman et al.'s own analysis uses ~50 PCs. We pick 32 as a midpoint that
captures ≥99% of the perturbation-relevant variance while remaining low enough that geometric
notions like Euclidean distance remain meaningful. This matters for RL: the dimensionality
curse says that in high dimensions, distances concentrate (everything is equidistant), the
volume of an ε-ball around a target shrinks exponentially, and reward signals become uniform.
A 32-dim latent is small enough that ‖z − z_ref‖ remains a useful gradient for the policy.
Latent dim is the most-ablated hyperparameter in this project: EXPERIMENTS.md compares
`{16, 32, 64}` on the same data with the same VAE training budget, measured by (i) ELBO,
(ii) silhouette on perturbation labels, (iii) downstream PPO success rate after dynamics
training. We expect 32 to dominate; if 16 wins on RL success, that signals our reward signal
is brittle and needs reshaping.

### Concept 3 — Why a residual connection in the dynamics model (predict Δz, not z')

CRISPRa perturbations applied to a single gene typically move a K562 cell by a small step in
latent space — usually ‖Δz‖ ≲ 1.0 in scVI 32-dim coordinates. The *identity transform*
`f(z, g) = z` is therefore already an extremely strong baseline. A neural network asked to
predict `z'` directly from `z` and `g` would waste most of its capacity learning to copy
`z` to its output. A residual parameterization `z' = z + Δz_pred` shifts the learning target
from "the absolute next latent" to "the perturbation-specific displacement", which is what the
data actually contains signal about. This is the same inductive bias that residual networks
(He et al. 2015) introduce into vision and that Latent ODEs / Neural ODEs use for continuous
dynamics — when the next state is *close to* the current state, learn the difference, not the
whole thing. The bias is especially important under our data constraints: cross-sectional
Perturb-seq gives us paired *populations*, not paired *individual cells* (see Concept 7), so
the signal for `z'` is corrupted by the choice of pseudo-pairing. The residual head reduces
the model's variance against pairing noise — if the OT pairing produces an imperfect match,
the worst the residual head can do is fall back to `Δz ≈ 0` (i.e. predict no effect), which
is a sensible posterior. Heteroscedastic outputs `(μ, log σ²)` further allow the model to
admit uncertainty when pairings are ambiguous (high `σ²` for OOD genes, low `σ²` for
well-sampled perturbations), which we then exploit as an exploration bonus and as a stopping
criterion in the RL agent.

### Concept 4 — Why PPO (and specifically MaskablePPO), not DQN, SAC, or MCTS

The choice of RL algorithm is forced by four properties of the CellPath MDP:

1. **Discrete action space of moderate size (~100–200)** — small enough that explicit
   per-action value heads are tractable, large enough that fine-grained exploration matters.
2. **Cheap surrogate environment** — each `step()` is one neural-network forward pass through
   the dynamics model, so we can afford millions of timesteps; sample-efficiency pressure is
   moderate, not extreme.
3. **Sparse-ish reward with a dense distance shaping term** — the negative-distance shaping
   makes the reward smooth on most of the state space, with sparseness only at the success
   threshold; this is friendly to policy-gradient methods.
4. **Per-step action constraints** — we mask repeated genes (a second activation of the same
   gene is biologically meaningless and reward-hackable) and may add budget constraints in
   ablations.

DQN handles discrete actions but its off-policy replay buffer + epsilon-greedy exploration
struggles when reward shaping is informative; DQN also has no native action-masking mechanism.
SAC-Discrete (Christodoulou 2019) is competitive for moderate discrete spaces and has good
sample efficiency, but its automatic entropy tuning is brittle in practice and it lacks native
masking. MCTS / MuZero-like model-based planning could exploit our learned dynamics directly,
but the implementation cost is high for a 14-day thesis. PPO is the practical sweet spot:
on-policy (no off-policy correction needed), entropy-bonus exploration that is well-understood
to balance discovery and convergence, stable in `stable-baselines3`, and `sb3-contrib`'s
`MaskablePPO` provides exact native support for per-step action masks via the
`gymnasium` info dict. We default to MaskablePPO and document SAC-Discrete as a planned
ablation in EXPERIMENTS.md to verify our choice empirically rather than dogmatically.

### Concept 5 — Why the sparsity penalty λ matters (biologically and computationally)

The reward `R = −‖z − z_ref‖ − λ·1[a ≠ NO_OP]` includes a *per-step intervention cost*. Setting
λ = 0 would let the agent freely use all 10 episode steps even when 2 would suffice, because
distance is monotonically non-increasing under a correctly-trained policy. Adding λ > 0 changes
the optimal-policy preference from "any sequence reaching the threshold" to "the *shortest*
sequence reaching the threshold." Biologically, this is non-trivial: a clinician contemplating
an actual reprogramming protocol would prefer two interventions over ten even if both reach the
target, because each intervention adds delivery cost, off-target risk, and immune response
likelihood; this preference is implicit in actual oncology trial design. Computationally, λ
acts as an L0-like regularizer on the action sequence and prevents reward-hacking by trivial
strategies (e.g., applying random genes repeatedly until distance stochastically falls below
the threshold). Choosing λ is a calibration problem: too small and policies become wasteful,
too large and policies refuse to act and accept the NO_OP penalty. EXPERIMENTS.md ablates
λ ∈ {0.0, 0.01, 0.05, 0.1, 0.2} and reports the Pareto frontier of (final distance, sparsity).
The recommended default `λ_sparse = 0.05` is chosen to put the median successful policy at
3–5 interventions, which matches the literature on minimal-cocktail reprogramming protocols
(Takahashi-Yamanaka 4-factor cocktail being the prototypical example, even if biologically
distinct from our setting).

### Concept 6 — What DepMap validation actually tests (and what it does not)

DepMap (Cancer Dependency Map; Tsherniak et al. 2017) is a public resource cataloguing
genome-wide CRISPR loss-of-function and RNAi essentiality scores across ~1,000 cancer cell
lines, including K562. For each gene, the "Chronos" score (Dempster 2021) summarizes how much
K562 viability drops when the gene is knocked out: more negative ⇒ more essential. CellPath's
RL agent selects genes for *CRISPRa activation* (gain of function). These are different
modalities — knocking out essential gene X kills cells, but activating it may have any effect.
So DepMap cannot directly validate that an RL-selected gene "works" in our forward direction.
What DepMap *can* test is **biological plausibility**: do RL-selected genes overlap with
functionally important genes in K562 more than chance? We measure this three ways:
(1) **Hypergeometric enrichment** of the top-K RL actions against (a) K562 essentials, (b)
MSigDB Hallmark gene sets, (c) hematopoietic-lineage and leukemia gene panels;
(2) **GSEA preranked** test of RL action frequencies against the K562 Chronos score
distribution; (3) **Null comparison** against (i) random gene sets of matched size and (ii)
random gene sets matched to the RL action expression-level distribution (to control for
selection bias toward high-expression / high-HVG genes). The test reports z-scores and
FDR-adjusted p-values. **What this does not test:** whether the discovered sequences would
actually reprogram K562 cells experimentally; whether the dynamics model has learned causal
rather than correlational structure; whether overlap with cancer-relevant genes implies the
RL agent has identified therapeutic targets. Those claims require wet-lab validation that is
out of scope for this thesis.

### Concept 7 — The Markov assumption and what it costs us

The MDP formulation `p(z_{t+1} | z_t, a_t)` asserts that the future is independent of the
past given the current latent state and the current action. This is convenient for RL but
biologically incorrect in three ways. First, **gene regulatory networks have epigenetic
memory**: a transcription factor activated five hours ago may have triggered chromatin
remodelling that persists even after the TF protein has degraded. Second, **single-cell
transcription is bursty**: the apparent state at time t reflects the *current burst* of a
small set of genes, not the time-averaged expression — the same cell measured a few minutes
later might look quite different. Third, and most importantly for our setting, **Perturb-seq
is cross-sectional**: we never observe `(cell_i at time t, cell_i at time t+1)`. We observe
populations of unperturbed cells and populations of perturbed cells, and we *infer* a
displacement function via optimal-transport pseudo-pairing (see DATA.md §pairing). This means
the dynamics model `f_θ(z, g)` learns *population-level perturbation effects*, not true
single-cell temporal transitions. We mitigate this in three ways: (i) the dynamics model has
heteroscedastic outputs so it can admit uncertainty where pairings are unreliable; (ii)
episode length is bounded (K ≤ 10 steps) so any accumulated Markov-violation error is
limited; (iii) ARCHITECTURE.md, AGENTS.md, and the thesis presentation all frame results as
*in-silico exploration over a Markov surrogate*, not as proven biological trajectories.
This is the single most important honesty constraint in the project. Any claim that says
otherwise is wrong and must be rejected in code review.

---

## 4. Design Decisions Log

| ID | Decision | Alternatives | Why this choice |
|---|---|---|---|
| **D1** | NB likelihood for scVI | Gaussian, Poisson, ZINB | NB handles overdispersion natively; ZINB adds a parameter with no measurable gain on 10x v3 K562 data (Svensson 2020). |
| **D2** | 32-dim latent | 16 / 32 / 64 / 128 | Empirical sweet spot per scVI K562 published analyses; small enough that Euclidean distance is informative for RL. Ablation matrix in EXPERIMENTS.md. |
| **D3** | Residual heteroscedastic dynamics MLP | Direct z→z', conditional VAE, Neural ODE | Residual head matches Δz ≪ z magnitude observed in Norman; heteroscedastic head gives the uncertainty needed for the validation gate and exploration. |
| **D4** | OT pseudo-pairing (Sinkhorn, ε=0.05) with random + mean-delta fallbacks | Random within-perturbation, k-NN in PCA, distributional NLL | OT is the published best practice (Bunne et al. 2023 *Nature Methods* CellOT); random/mean-delta kept as fallbacks so dynamics & RL development never block on OT instability. |
| **D5** | MaskablePPO baseline; SAC-Discrete ablation | DQN, vanilla PPO, MCTS, MuZero | On-policy + entropy exploration + native action masking is the cleanest fit; SAC-Discrete kept as ablation to verify empirically. |
| **D6** | Action repeat-mask on by default | Allow repeats, soft penalty | Re-activating an already-activated gene is biologically meaningless and reward-hackable. |
| **D7** | NO-OP terminates the episode but is success-conditional | NO-OP = success, NO-OP = failure, NO-OP doesn't exist | NO-OP allows policies to stop early to save sparsity cost, but **does not count as success unless ‖z − z_ref‖ < ε**. This prevents a degenerate policy that always emits NO-OP. |
| **D8** | ε_success is data-driven from NT-control distance distribution | Hardcoded ε = 0.5 | We compute `ε = percentile(‖z_ctrl − z_ref‖, 90)` once after VAE training. This makes the threshold scale with the geometry of the learned latent space rather than a fixed Euclidean number. The exact percentile (90 default, 95 / 99 ablated) is in `config/rl.yaml`. |
| **D9** | Combo perturbations: 80% train / 20% held-out composition test | Ignore, use all for training, use all for test | The 20% held-out combos answer "does the agent's *sequential* policy approximate *true* combinatorial biology?" — a genuinely novel evaluation question. |
| **D10** | Validation: primary gate on held-out cells within seen perturbations; OOD report on held-out genes | Single mixed gate, gene-only gate | Primary gate measures *generalization across cells* (most likely to hold up under domain shift), OOD measures *generalization across perturbations* (harder, reported but not gating, because Norman has no gene-feature side info). |
| **D11** | Polars for tabular; pandas only at scanpy/anndata boundary | All pandas, all polars | Polars is materially faster on the DepMap join and gene-vocab work; scanpy requires pandas internally, so we accept that boundary. |
| **D12** | Dual Docker (CUDA + CPU) + native uv venv for Mac dev | Single image, native-only, conda | MPS doesn't pass through to Linux containers; native venv preserves MPS for Mac dev while Docker images cover cluster + CI. |
| **D13** | Hydra for config; uv for env management; pre-commit for lint | argparse, conda, manual lint | User-specified Hydra; uv is the fastest viable Python package manager; pre-commit catches drift early. |
| **D14** | scVI save/load via official API (`model.save()` / `SCVI.load()`) | Raw `state_dict` serialization | scVI registers AnnData fields and stores them next to the state dict; manual `state_dict` loading silently breaks when AnnData schema differs. We test load-after-save in `tests/test_integration.py`. |
| **D15** | nbstripout for notebook outputs; metric logic stays in `src/analysis/*` | Notebooks may define ad-hoc metrics | Notebooks are visualization layers only. New metrics must live in `src/analysis/metrics.py` and be tested. Prevents metric drift between repo and presentation figures. |

---

## 5. Failure Modes & Recovery

### 5.1 VAE

| Failure | Symptom | Recovery |
|---|---|---|
| Posterior collapse | ELBO improves but `Z` clusters degenerate; control and perturbed cells overlap | Re-run with `n_layers=2`, `n_hidden=256`, longer warm-up; verify `library_size` is being estimated; check that `layer="counts"` is actually integer-valued |
| Dominant batch effect | UMAP shows technical (lane/batch) clusters | Set `batch_key` to the relevant `adata.obs` column in `setup_anndata` |
| Reconstruction fails on rare genes | Some HVGs have near-zero variance | Re-run HVG selection with `flavor="seurat_v3"` on raw counts before HVG filtering |

### 5.2 Pseudo-pairing

| Failure | Symptom | Recovery |
|---|---|---|
| OT Sinkhorn diverges | Loss = NaN or transport plan all-zero | Lower `reg` (ε), normalize cost matrix by its median, increase `numItermax` to 1000. If still failing, fall back to `pairing.method=random` |
| Per-perturbation cell count too small | Some perturbations have < 30 cells | Increase the minimum-cell threshold in preprocessing; merge low-count perturbations into the held-out set |

### 5.3 Dynamics model

| Failure | Symptom | Recovery |
|---|---|---|
| `log_var` collapses to lower clamp (−5) | Model is overconfident; calibration test fails | Increase `log_var` regularization or add deep ensemble (3 models, average mean, max variance) |
| Validation gate fails | Held-out R² ≤ per-gene mean-Δ baseline | Investigate: bad VAE? bad pairing? Check pairs/metadata.json for n_per_perturbation; check `ood_metrics` to distinguish memorization vs generalization failure |
| Composition loss spikes | `λ_combo` term dominates | Lower `λ_combo` from 0.5 to 0.1; verify combo pairs are well-formed |

### 5.4 RL

| Failure | Symptom | Recovery |
|---|---|---|
| Policy emits NO-OP every step | Sparsity penalty too high or reward too negative everywhere | Lower λ_sparse; verify ε_success is plausible (90th percentile, not 50th) |
| Policy ignores NO-OP, hits step limit | NO-OP path always dominated | Add small per-step cost; reduce K from 10 to 5 |
| Distance gradient is flat | Reward → constant across episode | Verify dynamics is actually moving z; check repeat-mask isn't blocking all genes too early |
| Action distribution collapses to single gene | Entropy too low | Increase `ent_coef` from 0.01 to 0.1 |

### 5.5 DepMap enrichment

| Failure | Symptom | Recovery |
|---|---|---|
| All p-values significant | Probably testing against wrong background | Verify background set = HVGs (or perturbed genes), not all 19,018 genes |
| All p-values insignificant | Either no signal or testing wrong gene set | Try alternate panels (Hallmark Hematopoiesis, leukemia driver genes); inspect raw action frequencies |

---

## 6. Integration Topology

```
                 docker compose --profile cuda up
                            │
   ┌────────────────────────┴──────────────────────────┐
   │                                                   │
   │  cellpath:cuda  (GPU training service)            │
   │    PYTHONPATH=/workspace                          │
   │    bind /workspace ← repo                         │
   │    bind /workspace/artifacts ← named volume       │
   │      └─► python -m src.pipeline run               │
   │             ├─ data       → data/processed/...    │
   │             ├─ train_vae  → artifacts/vae/...     │
   │             ├─ pairs      → artifacts/pairs/...   │
   │             ├─ train_dyn  → artifacts/dynamics/.. │
   │             ├─ (gate?)    → artifacts/dynamics/gate.json
   │             ├─ train_rl   → artifacts/rl/...      │
   │             └─ evaluate   → artifacts/eval/...    │
   │                                                   │
   │  tensorboard  (sidecar, port 6006)                │
   │    reads cellpath-artifacts volume                │
   │                                                   │
   └───────────────────────────────────────────────────┘
```

Local Mac dev (Apple Silicon):
```
   uv venv .venv --python 3.11
   uv pip install -e ".[dev]"
   source .venv/bin/activate
   PYTHONPATH=. python -m src.pipeline run --dry-run
```
Native uv venv is used for MPS support; Docker images are reserved for cluster + CI.

---

## 7. Cross-References

- **DATA.md** — preprocessing biology, OT pairing details, DepMap schema, cross-sectional honesty.
- **AGENTS.md** — interface contracts, agent missions, conflict zones.
- **PHASES.md** — 14-day plan referencing this architecture's validation gates.
- **EXPERIMENTS.md** — ablation matrix referencing this document's design decisions.
