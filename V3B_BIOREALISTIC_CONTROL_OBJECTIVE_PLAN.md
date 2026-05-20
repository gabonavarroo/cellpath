# V3B — Biorealistic Control-Objective Plan

> **Target output file (post-approval):** `V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md` at repo root.
> **Plan-mode constraint:** this draft is authored here; on ExitPlanMode, copy to the repo path.
> **Required sub-skill for execution:** `superpowers:subagent-driven-development` (per CLAUDE.md) once we exit plan mode and start V3B Phase 1.
> **Author role:** V3 research lead (AI). **Audience:** the future executor agent + the user.
> **Do not start implementation in plan mode. Do not interrupt the running Track N VAE.**

---

## 0. Context — why V3B exists

V2 shipped (`artifacts_v2/V2_FINAL_REPORT.md`) with the honest finding **"PPO compressed a 2-step lookahead into a feedforward controller, not PPO discovers a superior strategy"**: under the V2 hard benchmark (K=3, ε=p25=3.166, bin 8–10, OOD, 4 seeds × 300 ep), `RoR_corr010 × C2` reaches `0.941 ± 0.048` success at the primary cell, **+77 pp over random**, but `PPO − greedy_dyn_2 ∈ [−0.106, −0.012]` — never ≥ +0.05 pp at any of the 8 hardness-frontier cells.

V3A Track L (`artifacts_v3/interpretation/v3a_checkpoint.md`) tested the V3 latent-dim hypothesis on legacy 64D NB. Result: dynamics passes safety (OOD Pearson 0.515, beam reach 100% at K=3/bin 8-10/OOD, ε_p25 = 3.187 ≈ V2's 3.166, pairing-noise median 0.886 ≈ V2's 0.89) **but fails the V3 hypothesis** — `greedy_dyn_2` is **more** saturated in 64D-legacy than in V2 32D (`grd_2` at K=2/bin 8-10/OOD: 0.853 vs V2's 0.300; at K=2/bin 6-8/OOD: 0.963 vs V2's 0.790; at K=3 primary: 1.000 = V2). Track N (fresh 64D NB VAE) is still training (epoch ~43/400 at last checkpoint) and **must not be interrupted**; its result determines whether the legacy-VAE-config hypothesis lives.

V3A's diagnosis is correct but partial. The real issue is not "64D is the wrong dim" — it is that **the current MDP is too simplified to admit a biologically meaningful planning advantage**:

* Objective is **pure latent L2 to one centroid**. No safety, viability, uncertainty, incompatibility, or feasibility terms.
* Dynamics are **deterministic in mean** (heteroscedastic head exists but uncertainty is not in the reward at λ_unc=0).
* Horizon is **K=3** with a per-step penalty `β=0.05`. Anything longer is discouraged regardless of value.
* Reference state is **the unperturbed-K562 centroid** — a Gaussian centre in scVI space, so progress is locally smooth.
* No biological constraints on which gene combinations are permissible or safe.

These five properties make greedy_dyn_2 a near-Bayes-optimal planner; planning depth and reward shape just don't matter. **V3B reframes the problem so that planning depth, safety, and feasibility matter — without making greedy fail by arbitrary masks.**

---

## 1. Why is greedy performing so well? (Diagnosis)

### 1.1 Five compounding causes

1. **L2-to-centroid is a smooth, convex-near-the-target objective.** The reference centroid `z_ref` is the empirical mean of ~11 855 control cells in scVI latent. Around any point sufficiently close, the squared-L2 surface is locally a quadratic bowl. Greedy local descent on a learned mean dynamics converges fast on quadratic bowls.
2. **Mean dynamics are deterministic given (z, gene).** `PerturbationDynamicsModel.forward(z, g)` returns `(z + μ(z, g), μ, log_var)`. Greedy_dyn_2 evaluates `‖z + μ_a + μ_b‖²` exactly — no expectation, no risk. The heteroscedastic head exists but is **not used by greedy and not in the reward** by default (`reward.lambda_unc=0.0`).
3. **Short horizon ⇒ no temporal credit-assignment ambiguity.** K=3 with NO-OP termination + repeat-mask leaves at most `C(105, 3)` deterministic candidate plans per start state — a few thousand evaluations. The depth-2 beam search at width 20 covers > 99 % of value-relevant prefixes.
4. **No counterfactual constraint on gene choice.** Genes are interchangeable in the action space; no gene is forbidden, no pair is forbidden, no gene incurs a viability penalty. Every gene that pulls `z` toward `z_ref` is admissible.
5. **Curriculum and ε set together make K=3 cells partially saturated.** ε_p25 = 3.166 corresponds to a centroid-relative radius achievable in 1–2 effective interventions in the bin 8–10 OOD pool. `greedy_dyn_2` saturates at 1.000 at the primary cell, leaving **no room** for any controller to beat it by ≥ +0.05 pp without **changing the cell** (i.e., changing the objective).

### 1.2 When is greedy success legitimate?

Greedy is **the right answer** when:
* The dynamics field is locally well-conditioned (Hessian of distance-to-target is positive definite at every state under at least one admissible action).
* All actions are equally safe / feasible / certain.
* There is **no externality** that depends on *which* path is taken to the target — only the terminal state matters.

This is what V2 has been measuring. V2 is **honest about it**, which is why we ship that result and pivot to V3.

### 1.3 When does greedy success indicate oversimplification?

Greedy success is **misleading** when any of the following are true but **not represented in the reward**:

| Real-world feature absent from MDP | Why greedy still wins our benchmark |
|---|---|
| Temporary worsening required (a 2-step path that first increases distance to set up a state-dependent gene effect that crashes distance) | Our dynamics are smooth-in-z; no such non-monotonic structure is learned, so it doesn't appear at evaluation. |
| Some gene actions are toxic / essential / risky for K562 viability | We do not penalise them. DepMap K562 Chronos is loaded only for plausibility checks, never as a reward term. |
| Some gene pairs are mutually incompatible (synthetic lethal in K562 — Horlbeck reports 1,523 SL pairs in K562) | We have no SL mask; every pair is admissible. |
| The dynamics model is uncertain in some regions | `λ_unc = 0.0` by default; uncertainty does not flow into reward. |
| Long paths are valuable computationally even if hard in the lab | `β·t` linear step cost punishes them in proportion to length, no concave / threshold structure. |
| The "success" set is multi-dimensional (close to centroid AND on the unperturbed manifold AND no off-target hallmark scores) | Our ε is a single L2 ball. |

**Each of these is a falsifiable V3B hypothesis.** V3B tests them by adding constraints/penalties one at a time and observing whether `PPO − greedy_dyn_2` grows.

---

## 2. What would make greedy perform poorly for a scientifically valid reason?

The litmus test for "valid" is: **the constraint corresponds to a measurable biological signal we did not invent to defeat greedy**. The valid candidates:

1. **State-dependent gene effects from the heteroscedastic head.** If the action selected at step 1 lands in a high-uncertainty region under the learned `log σ²`, the *expected* downstream value differs from the *deterministic-mean* downstream value. A policy that accounts for the entire predictive distribution (via uncertainty penalty in reward) can avoid such states, while greedy_dyn_2 — which uses only the mean — cannot.
2. **Toxicity / essentiality penalties (DepMap K562 Chronos, OGEE v3, Replogle 2022 K562 essentials).** Some genes in the 105-gene action space are known K562 dependencies (Chronos < -0.5). Activating an essential gene in CRISPRa is not directly lethal, but it represents an experimentally hostile / off-target rich intervention. A safety-weighted objective will route around them; greedy_dyn_2 will not.
3. **Synthetic-lethal pair forbidness (Horlbeck 2018, SLKB, SynLethDB).** Of the 1,523 reported K562 SL gene pairs, an unknown but non-zero subset is contained in `pairs((84 train ∪ 21 OOD genes), 2)`. A path containing such a pair is biologically inadmissible; greedy_dyn_2 may select one if it minimises distance.
4. **Path-length non-linearity with a free band.** A `g(K) = β·K · 1[K ≤ K_free] + (β·K_free + γ·(K − K_free)^α) · 1[K > K_free]` schedule lets the policy explore K=4..K=10 cheaply, only paying super-linear cost when it would never be feasible in the lab. Greedy_dyn_2 is depth-limited and has no notion of "computational" vs "feasible" horizons.
5. **Stricter ε with reachability gating.** ε ∈ {p10, p15} would *reduce* greedy_dyn_2 saturation if and only if the reachability oracle confirms the cell remains solvable. V3A Track L already shows ε_p25 ≈ V2; tighter ε must be coupled with a beam-reachability check.
6. **A multi-objective target.** "Close to centroid AND off the leukemic-perturbation manifold AND no SL violation AND no high-essential picks AND low peak uncertainty". This is a *vector* objective; reduction to a scalar at evaluation time can use weighted sums or Pareto reporting.

**V3B does not ask "what would make greedy fail?". It asks: "what biology, present in the data but absent from our MDP, would make greedy *suboptimal* — and can we get PPO to discover the routes that respect it?"**

---

## 3. Biological data sources

Below: a tiered map of resources, what they give, how to map to the 105 CellPath perturbation genes, and the failure modes.

### 3.1 Already in repo (no download)

| Resource | What it gives | Map to 105 genes | Use as | Risk |
|---|---|---|---|---|
| **DepMap K562 Chronos** at `data/processed/depmap_k562_chronos.parquet` and `data/raw/depmap_chronos.csv` (loader: `src/analysis/depmap_validation.py::load_depmap_k562`). Columns: `gene_symbol`, `chronos`, `is_essential` (`Chronos < -0.5`) | Per-gene K562 dependency score in [−2.5, +0.5] approx. More negative = more essential. | Direct gene-symbol join on `gene_vocab.json::genes`. Coverage check: ~100/105 expected based on V1 DepMap comparison work. Genes missing in DepMap get `tox=0` (neutral). | Soft penalty in reward (V3B Reward C, D, E). Action mask only at extreme threshold (`Chronos < −1.5`) and only as ablation. | **Leakage**: Chronos was computed from CRISPR-Cas9 knockout, not CRISPRa activation. Direction mismatch — see §3.5. |
| **Norman 2019 K562 CRISPRa Perturb-seq** at `data/processed/norman_hvg.h5ad` and `data/raw/norman_2019.h5ad` | Our training data: 11 855 control cells + 19 264 genes; 105 perturbations (84 train + 21 OOD) including 100 singles + 31 pairs (per [Norman 2019, Science](https://www.science.org/doi/10.1126/science.aax4438)). | Ground truth — defines the 105-gene universe. | Post-hoc only; should not be used as a reward term (already the data-generating process). | None — already in pipeline. |
| **Norman 2019 combo pairs** in `combo_pairs.npz` (35 995) | Empirical post-double-perturbation latents for 31 dual-guide combos. | 31 specific pairs of the C(105, 2)=5460 possible. | Post-hoc realism scoring: compare PPO 2-step rollout endpoints to empirical Norman 2-perturbation cells when the gene pair coincides. | Sparse — only 31 pairs covered. |

### 3.2 Public datasets to download (V3B Phase 0)

| Resource | URL / DOI | What it gives | Map | Use as | Risk |
|---|---|---|---|---|---|
| **Replogle 2022 K562 essential CRISPRi Perturb-seq** | [Replogle et al. 2022, Cell](https://www.sciencedirect.com/science/article/pii/S0092867422005979); [GSE GSE264667 / figshare 20029387](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387) | ~400 K cells × 2 057 common essential genes in K562 (day-6 CRISPRi). | Gene-symbol join. ~95–100 of 105 expected to overlap with the essential set or its expressed neighbours. | (a) **K562-specific essentiality signal** to cross-validate Chronos (Chronos is multi-cell-line; Replogle is K562-only CRISPRi). (b) Post-hoc realism: for each PPO-selected gene, does Replogle show CRISPRi knockdown caused growth defect? | CRISPRa ≠ CRISPRi direction; same caveat as DepMap. |
| **Horlbeck 2018 K562 dual-CRISPRi GI map** | [Horlbeck et al. 2018, Cell](https://www.cell.com/cell/pdf/S0092-8674(18)30735-9.pdf); GEO GSE116198 | GI scores for 222 784 pairs across 472 K562 genes. 1 523 SL pairs reported (per [SLKB](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767912/)). | Restrict to pairs where **both** genes are in the 105-gene universe (intersection cardinality TBD — likely 30–100 pairs after intersection). | **Forbidden-pair soft penalty** in reward (E variant). Hard mask only as ablation; cannot mask Norman's own 31 measured combos because Norman saw the experimental outcome already. | Cell-line specificity (1,523 in K562, only 82 overlap Jurkat — SL is highly cell-line-specific). CRISPRi vs CRISPRa direction. |
| **OGEE v3** | [OGEE v3 NAR 2021](https://academic.oup.com/nar/article/49/D1/D998/5934414); v3.ogee.info | Cross-cell-line essentiality across 150 cancer lines via CRISPR-Cas9 + RNAi. | Gene-symbol join; provides "common-essential / conditionally-essential / non-essential" categorical label. | Cross-validation for DepMap. **Common-essential** is the safest "do not activate frivolously" set. | None unique beyond DepMap. |
| **COSMIC Cancer Gene Census (CGC)** | [Sondka 2018, Nature Rev Cancer](https://www.nature.com/articles/s41568-018-0060-1); cancer.sanger.ac.uk/cosmic/census | Tier-1 / Tier-2 oncogene / TSG / fusion-gene annotation. | Gene-symbol join. Expect 25–60 of 105 to be tier-1/2 (high prior — Norman selected GIs for cancer-relevance). | **Path-realism annotation** (post-hoc). Activating a TSG via CRISPRa makes biological sense as restoration; activating an oncogene does not. This *flips the sign* of the safety penalty depending on annotation. | Coarse-grained (binary); does not capture context. |
| **Open Targets tractability** | [Open Targets Platform NAR 2024](https://academic.oup.com/nar/article/53/D1/D1467/7917960); platform-docs.opentargets.org/target/tractability | SM/AB/PR/OC druggability tier per gene. | Gene-symbol join. | **Feasibility post-hoc score** only. Not a reward term (CRISPRa is not a drug; we are not selecting drug targets). | Pharma-modality scoring is irrelevant to CRISPR pre-clinical targets; we'd over-weight pharma. |

### 3.3 Synthetic-lethal / GI databases (auxiliary, not strictly K562)

| Resource | URL | Caveat |
|---|---|---|
| **SLKB** (synthetic lethality knowledge base) | [SLKB NAR 2024](https://academic.oup.com/nar/article/52/D1/D1418/7331024) | Aggregates 11 CDKO experiments × 22 cell lines; K562 is one. Five SL calculation methods (Median-B/NB, sgRNA-B/NB, Horlbeck, GEMINI, MAGeCK). Use the **K562 partition only**. |
| **SynLethDB** | [Guo et al. 2016 NAR](https://academic.oup.com/nar/article/44/D1/D1011/2502617) | Multi-species; manual curation; many entries are not K562. Filter to human + K562/leukemia + CRISPR-based. |

### 3.4 Recommended V3B biology layer (the *minimum* set)

To keep scope small and decisive, V3B initially uses **only DepMap K562 Chronos + Horlbeck-2018-restricted-to-K562**, plus a derived "common-essential" flag (`Chronos < -0.5` OR OGEE-common-essential). Other sources are post-hoc annotations to enrich the eval report — not reward terms.

### 3.5 Failure modes and leakage risks

1. **CRISPRa ≠ CRISPR-knockout direction.** A gene with `Chronos = −1.5` is essential under loss-of-function; CRISPRa *activates* it, which may be biologically neutral, hyperactivating-toxic, or differentiation-inducing. Treat Chronos as **prior probability of experimental disturbance**, not toxicity per se. Code comment + report caveat are mandatory.
2. **Evaluation-objective leakage.** DepMap was used in V1/V2 for *plausibility checks* on PPO action frequencies. If we now put DepMap **into the reward**, the V2 plausibility tests become tautological. Mitigation: when reward includes Chronos penalty, the plausibility eval must use **held-out** plausibility evidence (Replogle essentials, OGEE OOD subset, hallmark panels — not Chronos directly).
3. **Cell-line specificity of SL pairs.** Horlbeck reports 1,523 K562 SL pairs; only 82 overlap Jurkat. We use the **K562 set only** and document non-transferability.
4. **Coverage gaps.** Not all 105 genes are guaranteed to have Chronos / SLKB entries. Default = no penalty for missing data, but log the coverage in every run's metadata.json.
5. **Confounding with greedy.** If the safety penalty disproportionately punishes the genes greedy_dyn_2 picks, we may declare V3B success while having only handcrafted a discriminator. Mitigation: at acceptance time, **report PPO − greedy_dyn_2 on the same penalised reward**. If greedy is also re-evaluated under the new reward and PPO still wins by ≥ +0.05 pp, the win is real.

---

## 4. Reward and constraint redesign

Six reward families. All keep the existing terminal_only_step_cost mid-episode structure (R_t = 0 mid-episode) **except where noted**, to preserve the V2 hard-bench protocol. Variables: `d_T = ‖z_T − z_ref‖`, `is_success = 1[d_T < ε]`, `t = effective_step_count`, `a_i = i-th action in path`, `tox(g) = max(0, −Chronos_g − 0.5)`, `unc(g, z) = ‖σ(z, g)‖_2` (from heteroscedastic head), `SL = K562 SL pair set`. All `λ` are non-negative hyperparameters.

### A. Baseline (V2 primary) — `terminal_only_step_cost`
```
R_t   = 0                                            (t < T)
R_T   = 1·is_success − β·t                            (t = T)
β=0.05, ε=p25=3.166
```
Already implemented in `src/rl/reward.py::compute_reward`. Reference; do not change.

### B. Global path-length penalty (concave / free-band)
```
g(t)  = β·t                                          if t ≤ K_free
       β·K_free + γ·(t − K_free)^α                   if t > K_free
R_T   = 1·is_success − g(t)
```
Defaults: `K_free=5`, `β=0.02`, `γ=0.05`, `α=1.5`. *Intuition:* short paths (1–5 steps) are roughly free of step cost; only beyond 5 do steps incur super-linear cost, modelling the lab-feasibility cliff. *Reduces greedy advantage*: greedy_dyn_2 with this reward maps onto the same depth-2 evaluation, **but** PPO can now explore 4–5 step plans without being penalised; we should also evaluate `greedy_dyn_5` for an apples-to-apples ceiling. *Risk:* if `K_free` and `γ` are wrongly chosen we just re-shape the same MDP. *Decisive test:* train PPO with this reward + `env.max_steps=8`, evaluate at K=3, 5, 8 hardness cells; require `greedy_dyn_5` reachability ≥ 50 % at K=5/bin 8-10/OOD before declaring decisive.

### C. Safety-aware (toxicity) reward
```
R_t                = 0
tox_path           = Σ_{a_i ≠ NOOP} tox(a_i)
common_ess_penalty = Σ_{a_i ≠ NOOP} 1[CommonEssential(a_i)]
R_T                = 1·is_success − β·t − λ_tox·tox_path − λ_ce·common_ess_penalty
```
Defaults: `λ_tox=0.10`, `λ_ce=0.05`. *Intuition:* each gene activation incurs cost proportional to its K562 essentiality. Greedy_dyn_2 is re-evaluated under this reward and would still pick the highest-distance-reducer; PPO can prefer slightly worse-greedy-value genes that are safer, at K ≥ 2. *Risk:* may push policy toward NOOP (terminate early). Mitigation: keep `1·is_success` term dominant. *Decisive test:* under reward C, compute PPO action set vs greedy_dyn_2 action set, expected divergence on top-K with `Cliff's δ` ≤ −0.3 (PPO genes have more positive Chronos = less essential).

### D. Uncertainty-aware reward
```
R_t                  = 0
unc_path_max         = max_{i: a_i ≠ NOOP} ‖log σ²(z_i, a_i)‖_2
unc_path_mean        = mean_{i: a_i ≠ NOOP} ‖log σ²(z_i, a_i)‖_2
R_T                  = 1·is_success − β·t − λ_unc·unc_path_max
```
Default: `λ_unc=0.05`. *Intuition:* the heteroscedastic head provides `log σ²` per dim per state-action; route the policy through high-confidence trajectories. Greedy_dyn_2 does not use σ and cannot avoid uncertain regions. *Risk:* if the network is over-confident on OOD, uncertainty signal collapses. Mitigation: pre-flight check on V3A Track L OOD uncertainty Spearman 0.738 (PASS).

### E. Multi-objective combined
```
R_T = 1·is_success
      − g(t)                                  (path-length, from B)
      − λ_tox·tox_path                        (safety, from C)
      − λ_unc·unc_path_max                    (uncertainty, from D)
      − λ_sl·sl_violations(path)              (forbidden pair penalty)
```
where `sl_violations(path) = #{(i, j): i < j, (a_i, a_j) ∈ K562_SL ∪ (a_j, a_i) ∈ K562_SL}`. Defaults: `λ_tox=0.10`, `λ_unc=0.05`, `λ_sl=0.5`. *Intuition:* full biorealistic objective. **Should only be run after B, C, D individually have shown directional signal**, to avoid the conjunction trap where all terms simultaneously confuse PPO.

### F. Pareto evaluation (no reward change at training time)
Train PPO under A (V2 primary) **and** under B/C/D/E. At evaluation, compute the **5-vector** (success_rate, mean_steps, safety_score, peak_uncertainty, sl_violations). Plot Pareto frontier across configs. *Intuition:* greedy is one point; PPO under each reward is another point; the *frontier* defines the meaningful improvements. No single scalar required. Decisive output: a figure with at least one PPO config strictly dominating greedy on at least two of five axes without sacrificing the others by > 5 %.

### 4.1 Why each variant might reduce greedy dominance

| Variant | Greedy weakness it exploits | Strongest cell to test on |
|---|---|---|
| B (path-length free band) | greedy can't plan beyond depth=K of beam; K=5 cells unreachable by `greedy_dyn_2` | K=3 bin 10-12 OOD (out-of-band difficulty) |
| C (toxicity) | greedy picks highest-tox path; PPO learns slight-detour-around-essentials | any K where greedy's top-1 is a `Chronos < -0.5` gene |
| D (uncertainty) | greedy uses only μ, not σ; high-σ states cost extra | OOD cells where σ-Spearman is high |
| E (combined) | conjunction of all weaknesses | hardness frontier K=2/bin 8-10 OOD |
| F (Pareto) | scalar-greedy can't optimise multi-objective; PPO can | any cell |

### 4.2 Smallest decisive test (per variant)

For **each** of B, C, D, evaluate `PPO − greedy_dyn_K(same_reward)` at:
* primary cell K=3/bin 8-10/OOD (saturated under A; un-saturated under B/C/D is the first signal),
* informative cell K=2/bin 6-8/OOD,
* one cell where the reward variant is *predicted* to give PPO advantage (per the table above).

Decisive criterion: `PPO − greedy_dyn_K_same_reward ≥ +0.03 pp` at ≥ 1 cell on the V3A-track that passed safety, with 4-seed CI excluding zero on a Phase 2 seed sweep.

---

## 5. Hard masks vs soft penalties

### 5.1 Hard masks — apply only when constraints are biologically *impossible* to violate

| Constraint | Hard or soft | Why |
|---|---|---|
| Repeat-mask (gene already used in episode) | **Hard (existing)** | NO biological signal — a CRISPRa guide RNA is consumed per episode by design. Already in env. |
| NO-OP always available | **Hard (existing)** | The policy must be able to terminate. |
| `Chronos < −2.0` (extreme essential) genes blacklisted | **Soft only** as default; hard only as ablation | A gene being essential does not make CRISPRa of it experimentally impossible — just probably uninformative or toxic. Soft penalty preserves the option. |
| Norman-measured SL pairs (intra-31-pair set) | **Never masked** | Norman *measured* those pairs and we *use* the combo data; they are legitimate observations. |
| Horlbeck K562 SL pairs (intersected with our 105 genes, minus Norman's 31) | **Soft penalty in E** as default; hard mask only as ablation | The SL signal is statistical; cell-line context; we should not blacklist on one screen. |

### 5.2 Soft penalties — design principles

* **Bounded in [0, 1]** so that the reward scale stays comparable to `1·is_success`.
* **Aligned with measurable signals.** Each soft penalty must point to a row in a CSV / parquet on disk; if we cannot point to the data, we don't penalise it.
* **Reversible at inference time.** A safety-aware-trained PPO must still be evaluable under the V2 primary reward for back-compat. The env's reward_mode is the switch.
* **Composable.** Penalties add. No multiplicative interactions (those introduce coupled saturation).

### 5.3 Post-hoc scoring — for weak / single-screen evidence

For sources where the data is too narrow to put in the reward (Open Targets tractability, COSMIC CGC tier annotation, Norman-combo-overlap), V3B scores the *resulting paths* with these signals as **report columns**, not reward terms. The user reads `path_realism_scores.csv` alongside the success metrics.

### 5.4 How to avoid overfitting to external databases

* **Reserve one database for evaluation only.** V3B uses DepMap K562 Chronos **and** Horlbeck K562 SL as reward inputs; uses Replogle 2022 K562 essential CRISPRi and OGEE v3 as **held-out plausibility tests** at eval time. Train and test do not share the same biology source.
* **Permutation null on the safety reward.** For each safety reward run, also train a control PPO with the gene Chronos labels **permuted across the 105 genes** (preserves the marginal distribution but destroys the gene identity). If the permuted-Chronos PPO matches the real-Chronos PPO on success rate, the safety signal is doing nothing — declare null.
* **Report PPO − greedy_K_same_reward at every reward variant**, never PPO − greedy under a different reward (that would be apples-to-oranges).

---

## 6. Path length — should longer paths be allowed?

### 6.1 The K=3 lab constraint

V2 and V3A use K=3 because Norman 2019 measured exactly that (1 or 2 sgRNAs per cell experimentally). Beyond K=2, no empirical data confirms the dynamics field — the model is **extrapolating** the composition of effects. For lab-translation claims, K ≤ 2 is the safe band.

### 6.2 The computational-ideation regime

CellPath is **in-silico latent-space steering** (CLAUDE.md §1). We are not claiming therapeutic reprogramming. Computational ideation can permit K = 5, 8, 10 if we are honest about feasibility tiering:

| Tier | K range | Claim allowed | Comparator |
|---|---|---|---|
| Lab-feasible | K=1, 2 | Could plausibly be replicated experimentally tomorrow | Norman 2019 measured doubles |
| Plausible-extension | K=3, 4, 5 | Composition of measured effects; not directly measured | Beam-reachable under the dynamics |
| Speculative | K=6..10 | Computational exercise to probe the dynamics field's planning depth | Reachability-oracle-only; no biological claim |

### 6.3 Path-length penalty schedule

Recommended `g(t)` schedule for V3B:
```
g(t) = 0.02·t                     for t ∈ {1, 2}      (lab-feasible — cheap)
g(t) = 0.04 + 0.02·(t − 2)        for t ∈ {3, 4, 5}    (plausible — linear)
g(t) = 0.10 + 0.05·(t − 5)^1.5    for t > 5            (speculative — super-linear)
```
This replaces the V2 `β·t = 0.05·t` per-step penalty (which charges 0.15 for a K=3 path and 0.40 for a K=8 path). Under the new schedule, K=3 costs 0.08 (60 % less) and K=8 costs 0.10 + 0.05·3^1.5 ≈ 0.36 (10 % less but only after 3 super-linear steps). The result: PPO can ideate up to K=5 essentially for free, only paying for clearly-speculative depths.

### 6.4 Honest reporting per tier

Every V3B result table includes three rows per cell:
* "best lab-feasible path" (K ≤ 2),
* "best plausible path" (K ≤ 5),
* "best speculative path" (K ≤ 10),
each with own `mean_distance`, `safety_score`, `success_rate`. Greedy baselines reported at matching depths.

---

## 7. Should V3B be tested on 32D, 64D, or both?

### 7.1 Inputs to the decision

* **32D V2 primary** is the cleanest baseline. Diagnostics, hard-bench, seed CIs, final report, figures — all 32D. Track L showed 64D legacy is *more* greedy-saturated than 32D, but the V3B problem is **the reward, not the latent**. There is no a-priori reason 64D helps a biorealistic objective.
* **64D Track N** (fresh NB VAE, training as of last checkpoint) is the only valid 64D candidate. If Track N passes safety (OOD Pearson ≥ 0.40, beam reach ≥ 50 % at K=3/bin 8-10/OOD), it gives a second latent for V3B sanity check — does the biorealistic objective discover the same routes across two latents?
* **64D Track L is NOT a V3B candidate.** It failed the V3 hypothesis (saturated greedy_dyn_2 at K=2 too), and the legacy VAE has no Hydra snapshot. V3A artifacts kept for reference; V3B does not retrain on it.

### 7.2 Decision rule

Default: **start V3B on 32D V2 primary** (`RoR_corr010` dynamics + V2-OT pairs). Most experiments cost 5–10 min PPO retrain × N seeds; cheap. Once a reward variant produces directional signal on 32D, **replicate on Track N 64D** *if Track N is available and passed safety*. If Track N fails safety, skip 64D entirely and continue on 32D.

This avoids the V3A-Track-L trap of optimising on a latent that we already know is not the bottleneck.

### 7.3 Decision branches based on Track N outcome

| Track N outcome (when it finishes) | V3B branch |
|---|---|
| Track N PASSES safety AND `greedy_dyn_2` at K=3 primary < 0.95 | Treat Track N as a second V3B latent. Run Phase 1–4 on 32D first; replicate winning variants on Track N at Phase 5. |
| Track N PASSES safety AND `greedy_dyn_2` at K=3 primary ≥ 0.95 | The 64D field is also saturated under greedy_dyn_2 — same V3A signal. V3B stays on 32D; Track N becomes a sanity-check at the end (single seed, one variant). |
| Track N FAILS safety (OOD Pearson < 0.40 OR beam reach < 50 %) | Drop 64D entirely from V3B. V3.fallback.B (contraction regulariser) becomes the representation-level next step *after* V3B's control-objective axis is exhausted. |
| Track N still running when V3B Phase 1 starts | Phase 1 is post-hoc on V2 primary 32D — no Track N dependency. Proceed; revisit when Track N finishes. |

---

## 8. V3B experiment matrix

### 8.1 Two axes, separated to avoid the conjunction trap

**Axis A — representation-level fallbacks** (orthogonal to V3B's main pivot):
* A0: V2 32D (frozen — primary canvas)
* A1: Track N 64D NB (conditional on training completion + safety pass)
* A2: 64D ZINB (V3.3 from `V3_RESEARCH_PLAN.md` §4 row 4) — only if A0 and A1 both leave V3B objective unsupported
* A3: SCANVI 32D (perturbation-supervised) — only if A2 fails too
* A4: contraction-regulariser dynamics (V3.fallback.B from `V3_RESEARCH_PLAN.md` §5) — strict fallback

**Axis B — control-objective fallbacks** (V3B's core):
* B0: V2 primary reward (baseline; already done)
* B1: Path-length penalty (`g(t)` schedule from §6.3) on V2 primary
* B2: Safety-aware reward (`tox_path` from §4 Variant C)
* B3: Uncertainty-aware reward (`unc_path_max` from §4 Variant D)
* B4: SL-pair forbid penalty (variant E's `λ_sl` term only)
* B5: Combined B1+B2 (path + safety) — first conjunction; smallest risk of confusion
* B6: Combined B1+B2+B3+B4 (full Variant E)

V3B is principally an axis-B exploration on A0 (and conditionally A1). Axis A is the orthogonal fallback layer.

### 8.2 Phase-by-phase plan

**Sequential ablation order is mandatory** (user directive 2026-05-17): single-axis variants are tested **before** any conjunction. The order is **C → B → D → C+B → full E**, never combined-first. Each step's outcome must be interpreted in isolation before the next term is added.

| Phase | Goal | Latent | Reward | Cost (wall-clock) | Decisive output |
|---|---|---|---|---|---|
| **Phase 0** (biology layer build) | Build `src/analysis/path_feasibility.py`; produce per-gene safety table; download Horlbeck-2018 K562 SL pair table (GSE116198), intersect with 105-gene action space; produce CGC + OGEE annotations (post-hoc only) | none | none | 1.5 h (download adds ~30 min over Chronos-only) | `artifacts_v3/v3b_biology/{gene_safety.parquet, k562_sl_pairs.parquet, coverage.json}` |
| **Phase 1** (post-hoc scoring of V2 primary) | Run V2 primary PPO + greedy_dyn_{1,2,3,5} rollouts; score paths under biology layer; report safety_score, sl_violations, top-gene Chronos by policy | A0 | A (no retrain) | 1 h | `artifacts_v3/eval_v3b_posthoc/posthoc_summary.md`. Falsifies/confirms "greedy picks unsafe routes" — if greedy_dyn_2's top-10 genes have mean Chronos > PPO's, V3B's biology hypothesis is suspect; investigate before retraining. |
| **Phase 2** (safety on 32D — first single-axis retrain) | Train PPO on A0 with reward **C (safety-aware)**; full hardness eval; **also** re-evaluate greedy_dyn_2 under reward C; run permuted-Chronos null PPO | A0 | C | 30 min PPO + 30 min eval + 30 min permutation null | `PPO_C − greedy_dyn_2_C ≥ +0.03 pp` at ≥ 1 cell, AND Cliff δ on top-10 gene Chronos ≤ −0.3, AND permuted-Chronos PPO < real-Chronos PPO. |
| **Phase 3** (path-length on 32D — second single-axis retrain) | Train PPO on A0 with reward **B (path-length free-band)**; evaluate at K ∈ {2, 3, 5, 8} × bin {6-8, 8-10} × OOD; compare to greedy_dyn_{2, 5} **under reward B** | A0 | B | 30 min PPO + 30 min eval | `PPO_B − greedy_dyn_5_B ≥ +0.03 pp` at any K ≥ 4 cell → directional pass; AND PPO uses K ≥ 4 in ≥ 30 % of episodes. |
| **Phase 4** (uncertainty on 32D — third single-axis retrain) | Train PPO on A0 with reward **D (uncertainty-aware, `unc_path_max`)**; eval at OOD cells where dynamics OOD-σ Spearman ≥ 0.3 | A0 | D | 30 min + 30 min | `PPO_D − greedy_dyn_2_D ≥ +0.03 pp` at the high-σ-Spearman cell. |
| **Phase 5** (combined C+B on 32D — first conjunction) | **Only if both Phase 2 and Phase 3 produced directional signal.** Train PPO on A0 with reward C+B (no D, no SL yet). Compare to greedy under reward C+B. | A0 | C+B | 30 min + 30 min | `PPO_{C+B} − greedy_dyn_2_{C+B} ≥ +0.05 pp` (V3B headline) without losing any individual-phase win. |
| **Phase 5b** (full Variant E on 32D) | Only if Phase 5 passes. Add uncertainty term (D) and SL-pair penalty (E's λ_sl). Confirm or relax (drop terms whose individual phase failed). | A0 | E | 1 h | Confirm V3B headline holds under the full multi-objective reward AND each term's contribution is non-zero (ablation table). |
| **Phase 6** (transfer to Track N 64D) | Replicate the **best axis-B variant** from Phase 5 on Track N 64D **if** Track N has finished and passed safety | A1 | Best from C / B / C+B / E | 1 h (assumes pairs + dynamics already built by V3A) | Same headline criterion as Phase 5. |
| **Phase 7** (seed escalation) | If any Phase 2–6 shows `PPO − greedy_K_same_reward ≥ +0.05 pp` single-seed → escalate to 4 seeds {42, 0, 1, 7} | (whichever passed) | (the winning reward) | 4 × 30 min PPO + 4 × 30 min eval = ~4 h | 4-seed 95 % CI on Δ excludes zero. |
| **Phase 8** (representation-level fallback) | If axis B (Phases 2–7) all fail on A0 and A1, pivot to axis A — V3.3 ZINB or V3.4 SCANVI per `V3_RESEARCH_PLAN.md` §5 | A2 or A3 | TBD per result | ~3–4 h VAE + downstream | Same V2 hard-bench criterion. |

**Total V3B Phase 0–5 wall-clock**: ~6–8 hours on macOS MPS; ~3 hours on CUDA. Adding Phase 5b + Phase 6 adds ~2 h; Phase 7 (seed escalation) adds 4 h; Phase 8 (representation fallback) is a separate ~4 h block.

### 8.3 What is **always** reported per phase

Each phase's eval table includes:
* PPO success rate (mean ± seed std + 4-seed CI when Phase 7 done)
* random success rate
* greedy_dyn_{1, 2, 3, 5} success rate **under the same reward**
* `PPO − greedy_dyn_2_same_reward` and `PPO − greedy_dyn_5_same_reward`
* mean steps, mean final distance
* mean safety score (Chronos-weighted), mean SL violations, mean peak uncertainty
* fraction of safe actions (Chronos > −0.5)
* top-10 action genes per policy + their Chronos / CGC tier annotation
* path realism score (composite — defined §9.3)
* reachability oracle pass/fail at the cell

Greedy under the **same reward** is the load-bearing comparator. We never compare PPO under reward B to greedy under reward A.

---

## 9. Success metrics — what counts as V3B succeeding?

### 9.1 The V2 trap to avoid

V2's success metric was `PPO − greedy_dyn_2 ≥ +0.05 pp` at a saturated cell. We never hit it because the cell saturated *for both*. V3B's metric must be **multi-axis** and explicitly defined so that improving any one axis without degrading the others is a win.

### 9.2 V3B headline success criteria (any single one is sufficient)

1. **Planning-advantage in biorealistic objective.** `PPO_Bk − greedy_dyn_K_under_Bk ≥ +0.05 pp` at one V2-equivalent cell (K=3, ε=p25, bin 8-10, OOD) OR any K=2 cell, with 4-seed 95 % CI excluding zero. *This is the V3 stretch criterion adapted to the new reward.*
2. **Safety-equivalent success advantage.** PPO matches greedy_dyn_2 on **success rate** at the primary cell (within seed CI) but has **lower mean toxicity** (Cliff's δ ≤ −0.3 on per-episode `tox_path`), **fewer SL violations** (≥ 50 % reduction), and **lower peak uncertainty** (Cliff's δ ≤ −0.3). *This is a Pareto-dominance win on multi-objective.*
3. **Longer-path discovery.** Under reward B1, PPO reliably executes K ∈ {4, 5} paths that beam_search at depth 5 cannot find, OR has lower variance in K than greedy at matching success. *This is genuine credit assignment over a 5-step horizon.*

### 9.3 Path realism score (composite, post-hoc, **non-load-bearing** but reported)

For each episode:
```
realism = w_chr · (1 − norm_tox(path))
        + w_sl  · 1[no SL violations]
        + w_ess · (1 − fraction_common_essential(path))
        + w_nor · 1[final state on Norman-empirical-manifold]
where w_chr=0.4, w_sl=0.3, w_ess=0.2, w_nor=0.1
```
Reported alongside success_rate. *Not* used as a training signal; *not* used to declare V3B success on its own (to avoid over-fitting to the constructed score).

### 9.4 What "biological breakthrough" looks like under V3B

A V3B run that shows:
* PPO success ≥ 0.50 at the primary cell (lower than V2's 0.94 — accept the trade-off),
* PPO mean `tox_path` ≤ 0.3 (vs V2 PPO ≈ 0.6 — half the toxicity),
* PPO mean SL violations ≤ 0.05 (essentially zero),
* PPO mean steps = 3.5 (vs V2 PPO 2.7 — slightly longer, but uses safer routes),
* Cliff's δ on top-10 genes vs greedy_dyn_2: −0.4 to −0.6 (PPO picks meaningfully less essential genes)
* And `PPO − greedy_dyn_2_same_reward ≥ +0.05 pp`.

This is a defensible computational-ideation claim of *"in-silico, we discovered safer reprogramming routes than naïve planning"*. **Still not therapeutic reprogramming** (CLAUDE.md §3 rule 7) — we are reversing perturbation drift in CRISPRa K562 with a safety prior.

### 9.5 Stop conditions (V3B halts immediately)

| Trigger | Action |
|---|---|
| Reachability oracle drops below 50 % at K=3/bin 8-10/OOD on the V3B-trained dynamics | Halt; investigate; V3B uses **frozen** V2 dynamics (no retrain), so this cannot happen unless we accidentally retrained dynamics — pull the lever. |
| Phase 1 post-hoc shows greedy_dyn_2's safety/SL profile is already strictly better than PPO's | Halt and revisit hypothesis. The biorealistic-objective premise is wrong; rethink before retraining. |
| Permuted-Chronos PPO matches real-Chronos PPO on success | The safety reward is doing nothing; the comparison is null. Adjust λ or drop axis B2. |
| All B1, B2, B3 reach Phase 7 with no `PPO − greedy ≥ +0.03 pp` | Pivot to axis A (representation): V3.3 ZINB / V3.4 SCANVI. |
| Compute > 12 h wall-clock | Save state; partial interpretation; next session continues from current phase. |

---

## 10. What V3B must NOT do

1. **Do not make greedy fail by arbitrary action masks.** If we blacklist genes solely because greedy picks them, we are gerrymandering. Every mask must point to an external biological signal we can cite.
2. **Do not choose evaluation cells where the start pool is structurally empty.** Reachability ≥ 10 cells in pool is a hard pre-condition; do not run a cell that fails it (existing `evaluate_rl_hard.py::empty_start_pool_summaries` handles this — keep using it).
3. **Do not lower the dynamics-gate threshold.** CLAUDE.md §3 rule 4 + V3 stub §3. V3B uses `skip_gate=true` documented in PROGRESS.md per the V2 protocol; thresholds in `config/dynamics.yaml::gate.*` are locked.
4. **Do not remove greedy baselines from any eval table.** V3B's main risk is rediscovering the V2 trap under a new objective without noticing — the `greedy_dyn_2_under_same_reward` line is the alarm.
5. **Do not claim biological discovery or therapeutic effect.** CLAUDE.md §3 rule 7 stands. V3B is "in-silico safer reprogramming routes under a biorealistic prior", not "we cured CML".
6. **Do not use DepMap Chronos as a reward term AND as a plausibility test.** Train-test contamination. Reward = DepMap + Horlbeck SL; test = Replogle K562 essentials + OGEE + CGC + Open Targets. Mutual exclusion enforced in code.
7. **Do not penalise genes solely because greedy is strong on them.** Each penalty must point to a measurable biological signal.
8. **Do not amend or rewrite frozen V2 dynamics.** V3B does not retrain dynamics on 32D — it reuses V2 primary `RoR_corr010`. Track N produces its own dynamics in V3A separately; V3B reuses that.
9. **Do not modify any of `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/`.** Frozen tiers. V3B writes only to `artifacts_v3/`.
10. **Do not retrain PPO on the legacy 64D Track L.** V3A already showed it's more saturated than V2; spending compute there is V3A's known mistake.
11. **Do not interpret a single-seed PPO win as V3B success.** Phase 7 (4-seed escalation) is required before headline.

---

## 11. Code modules that change

### 11.1 New modules

| Path | Purpose | Size estimate |
|---|---|---|
| `src/analysis/path_feasibility.py` | Build & cache the V3B biology layer: `load_chronos()`, `load_horlbeck_k562_sl()`, `score_episode(actions, ...) → {tox_path, sl_violations, peak_uncertainty, common_essential_count, realism}`. | ~200 LOC |
| `src/rl/biology_rewards.py` | Pure-function reward extensions for variants B (`path_length_freeband`), C (`safety_aware`), D (`uncertainty_aware`), E (`combined`). Mirrors `src/rl/reward.py::compute_reward` signature; called from environment when `reward.mode ∈ {path_length_freeband, safety_aware, uncertainty_aware, combined}`. | ~250 LOC |
| `src/analysis/v3b_metrics.py` | Per-episode and per-run aggregators: `safety_adjusted_success_rate`, `pareto_frontier`, `permuted_chronos_null` (used by Phase 3). | ~150 LOC |
| `scripts/build_v3b_biology_layer.py` | Phase-0 driver. Reads DepMap parquet + downloads/imports Horlbeck SL table; writes to `artifacts_v3/v3b_biology/`. | ~100 LOC |
| `scripts/posthoc_score_paths.py` | Phase-1 driver. Loads existing V2 PPO + greedy rollouts (rollouts.parquet) and applies the biology layer. Writes `artifacts_v3/eval_v3b_posthoc/`. | ~150 LOC |
| `scripts/train_rl_v3b.py` | Thin wrapper around `scripts/train_rl.py` that asserts `cfg.rl.reward.mode ∈ {path_length_freeband, safety_aware, uncertainty_aware, combined}` and writes to `artifacts_v3/rl_v3b_*/`. | ~80 LOC |
| `scripts/evaluate_rl_v3b.py` | Wrapper around `scripts/evaluate_rl_hard.py` that adds per-episode biology scoring + permuted-Chronos null support + Pareto figure. Writes `artifacts_v3/eval_v3b_*/`. | ~200 LOC |
| `tests/test_biology_rewards.py` | Unit tests for each new reward mode: scale invariance, zero-action no-penalty, single-violation cost. | ~150 LOC |
| `tests/test_path_feasibility.py` | Unit tests for biology layer loader + scorer. | ~100 LOC |

### 11.2 Modified modules

| Path | Change | Why |
|---|---|---|
| `src/rl/reward.py` | Add `reward_mode ∈ {path_length_freeband, safety_aware, uncertainty_aware, combined}` branches dispatching to `src/rl/biology_rewards.py`. **Do not change** existing modes (V2 primary preserved). | Single source of truth for reward dispatch. |
| `src/rl/environment.py` | Pass through new reward kwargs: `safety_table`, `sl_pair_set`, `path_length_schedule`, `lambda_tox`, `lambda_unc`, `lambda_sl`. Plumb path-accumulators (`tox_path_so_far`, `sl_violations_so_far`) through `step`. | Env owns the per-episode accumulators; the reward function is pure. |
| `src/rl/baselines.py` | Add `GreedyDynamicsBeamPolicy` depth=5 variant by adjusting default arg — already supports it via `depth`. Add `select_action_with_reward_aware_distance` that incorporates path-accumulators (so greedy can be re-evaluated under any V3B reward at eval time). | "Greedy under V3B reward" must exist to be the fair comparator. |
| `config/rl.yaml` | Add `reward.lambda_tox`, `reward.lambda_unc_path`, `reward.lambda_sl`, `reward.path_length_schedule` (struct with K_free, beta, gamma, alpha). | Hydra single source of truth. |
| `config/paths.yaml` | Add V3B path keys: `v3b_biology_dir`, `v3b_rl_dir_template`, `v3b_eval_dir_template`, `v3b_figures_dir`, `v3b_interpretation_dir`. | CLAUDE.md §3 rule 3. |
| `src/analysis/metrics.py` | Add `chronos_weighted_action_score`, `sl_violation_count`, `pareto_frontier_2d` — single source of truth. | CLAUDE.md §3 rule 4. |
| `PROGRESS.md` | Append V3B session entries per CLAUDE.md §8 format. | Standard. |

### 11.3 Modules that DO NOT change

* `src/models/dynamics.py` — V3B reuses V2 primary dynamics.
* `src/data/*.py` — V3B uses existing pairs / latents.
* `scripts/train_vae.py`, `scripts/train_dynamics.py` — V3B touches neither (except via Hydra path overrides if Track N triggers a fallback in Phase 8).
* `config/dynamics.yaml`, `config/vae.yaml` — locked.
* `src/utils/{device,seeding,checkpointing,logging}.py` — touch only via reads.
* Anything in `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/`.

### 11.4 Where all V3B outputs go

```
artifacts_v3/
├── v3b_biology/
│   ├── gene_safety.parquet           # 105 rows × {gene, chronos, is_common_essential, cgc_tier, og_tier}
│   ├── k562_sl_pairs.parquet         # n rows × {gene_a, gene_b, sl_score, source}
│   ├── coverage.json                  # % of 105 with each annotation
│   └── README.md                      # provenance + caveats
├── rl_v3b_<reward>_<latent>_seed<N>/  # PPO checkpoints + rollouts
│   ├── ppo.zip
│   ├── rollouts.parquet
│   └── metadata.json
├── eval_v3b_posthoc/                  # Phase 1 outputs (no PPO retrain)
├── eval_v3b_<reward>_<latent>/        # Phases 2–7
│   ├── per_cell/*/summary.json
│   ├── pareto_frontier.png
│   ├── permutation_null_summary.json  # only if applicable
│   └── results_table.md
├── figures_v3b/                       # consolidated V3B figures
└── interpretation/
    └── v3b_phase{N}_interpretation.md
```

---

## 12. Decision rules conditional on Track N outcome

(See also §7.3 for the latent-axis decision.)

**Branch X — Track N finishes BEFORE V3B Phase 1 starts.** Phase 1 runs against V2 32D as planned; Track N's safety result is logged. If Track N passed safety, Phase 6 (replicate on Track N) is unlocked. If Track N failed safety, Phase 6 is skipped; the representation axis remains stalled at 32D (Phase 8 / V3.3 ZINB is the next representation pivot).

**Branch Y — Track N finishes DURING V3B Phase 2/3/4.** No change; Phase 6 schedule shifts but Phase 2/3/4 (on 32D) continue. Track N's `greedy_dyn_2` saturation level is recorded; if it is the same as Track L's (≥ 0.95 at primary), the latent axis is conclusively rejected without further V3 latent experiments — V3.fallback.B (contraction-regulariser) is the only remaining axis-A move.

**Branch Z — Track N has NOT finished by V3B Phase 6.** Phase 6 skipped this session. V3B records `Track N 64D — pending` in PROGRESS.md. Next session checks Track N and runs Phase 6 only if it has completed + passed safety. If it has, the best axis-B variant is replicated on Track N. If not, V3B ends with the 32D-only result.

**Branch W — Track N crashes / NaN.** Investigate root cause per CLAUDE.md general guidance; do not bypass. If unrecoverable, drop 64D from V3B; cite the V3A interpretation file. Phase 8 (V3.3 ZINB) becomes the axis-A fallback if axis B fails.

---

## 13. How ZINB and SCANVI fit into V3B

V3B's primary axis is **control-objective**. ZINB and SCANVI are **representation-level**:

* **ZINB** (`vae.gene_likelihood=zinb`) changes the per-gene observation distribution from negative-binomial to zero-inflated-NB. *Effect:* better fit on the dropout-rich tails of the Norman expression matrix; may produce a latent geometry where the dropout-heavy axes are de-emphasised. *V3B relevance:* none directly — ZINB does not encode safety, viability, or feasibility. **It is an axis-A fallback** if axis-B exhausts.
* **SCANVI** (`scvi.model.SCANVI`) extends scVI with a semi-supervised classifier head trained on partial cell-type labels ([scvi-tools docs](https://docs.scvi-tools.org/en/stable/user_guide/models/scanvi.html)). For us, the natural label is **perturbation identity** (105 classes + control), not cell type. *Effect:* a latent where perturbation directions are explicitly encoded as classifier-separable subspaces — potentially making "distance to centroid" a *less* sufficient statistic and forcing the policy to navigate a richer manifold. *V3B relevance:* **a stronger axis-A fallback than ZINB**, because perturbation-aware latent geometry can interact with the biorealistic objective in ways pure latent-dim increase cannot. SCANVI on 32D is the recommended Phase 8 axis-A move if axes B and A1 (Track N) both fail.

### 13.1 Combined experiments (axis A × axis B)

Only after both axes have been minimally tested:

| Combination | Trigger |
|---|---|
| Track N 64D × best axis-B variant | Phase 6 (if Track N passes safety) |
| SCANVI 32D × best axis-B variant | If Phases 2–5 fail on V2 32D AND Phase 6 fails on Track N — pivot to representation-level fallback per V3 stub §5 then replicate axis-B variant. |
| ZINB 64D × best axis-B variant | Only if SCANVI also fails. |
| Contraction-regulariser dynamics × best axis-B variant | Last-resort; only after all three axis-A fallbacks are exhausted. |

We never combine all axes simultaneously without a sequential ablation — the V2-trap is "I added 4 things and it works, which one mattered?". Sequential ablation answers it; conjunction does not.

---

## 14. Verification (end-of-V3B-session check)

Before declaring V3B Phase N complete:

1. `git status` shows `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/` clean.
2. Every PPO run logged `rl.train.skip_gate=true` with the per-run metadata.json containing the dynamics gate margin and beam reach (CLAUDE.md §3 rule 9; PROGRESS.md sacred-rule conformance).
3. `pytest -q` passes (no regressions); new tests in `tests/test_biology_rewards.py` and `tests/test_path_feasibility.py` pass.
4. Every result table includes both `PPO − greedy_dyn_2_same_reward` AND `PPO − random` columns.
5. For Phase 3+, permutation-null PPO summary is present and shows real-Chronos > permuted-Chronos.
6. The interpretation file under `artifacts_v3/interpretation/v3b_phase{N}_interpretation.md` contains:
   * Phase goal, reward variant, cells tested, n_seeds, n_episodes.
   * Headline `PPO − greedy_dyn_2_same_reward` with CI.
   * Pareto multi-axis comparison.
   * Caveats: CRISPRa ≠ CRISPRko, evaluation-objective leakage status, Track N status.
   * Decision: continue / stop / pivot per §9.5.

---

## 15. Concise implementation prompt for V3B Phase 0 + Phase 1

(Paste this into a fresh CC session after this plan is approved and exits plan mode.)

```
You are executing V3B Phases 0 and 1 from V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md. Goal:
build the biology-annotation layer and post-hoc score the V2 primary PPO rollouts under
the new biology layer. No PPO retraining in this session. Do NOT interrupt the running
Track N VAE training (background process).

REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

Read first (in this order):
  - V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §§3, 5, 9, 11 (biology layer + reward
    constraints + success metrics + module map)
  - artifacts_v2/V2_FINAL_REPORT.md §3 (cells to score)
  - artifacts_v3/interpretation/v3a_checkpoint.md (Track N status)
  - src/analysis/depmap_validation.py (existing Chronos loader to reuse)
  - scripts/evaluate_rl_hard.py (existing rollout structure)

Phase 0 — Biology layer build (1 h):
  1. Confirm data/processed/depmap_k562_chronos.parquet present; coverage of the 105 genes
     in artifacts/vae/gene_vocab.json::genes. Log to artifacts_v3/v3b_biology/coverage.json.
  2. Download Horlbeck-2018 K562 GI map (GEO GSE116198 or the SLKB-derived table). Intersect
     with the 105 gene set; keep only pairs where BOTH genes are in our action space.
     Save to artifacts_v3/v3b_biology/k562_sl_pairs.parquet. If both genes are not available
     for a Horlbeck-published pair, drop the pair with a logged warning.
  3. Implement src/analysis/path_feasibility.py with load_chronos, load_horlbeck_k562_sl,
     score_episode. Add unit tests in tests/test_path_feasibility.py.
  4. Write artifacts_v3/v3b_biology/README.md documenting provenance + caveats (CRISPRa vs
     CRISPRko, cell-line specificity, K562-only Horlbeck partition).
  5. Acceptance: pytest passes; coverage.json shows ≥ 90 of 105 genes annotated; sl_pairs
     parquet has > 0 rows.

Phase 1 — Post-hoc scoring of V2 primary (1 h):
  1. Locate V2 primary PPO rollouts: artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed{42,0,1,7}/rollouts.parquet.
  2. Locate V2 primary greedy rollouts: re-run with scripts/evaluate_rl_hard.py if the
     parquet does not already exist for greedy. Use n_episodes=300 to match V2 protocol.
  3. For each policy in {ppo_seed42, ppo_seed0, ppo_seed1, ppo_seed7,
     greedy_dyn_1, greedy_dyn_2, greedy_dyn_3, random_uniform_valid, always_noop}:
       For each cell in {K=3/bin8-10/OOD primary, K=2/bin6-8/OOD, K=2/bin8-10/OOD,
                          K=3/bin6-8/OOD}:
           Compute per-episode {tox_path, sl_violations, peak_uncertainty,
                                common_essential_count, realism}; aggregate to mean ± std.
  4. Write artifacts_v3/eval_v3b_posthoc/posthoc_summary.md with one table per cell:
        | policy | success | tox_path | SL_viol | unc_peak | realism |
  5. Decisive question to answer in posthoc_summary.md §"Verdict":
        Does greedy_dyn_2 already have lower (better) tox / SL / realism than PPO?
        If yes → V3B premise is suspect; halt before Phase 2 and write a halt note.
        If no  → V3B premise stands; Phase 2 (retraining under reward B1) is justified.
  6. Update PROGRESS.md with a new session entry per CLAUDE.md §8.

Hard rules:
  - No edits under artifacts/, artifacts_64/, artifacts_v2/, artifacts/rl_sweeps/.
  - All new outputs under artifacts_v3/.
  - Do not interrupt the running Track N VAE (check `ps aux | grep train_vae` first).
  - Do not invoke train_rl.py in this session.
  - Use existing src/analysis/depmap_validation.py::load_depmap_k562 — do not duplicate.
  - If Horlbeck SL data is hard to obtain in this session, document the gap in
    artifacts_v3/v3b_biology/README.md and proceed with Chronos-only Phase 1; Phase 2
    is delayed by one session.

Stop after Phase 1. Report the Verdict line (PROCEED or HALT) and the posthoc_summary.md
path. Wait for user approval before Phase 2.
```

---

## 16. References

* **V2 result** — [`artifacts_v2/V2_FINAL_REPORT.md`](../Developer/ITAM/IA/cellpath/artifacts_v2/V2_FINAL_REPORT.md) (in repo)
* **V3A latent audit** — [`V3A_LATENT_AUDIT_AND_64D_PLAN.md`](../Developer/ITAM/IA/cellpath/V3A_LATENT_AUDIT_AND_64D_PLAN.md) and [`artifacts_v3/interpretation/v3a_checkpoint.md`](../Developer/ITAM/IA/cellpath/artifacts_v3/interpretation/v3a_checkpoint.md)
* **V3 stub** — [`V3_RESEARCH_PLAN.md`](../Developer/ITAM/IA/cellpath/V3_RESEARCH_PLAN.md)
* **Norman et al. 2019** — "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes," *Science* 365, 786–793 (2019). DOI: 10.1126/science.aax4438. https://www.science.org/doi/10.1126/science.aax4438 (the dataset).
* **Replogle et al. 2022** — "Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq," *Cell* 185, 2559–2575. DOI: 10.1016/j.cell.2022.05.013. https://www.sciencedirect.com/science/article/pii/S0092867422005979. Data at https://gwps.wi.mit.edu/ (K562 genome-wide + essential CRISPRi Perturb-seq).
* **Horlbeck et al. 2018** — "Mapping the genetic landscape of human cells," *Cell* 174, 953–967. DOI: 10.1016/j.cell.2018.06.010. https://www.cell.com/cell/pdf/S0092-8674(18)30735-9.pdf. K562 dual-CRISPRi GI / SL map.
* **Chronos / DepMap** — Dempster et al. 2021. "Chronos: a cell population dynamics model of CRISPR experiments." *Genome Biology*. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8686573/. Portal: https://depmap.org/portal/. Score interpretation: https://forum.depmap.org/t/depmap-genetic-dependencies-faq/131.
* **OGEE v3** — Gurumayum et al. 2021. *NAR* 49, D998–D1003. https://academic.oup.com/nar/article/49/D1/D998/5934414. Cross-cell-line essentiality.
* **SLKB** — Gao et al. 2024. *NAR* 52, D1418–D1430. https://academic.oup.com/nar/article/52/D1/D1418/7331024. Aggregated SL knowledge base; K562 partition available.
* **SynLethDB** — Guo et al. 2016, updated 2021. *NAR* 44, D1011. https://academic.oup.com/nar/article/44/D1/D1011/2502617.
* **COSMIC Cancer Gene Census** — Sondka et al. 2018. *Nature Rev Cancer* 18, 696–705. https://www.nature.com/articles/s41568-018-0060-1. Portal: https://cancer.sanger.ac.uk/cosmic/census.
* **Open Targets** — Buniello et al. 2025. *NAR* 53, D1467–D1475. https://academic.oup.com/nar/article/53/D1/D1467/7917960. Tractability docs: https://platform-docs.opentargets.org/target/tractability.
* **SCANVI** — Xu et al. 2021. "Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models." *Mol Syst Biol* 17:e9620. Docs: https://docs.scvi-tools.org/en/stable/user_guide/models/scanvi.html.
* **CLAUDE.md** — `/Users/gabo/Developer/ITAM/IA/cellpath/CLAUDE.md` (project context + sacred rules).

---

## 17. Self-review — spec coverage check

Going back through the user's brief in §0–§16 of this plan:

| User brief point | Where addressed |
|---|---|
| "Why is greedy performing so well?" | §1 (five compounding causes + when legitimate vs oversimplified) |
| "What would make greedy fail for valid reasons?" | §2 (six axes; each cites the biological signal) |
| "Biological data sources, citations, mapping to 105 genes, failure modes" | §3 (tiered tables + §3.5 failure modes) |
| "Mathematical reward families" | §4 (Variants A–F with formulas, intuition, risks, decisive tests) |
| "Hard masks vs soft penalties" | §5 (decision table + design principles + leakage avoidance) |
| "Longer paths and path-length penalty schedule" | §6 (lab/plausible/speculative tiers + schedule in §6.3) |
| "Test on 32D, 64D, or both?" | §7 (32D first, Track N if it passes, decision branches §7.3) |
| "V3B experiment matrix" | §8 (Phase 0–8, costs, decisive criteria) |
| "Success metrics — multi-axis, biological breakthrough" | §9 (three sufficient criteria + Pareto + the not-load-bearing realism score + stop conditions §9.5) |
| "What NOT to do" | §10 (11 explicit prohibitions) |
| "Code modules that change" | §11 (new + modified + unchanged + output tree) |
| "Integrate ZINB / SCANVI as representation-level fallbacks separate from control-objective fallbacks" | §13 (axis A vs axis B; combined experiments only sequential) |
| "Conditional branches on Track N outcome" | §7.3, §12 |
| "Use current literature/datasets and cite" | §3 tables, §16 references |
| "Concise implementation prompt for the first phase" | §15 |
| "Do not implement now; do not interrupt Track N" | §0 header + §15 hard rules |
| "Keep all new outputs under artifacts_v3/" | §11.4 + §10 rule 9 |

**Coverage: complete.** No placeholder steps; every reward formula has explicit defaults; every cited resource has a URL; every prohibition has a rationale.
