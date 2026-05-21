# V3C Dynamics Utility Audit & Reformulation Plan

> **Plan-mode draft.** On approval, this content will be saved to `V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md` at repo root, with a matching pointer added to `PROGRESS.md` at end-of-execution.
>
> **For agentic workers:** Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Phase 0B steps use checkbox (`- [ ]`) syntax for tracking. Phases 0C, 0D, 1, 2+ are research workflows — track via `TaskCreate`, not checkboxes.

---

## Context — why we are doing this

V3B closed with verdict **`LOCKED_DESIGN_TECHNICAL_ONLY`**: the biorealistic B+C+D reward stack (`safety_path_freeband` + `uncertainty_aware` fused as `biorealistic_fused`) is implemented end-to-end, unit-tested (356 passed / 2 skipped), and produces numerically-bounded outputs across all 9 hardness cells. But no Bucket-B planning-advantage headline emerges on the V2 primary 32D RoR_corr010 dynamics field because **that field is saturated**: greedy_dyn_1/2 already reach success across most cells at p25, p15, p10, and even p5 ε for K ≥ 4 cells (see `artifacts_v3/interpretation/v3b_phase3b_strict_epsilon_diagnostic.md` Rule B verdict).

The bottleneck is **representation / dynamics**, not reward. The V3B reward stack is locked as the canonical future objective; it now needs a dynamics field under which it is informative. V3C is the dynamics-utility-reformulation phase.

A naive next move would be to immediately train contraction-regularized dynamics or pivot to SCANVI / ZINB. The user's prompt explicitly rejects that posture: we have already wasted budget on dynamics changes whose downstream utility was opaque (Soft-OT passed the gate but was control-hostile; 64D experiments increased greedy saturation rather than decreasing it). **V3C must first define a "dynamics utility gate" that is predictive of downstream control behavior, audit all existing fields against it, and only then commit to new training.**

The intended outcome of this plan:

1. A concrete, multi-bucket **dynamics-utility audit framework** that quantifies prediction sanity, reachability, greedy saturation, depth leverage, contraction geometry, action heterogeneity, reward leverage, and PPO learnability preconditions — for any existing or future dynamics field.
2. An **exploratory two-tier triage** that ranks/explains/flags candidates rather than over-filtering them, and allocates a small PPO_BCD smoke budget across an anchor (V2 RoR_corr010), best-by-audit fields, and wildcards.
3. A decision tree from audit → smoke → new-dynamics training, with clear escalation criteria and sacred-rule preservation.
4. A list of new dynamics candidates to train **only if** the existing roster fails to produce a usable control problem.

---

## Central question

**Which dynamics field produces the most useful, realistic, non-saturated control problem under the locked `biorealistic_fused` reward stack?**

Not: which field has the highest validation Pearson? Not: which field passes the V2 prediction gate? Both are necessary but neither is sufficient.

A useful V3C dynamics field must be:

- **Prediction-sane** (Bucket U-A): predictively reasonable on val and OOD pairs; uncertainty roughly calibrated. Otherwise the dynamics-model is fiction and downstream metrics are noise.
- **Reachable** (Bucket U-B): the success set is non-empty under realistic planners. Not Soft-OT-style dead/no-op.
- **Non-trivially-greedy-solved** (Bucket U-C): single-step greedy does not saturate every cell; depth leverage exists somewhere.
- **Action-heterogeneous** (Bucket U-E): different starts produce different first actions; no single universal-attractor gene.
- **Geometrically non-degenerate** (Bucket U-D): predicted deltas are not all uniformly pointing at the centroid (the structural cause of greedy saturation).
- **Reward-responsive** (Bucket U-F): the locked B+C+D objective changes optimal action/path selection relative to distance-only.
- **PPO-learnable** (Bucket U-G): standard preconditions for 1M-timestep MaskablePPO to acquire signal.

A field that fails one of these is **flagged** in the audit, not rejected outright; **hard rejection is reserved for fields that are invalid, broken, or genuinely control-dead** (see §4 Stage 0). A field that passes all is promoted to PPO_BCD smoke as **Best-by-audit**. A field that passes most with an interesting signature (unusual geometry, prediction/control disagreement, partial depth leverage) is promoted as a **Wildcard** with explicit written rationale. The audit ranks / explains / flags; smoke-target selection is interpretive (§4 Stage 3) — `util_score` is a ranking aid only.

---

## §1 — Why the classical dynamics gate is insufficient

The current `gate.json` validation, defined in `src/analysis/metrics.py` (`dynamics_validation_gate`), requires 5 margin checks + 1 uncertainty check on the validation split:

- MLP-vs-noop R² margin ≥ 0.10
- MLP-vs-global-mean R² margin ≥ 0.05
- MLP-vs-per-gene-mean R² margin ≥ 0.0
- **MLP-vs-linear-ridge Pearson margin ≥ 0.03** (the V2 binding constraint)
- MLP-vs-kNN R² margin ≥ 0.03
- Uncertainty Spearman ≥ 0.20

This catches gross prediction failures but is **orthogonal to control utility** for three reasons:

### 1.1. One-step prediction error does not predict planning utility under iterative composition

A learned dynamics field f̂(z, g) ≈ z + μ(z, g) is used by greedy/beam planners and PPO as a multi-step rollout: z₀ → z₁ → z₂ → … → z_T. Compounding error scales **non-linearly** with T (Janner et al. 2019, "When to Trust Your Model"; Asadi et al. 2018, "Lipschitz Continuity in MBRL"), and the curvature of the error landscape near the reference centroid matters more than mean error. A field with high mean Pearson but high local Lipschitz constant near z_ref may **diverge** at K = 3, while a lower-Pearson field with smooth, contractive geometry may steer reliably.

### 1.2. The Soft-OT cautionary case

`artifacts_v2/dynamics_soft_ot_default` passed the prediction gate (val ridge margin +0.0413, comfortably above the +0.03 threshold) but achieved **0/64 control successes** in V2 hard-bench (`V2_FINAL_REPORT.md`). Soft-OT pairs reduce per-pair noise by averaging over multiple targets, which improves regression metrics. The suspected failure mode is **barycentric / action-homogenized** dynamics — soft pairs encode "any gene moves you about equally toward a population mean," which would predict an approximately action-indiscriminate μ(z, g) (genes do similar things in similar contexts). Whether this manifests as literal μ ≈ 0, μ approximately independent of g, or μ concentrated on a single attractor direction is an **empirical question Bucket U-D answers** via the contraction-geometry and action-diversity diagnostics. Either way, the result is an *un-actionable* MDP that the prediction gate cannot see.

### 1.3. The V1-OT / RoR-corr010 inverse case

Both `artifacts/dynamics` (V1 OT, +0.0074 ridge margin) and `artifacts_v2/dynamics_v1ot_ror_corr010` (V2 primary, +0.0136 ridge margin) **fail** the prediction gate by the +0.03 threshold but are fully controllable: V2 primary achieves 17/17 beam reachability at K=3/bin8-10/OOD and PPO_C2 achieves 0.941 success at the primary cell (`V2_FINAL_REPORT.md`). The gate threshold is set by an OT-pairing-noise floor (~0.89 across all configurations tested), not by architecture, so the gate **systematically under-credits** fields that are noisy at the prediction level but geometrically coherent at the multi-step level.

### 1.4. What V3C must add

The audit must measure properties that emerge from **iterated** dynamics, **action discrimination**, and **interaction with the reward shape**:

- Beam reachability (U-B): does the success set exist under realistic planners?
- Greedy saturation and depth leverage (U-C): is the problem trivial or non-trivial at K=1, 2, 3, 5, 8?
- Contraction geometry (U-D): is the field *too* contractive (Soft-OT-like collapse, or universal-attractor)?
- Action heterogeneity (U-E): does the field discriminate between genes?
- Reward leverage (U-F): does the locked B+C+D objective change optimal behavior?
- PPO learnability (U-G): preconditions for 1M-step MaskablePPO acquisition?

These are the new metrics that constitute **Bucket U — dynamics utility** (sub-buckets U-A through U-G), orthogonal to the existing V3B-legacy buckets (Bucket A = reward-fit, Bucket B = reward-independent control, Bucket C = held-out biology, Bucket D = dynamics prediction). Legacy bucket meanings are preserved per `AGENTS.md` §4.

---

## §2 — Inventory of existing dynamics fields

Phase 0A builds an automated inventory via `scripts/inventory_dynamics_v3c.py`, which discovers every directory under `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts_v3/` containing `model.pt` + `config.json`, and extracts metadata.

**Initial roster from reconnaissance (subject to inventory script verification):**

| # | Field path | n_lat | Pair source | RoR | λ_corr | Val P | OOD P | Unc-Spearman | Gate margin | Status | Audit class |
|---|---|---:|---|:-:|---:|---:|---:|---:|---:|---|---|
| 1 | `artifacts/dynamics` | 32 | OT | no | 0.00 | 0.6085 | 0.4793 | 0.249 | +0.0074 | V1 baseline | **Eligible** |
| 2 | `artifacts/dynamics_ablation/*` (4 dirs) | 32 | OT | no | 0.00 | partial | — | — | — | ablations | Eligible-conditional |
| 3 | `artifacts/dynamics_sweeps/*` (5 dirs) | 32 | OT | no | 0.00 | partial | — | — | +0.0074 | lr / mse sweep | Eligible-conditional |
| 4 | `artifacts_64/dynamics` | 64 | OT | no | 0.00 | 0.5965 | — | — | −0.0191 | n=64 baseline | Eligible |
| 5 | `artifacts_64/dynamics_variants/*` (5 dirs) | 64 | OT | no | 0.00 | partial | — | — | — | n=64 ablations | Eligible-conditional |
| 6 | `artifacts_v2/dynamics_v1ot_ror` | 32 | OT | **yes** | 0.00 | 0.6007 | 0.4392 | 0.245 | +0.0069 | RoR no-corr | **Eligible** |
| 7 | `artifacts_v2/dynamics_v1ot_ror_corr005` | 32 | OT | **yes** | 0.05 | 0.6097 | 0.4899 | 0.250 | +0.0114 | RoR + corr 0.05 | **Eligible** |
| 8 | **`artifacts_v2/dynamics_v1ot_ror_corr010`** | 32 | OT | **yes** | 0.10 | 0.6146 | 0.5163 | 0.245 | +0.0136 | **V2 PRIMARY — ANCHOR** | **Anchor** |
| 9 | `artifacts_v2/dynamics_mean_delta_default` | 32 | MΔ | no | 0.00 | 0.6202 | — | — | +0.0214 | mean-delta no-corr | **Eligible** |
| 10 | `artifacts_v2/dynamics_mean_delta_corr_005` | 32 | MΔ | no | 0.05 | full | — | — | — | mean-delta + corr | **Eligible** |
| 11 | `artifacts_v2/dynamics_mean_delta_corr_010` | 32 | MΔ | no | 0.10 | full | — | — | — | mean-delta + corr | **Eligible** |
| 12 | `artifacts_v2/dynamics_mean_delta_corr_030` | 32 | MΔ | no | 0.30 | full | — | — | — | mean-delta + corr | **Eligible** |
| 13 | `artifacts_v2/dynamics_soft_ot_default` | 32 | sOT | no | 0.00 | partial | — | — | gate **passed** | Soft-OT cautionary | **Eligible (control reference)** |
| 14 | `artifacts_v2/dynamics_random_default` | 32 | rand | no | 0.00 | partial | — | — | — | random pair control | **Eligible (null reference)** |
| 15 | `artifacts_v3/dynamics_n64_legacy_ror_corr010` | 64 | OT | **yes** | 0.10 | — | — | — | +0.0043 | V3A Track L | **Eligible** |
| 16 | `artifacts_v3/dynamics_n64_nb_ror_corr010` | 64 | OT-NB | **yes** | 0.10 | — | — | — | +0.0074 | V3A Track N (NB likelihood) | **Eligible** |

**Inventory script (`scripts/inventory_dynamics_v3c.py`) outputs `artifacts_v3/v3c/utility_audit/dynamics_inventory.csv` with columns:**

`field_id, path, n_latent, pair_source, ror_flag, lambda_corr, val_pearson, val_r2, ood_pearson, ood_r2, ridge_margin, knn_margin, uncertainty_spearman, gate_passed, mtime, pairs_dir, parents_or_siblings, eligible, audit_class`

**Eligibility rules:**
- `eligible = True` iff: `model.pt` loads cleanly, `config.json` parses, `n_latent` is well-defined, dynamics is `PerturbationDynamicsModel`-compatible (i.e. exposes `forward(z, gene_idx) → (z_next, μ, log_var)`).
- Architecture-ablation variants (rows 2, 3, 5) are marked `eligible-conditional`: they are loaded only if their parent dynamics dir lacks coverage on a given audit metric.
- **All eligible fields are audited.** No pre-filtering by prediction metrics.

**Audit class semantics:**
- `Anchor` = always present in PPO_BCD smoke (V2 primary RoR_corr010).
- `Eligible` = candidate for top-by-audit and wildcard selection.
- `Eligible-conditional` = audited if needed for diagnostic coverage, not promoted to smoke.

---

## §3 — The V3C Dynamics Utility Gate (Bucket U)

The audit produces **seven sub-bucket scores** per field; these compose Bucket U and are named **U-A through U-G** (to avoid collision with V3B legacy Bucket A / B / C / D, which retain their existing meaning per `AGENTS.md` §4 and §12 below). No single overfit "utility score" is published; rankings are by per-bucket rules + a flagged composite (`util_score`) used purely as a ranking aid. Smoke-target selection from the audit always requires written qualitative rationale (§4 Stage 3) — never the composite alone.

### Bucket U-A — Prediction sanity (necessary, not sufficient)

**Source**: `gate.json`, `val_metrics.json`, `ood_metrics.json` already present per field. The audit re-computes only if missing.

**Metrics:**
- `val_pearson`, `val_r2`
- `ood_pearson`, `ood_r2`
- `mlp_minus_ridge_pearson` (the V2 gate margin)
- `mlp_minus_knn_r2`
- `uncertainty_spearman_val`, `uncertainty_spearman_ood`
- `per_dim_pearson` distribution (median + p10): catches latent dimensions where the field is dead
- `per_gene_r2` distribution (median + p10): catches genes the field cannot handle

**Soft thresholds (provisional, used for ranking only):**
- val_pearson ≥ 0.40 (broad sanity)
- ood_pearson ≥ 0.20 (held-out genes are not entirely broken)
- uncertainty_spearman_val ≥ 0.10 (some calibration; below this, Variant D is useless)

**Prediction-pathological flag (NOT a hard rejection):**
- `val_pearson < 0.10` OR `per_dim_pearson median < 0.05` → field is flagged `prediction_pathological`. Geometry (U-D) and reachability (U-B) diagnostics still run if the model loads — these are cheap and may reveal a coherent control geometry hidden behind noisy prediction (the V1-OT / RoR-corr010 inverse case in §1.3). Downstream prediction-derived metrics are reported with explicit "low-prediction" caveat.
- Hard rejection is reserved for Stage 0 load/forward failures (§4 Stage 0). A model that loads and runs gets audited.

### Bucket U-B — Reachability

**Source**: beam reachability oracle. We reuse `GreedyDynamicsBeamPolicy` from `src/rl/baselines.py` with `depth = K`, `beam_width = 64` (larger than the policy-evaluator default 20), distance-only scoring (no reward terms), and ε = p15 = 2.9898 (locked from V3B Phase 4).

**Metrics per field × cell:**
- `beam_reach_at_K_p15`: fraction of start states for which beam at depth K reaches d < ε at any step ≤ K
- **Canonical 7-cell V3B matrix (per `V3_CONTROLLER_OBJECTIVE_SPEC.md`):**
  K=2/bin6-8/OOD, K=2/bin8-10/OOD, K=3/bin6-8/OOD, K=3/bin8-10/OOD, K=4/bin8-10/OOD, K=5/bin8-10/OOD, K=8/bin8-10/OOD
- **Optional exploratory cells (run only if budget allows):** K=4/bin6-8/OOD, K=5/bin6-8/OOD (the "+2" cells defined in `evaluate_rl_v3b_phase4.py:CELL_DEFS`). When run, they are reported as supplementary columns, not in the headline.
- Also reported at p25 (3.1663) and p10 (2.8846) for stricter-ε sensitivity

**Soft thresholds:**
- For at least one bin∈{8-10} K∈{4, 5, 8} cell, `beam_reach_at_K_p15 ∈ [0.10, 0.95]` (problem is non-trivially-reachable)
- `beam_reach_at_K=8/bin8-10/OOD ≥ 0.30` at p15 (the problem is solvable in principle)

**Hard rejection ("dead field"):**
- `beam_reach_at_K=8/bin8-10/OOD < 0.05` at p15 (this is the Soft-OT signature: no plan can reach success)

### Bucket U-C — Greedy saturation and depth leverage

**Source**: `evaluate_rl_v3b_phase4.py`-style runs but with shorter episodes (`n_episodes = 200` per cell, sufficient for ±0.05 Wilson CI). Use **distance-only** greedy (`reward_mode = absolute_distance` or freeze λ_tox=λ_ce=λ_unc_path=0) AND fused-greedy (with locked B+C+D coefficients) — both reported.

**Metrics per field × cell:**
- `greedy_dyn_K_distance`: success at K ∈ {1, 2, 3, 5, 8}, distance-only scoring
- `greedy_dyn_K_fused`: same, with locked B+C+D scoring
- `depth_leverage_2_minus_1`, `depth_leverage_3_minus_2`, `depth_leverage_5_minus_3`, `depth_leverage_8_minus_5`: paired differences
- `cumulative_depth_leverage_K8_minus_K1`: total range
- `path_length_distribution`: histogram of T at success across K, useful for cross-checking freeband leverage

**Soft thresholds:**
- `greedy_dyn_1_distance ≤ 0.95` for at least 3 cells (single-step does not trivially solve everything)
- `greedy_dyn_5_distance ≤ 0.99` for at least 1 K ≥ 4 cell (some headroom for planning)
- `cumulative_depth_leverage ≥ 0.05` for at least 1 K ≥ 4 cell (depth has value)

**Concern flags (soft):**
- `greedy_dyn_1_distance ≥ 0.95` everywhere → trivially-solvable, no PPO will improve
- `greedy_dyn_5_distance = greedy_dyn_1_distance` everywhere → no depth value
- `greedy_dyn_5_distance = 0.000` anywhere → unreachability hidden behind earlier saturation

### Bucket U-D — Contraction geometry (NEW for V3C)

**Source**: dedicated `scripts/audit_dynamics_utility_v3c.py contraction` subcommand. Samples `(z, g)` pairs from two distributions, computes geometric statistics, **reports both separately** (per user direction; divergence is itself a signal).

**Sample 1 — "OOD start pool":**
- z ~ uniform from cells `bin8-10/OOD` ∪ `bin6-8/OOD` (N=1500 starts)
- g ∈ {1, …, n_genes} (all 105 actions)
- Total: 157,500 (z, g) pairs

**Sample 2 — "Held-out validation pairs":**
- (z_ctrl, gene_idx) drawn from `val_pairs.npz` + `ood_pairs.npz` (N≈4324 + 14549)
- Compute μ(z_ctrl, gene_idx)

**Metrics per sample:**
- `alignment_cos(μ, z_ref − z)`: cosine alignment between predicted delta and direction-to-target. Aggregations: median, p25, p75, fraction > 0.5, fraction < 0.0
- `contraction_fraction`: fraction of (z, g) pairs with `μ · (z_ref − z) > 0` (positive contraction)
- `delta_magnitude_distribution`: ‖μ‖ histogram. Pathology: ‖μ‖ ≈ 0 everywhere (Soft-OT collapse)
- `action_diversity_per_state`: `std_g[cos(μ(z, g), z_ref − z)]` averaged over z. Low = all genes do same thing
- `state_diversity_per_action`: `std_z[cos(μ(z, g), z_ref − z)]` averaged over g. Low = each gene is state-independent (universal attractor or universal pusher)
- `gene_universality_max`: `max_g (mean_z cos(μ(z, g), z_ref − z))`. Above 0.5 = one or few attractor genes dominate
- `gene_universality_gini`: Gini coefficient of `mean_z |cos|` across genes — concentration
- `null_gene_fraction`: fraction of genes with `mean_z ‖μ(·, g)‖ < 0.1·median` (effectively no-op genes; expected and healthy)

**Soft thresholds:**
- `contraction_fraction ∈ [0.50, 0.85]` (most actions point toward target but not all — discrimination exists)
- `action_diversity_per_state ≥ 0.10` (genes do different things in same state)
- `gene_universality_max ≤ 0.70` (no single super-attractor)
- `null_gene_fraction ≥ 0.10` (some genes do nothing — biologically realistic)

**Concern flags:**
- `contraction_fraction > 0.95` and `action_diversity_per_state < 0.05` → universal-contraction collapse → trivial greedy saturation
- `contraction_fraction < 0.30` → field does not steer toward reference at all (Soft-OT-like)
- `gene_universality_max > 0.85` → one gene dominates all decisions

**Divergence signal (Sample 1 vs Sample 2):**
- If `contraction_fraction(OOD pool) − contraction_fraction(val pairs)` differs by > 0.30, flag the field: it behaves geometrically differently on the control distribution vs the training distribution. Possible failure mode (under-confident or over-confident extrapolation).

### Bucket U-E — Action heterogeneity and path diversity

**Source**: same audit script. Reuses Bucket U-C rollouts (no re-running of greedy needed).

**Metrics per field:**
- `first_action_entropy_distance`: Shannon entropy of distance-only greedy_dyn_1 first-action distribution over N=500 starts (one per cell, OOD)
- `first_action_entropy_fused`: same under fused (B+C+D) greedy_dyn_1
- `first_action_top1_freq`, `first_action_top5_freq`, `first_action_top10_freq`: concentration measures
- `first_action_gini`: Gini coefficient over gene-frequency
- `path_diversity_depth2`: fraction of unique 2-step plans in beam-top-1 across starts
- `path_diversity_depth3`: same at depth 3
- `distance_vs_fused_first_action_overlap`: fraction of starts where distance-only and fused greedy pick the same first action

**Soft thresholds:**
- `first_action_entropy_fused ≥ 0.5 · log(n_genes)` (≥ ~2.3 nats for 105 genes — wide selection)
- `first_action_top5_freq ≤ 0.70` (top 5 don't dominate)
- `path_diversity_depth2 ≥ 0.30` (≥30% of 2-step plans are unique)

**Concern flags:**
- `first_action_entropy_fused < 0.3 · log(n_genes)` → single-gene policy dominates (will mask any PPO learning signal)
- `distance_vs_fused_first_action_overlap > 0.95` → fused reward changes nothing (no leverage)

### Bucket U-F — Reward leverage under locked B+C+D

**Source**: same Bucket U-C rollouts compared across reward modes.

**Metrics per field × cell:**
- `delta_success_fused_minus_distance(K)`: at K∈{2, 3, 5}, success rate of fused-greedy minus success of distance-greedy. Sign matters (fused often trades raw success for safety/uncertainty).
- `mean_tox_path` (distance vs fused): fused should reduce
- `mean_common_essential_count` (distance vs fused): fused should reduce, often to zero
- `mean_unc_path_max` (distance vs fused): fused should reduce
- `mean_T_at_success` (distance vs fused): freeband shaping should keep T ≤ 3 modally
- `safety_adjusted_success` (V3B-legacy Bucket A definition: `success ∧ (CE_count = 0)`)

**Soft thresholds:**
- For at least one K ≥ 3 cell, fused-greedy reduces `mean_unc_path_max` by ≥ 0.10 vs distance-only **without** reducing `safety_adjusted_success` by more than 0.05 (D is load-bearing)
- For all cells, fused-greedy has `mean_common_essential_count ≤ 0.10` (C is implementable; we know it works under V2 from Phase 2)
- `distance_vs_fused_first_action_overlap ≤ 0.85` for at least 3 cells (fused changes path; relates to Bucket U-E)

**Concern flags:**
- `delta_success_fused_minus_distance(K=5) ≤ −0.15` everywhere → fused reward is too punitive on this field (over-shaped)
- `mean_unc_path_max` identical between distance and fused → uncertainty is not action-discriminating (the V3B issue — log_var is state-dependent, not gene-dependent)

**Norman measured-combo consistency (optional, post-hoc realism diagnostic):**

When fused-greedy or PPO paths produce 2-step combinations `(gene_a, gene_b)` that overlap with Norman 2019 measured combo perturbations (the `combo_pairs.npz` universe, ~35,995 measured combo-cell pairs):

- `fraction_paths_with_measured_combo_overlap`: percent of successful 2-step trajectories whose `(gene_a, gene_b)` corresponds to a pair present in Norman's measured combo set. Report both ordered and unordered overlap.
- `measured_combo_latent_consistency`: for overlapping paths, mean cosine similarity between the predicted post-2-step latent and the empirical Norman post-combo latent (held-out validation distribution). Higher = the dynamics' predicted combo behavior agrees with measured combo behavior on the overlapping subset.
- `measured_combo_distance_consistency`: same but using L2 distance to the empirical post-combo centroid; reported as a sanity check.

This is a **realism diagnostic**, not a reward axis. A field that plans paths overlapping with measured biology is more credible than one that plans purely in unmeasured combinatorial space. Used as a tie-breaker between candidates and as a secondary contributor to the `CANDIDATE_SIGNAL_PARETO` verdict (§4 Stage 4). **Optional** — runs only when paths-with-combo-overlap subset is non-empty; otherwise reported as `n/a`.

### Bucket U-G — PPO learnability preconditions

**Source**: composite of Buckets U-A through U-F.

**Preconditions (all must hold for a field to be PPO-promoted via "Best-by-audit" route):**

1. Bucket U-A: `val_pearson ≥ 0.40` AND `ood_pearson ≥ 0.20` AND `uncertainty_spearman_val ≥ 0.10`
2. Bucket U-B: `beam_reach_at_K=2/bin8-10/OOD ≥ 0.10` at p15
3. Bucket U-C: ≥ 3 cells with `greedy_dyn_5_distance ∈ [0.10, 0.95]` AND `cumulative_depth_leverage ≥ 0.05` in ≥ 1 K ≥ 4 cell
4. Bucket U-D: `contraction_fraction ∈ [0.30, 0.95]` AND `action_diversity_per_state ≥ 0.05`
5. Bucket U-E: `first_action_entropy_fused ≥ 0.3 · log(n_genes)` AND `path_diversity_depth2 ≥ 0.10`
6. Bucket U-F: `distance_vs_fused_first_action_overlap ≤ 0.95` (fused reward has some effect)
7. **No-op collapse low**: `always_noop_success ≤ 0.10` in bin8-10 OOD cells (NOOP must not be the optimal trivial policy)
8. **Random clearly below planner**: `random_uniform_valid_success ≤ greedy_dyn_2_fused_success − 0.10` in ≥ 3 cells (the planner provides leverage)

These are the same preconditions the user's prompt enumerated. They are **soft** for ranking and **flag-only** for the "Best-by-audit" route (failing any precondition warrants written rationale but is not a hard rejection). Wildcards are not bound by them (see §4 Stage 3).

---

## §4 — The Exploratory Two-Tier Filtering Funnel

**Per user direction**: the audit ranks/explains/flags rather than over-filtering. Hard-reject only fields that are invalid, broken, dimension-incompatible, numerically unstable, or completely dead/no-reachability/no-action-utility. The remaining fields are audited and triaged into three PPO_BCD smoke classes.

### Stage 0 — Eligibility check (hard rejection)

Run `scripts/inventory_dynamics_v3c.py`. Reject only if:
- `model.pt` fails to load
- `config.json` malformed
- `n_latent` mismatches the available VAE / start pool
- Forward pass on a single sample raises an exception
- val_pearson < 0.10 (numerical collapse, NaN, gradient explosion)

This typically rejects 0–2 fields. The rest proceed.

### Stage 1 — Full Bucket U audit (no rejection)

Run `scripts/audit_dynamics_utility_v3c.py` on every eligible field. Computes Buckets U-A through U-G per §3. Outputs per-field summary:

```
artifacts_v3/v3c/utility_audit/<field_id>/
    prediction_metrics.json
    reachability.json
    greedy_saturation.json
    contraction_geometry.json
    action_heterogeneity.json
    reward_leverage_fused.json
    ppo_preconditions.json
    bucket_u_summary.md
```

**Composite utility score (for ranking, not for filtering):**

```
util_score = 0.20 · clip(val_pearson, 0, 1)                                 # U-A
           + 0.20 · mean over K∈{4,5,8} cells of clip(beam_reach_p15, 0, 1)  # U-B
           + 0.20 · clip(cumulative_depth_leverage, 0, 0.5) / 0.5           # U-C
           + 0.15 · clip(contraction_fraction × (1 − gene_universality_max), 0, 1) # U-D
           + 0.10 · clip(first_action_entropy_fused / log(n_genes), 0, 1)   # U-E
           + 0.10 · clip(1 − distance_vs_fused_first_action_overlap, 0, 1)  # U-F
           + 0.05 · (1 if Bucket U-G all preconditions met else 0)          # U-G
```

Score is in [0, 1]; reported alongside per-bucket flags. **It is a ranking aid only — never a verdict.** Smoke-target selection (Stage 3) always requires written qualitative rationale that cites specific Bucket U-A through U-G evidence; ranking by `util_score` alone is explicitly insufficient.

### Stage 2 — Audit aggregation and ranking

Run `scripts/aggregate_v3c_utility_audit.py`:

- Aggregates all per-field summaries into:
  - `dynamics_inventory.csv` (Stage 0 metadata)
  - `prediction_metrics.csv` (Bucket U-A across fields)
  - `reachability_matrix.csv` (Bucket U-B; rows=fields, cols=K×bin×ε)
  - `greedy_saturation_matrix.csv` (Bucket U-C)
  - `depth_leverage_matrix.csv` (Bucket U-C derived)
  - `contraction_geometry.csv` (Bucket U-D)
  - `action_heterogeneity.csv` (Bucket U-E)
  - `reward_leverage_fused.csv` (Bucket U-F; includes optional Norman-combo realism columns)
  - `ppo_preconditions.csv` (Bucket U-G)
  - `utility_summary.md` (cross-bucket synthesis per field, ranked by util_score, flagged with explicit concern labels)
  - `candidate_ranking.md` (top-K + wildcards + anchor)

### Stage 3 — Smoke budget allocation (exploratory two-tier)

Total **PPO_BCD smoke budget for Phase 1 = 4 fields** (1 anchor + 1–2 best + 1–2 wildcards), each smoked at **1M timesteps × seed 42 only**. Budget rationale: 4 × ~3 hours = 12 hours total wallclock on a single GPU; cheap relative to V3B Phase 4's 12 final runs.

**Allocation rule:**

1. **Anchor (1 field, mandatory):** `artifacts_v2/dynamics_v1ot_ror_corr010` (V2 primary). Provides the saturated reference point. If any candidate outperforms anchor on V3B-legacy Bucket B raw success (planning-advantage over greedy_dyn_5_fused) with seed-42 success ≥ anchor's seed-42 success + 0.05 **or** on Pareto axes (per `CANDIDATE_SIGNAL_PARETO` §4 Stage 4), escalate to 4-seed in Phase 4.

2. **Best-by-audit (1–2 fields):** Highest `util_score` among eligible fields that also pass Bucket U-G preconditions (all 8 criteria). If 1 field passes U-G, smoke it. If 2 fields pass U-G with `util_score` within 0.05, smoke both. If 0 fields pass U-G, take the highest `util_score` regardless and label "best-by-audit but failed U-G; expected weak signal." In every case, the choice is documented with a written rationale citing specific U-A through U-G evidence.

3. **Wildcards (1–2 fields):** Fields that **fail** Bucket U-G but have at least one unusual positive signal:
   - **Geometry-disagreement wildcard**: large `contraction_fraction` divergence between OOD-pool and val-pair samples (Bucket U-D Sample 1 vs Sample 2 divergence > 0.30). May behave very differently under control than under training, in either direction.
   - **Action-heterogeneous wildcard**: very high `first_action_entropy_fused` (top decile of audited fields) even at low `util_score`. Suggests the field is highly action-discriminating but maybe poorly reachable.
   - **Depth-leverage wildcard**: very high `cumulative_depth_leverage` (top decile) even at low Bucket U-A scores. Suggests the field has the planning structure we want even if predictively noisy.
   - **Soft-OT-vs-control disagreement wildcard**: a field where prediction-gate-passing and Bucket U-B reachability sharply disagree (in either direction; e.g., `gate.passed = True` but `beam_reach < 0.30`, or `gate.passed = False` but `beam_reach > 0.70`). Soft-OT, V1-OT, and mean-delta variants are likely candidates here.
   - **Pair-source diversity wildcard**: at least one non-OT pair-source (`mean_delta` or `soft_ot`) gets a wildcard slot if it has any other interesting signal, to cover the "pair geometry orthogonal to architecture" axis.
   - **Norman-combo-realism wildcard**: a field with notably high `fraction_paths_with_measured_combo_overlap` and `measured_combo_latent_consistency` (per Bucket U-F realism diagnostic), even at moderate `util_score`. Indicates the field plans biology-overlapping paths.

   The wildcards are **chosen by the human researcher** from the audit ranking, with a **mandatory written rationale** documented in `artifacts_v3/v3c/interpretation/v3c_phase0_utility_audit.md`. The audit script provides the candidate pool with explanatory flags; the choice is interpretive. Per-field rationale must cite at least two specific U-bucket signals; "high util_score" alone is not a valid rationale.

**Pre-commit policy**: list the 4 chosen smoke targets — with written rationale per target — in the audit interpretation doc **before** running smokes. No backfilling, no post-hoc reassignment.

### Stage 4 — Phase 1 PPO_BCD smoke (adaptive 500k → 1M)

**Anchor reuse pre-check**: before training the anchor, look for an existing V3B Phase 4 PPO_BCD seed-42 checkpoint under `artifacts_v3/rl_v3b_biorealistic_fused_epsp15_seed42/` (or equivalent). If the configuration matches (RoR_corr010 dynamics + ε = p15 + V3B-default freeband/λ_tox/λ_ce/λ_unc_path), **reuse** the checkpoint — no anchor retraining. If absent or incompatible, the anchor enters the adaptive schedule below like any other candidate.

**Adaptive smoke schedule:**

1. **Smoke phase — 4 fields × 500k timesteps × seed 42** under `biorealistic_fused`, ε = p15. Evaluate at 500k against the canonical 7-cell V3B matrix + the V3B baseline policy roster (random, always_noop, greedy_dyn_1/2/3/5_fused).

2. **Mid-smoke triage** per field (after 500k eval):
   - `EARLY_COLLAPSED`: PPO_BCD success on K=2/bin6-8/OOD < 0.30 — implementation likely broken on this dynamics; stop at 500k.
   - `EARLY_FLAT`: PPO_BCD ≤ greedy_dyn_5_fused − 0.05 at every cell AND no Pareto improvement (see below) — same V2 saturation pattern; stop at 500k.
   - `EARLY_PROMISING`: any cell where PPO_BCD ≥ greedy_dyn_5_fused − 0.03 with greedy not saturated (< 0.95), OR clear Pareto improvement on ≥ 2 reward axes — continue to 1M.

3. **Continuation phase**: continue only `EARLY_PROMISING` fields to 1M total (resume from 500k checkpoint). `EARLY_COLLAPSED` and `EARLY_FLAT` keep their 500k verdict.

**Per-field final decision rule (evaluated at whichever checkpoint the field reached — 500k or 1M):**

- `LOCKED_DESIGN_FAILED_IMPLEMENTATION` — `EARLY_COLLAPSED` confirmed at the field's final checkpoint.
- `WEAK_SIGNAL` — `EARLY_FLAT` confirmed at the field's final checkpoint (no raw advantage and no Pareto improvement).
- `CANDIDATE_SIGNAL_RAW` — PPO_BCD beats greedy_dyn_5_fused by ≥ 0.05 raw success at any cell **and** greedy_dyn_5 is not saturated at that cell (< 0.95). Escalate to Phase 4 (4-seed).
- `CANDIDATE_SIGNAL_PARETO` (new) — PPO_BCD ties greedy_dyn_5_fused on raw success within ±0.03 at ≥ 1 non-saturated cell, **AND** improves at least two of:
  - `mean_tox_path ↓`
  - `mean_common_essential_count ↓`
  - `mean_unc_path_max ↓`
  - `T_at_success_distribution` shifts into freeband free/mild bands (T ≤ 5 fraction ↑)
  - `fraction_paths_with_measured_combo_overlap ↑` (Norman-combo realism, Bucket U-F diagnostic)
  
  **without** worsening `mean_final_distance` by > 0.10 vs greedy_dyn_5_fused **and** without breaching the locked-design implementation guardrails (success on K=2/bin6-8/OOD ≥ 0.30). Escalate to Phase 4 (4-seed) — the smoke shows the V3B-intended biorealistic-control payoff (reward-axis improvement without success-rate regression).
- `STRONG_SIGNAL` (rare) — `CANDIDATE_SIGNAL_RAW` at multiple cells with raw margin ≥ 0.10.

Either `CANDIDATE_SIGNAL_RAW` or `CANDIDATE_SIGNAL_PARETO` triggers Phase 4 escalation. `STRONG_SIGNAL` is the gold-standard outcome.

**Reporting**: `artifacts_v3/v3c/interpretation/v3c_phase1_ppo_smoke_summary.md` with per-field final verdict labels, paired deltas (seed-42 PPO vs seed-42 greedy), explicit Pareto-axis evidence where applicable, mid-smoke vs final-smoke comparison, and recommendation for Phase 2 / Phase 4 escalation.

### Stage 5 — Phase 4 escalation (4-seed)

Triggered by `CANDIDATE_SIGNAL_RAW`, `CANDIDATE_SIGNAL_PARETO`, or `STRONG_SIGNAL` in Phase 1. Replicates V3B Phase 4 protocol (4 seeds {42, 0, 1, 7} × 1M steps), runs `aggregate_v3b_phase4.py` for paired CIs, and applies the V3B verdict heuristics (`LOCKED_DESIGN_POSITIVE_SIGNAL` if seed-paired δ vs greedy_dyn_5_fused excludes zero at a non-saturated cell, **or** if Pareto-axis improvements survive 4-seed aggregation).

### Stage 6 — New dynamics (Phase 2 / 3)

Triggered if **no** existing field reaches `CANDIDATE_SIGNAL_RAW` or `CANDIDATE_SIGNAL_PARETO` after Phase 1. See §8 for the candidate ladder.

---

## §5 — Use of the locked B+C+D reward

The audit, smokes, and final evaluations use the **locked** B+C+D reward stack as the canonical objective:

- `reward_mode = "biorealistic_fused"`
- ε = p15 = 2.9898 (locked from V3B Phase 4)
- λ_tox = 0.10, λ_ce = 0.05, λ_unc_path = 0.05
- freeband: free_steps=3, mild_until=5, mild_beta=0.02, heavy_beta=0.10, success_bonus=1.0
- uncertainty_reduce = "mean_sigma", clip_min = −5.0, clip_max = 3.0
- env.max_steps = 8 (V3B Phase 4 standard)

**Reward-aware greedy** at each depth K ∈ {1, 2, 3, 5} uses the **same** locked coefficients (per AGENTS.md §4 and `src/rl/baselines.py` `GreedyDynamicsBeamPolicy._score()`). This preserves the cross-baseline fairness invariant.

### Secondary reward modes (diagnostic, not headline)

For Bucket U-F decomposition, the audit also runs greedy_dyn_K under:
- `terminal_only_step_cost` (V2 baseline)
- `safety_aware` (Variant C alone)
- `path_length_freeband` (Variant B alone)
- `uncertainty_aware` (Variant D alone)
- `safety_path_freeband` (Variant B+C)

These are reported as decomposition columns in `reward_leverage_fused.csv`. They let us answer: when fused-greedy outperforms (or under-performs) distance-greedy, which axis is driving the change?

**Headline reward is always `biorealistic_fused`. Decomposition is for diagnosis only.**

---

## §6 — Bounded reward tuning policy

The user explicitly asks whether B+C+D should be tuned briefly before benchmarking. Reasoning:

### Arguments against any tuning before audit

- Full-grid tuning is the V2 trap (we burned phase budget tuning rewards on saturated dynamics).
- The locked stack is the canonical future objective; results published with the locked stack are comparable across dynamics fields.
- Tuning before utility audit risks selecting a (dynamics, λ) combination that exploits a quirk of one field and doesn't generalize.

### Arguments against zero tuning

- The V3B coefficients (λ_tox=0.10, λ_ce=0.05, λ_unc_path=0.05) were set per V2 design — they may interact differently with a less-saturated field. Variant D (uncertainty) in particular was state-dependent on V2 dynamics, plausibly different on a contraction-aware or perturbation-aware field.
- Reporting only locked-default results may under-credit a candidate that needs slight rebalancing to express its signal.

### Recommendation (locked default = primary; bounded tuning = secondary)

- **Phase 1 PPO_BCD smoke**: locked defaults only. The smoke is for utility detection, not coefficient optimization.
- **Phase 4 escalation (only if Phase 1 shows `CANDIDATE_SIGNAL_RAW` or `CANDIDATE_SIGNAL_PARETO`)**: locked defaults as primary; **plus** a two-step bounded tuning policy on the field that escalated:

  **Step 1 — 4-combination corner mini-grid (default, always run):**
  - `(λ_tox, λ_ce, λ_unc_path) ∈ { (0.05, 0.025, 0.025), (0.05, 0.025, 0.10), (0.10, 0.05, 0.025), (0.10, 0.05, 0.10) }`
  - 4 runs × seed 42 × 1M timesteps
  - Goal: detect coefficient sensitivity cheaply.

  **Step 2 — full 12-combination grid (only if sensitivity appears):**
  - Sensitivity criterion: `max(raw success across mini-grid) − min(raw success across mini-grid) ≥ 0.05` at any cell, OR `max(unc_path_max across mini-grid) − min ≥ 0.10` at any cell.
  - If sensitivity is present: expand to the full `2 × 2 × 3` grid (`λ_tox ∈ {0.05, 0.10}`, `λ_ce ∈ {0.025, 0.05}`, `λ_unc_path ∈ {0.025, 0.05, 0.10}`).
  - If sensitivity is absent (mini-grid is flat): stop — report locked-default as canonical and the mini-grid as appendix evidence that the B+C+D objective is robust on this field. This is a positive finding, not a failure.

  - Each tuned variant runs only seed 42 × 1M; do not 4-seed every tuned combination.
  - Freeband schedule **fixed** (per V3B test coverage); only re-tuned if a field shows path-length leverage > V2 (which V2 did not).
- **Reporting protocol**:
  - Locked default = canonical Phase 4 result, headlines the verdict.
  - Tuned variant = appendix-only, labeled "best bounded-tuned B+C+D for field X". Never claimed as primary; never compared across fields (each field is tuned independently).
  - The locked default vs best-tuned **delta** is reported per field as a sensitivity proxy.

### What is explicitly forbidden

- Tuning per-cell coefficients (would overfit the hardness matrix).
- Tuning per-seed coefficients (would be HARKing).
- Re-tuning freeband schedule on a per-field basis (it is locked).

This bounded policy preserves comparability across V3B and V3C results while allowing modest expression of a candidate field's signal.

---

## §7 — Literature and external support

The audit's bucket framework is grounded in three threads of MBRL / representation-learning literature. We are not doing a formal literature review — citations are to anchor the methodology and provide pointers for further reading.

### Model-based RL: why one-step error fails to predict planning utility

- **Janner, Fu, Zhang, Levine. "When to Trust Your Model: Model-Based Policy Optimization." NeurIPS 2019.** Demonstrates that 1-step model error does not predict downstream policy performance; introduces branched rollouts to mitigate compounding error. Direct support for §1.1.
- **Asadi, Misra, Littman. "Lipschitz Continuity in Model-based Reinforcement Learning." ICML 2018.** Compounding-error bound scales with Lipschitz constant of the learned dynamics. Suggests our Bucket U-D contraction metrics should also include a local Lipschitz proxy if computationally feasible (deferred to Phase 2 if needed).
- **Lambert, Wilcox, Zhang, Pister, Calandra. "Objective Mismatch in Model-based Reinforcement Learning." L4DC 2020.** Argues that prediction objective and control objective are not aligned; supports our split of Bucket U-A (prediction) vs the rest of Bucket U (utility).
- **Chua, Calandra, McAllister, Levine. "Deep RL in a Handful of Trials using Probabilistic Dynamics Models." NeurIPS 2018 (PETS).** Ensemble + heteroscedastic uncertainty for MBRL — direct relevance to Candidate B (ensemble dynamics) in §8.

### Single-cell perturbation latent representations

- **Lopez, Regier, Cole, Jordan, Yosef. "Deep generative modeling for single-cell transcriptomics." Nature Methods 2018 (scVI).** Our current latent backbone.
- **Xu, Lopez, Mehlman, Regier, Jordan, Yosef. "Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models." Mol. Syst. Biol. 2021 (scANVI).** Perturbation-supervised semi-supervised latent — direct relevance to Candidate C in §8.
- **Eraslan, Simon, Mircea, Mueller, Theis. "Single-cell RNA-seq denoising using a deep count autoencoder." Nature Communications 2019 (DCA / ZINB).** Direct relevance to Candidate D.
- **Roohani, Huang, Leskovec. "GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations." Nature Biotechnology 2024.** Perturbation-graph-aware latent for combinatorial extrapolation.
- **Lotfollahi, Naghipourfar, Theis, Wolf. "scGen — predicting single-cell perturbation responses." Nature Methods 2019.** Compositional latent arithmetic; relevant for understanding why mean-delta pairs may underperform OT.

### Optimal transport and pairing

- **Schiebinger, Shu, Tabaka et al. "Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming." Cell 2019.** Diagnostic for OT pairing assumptions; supports the V3B reading that Soft-OT noise reduction can be counterproductive.
- **Tong, Huang, Wolf, van Dijk, Krishnaswamy. "TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics." ICML 2020.** Latent dynamics on OT trajectories.

### Contraction / regularization in neural dynamics

- **Lohmiller, Slotine. "On Contraction Analysis for Non-linear Systems." Automatica 1998.** Mathematical basis for Bucket U-D — provides the framework for thinking about whether a learned vector field "contracts" toward an equilibrium.
- **Kolter, Manek. "Learning Stable Deep Dynamics Models." NeurIPS 2019.** Stable dynamics by construction. May be over-restrictive for our use case (we want partial contraction with action discrimination), but worth knowing.
- **Saemundsson, Terenin, Hofmann, Deisenroth. "Variational Integrator Networks for Physically Structured Embeddings." AISTATS 2020.** Different angle on structured dynamics.

### Anti-collapse and representation diversity

- **Bardes, Ponce, LeCun. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR 2022.** Anti-collapse regularizer that maintains per-dim variance. Could be adapted to action-conditional Δz field.
- **Zbontar et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." ICML 2021.** Off-diagonal decorrelation. Possibly applicable.

### Searches to run if more depth needed

Areas the audit script implementor should keyword-search if Phase 0 raises new questions:
- "Lipschitz regularization MBRL"
- "compounding model error learned dynamics control"
- "uncertainty-aware planning ensemble MBRL"
- "perturbation-aware single-cell latent control"
- "anti-collapse regularization action-conditional"
- "model bias optimization-induced model error"
- "implicit regularization model-based control"

If WebSearch is available at execution time, search these explicitly; otherwise mark for manual review.

---

## §8 — Candidate new dynamics (post-audit, contingent on Phase 1 result)

Only triggered if Phase 1 PPO smoke fails to produce either `CANDIDATE_SIGNAL_RAW` or `CANDIDATE_SIGNAL_PARETO` on any existing field. Listed in priority order; full implementation plans deferred to phase-specific spec docs that will be written **after** Phase 1 results are known.

### Candidate A — Contraction-aware / anti-collapse dynamics (highest priority)

**Hypothesis**: V2 RoR_corr010 over-contracts because the residual MLP head implicitly learns "shrink toward population mean" as the most regularized solution. An explicit penalty on excessive alignment of μ(z, g) with (z_ref − z) would push the model toward action-discriminating geometry.

**Formulations to consider** (one or more to be tried in Phase 2):

(i) **Direct alignment penalty:**
```
L_contract = λ_contract · max(0, cos(μ(z, g), z_ref − z) − τ)²
```
with τ ∈ {0.5, 0.7} the alignment threshold. Activated only when alignment is too high; otherwise inactive.

(ii) **Action-diversity penalty:**
```
L_diverse = λ_diverse · (1 − var_g[μ(z, g)] / ε_diverse)
```
encouraging per-state action variance.

(iii) **Universal-attractor penalty:**
```
L_universal = λ_universal · max_g (mean_z (cos(μ(z, g), z_ref − z)))
```
specifically penalizing the gene-universality observed in Bucket U-D.

(iv) **Composite:** (i) + (ii) at small λ each, plus the existing predictive NLL + composition loss + correlation loss. This is the safest first attempt: it adds geometric constraints without removing predictive structure.

**What it does NOT do (anti-sabotage criterion)**: it does not add input noise, dropout to the dynamics output, or generic regularization on ‖μ‖. Those would degrade prediction without specifically attacking the universal-contraction pathology.

**Training cost**: ~30 min on a single GPU per field; can train 2–3 formulations in parallel.

**Decision gate**: a contraction-aware field must **maintain** Bucket U-A scores (val_pearson within 0.02 of RoR_corr010) **and** show improved Bucket U-D (contraction_fraction < 0.85, gene_universality_max < 0.70).

### Candidate B — Ensemble dynamics with action-discriminating epistemic uncertainty

**Hypothesis**: V3B Variant D (uncertainty-aware) was not load-bearing because the heteroscedastic σ head learned state-dependent uncertainty (cells far from training distribution have high σ regardless of action). An **ensemble** of N=5 dynamics models would expose **action-dependent epistemic** uncertainty: actions that the ensemble disagrees on are uncertain, regardless of state.

**Architecture**: same `PerturbationDynamicsModel` ×5 with different seeds, each trained to convergence. Per (z, g), compute:
- `μ_ensemble = mean_i μ_i(z, g)` (point prediction)
- `σ_epi = std_i [μ_i(z, g)]` (epistemic uncertainty)
- `σ_ale = mean_i exp(0.5 · log_var_i(z, g))` (aleatoric, from each model's heterosce head)
- `σ_total = sqrt(σ_epi² + σ_ale²)` (used by env/baselines)

The env's `per_step_uncertainty_scalar()` accepts the ensemble σ in place of single-head log_var (a small refactor in `src/rl/biology_rewards.py`).

**Cost**: 5× training time (parallel-friendly); 5× inference (small overhead for greedy/PPO since dynamics calls are batched).

**Decision gate**: Bucket U-F under ensemble must show `mean_unc_path_max(distance) − mean_unc_path_max(fused)` ≥ 0.10 at some cell **with action-conditional variance** (i.e., the reduction is action-discriminating, not just state-averaged). Specifically, the ensemble σ for the top-5 fused-greedy actions must differ from the σ for the top-5 distance-greedy actions.

### Candidate C — SCANVI 32D (perturbation-aware latent)

**Hypothesis**: scANVI's semi-supervised loss (using perturbation_idx as labels) aligns the latent to the perturbation manifold rather than the cell-state manifold. This may break the universal-contraction pathology by making the latent itself perturbation-discriminating.

**Risks**:
- May **trivialize** the control task if the perturbation centroid clusters become too separable (then any action moves between clusters).
- May break the (latent, gene_action) interaction structure if the model collapses perturbation-conditional information into class labels rather than continuous geometry.

**Cost**: ~7 hours VAE training + 30 min OT pairs + 30 min dynamics retrain + audit re-run. Significant.

**Decision gate**: SCANVI latent must show meaningfully different Bucket U-D contraction geometry **and** at least one cell where `beam_reach_at_K=3/bin8-10/OOD ∈ [0.20, 0.95]` at p15 (non-trivial, non-saturated). Otherwise, the new latent is just a different saturation point.

### Candidate D — ZINB VAE (low priority unless other candidates fail)

**Hypothesis**: NB likelihood (V3A Track N) showed marginally less saturation than legacy scVI 64D, but both stayed in the same regime. ZINB's explicit dropout component may further alter the pairing-noise floor (currently 0.89 invariant across all configurations).

**Risks**: ZINB is more likely to affect prediction quality than control geometry. Worth running only if Candidates A, B, and C all fail.

**Cost**: ~2 hours VAE + downstream pipeline.

### Candidate E — Strict RL start-pool split (deferred per user direction)

Per the user's answer to the clarifying question, this is **deferred to Phase 4+**, after a new dynamics candidate shows real signal. Existing dynamics-action OOD remains the audit substrate.

When activated:
- Filter PPO training pool to **train-gene cells only** (~85,041 cells).
- Eval on **held-out-gene start states** (the existing 14,549 OOD cells).
- Refactor: pass `held_out_gene_filter` to `_load_start_pool()` in `evaluate_rl_v3b_phase4.py` and to the env's `start_pool_latents` accessor.
- Reports both pre- and post-strict-split numbers for comparability.

### Candidate F — Action-sparsity-prior dynamics (novel, lower priority)

**Hypothesis**: biologically, most gene activations should be approximately null for most cell states (genes don't perturb every cell). The current dynamics has no inductive bias for sparsity — it learns dense action-conditional deltas. An ℓ₁ prior on average ‖μ(·, g)‖ across g could break the universal-attractor pattern by making most actions effectively no-ops.

```
L_sparse = λ_sparse · Σ_g (1/N) Σ_z ‖μ(z, g)‖₁
```

with a moderate λ (e.g., 0.01–0.05).

**Risk**: under-shrinks all actions if λ is too high; effectively no-op everywhere.

**Cost**: ~30 min training per field; can be tried alongside Candidate A in Phase 2 parallel run.

### Candidate G — Distribution-matching dynamics (deferred, speculative)

cell-OT / score-matching dynamics where the loss is the Wasserstein-2 distance between the predicted post-perturbation distribution and held-out empirical post-perturbation distribution. Too speculative for V3C. Listed for completeness; not in the immediate ladder.

### Recommended Phase 2 order

If Phase 1 reaches neither `CANDIDATE_SIGNAL_RAW` nor `CANDIDATE_SIGNAL_PARETO`:
1. Candidate A (contraction-aware, formulation iv: composite) — first.
2. If Candidate A fails to improve audit, Candidate F (sparsity prior) — fast iteration.
3. If geometry still degenerate, Candidate B (ensemble dynamics) — addresses uncertainty axis.
4. If all of A/B/F fail, Candidate C (SCANVI) — last resort, larger investment.
5. Candidate D (ZINB) only if a different question arises (e.g., dropout calibration matters for biology validation).

---

## §9 — Artifact organization

**Recommendation**: do **not** create a top-level `artifacts_v3c/` directory. Keep the V3 sacred rule (all V3 outputs under `artifacts_v3/`). Use the following subtree:

```
artifacts_v3/
├── v3c/
│   ├── utility_audit/
│   │   ├── dynamics_inventory.csv
│   │   ├── prediction_metrics.csv
│   │   ├── reachability_matrix.csv
│   │   ├── greedy_saturation_matrix.csv
│   │   ├── depth_leverage_matrix.csv
│   │   ├── contraction_geometry.csv
│   │   ├── action_heterogeneity.csv
│   │   ├── reward_leverage_fused.csv
│   │   ├── ppo_preconditions.csv
│   │   ├── utility_summary.md
│   │   ├── candidate_ranking.md
│   │   └── <field_id>/                  # per-field detail
│   │       ├── prediction_metrics.json
│   │       ├── reachability.json
│   │       ├── greedy_saturation.json
│   │       ├── contraction_geometry.json
│   │       ├── action_heterogeneity.json
│   │       ├── reward_leverage_fused.json
│   │       ├── ppo_preconditions.json
│   │       └── bucket_u_summary.md
│   ├── interpretation/
│   │   ├── v3c_phase0_utility_audit.md
│   │   ├── v3c_phase1_ppo_smoke_summary.md
│   │   ├── v3c_phase2_new_dynamics_candidate_summary.md   # if applicable
│   │   ├── v3c_phase4_final.md                             # if escalation
│   │   └── v3c_final.md
│   ├── dynamics_candidates/             # new dynamics trained in V3C
│   │   ├── contraction_aware_v1/
│   │   ├── contraction_aware_v2/        # variant ii/iii/iv
│   │   ├── ensemble_v1/
│   │   ├── scanvi_v1/
│   │   └── zinb_v1/
│   ├── rl_smokes/                       # Phase 1 PPO_BCD smokes
│   │   ├── anchor_v2_ror_corr010_seed42/
│   │   ├── best_by_audit_<field_id>_seed42/
│   │   └── wildcard_<field_id>_seed42/
│   └── rl_final/                        # Phase 4 4-seed escalations (if any)
│       └── <field_id>_4seed_locked/
│       └── <field_id>_4seed_tuned/      # bounded tuning grid result
```

This preserves V3A and V3B artifacts in their existing flat layout under `artifacts_v3/`, and introduces a clean `v3c/` subtree without touching frozen tiers.

**Sacred rule check (no modification):**
- `artifacts/` ✓ untouched
- `artifacts_64/` ✓ untouched
- `artifacts_v2/` ✓ untouched (V2 primary dynamics is read-only)
- `artifacts/rl_sweeps/` ✓ untouched

The audit script may **read** any frozen tier (it must, to audit V2 RoR_corr010 anchor and V1 OT baseline) but writes **only** under `artifacts_v3/v3c/`.

---

## §10 — Deliverables

### New code (under `scripts/` and `src/`)

| Path | Type | Responsibility |
|---|---|---|
| `scripts/inventory_dynamics_v3c.py` | new | Auto-discover dynamics dirs under artifacts*, extract metadata, write `dynamics_inventory.csv`. Idempotent. |
| `scripts/audit_dynamics_utility_v3c.py` | new | Per-field utility audit driver. Subcommands: `prediction`, `reachability`, `greedy`, `contraction`, `heterogeneity`, `reward_leverage`, `preconditions`, `all`. Writes per-field outputs under `artifacts_v3/v3c/utility_audit/<field_id>/`. |
| `scripts/aggregate_v3c_utility_audit.py` | new | Cross-field aggregator. Reads all `<field_id>/*.json`, writes the top-level `*.csv` matrices and `utility_summary.md` + `candidate_ranking.md`. |
| `src/analysis/dynamics_utility.py` | new | Library functions: `compute_contraction_geometry(dynamics, z_sample, gene_idx_sample, z_ref) → dict`, `compute_action_heterogeneity(...) → dict`, `compute_reward_leverage(...) → dict`, `compute_utility_score(bucket_dict) → float`. All metric definitions live here (per `CLAUDE.md` Rule 4 — metrics in `src/analysis/`). |
| `tests/test_dynamics_utility.py` | new | Unit tests for utility metric functions: shape, sign, edge cases (zero μ, identity μ, single-attractor case). Mock dynamics with known geometry → expected metric values. |

### New audit outputs (under `artifacts_v3/v3c/utility_audit/`)

All listed in §9 tree. CSVs are long-format for Polars analysis; markdown summaries are human-readable.

### New interpretation docs

| Path | Phase |
|---|---|
| `artifacts_v3/v3c/interpretation/v3c_phase0_utility_audit.md` | After Phase 0D |
| `artifacts_v3/v3c/interpretation/v3c_phase1_ppo_smoke_summary.md` | After Phase 1 |
| `artifacts_v3/v3c/interpretation/v3c_phase2_new_dynamics_candidate_summary.md` | After Phase 2 (if applicable) |
| `artifacts_v3/v3c/interpretation/v3c_phase4_final.md` | After Phase 4 (if applicable) |
| `artifacts_v3/v3c/interpretation/v3c_final.md` | At V3C closeout |

### Repo-level

| Path | Type | Responsibility |
|---|---|---|
| `V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md` | new | This document, finalized at repo root. |
| `PROGRESS.md` | edit | New session entry per `CLAUDE.md` §8 format at end of each V3C phase. |
| `ARCHITECTURE.md` | edit | If Bucket U is added as a sacred-rule metric category, document it in §2 along with the existing Bucket A/B/C/D. (No interface contract changes; this is documentation only.) |

### What this plan does NOT deliver

- No new VAE training (Phase 3 if reached, but not in scope of Phase 0–1).
- No new pair construction (audit uses existing pair files; new pair files needed only for Candidate C/D).
- No 4-seed PPO training in Phase 1 (only seed 42 smoke).
- No new reward variants (locked B+C+D is the canonical reward; secondary modes already exist).
- No biological-discovery claims (per sacred rule).

---

## §11 — Execution plan (post-approval)

### Phase 0A — Inventory (≈1 hour)

1. Implement `scripts/inventory_dynamics_v3c.py`.
2. Run on `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts_v3/`.
3. Verify output `dynamics_inventory.csv` contains all expected fields from §2 roster.
4. Spot-check 2–3 entries manually against config.json.
5. Commit `feat(v3c): dynamics inventory script`.

### Phase 0B — Audit script implementation (≈4–6 hours, TDD)

Use `superpowers:test-driven-development`. One audit subcommand at a time:

- [ ] **0B-1: Prediction sub-audit (Bucket U-A)** — reuse existing `gate.json` or recompute on-demand. Write `prediction_metrics.json` per field. Unit test on synthetic dynamics. Apply prediction-pathological flag where applicable; do not exclude from downstream sub-audits.
- [ ] **0B-2: Reachability sub-audit (Bucket U-B)** — instantiate `GreedyDynamicsBeamPolicy(depth=K, beam_width=64)` with distance-only scoring on the OOD start pool. Run on the canonical 7-cell V3B matrix. Write `reachability.json`.
- [ ] **0B-3: Greedy saturation sub-audit (Bucket U-C)** — run greedy_dyn_{1,2,3,5,8} under both distance-only and fused on the 7-cell matrix. Write `greedy_saturation.json` and `depth_leverage.json`.
- [ ] **0B-4: Contraction geometry sub-audit (Bucket U-D)** — implement `compute_contraction_geometry()` in `src/analysis/dynamics_utility.py`. Sample from both OOD pool (Sample 1) and val/OOD pairs (Sample 2). Compute all 7 Bucket U-D metrics. Write `contraction_geometry.json` with separate fields for Sample 1 and Sample 2 plus divergence flag.
- [ ] **0B-5: Action heterogeneity sub-audit (Bucket U-E)** — implement `compute_action_heterogeneity()`. Reuse greedy rollouts from 0B-3.
- [ ] **0B-6: Reward leverage sub-audit (Bucket U-F)** — implement `compute_reward_leverage()`. Cross-tabulates greedy success/metrics between distance and fused. **Includes** the optional Norman measured-combo consistency diagnostic when the path-overlap set is non-empty.
- [ ] **0B-7: PPO preconditions sub-audit (Bucket U-G)** — composite check; emits a structured pass/fail with explanations.
- [ ] **0B-8: `all` subcommand** — orchestrates 0B-1 through 0B-7. Idempotent. Skips a sub-audit if output JSON already exists (unless `--force`).
- [ ] **0B-9: Aggregator** — implement `scripts/aggregate_v3c_utility_audit.py`. Produces all top-level CSVs + `utility_summary.md` + `candidate_ranking.md`. Computes `util_score`. Flags candidates.
- [ ] **0B-10: Unit + integration tests** — synthetic dynamics with known properties (universal-attractor, no-op, action-discriminating) should produce predictable bucket scores. Run `PYTHONPATH=. .venv/bin/pytest -q tests/test_dynamics_utility.py`.

### Phase 0C — Run audit on all eligible fields (≈2–4 hours runtime)

1. Confirm pair files and start pools exist for each field (some 64D fields may share pairs with V3A).
2. Run `python scripts/audit_dynamics_utility_v3c.py all --field_id <id>` per field. Parallelize if possible.
3. Run aggregator: `python scripts/aggregate_v3c_utility_audit.py`.
4. Open `utility_summary.md` and `candidate_ranking.md` for inspection.

### Phase 0D — Triage and smoke-target selection (≈1 hour)

1. Read `candidate_ranking.md` and `utility_summary.md`.
2. Write `artifacts_v3/v3c/interpretation/v3c_phase0_utility_audit.md` with:
   - Anchor: V2 RoR_corr010 (fixed)
   - Best-by-audit: top 1–2 fields by `util_score` that ideally pass Bucket U-G preconditions (with written rationale either way)
   - Wildcards: 1–2 from the audit ranking, chosen with explicit rationale per §4 Stage 3
   - Pre-commit list of all 4 smoke targets
3. Commit `docs(v3c): phase 0 utility audit interpretation`.

### Phase 1 — PPO_BCD smoke (adaptive 500k → 1M, parallel-friendly)

**0. Anchor reuse pre-check.** Before any training, search for an existing compatible V3B Phase 4 PPO_BCD seed-42 checkpoint:
   ```
   ls artifacts_v3/rl_v3b_biorealistic_fused_epsp15_seed42/ppo*.zip 2>/dev/null
   ```
   If a checkpoint exists AND its env/dynamics/ε configuration matches (`dynamics_v1ot_ror_corr010` + p15 + V3B-default freeband/λ_tox/λ_ce/λ_unc_path), **reuse** it as the anchor — no anchor retraining. Document the reuse in `v3c_phase0_utility_audit.md`. If incompatible, the anchor enters the schedule below.

**1. Smoke phase (500k, per non-reused smoke target):**
   ```
   python scripts/train_rl_v3b.py \
       --mode biorealistic_fused \
       --seed 42 \
       --timesteps 500_000 \
       --dynamics_dir <field_path> \
       --epsilon p15 \
       --out_dir artifacts_v3/v3c/rl_smokes/<smoke_class>_<field_id>_seed42_500k
   ```

**2. Mid-smoke triage (per §4 Stage 4).** Evaluate each 500k checkpoint:
   ```
   python scripts/evaluate_rl_v3b_phase4.py \
       --vae_dir <auto> \
       --dynamics_dir <field_path> \
       --pairs_dir <auto> \
       --ppo PPO_BCD_<field_id>_500k:<500k checkpoint> \
       --out_dir artifacts_v3/v3c/rl_smokes/<smoke_class>_<field_id>_seed42_500k/eval \
       --cells k2_bin6-8_splitood k2_bin8-10_splitood k3_bin6-8_splitood k3_bin8-10_splitood k4_bin8-10_splitood k5_bin8-10_splitood k8_bin8-10_splitood
   ```
   (Use the canonical 7-cell V3B matrix; pass `+k4_bin6-8 +k5_bin6-8` only if budget allows the exploratory cells.) Classify each field as `EARLY_COLLAPSED`, `EARLY_FLAT`, or `EARLY_PROMISING`.

**3. Continuation phase (1M, only for `EARLY_PROMISING` fields).** Resume from the 500k checkpoint to 1M total:
   ```
   python scripts/train_rl_v3b.py \
       --mode biorealistic_fused \
       --seed 42 \
       --timesteps 1_000_000 \
       --resume artifacts_v3/v3c/rl_smokes/<smoke_class>_<field_id>_seed42_500k/ppo.zip \
       --dynamics_dir <field_path> \
       --epsilon p15 \
       --out_dir artifacts_v3/v3c/rl_smokes/<smoke_class>_<field_id>_seed42
   ```
   (If `train_rl_v3b.py` lacks a `--resume` flag at execution time, the Phase 0B audit-script work also adds it; the V2 PPO training code in `scripts/train_rl.py` accepts an `init_model_path` argument that can be threaded through.)

**4. Final evaluation (at the field's final checkpoint — 500k or 1M):**
   ```
   python scripts/evaluate_rl_v3b_phase4.py \
       --vae_dir <auto> \
       --dynamics_dir <field_path> \
       --pairs_dir <auto> \
       --ppo PPO_BCD_<field_id>:<final checkpoint> \
       --out_dir artifacts_v3/v3c/rl_smokes/<smoke_class>_<field_id>_seed42/eval
   ```
   Same 7-cell matrix.

**5. Aggregate:** `python scripts/aggregate_v3b_phase4.py --eval_root artifacts_v3/v3c/rl_smokes`.

**6. Write `artifacts_v3/v3c/interpretation/v3c_phase1_ppo_smoke_summary.md`** with per-field final verdicts (`FAILED_IMPLEMENTATION` / `WEAK_SIGNAL` / `CANDIDATE_SIGNAL_RAW` / `CANDIDATE_SIGNAL_PARETO` / `STRONG_SIGNAL`), explicit Pareto-axis evidence where applicable, and 500k-vs-final comparison for continued fields.

**7. Commit:** `feat(v3c): phase 1 adaptive smoke + interpretation`.

### Phase 2 — New dynamics (≈4–8 hours, only if no `CANDIDATE_SIGNAL_RAW` and no `CANDIDATE_SIGNAL_PARETO` in Phase 1)

Trigger condition: no Phase 1 field reached either CANDIDATE flavor.

1. Write `V3C_PHASE2_CONTRACTION_DYNAMICS_SPEC.md` with the chosen formulation (Candidate A.iv composite recommended).
2. Implement loss term in `src/models/dynamics.py` (gated by `cfg.dynamics.lambda_contraction`).
3. Train under `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v1/`.
4. Run audit on the new field via `audit_dynamics_utility_v3c.py all --field_id contraction_aware_v1`.
5. Compare to V2 RoR_corr010 anchor — if improved Bucket U-D + maintained Bucket U-A, run PPO_BCD smoke (adaptive 500k → 1M).
6. If still no signal, iterate to next formulation (Candidate A variant, or Candidate F sparsity).

### Phase 3 — Representation candidates (only if Phase 2 fails)

Train SCANVI 32D or ZINB 64D (Candidates C/D in §8). Wallclock: 7–10 hours per candidate. Then run full pipeline (pairs → dynamics → audit → smoke).

### Phase 4 — 4-seed escalation (only if any Phase 1/2/3 field reached `CANDIDATE_SIGNAL_RAW` or `CANDIDATE_SIGNAL_PARETO`)

1. Train 3 additional seeds {0, 1, 7} of the winning configuration at 1M timesteps.
2. Run aggregator: `aggregate_v3b_phase4.py`.
3. Apply V3B verdict heuristics (`LOCKED_DESIGN_POSITIVE_SIGNAL` / `TECHNICAL_ONLY` / `FAILED_IMPLEMENTATION`), extended to credit Pareto-axis improvements that survive 4-seed aggregation.
4. Run bounded reward tuning per §6: **start with the 4-combination corner mini-grid** (seed 42 × 1M each). If sensitivity criterion fires, expand to the full 12-combination grid. If mini-grid is flat, stop and report B+C+D robustness as a positive finding.
5. Report locked-default as canonical headline, tuned variants as appendix.
6. Write `artifacts_v3/v3c/interpretation/v3c_phase4_final.md`.

### End-of-V3C closeout

1. Write `artifacts_v3/v3c/interpretation/v3c_final.md` with the consolidated narrative.
2. Update `PROGRESS.md` with the V3C entry.
3. Run frozen-tier check:
   ```
   git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/
   ```
   (Must show no modified files; only untracked files outside these dirs allowed.)
4. Run full test suite:
   ```
   PYTHONPATH=. .venv/bin/pytest -q
   ```
   (Expected: ≥ 356 + new utility-audit tests passing.)

---

## §12 — Sacred rules (preserved)

The following rules apply at execution time and are restated here for the implementer:

1. **Frozen tiers (read-only):** `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/`. Never modify. Never overwrite. May read for audit purposes.
2. **All V3C outputs** go under `artifacts_v3/v3c/`. Do not create `artifacts_v3c/` at repo root.
3. **No prediction gate lowering.** The `+0.03 ridge_margin` threshold in `dynamics_validation_gate` is **diagnostic** for Bucket U-A — do not change it to make a field "pass." It is reported as a signal, not used as a hard rejection gate.
4. **Greedy baselines mandatory.** Every PPO evaluation must include reward-aware greedy_dyn_{1, 2, 3, 5} under the same B+C+D objective. Distance-only greedy reported as a secondary column.
5. **Reward-aware greedy uses the same coefficients as PPO.** Maintains the cross-baseline fairness invariant (`src/rl/baselines.py:_score()`).
6. **No biological-discovery claims.** Chronos safety improvements are reward-fit (per V3B Phase 2b audit). Replogle held-out is the only Bucket-C source pending integration, and even then it's a validation check, not a discovery claim.
7. **Bucket separation strict (V3B-legacy meanings preserved, V3C sub-buckets renamed U-A through U-G):**
   - Bucket A (V3B-legacy) = reward-fit metrics (tox_path, CE_count, unc_path_max under fused)
   - Bucket B (V3B-legacy) = reward-independent control metrics (raw success, mean_steps, final_distance)
   - Bucket C (V3B-legacy) = held-out biology (Replogle 2022 K562 essentials when integrated; not in Phase 0)
   - Bucket D (V3B-legacy, implicit) = dynamics prediction metrics (val/OOD Pearson, ridge margin, uncertainty calibration)
   - **Bucket U (NEW, this plan) = dynamics utility metrics**, with sub-buckets **U-A through U-G** explicitly named to avoid collision with the V3B-legacy letters above.
   Reports never mix bucket sources for headline numbers. When referencing a sub-bucket of Bucket U, always use the U-prefix (e.g., "Bucket U-A" not "Bucket A").
8. **No `--no-verify` / hook skipping.** Pre-commit hooks must run.
9. **Idempotent audit.** `audit_dynamics_utility_v3c.py` re-uses existing outputs unless `--force`; safe to rerun.
10. **Pre-commit selection of smoke targets, with written rationale.** The 4 smoke fields are chosen and documented in `v3c_phase0_utility_audit.md` **before** the smokes run. Each target has a written rationale citing specific Bucket U-A through U-G evidence (or wildcard signal). No backfilling, no post-hoc reassignment.

If a sacred-rule update becomes necessary (e.g., introducing Bucket U formally in `ARCHITECTURE.md` §2 sacred-rule list), the implementer must propose the update explicitly and request user approval **before** committing the doc change.

---

## §13 — Final reasoning: "What makes a dynamics field good for CellPath now?"

A good V3C dynamics field is **not** the field with the highest validation Pearson.

A good V3C dynamics field is one that, under the locked B+C+D reward stack and the **canonical 7-cell V3B matrix**, satisfies the following as a set of **diagnostic flags** (not pass/fail gates — see triage policy in §4):

1. **Is prediction-sane** (Bucket U-A): val_pearson ≥ 0.40, ood_pearson ≥ 0.20, uncertainty_spearman ≥ 0.10. Otherwise flag as `prediction_pathological`; downstream prediction-derived metrics carry caveats but geometry/reachability sub-audits still run.
2. **Is reachable** (Bucket U-B): beam_reach_at_K=8/bin8-10/OOD ≥ 0.30 at p15. The Soft-OT failure mode is exactly this criterion failing.
3. **Is not trivially greedy-saturated** (Bucket U-C): greedy_dyn_1_distance ≤ 0.95 in ≥ 3 cells; greedy_dyn_5_distance ≤ 0.99 in ≥ 1 K ≥ 4 cell. Otherwise PPO has limited room to add value (but still worth smoking if other signals are unusual).
4. **Has depth leverage** (Bucket U-C): cumulative_depth_leverage ≥ 0.05 in ≥ 1 K ≥ 4 cell. Otherwise multi-step planning is decorative.
5. **Has action heterogeneity** (Bucket U-E): first_action_entropy_fused ≥ 0.3 · log(n_genes); path_diversity_depth2 ≥ 0.30. Otherwise one universal-attractor gene dominates.
6. **Has reward leverage under B+C+D** (Bucket U-F): distance_vs_fused_first_action_overlap ≤ 0.85 in ≥ 3 cells. Otherwise the locked reward stack changes nothing on this field.
7. **Has non-degenerate contraction geometry** (Bucket U-D): contraction_fraction ∈ [0.30, 0.95], action_diversity_per_state ≥ 0.05, gene_universality_max ≤ 0.70. Otherwise either dead (Soft-OT-like) or over-contractive (V2-saturation-like).
8. **Makes uncertainty / path / safety meaningful** (Bucket U-F sub-axis): fused-greedy reduces mean_unc_path_max relative to distance-greedy by ≥ 0.10 at ≥ 1 cell **without** killing V3B-legacy Bucket B success. Otherwise Variant D is decorative.
9. **Plans biology-overlapping paths** (Bucket U-F Norman-combo realism, optional): non-zero `fraction_paths_with_measured_combo_overlap` with positive `measured_combo_latent_consistency`. A tie-breaker, not a gate.
10. **Is learnable by MaskablePPO** (Bucket U-G): all 8 preconditions ideally hold for the "Best-by-audit" route. Wildcards explicitly relax this — a single strong positive signal can compensate for failed preconditions, with written rationale.
11. **Does not require ad-hoc per-cell masks or coefficient-overfitting** to express its signal: the locked B+C+D defaults (λ_tox=0.10, λ_ce=0.05, λ_unc_path=0.05, freeband {3, 5, 0.02, 0.10}) are the canonical objective; the bounded mini-grid (§6) is a secondary appendix only.

**Triage interpretation (per §4):** A field that passes all flags is promoted to PPO smoke as Best-by-audit. A field that fails some but has unusual positive signals (geometry-disagreement, depth-leverage, Norman-combo-realism, action-heterogeneity, pair-source diversity, soft-OT-vs-control disagreement) is promoted as a Wildcard with written rationale. **Hard rejection is reserved for Stage 0 failures** (model fails to load, malformed config, forward pass raises, numerical collapse) — not for failing any subset of (1)–(11).

The reason the V2 RoR_corr010 32D field stalled at `LOCKED_DESIGN_TECHNICAL_ONLY` is failure on (3), (4), and (7): greedy_dyn_1 saturates K=3/bin8-10/OOD at 1.000; depth leverage is essentially zero past K=2; contraction geometry shows near-universal alignment with z_ref. The V3B reward stack is correct — the V2 dynamics field cannot express what the reward is asking.

The Soft-OT case (failure on (2) and likely (5)) is the diagnostic-disagreement signature: prediction gate passes, yet beam reachability is near-zero and action heterogeneity is plausibly near-zero. Whether the failure mechanism is literal μ ≈ 0, action-homogenization, or single-attractor concentration is exactly what Bucket U-D Sample 1/Sample 2 diagnostics decide empirically.

V3C succeeds when an existing or newly-trained dynamics field reaches Phase 1 verdict `CANDIDATE_SIGNAL_RAW` (PPO_BCD beats greedy_dyn_5_fused by ≥ 0.05 raw success at a non-saturated cell) **or** `CANDIDATE_SIGNAL_PARETO` (PPO_BCD ties greedy within ±0.03 but improves ≥ 2 reward/utility axes without hurting final distance/path realism). Either flavor is a Phase 4 escalation trigger — the Pareto path explicitly credits the V3B-intended biorealistic-control payoff.

---

## §14 — Next normal-mode execution prompt

After this plan is approved via `ExitPlanMode`, the following prompt may be pasted in normal mode to kick off Phase 0A. **Do not paste this yet.** It is the post-approval starter.

> Begin V3C Phase 0A per `V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md`.
>
> 1. Implement `scripts/inventory_dynamics_v3c.py` per §10 / §11 Phase 0A. The script must:
>    - Walk `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts_v3/` (max depth 5).
>    - Identify each directory containing both `model.pt` and `config.json` as a candidate dynamics field.
>    - Extract metadata per §2 columns (n_lat, pair source, RoR flag, λ_corr, val Pearson, OOD Pearson, gate margin, gate passed, mtime).
>    - Mark eligibility per §2 rules and audit_class (Anchor / Eligible / Eligible-conditional).
>    - Write `artifacts_v3/v3c/utility_audit/dynamics_inventory.csv`.
>    - Be idempotent (re-running produces the same output unless artifacts change).
> 2. Run the script and verify the V2 primary `artifacts_v2/dynamics_v1ot_ror_corr010` is in the output, marked `Anchor`.
> 3. Spot-check 2–3 other entries against their `config.json`.
> 4. Commit: `feat(v3c): dynamics inventory script and initial roster`.
>
> Sacred rules: do not modify `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/`. All V3C outputs go under `artifacts_v3/v3c/`. Run `PYTHONPATH=. .venv/bin/pytest -q` after committing.
>
> After Phase 0A is committed, await my confirmation before proceeding to Phase 0B (audit-script TDD implementation).

---

## Appendix A — Glossary

| Term | Meaning |
|---|---|
| **Anchor** | V2 RoR_corr010 32D, always smoked as the fixed reference field; reuse existing V3B Phase 4 PPO_BCD seed-42 checkpoint when configuration matches |
| **Best-by-audit** | Top 1–2 fields by composite `util_score` that ideally pass Bucket U-G preconditions; written rationale required regardless |
| **Wildcard** | Field that fails Bucket U-G but has at least one strong positive signal (geometry-disagreement, depth-leverage, action-heterogeneity, pair-source diversity, Soft-OT-vs-control disagreement, Norman-combo realism); chosen interpretively with written rationale |
| **Bucket A (V3B-legacy)** | Reward-fit metrics (tox_path, CE_count, unc_path_max) — V3B-defined |
| **Bucket B (V3B-legacy)** | Reward-independent control metrics (raw success, T, final dist) — V3B-defined |
| **Bucket C (V3B-legacy)** | Held-out biology (Replogle 2022, pending) — V3B-defined |
| **Bucket D (V3B-legacy)** | Dynamics prediction metrics (val/OOD Pearson, ridge margin) — implicit in V2 |
| **Bucket U (new, this plan)** | Dynamics utility metrics; sub-buckets **U-A through U-G** |
| **Bucket U-A** | Prediction sanity (val/OOD Pearson, uncertainty calibration, per-dim/per-gene distribution) |
| **Bucket U-B** | Reachability (beam_reach_at_K_p15 on canonical 7-cell V3B matrix) |
| **Bucket U-C** | Greedy saturation + depth leverage (distance and fused) |
| **Bucket U-D** | Contraction geometry (contraction_fraction, action_diversity, gene_universality; Sample 1 = OOD pool, Sample 2 = val pairs) |
| **Bucket U-E** | Action heterogeneity + path diversity |
| **Bucket U-F** | Reward leverage under locked B+C+D; includes optional Norman measured-combo consistency diagnostic |
| **Bucket U-G** | PPO learnability preconditions (composite of U-A through U-F + no-op and random checks) |
| **CANDIDATE_SIGNAL_RAW** | Phase 1 verdict: PPO_BCD beats greedy_dyn_5_fused by ≥ 0.05 raw success at a non-saturated cell — escalate to Phase 4 |
| **CANDIDATE_SIGNAL_PARETO** | Phase 1 verdict: PPO_BCD ties greedy_dyn_5_fused on raw success within ±0.03 at a non-saturated cell, but improves ≥ 2 reward/utility axes without hurting final distance/realism — escalate to Phase 4 |
| **STRONG_SIGNAL** | Phase 1 verdict: `CANDIDATE_SIGNAL_RAW` at multiple cells with margin ≥ 0.10 (gold-standard) |
| **WEAK_SIGNAL** | Phase 1 verdict: no raw advantage and no Pareto improvement; V2-saturation-like outcome |
| **LOCKED_DESIGN_FAILED_IMPLEMENTATION** | Phase 1 verdict: PPO_BCD collapses on K=2/bin6-8/OOD (< 0.30); implementation broken on this dynamics |
| **EARLY_PROMISING / EARLY_FLAT / EARLY_COLLAPSED** | Mid-smoke triage labels at the 500k checkpoint, determining whether to continue to 1M |
| **fused-greedy** | `GreedyDynamicsBeamPolicy` with locked B+C+D coefficients |
| **distance-greedy** | `GreedyDynamicsBeamPolicy` with λ_tox = λ_ce = λ_unc_path = 0 |
| **Locked reward stack** | `biorealistic_fused` with V3B Phase 4 defaults (λ_tox=0.10, λ_ce=0.05, λ_unc_path=0.05, freeband {3, 5, 0.02, 0.10}) |
| **Canonical 7-cell V3B matrix** | K=2/bin6-8, K=2/bin8-10, K=3/bin6-8, K=3/bin8-10, K=4/bin8-10, K=5/bin8-10, K=8/bin8-10 — all OOD (per `V3_CONTROLLER_OBJECTIVE_SPEC.md`) |
| **Optional exploratory cells** | K=4/bin6-8/OOD, K=5/bin6-8/OOD (the "+2" cells in `evaluate_rl_v3b_phase4.py`) |
| **Prediction-pathological flag** | val_pearson < 0.10 or per_dim_pearson median < 0.05; **not** a hard rejection — geometry/reachability still audited |
| **Sample 1 / Sample 2 (Bucket U-D)** | OOD start pool / held-out val pairs; reported separately, divergence > 0.30 itself is a signal |
| **util_score** | Composite ranking aid, never a verdict; written rationale always required for smoke selection (§4 Stage 1, Stage 3) |
| **Norman-combo realism** | Optional Bucket U-F diagnostic: fraction of successful 2-step paths overlapping Norman 2019 measured combos + latent consistency on the overlap |

---

## Appendix B — Self-review checklist (writing-plans skill)

This plan has been self-reviewed against the writing-plans skill checklist:

**1. Spec coverage:**
- §1 covers user-prompt point 1 (why classical gate is insufficient).
- §2 + scripts/inventory_dynamics_v3c.py covers point 2 (inventory).
- §3 covers point 3 (new utility gate, multi-bucket).
- §4 covers point 4 (filtering funnel, with user's exploratory two-tier override).
- §5 covers point 5 (locked B+C+D as primary, secondary modes for decomposition).
- §6 covers point 6 (bounded reward tuning, deferred).
- §7 covers point 7 (literature support).
- §8 covers point 8 (new dynamics candidates A–F, with Candidate E deferred per user direction).
- §9 covers point 9 (artifact organization; recommends `artifacts_v3/v3c/`).
- §10 covers point 10 (deliverables listing).
- §11 covers point 11 (execution plan by phase, Phase 0A–4).
- §12 covers point 12 (sacred rules preserved).
- §13 covers point 13 (final reasoning, 11-flag diagnostic list with triage interpretation).
- §14 covers the requested next normal-mode execution prompt.

**1b. User-revision coverage (post-Plan-Mode iteration 1):**
- (1) "Fail any criterion = reject" → triage language: Central question, §3 Bucket U-A, §4 Stage 3, §13 all rewritten in triage flavor.
- (2) 9-cell → 7-cell canonical V3B matrix: §3 Bucket U-B, §11 Phase 1, §13, Glossary all updated.
- (3) Sub-buckets renamed U-A through U-G: §3 headings, §4 references, §11 Phase 0B tasks, §12 sacred rules, §13, Glossary all updated.
- (4) Prediction hard reject softened to "prediction_pathological flag": §3 Bucket U-A, §4 Stage 0 (model-load only), §12 rule 3, §13 item 1 all updated.
- (5) Soft-OT rephrased as suspected barycentric/action-homogenized: §1.2, §13 closing paragraph.
- (6) Phase 1 smoke adaptive 500k → 1M with anchor reuse: §4 Stage 4, §11 Phase 1.
- (7) CANDIDATE_SIGNAL_RAW + CANDIDATE_SIGNAL_PARETO: §4 Stage 4, §4 Stage 5/6, §11 Phase 2/4, §13, Glossary.
- (8) util_score ranking aid only + written rationale: §3 prelude, §4 Stage 1, Stage 3, §12 rule 10.
- (9) Mini-grid first: §6 (two-step expansion policy), §11 Phase 4.
- (10) Norman measured-combo realism diagnostic: §3 Bucket U-F, §11 Phase 0B-6, Pareto verdict, Glossary.

**2. Placeholder scan:**
- No TBDs, no "implement appropriately", no "fill in later". File paths are concrete, metric names match `src/rl/baselines.py` and `src/rl/biology_rewards.py` code, threshold values are quantitative.
- One area marked as "deferred to phase-specific spec": new-dynamics formulations in §8. This is intentional — formulations should be chosen with Phase 1 audit data in hand, not committed now.

**3. Type consistency:**
- `GreedyDynamicsBeamPolicy` signature referenced matches `src/rl/baselines.py:184–503`.
- `biorealistic_fused` mode name matches `src/rl/reward.py:200–221`.
- File paths under `artifacts_v3/v3c/` use kebab-case consistent with existing V3B conventions (`v3b_reward_stack_lock.md`, etc.).
- `util_score` formula sums to 1.00 weight (`0.20 + 0.20 + 0.20 + 0.15 + 0.10 + 0.10 + 0.05`).
- Sub-bucket naming U-A through U-G applied consistently across §1, §3, §4, §8, §11, §12, §13, Glossary. V3B-legacy Bucket A/B/C/D references are explicitly labeled "(V3B-legacy)" where they appear.

Self-review status: **PASS** — all 10 user revisions applied; no further inline edits needed.
