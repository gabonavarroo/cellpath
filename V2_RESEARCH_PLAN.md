# CellPath V2 Research Plan

> Status: approved for implementation. Author: V2 research lead. Date: 2026-05-15. Branch: `gabo2`.
> All artifact references are local paths under `/Users/gabo/Developer/ITAM/IA/cellpath`.
> V1 artifacts are **read-only**. V2 outputs go under `artifacts_v2/` (and `artifacts_v2_64/` if 64D is revisited). No V1 outputs are deleted, renamed, or overwritten.
> **Implementation order is strict: P0A first and alone; do not begin P0B until P0A results have been reviewed.**

## 1. Executive diagnosis

V1 is an honest in-silico steering pipeline that **shipped with one gate failure overridden and a learned dynamics field that is globally contractive**, so the headline number (PPO 0.988 vs random uniform-valid 0.840 success at p50 / start8 / K=10) is not a strong test of policy quality. Three findings dominate everything in this plan:

1. **The dynamics gate fails on exactly one term: `margin_vs_linear_ridge_pearson = +0.0074` against the required `+0.030`.** All four other R²-based margins pass with large positive margins (no-op +0.396, global-mean +0.386, per-gene-mean +0.241, kNN +0.308). Uncertainty Spearman passes (0.249). The pattern says the MLP is doing essentially what a `Ridge(z_ctrl ⊕ one_hot(gene))` does — it has not learned the residual nonlinearity that would separate it from a properly featurised linear model. The `use_state_linear_skip=true` flag, which adds a gene-independent linear `state_linear(z)` term to the output, mechanically pushes the MLP toward the ridge solution. We are training the MLP to be linear in `z`. *(`artifacts/dynamics/gate.json`; `src/models/dynamics.py:148-161`.)*
2. **The RL surrogate environment is too easy because the learned dynamics is a global attractor.** Contraction diagnostics show 100 % of 11 760 (start, action) pairs reduce ‖z − z_ref‖ with mean improvement 2.737 (manual starts) and 95.5 % of 105 000 (start, action) pairs reduce distance with mean improvement 1.008 (auto starts). With p50 ε = 3.531, min_start_distance = 8.0, and K = 10 steps, almost any sequence of actions reaches the goal: gap = 8 − 3.5 = 4.5 ≤ 1.0 × 10. Random success of 0.840 is the geometric consequence of contraction × steps, not a fair RL baseline. *(`artifacts/contraction/summary.json`, `artifacts/contraction_auto/summary.json`, `artifacts/rl_sweeps/.../random_baseline/summary.json`.)*
3. **The reward used to train PPO is dense absolute distance `−‖z_{t+1} − z_ref‖`**, which is the worst possible shape for separating PPO from a "global pull" policy: every action with a contractive Jacobian receives positive shaped reward, so a competent random policy and PPO collect very similar return per step. *(`reward_hparams.intermediate_shaping=true, reward_mode=absolute_distance` in `artifacts/rl_sweeps/.../eval_deterministic/metadata.json`; `src/rl/reward.py:82-102`.)*

Three secondary findings inform the prioritisation:

4. **64D is not categorically worse than 32D for dynamics quality**, despite v1's framing. The 64D `state_linear` variant has val Pearson 0.5965 and OOD Pearson 0.3686 vs 32D's 0.6085 / 0.4793. The 64D `state_linear_combo0` variant has val Pearson 0.6000 with OOD R² −0.120. What is worse at 64D is **`use_gene_delta_bias=true`**, which catastrophically degrades OOD (R² down to −1.825 / −2.317). The narrative "64D is worse" generalises a flag-specific failure into a dimension-wide claim. 32D should remain the primary track but 64D state_linear is a defensible ablation for V2.
5. **DepMap enrichment is statistically null but the test is underpowered and binary.** Hypergeometric on top-20 PPO genes against 2000 HVG background gives q = 0.837. The PPO action distribution is concentrated on 23 unique genes out of 105 available; CKS1B (274) and HK2 (in top-20) drive weighted mean Chronos −0.168 vs random −0.066. The right test is a *ranked-correlation* test over the entire 105-gene action universe, with PPO action frequency as the rank, not a top-K hypergeometric over an arbitrary cutoff.
6. **Random uniform-valid is the wrong upper baseline for "non-trivial control" claims.** It tests "are some learned actions better than uniform random gene choice", not "did PPO learn anything beyond what a one-step greedy planner over the dynamics already knows". With a globally contractive surrogate, a *greedy-dynamics oracle* that picks `argmin_a ‖f_θ(z, a) − z_ref‖` is the right *upper-reference* baseline. **However, greedy has oracle access to the learned dynamics**; if PPO ≥ random by a large margin and approaches greedy, that is a success. Only beating greedy is evidence of multi-step planning beyond one-step contraction.

The plan that follows treats (1) and (3) as the two main blockers but defers any model change until the **P0A forensic benchmark** has clarified exactly how easy the current task really is, which baselines are non-trivial, and where the dynamics gate concentrates its failure. (2) is then unblocked by (3). (4)–(6) shape the V2 reporting protocol.

## 2. Artifact inventory

### 2.1 VAE runs

| Path                 | n_latent | ε (p50) source                              | git commit | Notes                                                                             |
|----------------------|---------:|---------------------------------------------|-----------|-----------------------------------------------------------------------------------|
| `artifacts/vae`      |       32 | `epsilon_success.json` → 3.531 (p50, L2)    | 1e7969c   | Primary VAE; `epsilon_success_backup_p90.json` → 4.4398; trained on Norman 2019.   |
| `artifacts_64/vae`   |       64 | `epsilon_success.json` → 4.4347 (p50, L2)   | 1e7969c   | Isolated 64D VAE; epsilon naturally larger because higher-dim L2 norms inflate.    |

### 2.2 Dynamics runs (32D)

| Path                                              | state_linear / gene_bias | lr     | epochs | sel_metric  | val Pearson | ridge Pearson | margin    | OOD Pearson | OOD R² | unc Spearman | gate passed |
|---------------------------------------------------|:------------------------:|:------:|:------:|:-----------:|------------:|--------------:|----------:|------------:|-------:|-------------:|:-----------:|
| `artifacts/dynamics` (primary)                    | T / F                    | 1e-4   | 300    | gate_margin | 0.6085      | 0.6011        | **+0.0074** | 0.4793      | 0.263  | 0.249        | **false**   |
| `artifacts/dynamics_current_default`              | T / F                    | 1e-4   | 300    | gate_margin | 0.6085      | 0.6011        | +0.0074   | 0.4793      | 0.263  | 0.249        | false       |
| `artifacts/dynamics_default_check`                | T / F                    | 1e-4   | 300    | gate_margin | 0.6085      | 0.6011        | +0.0074   | 0.4793      | 0.263  | 0.249        | false       |
| ablation: `baseline`                              | F / F                    | 1e-4   | 300    | gate_margin | 0.6000      | 0.6011        | −0.0011   | —           | 0.251  | —            | false       |
| ablation: `state_linear`                          | T / F                    | 1e-4   | 300    | gate_margin | 0.6031      | 0.6011        | +0.0020   | —           | —      | —            | false       |
| ablation: `gene_bias`                             | F / T                    | 1e-4   | 300    | gate_margin | 0.5853      | 0.6011        | −0.0158   | —           | —      | —            | false       |
| ablation: `state_linear_gene_bias`                | T / T                    | 1e-4   | 300    | gate_margin | 0.5829      | 0.6011        | −0.0181   | —           | —      | —            | false       |
| `dynamics_sweeps/lr1e-4_mse0`                     | T / F                    | 1e-4   | —      | gate_margin | 0.6076      | 0.6011        | +0.0065   | 0.4789      | 0.263  | —            | false       |
| `dynamics_sweeps/lr3e-4_mse0`                     | T / F                    | 3e-4   | —      | gate_margin | 0.6057      | 0.6011        | +0.0046   | 0.4805      | 0.263  | —            | false       |
| `dynamics_sweeps/lr3e-4_mse005`                   | T / F                    | 3e-4   | —      | gate_margin | 0.6058      | 0.6011        | +0.0047   | 0.4805      | 0.263  | —            | false       |
| `dynamics_sweeps/lr3e-4_mse01`                    | T / F                    | 3e-4   | —      | gate_margin | 0.6058      | 0.6011        | +0.0047   | 0.4805      | 0.263  | —            | false       |
| `dynamics_sweeps/lr1e-3_mse0`                     | T / F                    | 1e-3   | —      | gate_margin | 0.6031      | 0.6011        | +0.0020   | 0.4854      | 0.265  | —            | false       |

**OOD ridge-margin actually passes** for the primary 32D model (+0.0401 vs +0.030). The failure is exclusively the *in-distribution* val ridge-Pearson margin. *(`artifacts/dynamics/gate.json::ood.margin_checks`.)*

### 2.3 Dynamics runs (64D)

| Path                                                    | state_linear / gene_bias | val Pearson | val R² | OOD Pearson | OOD R²  | gate passed |
|---------------------------------------------------------|:------------------------:|------------:|-------:|------------:|--------:|:-----------:|
| `artifacts_64/dynamics`                                 | T / F                    | 0.5965      | 0.4012 | 0.3686      | 0.254   | false       |
| `dynamics_variants/baseline_plain`                      | F / F                    | 0.5958      | 0.3992 | 0.3859      | 0.248   | false       |
| `dynamics_variants/state_linear`                        | T / F                    | 0.5965      | 0.4012 | 0.3686      | 0.254   | false       |
| `dynamics_variants/gene_bias`                           | F / T                    | 0.5157      | 0.3844 | 0.1191      | −1.825  | false       |
| `dynamics_variants/state_linear_gene_bias`              | T / T                    | 0.5617      | 0.3822 | 0.1053      | −2.317  | false       |
| `dynamics_variants/state_linear_combo0`                 | T / F                    | 0.6000      | 0.3984 | 0.3047      | −0.120  | false       |

### 2.4 RL runs

| Path                                                                                 | Policy        | ε source                | min_start_dist | n_eps | success | mean_steps | mean_final_d | top 5 actions                                                                 |
|--------------------------------------------------------------------------------------|---------------|-------------------------|:--------------:|:-----:|:-------:|:----------:|:------------:|-------------------------------------------------------------------------------|
| `artifacts/rl_sweeps/p50_start8_200k`                                                | PPO (200k)    | p50 = 3.531             | 8.0            | —     | 0.484   | 1.48       | 5.857        | NO_OP(258), CKS1B(119), TSC22D1(77), LYL1(73), MAP4K3(47) — heavy NO-OP bias  |
| `artifacts/rl/` (legacy)                                                             | PPO (2M)      | p50 = 3.531             | 8.0            | —     | —       | —          | —            | NO_OP(251), CKS1B(114), LYL1(79), TSC22D1(60), MAP4K3(60)                     |
| `p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic`                       | PPO det       | override 3.53108 (p50)  | 8.0            | 500   | 0.988   | 2.28       | 3.029        | CKS1B(274), TSC22D1(143), CELF2(136), CSRNP1(84), KIF2C(58)                   |
| `p50_start8_shaped_noopfix_500k_clean_eval/eval_stochastic`                          | PPO stoch     | override 3.53108 (p50)  | 8.0            | 500   | 0.988   | 2.29       | —            | (same head; entropy regularised tail)                                          |
| `p50_start8_shaped_noopfix_500k_clean_eval/random_baseline`                          | random        | override 3.53108 (p50)  | 8.0            | 500   | 0.840   | 5.53       | 3.411        | PTPN13(38), FOSB(38), IGDCC3(37), S1PR2(36), UBASH3A(35) — flat over 105 genes |

Confirmed PPO HPs (`metadata.json` of det eval): total_timesteps = 2 000 000, n_steps = 1024, batch = 256, n_epochs = 10, lr = 3e-4, γ = 0.99, λ_gae = 0.95, clip = 0.2, ent_coef = 0.01, vf_coef = 0.5, max_grad_norm = 0.5, policy = MlpPolicy [128, 128] tanh. Confirmed reward: `intermediate_shaping=true`, `reward_mode="absolute_distance"`, `lambda_sparse=0.05`, `lambda_unc=0.0`, `success_bonus=0.0`, `failure_penalty=0.0`, `distance_scale=1.0`. Confirmed flags: `dynamics_gate_passed=false`, `dynamics_gate_overridden=true`. Confirmed scope: max_steps = 10, deterministic_eval = true for the headline run.

### 2.5 Contraction diagnostics

| Path                                            | n_starts | n_pairs | frac improved | mean improvement | median | std   | worst    | interpretation                                                |
|-------------------------------------------------|---------:|--------:|--------------:|-----------------:|-------:|------:|---------:|---------------------------------------------------------------|
| `artifacts/contraction` (32D, state_linear)     | 112      | 11 760  | 1.000         | 2.737            | 2.731  | 0.707 | +0.597   | Per-cell action-set is unanimously distance-reducing.          |
| `artifacts/contraction_auto` (32D, 1k starts)   | 1 000    | 105 000 | 0.955         | 1.008            | 1.015  | 0.579 | −1.864   | On a random latent grid, 95.5 % of actions still contract.     |
| `artifacts_64/contraction` (64D, state_linear)  | 55       | 5 775   | 1.000         | 3.282            | 3.342  | —     | +0.946   | 64D state_linear also globally contractive.                    |
| `artifacts_64/contraction_baseline_plain`       | 55       | 5 775   | 1.000         | 3.097            | 3.164  | —     | +0.808   | Even without the linear skip, contraction holds.               |
| `artifacts_64/contraction_gene_bias`            | 55       | 5 775   | 0.932         | 2.329            | 2.664  | —     | −6.176   | gene_bias variant has rare large divergences (worst −6.2).     |
| `artifacts_64/contraction_state_linear_gene_bias` | 55     | 5 775   | 0.903         | 2.239            | 2.671  | —     | −4.852   | Combined flag worst-case is degradation.                        |
| `artifacts_64/contraction_auto`                 | 1 000    | 105 000 | 0.984         | 1.349            | 1.356  | —     | −1.499   | 64D auto starts contract more reliably than 32D auto.          |

**Reading:** the dynamics field learned by `state_linear`-on at any dimension behaves like an affine contraction toward a region containing z_ref. This is what makes the RL task easy and what makes ridge-Pearson hard to beat — both are downstream of the same fact. **V2 extends this with a per-gene action-contraction diagnostic (§7.A.2) that quantifies whether all genes are equally contractive or whether a meaningful subset stands out — the former would imply PPO has nothing to learn beyond "any non-NOOP action".**

### 2.6 Biological evaluation outputs

| Path                                                | Test                                                  | n  | Result                                                       | Significant? |
|-----------------------------------------------------|-------------------------------------------------------|:--:|--------------------------------------------------------------|:------------:|
| `artifacts/eval/depmap_enrichment.csv`              | Hypergeometric, PPO top-20 vs K562 essentials         | 20 | p = 0.837, q = 0.837                                          | No           |
| `artifacts/eval/depmap_gene_level_scores.csv`       | Per-gene scoring (Chronos, presence, essential flag)  | 105 | weighted mean Chronos: PPO det −0.168, random −0.066          | Marginal     |
| `artifacts/eval/depmap_comparison_summary.json`     | MWU + permutation null over top-20 lists              | —  | n_perm = 10 000, ppo_det unique = 23, ppo_stoch unique = 27   | Not stated   |
| `artifacts/eval/evaluate_report.json`               | Aggregator                                            | —  | gate_overridden = true; all stages succeeded                  | —            |

## 3. Reconstructed best V1 configuration

Verified from `artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic/metadata.json` and `artifacts/dynamics/config.json`:

- **VAE**: `n_latent=32`, primary VAE under `artifacts/vae`. ε = 3.5311 = p50 of `‖z_ctrl_i − z_ref‖₂` (n_ctrl = 11 855). z_ref = mean of control-cell latents (perturbation_idx == 0).
- **Pairing**: OT (entropic Sinkhorn), `ot_epsilon=0.05`, `ot_iter=500`, `normalize_cost=true`. 80/20 gene split for OOD; 90/10 cell split within train for in-distribution val (`config/default.yaml`, `src/data/perturbation_pairs.py:208-258`).
- **Dynamics (best V1)**: `use_state_linear_skip=true`, `use_gene_delta_bias=false`, `n_latent=32`, `n_hidden=256`, `n_layers=3`, `d_emb=64`, lr=1e-4, max_epochs=300, early_stop_patience=35, `selection_metric="gate_margin"`, `lambda_mse_delta=0.0`, `lambda_combo=0.5` (default; combos pairs file exists), `log_var_reg=0.01`. Heteroscedastic NLL on Δz = z_pert − z_ctrl. SHA256 of model.pt: `278028682e…b03510`.
- **RL eval**: epsilon_override = 3.53108 (i.e. p50), min_start_distance = 8.0, max_steps = 10, n_episodes = 500, deterministic = true (and stochastic ran with the same settings). PPO HP and reward HP exactly as listed in §2.4. Reward mode = `absolute_distance` with intermediate shaping. Reward = `−d_next − 0.05·𝟙[a≠NOOP]` per step (no terminal bonus, no truncation penalty, no uncertainty term).
- **Caveats locked in `artifacts/eval/caveats.md`**: (i) gate override, (ii) Markov assumption on cross-sectional Perturb-seq, (iii) globally contractive surrogate, (iv) DepMap non-significant. These are the four honest disclaimers and V2 must preserve them.

The summary (p50 / start8 / lr=1e-4 / 300 epochs / deterministic) is confirmed correct against these files.

## 4. Failure analysis

### 4.1 Dynamics gate

- **Which terms fail.** Exactly one: `margin_vs_linear_ridge_pearson = +0.00737` against threshold `+0.030`. R²-based margins against no-op, global-mean, per-gene-mean, and kNN all pass with large positive margins (+0.396, +0.386, +0.241, +0.308). Uncertainty Spearman passes (0.249, threshold 0.20). The model crushes naive baselines but ties a strong featurised linear model on Pearson.
- **What this means.** The ridge baseline is `Ridge(z_ctrl ⊕ one_hot(gene), Δz, α=1.0)`. The `one_hot(gene)` block encodes a per-gene additive Δz, and the z_ctrl block encodes a linear contraction in state. With `use_state_linear_skip=true` the MLP's forward pass is `mu = state_linear(z) + mlp(z, gene_emb)`. The state_linear branch is structurally identical to the ridge's z_ctrl half. If `mlp(z, gene_emb)` learns mostly bias-per-gene, then `mu ≈ state_linear(z) + per_gene_bias(gene)`, which is ridge with a slightly more expressive but redundantly parameterised solution. **This is exactly the model V1 trains.** The gate's purpose is to demand that the MLP do something nonlinearly conditional on gene × state — i.e. learn that gene A pushes differently from different start states — and the V1 loss does not directly reward this.
- **Where the failure concentrates.** TBD by P0A.1 diagnostic (gate_breakdown). Plausible decompositions are (a) a small subset of high-variance latent dimensions where Δz has strong gene-conditional nonlinearity that the current MLP under-captures, or (b) a small subset of high-effect-size genes whose Δz shape is poorly approximated by the linear ridge but the MLP only marginally improves. The existing `gate_diagnostics.json` carries per-dim and per-gene columns; reading these is the very first action of P0A.
- **Is ridge an unfair baseline?** No. The gate is correctly demanding: a *featurised linear model with full gene memorisation* is a real baseline. If the MLP does not beat that, the MLP's nonlinearity adds no predictive value. The right complaint is not "the gate is unfair" but "the loss does not directly optimise the gate metric".
- **Heteroscedastic NLL vs Pearson.** Per-element NLL is `0.5·exp(−log_var)·(Δ−μ)² + 0.5·log_var + 0.01·log_var²`. The gradient on μ is `exp(−log_var)·(μ − Δ)`. This is a precision-weighted MSE, not a Pearson optimiser. High-uncertainty dimensions get downweighted in the gradient, which can *reduce* the per-dim Pearson the gate cares about.
- **z_pert vs Δz consistency.** Training target is Δz = z_pert − z_ctrl (`src/models/dynamics.py:204-235`). Gate evaluates on `delta_true = z_pert_true_v − z_ctrl_v` (`src/analysis/metrics.py:300`). Consistent.
- **Checkpoint selection.** `selection_metric="gate_margin"` picks the best `val_mlp_minus_ridge_pearson` epoch. This is correct in spirit, but the *loss being optimised* during that epoch is NLL, so we are selecting the best NLL checkpoint that happens to coincide with the largest Pearson margin. A direct Pearson loss term would tighten this coupling.
- **Composition loss.** `lambda_combo = 0.5` is on by default. It optimises MSE between two-step rollouts and ground-truth combo outcomes. Effect on the single-step val ridge-Pearson margin is unclear from current artifacts and worth ablating (P1).
- **Uncertainty vs Pearson trade-off.** Uncertainty Spearman is +0.249 (passes). There is no current evidence that calibration is trading off against Pearson; both seem co-improvable.
- **VAE geometry.** The fact that ridge captures 0.601 Pearson on Δz from z_ctrl + gene-onehot is already a statement that the latent geometry encodes most of the gene-effect linearly. An expressive nonlinear function class should still beat that by ≥ 0.030 if the gene effects are state-dependent. A separate VAE diagnostic should ask: "given (z_ctrl, gene), how much of Δz variance is irreducible (residual noise of the OT pseudo-pair) versus latent-geometry-bound (no nonlinear lift can recover it)?"
- **OT pairing noise as upper bound.** OT pseudo-pairing introduces an upper bound: the "true" Δz for a given control cell against a given perturbed cell is identifiable only up to the matching. Ridge averages over this noise smoothly; a more flexible function class can overfit it. This may explain the small margin without invoking dynamics-model failure. **Quantify in P0A.4** by computing per-gene within-perturbation Δz variance under OT vs synthetic mean_delta.
- **Splits.** Train/val/OOD is constructed by gene-level 80/20 plus per-train-gene 90/10 cell split. The in-distribution val cells *share genes with training cells*, so the gene-onehot ridge baseline is very strong on val. On OOD (held-out genes), the gene-onehot column is all zeros for unseen genes and ridge becomes much weaker — which is exactly what `gate.json::ood` shows (ridge OOD Pearson 0.4392, MLP OOD Pearson 0.4793, margin +0.0401, passes). **The gate's in-distribution Pearson margin is hardest precisely because the gene-onehot column is most informative there.** This is a feature of the test, not a bug.

### 4.2 Random baseline too strong

- **Why random is at 0.840.** Min_start_distance = 8.0 → starts are 8 + units from z_ref. ε = 3.531. Gap to close = 4.469. K = 10 steps. Mean per-step contraction (auto pairs, the appropriate proxy) ≈ 1.008. Even worst-case auto contraction (`worst_improvement=−1.864`) is rare (≤ 4.5 %) and variance allows ε to be crossed within 5–6 steps on most random trajectories. PPO simply selects the highest-contraction action sequence and gets there in 2.28 steps.
- **Is NO-OP unfair?** No. Random `uniform_valid` does sample NO-OP, and `noop_first_action_rate = 0.008` (4/500). NO-OP terminates without success when d > ε, which is the correct semantics.
- **Repeat mask effect on random.** Mild. After step t, the random policy samples from a smaller set, but the mask never removes NO-OP. Random still has ~75–105 valid gene actions at every step.
- **Is start_distance = 8 enough?** Numerically, with mean contraction 1.0 per step and K = 10, the policy has roughly twice the budget it needs. The right knob is K, not min_start_distance alone.
- **Stricter epsilon (p25).** The right knob alongside K. p25 must be computed from the control distance distribution (P0A.3) before any RL re-eval; the value will be substantially smaller than 3.531.
- **Sample efficiency, not success rate.** Even at p50/K=10/start8 PPO already beats random on mean_steps (2.28 vs 5.53). This is a less-corrupted V2 metric and is primary alongside success.
- **Stronger baselines.** Random uniform-valid is far weaker than (a) **greedy-dynamics-1**: pick `argmin_a ‖f_θ(z, a) − z_ref‖`. (b) **ridge-greedy**: same with the ridge baseline as planner. (c) **mean-delta-greedy**: pick `argmin_a ‖z + mean_Δ_a − z_ref‖`. PPO ≈ greedy-dynamics is the right *upper-reference* comparison; PPO > greedy-dynamics is **bonus evidence** of multi-step planning, not a pass/fail requirement (greedy has oracle access to the learned dynamics that PPO had to estimate from experience).

### 4.3 RL reward and environment

- **Reward shape.** `R_t = −d_{t+1} − 0.05·𝟙[a_t ≠ NOOP]`. Both terms can be earned by a globally contractive policy without learning any gene-specific signal. There is no terminal bonus (`success_bonus = 0`), no truncation penalty, and intermediate shaping is on.
- **K = 10 is generous.** With contraction ~1.0 per step, K = 10 ≫ ⌈gap/contraction⌉. The honest budget is K = 1–3.
- **Start state stratification.** Currently none beyond the min_start_distance filter. For V2 we want stratified reporting over (start-distance bin × held-out-gene-of-origin), not a single aggregate.
- **NO-OP semantics.** Correct. *No change needed.*
- **Repeat mask.** Correct. *No change needed.*
- **Sample efficiency is unrewarded.** λ_sparse = 0.05 is too small to drive short trajectories.

### 4.4 VAE latent

- **32D vs 64D.** With the right flag, 64D is comparable to 32D on val Pearson and worse on OOD Pearson. The narrative "32D is optimal" is not fully supported; what's supported is "gene_bias is bad at 64D". A 32D-vs-64D re-ablation with state_linear-only and a *gate-margin-driven* loss is the honest comparison (P1).
- **Latent dim selection by reconstruction vs gate-margin.** scVI selects 32D for unsupervised quality. Whether 32D maximises the dynamics ridge-margin is a different question that has not been directly tested.
- **Latent label leakage risk.** SCANVI or any perturbation-label-informed latent would invalidate the OOD gene split's interpretation. Defer to P2 with a separate held-out gene split for VAE training, or skip entirely.

### 4.5 OT pairing

- **Noise as upper bound.** OT pseudo-pairing introduces an irreducible noise floor on Δz. Ridge averaging over residual noise smoothly = low-variance baseline; nonlinear MLP overfitting on residual noise = no Pearson gain. **This is a strong candidate explanation for the +0.0074 ceiling and must be probed even if a P0B fix passes the gate** — see P0A.4 (forensic) and P1 (corrective).
- **Falsifiable test.** Compare gate margin under `pairing.method = {ot, mean_delta, random}` with the same dynamics architecture. If margin under `mean_delta` ≥ +0.030 (because mean_delta zeroes out the within-perturbation noise) and margin under `ot` stays ~+0.007, the gate failure is pairing-noise-dominated. If both stay ~+0.007, it's architecture-dominated. *This is the single most informative diagnostic; we run a read-only version in P0A.4 and the retrain version in P1.*
- **Sinkhorn ε = 0.05.** Aggressive. Higher Sinkhorn ε would smear the matching across more perturbed cells, reducing noise but introducing systematic blur. Tune in P1.
- **Soft OT expectations.** Replace the argmax-per-column step with expectation over the row of the doubly-stochastic transport plan. Reduces matching noise without changing pair counts. P1.
- **Per-gene sample counts.** Genes with few perturbed cells get noisy pseudo-pairs. Cap or stratify in P1.

### 4.6 DepMap / biology

- **Top-20 hypergeometric is underpowered.** The 105-gene action universe vs 2000 HVG background means even a fully concordant top-20 has limited power.
- **Right test: ranked correlation.** Sort the 105 actions by PPO action frequency; correlate with Chronos using Spearman or rank-biserial. Uses the entire action universe and the entire effect distribution.
- **GSEA.** Preranked on (gene → freq weight) against (gene → −Chronos), within action universe and within MSigDB Hallmark/lineage panels. Permute over genes; BH-FDR.
- **CRISPRa vs CRISPRko honesty.** Norman is CRISPRa, DepMap is CRISPRko; the expected sign of correlation is non-trivial. *Carry caveat into V2 reporting.*
- **Action universe = 105 single-gene perturbations.** Anything beyond is overclaim until combinatorial action space is added (P2+).

## 5. Mathematical diagnosis

### 5.1 Dynamics loss vs gate metric

- **Current loss.** Per-element heteroscedastic NLL with weak log-var regularisation. Gradients on μ are `exp(−log_var)·(μ − Δ)`; gradients on log_var are `0.5·(1 − exp(−log_var)·(Δ−μ)²) + 0.02·log_var`. Composition loss (when active) is two-step MSE.
- **Gate metric.** Mean-per-dim Pearson of (Δz_pred, Δz_true) on held-out cells, minus the same on ridge predictions. Pearson is sensitive to sign agreement and scale-invariant per dimension.
- **Misalignment.** A model can achieve excellent NLL by predicting a small-magnitude μ with appropriate log_var on noisy dimensions; this can leave per-dim Pearson unchanged from ridge or slightly worse. There is no term in the loss that says "make `corr(μ_d, Δ_d)` large".
- **Repair candidates (to be tried in P0B, only after P0A inspection).**
  - (A) **Add a per-dim correlation loss**: `L_corr = mean_d (1 − corr(μ_d, Δ_d))`, computed per minibatch on the predicted-vs-true Δz, with **numerical safety baked in: ignore dimensions whose target variance in the minibatch is below 1e-6, clamp correlations to [−1+1e-7, 1−1e-7], log the term as a separate scalar in TensorBoard, and sweep `λ_corr ∈ {0.05, 0.10, 0.30}`**.
  - (B) **Residual-over-ridge architecture**: pre-fit ridge `R(z, gene)` on the train pairs once, **export coefficients and intercept as numpy arrays, then store them as `register_buffer` PyTorch tensors on the dynamics model (NOT as a stored sklearn object), so device/checkpointing/state-dict serialisation stays clean**. Predict `μ_total = ridge_pred(z, gene) + mlp_residual(z, gene_emb)`. MLP learns only what ridge cannot. Failure mode: residuals are mostly OT-pairing noise, but the diagnostic answer is clean.
  - (C) **Reduce/remove `state_linear_skip`** as an ablation alongside (B). *Use as an ablation, not as a primary fix.*

### 5.2 Reward vs non-trivial control

- **Current reward.** `R_t = −d_{t+1} − 0.05·𝟙[a_t ≠ NOOP]`. Maximised by any policy that contracts fast — including random over a contractive field.
- **Trivial-control criterion.** A reward is *trivially controllable* if `E[Σ R | π_random] / E[Σ R | π_*]` is close to 1.
- **Repair (to be tried in P0C, only after P0A and P0B).**
  - **Terminal-only success reward**: `R_terminal = 𝟙[terminal ∧ d < ε] − β·step_count`. No dense gradient on contraction. PPO must allocate K steps to reach ε.
  - **Δd shaped reward**: `R_t = d_t − d_{t+1} − λ_sparse·𝟙[a ≠ NOOP]`. Still rewards any contractive action but normalises across runs.
  - **Recommendation**: terminal-only + small step penalty first; Δd shaped as fallback if PPO does not learn at K ≤ 3.

### 5.3 Success threshold and start-state distribution

- **Current.** ε = p50 of control-cell distances; min_start_distance = 8.0; K = 10. Probability mass `P(success | π_random)` is large under contractive dynamics.
- **Repair (covered by P0A hard benchmark; can be applied to existing PPO without retraining).**
  - Use **p25** for primary success threshold.
  - **Stratify** start states by `d_0` bin and report per-bin.
  - **Reduce K** to 1, 2, 3 as primary; K = 8 / 10 as secondary.
  - **Held-out gene start cells**: evaluate only on cells whose source perturbation is in the OOD gene split.

### 5.4 Pairing and predictability

- Let `Δ_true(z, g) = E[z_pert − z_ctrl | gene = g, state = z]` be the population-level conditional mean. The MLP cannot do better than predicting `Δ_true(z, g)`; everything beyond is residual matching noise from OT. The MLP–ridge Pearson margin ≈ `(var(Δ_true_nonlin)) / (var(Δ_true) + var(OT_noise))`. If most gene-induced variation is well-described as a per-gene additive offset linear in z, ridge wins or ties.
- Repair via Soft OT / multi-pairing in P1.

## 6. V2 hypotheses (ranked, all falsifiable)

**H1 (highest prior, addressed by P0B).** *The dynamics gate fails because the loss does not directly optimise per-dim correlation, and the linear-skip architecture mechanically aligns the MLP with the ridge baseline. Adding a numerically safe correlation loss term and/or making the MLP a residual-over-ridge predictor will close the +0.0074 margin to ≥ +0.030 without sacrificing OOD Pearson or uncertainty calibration.*
- **Test:** retrain dynamics with `λ_corr ∈ {0.05, 0.10, 0.30}`; with `use_residual_over_ridge=true`; and with both.
- **Falsification:** if no `λ_corr` setting and no residual-over-ridge config exceeds margin +0.020, fall through to H2 (pairing-noise upper bound).

**H2 (medium prior, addressed by P1).** *The gate failure is bounded above by OT pseudo-pairing noise. Switching to `mean_delta` pairing or soft-OT expectation reduces target variance and lifts the achievable margin to ≥ +0.030.*
- **Test:** rebuild pairs with `pairing.method = mean_delta`; retrain dynamics under best P0B variant.
- **Falsification:** if `mean_delta` margin stays at +0.007–+0.012, H1 alone explains the gap. *Even if H1 succeeds in P0B, run H2 in P1 — pairing-noise scoping is independently important for scientific credibility.*

**H3 (high prior, P0A immediately verifiable, P0C definitive).** *The narrow PPO–random gap is caused by (a) globally contractive dynamics, (b) dense `−d_next` reward, and (c) overgenerous K = 10. Evaluating the existing PPO under V2 hard benchmark (K = 3, ε = p25, OOD gene split) will already widen the gap; retraining with terminal-only reward in P0C will widen it further.*
- **Test:** P0A hard benchmark on V1 PPO; if needed, P0C retraining.
- **Acceptance:** under primary cell (K = 3, ε = p25, distance bin 8–10, OOD gene split): random ≤ 0.40; PPO ≥ 0.70; PPO − random ≥ +25 pp; PPO mean_steps ≤ 2.0; **PPO approaches greedy-dynamics-1 within ±0.10 success**, with PPO > greedy treated as bonus evidence rather than a pass/fail.

**H4 (medium prior, P0A).** *PPO action selection at the population level correlates with biological essentiality more strongly than random, but the V1 top-20 hypergeometric is underpowered to detect this. Full-action-universe Spearman correlation between PPO action frequency and Chronos score, plus preranked GSEA over Hallmark/lineage panels with BH-FDR, will yield at least one panel at q ≤ 0.10 — or it will not, and we will report the negative result with the right test attached.*
- **Test:** new metric on existing V1 rollouts (read-only).
- **Acceptance:** new test is run; outcome reported honestly either way.

**H5 (P0A).** *The dynamics field's contractivity is not uniform across genes — a meaningful subset of genes contracts more strongly and consistently than others. PPO's top actions concentrate on this subset. Random is strong because most genes are at least mildly contractive, but PPO's edge is real (it picks the strongly-contractive subset).*
- **Test:** per-gene action-contraction diagnostic (§7.A.2): for each gene g, compute `improvement(z, g) = d(z, z_ref) − d(f(z, g), z_ref)` over a large random sample of starts; report mean, std, fraction positive, top/bottom genes, and Gini/entropy of mean-improvement distribution; check overlap with PPO top actions.
- **Falsification:** if Gini coefficient < 0.10 (i.e. all genes are roughly equally contractive), PPO has very little to learn from gene choice, and V2 cannot honestly claim "PPO learned biologically prioritised actions" — only "PPO learned to spam any non-NOOP action". This would be a critical scientific finding.

**H6 (low prior, P2).** *An ensemble of 3 dynamics models with different seeds gives substantially better uncertainty estimates and tighter gate margins than the single heteroscedastic head.*
- **Test:** train 3 seeds; report mean and disagreement-based uncertainty.
- **Falsification:** ensemble margin gain < +0.005.

## 7. Phased implementation plan

V2 implementation proceeds in three sequential phases. **Each phase concludes with a review checkpoint; do not begin the next phase until prior results have been inspected and the user has approved.**

### 7.A — P0A: Forensic benchmark (read-only, no retraining)

Goal: clean, hard, baseline-rich benchmark of the V1 pipeline before any model change. Output: a results matrix and a set of diagnostics that *justify* what comes next. **No retraining of VAE, dynamics, or PPO in this phase. No reward changes. No pairing changes.**

#### P0A.1 — Gate breakdown diagnostic (tiny)

**Files (NEW, read-only utilities):**
- `src/analysis/gate_breakdown.py`: re-load the V1 dynamics predictions on val and OOD pairs; recompute per-dim Pearson, per-gene Pearson, and (MLP − ridge) margins per dim and per gene. Use the existing `gate_diagnostics.json` where possible; recompute from scratch where it is missing. Write `artifacts_v2/diagnostics/per_dim_margin.csv` and `per_gene_margin.csv`.

**Commands:**
```bash
PYTHONPATH=. python -m src.analysis.gate_breakdown \
  --dynamics_dir artifacts/dynamics \
  --pairs_dir artifacts/pairs \
  --out artifacts_v2/diagnostics
```

**Acceptance:** files exist; top-5 worst per-dim and worst per-gene margins identified.

**Artifact:** `artifacts_v2/diagnostics/per_dim_margin.csv`, `per_gene_margin.csv`.

#### P0A.2 — Per-gene action-contraction diagnostic (tiny)

**Goal:** quantify whether the contractive field is uniform across genes or concentrated in a subset. This is the answer to "does PPO have real action structure to learn".

**Files (NEW):**
- `scripts/diagnose_action_contraction.py`: for each gene `g` in the action universe, sample `n_starts` (default 500) latent states uniformly from the V1 perturbed-cell pool (or the V1 RL start_pool with `min_start_distance=0`), compute `improvement(z, g) = d(z, z_ref) − d(f_θ(z, g), z_ref)`; aggregate per gene to a row of (mean, std, fraction_positive, n_starts). Then aggregate across genes: top-10 most contractive, bottom-10 least contractive (or expanding), Gini coefficient of the mean-improvement distribution, Shannon entropy of |mean| normalised distribution, overlap between top-N most contractive and PPO top-N actions. Write `artifacts_v2/diagnostics/per_gene_contraction.csv` (per-gene rows) and `per_gene_contraction_summary.json` (aggregate).
- Use existing `f_θ` via the standard `PerturbationDynamicsModel.predict(z, gene_idx)` API; do not modify the model.

**Commands:**
```bash
PYTHONPATH=. python scripts/diagnose_action_contraction.py \
  --dynamics_dir artifacts/dynamics \
  --vae_dir artifacts/vae \
  --n_starts 500 \
  --out artifacts_v2/diagnostics
```

**Expected runtime:** tiny (~3–5 min CPU).

**Acceptance:** per-gene CSV and summary written; top/bottom genes printed; Gini and entropy reported.

**Interpretation guide (do not implement as gating thresholds — these are reading rules for the user):**
- Gini > 0.30 and entropy < 0.6 × log(|G|): clear action-importance heterogeneity → PPO has signal to learn; H5 supported.
- Gini < 0.10 and entropy > 0.9 × log(|G|): contraction is uniform across genes → PPO's gene choice is largely irrelevant; H5 rejected; V2 narrative must shift to "this is not a biological-prioritisation task".

#### P0A.3 — V2 hard RL benchmark (small, no training; smoke matrix first)

**Goal:** rerun the V1 PPO and a suite of baselines under the V2 hard evaluation matrix (§9), starting with a tiny smoke matrix to validate plumbing.

**Files (NEW):**
- `src/rl/baselines.py`: `RandomUniformValidPolicy`, `AlwaysNoopPolicy`, `GreedyDynamicsPolicy(dynamics, n_step=1)`, `RidgeGreedyPolicy(ridge_buffers)`, `MeanDeltaGreedyPolicy(mean_delta_table)`. Each implements `select_action(z, mask, info) -> int` and is composable with the existing env. The ridge baseline must load coefficients via `np.load`/`register_buffer` consistent with what P0B will later do for the dynamics model.
- `scripts/evaluate_rl_hard.py`: driver. Loads V1 dynamics + VAE + epsilon. Composes env with `epsilon_override`, `max_steps`, and optionally `held_out_genes_only=true` (new env reset filter — implemented as a read-side filter on the start pool only, no env-class change). For each (K, ε, distance bin, gene split, policy), runs `n_episodes` and writes `artifacts_v2/eval_hard_v1policy/<config>/summary.json`. Driver supports both a tiny `--smoke` mode (a single primary cell, n_episodes=20) and a full matrix.
- `config/experiments/rl_hard_eval.yaml`: Hydra config parameterising the matrix.
- `src/utils/epsilon_percentile.py`: helper that reads `artifacts/vae/latents.h5ad` and computes a chosen percentile.

**Step 1: smoke matrix (tiny, mandatory first).**

Run only the primary cell at K=3, ε=p25, distance bin 8–10, OOD=true, n_episodes=20, for *all* baselines + PPO det. Confirm the output schema matches `summary.json` expectations, the env composes correctly, and no baseline silently errors out.

```bash
PYTHONPATH=. python -m src.utils.epsilon_percentile --vae_dir artifacts/vae --p 25 \
  > artifacts_v2/epsilon_p25.txt
PYTHONPATH=. python scripts/evaluate_rl_hard.py --smoke \
  --vae_dir artifacts/vae \
  --dynamics_dir artifacts/dynamics \
  --ppo_zip artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip \
  --out_dir artifacts_v2/eval_hard_v1policy_smoke \
  --k_values 3 \
  --epsilon_values p25 \
  --distance_bins 8-10 \
  --held_out_genes_only true \
  --n_episodes 20 \
  --baselines random,always_noop,greedy_dyn_1,ridge_greedy,mean_delta_greedy
```

**Expected runtime:** tiny (~3–8 min).

**Smoke acceptance:** a single `summary.json` per policy at the primary cell; success_rate is in [0, 1]; mean_steps is in [0, 3]; no exceptions; PPO and at least one baseline both produced non-trivial output. **If smoke fails, stop and debug; do not run the full matrix.**

**Step 2: full matrix (only after smoke passes).**

```bash
PYTHONPATH=. python scripts/evaluate_rl_hard.py \
  --vae_dir artifacts/vae \
  --dynamics_dir artifacts/dynamics \
  --ppo_zip artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip \
  --out_dir artifacts_v2/eval_hard_v1policy \
  --k_values 1 2 3 8 \
  --epsilon_values p25 p50 \
  --distance_bins 4-6 6-8 8-10 10-12 \
  --held_out_genes_only true,false \
  --n_episodes 500 \
  --baselines random,always_noop,greedy_dyn_1,ridge_greedy,mean_delta_greedy
```

**Expected runtime:** small-to-medium (matrix ≈ 224k rollouts; 2–4 h on CPU; parallelise by Hydra multirun).

**Acceptance:** matrix populated; per-cell summary.json written; aggregate `results_table.md` produced.

**Rollback / honesty:** if V1 PPO ≤ random in the primary cell (K=3, ε=p25, bin 8–10, OOD), V2 must report this. *V1 success was an artifact of easy eval.* That outcome is acceptable and publishable as a methodological correction.

#### P0A.4 — Pairing-noise scoping (tiny, no retraining)

**Goal:** establish the OT-pairing noise floor that bounds the achievable gate margin. Even if H1 closes the gate in P0B, this measurement is required for the scientific story.

**Files (NEW):**
- `scripts/diagnose_pairing_noise.py`: for each gene `g`, compute (i) within-perturbation variance of Δz under current OT pairs, (ii) variance of Δz around the per-gene mean (i.e. what `mean_delta` would target), (iii) the ratio (irreducible-noise-floor / total-Δ-variance). Write `artifacts_v2/diagnostics/pairing_noise.json` and a markdown summary.

**Commands:**
```bash
PYTHONPATH=. python scripts/diagnose_pairing_noise.py \
  --pairs_dir artifacts/pairs \
  --out artifacts_v2/diagnostics/pairing_noise.json
```

**Expected runtime:** tiny.

**Acceptance:** json + md exist; ratio per gene reported; estimate of "ceiling" for what any dynamics model can predict on top of ridge written into the summary.

#### P0A.5 — Biological eval upgrade (tiny, no training; new V2 script)

**Goal:** replace top-20 hypergeometric as the primary biology test with full-action-universe ranked correlation + preranked GSEA. **Do NOT modify `scripts/aggregate_eval.py`** — V1 reporting must remain bit-stable. Write a new V2-specific script instead.

**Files (NEW / MODIFIED):**
- `src/analysis/metrics.py`: NEW `action_freq_chronos_spearman(action_freq: dict, chronos: pd.Series, seed: int = 42, n_boot: int = 10_000) -> dict`. Returns Spearman ρ, two-sided p, bootstrap 95 % CI, n_overlap. Docstring includes math.
- `src/analysis/depmap_validation.py`: NEW `preranked_gsea(action_freq, chronos, panel_sets, n_perm=10_000, seed=42)`. Permute over genes for null; Benjamini-Hochberg across panels.
- `scripts/evaluate_biology_v2.py` (NEW; does **not** import from or modify `scripts/aggregate_eval.py`): reads V1 rollout action_freq JSONs (PPO det, PPO stoch, random) and DepMap Chronos, computes (B1) and (B2), writes `artifacts_v2/eval/depmap_v2.csv` and `depmap_v2_summary.json`. Adds its own `metadata.json` recording git commit, config, and source paths.

**Commands:**
```bash
PYTHONPATH=. python scripts/evaluate_biology_v2.py \
  --rl_dir artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval \
  --action_freq_ppo_det artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic/action_freq.json \
  --action_freq_random artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/random_baseline/action_freq.json \
  --depmap_csv data/processed/depmap_k562_chronos.parquet \
  --out artifacts_v2/eval/depmap_v2
```

**Expected runtime:** tiny.

**Acceptance:** v2 biology table + summary written; significant or not, reported honestly; V1 outputs untouched.

#### P0A — Review checkpoint

Read all P0A artifacts and confirm:

- Where the gate failure concentrates (dim-level, gene-level).
- Whether the dynamics field is uniformly contractive across genes (H5).
- The V2-hard-benchmark behaviour of the *existing* V1 PPO and all baselines.
- The OT-pairing-noise ceiling.
- The new biology test result on existing rollouts.

*Only after this review do we proceed to P0B.*

### 7.B — P0B: Dynamics gate fix (small, retraining; only after P0A review)

**Goal:** close the dynamics gate using the loss-and-architecture levers identified by H1, informed by the P0A breakdown.

#### P0B.1 — Correlation loss (numerically safe) + λ_corr sweep

**Files (MODIFIED / NEW):**
- `src/analysis/metrics.py`: NEW `correlation_loss(mu, target, eps=1e-7, min_var=1e-6)` — for each latent dim, skip dims whose `target.var(dim=0) < min_var`; for the rest compute `1 − corr_d`, clamp `corr_d ∈ [−1+eps, 1−eps]`; return `mean_d(1 − corr_d)` over kept dims. Unit-test on synthetic batches: zero when inputs are perfectly correlated; one when uncorrelated; numerically finite at boundary.
- `src/models/dynamics.py`: no change.
- `scripts/train_dynamics.py`: read `cfg.dynamics.lambda_corr`; if > 0, add `lambda_corr * correlation_loss(mu, target_delta)` to the loss. Log `loss_corr` separately in TensorBoard; record in `metadata.json`.
- `config/dynamics.yaml`: add `lambda_corr: 0.0` (default off).
- `config/experiments/dynamics_v2_corr_005.yaml`, `…_010.yaml`, `…_030.yaml`: three experiment configs sweeping `λ_corr ∈ {0.05, 0.10, 0.30}` keeping `use_state_linear_skip: true`.

#### P0B.2 — Residual-over-ridge architecture (buffer-based)

**Files (MODIFIED / NEW):**
- `src/models/dynamics.py`: add `use_residual_over_ridge: bool = False` config flag. When `True`, the model stores three `register_buffer`s: `ridge_W_z: (n_latent, n_latent)`, `ridge_W_gene: (n_genes, n_latent)`, `ridge_b: (n_latent,)`. Forward pass: `ridge_pred = z @ W_z + W_gene[gene_idx] + b`. Final `μ = ridge_pred + mlp_residual(z, gene_emb)`. The `state_linear_skip` and `use_residual_over_ridge` flags are *mutually exclusive*; configs that set both raise `ValueError`. **No sklearn objects are stored on the model.**
- `src/models/dynamics.py` (helper): `fit_ridge_baseline_from_pairs(train_pairs) -> (W_z, W_gene, b)` — calls `_fit_ridge_baseline` from `src/analysis/metrics.py` (single source of truth, CLAUDE.md rule 4), extracts `.coef_` and `.intercept_`, splits the coefficient block into state-vs-gene halves, returns numpy arrays. Caller is `scripts/train_dynamics.py` which converts to torch tensors and assigns to model buffers before training starts.
- `scripts/train_dynamics.py`: when `use_residual_over_ridge=true`, fit-or-load the ridge baseline at start, write `ridge_baseline.npz` (W_z, W_gene, b, n_genes) into the run dir, and assign to the model's buffers. Buffers are saved as part of the model state_dict naturally — no extra serialisation logic.
- `config/experiments/dynamics_v2_ror.yaml`: `use_residual_over_ridge: true`, `use_state_linear_skip: false`, `lambda_corr: 0.0`.
- `config/experiments/dynamics_v2_both.yaml`: `use_residual_over_ridge: true`, `lambda_corr: 0.10` (best of P0B.1 if known, else 0.10).

**Commands (after P0A review):**
```bash
PYTHONPATH=. python scripts/train_dynamics.py --config-name dynamics_v2_corr_005 \
  paths.dynamics_dir=artifacts_v2/dynamics_v2_corr_005
PYTHONPATH=. python scripts/train_dynamics.py --config-name dynamics_v2_corr_010 \
  paths.dynamics_dir=artifacts_v2/dynamics_v2_corr_010
PYTHONPATH=. python scripts/train_dynamics.py --config-name dynamics_v2_corr_030 \
  paths.dynamics_dir=artifacts_v2/dynamics_v2_corr_030
PYTHONPATH=. python scripts/train_dynamics.py --config-name dynamics_v2_ror \
  paths.dynamics_dir=artifacts_v2/dynamics_v2_ror
PYTHONPATH=. python scripts/train_dynamics.py --config-name dynamics_v2_both \
  paths.dynamics_dir=artifacts_v2/dynamics_v2_both
```

**Expected runtime:** small (5 × ~30 min on Apple Silicon; or 5 × ~10 min on CUDA).

**Acceptance:** at least one variant produces `gate.json.passed=true` with `margin_vs_linear_ridge_pearson ≥ 0.030`, OOD Pearson ≥ 0.40, uncertainty Spearman ≥ 0.20.

**Rollback:** if no variant passes, advance to P1 pairing-noise corrective (soft-OT / mean_delta retraining) before exploring further architectural changes.

**Artifact:** `artifacts_v2/dynamics_v2_*/gate.json` with `passed=true`.

#### P0B — Review checkpoint

Inspect gate.json across all P0B variants; pick the best on (ridge-margin, OOD Pearson, unc Spearman). Re-run V2 hard benchmark from P0A.3 on the best V2 dynamics with the *existing* V1 PPO. *Only then proceed to P0C.*

### 7.C — P0C: Reward / RL retraining (medium; only after P0B review)

**Goal:** retrain PPO under V2 reward modes on the V2 dynamics; re-evaluate under V2 hard benchmark; compare to baselines.

#### P0C.1 — Reward modes

**Files (MODIFIED):**
- `src/rl/reward.py`: add reward modes `delta` (`R_t = d_t − d_{t+1} − λ_sparse·𝟙[a ≠ NOOP]`) and `terminal_only_step_cost` (`R_t = 0` for non-terminal, `R_T = 𝟙[d < ε] − β·step_count` at terminal). Existing `absolute_distance` remains the default for back-compat.
- `config/rl.yaml`: add `reward_mode` and `beta_step_cost`; default `reward_mode=absolute_distance`, `beta_step_cost=0.05`.

#### P0C.2 — PPO retraining

**Files (MODIFIED):**
- `scripts/train_rl.py`: must continue to check `dynamics_gate_passed` (sacred rule 9). If P0B did not pass, P0C is blocked.

**Configs:**
- `config/experiments/rl_v2_terminal.yaml`: `reward_mode=terminal_only_step_cost`, `max_steps=3`, uses best V2 dynamics.
- `config/experiments/rl_v2_delta.yaml`: `reward_mode=delta`, `max_steps=3`, uses best V2 dynamics.

**Commands:**
```bash
PYTHONPATH=. python scripts/train_rl.py --config-name rl_v2_terminal \
  paths.dynamics_dir=artifacts_v2/dynamics_v2_<best> \
  paths.rl_dir=artifacts_v2/rl_v2_terminal
PYTHONPATH=. python scripts/train_rl.py --config-name rl_v2_delta \
  paths.dynamics_dir=artifacts_v2/dynamics_v2_<best> \
  paths.rl_dir=artifacts_v2/rl_v2_delta
```

**Expected runtime:** medium (2 × 2M timesteps ≈ 2 × 4–8 h on CPU; faster on GPU).

#### P0C.3 — V2 hard benchmark re-eval

Re-run `scripts/evaluate_rl_hard.py` from P0A.3 on the V2 PPO + V2 dynamics; write `artifacts_v2/eval_hard_v2policy_v2dyn/`.

**Acceptance:** primary cell (K=3, ε=p25, bin 8–10, OOD): PPO det success ≥ 0.70, random ≤ 0.40, PPO − random ≥ +25 pp, PPO mean_steps ≤ 2.0, PPO within ±0.10 of greedy-dynamics-1 (or better).

## 8. P1 / P2 experiment plan

### P1 (run after P0C completes; strong next experiments)

- **P1.1 — Pairing alternatives (preserves H2 scoping even if P0B passed).** Rebuild with `pairing.method=mean_delta` (variance baseline). Retrain best V2 dynamics. Compare gate margin to P0B baseline. Then implement Soft-OT (expectation over transport-plan rows) and retrain. Diagnostic and corrective.
- **P1.2 — 32D vs 64D under V2 dynamics loss.** Retrain best P0B variant on 64D; pick the better of {32D V2, 64D V2}.
- **P1.3 — Stratified-distance, held-out-gene start states in the env.** Augment env reset to draw stratified start states (currently the bins are imposed *eval-side* via filtering; making them training-side requires env changes).
- **P1.4 — Ensemble dynamics (3 seeds) + uncertainty-aware PPO.** Train 3 seeds; report disagreement; use it in `lambda_unc > 0`.
- **P1.5 — Per-gene Pearson loss weighting.** Use P0A.1 to upweight bottom-decile genes.

### P2 (stretch)

- **P2.1 — FiLM conditioning.** `γ(gene_emb) ⊙ z + β(gene_emb)` instead of concat.
- **P2.2 — Low-rank gene embeddings.** Constrain to rank ≤ 16.
- **P2.3 — Contractive-action regulariser.** Penalise gene-independent contraction during dynamics training.
- **P2.4 — Latent dim ablation (16, 24, 32, 48, 64) under V2 loss.**
- **P2.5 — Combinatorial action space.** Defer to V3.
- **P2.6 — External healthy reference.** Defer to V3. Sacred-rule 7 caution.
- **P2.7 — CRISPRi support.** Defer. Sacred-rule 10.

## 9. New evaluation protocol (V2 hard benchmark)

This is the V2 RL evaluation contract.

**Axes:**
- `K ∈ {1, 2, 3, 8}`. Primary K = 3. Secondary K = 1, 2 (one-shot, two-shot). Tertiary K = 8 (matches V1).
- `ε ∈ {p25, p50}`. Primary p25. Secondary p50.
- `distance bin ∈ {4–6, 6–8, 8–10, 10–12}`. Primary 8–10. Secondary all bins.
- `gene split ∈ {seen, ood, mixed}`. Primary OOD. Secondary seen. Tertiary mixed.

**Baselines (must include all):**
- `random_uniform_valid`: V1 random.
- `always_noop`: sanity check.
- `greedy_dynamics_1`: `argmin_a ‖f_θ(z, a) − z_ref‖`. *Upper-reference baseline with oracle access to the dynamics.*
- `ridge_greedy`: same with ridge as planner.
- `mean_delta_greedy`: same with per-gene mean Δ as planner.
- `ppo_deterministic` and `ppo_stochastic`: V2 trained policy.

**Reporting (per (K, ε, bin, split) cell):**
- success_rate ± Wilson 95 % CI
- mean_steps ± SE
- mean_final_distance
- weighted_action_freq_chronos_spearman + p
- top-10 actions
- PPO − random delta (pp)
- PPO − greedy_dynamics_1 delta (pp)

**Primary V2 headline (single cell):** K = 3, ε = p25, bin = 8–10, split = ood.

**V2 acceptance thresholds (direction-of-improvement; reconcile against empirical distributions once P0A runs):**
- random success ≤ 0.40 in primary cell.
- PPO det success ≥ 0.70 in primary cell.
- PPO − random ≥ +25 pp; aspirational ≥ +40 pp.
- **PPO within ±0.10 of greedy_dynamics_1 in primary cell**. If PPO > greedy, that is **bonus** evidence of multi-step planning; not required for V2 to be a "pass". If PPO < greedy by > 0.10, document it: PPO did not match a one-step oracle and may need more training or a different reward.
- PPO mean_steps ≤ 2.0 (out of K = 3).
- Dynamics uncertainty Spearman ≥ 0.20 on OOD pairs.

**V2 unacceptable outcomes (report honestly):**
- PPO success > random by < 10 pp in primary cell.
- PPO < random in any cell.
- Random > 0.60 in primary cell (task is still too easy; restructure the benchmark).

## 10. Biological validation plan

Ranked tests; primary is (B1).

(B1) **Weighted-action-freq vs Chronos Spearman (whole action universe).** Across the 105-gene action universe, rank by PPO action frequency vs Chronos. Bootstrap 95 % CI (n=10k). Compare to random on the same metric.
- *Evidence threshold:* Spearman p < 0.05 with effect sign consistent with the CRISPRa-vs-CRISPRko expectation. Do not invert sign to chase significance.

(B2) **Preranked GSEA over Hallmark / lineage panels.** Weighted action freq vs −Chronos; permutation p with BH-FDR.
- *Evidence threshold:* at least one panel q ≤ 0.10.

(B3) **Top-K hypergeometric (legacy V1).** Keep for back-compat; not primary.

(B4) **Gene functional clustering of top PPO actions.** Submit top-20 to STRING / Hallmark; qualitative.

(B5) **CRISPRa-vs-CRISPRko sign discussion.** Explicit caveat section.

(B6) **Negative evidence is evidence.** If (B1) and (B2) are p > 0.05 / q > 0.10, report "no detectable enrichment beyond expectation". The thesis-defense talking point becomes "the surrogate produces a learnable steering policy but does not learn biologically prioritised genes beyond chance", which is honest and publishable.

**Overclaim guardrails:**
- No "cancer reprogramming" or "therapeutic discovery".
- Top genes are *computational selections from a CRISPRa action space*, not drug targets.
- No 32D-vs-64D rolling narratives without the matrix.
- Do not silently lower gate thresholds.

## 11. Risks and anti-goals

**Risks:**
- (R1) `λ_corr` may destabilise NLL or hurt uncertainty calibration. **Mitigation:** clamp + min_var filter + separate logging; sweep three values.
- (R2) Terminal-only reward may not give PPO enough signal at K=3. **Mitigation:** small terminal proximity reward `−γ·d_final` (P0C.1 already includes step penalty β) keeps gradient non-zero without rewarding contraction at every step.
- (R3) p25 epsilon may make the task structurally infeasible in some bins. **Mitigation:** report per-bin so the reachable region is identified.
- (R4) Replacing OT pairing may inflate gate margins while degrading downstream behavioural diversity. **Mitigation:** P1.1 rebuilds and reruns the full RL benchmark — gate is necessary but not sufficient.
- (R5) Subagent-driven V2 implementation may regress V1 artifacts. **Mitigation:** V2 writes only to `artifacts_v2/`; V1 immutable; new biology script is a separate file from `scripts/aggregate_eval.py`.
- (R6) Implementation may bake sklearn objects into the dynamics model. **Mitigation:** ridge coefficients live as `register_buffer` numpy-derived tensors; no sklearn import in the model module beyond the helper that produces buffers at construction time.

**Anti-goals:**
- Do not lower the V1 gate thresholds (`config/dynamics.yaml::dynamics.gate.thresholds`). Improve the model.
- Do not weaken random baseline by removing repeat-mask or forcing NO-OP-first. Make the *environment* harder; never make random *broken*.
- Do not use ad-hoc fudges to the success threshold. p25 means p25.
- Do not require PPO to beat greedy-dynamics as a pass/fail gate. PPO ≥ random by large margin is the gate; PPO ≥ greedy is bonus.
- Do not delete or overwrite anything in `artifacts/`, `artifacts_64/`, or `artifacts/rl_sweeps/`.
- Do not modify `scripts/aggregate_eval.py`. V2 biology runs through `scripts/evaluate_biology_v2.py`.
- Do not retrain the VAE without a documented hypothesis and a separate ablation directory.
- Do not enable `use_knockout`. Sacred rule 10.
- No "normal", "healthy", "non-leukemic", "therapeutic", or "drug target" language. Sacred rule 7.
- Do not modify shared interfaces (`src/utils/device.py`, `src/utils/seeding.py`, `src/analysis/metrics.py`, `config/paths.yaml`, AGENTS.md interface contracts) without updating ARCHITECTURE.md §2 and AGENTS.md "Interface Contract" in the same commit.
- Do not pursue residual-over-ridge or correlation loss before P0A has been run and inspected.
- Do not skip pairing-noise scoping. If P0B passes the gate, run pairing-noise scoping anyway in P1 — the scientific story requires knowing where the OT ceiling is.
- Do not run the full V2 hard-benchmark matrix before the smoke matrix in P0A.3 has passed.

## 12. Final recommended next prompt for implementation (P0A only)

> **CellPath V2 — P0A forensic benchmark only.**
>
> You are working in the local repo `/Users/gabo/Developer/ITAM/IA/cellpath` on a fresh worktree off `gabo2`. Read `V2_RESEARCH_PLAN.md` end-to-end; in particular, §7.A (P0A), §9 (V2 hard benchmark), §10 (biological validation), and §11 (anti-goals). Then implement **only** P0A.1 through P0A.5. Do NOT modify the dynamics model, the VAE, the reward, or the environment class. Do NOT retrain anything. Do NOT begin P0B or P0C.
>
> All V2 outputs go under `artifacts_v2/`. All new configs go under `config/experiments/`. All new metrics go in `src/analysis/metrics.py` (or `src/analysis/depmap_validation.py` for biology) with full docstring math, per CLAUDE.md rule 4. Do NOT modify `scripts/aggregate_eval.py` — the V2 biology pipeline lives in a new script `scripts/evaluate_biology_v2.py`. Every new utility / baseline / script must have a unit test in `tests/` written *before* the implementation (TDD). Every script must accept `--config-name` (Hydra) and write a `metadata.json` containing the full Hydra config, git commit hash, and read-only-mode flag.
>
> Concrete task order:
>
> 1. **P0A.1 — Gate breakdown.** Write `src/analysis/gate_breakdown.py` (re-loads V1 dynamics on val + OOD pairs, recomputes per-dim and per-gene MLP-vs-ridge margins). Add tests against a small mock pair set. Run on `artifacts/dynamics` and `artifacts/pairs`; write `artifacts_v2/diagnostics/per_dim_margin.csv` and `per_gene_margin.csv`. **Stop and report top-5 worst per-dim and per-gene margins.**
> 2. **P0A.2 — Per-gene action-contraction diagnostic.** Write `scripts/diagnose_action_contraction.py`. For each gene in the action universe, sample 500 starts from the V1 perturbed-cell pool, compute `improvement(z, g) = d(z, z_ref) − d(f_θ(z, g), z_ref)`, aggregate per gene (mean, std, fraction_positive, n_starts), aggregate across genes (top-10 most/bottom-10 least contractive, Gini coefficient of mean-improvement distribution, Shannon entropy of |mean-improvement| normalised distribution, overlap with PPO top-10 actions from `artifacts/rl_sweeps/.../action_freq.json`). Add a `metrics.py` helper for Gini and entropy if not present (with docstring math). Write `artifacts_v2/diagnostics/per_gene_contraction.csv` and `per_gene_contraction_summary.json`. **Stop and report Gini, entropy, top/bottom-10 genes, and PPO-top-10 overlap.**
> 3. **P0A.3 — V2 hard benchmark (smoke first, then full).**
>    (a) Write `src/utils/epsilon_percentile.py` and run for `p=25`; record value in `artifacts_v2/epsilon_p25.txt`.
>    (b) Write `src/rl/baselines.py` with `RandomUniformValidPolicy`, `AlwaysNoopPolicy`, `GreedyDynamicsPolicy(dynamics, n_step=1)`, `RidgeGreedyPolicy(ridge_buffers)`, `MeanDeltaGreedyPolicy(mean_delta_table)`. Each implements `select_action(z, mask, info) -> int`. Add tests for each on a 2D mock env. The ridge baseline must load coefficients via `np.load` and hold them as numpy arrays (or torch buffers if used in a torch context). Do NOT store sklearn objects inside any policy.
>    (c) Write `scripts/evaluate_rl_hard.py`. The script composes the existing `CellReprogrammingEnv` with `epsilon_override`, `max_steps`, and an optional `held_out_genes_only=true` filter applied *only on the start pool* (no env-class modification). For each (K, ε, distance bin, gene split, policy), run `n_episodes` and write per-cell `summary.json` with success_rate (+ Wilson CI), mean_steps, mean_final_d, weighted_action_freq_chronos_spearman + p, top-10 actions, PPO−random delta, PPO−greedy_dyn_1 delta. Aggregate into `results_table.md`. The script must support a `--smoke` flag that overrides the matrix to a single primary cell (K=3, ε=p25, bin=8–10, OOD=true, n_episodes=20). Add a unit test that the smoke run completes end-to-end and produces a valid summary.json for each policy.
>    (d) **Run the smoke matrix first** (see §7.A P0A.3 Step 1). Confirm output schema, env composition, no silent baseline errors. **Stop and report the smoke cell numbers.**
>    (e) Only if smoke passes: run the full matrix on the V1 PPO. **Stop and report the primary cell (K=3, ε=p25, bin 8–10, OOD).**
> 4. **P0A.4 — Pairing-noise scoping.** Write `scripts/diagnose_pairing_noise.py`. For each gene, compute (i) within-perturbation variance of Δz under current OT, (ii) variance around per-gene mean (mean_delta target), (iii) ratio (irreducible noise / total variance). Write `artifacts_v2/diagnostics/pairing_noise.json` plus a `pairing_noise.md` narrative summary. **Stop and report the median noise-ceiling ratio across genes.**
> 5. **P0A.5 — Biology rerank on existing rollouts (new V2 script).** Add `action_freq_chronos_spearman` to `src/analysis/metrics.py` and `preranked_gsea` to `src/analysis/depmap_validation.py` (both with docstring math, both unit-tested with synthetic Chronos vectors). Write a **new** `scripts/evaluate_biology_v2.py` that reads V1 rollout action_freq JSONs (PPO det, PPO stoch, random) and DepMap Chronos, computes (B1) and (B2), writes `artifacts_v2/eval/depmap_v2.csv` and `depmap_v2_summary.json`. **Do not modify `scripts/aggregate_eval.py`.** **Stop and report (B1) Spearman + p and (B2) most-enriched panels at q ≤ 0.10 (or explicit negative result).**
>
> After each P0A.x step, commit. Use commit messages of the form `v2/p0a.<n> <what>`. Open a PR titled `V2 P0A — forensic benchmark` against `gabo2` (do NOT push to `main`). At the end of P0A, write a summary file `artifacts_v2/p0a_summary.md` with: gate-failure concentration, contraction-heterogeneity verdict (H5 supported / rejected), V2-hard-benchmark primary-cell numbers for V1 PPO + all baselines, pairing-noise ceiling, and biology rerank result. **Then stop — do not begin P0B.**
>
> Constraints:
> - Do not delete or modify any V1 file in `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/`.
> - Do not modify `src/models/dynamics.py`, `src/models/vae.py`, `src/rl/environment.py`, `src/rl/reward.py`, `src/rl/train_ppo.py`, `scripts/train_dynamics.py`, `scripts/train_rl.py`, `scripts/train_vae.py`, `scripts/aggregate_eval.py`. P0A is read-only with respect to model, training, and V1 reporting code; only NEW files and read-side hooks are permitted.
> - Do not lower any gate threshold in `config/dynamics.yaml`.
> - Do not claim therapeutic discovery; honour sacred rule 7 in every docstring you write.
> - Do not call `torch.device()` outside `src/utils/device.py`. Do not call random seeds outside `src/utils/seeding.py`. Do not redefine any metric outside `src/analysis/metrics.py` (biology lives in `src/analysis/depmap_validation.py`).
> - If you run into an unexpected obstacle, do not silently rationalise — stop and write the obstacle to `artifacts_v2/diagnostics/p0a_blockers.md` and ask the user before proceeding.
> - When in doubt about runtime, prefer to lower `n_episodes` and `n_starts` to a tiny smoke value first and only scale up once the full pipeline runs end-to-end.

---

## Summary (top-of-mind for V2 lead)

1. **Top likely root causes.**
   (a) The dynamics MLP, regularised with state_linear_skip and trained with heteroscedastic NLL on Δz, is structurally and statistically very close to the `Ridge(z_ctrl ⊕ one_hot(gene))` baseline; the +0.0074 margin is what NLL leaves over Pearson once linear-in-z and per-gene-additive variation are accounted for.
   (b) The RL surrogate environment combines globally contractive dynamics, dense `−d_next` reward, K = 10, and p50 epsilon — together making "any contractive sequence" a successful policy.
   (c) DepMap top-20 hypergeometric is the wrong biological test; the right test is full-action-universe ranked correlation.

2. **Top P0 fixes (phased).**
   - **P0A (now, no retraining):** gate breakdown, per-gene contraction diagnostic, V2 hard benchmark on V1 PPO (smoke first, then full), pairing-noise scoping, biology rerank via a new V2 script. *This phase reveals which of the candidate fixes is most likely to work.*
   - **P0B (later, after P0A review):** numerically safe correlation loss + buffer-based residual-over-ridge dynamics; pick variant with best gate margin.
   - **P0C (later, after P0B review):** terminal-only and Δd reward modes; PPO retrain at K=3; final V2 hard benchmark.

3. **Exact first command sequence to run (P0A only):**
   ```bash
   # 0. branch + diagnostics scaffold
   git checkout -b v2-p0a
   mkdir -p artifacts_v2/diagnostics artifacts_v2/eval artifacts_v2/eval_hard_v1policy_smoke artifacts_v2/eval_hard_v1policy

   # 1. P0A.1 gate breakdown
   PYTHONPATH=. python -m src.analysis.gate_breakdown \
     --dynamics_dir artifacts/dynamics \
     --pairs_dir artifacts/pairs \
     --out artifacts_v2/diagnostics

   # 2. P0A.2 per-gene contraction
   PYTHONPATH=. python scripts/diagnose_action_contraction.py \
     --dynamics_dir artifacts/dynamics \
     --vae_dir artifacts/vae \
     --n_starts 500 \
     --out artifacts_v2/diagnostics

   # 3a. P0A.3 V2 hard benchmark — SMOKE first (single primary cell, n_episodes=20)
   PYTHONPATH=. python -m src.utils.epsilon_percentile --vae_dir artifacts/vae --p 25 \
     > artifacts_v2/epsilon_p25.txt
   PYTHONPATH=. python scripts/evaluate_rl_hard.py --smoke \
     --vae_dir artifacts/vae \
     --dynamics_dir artifacts/dynamics \
     --ppo_zip artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip \
     --out_dir artifacts_v2/eval_hard_v1policy_smoke \
     --k_values 3 \
     --epsilon_values p25 \
     --distance_bins 8-10 \
     --held_out_genes_only true \
     --n_episodes 20 \
     --baselines random,always_noop,greedy_dyn_1,ridge_greedy,mean_delta_greedy

   # 3b. P0A.3 V2 hard benchmark — FULL matrix (only after smoke passes)
   PYTHONPATH=. python scripts/evaluate_rl_hard.py \
     --vae_dir artifacts/vae \
     --dynamics_dir artifacts/dynamics \
     --ppo_zip artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip \
     --out_dir artifacts_v2/eval_hard_v1policy \
     --k_values 1 2 3 8 \
     --epsilon_values p25 p50 \
     --distance_bins 4-6 6-8 8-10 10-12 \
     --held_out_genes_only true,false \
     --n_episodes 500 \
     --baselines random,always_noop,greedy_dyn_1,ridge_greedy,mean_delta_greedy

   # 4. P0A.4 pairing-noise scoping
   PYTHONPATH=. python scripts/diagnose_pairing_noise.py \
     --pairs_dir artifacts/pairs \
     --out artifacts_v2/diagnostics/pairing_noise.json

   # 5. P0A.5 biology rerank — NEW V2 script; do NOT touch scripts/aggregate_eval.py
   PYTHONPATH=. python scripts/evaluate_biology_v2.py \
     --rl_dir artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval \
     --action_freq_ppo_det artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic/action_freq.json \
     --action_freq_random artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/random_baseline/action_freq.json \
     --depmap_csv data/processed/depmap_k562_chronos.parquet \
     --out artifacts_v2/eval/depmap_v2
   ```

4. **Expected success criteria for V2.**
   - **End of P0A:** clean, reviewable benchmark of V1 under V2-hard eval; gate-failure concentration identified; contraction heterogeneity quantified; pairing-noise ceiling estimated; biology rerank reported. *Then we decide whether the next step is dynamics fix or reward fix or both.*
   - **End of P0B:** `artifacts_v2/dynamics_v2_*/gate.json` with `passed=true` and `margin_vs_linear_ridge_pearson ≥ 0.030`, OOD Pearson ≥ 0.40, uncertainty Spearman ≥ 0.20.
   - **End of P0C:** `artifacts_v2/eval_hard_v2policy_v2dyn/results_table.md` primary cell (K=3, ε=p25, bin 8–10, OOD): PPO det success ≥ 0.70, random ≤ 0.40, PPO − random ≥ +25 pp, PPO within ±0.10 of greedy_dynamics_1, PPO mean_steps ≤ 2.0.
   - **End of P1:** pairing-noise corrective tested even if P0B passed; one of {32D V2, 64D V2} selected as primary; ensemble dynamics tested.
   - **Throughout:** V1 artifacts intact and reproducible. All caveats from `artifacts/eval/caveats.md` carried forward. No sacred rule violated.
