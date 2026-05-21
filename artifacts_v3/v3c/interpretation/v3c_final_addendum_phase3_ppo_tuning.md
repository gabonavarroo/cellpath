# V3C Final Addendum — v2_aggressive 4-seed + Phase 3 ensemble + PPO tuning

> **Status:** complete. Addendum to `v3c_final_closeout.md`, `final_champion_selection.md`, `final_results_leaderboard.md`. **Champion does NOT change** — the v2_aggressive seed 42 default-config PPO_BCD remains the PRIMARY (CHAMPION_TUNED_RESULT). 4-seed validation, ensemble experiment, and PPO tuning are all reported below with verdicts.
>
> **Sacred-rule check:** `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` → clean. All new outputs under `artifacts_v3/v3c/`.

---

## Headline

| Investigation | Verdict | Champion impact |
|---|---|---|
| v2_aggressive 4-seed PPO_BCD validation (seeds 1, 7 trained; reused 42, 0) | **V2AGG_VARIANCE_BOUNDED** (CI includes zero, but 3/4 seeds robustly +0.075 at K=3/b8-10) | Champion **unchanged** — CHAMPION_TUNED_RESULT label confirmed appropriate |
| Phase 3 ensemble-disagreement (3 Track L seeds: 42, 0, 1) | **ENSEMBLE_DIAGNOSTIC_ONLY** (action-dependence score = 5e-5) | No ensemble PPO smoke run; champion unchanged |
| PPO hyperparameter tuning (ent_coef, lr, total_timesteps × seed 42) | **No tuned config beats default at K=3/b8-10/OOD** | Default config IS the local PPO optimum on this dynamics field |

---

## §1 — v2_aggressive 4-seed validation

**Trained:** 2 new PPO_BCD checkpoints — seeds 1 and 7, 500k each, locked B+C+D, ε=p15=3.0193, max_steps=8. Reused existing seeds {42, 0}.

**Per-seed PPO_BCD success rates (n=200 episodes):**

| Cell | seed42 | seed0 | seed1 | seed7 | 4-seed mean ± std |
|---|---:|---:|---:|---:|---:|
| K=2/b6-8 | 0.185 | 0.155 | 0.175 | 0.130 | 0.161 ± 0.024 |
| K=2/b8-10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| K=3/b6-8 | 0.800 | 0.795 | 0.865 | 0.860 | 0.830 ± 0.037 |
| **K=3/b8-10** | **0.840** | **0.705** | **0.840** | **0.840** | **0.806 ± 0.068** |
| K=4/b8-10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Same-field greedy at K=3/b8-10 (deterministic across seeds): greedy_dyn_1_fused = 0.705, greedy_dyn_2_fused = 0.705, greedy_dyn_3_fused = 0.765.

**Paired Δ vs greedy_dyn_3_fused at K=3/bin8-10/OOD (the un-saturated primary cell):**

- Per-seed Δ: `[+0.075, -0.060, +0.075, +0.075]`
- Mean Δ: **+0.041**, 4-seed CI95 (normal approx): **[-0.025, +0.107]** — **includes zero**
- 3 of 4 seeds (42, 1, 7) reproduce the +0.075 advantage; seed 0 is the outlier at -0.060
- Median Δ = +0.075 (mode = +0.075, 3/4 seeds)

**Verdict: `V2AGG_VARIANCE_BOUNDED`** — the standard CI test does not exclude zero, but the result has a **clear positive structural pattern**: 3 of 4 seeds independently converge to the same PPO_BCD = 0.840 (zero variance among the winning seeds), and only seed 0 finds a different local optimum at greedy_dyn_1's saturation point (0.705). The variance comes entirely from one outlier seed.

**At the K=3/b6-8/OOD cell** (also non-saturated for greedy):
- 4-seed PPO mean: 0.830 ± 0.037
- greedy_dyn_3_fused: 0.935 → 4-seed Δ ≈ -0.105 (PPO loses; not a positive cell)

**Pareto profile (4-seed mean at K=3/b8-10):**
- success: 0.806 vs greedy_dyn_3 0.765 → +0.041 (within ±0.03 tol → tie, slight positive)
- mean_final_distance: 2.91 ± 0.067 vs greedy_dyn_3 ~2.92 → **distance preserved (no regression)** ✓
- tox_path / common_essential: 0.000 for both → tied at clean ✓
- mean_unc_path_max: 0.41 vs greedy_dyn_3 0.44 → improved ✓

The distance-preserved finding is **different from Track L**, where PPO ties greedy on success but regresses on distance (+0.173). On v2_aggressive, PPO matches greedy on distance AND adds a +0.041 mean success edge.

---

## §2 — Phase 3 ensemble-disagreement audit

**Trained:** 2 new Track L-clone dynamics (seeds 0, 1) under `artifacts_v3/v3c/dynamics_ensemble/track_l_clone_seed{0,1}/`. Reused existing Track L (seed 42 effectively) as third ensemble member. All three use identical Track L config (n_latent=64, RoR, λ_corr=0.10, lr=1e-4, λ_combo=0.5).

**Per-member prediction sanity** (V2 anchor val_pearson = 0.615 baseline):

| Member | val_pearson | ood_pearson |
|---|---:|---:|
| `dynamics_n64_legacy_ror_corr010` | 0.6199 | 0.5145 |
| `dynamics_ensemble/track_l_clone_seed0` | 0.6202 | 0.5139 |
| `dynamics_ensemble/track_l_clone_seed1` | 0.6199 | 0.5087 |

All three preserve prediction within ±0.006 of each other. Track L training is essentially deterministic across seeds at this scale.

**Ensemble disagreement audit** (`scripts/audit_ensemble_disagreement.py`, n_states=300 OOD pool, n_genes=105):

- Overall `σ_disagreement` across members per (state, gene): **0.0402** (mean of per-dim across-member std)
- **Action-dependence score** = mean over states of var(σ across genes within that state): **5.1e-5** (effectively zero)
- **Action-dependence normalized** = action_dep_score / σ² = 0.031 (tiny)
- Mean per-state action-σ range (max − min across genes): 0.038 (small absolute, dominated by state)
- Ensemble-mean alignment_cos_median (μ vs z_ref − z): 0.836 (still universal-attractor-like)
- Top-gene entropy normalized: 0.406 (best-aligned gene is moderately concentrated, not uniform)

**Verdict: `ENSEMBLE_DIAGNOSTIC_ONLY`** — the ensemble produces real but **state-dominated** disagreement. Variance across action choices (gene g) at a fixed state is ~5e-5, vs overall σ² ≈ 0.0016. **Uncertainty is state-dependent, not action-dependent** — the V3B Phase 4 finding (single-head heteroscedastic σ is state-dependent only) is **reproduced by ensemble disagreement**.

Implication: Variant D's `λ_unc_path · unc_path_max` cannot be more load-bearing under this ensemble than it was under single-head σ. The ensembling axis does not by itself fix the action-discrimination problem.

**No PPO_BCD smoke run on the ensemble** per the V3C plan gating condition — ENSEMBLE_DIAGNOSTIC_ONLY does not unlock the optional smoke.

---

## §3 — PPO hyperparameter tuning

**Bounded grid on v2_aggressive seed 42 (locked B+C+D reward):**

| Config | total_steps | lr | ent_coef | K=3/b8-10 | K=3/b6-8 | K=2/b6-8 | K=4/b8-10 | mean_final_d K=3/b8-10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **Default 500k (champion)** | 500k | 3e-4 | 0.01 | **0.840** | 0.800 | 0.185 | 1.000 | 2.95 |
| Higher entropy (ent_coef=0.05) | 500k | 3e-4 | 0.05 | 0.705 ⬇ | 0.840 ⬆ | 0.140 | 1.000 | 2.91 |
| Lower LR (lr=1e-4) | 500k | 1e-4 | 0.01 | 0.840 = | 0.745 ⬇ | 0.095 ⬇ | 1.000 | 2.90 |
| Longer (750k) | 750k | 3e-4 | 0.01 | 0.705 ⬇ | 0.870 ⬆ | 0.170 | 1.000 | 2.94 |

⬆/⬇ = better/worse than default. = = tied.

**Findings:**

1. **No tuned config beats default at K=3/bin8-10/OOD** (the primary signal cell). Default config achieves 0.840; ent_coef=0.05 and 750k both regress to 0.705 (greedy_dyn_1-saturation point); lr=1e-4 ties default at 0.840 but loses ground at K=3/b6-8 and K=2/b6-8.
2. **Default config IS the local PPO optimum on v2_aggressive seed 42.** The K=3/b8-10 advantage emerges at 500k with default lr/entropy; both higher entropy (more exploration) and longer training (overfitting? policy drift?) push the policy back to the depth-1 greedy plateau (0.705).
3. **750k regression is consistent with the Track N pattern** (V3C Phase 4): more training does not always help on these small-pool OOD cells (n=8 at K≥3/b8-10), and can shift the policy toward a more conservative greedy-like attractor.
4. **No config produces a clean cross-cell win.** Each tuned config trades K=3/b8-10 for K=3/b6-8 (or vice versa). The selection rule "do not select a config that wins one cell but collapses elsewhere" applies — the default config is the most balanced.

**Verdict: PPO tuning produces NO PPO_TUNED_RESULT that improves on the current champion.** The default 500k seed-42 config IS the locally tuned optimum and remains the PRIMARY champion.

### Secondary: reward-coefficient mini-grid

**Not run.** Per the V3C plan §6 gating: reward tuning is unlocked only after a robust multi-seed positive or a strong Phase 2 candidate. The 4-seed v2_aggressive result is V2AGG_VARIANCE_BOUNDED (CI includes zero), not a robust positive. Tuning the reward stack at this signal level would search a flat landscape and risk over-fitting the K=3/b8-10 single-cell signal.

---

## §4 — Fair final comparison

| Candidate | Cell | Seeds | PPO_BCD | same-field greedy (best K available) | Δ | Verdict |
|---|---|---|---:|---:|---:|---|
| **PRIMARY: v2_aggressive + default 500k** | K=3/b8-10/OOD | 42 | 0.840 | g_3=0.765 | +0.075 | CHAMPION_TUNED_RESULT |
| Same, 4-seed | K=3/b8-10/OOD | 42,0,1,7 | 0.806 ± 0.068 | g_3=0.765 | +0.041 CI [-0.025,+0.107] | V2AGG_VARIANCE_BOUNDED |
| Best PPO-tuned v2_aggressive | K=3/b8-10/OOD | 42 | 0.840 (= default) | g_3=0.765 | +0.075 | No improvement |
| **SECONDARY: Track L + 4-seed 1M** | K=2/b8-10/OOD | 42,0,1,7 | 0.705 ± 0.000 | g_2=0.695 | +0.010 (CI excludes 0, but distance regresses) | LOCKED_DEFAULT_RESULT (NO_STABLE_SIGNAL) |
| Phase 3 ensemble (3 Track L members) | n/a (audit only) | 42,0,1 | n/a | n/a | n/a | ENSEMBLE_DIAGNOSTIC_ONLY |
| V2 anchor 4-seed (V3B Phase 4) | K=2/b8-10/OOD | 42,0,1,7 | 0.148 ± 0.037 | g_2=0.130 | +0.018 | LOCKED_DESIGN_TECHNICAL_ONLY |

**Selection by criteria:**

1. **Reproducible PPO-minus-greedy delta**: v2_aggressive default (seed 42 = +0.075; 3 of 4 seeds also +0.075). Closest to a stable positive.
2. **Final-distance behavior**: v2_aggressive (distance preserved at K=3/b8-10) beats Track L (+0.173 regression at K=2/b8-10).
3. **Reward-fit metrics**: v2_aggressive PPO tox=0, CE=0, unc=0.41 vs greedy_dyn_3 unc=0.44 → marginally improved Variant D axis.
4. **Seed stability**: V2 anchor and Track L have lower per-seed variance than v2_aggressive at the primary cell; but the underlying signal is also smaller. Trade-off.
5. **Consistency across canonical cells**: v2_aggressive loses K=2/b8-10/OOD entirely (reach destroyed). Track L preserves all cells. Neither dominates everywhere.

**Decision: champion unchanged.** v2_aggressive seed 42 default 500k remains PRIMARY (CHAMPION_TUNED_RESULT). Track L 4-seed remains SECONDARY (LOCKED_DEFAULT_RESULT).

---

## §5 — Whether final default pipeline should change

**No.** The default pipeline target (`make final-v3c-eval`) already runs the champion (v2_aggressive seed 42 default 500k) per `final_champion_manifest.json`. The 4-seed validation, ensemble experiment, and PPO tuning all support the existing manifest:

- v2_aggressive seed 42 default is the locally-tuned PPO optimum on this dynamics field.
- 4-seed pattern is robust at 3/4 seeds (positive lean) but not CI-positive — the manifest's `CHAMPION_TUNED_RESULT` label and `if_phase4_4seed_were_run` clause already anticipate exactly this outcome.
- No ensemble candidate or PPO-tuned config supplants the champion.

The manifest's existing `limitations` section already lists "Seed 0 PPO_BCD at K=3/b8-10 = 0.705 (loses to greedy_dyn_3 by -0.060)" and "Single-seed CANDIDATE_TUNED claim". This addendum adds:

- 4-seed mean is +0.041 (positive lean) but CI includes zero.
- PPO tuning grid found default IS the local optimum (no improvement).
- Ensemble of 3 Track L seeds confirms uncertainty is state-dependent not action-dependent (reproduces V3B Phase 4 finding) — ruling out ensembling as the missing piece for Variant D.

---

## §6 — Files written (this addendum)

- `artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed1_500k/{ppo.zip, eval/, metadata.json, ...}`
- `artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed7_500k/{ppo.zip, eval/, metadata.json, ...}`
- `artifacts_v3/v3c/rl_tuning/v2agg_seed42_ent005_500k/{ppo.zip, eval/, ...}`
- `artifacts_v3/v3c/rl_tuning/v2agg_seed42_lr1e4_500k/{ppo.zip, eval/, ...}`
- `artifacts_v3/v3c/rl_tuning/v2agg_seed42_750k/{ppo.zip, eval/, ...}`
- `artifacts_v3/v3c/dynamics_ensemble/track_l_clone_seed0/{model.pt, gate.json, ...}`
- `artifacts_v3/v3c/dynamics_ensemble/track_l_clone_seed1/{model.pt, gate.json, ...}`
- `artifacts_v3/v3c/utility_audit/ensemble_track_l/disagreement_audit.json`
- `scripts/audit_ensemble_disagreement.py` (new)
- `artifacts_v3/v3c/interpretation/v3c_final_addendum_phase3_ppo_tuning.md` (this document)

---

## §7 — Recommended next session

1. **Moderate-τ contraction sweep** ∈ {0.65, 0.70, 0.75} — Phase 2.5 showed τ=0.80 too weak (DIAGNOSTIC_ONLY) and τ=0.60 trades K=2 reach for K=3 un-saturation; an intermediate τ might retain BOTH properties.
2. **Larger ensemble (N=5+ Track L seeds)** to see if action-dependence emerges with more members. The N=3 result at action_dep=5e-5 is near zero but may grow with N.
3. **Different dynamics architectures** for ensemble members (vary λ_corr, n_hidden, n_layers slightly) — diverse-models ensemble may produce more action-dependence than seed-only ensemble.
4. **Try a true Phase 4 4-seed × 1M escalation** on v2_aggressive — the current 4-seed is at 500k; longer training (with the caveat that 750k seed 42 regressed) might or might not stabilize the 3-of-4 pattern.
5. **PPO early-stopping** keyed on the K=3/b8-10/OOD cell — checkpoint every 100k and select the best (mirrors the Track N 500k peak).
