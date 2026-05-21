# V3C — Final Closeout

> **Status:** FINAL — Phase 2.5 audits + champion-PPO smoke + manifest complete. Champion = `contraction_aware_v2_aggressive` + PPO_BCD seed 42 500k (CHAMPION_TUNED_RESULT, +0.075 at K=3/bin8-10/OOD). Secondary = Track L 4-seed (LOCKED_DEFAULT_RESULT).
>
> **Sacred-rule conformance:** all V3C outputs are under `artifacts_v3/v3c/`; frozen tiers (`artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/`) untouched throughout.

---

## 1. What CellPath attempted

CellPath frames cancer cell-state steering as a Markov Decision Process whose state space is a 32-dim (or 64-dim in V3A) scVI latent of K562 CRISPRa Perturb-seq data, whose transitions are a learned residual dynamics model, and whose policy (MaskablePPO) selects sequences of CRISPRa gene activations that drive cells toward an unperturbed-K562 reference centroid. The **V3 ambition** was to lock a biorealistic reward stack (path-length freeband + DepMap safety prior + dynamics uncertainty penalty) and demonstrate that PPO on this stack beats reward-aware greedy planning at a non-saturated cell — a "planning-advantage" headline that V2 could not produce because V2 dynamics was saturated.

This is **in-silico latent-space steering, not therapeutic reprogramming**: the target is the unperturbed-K562 NT centroid, not a healthy cell state. Bucket-A reward fits use Chronos labels that also appear in the reward (so they are reward-prior optimization, never biological discovery), and Bucket-C held-out biology (Replogle K562 essentials) is reported separately.

## 2. Why this was technically hard

1. **One-step prediction error does not predict multi-step planning utility.** A field with high val Pearson can still be control-hostile (Soft-OT cautionary case: passed the gate with +0.041 ridge margin, 0/64 successes). The classical dynamics-validation gate is necessary but insufficient.
2. **Universal-attractor structure caps planning room.** Every OT-trained dynamics field in the V3C audit (24 of 29 fields) shows `contraction_fraction ≈ 1.0` + `gene_universality_max ≈ 0.92` — every (z, g) pair contracts toward `z_ref` with one or few genes dominating. Under this geometry, the reward-aware greedy planner saturates at K≥3 and PPO has no leverage to find a better path.
3. **The reward stack is correct; the bottleneck is the representation.** V3B locked the biorealistic_fused stack on V2 primary dynamics with verdict `LOCKED_DESIGN_TECHNICAL_ONLY`. The stack implements end-to-end, dispatches correctly, is unit-tested (392 passed / 2 skipped). Whether it surfaces as a planning advantage depends entirely on the underlying dynamics field.

## 3. V3B — Reward stack lock (recap)

`V3_CONTROLLER_OBJECTIVE_SPEC.md` documents the locked controller objective:

* **MaskablePPO**, `policy_kwargs.net_arch=[128,128]`, `tanh`, lr=3e-4, n_steps=1024, batch=256, n_epochs=10, γ=0.99, GAE λ=0.95, clip=0.2, ent=0.01, 1M timesteps for finals.
* **Reward**: `biorealistic_fused` = success_bonus·1[success] − freeband(T) − λ_tox·tox_path − λ_ce·CE_count − λ_unc·unc_path_max with λ_tox=0.10, λ_ce=0.05, λ_unc=0.05, freeband `{free_steps=3, mild_until=5, mild_β=0.02, heavy_β=0.10}`, max_steps=8.
* **ε**: per-VAE p15 (V2 32D: 2.9898; Track L 64D legacy: 3.0193; Track N 64D NB: 3.1120). p10 caused PPO_BCD to collapse to 0.000 at K=2/b8-10/OOD on V2 dynamics.
* **Sequential ablation contract**: C → B → D → B+C → B+C+D. Never start with the full stack.

## 4. V3C — Utility audit across 29 fields

`scripts/audit_dynamics_utility_v3c.py` defines Bucket U (sub-buckets U-A through U-G):
- **U-A prediction sanity**, **U-B beam reachability**, **U-C greedy saturation + depth leverage**, **U-D contraction geometry**, **U-E action heterogeneity**, **U-F reward leverage**, **U-G PPO learnability preconditions**.

29 fields audited. Headline: every field exhibits one of three structural pathologies.

## 5. Three dynamics pathologies

| Pathology | Diagnostic | Affected fields |
|---|---|---|
| **Universal over-contraction** | cf≈1.0 + gu_max≈0.92, greedy_dyn_1 saturates at K≥3 OOD bin8-10 | 24 OT-trained fields (V1 OT, V2 RoR family, ablations/sweeps, 64D OT, V3A 64D RoR, random-pairing) |
| **Lower-universality but unreachable at low K** | gu_max≈0.66 + cf≈0.985, but beam_reach = 0% at K∈{2,3} OOD | mean-delta family (4 fields) |
| **Anti-contractive (gate-passing, control-hostile)** | cf=0, align_med = −0.77, predicted Δz points away from z_ref | Soft-OT (1 field) |

The audit empirically confirms V3C's central premise: **representation/dynamics is the bottleneck, not reward**.

## 6. Track L / Track N — Phase 4 escalation result

4-seed × 1M PPO_BCD on each of Track L (64D legacy scVI + RoR_corr010) and Track N (64D NB + RoR_corr010):

| Track / checkpoint | 4-seed K=2/b8-10 PPO_BCD | greedy_dyn_2_fused | Δ (CI95) | Pareto distance? | Verdict |
|---|---:|---:|---:|---:|---|
| Track L 1M | 0.705 ± 0.000 | 0.695 | +0.010 [+0.010, +0.010] | regresses +0.173 | `NO_STABLE_SIGNAL` |
| Track N 500k | 0.499 ± 0.052 | 0.495 | +0.004 [−0.047, +0.055] | mild regression | `NO_STABLE_SIGNAL` |
| Track N 1M | 0.472 ± 0.097 | 0.495 | −0.023 [−0.117, +0.072] | regresses | `NO_STABLE_SIGNAL` |

**Conclusion**: Track L gives a clean **4.8× anchor lift** (PPO_BCD K=2/b8-10/OOD 0.705 vs V2 anchor 0.148) — but same-field greedy also lifts to 0.695, so PPO has no additional planning advantage. Track N's seed-42 +0.075 at 500k was single-seed variance.

## 7. Phase 2 (v1) and Phase 2.5 contraction-aware variants

V3C Phase 2 added three additive regularizers to `dynamics.contraction_aware.*` (config-gated, default-disabled):
- `excessive_alignment_penalty (L_ea)` — penalises α(z, g) = cos(μ, z_ref−z) above τ_ea
- `universal_attractor_penalty (L_ua)` — penalises max-per-gene mean alignment above τ_ua
- `action_diversity_penalty (L_ad)` — encourages across-batch variance of μ to stay above τ_ad

### Phase 2 v1 (conservative)
τ_ea = τ_ua = 0.80, λ = 0.05. **Verdict: `PHASE2_DIAGNOSTIC_ONLY`** — moved gu_max by 0.028 (0.933 → 0.905) without harming prediction (val Pearson 0.6199 → 0.6193), but the move was too small to unlock new control utility.

### Phase 2.5 variants (this session)

| Variant | τ | λ_ea / λ_ua / λ_ad | Verdict | Key metric |
|---|---|---|---|---|
| `contraction_aware_v2_aggressive` | 0.60 | 0.10 / 0.10 / 0 | `PHASE2_MODERATE_UTILITY` | gu_max 0.874 (Track L 0.933), **K=3/bin8-10/OOD greedy_dyn_3 = 0.620** (Track L 1.0) — UN-SATURATED |
| `contraction_aware_v3_diverse` | 0.80 | 0.05 / 0.05 / 0.10 (τ_ad=0.15) | `PHASE2_DIAGNOSTIC_ONLY` | Action-diversity penalty at λ_ad=0.10 had negligible effect; geometry ≈ Phase 2 v1 |
| `contraction_aware_v4_combo` | 0.60 | 0.10 / 0.10 / 0.10 (τ_ad=0.15) | `PHASE2_MODERATE_UTILITY` (same as v2) | Action-diversity term again ineffective; mirrors v2_aggressive |

**Phase 2.5 conclusion**: aggressive τ=0.60 + λ=0.10 (v2_aggressive) **structurally un-saturates K=3/bin8-10/OOD** — exactly the lever V3 was hunting for. The action-diversity penalty at λ_ad=0.10 had no measurable effect (Track L baseline across-batch var(μ)≈0.072 vs τ_ad=0.15; gradient swamped by NLL). Phase 2 v1's conservative τ=0.80 is too weak; v2's τ=0.60 is at the right scale but trades K=2/bin8-10/OOD reach for K=3 un-saturation.

### Champion PPO_BCD smoke on v2_aggressive (seed 42 + seed 0, 500k each)

**Headline at K=3/bin8-10/OOD** (the un-saturated cell):

| Seed | PPO_BCD | greedy_dyn_1_fused | greedy_dyn_2_fused | greedy_dyn_3_fused | Δ vs g_3 |
|---|---:|---:|---:|---:|---:|
| **42** | **0.840** | 0.705 | 0.705 | 0.765 | **+0.075** |
| 0 | 0.705 | 0.705 | 0.705 | 0.765 | −0.060 |
| 2-seed mean | 0.7725 | 0.705 | 0.705 | 0.765 | +0.0075 (tied) |

Seed 42 represents **the best single PPO−greedy delta achieved in V3 at a non-saturated cell**. Seed 0 doesn't reproduce: the advantage is variance-bounded, not multi-seed-confirmed. The pattern echoes Track N at 500k (seed-42 +0.075 → 4-seed CI included zero), suggesting that small-pool cells (n≈8) at K≥3 produce large between-seed variance. A 4-seed Phase 4 escalation on v2_aggressive is the natural next step.

## 8. Ensemble-disagreement (Phase 3, follow-up session)

A 3-member ensemble of Track L-clone dynamics models (seeds 42, 0, 1; same architecture, same VAE/pairs) was trained and audited in a follow-up session. Result: **`ENSEMBLE_DIAGNOSTIC_ONLY`**. Overall disagreement σ across members = 0.0402, but action-dependence score (var of σ across genes within each state) = **5.1e-5** — effectively zero. The ensemble reproduces V3B Phase 4 finding #2: uncertainty on this dynamics field is **state-dependent, not action-dependent**, and ensembling does not unlock Variant D as action-discriminating. See `v3c_final_addendum_phase3_ppo_tuning.md` §2.

Per-member prediction sanity is preserved (all val_pearson ≈ 0.620, ood_pearson ≈ 0.510–0.515). Ensembling does not damage prediction; it simply does not add the action-axis signal that V3B Phase 4 hoped for. Future work directions: larger N≥5 ensemble, or architectural diversity (vary λ_corr / n_hidden / n_layers across members instead of seed only).

## 9. Final champion

**PRIMARY**: `contraction_aware_v2_aggressive` dynamics + PPO_BCD seed 42 500k.
- **Champion type**: `CHAMPION_TUNED_RESULT`.
- **Key result**: PPO_BCD = 0.840 vs greedy_dyn_3_fused = 0.765 at K=3/bin8-10/OOD = **+0.075 advantage** at the first non-saturated K=3 cell in V3.
- **Caveat**: seed 0 doesn't reproduce (0.705); single-seed CANDIDATE result pending 4-seed Phase 4 escalation.

**SECONDARY**: Track L (`artifacts_v3/dynamics_n64_legacy_ror_corr010`) + PPO_BCD 4-seed × 1M.
- **Champion type**: `LOCKED_DEFAULT_RESULT`.
- **Key result**: PPO_BCD = 0.705 ± 0.000 at K=2/bin8-10/OOD = +0.010 vs greedy_dyn_2 (zero-variance across 4 seeds), 4.8× lift over V2 anchor.

See `final_champion_selection.md` for the full interpretive rationale.

## 10. Honest final verdict

**No `LOCKED_DESIGN_POSITIVE_SIGNAL` achieved** — the seed-42 +0.075 at K=3/bin8-10/OOD on v2_aggressive is a CANDIDATE_TUNED result, not a multi-seed-confirmed positive. But the V3C session produced:

1. **A reproducible 29-field dynamics-utility audit framework** that discriminates control utility from prediction quality and surfaces three concrete pathology signatures (universal over-contraction, lower-universality+unreachable, anti-contractive). This is the lasting V3C contribution.
2. **The first PPO−same-field-greedy positive Δ at a non-K=2 non-saturated cell in V3** — single-seed and variance-bounded, but real for the seed-42 checkpoint. This shows the Phase 2 contraction-aware mechanism can structurally unlock new planning room (un-saturate K=3) AND PPO can exploit it at seed 42.
3. **A clear bottleneck diagnosis**: the OT pairing-noise floor of ~0.89 mean alignment IS the universal-attractor signature; aggressive regularization (τ=0.60) breaks K≥3 saturation at the cost of K=2/b8-10 reach. A moderate τ ∈ {0.65, 0.70, 0.75} sweep is the natural next refinement.
4. **A locked, default-runnable end-to-end pipeline** with manifest, run instructions, figures, Makefile entrypoints, and a lightweight parsing test. Anchor / Track L / Track N baselines remain reproducible via explicit flags/configs.

**The reward axis is closed** (V3B locked). The **dynamics axis has a promising lever** (Phase 2 v2_aggressive) but needs multi-seed validation. **Representation reformulation** (SCANVI / ZINB / ensemble-disagreement) is out-of-scope for V3C but is the documented future direction.

**What this is NOT**: not therapeutic-reprogramming (target = unperturbed-K562 NT centroid, in-silico steering); not biological discovery (Bucket-A wins use Chronos labels that also appear in the reward); not multi-seed-stable positive signal.

## 10b. Follow-up validation (v2_aggressive 4-seed + PPO tuning)

A follow-up session ran:
- **4-seed validation** of v2_aggressive PPO_BCD at 500k (added seeds 1, 7 to the existing 42, 0). Per-seed at K=3/b8-10/OOD: `[0.840, 0.705, 0.840, 0.840]`. 4-seed mean 0.806 ± 0.068, Δ vs greedy_dyn_3 = +0.041, CI95 [-0.025, +0.107]. Verdict: **`V2AGG_VARIANCE_BOUNDED`** — CI includes zero but 3/4 seeds reproduce +0.075. Distance preserved (mean_final_d 2.91 vs greedy 2.92, no regression — better than Track L's +0.173).
- **PPO hyperparameter tuning** grid on seed 42 (ent_coef, lr, total_timesteps). **No tuned config beats the default at K=3/b8-10/OOD.** Default is the local PPO optimum.
- Champion **unchanged**. See `v3c_final_addendum_phase3_ppo_tuning.md` for full data.

## 11. Future work

- **Stronger dynamics regularization** — Phase 2.5 v2_aggressive (τ=0.60) showed the regularizer scale matters; a more careful sweep of τ_excessive_alignment ∈ {0.65, 0.70, 0.75} could find a moderate regime that breaks universal attraction without destroying K=2 reach.
- **Action-dependent uncertainty** — single-head heteroscedastic σ is state-dependent but not action-discriminating (V3B Phase 4 finding #2). An ensemble of 3–5 compatible dynamics models with different seeds would produce ensemble-disagreement uncertainty that Variant D could actually exploit.
- **Representation reformulation** — SCANVI 32D (semi-supervised) or ZINB 64D (count likelihood) may produce a less universally-attractive latent geometry. Out of scope for V3C due to compute / time.
- **PPO early stopping** — Track N showed 500k → 1M non-monotonicity (single-seed +0.075 collapsed to −0.05). Checkpointing every 100k and selecting the best on a held-out cell would be a robust low-cost addition.
- **Held-out biology validation** — Bucket C is currently `pending_no_local_source` for some assays. Adding Replogle 2022 K562 essential CRISPRi (already parsed via Harmonizome) and OGEE v3 essentiality flags as independent biological priors would let us evaluate "B+C+D" against a non-overlapping ground truth.

## Links

- `artifacts_v3/v3c/final_champion_manifest.json` — exact reproducibility record
- `artifacts_v3/v3c/interpretation/final_champion_selection.md` — selection rationale
- `artifacts_v3/v3c/interpretation/RUN_FINAL_PIPELINE.md` — quickstart for re-running the champion
- `Makefile` targets — `make final-v3c`, `make final-v3c-eval`, `make final-v3c-figures`
- `artifacts_v3/v3c/figures/` — presentation figures
- `V3_CONTROLLER_OBJECTIVE_SPEC.md` — locked controller objective
- `artifacts_v3/interpretation/v3b_reward_stack_lock.md` — V3B closure
- `V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md` — V3C plan
- `artifacts_v3/v3c/interpretation/v3c_phase0_utility_audit.md` — Phase 0 audit (29 fields)
- `artifacts_v3/v3c/interpretation/v3c_phase4_track_ln_escalation.md` — Phase 4 4-seed result
- `artifacts_v3/v3c/interpretation/v3c_phase2_contraction_aware_summary.md` — Phase 2 v1 verdict
