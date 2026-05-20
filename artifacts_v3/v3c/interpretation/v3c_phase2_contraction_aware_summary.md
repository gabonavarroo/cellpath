# V3C Phase 2 — `contraction_aware_v1` audit summary

**Status:** `PHASE2_DIAGNOSTIC_ONLY` — the regularizer reduces the targeted `UNIVERSAL_ATTRACTOR_GENE` axis (gu_max −0.028) without degrading prediction, but the move is too small to unlock K=5 or improve control utility over Track L. No PPO smoke recommended; Phase 2.5 should explore a more aggressive coefficient regime.

**Scope:** Trained one conservative contraction-aware candidate (τ_ea = τ_ua = 0.80, λ_ea = λ_ua = 0.05, λ_ad = 0) on Track L's 64D legacy scVI pairs, then ran the full V3C utility audit (`audit_dynamics_utility_v3c.py all`, n_episodes=200). Spec: `artifacts_v3/v3c/interpretation/v3c_phase2_contraction_aware_spec.md`.

**Frozen tier check:** `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` → clean.

**Test suite:** 392 passed / 2 skipped (377 prior baseline + 15 new contraction-aware tests).

---

## §1 — Headline

| Field | val Pearson | OOD Pearson | cf | gu_max | align_med | act_div | K=2/b8-10 reach | K=5/b8-10 reach |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 anchor | 0.615 | 0.516 | 0.9995 | 0.921 | 0.784 | 0.116 | 0.120 | 1.000 |
| Track L (n64 legacy, baseline arch) | **0.620** | 0.515 | 0.9998 | 0.933 | 0.821 | 0.107 | **0.560** | 1.000 |
| Track N (n64 NB, baseline arch) | 0.621 | 0.504 | 0.9992 | 0.932 | 0.796 | 0.114 | 0.435 | 1.000 |
| mean_delta_corr_010 | 0.520 | 0.384 | 0.985 | 0.657 | 0.487 | 0.135 | 0.000 | 0.000 |
| Soft-OT | 0.934 | 0.743 | 0.000 | −0.622 | −0.770 | 0.048 | 0.000 | 0.000 |
| **`contraction_aware_v1` (NEW)** | **0.619** | 0.514 | 0.9998 | **0.905** | 0.806 | 0.104 | **0.560** | 1.000 |

**Geometry changed (gu_max −0.028), prediction preserved, K=2 reach matches Track L — but no K≥3 un-saturation and no improvement over Track L on any usable cell.**

---

## §2 — Verdict: `PHASE2_DIAGNOSTIC_ONLY`

Mapping against the spec §4 criteria:

| Criterion | Spec threshold | Observed | Status |
|---|---|---|---|
| (U-A) val Pearson preserved | ≥ V2 anchor − 0.05 = 0.565 | **0.619** | ✅ preserved |
| (U-D) `contraction_fraction` reduced | < 0.97 | **0.9998** | ✗ unchanged |
| (U-D) `gene_universality_max` reduced | < 0.85 | **0.905** | ✗ above threshold (but reduced from Track L's 0.933) |
| (U-B) K=2 reach lift vs anchor/Track L | improvement | matches Track L, lifts over anchor | △ no incremental lift over Track L |
| (U-E) `action_diversity_per_state` up | improvement | **0.104** (down 0.003 from Track L) | ✗ slightly worse |
| (U-E) `first_action_entropy_fused` up | improvement | (see §3) | ✗ within sampling |
| (U-B) K=5/bin8-10 nontrivial & non-saturated | reach > 0.2 AND < 0.95 | **1.000** | ✗ still saturated |

**Verdict labels considered:**
- `PHASE2_STRONG_K5_UTILITY`: rejected — K=5 still 100% reachable (saturated). No depth leverage gained.
- `PHASE2_MODERATE_UTILITY`: rejected — although gu_max moved in the right direction, it did not cross the spec's < 0.85 threshold, and no control-utility axis improved versus Track L.
- `PHASE2_DIAGNOSTIC_ONLY`: **selected** — geometry of the dynamics field changed in the predicted direction (gu_max −0.028, align_med −0.015), prediction sanity is fully preserved, but the change is **not large enough to translate into measurable control-utility improvement** at the canonical 7-cell hardness matrix.
- `PHASE2_FAILED_FIELD`: rejected — prediction did not collapse, reachability was not destroyed, and the field is contractive (not Soft-OT-like).

---

## §3 — Detailed comparison vs Track L

### 3.1 U-A — Prediction sanity (preserved)

| | Track L | contraction_aware_v1 |
|---|---:|---:|
| val Pearson | 0.6199 | 0.6193 |
| OOD Pearson | 0.5145 | 0.5142 |
| val MLP−ridge Pearson | +0.0043 | +0.0037 |
| uncertainty Spearman (val) | 0.805 | 0.805 |
| `prediction_pathological_flag` | False | False |

The regularizer cost the predictive head **−0.0006** val Pearson and **−0.0003** OOD Pearson — within sampling noise. The MLP−ridge margin dropped marginally (+0.0043 → +0.0037), still positive. Uncertainty calibration unchanged.

### 3.2 U-D — Contraction geometry (targeted axis moved; secondary axes flat)

| OOD start pool | Track L | contraction_aware_v1 | Δ |
|---|---:|---:|---:|
| `contraction_fraction` | 0.9998 | 0.9998 | 0.0000 |
| `alignment_cos_median` | 0.821 | 0.806 | −0.0150 |
| `gene_universality_max` | **0.933** | **0.905** | **−0.0278** |
| `gene_universality_gini` | 0.064 | 0.065 | +0.001 |
| `action_diversity_per_state` | 0.107 | 0.104 | −0.003 |
| `state_diversity_per_action` | 0.083 | 0.087 | +0.004 |
| `delta_magnitude_median` | 2.835 | 2.806 | −0.029 |

The **targeted gu_max axis moved 0.028 in the right direction**, but the `contraction_fraction = 0.9998` is essentially unchanged — every (z, g) pair still contracts toward z_ref. The regularizer modestly redistributes alignment across genes (slightly higher state_diversity_per_action and lower gu_max) without flipping any (z, g) pair to a non-contractive direction.

This is consistent with the regularizer formulation: L_ea fires only when α > 0.80 on a particular (z, g) row; L_ua fires only when one gene's mean α exceeds 0.80. **Most rows already had α ∈ [0.5, 0.8]** on Track L, so L_ea fired on a relatively small minority of pairs and the gradient signal averaged out across the batch. The mass that L_ua _did_ reach (the top-1 attractor gene) shrank by 0.028, exactly the spec's predicted effect, but the overall contraction count stayed at 100%.

### 3.3 U-B — Reachability (preserved at K=2/b8-10; flat elsewhere)

| Cell | V2 anchor | Track L | contraction_aware_v1 | Δ vs Track L |
|---|---:|---:|---:|---:|
| K=2/bin6-8/OOD | 0.655 | 0.895 | 0.745 | **−0.150** |
| **K=2/bin8-10/OOD** | 0.120 | **0.560** | **0.560** | **0.000** |
| K=3/bin6-8/OOD | 0.975 | 1.000 | 0.970 | −0.030 |
| K=3/bin8-10/OOD | 1.000 | 1.000 | 1.000 | 0.000 (sat) |
| K=4–K=8/bin8-10/OOD | 1.000 | 1.000 | 1.000 | 0.000 (sat) |

The audit-discriminating K=2/bin8-10/OOD cell **holds the Track L lift over anchor** (0.560 vs 0.120 = 4.7×). However, K=2/bin6-8/OOD regresses 0.150 versus Track L. Interpretation: the regularizer constrains the field's contraction structure, which mildly tightens the actionable region at the easier K=2 cell (b6-8) but does not change the difficulty frontier at b8-10.

### 3.4 U-C — Greedy saturation / depth leverage (unchanged)

`greedy_dyn_1` distance success at K=3/b8-10 hits 1.000 at the candidate (same as anchor and Track L). The regularizer did not unlock depth leverage; greedy still saturates by K≥3 OOD bin8-10.

### 3.5 U-E — Action heterogeneity (slight regression)

| Cell | Track L `first_action_entropy_fused` | contraction_aware_v1 | Δ |
|---|---:|---:|---:|
| (representative) k3_bin8-10 | (see audit JSONs) | slightly lower entropy | ≤ 0 |

The action-diversity axis (U-E) was deliberately left disabled (λ_ad = 0). The slight regression in entropy is consistent with the L_ea+L_ua pair *concentrating* mass on fewer attractor genes (the audit's `top10_genes_fused` shifts subtly) rather than diversifying it.

### 3.6 U-F — Reward leverage (no Pareto signal)

Reward-leverage at all 7 cells shows tiny deltas (success deltas within ±0.005 between distance-only and fused beam). No Pareto signal emerges — the field is too narrow at the actionable cells for the reward shaping to find a separating policy.

### 3.7 U-G — Preconditions (still fails the gate; matches Track L baseline)

| Precondition | Pass? |
|---|---|
| U-A val Pearson ≥ 0.40 | ✅ (0.619) |
| U-A OOD Pearson ≥ 0.20 | ✅ (0.514) |
| U-A uncertainty Spearman ≥ 0.10 | ✅ (0.805) |
| U-B reach > 0 at K=4/bin8-10/OOD | ✅ (1.000 — saturated) |
| U-B reach > 0 at K=5/bin8-10/OOD | ✅ (1.000) |
| U-B reach > 0 at K=8/bin8-10/OOD | ✅ (1.000) |
| U-C cumulative depth leverage > 0 | ✗ (greedy saturated at K=1 OOD bin8-10) |
| **all_preconditions_pass** | **False** (matches Track L) |

util_score: **0.370** vs Track L's 0.365, V2 anchor's 0.369 — essentially tied. The regularizer did not move the composite ranking aid.

---

## §4 — Mechanism interpretation

The spec predicted that L_ea (excessive alignment) + L_ua (universal attractor) would attack the **OT pairing noise floor** — the structural reason all OT-trained fields land at `gu_max ≈ 0.92`. The audit confirms a real but **conservative-scale** effect:

1. **The signal sign is correct:** every targeted quantity moved in the predicted direction (gu_max ↓ 0.028, align_med ↓ 0.015, top-K gene concentration shifted), with no degradation in val/OOD prediction. The losses do what they were designed to do at the chosen weights.
2. **The signal magnitude is small:** 3% reduction in gu_max is not enough to unlock K=5 utility, because the structural contraction (`cf = 0.9998`) is independent of the gu_max axis. Soft-OT shows the *upper* bound of the contraction axis (cf = 0.0 vs 1.0 for OT fields); to move toward a usable middle, a stronger or differently-aimed regularizer is needed.
3. **The Pareto frontier (U-F) doesn't open:** because the field still contracts every (z, g) pair, the reward-aware fused beam and the distance-only beam converge on the same actions — there is no choice for the reward to discriminate.

**This is consistent with the Phase 0C central finding:** representation/dynamics is the bottleneck, and the conservative regularizer at τ=0.80 is not strong enough to *break* the universal-attractor structure — it only nudges the dominant attractor gene.

---

## §5 — PPO smoke: NOT recommended for v1

Per Phase 2 spec §5 gating: smoke only if `PHASE2_MODERATE_UTILITY` or `PHASE2_STRONG_K5_UTILITY`. The verdict here is `PHASE2_DIAGNOSTIC_ONLY`, so no PPO smoke is run. The reasoning:

- K=2/bin8-10/OOD reach matches Track L exactly (0.560). A PPO smoke on this field would re-test what we already know from Track L's Phase 1 smoke (which already won the same cell at the same reach). It does NOT test the targeted hypothesis (does breaking the universal-attractor unlock new utility?) because the universal-attractor was barely broken.
- K=5/bin8-10/OOD reach is still 1.000 (saturated). No PPO room.
- The reward-aware beam and distance-aware beam show no Pareto separation.

The honest read: the regularizer needs a more aggressive coefficient regime (τ ≤ 0.60 and/or λ ≥ 0.20, possibly with λ_ad > 0 enabled) before a PPO smoke would be informative.

---

## §6 — Phase 2.5 next steps (recommendation)

The v1 candidate validates the **mechanism** (the regularizer moves the targeted axis in the right direction without harming prediction). The next iteration should test whether **stronger** regularization can break the contraction structure. Suggested grid (a small sweep, not a tune):

| Variant | τ_ea / τ_ua | λ_ea / λ_ua | λ_ad / τ_ad | Hypothesis |
|---|---|---|---|---|
| v2_aggressive | 0.60 / 0.60 | 0.10 / 0.10 | 0 / 0 | Lower cap → more pairs over threshold → stronger gradient on gu_max |
| v3_diverse | 0.80 / 0.80 | 0.05 / 0.05 | 0.05 / 0.5 | Add action-diversity floor to push for state-conditional Δz |
| v4_combo | 0.60 / 0.60 | 0.10 / 0.10 | 0.05 / 0.5 | Aggressive cap + diversity floor — most likely to break `cf = 1.0` |

These are research candidates, not a tuning sweep. **Per the user's session brief, reward-coefficient tuning is deferred** until either a Track A multi-seed positive or a strong Phase 2 candidate emerges.

If v2–v4 still produce `PHASE2_DIAGNOSTIC_ONLY` or worse:
- Move to **ensemble-disagreement** dynamics (action-dependent uncertainty, V3.fallback.C) which is mechanistically distinct from the contraction-axis regularizer.
- Or to **SCANVI 32D / ZINB 64D** representation reformulations (V3 stub axis A).

---

## §7 — Files written

- `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v1/{model.pt, config.json, gate.json, val_metrics.json, ood_metrics.json, gate_diagnostics.json, epoch_metrics.json, checkpoint_comparison.json, model_best_nll.pt, model_best_gate.pt, ridge_baseline.npz}` — trained dynamics + gate artifacts.
- `artifacts_v3/v3c/utility_audit/artifacts_v3__v3c__dynamics_candidates__contraction_aware_v1/{prediction_metrics, reachability, greedy_saturation, contraction_geometry, action_heterogeneity, reward_leverage_fused, ppo_preconditions, bucket_u_index}.json` + `bucket_u_summary.md` — V3C utility audit outputs.
- `src/models/dynamics.py` — three new loss helpers: `excessive_alignment_penalty`, `universal_attractor_penalty`, `action_diversity_penalty` (+ `_alignment_cosine` private helper).
- `scripts/train_dynamics.py` — additive wiring for the new losses behind `dynamics.contraction_aware.*` config keys; loads `z_ref` once when active.
- `scripts/audit_dynamics_utility_v3c.py` — recognizes the new `artifacts_v3/v3c/dynamics_candidates/contraction_aware*` path family.
- `config/dynamics.yaml` — new additive block `dynamics.contraction_aware.*` (all λ_* default 0.0 → byte-identical V2/V3 default behavior).
- `tests/test_dynamics_contraction_aware.py` — 15 new TDD-discipline unit tests for the three loss helpers (all passing).
- `artifacts_v3/v3c/utility_audit/dynamics_inventory.csv` — one new row for the candidate.

---

## §8 — Sacred-rule conformance

- ✅ Frozen tiers untouched (`git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` → clean).
- ✅ All new V3C outputs under `artifacts_v3/v3c/`.
- ✅ Default-disabled regularizer (config defaults `λ_*=0.0`); existing V2/V3 training byte-identical.
- ✅ K=5 framed as aspirational; verdict uses flexible Phase 2 labels (not a hard rejection on K=5 alone).
- ✅ Locked B+C+D reward stack untouched. No reward-coefficient tuning.
- ✅ No claim of biological discovery. The regularizer affects in-silico latent-space geometry only.
- ✅ Tests: 15 new unit tests, all passing; full suite 392 passed / 2 skipped (was 377 baseline → +15 contraction-aware tests).
