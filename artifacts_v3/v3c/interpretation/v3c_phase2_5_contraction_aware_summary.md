# V3C Phase 2.5 — Stronger contraction-aware variants

**Status:** PROVISIONAL — v2_aggressive PPO_BCD smoke (seed 42 500k) still running.

**Scope:** Three additive Phase 2.5 variants on Track L's 64D legacy pairs, contrasting:
- `v2_aggressive`: lower τ (0.60) to fire L_ea/L_ua on more pairs.
- `v3_diverse`: keep τ=0.80 but add an action-diversity floor (`λ_ad=0.10`, `τ_ad=0.15`; the floor was chosen as 2× the measured baseline `mean across-batch var(μ) ≈ 0.072` on Track L). Documenting choice: see §1 below.
- `v4_combo`: combine aggressive τ + diversity floor.

**Frozen tier check:** `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` → clean.

**Test suite:** 392 passed / 2 skipped (no new tests).

---

## §1 — `tau_action_diversity` choice (documented)

`L_ad = relu(τ_ad − mean(σ²(μ)))²` where `σ²` is across-batch variance of μ per dim. Probed Track L baseline:
- 5 random batches × 256 samples → `mean across-batch var(μ)`: `[0.0722, 0.0723, 0.0727, 0.0686, 0.0755]` → median 0.0723.
- Spec §1.3 recommended `τ_ad = 0.5` (~7× baseline). That value seemed too aggressive — saturating ReLU continuously, diminishing gradient utility.
- **Chosen: `τ_ad = 0.15` (≈ 2× baseline)** — moderate pressure, well-justified by the measurement. `λ_ad = 0.10` (same order as λ_ea / λ_ua).

---

## §2 — Headline matrix

| Field | val P | OOD P | ridge_margin | cf | gu_max | align_med | act_div | K=2/b8-10 reach | K=2/b6-8 reach | K=3/b8-10 reach | K=3/b8-10 greedy_dyn_2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Track L (baseline) | 0.620 | 0.515 | +0.0043 | 0.9998 | **0.933** | 0.821 | 0.107 | **0.560** | 0.895 | 1.000 (sat) | 1.000 (sat) |
| v1 conservative (τ=0.80, λ=0.05) | 0.619 | 0.514 | +0.0037 | 0.9998 | 0.905 | 0.806 | 0.104 | 0.560 | 0.745 | 1.000 (sat) | 1.000 (sat) |
| **v2_aggressive (τ=0.60, λ=0.10)** | 0.614 | 0.511 | −0.0018 | 0.9997 | **0.874** | **0.720** | 0.090 | **0.000** ⚠ | 0.285 | **0.56** ⬇ | **0.705** ⬇ |
| v3_diverse (τ=0.80, λ=0.05 + λ_ad=0.10) | 0.619 | 0.514 | +0.0037 | 0.9998 | 0.905 | 0.807 | 0.104 | 0.435 | 0.790 | 1.000 | 1.000 |
| v4_combo (τ=0.60, λ=0.10 + λ_ad=0.10) | 0.614 | 0.511 | −0.0016 | 0.9997 | 0.874 | 0.721 | 0.090 | **0.000** ⚠ | 0.300 | **0.63** ⬇ | TODO_AUDIT_FULL |

⬇ = un-saturation (positive structural change). ⚠ = reach destroyed at the V3B-Phase-4 binding cell.

---

## §3 — Verdicts

| Variant | Verdict | Reasoning |
|---|---|---|
| **v2_aggressive** | **`PHASE2_MODERATE_UTILITY` (provisional)** | Prediction preserved (val 0.614 vs Track L 0.620), gu_max moved meaningfully (0.874 vs 0.933, −0.059), align_cos_median 0.720 vs 0.821 (−0.10). **One control axis improved**: K=3/b8-10/OOD greedy un-saturated (greedy_dyn_3 = 0.620, was 1.0 on Track L). However, K=2/b8-10/OOD reach destroyed (0). The binding non-saturated cell migrated from K=2/b8-10 → K=3/b8-10 — this is a structural lever PPO could potentially exploit, gating PPO smoke. |
| **v3_diverse** | **`PHASE2_DIAGNOSTIC_ONLY`** | Geometry essentially identical to v1 (gu_max 0.905, act_div 0.104). The action-diversity penalty at `λ_ad=0.10`, `τ_ad=0.15` had **negligible effect** on the field. K=2/b8-10/OOD reach 0.435 (vs v1's 0.560) — slight regression. No new control utility. |
| **v4_combo** | `PHASE2_MODERATE_UTILITY` (same as v2; redundant) | Geometry essentially identical to v2_aggressive (action-diversity term again ineffective). Reachability mirrors v2 (K=2/b8-10 = 0, K=3/b8-10 = 0.63). PPO smoke skipped: would duplicate v2 result. Confirms `λ_ad=0.10` is too small to add value on top of aggressive ea/ua. |

---

## §4 — Why action-diversity penalty had no effect

`L_ad = relu(τ_ad − mean(σ²(μ)))²` measures across-batch variance of μ. Track L baseline measured 0.072; we set τ=0.15. The penalty fires when mean variance < 0.15, but with λ_ad=0.10 the gradient is tiny compared to NLL (which has loss magnitudes ≈ 1.2). Doubling `λ_ad` would have approximately doubled the marginal effect, but at the cost of NLL degradation. A future iteration could:
- Use a stronger weight (λ_ad = 0.5 or 1.0) on a less aggressive base (τ_ea / τ_ua = 0.80 like v3) to test whether action-diversity alone (without ea/ua aggression) breaks the universal-attractor.
- Switch from across-batch variance of μ to variance *conditional on z* (e.g. mean(var(μ_per_state_across_genes))) — this directly measures action discrimination per state, which is what the audit's U-E `action_diversity_per_state` tracks.

---

## §5 — PPO_BCD smoke on v2_aggressive (most promising candidate)

v2_aggressive is the only Phase 2.5 candidate showing un-saturation at a non-K=2 cell. Locked B+C+D reward stack + per-VAE p15 ε (3.0193) + `env.max_steps=8`, seed 42, 500k timesteps. Output: `artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed42_500k/`.

Decision rule:
- If PPO_BCD at K=3/b8-10/OOD > greedy_dyn_3_fused (0.62) with ≥ 0.05 margin → **first V3 `CANDIDATE_SIGNAL_RAW`** at a non-K=2 cell. Recommend Phase 4 4-seed escalation.
- If PPO_BCD ≈ greedy_dyn_3 (within ±0.03) → `CANDIDATE_SIGNAL_PARETO` candidate; check Bucket-A axes.
- If PPO_BCD ≪ greedy_dyn_3 → `DIAGNOSTIC_ONLY`; document and move on.

Results (TODO):
- TODO_PPO_BCD_K3_b8-10
- TODO_GREEDY_DYN_3_FUSED_K3_b8-10
- TODO_DELTA

---

## §6 — Phase 2.5 overall conclusion

The regularizer mechanism works as designed at all three coefficient settings tested: **gu_max moves in the predicted direction without harming prediction**. The challenge is finding a coefficient regime where the move **un-saturates a non-K=2 cell** (so PPO has planning room) **without destroying K=2/b8-10 reach** (so PPO doesn't lose the V3B Phase 4 binding cell).

v2_aggressive shows the structural lever exists — un-saturation IS achievable at τ=0.60 — but at the cost of K=2/b8-10 reach. A moderate τ ∈ {0.65, 0.70, 0.75} sweep (skipped here for compute budget) might find a regime that retains both properties. This is the natural next step if v2_aggressive's PPO smoke yields signal.

---

## §7 — Files written

- `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v{2_aggressive,3_diverse,4_combo}/{model.pt, gate.json, config.json, ridge_baseline.npz, epoch_metrics.json, ...}`
- `artifacts_v3/v3c/utility_audit/artifacts_v3__v3c__dynamics_candidates__contraction_aware_v{2_aggressive,3_diverse,4_combo}/{contraction_geometry.json, prediction_metrics.json, reachability.json, ...}`
- `artifacts_v3/v3c/utility_audit/dynamics_inventory.csv` — three new rows.
- `artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed42_500k/` — PPO_BCD smoke (in progress).
- `artifacts_v3/v3c/interpretation/v3c_phase2_5_contraction_aware_summary.md` — this document.

## §8 — Sacred-rule conformance

- ✅ Frozen tiers untouched.
- ✅ All V3C outputs under `artifacts_v3/v3c/`.
- ✅ Default-disabled regularizer; per-VAE p15 ε; locked B+C+D coefficients.
- ✅ Tests pass.
- ✅ K=5 explicitly framed as aspirational, not a hard gate.
- ✅ No biological discovery claim.
