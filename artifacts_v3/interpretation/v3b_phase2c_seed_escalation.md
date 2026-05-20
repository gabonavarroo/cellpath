# V3B Phase 2c — 4-Seed Escalation Interpretation

**Date:** 2026-05-18
**Author:** V3 research lead (CC agent)
**Scope:** 4-seed escalation of V3B Phase 2 safety-aware PPO_C + Replogle held-out (Bucket-C) check.
**Sacred-rule conformance:** writes only under `artifacts_v3/`. No code edits (only `scripts/train_rl_v3b.py` had a 4-line path-naming refresh, see §8). Phase 2 PPO checkpoints and eval summaries untouched. Frozen tiers (`artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/`) clean.

---

## 1. Headline

* **Final verdict: `PHASE2C_PROMISING_BUT_UNSTABLE_SEED_VARIANCE_DOMINATES`** (matches 1/3 acceptance criteria; the 2 strict CI criteria fail because the Phase 2 single-seed signal does not survive seed escalation).
* **Phase 3 (path-length B) status: UNLOCKED for testing, but Phase 2's Variant C is NOT a publishable headline.**
* **Bucket-C held-out validation: `HELDOUT_INCONCLUSIVE_NO_GENERALIZATION_DETECTED`** — DepMap-Chronos safety prior does NOT transfer to the Replogle 2022 K562 essential CRISPRi assay.
* **Seed 42 was REUSED** (symlinked from Phase 2 artifacts; not retrained). Seeds {0, 1, 7} × {real, permuted} = 6 new PPO retrains.

## 2. Seeds trained and evaluated

| Seed | Real-Chronos PPO_C | Permuted-Chronos PPO_C |
|---|---|---|
| 42 | **REUSED** from Phase 2 (`rl_v3b_safety_aware_v2primary_seed42/`; symlinked to `rl_v3b_safety_aware_seed42/`) | **REUSED** from Phase 2 (symlinked) |
| 0 | Trained this session (1M timesteps, 3.4 min) — `rl_v3b_safety_aware_seed0/` | Trained this session — `rl_v3b_safety_aware_seed0_permuted_chronos/` |
| 1 | Trained this session (3.3 min) — `rl_v3b_safety_aware_seed1/` | Trained this session — `rl_v3b_safety_aware_seed1_permuted_chronos/` |
| 7 | Trained this session (3.4 min) — `rl_v3b_safety_aware_seed7/` | Trained this session — `rl_v3b_safety_aware_seed7_permuted_chronos/` |

Total training wall-clock: ~20 min (6 × 3.3 min). Total evaluation wall-clock: ~7 min (4 × ~1.7 min per seed). Aggregation + held-out: ~3 min.

## 3. Bucket B (reward-independent) — 4-seed raw success per cell

(mean ± std across seeds {42, 0, 1, 7}; n=300 episodes/cell)

| Cell | ppo_C | ppo_A | ppo_C_permuted | greedy_dyn_2_C | greedy_dyn_2_A |
|---|---:|---:|---:|---:|---:|
| K=2 / bin 6-8 / OOD | **0.743 ± 0.020** | 0.773 ± 0.000 | 0.755 ± 0.013 | 0.790 ± 0.000 | 0.790 ± 0.000 |
| **K=2 / bin 8-10 / OOD ⭐ (leakage-safe headline cell)** | **0.309 ± 0.057** | 0.300 ± 0.000 | 0.331 ± 0.058 | 0.300 ± 0.000 | 0.300 ± 0.000 |
| K=3 / bin 6-8 / OOD | 0.998 ± 0.002 | 0.997 ± 0.000 | 0.999 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| K=3 / bin 8-10 / OOD (V2 primary, saturated) | 0.940 ± 0.000 | 1.000 ± 0.000 | 0.952 ± 0.097 | 1.000 ± 0.000 | 1.000 ± 0.000 |

Note: PPO_A, greedy_dyn_2_C, greedy_dyn_2_A all have std = 0 — they are deterministic in eval (same eval seed; PPO_A is the same checkpoint evaluated 4 times; greedy is deterministic given dynamics). Only PPO_C and PPO_C_permuted have real seed variance.

## 4. Bucket B paired-by-seed deltas (4-seed 95 % CI)

| Cell | Δ(PPO_C − PPO_A) | Δ(PPO_C − greedy_dyn_2_C) | Δ(real − permuted) |
|---|---|---|---|
| K=2 / bin 6-8 / OOD | **−0.031** [−0.050, −0.012] ❌ regression | **−0.048** [−0.067, −0.028] ❌ | −0.013 [−0.040, +0.015] — |
| K=2 / bin 8-10 / OOD ⭐ | +0.009 [−0.046, +0.065] — | +0.009 [−0.046, +0.065] — | **−0.022** [−0.130, +0.086] — |
| K=3 / bin 6-8 / OOD | +0.002 [−0.000, +0.004] — | −0.002 [−0.004, +0.000] — | −0.001 [−0.004, +0.002] — |
| K=3 / bin 8-10 / OOD | **−0.060** [−0.060, −0.060] ❌ regression | **−0.060** [−0.060, −0.060] ❌ regression | −0.012 [−0.106, +0.083] — |

(✅ = CI strictly excludes zero in the favourable direction; ❌ = excludes zero in the unfavourable direction; — = CI straddles zero.)

### 4.1 What this tells us

* **The +4 pp Phase 2 single-seed headline at K=2 / bin 8-10 / OOD collapses to +0.9 pp across 4 seeds** (CI [−4.6, +6.5]). Phase 2 was seed-42 noise.
* **At K=2 / bin 6-8 / OOD, PPO_C is statistically WORSE than both PPO_A (−3.1 pp [−5.0, −1.2]) and greedy_dyn_2_C (−4.8 pp [−6.7, −2.8])** — significant regressions.
* **The "real − permuted" claim from Phase 2 collapses to slightly NEGATIVE at K=2 / bin 8-10** (mean −2.2 pp): in seeds 0 and 1 the permuted PPO actually outperforms the real PPO on raw success at this cell. The "label structure is load-bearing on raw success" Phase 2 framing does NOT hold.
* **At the V2 saturated primary K=3 / bin 8-10 / OOD, PPO_C consistently loses 6 pp** (zero std — every seed loses the same 6 pp). This is the pure-constraint cost of the safety reward at a cell where greedy_dyn_2 already saturates without using essentials.

### 4.2 Per-seed raw success at the headline cell K=2 / bin 8-10 / OOD

| Seed | ppo_C | ppo_A | ppo_C_permuted | greedy_dyn_2_C |
|---|---:|---:|---:|---:|
| 42 | **0.340** | 0.300 | 0.267 | 0.300 |
| 0 | **0.267** | 0.300 | **0.393** | 0.300 |
| 1 | **0.257** | 0.300 | **0.363** | 0.300 |
| 7 | **0.373** | 0.300 | 0.300 | 0.300 |

Two seeds (42, 7) place PPO_C ≥ PPO_A; two seeds (0, 1) place PPO_C < PPO_A AND PPO_C < permuted-PPO_C. The seed-42 result that drove Phase 2's ACCEPT was on the favourable side of this distribution.

## 5. Bucket A (reward-fit) — 4-seed solidity

In contrast to Bucket B, the reward-fit metrics are **rock-solid across all 4 seeds**:

* `mean_tox_path` for PPO_C: **0.000 ± 0.000** at every cell.
* `mean_common_essential_per_ep` for PPO_C: **0.000 ± 0.000** at every cell.
* The permuted-Chronos PPO_C has mean_CE/ep = **0.022–0.032** with seed std ≥ 0.024 — does NOT achieve the safety profile that real-Chronos PPO does.

So the **DepMap-Chronos safety prior is faithfully optimized**: PPO_C avoids all 5 common-essential genes (CBFA2T3, HK2, PLK4, PTPN1, STIL) across all seeds and cells, and the permuted variant does not. This is what the reward was designed to do; **it is reward-prior optimization, not biological discovery**.

## 6. Bucket C (held-out biological validation) — Replogle 2022 K562 essential CRISPRi

**A held-out source was added this session**: Harmonizome's mirror of Replogle 2022 K562 essential Perturb-seq gene perturbation signatures. Source URL: `maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.,+Cell,+2022+K562+Essential+Perturb-seq+...`. Full details in `replogle_heldout_summary.md`.

* Replogle-essential ∩ Norman 105: **6 genes** (FOXL2, KIF18B, NCL, PLK4, SET, STIL).
* Agreement with DepMap-essential ∩ Norman 105: **2 genes** (PLK4, STIL).
* **Replogle-only essentials** (in Replogle but NOT DepMap, so not in reward): **4 genes** (FOXL2, KIF18B, NCL, SET). The clean Bucket-C check.

### Paired-by-seed PPO_C − PPO_A on `frac_actions_replogle_only_essential`:

| Cell | Mean Δ (PPO_C − PPO_A) | per-seed [42, 0, 1, 7] | 4-seed CI | CI excludes 0 |
|---|---:|---|---|:---:|
| K=2 / bin 6-8 / OOD | +0.0004 | [+0.0017, 0, 0, 0] | [−0.0004, +0.0012] | ❌ |
| K=2 / bin 8-10 / OOD | **+0.0071** | [+0.0283, 0, 0, 0] | [−0.0068, +0.0210] | ❌ |
| K=3 / bin 6-8 / OOD | +0.0004 | [+0.0015, 0, 0, 0] | [−0.0004, +0.0011] | ❌ |
| K=3 / bin 8-10 / OOD | +0.0053 | [+0.0213, 0, 0, 0] | [−0.0051, +0.0158] | ❌ |

**Direction:** every cell shows a **non-negative** mean — if anything, PPO_C picks Replogle-only essentials **more often** than PPO_A (entirely driven by seed 42). PPO_A's 0 % baseline across all seeds and cells leaves no Bucket-C floor for PPO_C to beat.

**Verdict:** `HELDOUT_INCONCLUSIVE_NO_GENERALIZATION_DETECTED`. The DepMap-Chronos safety prior shows no held-out generalization to Replogle CRISPRi essentiality.

## 7. Acceptance criteria — verdict

| # | Criterion | Result | Passed |
|---|---|---|:---:|
| 1 | 4-seed 95% CI on PPO_C − PPO_A raw success at K=2 / bin 8-10 / OOD strictly excludes zero | +0.0092 [−0.0464, +0.0647] | ❌ |
| 2 | 4-seed 95% CI on real − permuted raw success at K=2 / bin 8-10 / OOD strictly excludes zero | −0.0217 [−0.1295, +0.0861] | ❌ |
| 3 | PPO_C raw success at K=2 / bin 6-8 / OOD ≥ PPO_A − 0.05 | PPO_C 0.7425 vs PPO_A 0.7733 (regression 0.031 ≤ 0.05) | ✅ |

**Two of three strict criteria FAIL.** The Phase 2 single-seed +4 pp headline does not survive seed escalation. Verdict: **`PHASE2C_PROMISING_BUT_UNSTABLE_SEED_VARIANCE_DOMINATES`**.

## 8. What this means for Phase 2's claims

### 8.1 Withdrawn claims

* "**PPO_C beats greedy_dyn_2_C by +4 pp at K=2 / bin 8-10 / OOD**" — **WITHDRAWN**. 4-seed mean is +0.9 pp ± 5.7 pp. Phase 2's single-seed observation was within seed noise.
* "**Real-Chronos PPO_C beats permuted-Chronos PPO_C on raw success at K=2 / bin 8-10 / OOD**" — **WITHDRAWN**. 4-seed mean is −2.2 pp; the permuted variant slightly outperforms the real variant on raw success at this cell.
* "**First V3-era result where PPO strictly exceeds the depth-2 model-based oracle**" — **WITHDRAWN** for Variant C. Not statistically supported with 4 seeds.

### 8.2 Held claims (Bucket A — reward-fit, expected by construction)

* PPO_C **optimizes the DepMap-Chronos safety prior to perfection**: mean_CE/ep = 0.000 across all 4 cells × all 4 seeds; mean_tox = 0.000 likewise. The reward is well-formed and the policy learns to satisfy it.
* The permuted-Chronos PPO does NOT achieve this safety profile (mean_CE/ep = 0.022–0.032), confirming the **label structure is load-bearing for the reward-fit metric**, even though it isn't load-bearing for raw success.

### 8.3 Held claims (Bucket B — reward-independent)

* "PPO_C raw success is **bounded** by PPO_A − 0.05 at K=2 / bin 6-8 / OOD" — held with margin (0.031 ≤ 0.05).
* "At the saturated K=3 / bin 8-10 / OOD primary cell, PPO_C costs **6 pp of raw success** vs all baselines" — held, deterministic across seeds.

### 8.4 Bucket-C status

`HELDOUT_INCONCLUSIVE_NO_GENERALIZATION_DETECTED`. The DepMap safety prior does not transfer to Replogle essentiality. This is a partial **negative** finding (with caveats — CIs straddle zero, magnitudes are small).

## 9. Recommended next action

The user's spec offers three options:
* (a) Tune λ_tox / λ_ce to recover the Bucket-B signal
* (b) Proceed to Phase 3 path-length B before combining C+B
* (c) Halt / revise biology objective

**My recommendation: (b) Proceed to Phase 3 path-length B**, with the following caveats:

1. **Variant C is NOT the V3B headline**. Carry it forward as "validated reward-prior optimization" (Bucket-A clean) but not as a planning-advantage claim.
2. **Do NOT spend further compute tuning λ_tox / λ_ce alone**. The seed variance dominates the Bucket-B signal; tuning λ within the seed-noise window is unlikely to flip the verdict cleanly. If we want a clean reward-fit-vs-Bucket-B tradeoff curve, we can sweep λ in conjunction with Phase 3, not standalone.
3. **Phase 3 (path-length free-band B) is the right next test** because it does NOT depend on the biology layer. The reward-independent question becomes: "does PPO with `g(t)` schedule + extended horizon K=8 find longer plans that depth-5 beam search misses?". Independent of biology.
4. **Plan the Phase 5 (C + B conjunction) carefully**. If Phase 3 shows clean Bucket-B signal, combining with Variant C as a secondary axis is fine — but the headline must come from Phase 3, with Variant C contributing the Bucket-A safety guarantee.

### 9.1 Why I am NOT recommending (a) λ tuning

The 4-seed std on PPO_C at K=2 / bin 8-10 / OOD raw success is 0.057, while PPO_A's std at the same cell across V2's 4 seeds is 0.045 (from V2_FINAL_REPORT.md §3). λ tuning changes the *mean* and possibly the *variance ratio*, but with seed-CI width ~13 pp (1.96 × 0.057 × 2), recovering a +4 pp mean signal would require a 4× variance reduction — not a realistic outcome of λ rescaling.

### 9.2 Why I am NOT recommending (c) halt

Bucket-A is solid; the safety prior optimizes well; Variant C is a known-good *constraint* even if not a *planning advantage*. Carrying it forward into Phase 5 as a constraint is valuable. The biorealistic-control hypothesis is alive — just not via Variant C alone.

### 9.3 Ideal alternative consolidated plan

**Phase 3 (single-axis B = path-length free-band)** as the next standalone test:
* Build `path_length_freeband` reward mode in `src/rl/biology_rewards.py` (no biology required).
* Train PPO_B at seed 42 first (smoke); if directional signal at K∈{4, 5} cells, escalate to 4 seeds.
* Apply 4-seed CI acceptance from the start (no more single-seed headlines).

**Then Phase 5 (C + B conjunction)** with the 4-seed protocol built in. Compare against greedy_dyn_5_{C+B}.

**Defer Phase 4 (D = uncertainty) and Phase 6 (Track N 64D)** until after Phase 3/5 — uncertainty needs Bucket-B leverage somewhere first; Track N replication needs a winning axis-B variant.

## 10. Sacred-rule conformance

* `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` clean (only pre-existing untracked entries).
* Phase 2 PPO checkpoints (`artifacts_v3/rl_v3b_safety_aware_v2primary_seed42/`) **untouched**; new naming `rl_v3b_safety_aware_seed42*` are **symlinks** to the existing seed-42 dirs.
* Phase 2 eval (`artifacts_v3/eval_v3b_phase2/`) untouched.
* Phase 2 interpretation (`artifacts_v3/interpretation/v3b_phase2_interpretation.md`) untouched.
* Phase 2b audit (`artifacts_v3/eval_v3b_phase2b/`, `artifacts_v3/interpretation/v3b_phase2b_leakage_split_audit.md`) untouched.
* Only code edit: `scripts/train_rl_v3b.py` line ~46 — changed naming template from `rl_v3b_safety_aware_v2primary_seed{N}` to `rl_v3b_safety_aware_seed{N}` per user spec. Old paths still work via the symlinks.
* New code: `scripts/aggregate_v3b_phase2c.py` (4-seed aggregator + acceptance check).
* Test suite: 305 passed / 2 skipped, no regressions (no test changes).

## 11. Files produced this session

```
artifacts_v3/rl_v3b_safety_aware_seed0/                   # NEW: real-Chronos seed 0
artifacts_v3/rl_v3b_safety_aware_seed1/                   # NEW: real-Chronos seed 1
artifacts_v3/rl_v3b_safety_aware_seed7/                   # NEW: real-Chronos seed 7
artifacts_v3/rl_v3b_safety_aware_seed0_permuted_chronos/  # NEW: permuted seed 0
artifacts_v3/rl_v3b_safety_aware_seed1_permuted_chronos/  # NEW: permuted seed 1
artifacts_v3/rl_v3b_safety_aware_seed7_permuted_chronos/  # NEW: permuted seed 7
artifacts_v3/rl_v3b_safety_aware_seed42*                  # SYMLINK to Phase 2 dirs

artifacts_v3/eval_v3b_phase2c/
├── seed42/  seed0/  seed1/  seed7/        # per-seed eval outputs
├── seed_escalation_results.csv             # 112 rows: per (seed, cell, policy) raw + bucket-A + bucket-B
├── seed_escalation_results.json            # 4-seed CIs + acceptance + verdict
├── seed_escalation_summary.md              # human-readable
├── replogle_norman_intersection.json       # Bucket-C gene-set intersection
├── replogle_heldout_per_seed.csv           # 80 rows: held-out frac per (seed, cell, policy)
├── replogle_heldout_paired_deltas.json     # held-out 4-seed CIs
└── replogle_heldout_summary.md             # Bucket-C interpretation

artifacts_v3/interpretation/
└── v3b_phase2c_seed_escalation.md          # (this file)

scripts/
├── train_rl_v3b.py                          # MODIFIED (naming template, 4 lines)
└── aggregate_v3b_phase2c.py                 # NEW
```

Bytes added to disk (mostly PPO checkpoints): ~5 MB.

## 12. Final summary table — Phase 2 → Phase 2b → Phase 2c

| | Phase 2 (single seed 42) | Phase 2b (audit, no retrain) | Phase 2c (4-seed) |
|---|---|---|---|
| Bucket-A (reward-fit) | ✅ Zero CE/ep, low tox | ✅ Reframed as expected | ✅ Solid across all 4 seeds |
| Bucket-B at K=2 / bin 8-10 / OOD raw | **+4 pp** (single seed) | Single-seed within V2 noise band | **+0.9 pp [−4.6, +6.5]** — CI straddles 0 |
| Real vs permuted at headline cell raw | +7.3 pp (single seed) | Single-seed | **−2.2 pp [−13.0, +8.6]** — direction flipped on 2/4 seeds |
| Bucket-B at K=2 / bin 6-8 / OOD raw | −3.7 pp (single seed) | Counterexample to headline | **−3.1 pp [−5.0, −1.2]** — significant regression |
| Bucket-C (Replogle) | Pending | Pending | `HELDOUT_INCONCLUSIVE_NO_GENERALIZATION_DETECTED` |
| Phase 2 verdict | `ACCEPT` (5/5 rules) | `PHASE2_VALID_REWARD_FIT_NEEDS_SEED_ESCALATION` | **`PHASE2C_PROMISING_BUT_UNSTABLE_SEED_VARIANCE_DOMINATES`** |
| Phase 3 unlocked? | Implied yes | Pending seed escalation | **YES** (but Variant C is NOT the headline) |

The progression is the audit-and-escalation pipeline doing its job: a flashy single-seed result was inspected, escalated, and downgraded to its honest level. Phase 3 path-length B is the next reward-independent test. Variant C remains as a known-good Bucket-A constraint, not a V3B planning-advantage headline.
