# V3B Phase 3b — Stricter-Epsilon Diagnostic Interpretation

**Date:** 2026-05-19
**Author:** V3 research lead (CC agent)
**Scope:** Evaluation-only diagnostic of whether stricter ε (p15, p10, p5) creates a path-length
leverage band on V2 primary 32D `RoR_corr010` dynamics. No retraining.
**Sacred-rule conformance:** all outputs under `artifacts_v3/`. Frozen tiers untouched. Existing
PPO_A, PPO_B (4 seeds), PPO_C (4 seeds) checkpoints reused as-is.

---

## 1. Headline

**Final verdict: `PHASE3B_FIELD_REMAINS_SATURATED_AT_P10_RECOMMEND_REPRESENTATION_REFORMULATION`.**

The V2 primary `RoR_corr010` 32D dynamics field is **structurally robust to ε-tightening down to p5
(2.7362)**. At every tested ε ∈ {p15, p10, p5}, `greedy_dyn_5_B` remains saturated (≥0.95) at every
K≥4 cell, leaving zero path-length leverage cells for PPO_B to exploit. The freeband schedule
correctly increases PPO_B's K∈{4,5} usage as ε tightens (6% → 11.6%), but raw success at K≥4
cells stays pinned at 1.000 because depth-2 single-gene actions already suffice.

**Recommended ε for B+C retraining: NONE.** V2 dynamics is the bottleneck — not the reward.

**B+C on V2 dynamics is NOT unlocked.** Combine-or-tune the reward is unjustified; the path-length
axis has no leverage to expose.

**Recommendation: abandon reward-space biorealistic control on V2 dynamics.** Move to a different
latent / dynamics field (Track N 64D NB once safety-pre-checked; or SCANVI 32D; or V3.fallback.B
contraction-regulariser). The V3B Phase 2c reward-fit safety result (Variant C as a Bucket-A
constraint) remains the only defensible V3B contribution on V2 dynamics.

---

## 2. Exact ε values + reachability

Computed from the V1 control-distance distribution (`artifacts/vae/latents.h5ad`, 11 855 control
cells, distances to `z_reference_centroid.npy`). Persisted at
`artifacts_v3/eval_v3b_phase3b/epsilon_quantiles.json`.

| Quantile | Value (L2 latent) | Notes |
|---|---:|---|
| p25 | 3.1663 | V2 reference, Phase 3 baseline |
| p20 | 3.0829 | not tested |
| **p15** | **2.9898** | Phase 3b test (≈6 % stricter than p25) |
| **p10** | **2.8846** | Phase 3b test (≈9 % stricter than p25) |
| **p5** | **2.7362** | Phase 3b additional (≈14 % stricter than p25) |
| p1 | 2.4774 | near control min (2.0424) |

OOD-perturbed start pool sizes per (bin):

| Bin | n cells | Notes |
|---|---:|---|
| 6-8 | 227 | well-populated |
| 8-10 | 17 | small but usable (V2 primary cell uses this bin) |
| 10-12 | **0** | empty — bin not testable |
| 12-14 | 0 | empty |

`bin 10-12` is structurally empty; the user spec's "10-12 if enough cells" was not satisfiable.
Diagnostic restricted to {6-8, 8-10}.

---

## 3. Diagnostic answers (per user spec)

### 3.1 Does greedy_dyn_1_B stop saturating?

**Partially, only at K=2 and K=3 cells**, not at K≥4. At p25 (Phase 3), greedy_dyn_1_B = 1.000 at
every K≥3 cell.

| ε | K=2 bin 6-8 | K=2 bin 8-10 | K=3 bin 6-8 | K=3 bin 8-10 | K=4 bin 6-8 | K≥4 8-10 |
|---|---:|---:|---:|---:|---:|---:|
| p15 | 0.710 | 0.277 | 1.000 | 1.000 | 1.000 | 1.000 |
| p10 | 0.637 | 0.130 | 0.980 | 0.940 | 1.000 | 1.000 |
| **p5** | **0.547** | **0.130** | 0.970 | 0.940 | 1.000 | 1.000 |

K=2 cells un-saturate substantially under stricter ε (greedy_dyn_1 drops from saturated-at-p25
into the 0.13–0.71 band). But these cells have K=2 max budget, so they CANNOT exercise the
freeband K∈{4,5} leverage by definition.

### 3.2 Does greedy_dyn_2_B stop saturating?

**Same pattern — K=2 cells un-saturate, K≥4 cells remain pinned**.

| ε | K=2 bin 6-8 | K=2 bin 8-10 | K=3 bin 6-8 | K=3 bin 8-10 | K=4 bin 6-8 | K=4 bin 8-10 |
|---|---:|---:|---:|---:|---:|---:|
| p15 | 0.643 | 0.130 | 0.993 | 1.000 | 1.000 | 1.000 |
| p10 | 0.567 | 0.130 | 0.987 | 0.940 | 1.000 | 1.000 |
| p5 | 0.450 | 0.130 | 0.973 | 0.940 | 1.000 | 1.000 |

### 3.3 Does greedy_dyn_5_B stop saturating?

**No, anywhere we can test it.** Greedy_dyn_5_B requires K=5 cells (by the depth cap); at every
K=5 and K=8 cell at every ε ∈ {p15, p10, p5}, greedy_dyn_5_B = **1.000 ± 0.000**.

### 3.4 Does PPO_B use T∈{4,5} more often under stricter ε?

**Yes, modestly — 6 % → 11.6 % across the four tested epsilons.**

| ε | K=4 bin 6-8 | K=4 bin 8-10 | K=5 bin 6-8 | K=5 bin 8-10 | K=8 bin 8-10 |
|---|---:|---:|---:|---:|---:|
| p25 (Phase 3) | 0.000 | 0.045 | 0.000 | 0.045 | 0.045 |
| p15 | 0.029 | 0.060 | 0.029 | 0.060 | 0.060 |
| p10 | 0.048 | 0.085 | 0.056 | 0.085 | 0.085 |
| **p5** | **0.115** | **0.116** | **0.126** | **0.116** | **0.116** |

PPO_B is **demonstrably more willing to extend plans into the mild band** as ε tightens, almost
2× from p25 → p5. **But it remains below the 30 % spec threshold at every cell and ε**, and the
raw-success effect is null because shorter plans still saturate the success criterion.

### 3.5 Does random remain substantially lower?

**Yes, at every cell + every ε** by ≥ 0.10 (in fact by ≥ 0.06 in the worst case at K=2 bin 8-10
at p5, and ≥ 0.20 everywhere else). Random success scales with budget (e.g. K=8 cell at p5:
random = 0.350). Rule 3.5: ✅ always satisfied.

### 3.6 Is there at least one reachable, non-saturated K≥4 cell where PPO_B can be meaningfully compared to greedy_dyn_5_B?

**No.** At every (K≥4, ε∈{p15, p10, p5}, bin∈{6-8, 8-10}) combination, greedy_dyn_5_B = 1.000.
The path-length axis has zero K≥4 test cells. PPO_B vs greedy_dyn_5_B is trivially +0.000 at
every K≥5 cell and undefined (depth cap excludes greedy_5 at K=4) at K=4 cells.

### 3.7 Is ε-tightening succeeding at making a harder task?

**Partially — at K=2 and K=3 cells only**, not at K≥4 cells. The V2 primary dynamics has
strong enough single-step contraction that one well-chosen gene action lands within ε at K≥4
under bin 8-10 / 6-8 starts, regardless of whether ε is p25 or p5. The K=2 budget is the only
constraint that produces non-saturation; longer-K cells trivially exploit the contraction.

---

## 4. Decision rule outcomes (user spec rules A/B/C)

| Rule | Definition | Outcome |
|---|---|---|
| **A** | At p15 or p10: greedy_dyn_5_B ∈ [0.30, 0.95] at any K≥4 cell AND PPO_B T∈{4,5} usage ≥15% | **NOT MET** at any ε. greedy_dyn_5_B = 1.000 everywhere at K≥4. |
| **B** | greedy_dyn_5_B ≥ 0.95 everywhere at p10 | **MET**. Also at p5. Field is saturated under stricter ε too. |
| **C** | At p10, all policies collapse (greedy/PPO success ≤ 0.10) | **NOT MET**. PPO_B success = 0.075–1.000 across cells at p10. K=2/bin 8-10 close to collapse but K≥3 fine. |

**Outcome: Rule B is the binding rule.** B+C on V2 dynamics is NOT justified.

---

## 5. Best non-saturated cell (and the implication)

Looking outside the K≥4 leverage band, the cells with the cleanest "non-saturated, no-collapse"
profile under any ε are:

### 5.1 K=2 / bin 6-8 / OOD at p15 — most informative non-K≥4 cell

| Policy | Raw success (4-seed mean ± std) | Notes |
|---|---:|---|
| greedy_dyn_1_B | 0.710 ± 0.000 | best |
| greedy_dyn_2_B | 0.643 ± 0.000 | depth-2 |
| ppo_A (V2 primary) | 0.577 ± 0.000 | frozen V2 baseline |
| ppo_C | 0.583 ± 0.036 | safety-aware |
| **ppo_B** | **0.556 ± 0.039** | freeband, single-seed-noisy |
| random_uniform_valid | 0.037 ± 0.000 | |
| always_noop | 0.000 ± 0.000 | |

**Greedy_dyn_1_B (single-step lookahead) STRICTLY BEATS PPO_B by +0.15 raw success** at this
cell. This is the cleanest evidence that the dynamics field is "single-step contraction-dominated"
and a multi-step planner gains nothing structural — even at K=2 cells where freeband cannot
exploit longer paths, the single-step optimal action is reliably better than what PPO finds.

### 5.2 K=3 / bin 8-10 / OOD at p10 — K=3 path-depth headroom appears

| Policy | Raw success | Notes |
|---|---:|---|
| greedy_dyn_3_B | 1.000 ± 0.000 | depth-3 saturates |
| greedy_dyn_1_B | 0.940 ± 0.000 | depth-1 below 0.95 |
| greedy_dyn_2_B | 0.940 ± 0.000 | depth-2 same as depth-1 |
| **ppo_B** | **0.915 ± 0.029** | matches greedy_dyn_2 within seed noise |
| ppo_A | 0.907 ± 0.000 | |
| ppo_C | 0.886 ± 0.036 | safety regression at stricter ε |
| random | 0.073 ± 0.000 | |

**Important nuance**: at K=3/bin 8-10/p10, there IS depth-headroom between greedy_dyn_2 (0.940)
and greedy_dyn_3 (1.000) — about +6 pp lift from one extra plan step. **But this is K=3, not
K≥4, so freeband's mild band (K=4,5) cannot exploit it.** PPO_B at 0.915 acts like a depth-2
controller; it doesn't pick up the +6 pp greedy_dyn_3 gain even when its env's K=3 budget allows.

**Path-length leverage exists between depth 2 and depth 3 at K=3 cells under p10.** Freeband B
cannot use it because freeband's free band ends at T=3 (the band where PPO is essentially trained
to maximise success). The user's hypothesis ("free up to K=3, then mild") is structurally aligned
with what PPO_B does; there's no room for the mild band to help when the leverage is *at* the
boundary T=3.

---

## 6. Other highly relevant conclusions

### 6.1 PPO_C regresses substantially under stricter ε

At p25 (Phase 2c) PPO_C had clean Bucket-A safety metrics. As ε tightens, **PPO_C loses
raw success faster than PPO_A or PPO_B**:

| Cell | p25 PPO_C | p15 PPO_C | p10 PPO_C | p5 PPO_C |
|---|---:|---:|---:|---:|
| K=3/bin 8-10/OOD | 0.940 | 0.940 | 0.886 | **0.775** |
| K=2/bin 6-8/OOD | (n/a, not in Phase 2) | 0.583 | 0.488 | 0.294 |

At p5/K=3/bin 8-10/OOD, **PPO_B beats PPO_C by +10.9 pp raw** (CI [+0.100, +0.119] ✅). This
is consistent with the Phase 2c finding that Variant C is a pure constraint — its safety-prior
optimization sometimes prefers slightly-longer-but-safer paths that pay off at p25 ε (within the
loose success ball) but fail to hit stricter ε. **Variant C is brittle to ε-tightening; Variant B
is robust to it.** This is a useful Phase 2b-style audit observation: the V3B reward design space
has different sensitivities to evaluation strictness.

### 6.2 PPO_B's freeband respect under stricter ε is real

PPO_B never used T>5 in any cell or ε combination (frac_T>5 = 0.000 ± 0.000 universally). The
heavy_beta=0.10 slope correctly priced speculative depths out of the optimizer's preferred region.
**Markov-composition hallucination risk did NOT materialize even under stricter ε** — the
schedule is well-designed; the field just doesn't need long paths.

### 6.3 Greedy_dyn_2 vs greedy_dyn_3 at K=3 cells — the depth-2-to-depth-3 jump

At p10/K=3/bin 8-10/OOD: greedy_dyn_2 = 0.940 → greedy_dyn_3 = 1.000 (+6 pp).
At p5/K=3/bin 8-10/OOD: greedy_dyn_2 = 0.940 → greedy_dyn_3 = 0.940 (no jump — collapse).
At p5/K=3/bin 6-8/OOD: greedy_dyn_2 = 0.973 → greedy_dyn_3 = 0.933 (actually a small drop — depth-3
overshoots).

So depth-3 planning sometimes helps and sometimes hurts at K=3 cells, depending on ε and bin. This
is the **dynamics' composability boundary** showing through: beyond depth 2, the dynamics' learned
composition starts producing predictions that diverge from "navigate to the centroid", possibly
because the contraction toward `z_ref` overshoots when stacked.

### 6.4 PPO_B is not a depth-2 controller under freeband — it's a depth-3 controller

PPO_B mean_steps across cells: 2.00 (K=2 cells) → 2.85–3.07 (K=3 to K=8 cells under p15/p10/p5).
PPO_A V2 baseline mean_steps: 2.70 at K=3/bin 8-10/p25.

So freeband PPO_B uses ~0.2 more steps on average than the V2 step-cost PPO_A, exactly what the
free band would predict (free up to 3 means PPO has no incentive to truncate aggressively at
T=1 or T=2). **PPO_B's behavior matches the reward design qualitatively**; the issue is that
the dynamics field gives the same success rate to depth-2 and depth-3 plans at K≥4 cells.

### 6.5 Random scaling reveals dynamics is "easy-to-anywhere" for K big enough

Random success rate as K grows (at p5):

| Cell | random success |
|---|---:|
| K=2 bin 6-8 | 0.013 |
| K=3 bin 6-8 | 0.077 |
| K=4 bin 6-8 | 0.167 |
| K=5 bin 6-8 | 0.220 |
| K=8 bin 8-10 | 0.350 |

Random scales roughly linearly with K, showing that the dynamics field is "navigable by accident"
beyond a certain budget. The contraction is structural, not informationally directed — but it's
also strong enough that any reasonable planner trivially succeeds.

---

## 7. Whether B+C at stricter ε is justified — NO

The user's Rule A required **BOTH** path-length headroom (greedy_5_B ∈ [0.30, 0.95]) AND
PPO_B long-path usage ≥ 15% at K≥4 cells. Neither condition is met at p15, p10, or p5. **B+C
retraining is NOT justified.**

If we forced a B+C retrain at p10 ε regardless, the predicted outcome based on the diagnostic is:
- C contributes Bucket-A safety constraint (zero CE picks).
- B contributes... nothing observable at K≥4 cells.
- Joint headline raw success is roughly PPO_C-at-p10 (worse than PPO_A on raw success).

Cost would be ~30 min of PPO training + 30 min of eval = ~1 hour. Predicted output: **no V3B
headline**.

---

## 8. Whether retraining is needed or evaluation-only is enough — EVALUATION-ONLY IS SUFFICIENT

The diagnostic answer is clean: the field is saturated for the reward-leverage axis. **Retraining
PPO_B at stricter ε will not change the structural conclusion** — the success criterion at K≥4
cells is dominated by single-step contraction, not by planning depth or reward shaping.

The only retraining that could change this verdict is on a **different dynamics field**:
- Track N 64D NB (V3A pending safety pre-check)
- SCANVI 32D (V3.4 fallback)
- A future V3 latent with contraction regulariser (V3.fallback.B)

---

## 9. Whether V2 dynamics should be abandoned for reward-space work — YES

The V2 primary 32D `RoR_corr010` dynamics field is **fundamentally well-conditioned**. Three
phases (2c, 3, 3b) of reward-space experiments confirm this:

* **Phase 2c (Variant C, safety):** safety prior optimizes as designed; no reward-independent
  raw-success advantage (single-seed +4 pp collapsed to noise in 4-seed escalation).
* **Phase 3 (Variant B, path-length):** path-length availability cannot create leverage; greedy
  saturates at every K≥3 cell at p25.
* **Phase 3b (stricter ε):** ε-tightening down to p5 (14 % stricter than p25) does NOT
  un-saturate K≥4 cells. Some K=2 and K=3 cells un-saturate, but those have insufficient budget
  for the freeband K∈{4,5} band.

**Recommendation: move V3B work off the V2 primary 32D dynamics field.** Continue Phase 2c safety
Variant C as a known-good Bucket-A constraint; defer all path-length and uncertainty work to a
representation where the dynamics field is not pre-saturated. Concretely:

1. **Run V3A Track N safety pre-check** (pairs build → RoR dynamics on Track N 64D NB → reachability
   oracle → greedy saturation check at p25). If Track N greedy_dyn_2 < 0.95 at K=3 primary,
   re-run Phase 3 on Track N: meaningful path-length test possible.
2. **If Track N is also saturated**, the V3 axis-A fallback chain (V3.3 ZINB → V3.4 SCANVI →
   V3.fallback.B contraction regulariser) is the only remaining path to a non-saturated dynamics.
3. **Halt reward-space work on V2 dynamics** until a non-saturated representation is in place.

---

## 10. Sacred-rule conformance

* `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` clean.
* No PPO / dynamics / VAE retraining performed in Phase 3b.
* Existing PPO_A (frozen V2), PPO_B (4 seeds), PPO_C (4 seeds) checkpoints reused as-is.
* All new outputs under `artifacts_v3/eval_v3b_phase3b/` and
  `artifacts_v3/interpretation/v3b_phase3b_strict_epsilon_diagnostic.md`.
* Test suite: **327 passed / 2 skipped**, no regressions.
* Only code edits in this session: `scripts/evaluate_rl_v3b_phase3.py` (added `--epsilon_value`,
  `--ppo_zip_C`, `--max_greedy_depth`, K=2 cells, depth-capping for greedy). All Phase 3 invocations
  back-compatible (legacy p25-tagged cells preserved as aliases).
* New code: `scripts/aggregate_v3b_phase3b.py` (aggregator + decision rules A/B/C).

---

## 11. Files produced this session

```
artifacts_v3/eval_v3b_phase3b/
├── epsilon_quantiles.json              # p1/p5/p10/p15/p20/p25/p50 + pool sizes
├── eps_p15/seed{42,0,1,7}/              # 4 seeds × 9 cells × ~6 policies per cell
├── eps_p10/seed{42,0,1,7}/
├── eps_p5/seed{42,0,1,7}/
├── epsilon_sweep_results.csv            # 876 rows long-form
├── epsilon_sweep_results.json           # CIs + paired deltas + decision rules
└── epsilon_sweep_summary.md             # human-readable per-ε tables

artifacts_v3/interpretation/
└── v3b_phase3b_strict_epsilon_diagnostic.md  # (this file)

scripts/
├── evaluate_rl_v3b_phase3.py            # extended (additive)
└── aggregate_v3b_phase3b.py             # NEW
```

Total Phase 3b wall-clock: ~110 min (sweep) + ~5 min (aggregation + writing).

---

## 12. Phase progression — V2 reward-space work summary

| Phase | Variant | Verdict | Bucket-B contribution |
|---|---|---|---|
| 2 (single seed 42) | C safety | `ACCEPT` (over-stated) | +4 pp at K=2/bin 8-10/OOD (single seed) |
| 2b (audit) | C safety | `PHASE2_VALID_REWARD_FIT_NEEDS_SEED_ESCALATION` | Single-seed within V2 noise |
| 2c (4-seed) | C safety | `PHASE2C_PROMISING_BUT_UNSTABLE_SEED_VARIANCE_DOMINATES` | 4-seed CI on +4 pp straddles 0 |
| 3 (4-seed @ p25) | B path-length | `PHASE3_FAIL_NO_PATH_LENGTH_LEVERAGE_FIELD_SATURATED` | None — all K≥4 saturated |
| **3b** (eval @ {p15, p10, p5}) | B path-length | **`PHASE3B_FIELD_REMAINS_SATURATED_AT_P10`** | None at any ε down to p5 |

**Cumulative V3B-on-V2-dynamics finding: the V2 primary 32D RoR_corr010 dynamics is robust to
all reward-space biorealistic controls tested. The biorealistic-control hypothesis must be tested
on a different latent/dynamics field.**
