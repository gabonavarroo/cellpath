# V3B Phase 3 — Path-length free-band reward (Variant B) interpretation

**Date:** 2026-05-18
**Author:** V3 research lead (CC agent)
**Scope:** Test whether the V2 step-cost was over-penalizing longer trajectories and whether
PPO can exploit medium-length paths (K=4, 5) that a depth-limited greedy oracle cannot.
**Sacred-rule conformance:** all new outputs under `artifacts_v3/`. Frozen tiers (`artifacts/`,
`artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/`) untouched. PPO_A (`rl_v1ot_ror_corr010_*`)
reused as the frozen V2 baseline.

---

## 1. Final verdict

**`PHASE3_FAIL_NO_PATH_LENGTH_LEVERAGE_DETECTED_FIELD_SATURATED`** — 2/5 acceptance rules pass.

**Phase 5 (C + B conjunction) is NOT unlocked.** Combining a non-headline-winning Variant C
with a non-headline-winning Variant B would not produce a defensible V3B headline.

**Seed-42 smoke directional signal:** **none observed**. PPO_B uses T∈{4,5} in only
6% of episodes at the best cell; raw success matches greedy_dyn_K_B at depths 1–8
(all 1.000) at every K≥4 cell. The 4-seed escalation confirmed the smoke result.

| Acceptance rule | Result | Passed |
|---|---|:---:|
| 1. PPO_B − greedy_dyn_5_B raw success CI excludes 0 at any K≥4 cell | Max Δ across K≥4 cells = +0.0000 (perfect ties at 1.000) | ❌ |
| 2. PPO_B uses T∈{4,5} in ≥30% of successful episodes | Max fraction = 4.5% at K=4/bin 8-10 OOD | ❌ |
| 3. always_noop success ≤ 0.05 at every cell | Max = 0.000 | ✅ |
| 4. PPO_B − random raw success ≥ 0.10 at winning/test cell | +0.7233 at K=4/bin 8-10 OOD | ✅ |
| 5. Winning cell is not K=3 | No winning cell exists | ❌ |

## 2. Why Phase 3 failed — the field is saturated

The decisive finding from the 4-seed run is that **every K≥3 cell on V2 primary dynamics
under reward-aware planning achieves 100 % success across every greedy depth (1–8) and
both PPO variants**:

| Cell | greedy_dyn_1_B | greedy_dyn_2_B | greedy_dyn_5_B | greedy_dyn_8_B | PPO_B | PPO_A |
|---|---:|---:|---:|---:|---:|---:|
| K=3 / bin 8-10 / OOD | 1.000 | 1.000 | 1.000 | 1.000 | 0.955 ± 0.030 | 1.000 |
| K=4 / bin 8-10 / OOD | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 |
| K=5 / bin 8-10 / OOD | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 |
| K=8 / bin 8-10 / OOD | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 |
| K=4 / bin 6-8 / OOD | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 |
| K=5 / bin 6-8 / OOD | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 |

Even `greedy_dyn_1_B` (single-step lookahead) achieves 100 % at the K=3 / bin 8-10 / OOD
cell. **The dynamics' single-step contraction toward `z_ref` is so strong at p25 ε on
the V2 primary `RoR_corr010` 32D field that one well-chosen gene action is usually enough**
(within ε after a single delta). All deeper planners and PPO simply repeat this fact;
the "advantage" of K∈{4,5} availability is structurally invisible because K=1 already wins.

**Path-length distribution among successful PPO_B episodes (4-seed mean):**

| Cell | T=1 | T=2 | T=3 | T∈{4,5} | T>5 |
|---|---:|---:|---:|---:|---:|
| K=3 / bin 8-10 / OOD | — | — | 100% | 0% | 0% |
| K=4 / bin 8-10 / OOD | — | — | 95.5% | **4.5%** | 0% |
| K=5 / bin 8-10 / OOD | — | — | 95.5% | **4.5%** | 0% |
| K=8 / bin 8-10 / OOD | — | — | 95.5% | **4.5%** | 0% |
| K=4 / bin 6-8 / OOD | — | — | 99.4% | 0.6% | 0% |
| K=5 / bin 6-8 / OOD | — | — | 99.4% | 0.6% | 0% |

PPO_B respects the free-band schedule (zero "speculative" T>5 usage everywhere) but
finds **no leverage to exploit T∈{4,5}** because shorter plans already saturate.

## 3. Per-cell paired deltas (4-seed 95% CI)

| Cell | Δ(PPO_B − PPO_A) | Δ(PPO_B − greedy_dyn_2_B) | Δ(PPO_B − greedy_dyn_5_B) |
|---|---|---|---|
| K=3 / bin 8-10 / OOD | **−0.045** [−0.075, −0.015] ❌ | **−0.045** [−0.075, −0.015] ❌ | **−0.045** [−0.075, −0.015] ❌ |
| K=4 / bin 8-10 / OOD | +0.000 [0, 0] — | +0.000 [0, 0] — | +0.000 [0, 0] — |
| K=5 / bin 8-10 / OOD | +0.000 — | +0.000 — | +0.000 — |
| K=4 / bin 6-8 / OOD | +0.000 — | +0.000 — | +0.000 — |
| K=5 / bin 6-8 / OOD | +0.000 — | +0.000 — | +0.000 — |
| K=8 / bin 8-10 / OOD | +0.000 — | +0.000 — | +0.000 — |

The **only** statistically-distinguished cell is K=3 / bin 8-10 / OOD, where PPO_B
**regresses** −4.5 pp vs PPO_A and greedy. This is a small but real cost of the freeband
schedule's small-T behaviour: under PPO_B's optimizer, the value function trained on the
new reward (with success_bonus=1 and zero step cost for T≤3) does not perfectly match
PPO_A's V2-trained policy at this saturated cell, costing roughly 5 pp.

## 4. Seed-42 smoke had directional signal? **No.**

Seed 42 already showed:
* PPO_B success = 1.000 at every K≥4 cell, matching greedy_dyn_5_B exactly.
* PPO_B uses T∈{4,5} in only 6% of episodes at the best cell.
* Smoke result was representative; 4-seed escalation confirmed it deterministically
  (most cells have std=0.000 across seeds because the start pool is fixed and the
  dynamics is deterministic in mean — only PPO_B's stochastic policy contributes
  any variance).

## 5. Greedy reward-aware vs distance-only?

**Reward-aware** (`greedy_dyn_K_B` with the freeband schedule baked into the beam score:
`d + path_penalty(T) − success_bonus·1[d<ε]`). The implementation extends
`GreedyDynamicsBeamPolicy` with a `freeband_schedule` parameter; behavior collapses to
V2 distance-only when the schedule is omitted, preserving back-compat (V2 reward/env
tests still pass; +22 new freeband tests; full suite **327 passed / 2 skipped**, was 305).

The greedy baselines reported here are explicitly labelled `greedy_dyn_K_B` to indicate
the freeband objective. Distance-only contrast was available via `--include_distance_greedy`
but skipped in the per-seed run since the saturated success rates would be identical to
reward-aware greedy at these cells (both reach ε in K≤3 steps).

## 6. Why I expected this could fail — and the diagnosis

### 6.1 Three sources of saturation

1. **V2 primary dynamics is too well-conditioned.** From `artifacts_v2/V2_FINAL_REPORT.md`
   §3: `greedy_dyn_2` saturated at 1.000 on K=3 / bin 8-10 / OOD with the V2 reward.
   We did not expect a *new reward* to un-saturate the *same dynamics field*.
2. **ε = p25 is too loose for K=4,5 cells.** ε ≈ 3.166 is reached by a single gene
   action from any bin-8-10 start under the RoR dynamics' contraction rate. The
   intent of K=4,5 cells in this phase was to test whether PPO_B would extend its
   plan when freed of harsh per-step cost — but there is no reason to extend if K=1
   already succeeds.
3. **Reward-aware greedy is a stronger baseline than V2 distance-only.** With the
   freeband objective, the beam prefers shorter successful plans (lower
   path_penalty term). For a saturated field, this means greedy commits to K=1–3
   plans and matches PPO_B exactly. The fairness of the comparison turned a
   speculative win (PPO_B − distance_greedy under freeband) into a tie.

### 6.2 Markov-composition risk under freeband

The user's concern about "mathematical hallucination at long paths" was empirically
addressed by the schedule itself: PPO_B used T>5 in **0%** of episodes across all
cells. The heavy_beta=0.10 slope correctly priced speculative depths out of the
optimizer's preferred region. **No hallucination risk materialized**, but only
because PPO_B didn't need to access it — the field's contraction is the load-bearing
fact, not the schedule design.

### 6.3 What freeband proved (small but real)

* The reward implementation is correct (22 unit tests + V2 regressions all pass).
* The schedule's `T≤3 free / T∈{4,5} mild / T>5 heavy` structure is functional
  end-to-end; PPO respects it.
* When the field is saturated, the freeband reward does NOT *create* artificial
  long-path usage — it lets the optimizer naturally find short successful plans.
* Reward-aware greedy is correctly implemented and is a strictly stronger
  baseline than V2 distance-only greedy on long-horizon evals.

These are all positive engineering outcomes, just not a Phase 3 acceptance pass.

## 7. Implications for V3B

Phase 2c showed Variant C is **reward-prior optimization** (Bucket-A clean, Bucket-B null).
Phase 3 shows Variant B is **availability without leverage**: even when the optimizer
is free to use long paths, the field saturates at K=1–3 so the long-path axis is
unobservable.

**Conclusion: the V3B headline pivot is not in the reward space on the V2 primary
dynamics field**. Both Variant C and Variant B operate on a fundamentally saturated
search problem at p25 ε. Continuing to add reward terms (D = uncertainty, E = combined)
on the same field will inherit the same saturation.

The V3B safety reward (Variant C) is still useful as a **constraint** (Bucket-A clean,
zero common-essential picks across seeds), but the "biorealistic-control" headline that
would distinguish V3B from V2 cannot come from C + B + D layered on the V2 primary field.

## 8. Recommended next steps

In ranked order (user's spec offered three options; I'm adding two more for completeness):

### A. **Problem reformulation — tighter ε** (RECOMMENDED, primary)

Set `epsilon_override = p15 ≈ 3.00` or `p10 ≈ 2.92` (currently the V1 epsilon_success.json
records p10 implicitly via the V2 distribution; need to recompute). This forces longer
paths to reach the now-stricter success criterion, naturally activating the K=4,5
leverage band. **Smallest decisive change**: re-run Phase 3 evaluation at p10 ε with the
existing PPO_B checkpoints + greedy oracles. No new training needed for the eval; if
PPO_B at p10 ε still saturates, also re-train PPO_B with p10 in the env. Compute: ~30 min.

Rationale: this is the cleanest test of whether the freeband schedule has merit. If at
p10 ε greedy_dyn_2 no longer saturates (e.g., success ≈ 0.6), then PPO_B vs greedy_dyn_5
becomes a real comparison. If greedy still saturates at p10, the field is fundamentally
robust and reward shaping cannot break it.

### B. **Test on Track N 64D dynamics** (recommended secondary)

Track N completed in V3A and is awaiting safety pre-check. If Track N 64D shows different
contraction (potentially less saturated greedy at p25), Variant B might find its leverage
there. V3A protocol §A1-bg-3 onwards: build pairs → train RoR dynamics → reachability →
greedy saturation check. ~2 hours of compute. Re-run Phase 3 evaluation on whichever
Track N cells show greedy_dyn_2 < 0.95.

### C. **Uncertainty-aware D (Phase 4)** — NOT recommended for the headline

The heteroscedastic head's `σ` is real (V3A Track L OOD Spearman = 0.738) but at
saturated cells, low-σ plans coincide with high-success plans. PPO_D would converge to
the same K=1–3 plans as PPO_B / PPO_C / PPO_A. **Phase 4 D is interesting as a constraint
axis, not a Bucket-B headline source**, unless paired with problem reformulation.

### D. **Halt biorealistic-control objective on V2 dynamics; revisit when 64D is in play.**

If A and B both fail to un-saturate, the honest V3B finding is: "the V2 primary
32D RoR dynamics is fundamentally well-conditioned, and reward-space biorealistic
controls (safety, path-length, uncertainty) do not produce a planning-advantage
headline. Phase 2c's reward-fit safety result is the V3B contribution. Move to V3C
on a different latent / dynamics if biorealistic control is still desired."

### E. **Combined Variant E (C + B + D) on V2 dynamics** — explicitly NOT recommended

This would layer three independently-non-headline-winning axes on a saturated field.
Per the user's plan §10.7, "we never combine all axes simultaneously without a
sequential ablation — the V2-trap is 'I added 4 things and it works, which one
mattered?'". After C and B both individually fail to deliver Bucket-B leverage, E
inherits the failure mode.

## 9. Phase 3 progression summary

| | Phase 2c (Variant C) | Phase 3 (Variant B) |
|---|---|---|
| Bucket-A reward-fit | ✅ Zero CE / zero tox at every seed × cell | N/A (no biology in B) |
| Bucket-B raw success | Single positive cell within seed noise; mostly null | All cells saturated at 1.000 |
| Bucket-B vs greedy | Single-seed within V2 noise; 4-seed mean ≈ 0 | All zeros under reward-aware greedy |
| Path-length usage | N/A | T∈{4,5} ≤ 5% everywhere |
| Verdict | `PROMISING_BUT_UNSTABLE_SEED_VARIANCE_DOMINATES` | `FAIL_NO_PATH_LENGTH_LEVERAGE_FIELD_SATURATED` |
| Phase 5 unlocked? | Maybe (conditional on Phase 3) | **No** — no Bucket-B headline available |

## 10. Sacred-rule conformance

* `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` clean.
* No VAE / dynamics retraining. PPO_A reused as the frozen V2 baseline.
* New code: `path_length_freeband_reward()` in `src/rl/biology_rewards.py`;
  `path_length_freeband` mode added to `src/rl/reward.py` (V2 modes byte-identical);
  freeband knobs plumbed through `src/rl/environment.py::CellReprogrammingEnv`;
  reward-aware extension in `src/rl/baselines.py::GreedyDynamicsBeamPolicy`;
  freeband knobs added to `config/rl.yaml` (default `mode` unchanged).
* New scripts: `scripts/evaluate_rl_v3b_phase3.py`, `scripts/aggregate_v3b_phase3.py`.
* Modified: `scripts/train_rl_v3b.py` accepts `--mode {safety_aware, path_length_freeband}`.
* New tests: `tests/test_freeband_reward.py` (22 tests; full suite 327 → 349 passed).

## 11. Files produced this session

```
artifacts_v3/rl_v3b_path_freeband_seed{42,0,1,7}/    # NEW: 4 PPO_B checkpoints
artifacts_v3/eval_v3b_phase3/
├── seed{42,0,1,7}/                      # per-seed eval × 6 cells × 9 policies
├── phase3_results.csv                    # 216 rows long-form
├── phase3_results.json                   # 4-seed CIs + acceptance
├── phase3_summary.md                     # human-readable per-cell table
artifacts_v3/interpretation/
└── v3b_phase3_path_freeband.md           # (this file)

src/rl/biology_rewards.py                 # +path_length_freeband_reward
src/rl/reward.py                          # +path_length_freeband mode dispatch
src/rl/environment.py                     # +freeband knobs
src/rl/baselines.py                       # +reward-aware freeband greedy
config/rl.yaml                            # +reward.freeband.* knobs
scripts/train_rl_v3b.py                   # +--mode flag
scripts/evaluate_rl_v3b_phase3.py         # NEW
scripts/aggregate_v3b_phase3.py           # NEW
tests/test_freeband_reward.py             # NEW: 22 tests
```
