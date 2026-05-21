# V2_STRATEGY_P0E_PLAN.md — Combinatorial Hardening (P0E)

> **Implementation plan for: `V2_STRATEGY_P0E_PLAN.md`.**
> When implementation begins, Task 1 commits a verbatim copy of this file to
> `/Users/gabo/Developer/ITAM/IA/cellpath/V2_STRATEGY_P0E_PLAN.md`.
> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:executing-plans`.

---

## 1. Context (why we are running another phase)

P0D ended with two open questions that the user reasonably challenged:

1. **Did we declare V1 OT the V2 primary too early?** Track C was skipped because no Track-A
   RoR variant *passed* the gate, but the RoR variants are objectively better fields than V1 OT
   on every diagnostic that matters for RL: val margin +0.0136 (vs +0.0074), OOD Pearson 0.516
   (vs 0.479), beam best_final_distance 1.51 (vs 1.59), and beam reachability still 17/17. We
   never tried retraining PPO on them.
2. **Is PPO actually planning?** B5 produced PPO=1.000 at the V2 hard primary cell, but
   `greedy_dyn_1` also produced 1.000 at the same cell. PPO−greedy = 0.000 at the headline.
   The +0.010 pp PPO−greedy at K=3 / bin 6-8 is encouraging but the field is dominated by a
   one-step-greedy oracle; a 2- or 3-step lookahead greedy is the right *upper-reference*
   baseline (V2_RESEARCH_PLAN.md §4.6) and we never built it.

Two follow-up issues:

3. **Mean-delta dynamics were rejected purely on the frozen-V1-PPO transfer result.** Per
   P0B2 the dynamics field is directionally contractive (fraction_positive = 0.826) but
   under-shoots at k=3 (beam best = 4.09). At K=8 the field might be reachable; we never tried
   a fresh PPO retrain on it.
4. **Reward modes we did not try.** Hybrid reward (delta-distance shaping + terminal success
   bonus) was listed as an option in the original strategy but never implemented or tested.

P0E addresses all four gaps without violating any sacred rule.

---

## 2. Hypotheses (preregistered before runs)

* **H_planning_baseline:** `greedy_dyn_2` (2-step lookahead beam search using the dynamics
  model as planner) is a measurably stronger upper-reference baseline than `greedy_dyn_1` at
  *some* (K, ε, bin, split) cell on V1 OT dynamics. If true, "PPO == greedy_dyn_2" becomes the
  more honest no-planning ceiling.
* **H_ror_ppo:** Retraining the B5 reward+curriculum config on `dynamics_v1ot_ror_corr010`
  (best Track-A field by gate margin AND OOD Pearson) produces a *combination* that either
  (a) matches B5-on-V1OT at the primary cell **and** beats `greedy_dyn_2` at any (K, bin),
  or (b) reduces mean_final_distance at the primary cell by ≥ 0.10. Either outcome would
  reverse the P0D conclusion that V1 OT is the V2 primary.
* **H_meandelta_k8:** Fresh PPO retraining on `dynamics_mean_delta_corr_030` at K=8
  (giving the agent more steps to cover the 4.09 → 3.17 gap) produces success rate ≥ 0.30 at
  K=8 / bin 8-10 / OOD. If true, mean-delta is RL-learnable despite gate failure.
* **H_hybrid_reward:** A hybrid reward `R_t = (d_t − d_{t+1}) − λ·1[a≠NOOP];  R_T += B·1[success]`
  at K=3 on V1 OT improves PPO success vs B5 (terminal-only) by ≥ 0 pp at primary cell **and**
  improves mean_final_distance by ≥ 0.10. (B5's "terminal_only_step_cost" provides no shaping;
  this tests whether dense+terminal beats terminal-only.)
* **H_k_ablation:** At K=8 on V1 OT (V1 setting), retrained PPO with terminal+curric reward
  achieves PPO > greedy_dyn_2 at primary cell by ≥ +0.05 pp. If true, planning beyond one step
  emerges when given a longer horizon.

Any one supported hypothesis is informative; all five together provide a complete picture.

---

## 3. Sacred rules (unchanged, restated verbatim)

* No VAE retraining; uses `artifacts/vae` read-only.
* No modification of `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/`.
* All new outputs under `artifacts_v2/`.
* No gate threshold lowering. `config/dynamics.yaml::gate.*` is untouched.
* PPO retraining on V1 OT or any non-gate-passing dynamics requires `rl.train.skip_gate=true`
  and a PROGRESS.md note documenting the override.
* No `torch.device()` outside `src/utils/device.py`; no `random.seed`/`np.random.seed`/
  `torch.manual_seed` outside `src/utils/seeding.py`.
* No path hardcoding; all new params flow through Hydra.
* No inline metrics; reuse existing modules.
* No force-add of large artifact directories. Only code, tests, configs, plan, interpretation,
  and PROGRESS.md are committed.
* Stop after writing `artifacts_v2/interpretation_p0e_v1ot_hardening.md`. Do not promote any
  winner to V2 primary in this session.

---

## 4. Phase-by-phase plan

### Phase 0 — Multi-step greedy baseline (the missing upper-reference)

**Why:** Without `greedy_dyn_2`, we cannot honestly evaluate "PPO plans beyond one step".

**Files to modify:**
* `src/rl/baselines.py` — add `GreedyDynamicsBeamPolicy` (configurable `depth ∈ {1, 2, 3}`,
  `beam_width` default 20). At each env step it does a depth-limited beam search using the
  dynamics model to find the gene sequence that minimises `||z_final − z_ref||`, then
  executes the first action of that sequence (receding-horizon planning). When `depth=1` it
  must reduce *exactly* to `GreedyDynamicsPolicy`.
* `scripts/evaluate_rl_hard.py` — wire `greedy_dyn_2` and `greedy_dyn_3` into `_policy_names()`
  and the construction block (mirror the existing `greedy_dyn_1_noop_free` pattern).
* `tests/test_baselines_multistep.py` (NEW) — three tests:
  - `test_depth_1_matches_greedy_dyn_1` (exact equivalence under fixed inputs)
  - `test_forward_shape_and_action_space` (returns int in [0, n_genes])
  - `test_beam_width_respects_max_candidates` (no error when beam_width > available genes)

**Re-evaluate existing PPO checkpoints with the new baselines:**
* Re-run `scripts/evaluate_rl_hard.py` on the existing B5 (`artifacts_v2/rl_v1ot_terminal_curriculum_k3_1M`)
  at the primary cell + extended (K=1,2,3 × bin 6-8, 8-10) with baselines
  `random,always_noop,greedy_dyn_1,greedy_dyn_2,greedy_dyn_3`.
* Output: `artifacts_v2/eval_p0e_b5_extended_with_beam_baselines/`.

**Smoke gate:** if greedy_dyn_2 evaluation cost > 5 min for a single cell at n=200, halve
beam_width to 10 and rerun. If still too slow, fall back to greedy_dyn_2 only (skip depth=3).

---

### Phase 1 — Track C: B5 PPO retrained on RoR dynamics

**Why:** The P0D plan §5.10 rollback rule skipped Track C when no RoR variant passed the
gate. The user is correct that this was premature — RoR is a strictly better field on every
RL-relevant metric. We test the combination directly.

**Files: none new (already wired in P0D).**

**Training runs (all on `artifacts_v2/dynamics_v1ot_ror_corr010`, the best Track-A run):**

| # | Reward | Curric. | K | TS | rl_dir |
|---|---|---|---:|---:|---|
| C1 | terminal_only_step_cost | yes (4→10) | 3 | 200k | `artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_200k_smoke` |
| C2 | terminal_only_step_cost | yes (4→10) | 3 | 1M | `artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M` |

**Smoke gate:** If C1 (200k) produces success_rate < 0.70 at K=3 / bin 8-10 / OOD on the
training-side final eval, skip C2 (likely architectural mismatch). Otherwise proceed.

**Per-run eval:** Same extended grid as Phase 0.

---

### Phase 2 — Mean-delta dynamics with full RL retraining

**Why:** P0B2 only tested mean-delta with frozen-V1-PPO transfer (PPO=0.000 at primary cell).
A fresh PPO might find a contractive policy at K=8 since beam best = 4.09 is < 5.0
and the field has fraction_positive = 0.826.

**Files: none new.**

**Training runs:**

| # | Dynamics | Reward | Curric. | K | TS | rl_dir |
|---|---|---|---|---:|---:|---|
| D1 | mean_delta_corr_030 | terminal_only_step_cost | no | 3 | 200k | `…/rl_meandelta_corr030_terminal_k3_200k_smoke` |
| D2 | mean_delta_corr_030 | terminal_only_step_cost | no | 8 | 500k | `…/rl_meandelta_corr030_terminal_k8_500k` |
| D3 | mean_delta_default | terminal_only_step_cost | no | 8 | 500k | `…/rl_meandelta_default_terminal_k8_500k` |

**Smoke gate:**
* D1 at K=3 is expected to fail (geometry gap). If D1 ≥ 0.30, that's surprising but informative.
* D2/D3 at K=8: if either ≥ 0.30, mean-delta is RL-learnable. If both < 0.10, mean-delta is
  dead for V2.

`rl.train.skip_gate=true` is required for all three (mean-delta variants fail the gate).

---

### Phase 3 — Hybrid reward mode (new RL method)

**Why:** B2 (delta-only) gave 0.000 success; B5 (terminal-only) gave 1.000. A hybrid `δd
shaping + terminal bonus` is the standard middle ground and was listed in the original
strategy under "future RL exploration" but never implemented.

**Files to modify:**
* `src/rl/reward.py` — add a fourth `reward_mode = "hybrid_delta_terminal"`:
  `R_t = α·(d_t − d_{t+1}) − λ·1[a≠NOOP];  R_T += B·1[success] − β·step_count·1[truncated]`.
  Adds two new params: `hybrid_alpha` (default 1.0) and `hybrid_terminal_bonus` (default 1.0).
* `src/rl/environment.py` — thread `hybrid_alpha`, `hybrid_terminal_bonus` from `reward_cfg`.
* `config/rl.yaml` — add the two new keys (default 1.0 each).
* `tests/test_reward.py` — append:
  - `test_hybrid_zero_terminal_matches_delta_distance`
  - `test_hybrid_zero_alpha_matches_terminal_only_with_distance_scale_eq_1`
  - `test_hybrid_terminal_bonus_applied_only_on_success`

**Training runs:**

| # | Dynamics | Reward | Curric. | K | TS | rl_dir |
|---|---|---|---|---:|---:|---|
| E1 | artifacts/dynamics (V1 OT) | hybrid_delta_terminal | no | 3 | 200k | `…/rl_v1ot_hybrid_k3_200k_smoke` |
| E2 | artifacts/dynamics (V1 OT) | hybrid_delta_terminal | yes (4→10) | 3 | 1M | `…/rl_v1ot_hybrid_curric_k3_1M` (only if E1 ≥ 0.70) |

---

### Phase 4 — K-ablation on V1 OT (B5 + longer horizon)

**Why:** B5 uses K=3 (V2 hard primary cell). At K=8 (V1 setting), PPO has 5 extra steps of
budget; if it actually plans, those extra steps should produce PPO > greedy_dyn_2 by a
visible margin.

**Training runs (on `artifacts/dynamics`):**

| # | Reward | Curric. | K | TS | rl_dir |
|---|---|---|---:|---:|---|
| F1 | terminal_only_step_cost | yes (4→10) | 2 | 500k | `…/rl_v1ot_terminal_curric_k2_500k` |
| F2 | terminal_only_step_cost | yes (4→10) | 8 | 1M | `…/rl_v1ot_terminal_curric_k8_1M` |

F1 investigates the K=2 / bin 8-10 weakness B5 showed (0.295 success). F2 tests planning
emergence at K=8.

---

### Phase 5 — Combinatorial evaluation matrix

Every PPO checkpoint trained in Phases 1–4 (plus the existing B3, B5) is evaluated on the
V2 hard benchmark with the full baseline set including the new multi-step greedy oracles.

**Grid:** K ∈ {1, 2, 3, 8} × ε=p25 × distance bin ∈ {6-8, 8-10} × OOD = true. Skip cells the
harness reports as empty (e.g. bin 10-12 OOD).

**Baselines:** `random_uniform_valid, always_noop, greedy_dyn_1, greedy_dyn_2, greedy_dyn_3,
ridge_greedy`. (Add `mean_delta_greedy` for the mean-delta runs only.)

**Output:** `artifacts_v2/eval_p0e_matrix/<run_name>/` with the standard `results_table.md`
per run, plus one aggregated comparison table `artifacts_v2/eval_p0e_matrix/comparison.md`
written by a small one-shot Python helper.

**Primary metrics per cell:**
* success_rate (Wilson 95 % CI)
* mean_final_distance
* mean_steps
* PPO − random
* PPO − greedy_dyn_1
* **PPO − greedy_dyn_2** ← *the new headline planning-vs-greedy delta*
* action diversity (entropy)

---

### Phase 6 — Interpretation + PROGRESS + tests + commit

**Files to write:**
* `artifacts_v2/interpretation_p0e_v1ot_hardening.md` — fills the template in §6 below.
* `PROGRESS.md` — one new session entry per CLAUDE.md §8 (append above the previous P0D
  entry; older entries unchanged).
* `V2_STRATEGY_P0E_PLAN.md` — verbatim copy of this file, committed.

**Final checks:**
* `pytest -q` — must show 252 + (new tests) passed, 2 skipped, zero regressions.
* `git status -- artifacts/ artifacts_64/ artifacts/rl_sweeps/` — must be clean.

---

## 5. Files to create / modify (consolidated)

| File | Change |
|---|---|
| `src/rl/baselines.py` | NEW class `GreedyDynamicsBeamPolicy(depth, beam_width)` |
| `scripts/evaluate_rl_hard.py` | Wire `greedy_dyn_2`, `greedy_dyn_3` in `_policy_names()` + construction |
| `src/rl/reward.py` | NEW reward mode `hybrid_delta_terminal` + two params |
| `src/rl/environment.py` | Plumb `hybrid_alpha`, `hybrid_terminal_bonus` from config |
| `config/rl.yaml` | Add `hybrid_alpha`, `hybrid_terminal_bonus` defaults |
| `tests/test_baselines_multistep.py` | NEW: 3 tests for `GreedyDynamicsBeamPolicy` |
| `tests/test_reward.py` | Append 3 tests for `hybrid_delta_terminal` |
| `scripts/compare_p0e_matrix.py` | NEW: one-shot aggregator that reads `eval_p0e_matrix/*/summary.json` and emits `comparison.md` |
| `artifacts_v2/interpretation_p0e_v1ot_hardening.md` | NEW interpretation document |
| `PROGRESS.md` | One new session entry |
| `V2_STRATEGY_P0E_PLAN.md` | Verbatim copy of this plan |

**Files NOT to modify:**
* `src/analysis/metrics.py` (gate logic locked)
* `src/models/dynamics.py` (architecture frozen for P0E — RoR work done in P0D)
* `config/dynamics.yaml::gate.*` (thresholds locked)
* `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/` (V1 frozen)
* `artifacts/vae/*` (sacred rule)

---

## 6. Interpretation template (`artifacts_v2/interpretation_p0e_v1ot_hardening.md`)

```markdown
# P0E — Combinatorial Hardening Interpretation (YYYY-MM-DD)

## Phase 0 — Multi-step greedy on existing B5
| Cell | PPO (B5) | greedy_dyn_1 | greedy_dyn_2 | greedy_dyn_3 | PPO−grd2 | PPO−grd3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K=3, bin 8-10, OOD | … | 1.000 | … | … | … | … |
| K=3, bin 6-8, OOD  | 0.995 | 0.985 | … | … | … | … |
| K=2, bin 8-10, OOD | 0.295 | 0.740 | … | … | … | … |
| K=2, bin 6-8, OOD  | 0.525 | 0.640 | … | … | … | … |

H_planning_baseline: <supported|rejected>. greedy_dyn_2 is a meaningful upper reference if …

## Phase 1 — Track C (B5-style PPO on RoR_corr010)
| Cell | PPO_RoR | PPO_V1OT (B5) | Δ | greedy_dyn_2 (on RoR) | PPO_RoR − grd2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary | … | 1.000 | … | … | … |
| K=3 / bin 6-8 | … | 0.995 | … | … | … |
| K=2 / bin 8-10 | … | 0.295 | … | … | … |

H_ror_ppo: <supported|rejected>. Why: ….

## Phase 2 — Mean-delta + full RL
| Run | Dyn | K | Final PPO | Best cell | Best success |
| --- | --- | ---: | ---: | --- | ---: |
| D1 | mean_delta_corr030 | 3 | … | … | … |
| D2 | mean_delta_corr030 | 8 | … | … | … |
| D3 | mean_delta_default | 8 | … | … | … |

H_meandelta_k8: <supported|rejected>. Why: ….

## Phase 3 — Hybrid reward
| Run | TS | Primary PPO | PPO − random | PPO − grd2 | mean_d | vs B5 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E1 | 200k | … | … | … | … | … |
| E2 | 1M   | … | … | … | … | … |

H_hybrid_reward: <supported|rejected>. Why: ….

## Phase 4 — K-ablation on V1 OT
| Run | K | Primary PPO | PPO − grd1 | PPO − grd2 | mean_steps | mean_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 2 | … | … | … | … | … |
| F2 | 8 | … | … | … | … | … |

H_k_ablation: <supported|rejected>. Why: ….

## Phase 5 — Final combinatorial matrix
| dynamics × PPO | primary success | PPO − rand | PPO − grd2 | best non-trivial cell |
| --- | ---: | ---: | ---: | --- |
| V1 OT × B5            | 1.000 | +0.86 | … | … |
| V1 OT × E2 (hybrid)   | …     | …     | … | … |
| V1 OT × F2 (K=8)      | …     | …     | … | … |
| RoR_corr010 × C2      | …     | …     | … | … |
| meandelta_corr030 × D2| …     | …     | … | … |
| …                     | …     | …     | … | … |

## Verdict
- Best end-to-end combination: ….
- First V2 cell where PPO > greedy_dyn_2 with margin ≥ +0.05: <yes/no>. If yes: <which combo>.
- Updated recommendation for V2 primary: ….

## What changed vs P0D
- P0D recommended V1 OT × B5. P0E either confirms or replaces this with <combination>.
- Track C was skipped in P0D; P0E ran it as <C1, C2>. Outcome: ….
- Mean-delta was declared dead in P0D; P0E ran D1-D3. Outcome: ….
- Multi-step greedy was missing in P0D; P0E adds it. Conclusion about "PPO plans": ….

## What is NOT recommended (still)
- VAE retraining (sacred rule).
- Gate threshold lowering.
- CRISPRi / knockout action space.
- External healthy reference.
```

---

## 7. Acceptance criteria

**Phase 0 (baselines):**
* `greedy_dyn_2` returns at least one (K, bin) cell where it differs from `greedy_dyn_1` by
  ≥ 0.01 absolute success rate.
* All 3 new baseline-tests pass.

**Phase 1 (Track C):**
* SMOKE (C1, 200k): primary-cell PPO ≥ 0.70. Otherwise skip C2.
* FULL (C2, 1M): produce one cell where `PPO_RoR > greedy_dyn_2` by ≥ +0.05 pp, OR
  reduce mean_final_distance at primary cell by ≥ 0.10 vs B5-on-V1OT.

**Phase 2 (mean-delta):**
* At least one of D1/D2/D3 produces success rate ≥ 0.30 at *any* cell. Otherwise declare
  mean-delta dead with PPO retraining.

**Phase 3 (hybrid reward):**
* SMOKE (E1, 200k): primary-cell PPO ≥ 0.70. Otherwise skip E2.
* FULL (E2, 1M): match B5 at primary cell **and** reduce mean_final_distance by ≥ 0.10.

**Phase 4 (K-ablation):**
* F2 (K=8, V1 OT): PPO > greedy_dyn_2 at primary cell by ≥ +0.05 pp.

**Overall:** At least one of (Phase 1, Phase 3, Phase 4) produces evidence of *planning*
(PPO > greedy_dyn_2 by ≥ +0.05 pp). If none does, the verdict is "V1 OT × B5 remains V2
primary; PPO matches but does not exceed a 2-step greedy oracle."

---

## 8. Rollback rules

* Phase 0: if greedy_dyn_2 evaluation is too slow at beam_width=20, halve to 10 and rerun.
  If still too slow, drop greedy_dyn_3 and proceed with only greedy_dyn_2.
* Phase 1 C1 smoke regresses (success < 0.70 at primary cell): skip C2; H_ror_ppo rejected.
* Phase 2 D1/D2/D3 all < 0.10 success at primary cell: H_meandelta_k8 rejected; mean-delta
  declared dead.
* Phase 3 E1 smoke regresses: skip E2; H_hybrid_reward rejected.
* Phase 4 F1/F2 regress vs B5: H_k_ablation rejected; document negative result.
* Any training NaN or evaluator crash: record, do not retry with hyperparam tweaks; that is
  V3 scope.

If *all* hypotheses are rejected, the interpretation must explicitly confirm the P0D
conclusion (V1 OT × B5 is V2 primary) and recommend stopping V2 here.

---

## 9. Estimated runtime (Apple Silicon CPU; P0D timings confirm PPO is ~0.5–3 min)

| Phase | Wall-clock |
|---|---|
| 0 — multi-step greedy code + tests + B5 re-eval with new baselines | ~30 min dev + ~20 min eval |
| 1 — Track C (C1 smoke + C2 full) | ~10 min train + ~10 min eval |
| 2 — Mean-delta (D1, D2, D3) | ~15 min train + ~15 min eval |
| 3 — Hybrid reward code + tests + E1, E2 | ~30 min dev + ~10 min train + ~10 min eval |
| 4 — K-ablation (F1, F2) | ~15 min train + ~10 min eval |
| 5 — Combinatorial matrix + comparison aggregator | ~30 min eval + ~20 min dev |
| 6 — Interpretation + PROGRESS + commit | ~30 min |
| **Total** | **~4.0 hours** |

---

## 10. Self-review

* **Spec coverage:** addresses every user concern: (a) "PPO ≈ greedy" → Phase 0 + 4
  (multi-step greedy + K-ablation); (b) "Track C was premature" → Phase 1; (c) "mean-delta
  not fully tested" → Phase 2; (d) "consider new RL methods" → Phase 3 (hybrid reward).
  (e) "combinatorial dynamics × RL" → Phase 5.
* **Sacred rules:** no VAE retrain, no gate-threshold lowering, no V1 artifact modification,
  `skip_gate=true` documented for every non-passing-gate PPO retrain.
* **No half-finished implementations:** every new piece of code has unit tests; smoke runs
  precede full runs.
* **No path hardcoding:** every PPO retrain uses Hydra overrides.
* **No premature conclusions:** the interpretation template requires explicit per-hypothesis
  verdicts, and acceptance criteria distinguish "supported" from "rejected" for each.
* **Honest reporting:** if every hypothesis is rejected, the interpretation must say so and
  recommend stopping V2.

---

## 11. Concise implementation prompt (for the executor)

> You are implementing CellPath V2 P0E — Combinatorial Hardening.
>
> Constraints (verbatim, do not violate):
> * Do not retrain VAE.
> * Do not modify V1 artifacts under `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/`.
> * All new outputs under `artifacts_v2/`.
> * Do not lower the dynamics gate thresholds.
> * PPO retraining on any non-gate-passing dynamics requires `rl.train.skip_gate=true` and
>   a PROGRESS.md note documenting the override.
> * Do not force-add large artifact directories to git.
> * Stop after writing `artifacts_v2/interpretation_p0e_v1ot_hardening.md`. Do not promote
>   any winner to V2 primary in this session.
>
> Execution order:
> 1. Commit a verbatim copy of `V2_STRATEGY_P0E_PLAN.md` to the repo root.
> 2. Phase 0: add `GreedyDynamicsBeamPolicy` in `src/rl/baselines.py`; wire `greedy_dyn_2`
>    and `greedy_dyn_3` into `scripts/evaluate_rl_hard.py`; write
>    `tests/test_baselines_multistep.py` (3 tests). Run pytest. Re-eval existing B5
>    against the new baselines on the extended grid.
> 3. Phase 1 (Track C): retrain B5-style PPO on `dynamics_v1ot_ror_corr010` at K=3, 200k
>    smoke then 1M full. Evaluate on the extended grid.
> 4. Phase 2 (mean-delta): retrain on `dynamics_mean_delta_corr_030` at K=3 (200k smoke),
>    then K=8 (500k). Plus K=8 control on `dynamics_mean_delta_default`. Evaluate.
> 5. Phase 3 (hybrid reward): implement `hybrid_delta_terminal` in `src/rl/reward.py`;
>    thread `hybrid_alpha` + `hybrid_terminal_bonus` through `src/rl/environment.py`;
>    update `config/rl.yaml`; add 3 tests in `tests/test_reward.py`. Run pytest. Train
>    E1 (200k smoke) and conditionally E2 (1M) on V1 OT.
> 6. Phase 4 (K-ablation): retrain B5-style on V1 OT at K=2 (500k) and K=8 (1M).
> 7. Phase 5 (matrix): write `scripts/compare_p0e_matrix.py` (one-shot aggregator). Run
>    all (dynamics, PPO) combos on the extended grid with full baseline set.
> 8. Write `artifacts_v2/interpretation_p0e_v1ot_hardening.md` per §6 template (fill in
>    measured numbers; explicit per-hypothesis verdicts).
> 9. Update `PROGRESS.md` with one new session entry.
> 10. Run `pytest -q` and `git status -- artifacts/ artifacts_64/ artifacts/rl_sweeps/`.
>
> Final response must report: code changed, tests passed, every dynamics × PPO combination
> with exact primary-cell + best-cell metrics, the H_planning_baseline / H_ror_ppo /
> H_meandelta_k8 / H_hybrid_reward / H_k_ablation verdicts, the recommended V2 primary
> (or confirmation that V1 OT × B5 from P0D remains the recommendation), the first cell
> (if any) where PPO > greedy_dyn_2 by ≥ +0.05 pp, and confirmation that no VAE / V1
> artifacts / gate thresholds were changed.
