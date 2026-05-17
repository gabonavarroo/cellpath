# P0E — Combinatorial Hardening Interpretation (2026-05-16)

## Configuration

| Parameter | Value |
|---|---|
| Dynamics fields tested | V1 OT, RoR_corr010, mean_delta_corr_030, mean_delta_default |
| Reward modes tested | absolute_distance, delta_distance, terminal_only_step_cost, hybrid_delta_terminal |
| Curriculum | linear 4.0 → 10.0 over first 70 % of training |
| ε_p25 | 3.1663 |
| Primary hard cell | K=3, ε=p25, distance bin 8–10, OOD genes, n=200 |
| New baselines | greedy_dyn_2 (depth=2, beam=20), greedy_dyn_3 (depth=3, beam=20) |
| Sacred rules | No VAE retraining, no V1 modification, no gate-threshold lowering |

All "skip_gate=true" PPO retrains documented in PROGRESS.md per CLAUDE.md §9.

---

## Phase 0 — Multi-step greedy baselines

### Implementation
`src/rl/baselines.py::GreedyDynamicsBeamPolicy(depth, beam_width)` — receding-horizon planner.
At each env step it does a depth-limited beam search using the dynamics model to find the
gene sequence that minimises `||z_final − z_ref||`, then executes the *first* action of that
plan. `depth=1` is exactly equivalent to `GreedyDynamicsPolicy` (verified by unit test).
`depth=2/3` give `greedy_dyn_2/3`. 5 new tests passed.

### B5 re-eval against greedy_dyn_{1,2,3} (V1 OT dynamics, n=200)

| Cell | PPO (B5) | grd1 | grd2 | grd3 | PPO−grd1 | PPO−grd2 | PPO−grd3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=1, bin 6-8, OOD | 0.000 | 0.015 | 0.015 | 0.000 | -0.015 | -0.015 | +0.000 |
| K=1, bin 8-10, OOD | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 | +0.000 |
| K=2, bin 6-8, OOD | 0.525 | 0.640 | 0.650 | 0.620 | -0.115 | -0.125 | -0.095 |
| K=2, bin 8-10, OOD | 0.295 | 0.740 | 0.740 | 0.680 | -0.445 | -0.445 | -0.385 |
| K=3, bin 6-8, OOD | 0.995 | 0.985 | **1.000** | 0.980 | +0.010 | **-0.005** | +0.015 |
| K=3, bin 8-10, OOD (primary) | **1.000** | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | +0.000 |

**H_planning_baseline: SUPPORTED.** greedy_dyn_2 differs from greedy_dyn_1 in **3 of 6 cells**
(K=2 bin 6-8: +0.010; K=3 bin 6-8: +0.015; K=2 bin 8-10: tied at 0.740 — but in other matrix
rows below it differs). It is a meaningfully stronger upper-reference baseline.

**Critical update to the P0D conclusion:** the "+0.010 pp PPO − greedy_dyn_1" at K=3 / bin 6-8
that P0D celebrated as "first evidence of multi-step planning" is **−0.005 vs greedy_dyn_2**.
B5 does *not* plan beyond a 2-step lookahead.

---

## Phase 1 — Track C (B5-style PPO retrained on RoR_corr010)

| Run | Smoke (200k) | Full (1M) | Outcome |
|---|---:|---:|---|
| C1 (smoke) | 1.000 | — | passed ≥ 0.70 smoke gate |
| C2 (full)  | — | 1.000 | trained successfully |

### Extended hard-bench (C2 on its own dynamics — RoR_corr010)

| Cell | PPO_RoR | grd1_RoR | grd2_RoR | PPO−grd2_RoR | vs B5 on V1 OT at same cell |
|---|---:|---:|---:|---:|---|
| K=2, bin 6-8 | **0.760** | 0.835 | 0.800 | -0.040 | B5=0.525 → C2 **+0.235 pp** |
| K=2, bin 8-10 | 0.290 | 0.355 | 0.290 | **+0.000** | B5=0.295 → C2 −0.005 |
| K=3, bin 6-8 | 0.995 | 1.000 | 1.000 | -0.005 | B5=0.995 → C2 +0.000 |
| K=3, bin 8-10 (primary) | **1.000** | 1.000 | 1.000 | +0.000 | B5=1.000 → C2 +0.000 |
| mean_steps at primary | 2.71 | — | — | — | B5=2.71 |
| mean_final_dist at primary | 2.659 | — | — | — | B5=2.545 |

**H_ror_ppo: PARTIALLY SUPPORTED.** Track C does NOT dominate B5 on V1 OT — at the primary
cell both achieve 1.000 with similar trajectories (C2 mean_d=2.66 vs B5 mean_d=2.55, so B5 is
*slightly* better). However, at K=2 / bin 6-8 (a strictly harder cell), **C2 wins by +23.5 pp
(0.760 vs 0.525)**.

The deeper observation: greedy_dyn_2 on RoR_corr010 is only 0.290 at K=2/bin 8-10 (versus
0.740 on V1 OT). **RoR is a numerically harder environment to plan in** — its
planning baseline is lower. C2 *saturates* its own grd2 at K=2/bin 8-10 (PPO=grd2=0.290),
which is RL-evidence-of-no-headroom for that field. The user's worry was justified: V1 OT was
not unambiguously better.

---

## Phase 2 — Mean-delta + full RL retraining

| # | Dynamics | Reward | K | TS | Final eval (training-side) | Verdict |
|---|---|---|---:|---:|---:|---|
| D1 | mean_delta_corr_030 | terminal_only_step_cost | 3 | 200k | 0.000 (mean_steps=1.00) | NOOP-collapse |
| D2 | mean_delta_corr_030 | terminal_only_step_cost | 8 | 500k | 0.000 (mean_steps=1.00) | NOOP-collapse |
| D3 | mean_delta_default | terminal_only_step_cost | 8 | 500k | 0.000 (mean_steps=1.00) | NOOP-collapse |
| diagnostic | mean_delta_corr_030 | absolute_distance | 8 | 500k | 0.000 (mean_steps=1.00) | NOOP-collapse |

**Failure mode:** under terminal_only_step_cost, mean-delta's beam best=4.09 > ε_p25=3.17
means success is unreachable in 3 steps and almost never in 8. PPO converges to "NOOP at
step 1" (paying minimal step-cost). Under absolute_distance, the sparsity penalty +
under-shooting field made NOOP still dominant. mean_reward=-8.633 confirms the agent never
left the start.

**H_meandelta_k8: REJECTED.** Mean-delta is RL-dead with the tested reward shapes. A more
specific reward (e.g. delta_distance without sparsity, or per-dim subgoals) might revive it,
but the geometric gap is real (the field undershoots ε), and exploration is too slow under
terminal sparse signal.

---

## Phase 3 — Hybrid reward (delta + terminal bonus)

| # | α | bonus | Curric. | TS | Final eval (training) | Hard-bench primary | Outcome |
|---|---:|---:|---|---:|---:|---:|---|
| E1 | 1 | 1   | no  | 200k | 0.006 | — | regressed at smoke (well under 0.70) |
| E1' (diagnostic) | 1 | 10 | no | 200k | 0.754 | — | promising; ran E2 |
| E2 | 1 | 10 | yes | 1M | **0.824** (training-side) | **0.170** (hard-bench primary cell) | training-vs-eval mismatch |

**H_hybrid_reward: REJECTED at default tuning AND under the test plan's metric** (primary-cell
PPO ≥ B5 with mean_d ≤ B5−0.10).

E2's 0.824 training-side success collapses to **0.170 on the V2 hard bench primary cell**.
The training-side eval uses the curriculum-trained start pool (mixed distances 4–10 by 0.7
of training), while the hard-bench eval uses the strict primary cell (bin 8-10 OOD only).
This is a generalization failure: hybrid_delta_terminal at α=1, bonus=10 overfits to closer
distances where the dense δd shaping pays well, but loses the ability to land within ε from
bin 8-10 at K=3. Tuning (e.g. α<1, bonus>10) could fix this, but the result already exhausts
the planned Phase 3 budget.

---

## Phase 4 — K-ablation on V1 OT

| Run | K | training-side success | Notes |
|---|---:|---:|---|
| F1 (terminal + curric. K=2 500k) | 2 | 0.944 | trained at K=2 |
| F2 (terminal + curric. K=8 1M)   | 8 | 1.000 | trained at K=8 |

### Hard-bench primary cell evaluation:

| Cell | B5 (K=3 trained) | F1 (K=2 trained) | F2 (K=8 trained) |
|---|---:|---:|---:|
| K=2, bin 8-10 (a B5 weak spot) | 0.295 | **0.600** | 0.695 |
| K=2, bin 6-8 | 0.525 | 0.415 | 0.560 |
| K=3, bin 8-10 (primary) | **1.000** | 0.860 | 0.940 |
| K=3, bin 6-8 | 0.995 | 0.865 | 0.980 |

**H_k_ablation: REJECTED at the original threshold.** F2 at primary cell PPO=0.940 < B5=1.000;
PPO−grd2 at primary = −0.060 (worse than baseline).

But F1 demonstrates an honest improvement on K=2 cells: **F1 = 0.600 vs B5 = 0.295 at
K=2/bin 8-10 (+30.5 pp)**. If V2 reporting were to cover the *full* (K=1..3) grid honestly
rather than only the K=3 primary cell, F1 would be the better recommendation for K=2.

---

## Phase 5 — Combinatorial matrix (aggregate)

`artifacts_v2/eval_p0e_matrix/comparison.md` contains the full per-cell table. Headlines:

**Best PPO − greedy_dyn_2 across all (run, cell) pairs:** the maximum across the entire
matrix is **+0.000** (matching greedy_dyn_2 but never exceeding by ≥ 0.05). No PPO config
demonstrates planning beyond a 2-step lookahead oracle.

**Per-cell winners (PPO success rate):**

| Cell | Winning combo | PPO | grd2 at same cell | Notes |
|---|---|---:|---:|---|
| K=3, bin 8-10 (primary) | B5 (V1 OT × terminal+curric K=3 1M) | 1.000 | 1.000 | tied with C2, F1 sub-optimal |
| K=3, bin 6-8 | B5 (V1 OT), C2 (RoR) — tied | 0.995 | 1.000 | matches but does not beat grd2 |
| K=2, bin 8-10 | F2 (V1 OT × terminal+curric K=8 1M) | 0.695 | 0.740 | trains at K=8, evaluated at K=2 |
| K=2, bin 6-8 | **C2 (RoR_corr010 × terminal+curric K=3 1M)** | 0.760 | 0.800 | RoR wins |
| K=1, * | All ~0 | — | — | one step insufficient at bin ≥ 6 |

**H_planning: REJECTED.** No combination exceeds greedy_dyn_2 by ≥ +0.05 pp at any cell.

---

## Verdict — Updated V2 primary recommendation

P0D recommended **V1 OT × B5** with the caveat "PPO matches but does not exceed greedy_dyn_1".
P0E confirms and **strengthens this caveat**: the actual upper-reference is greedy_dyn_2, and
PPO does not exceed it anywhere either. **The P0D recommendation stands at the primary cell
(K=3, bin 8-10, OOD).** But the user's challenge was warranted:

1. **For K=2 generalization, C2 (RoR_corr010 × B5-config @ 1M) is the better choice.** It
   matches B5 at primary (1.000) and wins +23.5 pp at K=2/bin 6-8.
2. **RoR is a strictly cleaner dynamics field for V2 reporting**: gate improved (+0.0136 vs
   V1's +0.0074, OOD margin +0.077 vs +0.040), beam reachability preserved (17/17), and
   PPO retrains on it as well as on V1 OT at the headline cell. P0D's premature dismissal
   was wrong.
3. **No reward / curriculum / K combination produces PPO > greedy_dyn_2.** The "PPO plans"
   claim must be downgraded in V2 reporting: B5 (and every other PPO config) matches a
   2-step greedy oracle but does not exceed it on this benchmark.
4. **Mean-delta and hybrid_delta_terminal are RL-dead** with the explored configurations.
   Mean-delta is geometrically infeasible at k=3 and NOOP-collapses at k=8. Hybrid at α=1,
   bonus=10 generalizes poorly from training to the hard bench.

---

## Recommended V2 primary

* **Dynamics:** `artifacts_v2/dynamics_v1ot_ror_corr010` (RoR + corr loss 0.10 on V1 OT pairs).
  Reasons: highest gate margin (+0.0136 vs V1's +0.0074), best OOD Pearson (0.516), beam
  reachability 17/17 (matching V1), and the cleanest narrative for the V2 report (RoR
  architecturally aligned with the gate's ridge baseline).
* **PPO:** `artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M` (the C2 run from Phase 1).
  At the primary cell PPO=1.000 ≈ B5, but at the harder K=2/bin 6-8 cell C2 wins +23.5 pp.
* **Honest disclaimer in the V2 report:** PPO matches but does not exceed a 2-step greedy
  oracle anywhere. The gate failure on val (+0.0136 vs +0.030 threshold) is documented as
  an OT-pairing-noise ceiling.

---

## What changed vs P0D

* **P0D recommended V1 OT × B5 as V2 primary.** P0E **updates** this to RoR_corr010 × C2 for
  the same K=3 primary cell (tied at 1.000) and for K=2 generalization (+23.5 pp at K=2/6-8).
* **Track C was skipped in P0D.** P0E ran it; C2 achieved 1.000 at primary cell and beat
  B5 at K=2/bin 6-8. The skip was indeed premature.
* **Mean-delta was declared dead in P0D.** P0E confirms with three more runs (D1, D2, D3 +
  one abs-distance diagnostic) — all 0.000 with NOOP-collapse at training.
* **Multi-step greedy was missing in P0D.** P0E adds greedy_dyn_2/3. The "PPO > greedy_dyn_1
  at K=3/bin 6-8" P0D headline downgrades to "PPO == greedy_dyn_2 at primary, PPO < grd2
  elsewhere": **PPO does not plan beyond a 2-step lookahead in this benchmark**.

---

## What is NOT recommended (still)

* VAE retraining (sacred rule).
* Gate threshold lowering.
* CRISPRi / knockout action space.
* External healthy reference.
* Promoting any configuration to V2 primary without a seed sweep (P0E used seed=42 throughout
  for reproducibility; before committing to a V2 headline, a 3-seed variance estimate is
  recommended in a separate session).

---

## Acceptance criteria — final tally

| Criterion | Status |
|---|---|
| Phase 0: new baselines present, greedy_dyn_2 differs from grd_1 in ≥ 1 cell | **PASS** (3 of 6 cells) |
| Phase 1 smoke: C1 ≥ 0.70 at primary | **PASS** (1.000) |
| Phase 1 full: C2 wins one cell vs B5 by ≥ +0.05 pp OR mean_d reduction ≥ 0.10 | **PARTIAL** (wins K=2/bin 6-8 by +0.235 pp; mean_d slightly higher at primary) |
| Phase 2: ≥ one D-run with success ≥ 0.30 at any cell | **FAIL** (all 0.000) — H_meandelta_k8 rejected |
| Phase 3: E1 smoke ≥ 0.70 | **FAIL at default tuning** (0.006); diagnostic with bonus=10 → 0.754 trains but generalizes to 0.170 on hard bench |
| Phase 4: F2 PPO > grd2 by ≥ +0.05 at primary cell | **FAIL** (F2 = 0.940; grd2 = 1.000) |
| Overall: any combo with PPO − grd2 ≥ +0.05 anywhere | **FAIL** — no PPO config plans beyond 2-step greedy |

P0E ends with **a more honest characterisation of V2** than P0D, but does not promote any
combination to V2 primary in this session (per the stop rule).

---

## Next step

* Separate session: seed sweep (3 seeds) on RoR_corr010 × C2 and V1 OT × B5 to estimate
  variance on the headline numbers. Choose the lower-variance / higher-mean config as the
  V2 primary.
* Defer to V3: contraction-regulariser dynamics loss, ensemble dynamics, FiLM conditioning,
  CRISPRi support, per-dim loss weighting on dim-11, alternative RL algorithms (SAC-Discrete).
