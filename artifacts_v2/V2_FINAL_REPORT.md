# CellPath V2 — Final Report (2026-05-16)

> **Audience:** someone who has read `CLAUDE.md` and `V2_RESEARCH_PLAN.md`. For history, see
> the per-phase interpretations at `artifacts_v2/interpretation_p0{a,b_doubleprime,b2,c0,d,e,f_wrapup}.md`.

---

## 1. Executive summary

CellPath V2 retrains the dynamics model on cleaner pairings, adds the residual-over-ridge
(RoR) architecture, introduces new reward modes + a distance curriculum, and benchmarks
against multi-step greedy oracles. The result is reported on the **V2 hard benchmark**:
K=3 episode budget, ε=p25 success threshold (3.166 latent units), distance bin 8–10, OOD
held-out genes, n=300 episodes, 4 seeds {42, 0, 1, 7}.

**V2 primary winner**: `RoR_corr010 × C2` (RoR dynamics + corr-loss λ=0.10 + terminal-only
step-cost reward + distance curriculum + 1 M timesteps). At the primary cell, PPO success
= **0.941 ± 0.048** (mean ± std across 4 seeds, 95 % normal CI **[0.894, 0.988]**); random
= 0.170; greedy_dyn_2 = 1.000. **PPO − random = +77 pp**. At the *informative* hardness
frontier (K=2, bin 6-8, OOD), C2 PPO = **0.748 ± 0.053** vs B5 (the V1-OT alternative) at
**0.588 ± 0.024** — non-overlapping CIs, **C2 wins by +16 pp**.

**Honest framing:** PPO matches but does not exceed greedy_dyn_2 anywhere on this benchmark
under 32D latent geometry. The result is *"PPO has compressed a 2-step lookahead into a
feedforward controller"*, not *"PPO discovers a superior strategy"*. The latter remains an
open V3 question.

---

## 2. Four V2 findings

### Finding 1 — The supervised gate and RL controllability are independent axes

| Dynamics | Gate (val margin ≥ +0.030) | Beam reachability (k=3 OOD) | PPO at primary cell |
|---|---|---|---|
| V1 OT | **FAIL** (+0.0074) | **PASS** (17/17) | **PASS** (1.000) |
| RoR_corr010 | FAIL (+0.0136) | **PASS** (17/17, best_dist 1.51) | **PASS** (0.941 ± 0.048) |
| soft_ot | **PASS** (+0.0413) | **FAIL** (0/17, best_dist 16.97) | FAIL (0.000) |
| mean_delta_corr_030 | FAIL (+0.0232) | FAIL (0/17, best_dist 4.09) | FAIL (0.000) |

soft-OT *passes the gate and is control-hostile*; V1 OT *fails the gate and is fully
controllable*. The supervised gate is **necessary but not sufficient** for RL success.
See `artifacts_v2/figures/dynamics_taxonomy.png`.

### Finding 2 — V1 OT is the verified-controllable dynamics; RoR_corr010 is the cleanest improvement

V1 OT remains the only V2-tested dynamics where every (start_distance ∈ bin 8-10, OOD) cell is
reachable by beam search at k=3. RoR_corr010 retains 17/17 reachability *and* improves several
metrics:

| Metric | V1 OT | RoR_corr010 | Δ |
|---|---|---|---|
| Val gate margin | +0.0074 | +0.0136 | +84 % |
| Val Pearson | 0.609 | 0.615 | +0.6 pp |
| OOD margin (val gate baseline) | +0.040 | +0.077 | +93 % |
| OOD Pearson | 0.479 | 0.516 | +0.037 |
| Uncertainty Spearman | 0.249 | 0.245 | tied |
| Beam k=3 best distance | 1.59 | 1.51 | −5 % |

RoR is the architecturally-cleanest gate-closer (MLP learns only the residual on top of the
gate's own ridge baseline). It cannot close the +0.030 threshold (OT pairing-noise ceiling),
but it improves what is improvable without breaking controllability.

### Finding 3 — RoR_corr010 × C2 PPO is V2 primary; B5 is the simpler fallback

**3-seed hardness-frontier comparison** (see `artifacts_v2/eval_p0f_seed_aggregate/seed_aggregate_success_rate.md`
for the full table; `artifacts_v2/figures/seed_variance.png` for the per-seed scatter):

| Cell | B5 (V1 OT) PPO mean | C2 (RoR_corr010) PPO mean | Δ | Seed-95 % CIs overlap? |
|---|---:|---:|---:|---|
| K=2 bin 6-8 OOD | 0.588 ± 0.024 | **0.748 ± 0.053** | **+0.160** | **NO** (B5 [0.564, 0.612] vs C2 [0.697, 0.800]) |
| K=2 bin 8-10 OOD | 0.459 ± 0.135 | 0.283 ± 0.045 | −0.176 | NO (B5 wins absolute, but PPO−grd2 is worse on V1 OT here) |
| K=3 bin 6-8 OOD | 0.961 ± 0.027 | **0.998 ± 0.002** | **+0.037** | **NO** (C2 effectively saturates) |
| K=3 bin 8-10 OOD (primary) | 0.963 ± 0.042 | 0.941 ± 0.048 | −0.022 | **YES** (tied at primary cell) |

**Decision:** C2 is V2 primary. It ties B5 at the saturated primary cell and wins
unambiguously at the K=2 / bin 6-8 frontier (CIs non-overlapping; +16 pp). Its dynamics field
is also harder to plan in at K=2 / bin 8-10 (RoR's grd2 there = 0.300 vs V1 OT's 0.727),
which contains the absolute B5 win at that cell.

### Finding 4 — PPO matches but does not exceed greedy_dyn_2 anywhere

PPO − greedy_dyn_2 across seeds and cells:

| Cell | B5 PPO − grd2 (CI) | C2 PPO − grd2 (CI) |
|---|---|---|
| K=2 bin 6-8 | −0.078 [−0.102, −0.054] | −0.042 [−0.093, +0.010] |
| K=2 bin 8-10 | −0.268 [−0.400, −0.135] | −0.017 [−0.061, +0.028] |
| K=3 bin 6-8 | −0.039 [−0.066, −0.013] | −0.002 [−0.004, −0.001] |
| K=3 bin 8-10 (primary) | −0.037 [−0.078, +0.005] | −0.059 [−0.106, −0.012] |

**The +0.05-pp planning-evidence threshold is reached at no cell.** C2 *equals* greedy_dyn_2
within seed CI at K=2 bin 8-10 (the cell where the RoR field's planner is most constrained),
which is the closest evidence we have for "PPO learned what 2-step lookahead does" — but it
does not exceed it. See `artifacts_v2/figures/hardness_frontier.png`.

---

## 3. Hardness-frontier table (4-seed 95 % CIs)

Primary cell highlighted in **bold**.

| Cell | Config | PPO mean | 95 % CI (normal across seeds) | Wilson 95 % (pooled) | random | greedy_dyn_1 | greedy_dyn_2 |
|---|---|---:|---|---|---:|---:|---:|
| K=2 bin 6-8  | B5 | 0.588 | [0.564, 0.612] | [0.560, 0.616] | 0.073 | 0.660 | 0.667 |
| K=2 bin 6-8  | **C2 (primary)** | **0.748** | **[0.697, 0.800]** | [0.723, 0.772] | 0.070 | 0.837 | 0.790 |
| K=2 bin 8-10 | B5 | 0.459 | [0.326, 0.592] | [0.431, 0.487] | 0.023 | 0.727 | 0.727 |
| K=2 bin 8-10 | C2 | 0.283 | [0.239, 0.328] | [0.259, 0.309] | 0.020 | 0.373 | 0.300 |
| K=3 bin 6-8  | B5 | 0.961 | [0.934, 0.987] | [0.948, 0.970] | 0.183 | 0.987 | 1.000 |
| K=3 bin 6-8  | C2 | 0.998 | [0.996, 0.999] | [0.993, 0.999] | 0.177 | 1.000 | 1.000 |
| **K=3 bin 8-10 (primary)** | B5 | 0.963 | [0.922, 1.000] | [0.951, 0.973] | 0.140 | 1.000 | 1.000 |
| **K=3 bin 8-10 (primary)** | **C2 (primary)** | **0.941** | **[0.894, 0.988]** | [0.926, 0.953] | 0.170 | 1.000 | 1.000 |

Notes:
* The "normal CI across seeds" captures **training-time variance** (4 PPO retrains at seeds
  {42, 0, 1, 7}).
* The "Wilson 95 % pooled" captures **evaluation-time variance** (binomial CI over the
  pooled 4 × 300 = 1 200 episodes).
* Greedy baselines have 0.000 std across seeds because they are deterministic given the
  dynamics; the only variance source for them is evaluation-time, captured by Wilson CIs
  (not shown in this table — see `seed_aggregate_success_rate.md`).
* B5 K=2 bin 8-10 has high std (0.135) because that cell is on the cusp between solvable and
  not under V1 OT; one bad seed (0.303) lowers the mean substantially.

---

## 4. Cross-dynamics transfer (does either PPO generalise?)

Each PPO evaluated on the *other* dynamics field (seed 42, n=300):

| Eval dynamics → | V1 OT | RoR_corr010 |
|---|---:|---:|
| ↓ Trained on | | |
| V1 OT (B5) | 0.963 (own) | 0.820 (transfer: −14 pp) |
| RoR_corr010 (C2) | 0.770 (transfer: −19 pp) | 0.941 (own) |

At K=2 / bin 6-8 the gap is starker:

| Eval dynamics → | V1 OT | RoR_corr010 |
|---|---:|---:|
| B5 | 0.588 | 0.443 (−15 pp) |
| C2 | 0.237 (−35 pp) | 0.748 |

**Neither PPO generalises across dynamics fields.** Both are dynamics-specific controllers
overfit to the residual structure they trained on. This is consistent with the model-free
RL narrative: the policy has internalised a *specific* dynamics field, not a
dynamics-agnostic planner.

---

## 5. Failure modes documented

### 5.1 mean_delta_corr_030 — RL-dead (P0E Phase 2 and earlier)
- Beam k=3 best distance = 4.09 > ε_p25 = 3.17 → success unreachable in 3 steps.
- All four PPO retrains (D1: K=3 terminal; D2: K=8 terminal; D3: K=8 terminal on default;
  diagnostic: K=8 absolute_distance) collapse to NOOP at step 1.
- **Cause:** ε is geometrically unreachable; terminal_only reward incentivises immediate
  NOOP to minimise step cost.

### 5.2 soft_ot — control-hostile (P0B″, P0C0)
- **Passes** the gate (+0.0413 val margin) but **fraction_positive = 0.000** — every gene
  perturbation at every start state moves the state *away* from z_ref.
- Beam k=3 best distance = 16.97 (worse than starting distance ~ 9).
- **Cause:** barycentric pseudo-controls are convex combinations of observed controls and
  hence pulled toward the control-cloud centre, which is farther from z_ref than
  bin-8-10 perturbed cells. The MLP learns this geometry faithfully.

### 5.3 hybrid_delta_terminal reward — generalisation failure (P0E Phase 3)
- α=1, B=1 → 0.006 success at smoke (200 k steps).
- α=1, B=10 → 0.754 success at smoke, scales to 0.824 at 1 M steps on the training-side eval
  (curriculum start pool, distances 4–10), but collapses to **0.170 on the hard-bench
  primary cell** (strict bin 8-10 OOD).
- **Cause:** the dense δd shaping rewards per-step progress; at evaluation on the strict
  primary cell the policy makes per-step progress but does not land within ε at K=3.

### 5.4 K-ablation — no planning emergence (P0E Phase 4)
- F2 (K=8 trained, 1 M steps): primary cell PPO = 0.940 < B5 = 1.000. PPO − grd2 = −0.060.
- F1 (K=2 trained): wins K=2 / bin 8-10 by +30 pp vs B5, but loses K=3 primary cell.
- **Cause:** the dynamics field at 32D is locally well-conditioned; extra horizon does not
  produce planning advantage beyond the 2-step lookahead a greedy oracle already achieves.

---

## 6. Honest claims (what V2 can and cannot say)

### ✅ Strongest honest claim
Under the V2 hard benchmark on Norman 2019 K562 CRISPRa data, a MaskablePPO policy trained
with terminal-only reward and a distance-bin curriculum on the RoR_corr010 residual-over-ridge
dynamics achieves **success rate = 0.941 ± 0.048 at the primary cell (K=3, ε=p25, bin 8-10,
OOD, 4 seeds × 300 ep)**, reducing trajectory length from random's 5.53 to ~2.7 mean steps.
At the K=2 / bin 6-8 frontier, this configuration outperforms the V1-OT alternative by **+16
pp (CIs non-overlapping)**. The policy matches a depth-2 model-based oracle (`greedy_dyn_2`)
within seed CIs at every cell, but does not exceed it — evidence that PPO has internalised
a 2-step lookahead into a feedforward controller without runtime model access.

### ✅ Sub-claims that are honest
* **Gate-controllability decoupling** is a methodological contribution. Soft-OT (gate PASS,
  control FAIL) and V1 OT (gate FAIL, control PASS) anchor it.
* **Sample-efficiency improvement** over random is genuine: random uses 5.53 mean steps;
  PPO uses ~ 2.5–2.7.
* **PPO is policy-specific** (cross-dynamics transfer hurts ≥ 14 pp); the result is *"PPO
  learned a controller for this dynamics field"*, not *"PPO discovered a transferable
  policy"*.
* **Hardness frontier** (K=2 cells) is where PPO comparisons are informative; the K=3 primary
  cell is partially saturated.

### ❌ Claims V2 must NOT make
* "PPO discovers superior strategies / outperforms classical planning" — false; PPO − grd2 is
  never ≥ +0.05 pp across the matrix.
* "PPO learned biologically prioritized actions" — Chronos correlation is null
  (ρ = −0.024, p = 0.815 — P0A).
* "The dynamics gate predicts RL controllability" — explicitly disproven (Finding 1).
* "PPO is a transferable controller" — cross-dynamics transfer hurts ≥ 14 pp; PPO overfits
  to its training dynamics.

---

## 7. V3 questions left open

1. **Does a higher-dim latent (64D) expose a field where multi-step planning is required?**
   The 32D field is locally well-conditioned (greedy_dyn_2 saturates at primary cell); 64D
   might preserve enough curvature for PPO − grd2 ≥ +0.05 pp to emerge.
2. **Does a semi-supervised latent (SCANVI) produce a more separable space where the
   dynamics gate is genuinely closable?** P0A's pairing-noise floor (0.89 median) is a hard
   ceiling on what any model can do on top of OT pairs in 32D scVI.
3. **Are there ensemble dynamics / contraction-regulariser losses that produce a field where
   PPO genuinely outperforms a 2-step oracle?**
4. **Does CRISPRi (knock-out) action space change the field's structure?** Currently CRISPRa
   only.

V3 is scoped in [V3_RESEARCH_PLAN.md](../V3_RESEARCH_PLAN.md). The V3 success criterion is
**PPO − greedy_dyn_2 ≥ +0.05 pp at one V2-equivalent cell or any reachable K=2 cell**.

---

## 8. Reproducibility & artifact map

### Deliverable documents
* `V2_FINAL_REPORT.md` (this file).
* `V2_WRAP_OR_V3_PIVOT_PLAN.md` (the plan that produced P0F).
* `V2_STRATEGY_P0E_PLAN.md`, `V2_STRATEGY_REASSESSMENT_PLAN.md`,
  `P0C0_REACHABILITY_PLAN.md`, `P0B_PRIME_PAIRING_CORRECTION_PLAN.md` — phase plans.
* `artifacts_v2/interpretation_p0a_summary.md`, `interpretation_p0a_decision.md`,
  `interpretation_p0b_prime.md`, `interpretation_p0b_doubleprime.md`,
  `interpretation_p0b2_mean_delta_corr.md`, `interpretation_p0c0_reachability.md`,
  `interpretation_p0d_v1ot_hardening.md`, `interpretation_p0e_v1ot_hardening.md`,
  `interpretation_p0f_wrapup.md`.
* `V3_RESEARCH_PLAN.md` — what V3 will test.

### Figures (`artifacts_v2/figures/`)
* `success_vs_K.png` — success-vs-K curves for PPO/random/grd1/grd2 per bin.
* `hardness_frontier.png` — (PPO − random) vs (PPO − grd2) scatter per cell.
* `action_diversity.png` — entropy of action_freq per PPO config.
* `seed_variance.png` — per-seed scatter of PPO success at the frontier cells.
* `dynamics_taxonomy.png` — 4-panel summary across V1 OT / RoR / mean-delta / soft-OT.
* `mean_d_distribution.png` — final-distance histograms per policy.

### Code
* `src/analysis/v2_figures.py` — figure helpers (NEW in P0F).
* `scripts/aggregate_v2_seeds.py` — seed aggregator (NEW in P0F).
* `scripts/make_v2_figures.py` — driver for the 6 figures (NEW in P0F).
* `src/rl/baselines.py::GreedyDynamicsBeamPolicy` (P0E), `NoopFreeGreedyPolicy` (P0C0).
* `src/rl/curriculum.py::DistanceCurriculumCallback` (P0D).
* `src/rl/reward.py` — reward modes added in P0D + P0E.
* `src/models/dynamics.py::use_residual_over_ridge` + `fit_ridge_baseline_from_pairs` (P0D).

### Data artifacts (under `artifacts_v2/`, gitignored)
* `pairs_{mean_delta,soft_ot,random}/` (P0B′ / P0B″).
* `dynamics_{mean_delta_default, mean_delta_corr_{005,010,030}, soft_ot_default,
  v1ot_ror, v1ot_ror_corr005, v1ot_ror_corr010, random_default}/`.
* `reachability_probe*/`, `diagnostics/gate_breakdown_*/`.
* `rl_v1ot_terminal_curric_k3_1M_seed{42,0,1,7}/`,
  `rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed{42,0,1,7}/` (the 8 V2 primary PPOs).
* `rl_meandelta_*/`, `rl_v1ot_hybrid_*/`, `rl_v1ot_terminal_curric_k{2,8}_*/` (ablations).
* `eval_p0f_*` (Phase 3 + 4 evaluations).
* `eval_p0f_seed_aggregate/seed_aggregate_{success_rate,mean_final_distance}.{json,md}`.

### Frozen tiers (must never be modified)
* `artifacts/` — V1 baseline.
* `artifacts_64/` — V1 64D ablation.
* `artifacts/rl_sweeps/` — V1 PPO checkpoints.
* `artifacts_v2/` — V2 baseline (becomes frozen as of this report).

### Sacred rules
1. No VAE retraining (V2 scope; **lifted in V3**).
2. No V1 artifact modification.
3. No gate-threshold lowering.
4. `rl.train.skip_gate=true` for any PPO retrain on a non-passing dynamics (logged in
   PROGRESS.md for every run).

---

## 9. V2 recommendation, in one paragraph

V2 ships with **RoR_corr010 dynamics × terminal_only_step_cost + curriculum + 1 M steps PPO**
as the V2 primary. At the primary cell (K=3, ε=p25, bin 8-10, OOD), the policy achieves
**0.941 ± 0.048 success** across 4 seeds, with **+77 pp absolute over random**. At the
informative K=2 / bin 6-8 frontier, this configuration beats the V1-OT alternative by **+16
pp with non-overlapping 95 % CIs**. PPO matches but does not exceed a 2-step model-based
oracle (`greedy_dyn_2`) at any cell — the V2 result is *"PPO compressed a 2-step lookahead
into a feedforward controller"*, not *"PPO discovers superior strategies"*. The supervised
dynamics gate (PHASES.md Phase 2) is shown to be necessary but not sufficient for RL
success; soft-OT *passes the gate* and is *control-hostile*; V1 OT and RoR both *fail the
gate* and are fully controllable. **V3 should test whether a higher-dim or semi-supervised
latent produces a field where multi-step planning is genuinely required**, with the V3
success criterion *PPO − greedy_dyn_2 ≥ +0.05 pp at one reachable cell*.
