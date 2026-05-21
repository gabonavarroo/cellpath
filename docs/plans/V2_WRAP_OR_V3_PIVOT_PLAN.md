# V2_WRAP_OR_V3_PIVOT_PLAN.md — V2 Wrap-up + V3 Scoping

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:executing-plans`. When
> implementation begins, Task 1 commits a verbatim copy of this file to the repo root.

---

## 1. Context

CellPath V2 is at a decision point. P0E established the following, with high confidence:

* `RoR_corr010 × C2 PPO` and `V1 OT × B5 PPO` both reach success = 1.000 at the V2 hard
  primary cell (K=3, ε=p25, bin 8–10, OOD, n=200).
* PPO − random = **+86 pp** at primary cell (genuinely useful learning).
* PPO − greedy_dyn_2 = **+0.000 pp at primary**, **never ≥ +0.05 pp anywhere** in the matrix.
* The primary cell is saturated for every reasonable controller (PPO, grd1, grd2 all = 1.000).
* The hardness frontier sits at **K=2** (PPO 0.29–0.76, grd2 0.29–0.80) and **K=1** (all ~0).
* RoR_corr010 ties B5 at primary, **wins K=2/bin 6-8 by +23.5 pp** (0.76 vs 0.525).
* Mean-delta is RL-dead. Hybrid reward overfits; K-ablation does not yield planning evidence.

The user's concern is real: **the headline result risks being trivial** — "PPO learns a
greedy-equivalent controller" rather than "PPO discovers a superior strategy". The
recommended action is not a binary wrap-vs-pivot; it is a **single short phase (P0F)** that
honestly wraps V2 around the hardness frontier (where the result *is* informative) while
defining a focused V3 agenda for the latent-dim hypothesis.

---

## 2. Answers to the strategic questions

### Q1 — Is the current V2 result scientifically meaningful? **Yes, with proper framing.**

**Strongest honest claim:**
> Under the V2 hard benchmark on Norman 2019 K562 CRISPRa data, a MaskablePPO policy trained
> with terminal-only reward and a distance-bin curriculum achieves **success = 1.000 at the
> primary cell (K=3, ε=p25, bin 8-10, OOD), reducing trajectory length from random's 5.53 to
> 2.55 mean steps**. The policy matches a depth-2 model-based oracle (`greedy_dyn_2`) at the
> primary cell — evidence that PPO has compressed a 2-step lookahead into a feedforward
> controller without runtime model access. On the hardness frontier (K=2, OOD), success rates
> spread from 0.29 to 0.76 across configurations; this is where the comparison is informative.

**Claims to avoid:**
* "PPO discovers superior strategies / outperforms classical planning" — false; PPO never
  exceeds `greedy_dyn_2` by ≥ +0.05 pp anywhere.
* "PPO learned biologically prioritized actions" — Chronos correlation is null (P0A `ρ=−0.024,
  p=0.815`).
* "The dynamics gate predicts RL controllability" — P0E proved gate-vs-control independence
  (soft-OT passes gate, fails control; V1 OT fails gate, passes control).
* "+0.010 pp at K=3 / bin 6-8 over greedy_dyn_1 demonstrates multi-step planning" — downgrades
  to −0.005 vs greedy_dyn_2.

**Is "PPO ≈ greedy_dyn_2" acceptable as the V2 outcome?** **Yes**, conditional on (a) honest
framing that names greedy_dyn_2 as the upper-reference baseline and (b) demonstrating that the
result is robust across seeds. The result is *not* trivial — PPO is a feedforward controller
that internalised what a beam search recomputes per step. But the V2 report must say "match"
not "exceed".

### Q2 — Is the current benchmark too easy?

| Cell | Saturation status | PPO range | Informative? |
|---|---|---|---|
| K=1, * | Universally near-zero | 0.000–0.005 | No (impossible at one step) |
| K=2, bin 6-8 | Frontier | 0.140–0.760 | **Yes** |
| K=2, bin 8-10 | Frontier | 0.065–0.695 | **Yes** |
| K=3, bin 6-8 | Top end of frontier | 0.470–0.995 | Marginal |
| K=3, bin 8-10 (primary) | Saturated | 0.170–1.000 (most ≈ 1.000) | No for headline; yes for variance |
| Bin 10–12 (OOD) | Empty start pool | — | n/a |

**Hardness frontier evaluation = K=2 cells under OOD held-out genes.** A lower PPO score IS
more meaningful at K=2 because random ≈ 0.01–0.06 and grd2 ≈ 0.74–0.80 — the spread reveals
whether the policy is learning. At K=3/bin 8-10, every reasonable controller wins; the cell
provides no comparative signal.

### Q3 — Is making the environment harder valid? **Yes — within strict guardrails.**

**Legitimate stress tests** preserve the property "an oracle planner with full dynamics access
can solve a non-trivial fraction of cells":
* K ∈ {1, 2, 3, 8} — already covered.
* ε ∈ {p10, p25, p50} — p10 is a stricter goal; only valid if beam_reach@k ≥ ~10 % under p10.
* Distance bin — bin 10-12 OOD is *empty*; only mixed has cells there. Mixed split is a valid
  axis we have not exploited.
* Action constraints — e.g., mask top-K most-used genes (forces PPO to use under-explored
  actions). Legitimate because both PPO and greedy see the same mask.
* Uncertainty-aware filtering — count only successes where `unc_calibration_spearman > 0.2`.

**Illegitimate (cherry-picking) tests:**
* Tests where `beam_reach < 10 %` — the task is impossible; everyone fails, no signal.
* Tests constructed to handicap greedy specifically (e.g., randomly perturbing the dynamics
  during eval) — that measures robustness to model error, not planning ability.
* Removing the noop action — already explored (`greedy_dyn_1_noop_free`); collapses on
  control-hostile fields.

**Required reachability/oracle check:** every new (K, ε, bin, split) cell must pass the
existing `scripts/probe_reachability.py` smoke (beam reachability ≥ 10 % at depth=k under
beam_width=50). This is already implemented and used in P0C0.

### Q4 — Should we run the final seed sweep now?

**Yes, but as part of P0F (the hardness-frontier evaluation), not standalone.** Stand-alone
seed sweep on the primary cell is uninformative because both candidates saturate at 1.000.
The seed sweep is only decisive when applied to the K=2 cells (the actual frontier).

**Exactly what a 3-seed sweep proves:**

| Question | Answered by |
|---|---|
| Is the +23.5 pp RoR_corr010 × C2 advantage at K=2/bin 6-8 robust or seed noise? | seed CIs at K=2 cells |
| Does B5 mean_final_distance = 2.55 (vs C2's 2.66) survive resampling? | seed CIs at primary cell |
| Is RoR's "saturating its own grd2" at K=2/bin 8-10 a property of the field or a fluke? | seed CIs at K=2 cells |

If C2's K=2 advantage is within seed variance → tied; **V2 primary is V1 OT × B5** (simpler).
If C2's K=2 advantage exceeds 2σ → C2 wins; **V2 primary is RoR_corr010 × C2**.

### Q5 — Should VAE-64 / VAE retraining become V3?

**Yes. The "no VAE retraining" rule was a V2-only scoping constraint, not a permanent ban.**

**Evidence VAE-64 might help:**
* Greedy_dyn_2 saturating at 1.000 across multiple cells suggests the field is locally
  well-conditioned. A higher-dim latent could expose more curvature → harder for 2-step
  lookahead → bigger PPO − grd2 gap.
* `artifacts_64/contraction_auto` shows 64D contraction mean = 1.349 vs 32D's 1.008 (per-step
  reductions larger), but worst_improvement = −1.499 (more variance). This is more headroom
  for *non-trivial* policy learning.
* 64D ε (p90) = 4.43 vs 32D ε (p50) = 3.53 — different success thresholds; need ε redefined.

**Evidence VAE-64 might NOT help (be honest):**
* Legacy 64D dynamics had val Pearson 0.5965 < 32D's 0.6085 (worse on gate).
* OOD Pearson 0.3686 < 32D's 0.4793 (worse OOD).
* 64D + gene_bias collapses OOD (R² = −1.825 / −2.317). This means dynamics architecture is
  sensitive to latent dim.

**Other VAE hyperparameters worth a single ablation in V3:**
* `gene_likelihood`: NB (current) vs ZINB (Norman 2019 has high dropout rates; ZINB might
  capture them better → tighter latent → harder benchmark).
* `n_hidden` in scVI encoder: 128 (current) vs 256 (more capacity).
* KL weight: 1.0 (current) vs 0.5 (less Gaussian prior pressure → more separation).
* SCANVI (semi-supervised; uses perturbation labels) — strongest known separator. Listed in
  V2_RESEARCH_PLAN.md §4.4 as "P2 stretch", but a focused SCANVI ablation belongs in V3.

**Minimal V3 latent ablation matrix:**

| # | Config | TS (VAE + dynamics + PPO) | Purpose |
|---|---|---|---|
| V3.1 | 32D current (baseline) | already done | reference |
| V3.2 | 64D + V1-default dynamics + B5 PPO | ~2 h VAE + ~30 min dyn + ~3 min PPO | does latent dim alone change saturation? |
| V3.3 | 64D + RoR + B5 PPO | ~30 min dyn + ~3 min PPO | does RoR transfer to 64D? |
| V3.4 (conditional) | 64D + ZINB + V1-default dynamics + B5 PPO | ~2 h VAE + ~30 min dyn + ~3 min PPO | only if V3.2 shows ≥ +0.03 pp PPO − grd2 |

Do **not** try 16D or 128D unless V3.2/V3.3 show clear directional signal.

**Decision metrics for "V3 better than V2":**

| Metric | Pass if |
|---|---|
| ε_p25 distribution | exists and is non-degenerate |
| Pairing-noise median | < 0.85 (V2 32D = 0.89) |
| Dynamics gate margin (val) | improved over 32D's +0.0136 — informative even if not ≥ +0.030 |
| Beam reachability (k=3, OOD bin 8-10) | ≥ 13/n cells (preserve controllability) |
| greedy_dyn_2 at primary cell | < 0.95 (i.e. NOT saturated; field is harder) |
| **PPO − greedy_dyn_2** at any cell | **≥ +0.05 pp** ← the V3 headline criterion |

If V3 fails all six, the latent-dim hypothesis is rejected and V3 must pivot (likely to
ensemble dynamics or contraction regulariser).

### Q6 — Ranked recommendation

| Rank | Option | Decision |
|---|---|---|
| **1** | **B + A (P0F: frontier eval + seed sweep + V2 wrap-up)** | **RECOMMENDED. Cheap, decisive, honest.** |
| 2 | C (V3 latent ablation) | Launch in next session AFTER P0F. Don't skip the wrap-up. |
| 3 | E (write the V2 report directly, no new runs) | Acceptable fallback if P0F seed sweep finds C2 is tied with B5 and we want to ship. |
| 4 | D (more V2 reward/curriculum attempts) | REJECTED. Saturated primary cell cannot be beaten without changing the field; that's V3. |

**On the user's note** ("Do not assume PPO must beat greedy_dyn_2"): correct — the
expectation is reasonable only when the field has multi-step dependence (deep credit
assignment); on a locally-contractive field like V1 OT, 2-step beam is already close to
optimal, and matching it is the right outcome. V3's purpose is to test whether a different
latent geometry produces a field that *requires* > 2-step planning.

---

## 3. Recommended next phase: **P0F — V2 Honest Wrap-up**

### 3.1 Objective
Produce the final V2 deliverable: a hardness-frontier evaluation with 3-seed CIs on the two
candidate combinations, a single-document V2 final report, and a focused V3 research agenda
referencing the unresolved 32D-saturation hypothesis.

### 3.2 Sacred rules
* No VAE retraining (V2 scope; V3 starts in a separate session).
* No V1 artifact modification (`artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/` frozen).
* All new outputs under `artifacts_v2/`.
* No gate threshold lowering.
* No force-add of large artifact directories.
* `rl.train.skip_gate=true` for all V1 OT and RoR PPO retrains (logged in PROGRESS.md).
* Reachability check required before adding any new (K, ε, bin) cell.

### 3.3 Hypotheses (preregistered)
* **H_seed_robust:** RoR_corr010 × C2 and V1 OT × B5 are tied (within 2σ across 3 seeds) at
  the primary cell. Differ by > 2σ at K=2/bin 6-8 (C2 wins).
* **H_frontier_reveals_gap:** At K=2, PPO − greedy_dyn_2 has a meaningful negative gap (−0.10
  to −0.20 pp), proving the benchmark is non-saturated and PPO has measurable headroom that
  the current dynamics doesn't allow it to exploit.
* **H_action_diversity:** PPO action diversity (entropy of `action_freq`) is lower than random
  but higher than greedy_dyn_2 — evidence of "learned policy" rather than "memorised oracle".

### 3.4 Phase-by-phase plan

#### Phase 1 — Reachability oracle check (15 min)
Run `scripts/probe_reachability.py` at the K=2 frontier cells under both V1 OT and
RoR_corr010 dynamics to confirm beam reachability ≥ 10 %. If a cell fails, exclude it from
the hardness frontier. **Existing script — no new code.**

```bash
PYTHONPATH=. .venv/bin/python scripts/probe_reachability.py \
  --dynamics_dirs v1_ot:artifacts/dynamics ror_corr010:artifacts_v2/dynamics_v1ot_ror_corr010 \
  --vae_dir artifacts/vae --pairs_dir artifacts/pairs \
  --out artifacts_v2/reachability_probe_p0f_k2 \
  --epsilon 3.1662898064 --distance_bin 6-8 \
  --held_out_genes_only --max_depth 2 --beam_width 50 --n_genes 105 --device cpu
```

Repeat with `--distance_bin 8-10`. Acceptance: ≥ 10/17 cells reachable in ≤ 2 steps. If
fewer, exclude from headline.

#### Phase 2 — 3-seed PPO sweep on the two finalists (~30 min)
Retrain B5 and C2 at seeds {0, 1, 7} on top of the existing seed-42 results.

```bash
# B5 variants (V1 OT × terminal+curric K=3 1M) at seeds 0, 1, 7:
for SEED in 0 1 7; do
  PYTHONPATH=. .venv/bin/python scripts/train_rl.py --config-name default \
    paths.dynamics_dir=artifacts/dynamics \
    paths.rl_dir=artifacts_v2/rl_v1ot_terminal_curric_k3_1M_seed${SEED} \
    rl.env.max_steps=3 rl.env.epsilon_override=3.1662898064 \
    rl.env.min_start_distance=4.0 \
    rl.reward.mode=terminal_only_step_cost rl.reward.beta_step_cost=0.05 \
    rl.reward.lambda_sparse=0.05 \
    rl.ppo.total_timesteps=1000000 \
    rl.train.skip_gate=true \
    rl.train.curriculum.enabled=true \
    rl.train.curriculum.start_d=4.0 rl.train.curriculum.end_d=10.0 \
    rl.train.curriculum.end_fraction=0.7 \
    rl.train.curriculum.apply_threshold=0.25 rl.train.curriculum.check_every=10000 \
    seed=${SEED}
done

# C2 variants (RoR_corr010 × terminal+curric K=3 1M) at seeds 0, 1, 7:
for SEED in 0 1 7; do
  PYTHONPATH=. .venv/bin/python scripts/train_rl.py --config-name default \
    paths.dynamics_dir=artifacts_v2/dynamics_v1ot_ror_corr010 \
    paths.rl_dir=artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed${SEED} \
    rl.env.max_steps=3 rl.env.epsilon_override=3.1662898064 \
    rl.env.min_start_distance=4.0 \
    rl.reward.mode=terminal_only_step_cost rl.reward.beta_step_cost=0.05 \
    rl.reward.lambda_sparse=0.05 \
    rl.ppo.total_timesteps=1000000 \
    rl.train.skip_gate=true \
    rl.train.curriculum.enabled=true \
    rl.train.curriculum.start_d=4.0 rl.train.curriculum.end_d=10.0 \
    rl.train.curriculum.end_fraction=0.7 \
    rl.train.curriculum.apply_threshold=0.25 rl.train.curriculum.check_every=10000 \
    seed=${SEED}
done
```

#### Phase 3 — Hardness-frontier evaluation across all 8 PPO runs (~20 min)
Evaluate the 4 B5 seeds (42, 0, 1, 7) on V1 OT + the 4 C2 seeds on RoR_corr010 at the
frontier cells K ∈ {2, 3} × bin ∈ {6-8, 8-10} × OOD = true, n=300 per cell (raise from 200
for tighter CIs). Skip K=1 (already known to be 0).

Reuse `scripts/evaluate_rl_hard.py` (existing); the only new code is a small aggregator that
computes 95 % CIs per cell across seeds.

#### Phase 4 — Add the cross-dynamics evaluation (the missing comparison)
Currently each PPO is evaluated *on its own training dynamics*. For an honest comparison we
must also evaluate `C2 PPO` on `V1 OT dynamics` and `B5 PPO` on `RoR_corr010 dynamics`
(transfer eval — does either policy generalize across dynamics fields?). This is **one extra
matrix** with 2 PPO × 2 dynamics × hardness-frontier cells.

#### Phase 5 — Figures + comparison tables (~1 h)

**Files to create under `artifacts_v2/figures/`:**

| Figure | Content | Reused infra |
|---|---|---|
| `success_vs_K.png` | success rate vs K for PPO/random/grd1/grd2 (per cell, faceted by bin) | matplotlib; data from Phase 3 |
| `hardness_frontier.png` | scatter of (PPO−random) vs (PPO−greedy_dyn_2) per cell, coloured by K | same |
| `action_diversity.png` | entropy histogram of `action_freq` for PPO, random, greedy | existing action_freq.json files |
| `seed_variance.png` | violin plot of success rate across seeds per cell | new aggregator output |
| `dynamics_taxonomy.png` | 4-panel: V1 OT, RoR_corr010, soft-OT, mean-delta on (gate, beam, PPO) axes | existing eval JSONs |
| `mean_d_distribution.png` | histogram of final distances per policy at primary cell | rollouts.parquet |

Use existing `scripts/visualize.py` as the base (per CLAUDE.md §3 rule 8: "Notebooks visualise
results computed by `src/analysis/*`. Add new metrics to `metrics.py`"). All figure-generation
code goes in a new module `src/analysis/v2_figures.py` (NOT in notebooks). No new metric
definitions; only plotting helpers.

#### Phase 6 — V2 final documentation (~1 h)

**Files to create:**

| File | Content |
|---|---|
| `artifacts_v2/V2_FINAL_REPORT.md` | Single-document V2 summary (audience: someone who has read CLAUDE.md and V2_RESEARCH_PLAN.md). Sections: (1) Executive summary; (2) The four findings (gate-control decoupling, V1 OT controllability, RoR-as-best-dynamics, PPO-matches-grd2-not-exceeds); (3) Hardness frontier table with 3-seed 95% CIs; (4) Failure modes documented (mean-delta dead, hybrid generalisation failure, K-ablation flat); (5) Honest claims + claims avoided; (6) V3 questions left open. |
| `artifacts_v2/interpretation_p0f_wrapup.md` | Per-phase results from P0F. |
| `V3_RESEARCH_PLAN.md` | Separate document at repo root. The V3 plan (see §4 of this plan). |
| `PROGRESS.md` (modify) | One new session entry. |

**Update existing files:**

| File | Edit |
|---|---|
| `README.md` | Replace V1 headline numbers with V2 final numbers; add caveat about saturation; reference V2_FINAL_REPORT.md |
| `PHASES.md` | Mark V2 phase complete; add Phase 6 (V3 scoping) to the roadmap |
| `EXPERIMENTS.md` | Add the V2 ablation matrix (B5 × C2 × E2 × F1 × F2 × mean-delta × soft-OT) |

#### Phase 7 — Tests + V1 artifact check (~10 min)

* `pytest -q` must show 261 + (new tests if any) passed; the figure-generation module gets
  one smoke test (`tests/test_v2_figures.py`) verifying it produces PNG files without error
  on the existing data — 1 test.
* `git status -- artifacts/ artifacts_64/ artifacts/rl_sweeps/` must be clean.

### 3.5 Files to create / modify

| Path | Change |
|---|---|
| `src/analysis/v2_figures.py` | NEW — plotting helpers; no new metrics |
| `scripts/aggregate_v2_seeds.py` | NEW — one-shot aggregator that reads 8 seed evals and emits 95% CIs |
| `tests/test_v2_figures.py` | NEW — 1 smoke test |
| `artifacts_v2/V2_FINAL_REPORT.md` | NEW |
| `artifacts_v2/interpretation_p0f_wrapup.md` | NEW |
| `V3_RESEARCH_PLAN.md` | NEW (repo root, separate document) |
| `V2_WRAP_OR_V3_PIVOT_PLAN.md` | NEW (this file's verbatim copy to repo root) |
| `README.md` | Update — V2 headline; remove V1 hyperboles |
| `PHASES.md` | Mark V2 complete; add V3 row |
| `EXPERIMENTS.md` | Add V2 ablation matrix |
| `PROGRESS.md` | One new session entry |

**Files NOT to modify:**
* Anything under `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/`.
* `src/analysis/metrics.py` gate logic.
* `config/dynamics.yaml::gate.*`.
* `src/models/dynamics.py` architecture.
* `artifacts/vae/*`.

### 3.6 Acceptance criteria

| # | Criterion | Pass if |
|---|---|---|
| 1 | Reachability oracle pre-check | ≥ 10/n cells reachable at depth=K, beam=50 for each (K, bin) in the frontier |
| 2 | 8 PPO retrains complete | 4 B5 + 4 C2 checkpoints at seeds {42, 0, 1, 7} written |
| 3 | Hardness frontier evaluation | per-cell summary.json + 95% CI tables for all 4 frontier cells (K=2 × bin {6-8, 8-10} × OOD, plus K=3 × bin {6-8, 8-10} for context) |
| 4 | Cross-dynamics transfer evaluation | 2 × 2 matrix produced |
| 5 | All 6 figures generated under `artifacts_v2/figures/` | PNGs exist, are non-empty, render correctly |
| 6 | V2_FINAL_REPORT.md is complete | All 6 sections populated with measured numbers |
| 7 | V3_RESEARCH_PLAN.md is complete | minimum: V3.1–V3.4 matrix, decision metrics, sacred rules update |
| 8 | pytest -q passes; V1 artifacts clean | as above |

### 3.7 Rollback rules

* Phase 1 reachability fails (< 10/n at frontier): replace bin 6-8 with bin 4-6, rerun once.
  If still failing, restrict the frontier to K=2/bin 8-10 only (the cell already known to be
  reachable).
* Phase 2 seed-sweep training NaN: record, do not retry. Drop that seed from the analysis.
* Phase 5 figures fail to render due to missing dependencies: fall back to plain markdown
  tables; the report is still complete without PNGs.
* Phase 6 documentation must be written even if Phase 2–5 are partial — the report describes
  what was measured, not what we hoped to measure.

### 3.8 Expected runtime (Apple Silicon CPU; P0E confirms PPO ≈ 0.5–3 min per run)

| Phase | Wall-clock |
|---|---|
| 1 — reachability oracle | ~15 min |
| 2 — 8 PPO retrains (3 B5 + 3 C2 new + 2 existing) | ~30 min |
| 3 — hardness frontier eval (8 × 4 cells × 300 ep) | ~20 min |
| 4 — cross-dynamics transfer eval | ~10 min |
| 5 — figures + aggregator | ~1.5 h dev + ~10 min generation |
| 6 — documentation | ~1.5 h writing |
| 7 — pytest + checks | ~10 min |
| **Total** | **~4–5 hours** |

---

## 4. V3 Research Agenda (separate document — V3_RESEARCH_PLAN.md, written in P0F Phase 6)

### 4.1 The V3 hypothesis

The V2 result is "PPO matches a 2-step model-based oracle but does not exceed it". This is a
direct consequence of the dynamics field being locally well-conditioned in 32D latent space.
**V3's central hypothesis (H_V3_latent):** a different latent geometry — higher dim or
better-separated — produces a field where multi-step credit assignment is genuinely required,
and PPO will then exceed greedy_dyn_2 by ≥ +0.05 pp at the primary or frontier cells.

### 4.2 Sacred rules update

* `artifacts_v2/` becomes the V2-frozen frozen baseline (NEW frozen tier alongside `artifacts/`
  and `artifacts_64/`).
* `artifacts_v3/` is the V3 working directory.
* VAE retraining is now PERMITTED but must produce a *new* model in
  `artifacts_v3/vae_<config>/`, not modify any V1/V2 artifact.
* All other V2 rules carry forward (no path hardcoding, no inline metrics, etc.).

### 4.3 Minimal V3 experiment matrix

| # | VAE config | Dynamics | PPO | Purpose |
|---|---|---|---|---|
| V3.0 | (frozen 32D from V2) | (frozen RoR_corr010 or V1 OT) | (frozen B5 or C2) | reference |
| V3.1 | n_latent=64, NB, default | V1 default + RoR + corr 0.10 | B5-config 1M | does latent dim alone change saturation? |
| V3.2 (conditional) | n_latent=64, ZINB | V1 default + RoR + corr 0.10 | B5-config 1M | only if V3.1 PPO−grd2 ≥ +0.03 pp at any cell |
| V3.3 (conditional) | SCANVI 32D (semi-supervised) | V1 default + RoR + corr 0.10 | B5-config 1M | only if V3.1 fails — try a fundamentally different latent |

Reuse existing `scripts/train_vae.py`, `scripts/build_pairs.py`, `scripts/train_dynamics.py`,
`scripts/train_rl.py`. The VAE retraining is the only fundamentally new compute (~2 h per VAE
on CPU; minutes on GPU).

### 4.4 V3 success criterion

A V3 latent passes if it shows `PPO − greedy_dyn_2 ≥ +0.05 pp at one V2-equivalent cell
(K=3, ε=p25, bin 8-10, OOD) OR at any reachable K=2 cell`. Anything less and V3.1 falls back
to V3.2; if V3.2 also fails, V3 rejects the latent-dim hypothesis and pivots to ensemble
dynamics or contraction regulariser.

### 4.5 V3 deliverables

* `V3_RESEARCH_PLAN.md` (written in P0F Phase 6 as a stub; full plan in next session).
* No code changes in P0F — V3 implementation is its own session.

---

## 5. Concise implementation prompt for P0F

> You are implementing CellPath V2 P0F — V2 Honest Wrap-up. Read
> `V2_WRAP_OR_V3_PIVOT_PLAN.md` (this document, after Task 1 commits it). Implement Phases 1
> through 7 completely.
>
> Constraints (verbatim, do not violate):
> * No VAE retraining (V2 scope).
> * No modification of `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/`.
> * All new outputs under `artifacts_v2/` (the figures, reports, aggregator output) and
>   repo-root files (`V2_WRAP_OR_V3_PIVOT_PLAN.md`, `V3_RESEARCH_PLAN.md`, README, PHASES,
>   EXPERIMENTS, PROGRESS).
> * No gate threshold lowering.
> * `rl.train.skip_gate=true` for every PPO retrain.
> * No force-add of large artifact directories.
> * Pre-check reachability before adding any new (K, ε, bin) cell.
> * Stop after writing `V3_RESEARCH_PLAN.md` and updating PROGRESS.md.
>
> Execution order:
> 1. Commit a verbatim copy of `V2_WRAP_OR_V3_PIVOT_PLAN.md` to the repo root.
> 2. Phase 1: reachability oracle pre-check at K=2 frontier cells on V1 OT and RoR_corr010.
> 3. Phase 2: train 6 new PPO checkpoints (3 B5 seeds × 3 C2 seeds at {0, 1, 7}). Use seed=42
>    runs already in `artifacts_v2/` as the 4th seed.
> 4. Phase 3: evaluate all 8 checkpoints at the K=2/K=3 × bin {6-8, 8-10} × OOD cells.
> 5. Phase 4: cross-dynamics transfer eval (B5 × RoR_corr010 dyn; C2 × V1 OT dyn).
> 6. Phase 5: write `src/analysis/v2_figures.py` + `scripts/aggregate_v2_seeds.py` + 1 smoke
>    test. Generate the 6 figures.
> 7. Phase 6: write `artifacts_v2/V2_FINAL_REPORT.md`, `artifacts_v2/interpretation_p0f_wrapup.md`,
>    `V3_RESEARCH_PLAN.md` (concise — V3.1–V3.4 matrix + decision metrics + sacred rules update),
>    update README, PHASES, EXPERIMENTS, PROGRESS.
> 8. Phase 7: pytest -q and V1 artifact check.
>
> Final response must report: code changed, tests passed, exact 8 PPO checkpoints (seeds and
> primary-cell success rates), 95% CIs on the seed sweep at primary cell AND K=2 cells, the
> cross-dynamics transfer matrix, all 6 figure paths, the H_seed_robust / H_frontier_reveals_gap /
> H_action_diversity verdicts, and confirmation that no VAE / V1 artifacts / gate thresholds
> were changed. Conclude with the single-line V2 primary recommendation (B5 or C2) backed by
> the seed CIs, and one paragraph summarising the V3 next-session prompt.

---

## 6. Wrap-up structure (everything V2 needs to ship)

If P0F executes cleanly, V2 closes with these deliverables (all referenced from
`V2_FINAL_REPORT.md`):

**Documents:**
* `V2_FINAL_REPORT.md` — single-document headline (this is the deliverable).
* `V2_WRAP_OR_V3_PIVOT_PLAN.md` — plan that produced the wrap-up.
* `V2_STRATEGY_P0E_PLAN.md`, `V2_STRATEGY_REASSESSMENT_PLAN.md`, `P0C0_REACHABILITY_PLAN.md`,
  `P0B_PRIME_PAIRING_CORRECTION_PLAN.md` — phase plans (already committed).
* `artifacts_v2/interpretation_p0a_summary.md` (existing), `interpretation_p0b_doubleprime.md`,
  `interpretation_p0b2_mean_delta_corr.md`, `interpretation_p0c0_reachability.md`,
  `interpretation_p0d_v1ot_hardening.md`, `interpretation_p0e_v1ot_hardening.md`,
  `interpretation_p0f_wrapup.md` — phase interpretations.
* `V3_RESEARCH_PLAN.md` — what V3 will test.

**Figures (under `artifacts_v2/figures/`):**
* `success_vs_K.png`, `hardness_frontier.png`, `action_diversity.png`,
  `seed_variance.png`, `dynamics_taxonomy.png`, `mean_d_distribution.png`.

**Tables (inline in V2_FINAL_REPORT.md):**
* Hardness-frontier matrix with 95% CIs.
* Dynamics × control 2-axis taxonomy.
* Cross-dynamics transfer matrix.
* Mean-delta / soft-OT / hybrid failure modes summary.

**Comparisons:**
* B5 vs C2 (with seed CIs at primary AND K=2 cells).
* V1 OT × B5 vs V1 OT × random vs V1 OT × greedy_dyn_2 (the headline triplet).
* PPO ablation matrix (B1, B2, B3, B4, B5, C2, E2, F1, F2, D1, D2, D3).

**Code:**
* `src/analysis/v2_figures.py` (NEW).
* `scripts/aggregate_v2_seeds.py` (NEW).
* `tests/test_v2_figures.py` (NEW, 1 smoke).

**Tests:** pytest -q must still pass (261 + 1 = 262 tests, 2 skipped).

---

## 7. Self-review

* **Spec coverage:** answers all 6 user questions explicitly (§2). Provides a clear diagnosis
  (§2.Q1), a recommended next action (§3), an implementation prompt (§5), a separate V3
  agenda (§4), and a full V2 wrap-up structure (§6).
* **Constraints respected:** does not assume PPO must beat grd2 (§2.Q6); does not propose
  changing the benchmark to make greedy fail (§2.Q3 distinguishes legit from cherry-picking);
  does not propose broad open-ended sweeps (only 8 PPO retrains + 4 new evaluations); does
  not overclaim biological discovery (§2.Q1 explicitly excludes Chronos claim); prefers the
  smallest decisive next action (P0F is ~4–5 h vs a multi-day V3 dive).
* **Sacred rules:** all V2 rules preserved; V3 rules updated in §4.2.
* **Honest reporting:** if P0F finds C2 ≈ B5 at all cells, V2 primary defaults to V1 OT × B5
  (simpler) and V3 must proceed; if C2 > B5 at K=2, V2 primary is RoR × C2 — either way the
  V2 headline number ("PPO = 1.000 at primary, +86 pp over random, matches grd2") is
  unchanged and the K=2 hardness frontier becomes the report's secondary story.
