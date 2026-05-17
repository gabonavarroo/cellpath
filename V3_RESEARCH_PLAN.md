# V3_RESEARCH_PLAN.md — Latent / Dynamics Redesign (stub)

> **Status:** stub written by P0F. The full V3 plan is to be drafted in a separate session
> using `superpowers:writing-plans`. This document scopes V3, lists hypotheses, defines
> success criteria, and updates the sacred rules. Implementation deferred.

---

## 1. Why V3 exists

CellPath V2 (`artifacts_v2/V2_FINAL_REPORT.md`) ships with the honest result
*"PPO matches but does not exceed a 2-step model-based oracle (`greedy_dyn_2`)"* at every
cell of the V2 hard benchmark. The primary cell (K=3, ε=p25, bin 8-10, OOD) is saturated for
every reasonable controller. The informative comparison is at K=2, where PPO and greedy_dyn_2
diverge — but never by ≥ +0.05 pp in PPO's favour.

This is most likely a property of the **32D latent geometry**: the field is locally
well-conditioned, so 2-step lookahead is already near-optimal. V3 tests whether a different
latent geometry produces a field where multi-step credit assignment is genuinely required.

---

## 2. Central hypothesis (V3)

**H_V3_latent:** A higher-dim or semi-supervised latent space produces a dynamics field
where multi-step planning is required for the V2 hard benchmark, and PPO retrained on that
field will exceed `greedy_dyn_2` by ≥ +0.05 pp at one V2-equivalent cell (K=3, ε=p25, bin
8-10, OOD) OR at any reachable K=2 cell.

If this hypothesis is rejected, V3 must pivot (see §5 — fallback agenda).

---

## 3. Sacred rules — V3 update

| Rule | V2 status | V3 status |
|---|---|---|
| No VAE retraining | enforced | **LIFTED** — V3 retrains the VAE under controlled ablations |
| All new outputs under `artifacts_v2/` | enforced | replaced by `artifacts_v3/` |
| No V1 artifact modification | enforced | enforced |
| No `artifacts_v2/` modification | n/a | **NEW** — V2 becomes a frozen tier |
| `artifacts_64/` is V1 64D legacy | frozen | frozen |
| No gate-threshold lowering | enforced | enforced |
| No path hardcoding (Hydra everywhere) | enforced | enforced |
| No inline metric definitions | enforced | enforced |
| `rl.train.skip_gate=true` for non-passing dynamics | required | required, logged |
| No CRISPRi action space | enforced (V2 dataset constraint) | **LIFTED** if a CRISPRi dataset is registered (V2_RESEARCH_PLAN.md §P2.7) |
| No external healthy reference | enforced | enforced (V3+ ethics review required) |

`artifacts_v3/` is the V3 working directory. All VAE / dynamics / RL outputs go there;
no overwrites of any prior tier.

---

## 4. Minimal V3 experiment matrix

The matrix is **conditional**: each row only triggers if the preceding row produces a
positive directional signal (≥ +0.03 pp PPO − greedy_dyn_2 at any cell).

| # | VAE config | Dynamics | PPO | Trigger | Compute |
|---|---|---|---|---|---|
| V3.0 | (frozen 32D) | (frozen RoR_corr010 or V1 OT) | (frozen B5 or C2) | always — V2 baseline | done |
| V3.1 | n_latent=64, NB, default scVI HPs | V1-default + RoR + corr 0.10 | B5-config 1M | always | ~2 h VAE + ~30 min dyn + ~3 min PPO |
| V3.2 | n_latent=64, NB | V1-default + RoR + corr 0.10 | C2-config 1M | always (parallel with V3.1) | ~3 min PPO (reuses V3.1 dyn) |
| V3.3 | n_latent=64, ZINB | V1-default + RoR + corr 0.10 | B5-config 1M | only if V3.1/V3.2 produces ≥ +0.03 pp PPO − grd2 at any cell | ~2 h VAE + ~30 min dyn + ~3 min PPO |
| V3.4 | SCANVI 32D (semi-supervised) | V1-default + RoR + corr 0.10 | B5-config 1M | only if V3.1/V3.2 fails (PPO − grd2 < +0.01 pp everywhere) | ~3 h VAE + ~30 min dyn + ~3 min PPO |

**Why not also try 16D or 128D?** 16D was rejected in V1 latent-dim ablation
(`V2_RESEARCH_PLAN.md` §4.4); 128D is speculative without intermediate evidence. Add either
only if V3.2/V3.3 show clear directional signal.

---

## 5. Success criterion (V3 headline)

**V3 succeeds if any (V3.1–V3.4) run produces `PPO − greedy_dyn_2 ≥ +0.05 pp` at one
V2-equivalent cell (K=3, ε=p25, bin 8-10, OOD) OR any reachable K=2 cell (K=2 / bin 6-8 OOD
preferred — V2 already showed this is the most informative cell).**

If V3.1 through V3.4 all fail, **V3 rejects the latent-dim hypothesis** and pivots to:

* **V3.fallback.A — Ensemble dynamics:** train 3 dynamics models with different seeds; use
  disagreement as the dynamics uncertainty and add it to the reward.
* **V3.fallback.B — Contraction regulariser:** add an explicit contraction-rate penalty to
  the dynamics loss to break the "locally well-conditioned" property.

---

## 6. Decision metrics (per V3 run)

| Metric | Value at V2 (RoR_corr010 × C2, 32D) | V3 expectation if H_V3_latent supported | V3 expectation if rejected |
|---|---|---|---|
| ε_p25 distribution | exists, p25 = 3.166 | exists, possibly tighter or wider | exists, similar |
| Pairing-noise median | 0.89 (32D); 0.78 (32D soft-OT) | < 0.85 (cleaner pairs in higher-dim) | similar or worse |
| Dynamics gate margin (val) | +0.0136 (RoR) | improves towards +0.030 | similar saturation |
| OOD Pearson | 0.516 | ≥ 0.40 (preserve generalization) | similar or worse |
| Beam reachability (k=3 OOD bin 8-10) | 17/17 (100 %) | ≥ 50 % (controllability preserved) | similar — must NOT degrade |
| greedy_dyn_2 at primary cell | 1.000 (saturated) | **< 0.95** (field is harder) | still 1.000 (field is similar) |
| **PPO − greedy_dyn_2** at any cell | never ≥ +0.05 | **≥ +0.05** at ≥ 1 cell | never ≥ +0.05 |

---

## 7. Files V3 will modify (preview)

| Path | Change |
|---|---|
| `config/vae.yaml` | add `n_latent_v3: 64`; new `gene_likelihood: zinb` variant; new `scanvi_label: perturbation` for SCANVI |
| `scripts/train_vae.py` | already Hydra-driven; no change needed beyond config overrides |
| `scripts/build_pairs.py` | already supports `pairing.method=ot|mean_delta|soft_ot|random`; reused |
| `scripts/train_dynamics.py`, `src/models/dynamics.py` | reused as-is from V2 (RoR already supported) |
| `src/rl/environment.py`, `src/rl/baselines.py`, `scripts/evaluate_rl_hard.py` | reused — `n_latent` is inferred from `gene_vocab.json` |
| `artifacts_v3/vae_n64_nb/`, `artifacts_v3/dynamics_v3_*/`, `artifacts_v3/rl_v3_*/` | NEW V3 outputs |
| `artifacts_v3/interpretation_v3_latent.md` | NEW |
| `V3_RESEARCH_PLAN.md` (this file) | converted from stub to full plan in V3 first session |

**Files V3 will NOT modify:**
* `src/analysis/metrics.py` (gate logic locked).
* `config/dynamics.yaml::gate.*` (thresholds locked).
* Any of: `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/`, `artifacts_v2/`.

---

## 8. Compute budget (V3 first session)

| Phase | Work | Wall-clock (CPU) |
|---|---|---|
| 1 | V3.1 VAE training (64D NB) | ~2 h |
| 2 | V3.1 build_pairs + train_dynamics (RoR + corr 0.10) | ~45 min |
| 3 | V3.1 reachability oracle pre-check | ~15 min |
| 4 | V3.1 PPO retrain (B5-config and C2-config @ 1M, seed 42) | ~10 min |
| 5 | V3.1 hardness frontier eval (K=2, K=3 × bin {6-8, 8-10} × OOD, n=300) | ~20 min |
| 6 | V3.1 interpretation + decision | ~1 h |
| **Total V3.1** | | **~4-5 h** |
| 7 (conditional) | V3.3 ZINB retrain end-to-end | ~3-4 h |
| 8 (conditional) | V3.4 SCANVI retrain end-to-end | ~4 h |

V3.2 reuses V3.1's VAE and dynamics; only differs in PPO config (C2 vs B5).

---

## 9. Honest framing (what V3 must not claim)

If V3 succeeds (PPO − grd2 ≥ +0.05 pp at any cell):
* Honest claim: *"In V3's higher-dim/SCANVI latent space, the dynamics field is non-trivially
  multi-step; PPO exceeds a 2-step model-based oracle by X pp at the K=Y / bin Z cell, while
  matching it at primary."*
* Do **not** claim that the V3 PPO is biologically grounded (Chronos test still required and
  still must be reported honestly — same protocol as V2 §6).
* Do **not** claim V3 invalidates V2 — both are evidence for the gate-control decoupling
  thesis.

If V3 fails (PPO − grd2 < +0.05 pp everywhere):
* Honest claim: *"The V3 latent ablation rejects the latent-dim hypothesis; the
  saturation observed in V2 is a property of the (OT pairs × CRISPRa Norman 2019 × scVI)
  pipeline at large, not specifically 32D."*
* This is a **publishable negative result** with the same V2 deliverables (gate-control
  taxonomy, hardness frontier, RoR architecture).
* Next pivot: ensemble dynamics or contraction regulariser, per §5.

---

## 10. Stop conditions

V3 stops immediately and writes the interpretation if:
* Beam reachability at the V3 dynamics drops below 50 % at K=3 / bin 8-10 OOD
  (controllability lost).
* Dynamics-gate uncertainty Spearman < 0.20 (OOD calibration collapsed).
* Any training NaN or evaluator crash that cannot be resolved without changing the V3 sacred
  rules.

V3 declares the latent-dim hypothesis rejected and pivots to the fallback agenda (§5) if:
* V3.1 AND V3.2 produce PPO − grd2 < +0.01 pp at every cell.
* V3.3 or V3.4 produces PPO − grd2 ≥ +0.05 pp at *no* cell.

---

## 11. References

* `V2_FINAL_REPORT.md` — V2 result that motivates V3.
* `V2_WRAP_OR_V3_PIVOT_PLAN.md` — the plan that produced V2's wrap and this V3 stub.
* `V2_RESEARCH_PLAN.md` — original V2 plan; lists §P1–P2 deferred items that V3 inherits.
* `artifacts_v2/interpretation_p0f_wrapup.md` — measured numbers underlying this stub.
* `artifacts/contraction_auto/`, `artifacts_64/contraction_auto/` — empirical evidence on 32D
  vs 64D contraction.
* CLAUDE.md §3 — sacred rules that V3 inherits (rules 1–10), with rule 1 lifted as
  documented in §3 above.
