# V3A Checkpoint — Track L Complete, Track N In-Progress

**Date:** 2026-05-17
**Phases complete:** A0, A1 (Track L), A2 (Track L), A3 (Track L); A1-bg in progress
**Halt:** before PPO (A4) per user instruction

---

## 1. Track L (legacy 64D VAE reuse) — full pipeline results

### 1.1 Phase A0 audit (recap)

Verdict: **REUSE OK**. Gene vocab and held-out (OOD) split bit-identical with V1/V2.
scVI config: `n_latent=64`, `gene_likelihood='nb'`, `dispersion='gene'`,
`latent_distribution='normal'`, `n_layers=2`, `n_hidden=128`. See
`artifacts_v3/audit_v3a.md`.

### 1.2 Phase A1 — ε quantiles, pairs, pairing-noise

`epsilon_success.json` (recomputed at p10/p25/p50/p75/p90 over 11 855 control cells):

| Quantile | Track L 64D | V2 32D (reference) |
|---|---:|---:|
| p10 | 2.918 | (not reported) |
| **p25** | **3.187** | 3.166 |
| p50 | 3.548 | 3.531 |
| p75 | 3.969 | (not reported) |
| p90 | 4.435 | (not reported) |
| mean | 3.639 | — |
| std | 0.662 | — |

**Finding:** 64D ε quantiles are within 1 % of V2 32D — the centroid-relative
distance distribution is essentially preserved. The plan's expected `√(64/32) ≈
1.4×` scaling did NOT materialize: 64D scVI has an effective dimensionality
much lower than the nominal 64.

OT pairs (current `scripts/build_pairs.py`, seed 42):

| Split | Track L count | V2 count |
|---|---:|---:|
| train_pairs | 38 958 | 38 958 |
| val_pairs | 4 324 | 4 324 |
| ood_pairs | 14 549 | 14 549 |
| combo_pairs | 35 995 | 35 995 |

OOD gene set: bit-identical (`AHR, ARRDC3, BAK1, CELF2, COL2A1, …` — 21 genes).
`pair_seed=42`, `ot_epsilon=0.05`, `pairing_method='ot'`.

Pairing-noise (per-gene residual variance / total variance of Δz):

| Statistic | Track L 64D | V2 32D (reference) |
|---|---:|---:|
| median | **0.886** | 0.89 |
| mean | 0.859 | — |
| min | 0.518 | — |
| max | 0.994 | — |
| std | 0.108 | — |

**Finding:** 64D does NOT reduce OT pairing noise. The 0.89 floor in 32D was
not a dimensionality artifact — it's intrinsic to the OT pairing of unmatched
cell observations.

### 1.3 Phase A2 — dynamics (RoR + corr 0.10)

Training: early stopped at epoch 89 of 300 (`patience=35`), best epoch 54
(11 min wall-clock on MPS). `selection_metric=gate_margin` → `model.pt` ←
`model_best_gate.pt`. `recommend_checkpoint=keep_best_nll` flagged because
best_gate and best_nll coincided at epoch 54.

| Metric | Track L 64D | V2 RoR 32D (reference) | Δ |
|---|---:|---:|---:|
| **val Pearson** | **0.620** | 0.615 | +0.005 (tied/slight gain) |
| val R² | 0.399 | 0.398 | tied |
| val MLP−ridge Pearson margin | **+0.0043** | +0.0136 | **−0.0093 (worse 3×)** |
| val ridge Pearson | 0.616 | 0.601 | +0.015 (ridge captures *more*) |
| **OOD Pearson** | **0.515** | 0.516 | tied |
| OOD R² | 0.302 | — | — |
| OOD MLP−ridge Pearson margin | **+0.0470** | +0.077 | **−0.030 (worse)** |
| OOD ridge Pearson | 0.468 | — | — |
| Uncertainty Spearman (val) | **0.805** | 0.245 (reported) | +0.560 |
| Uncertainty Spearman (OOD) | 0.738 | — | — |
| Gate val passed | FAIL (0.0043 < 0.030) | FAIL (0.0136 < 0.030) | same |
| OOD margin checks | all PASS | all PASS | same |

**Finding:** Track L RoR matches V2 RoR on raw Pearson but is architecturally
WORSE: the ridge baseline is stronger in 64D (val ridge Pearson 0.616 vs V2's
0.601), leaving less residual for the MLP to capture. The MLP-vs-ridge gap
shrinks from +0.0136 to +0.0043 — the dynamics gate is *farther* from passing,
not closer. The uncertainty calibration is much better (0.805 vs reported
0.245) — likely a metric definition difference between V2 and current
implementation; same scale in both Track L val and OOD.

**OOD Pearson 0.515 ≥ 0.40 threshold: PASS.** Hard gate met. RoR retraining
lifted legacy 64D OOD Pearson from 0.369 to 0.515 — the legacy `state_linear_skip`
architecture was the bottleneck, not the latent itself.

### 1.4 Phase A3 — reachability + greedy saturation

#### Beam reachability (depth-3 beam, beam_width 50, OOD only)

| Cell | Track L reach | best_dist | V2 RoR 32D reach |
|---|---|---:|---|
| K=3 / bin 8-10 / OOD | **8/8 = 100 %** | 1.53 | 17/17 = 100 % |
| K=2 / bin 6-8 / OOD | **179/183 = 97.8 %** | 1.61 | (V2 K=3 only) |

Notable: bin 8-10 OOD pool has only 8 cells in Track L (vs V2's 17). The
tighter ε distribution in 64D means fewer control cells fall into bin 8-10
relative to the centroid. This is geometry, not a contamination — the OOD
gene set is identical; only the bin-distance assignment differs.

**Field is fully controllable. Stop condition (reach < 50 %) is NOT
triggered.**

#### Greedy saturation (n_episodes=300, seed=42, deterministic)

| Cell | random | grd_1 | **grd_2** | grd_3 | V2 grd_2 (reference) |
|---|---:|---:|---:|---:|---:|
| K=2 / bin 6-8 / OOD | 0.173 | 0.963 | **0.963** | 0.943 | 0.790 |
| K=2 / bin 8-10 / OOD | 0.090 | 0.853 | **0.853** | 0.853 | 0.300 |
| K=3 / bin 6-8 / OOD | 0.397 | 1.000 | **1.000** | 1.000 | 1.000 |
| **K=3 / bin 8-10 / OOD (primary)** | 0.337 | 1.000 | **1.000** | 1.000 | 1.000 |

Wilson 95 % CIs in `summary.md`; e.g. grd_2 at K=3/bin 8-10/OOD has CI
[0.987, 1.000].

**Finding:** greedy_dyn_2 SATURATES at 1.000 at K=3 primary on Track L
(matching V2). More strikingly, greedy_dyn_2 at K=2/bin 8-10/OOD jumps from
V2's 0.300 to **0.853**, and at K=2/bin 6-8/OOD from V2's 0.790 to **0.963**.

**The Track L 64D field is MORE saturated than V2's 32D field, not less.**
The V3 latent-dim hypothesis (a higher-dim latent exposes a less-saturated
field where PPO − greedy_dyn_2 ≥ +0.05 pp becomes achievable) is **rejected
on Track L**.

Additional observation: greedy_dyn_3 = greedy_dyn_2 at every saturated cell.
At K=2/bin 6-8, grd_3 (0.943) trails grd_2 (0.963) — receding-horizon at
depth 3 with only 2 env steps remaining occasionally commits to a 3-step
path that cannot be completed.

---

## 2. Track N (fresh 64D NB VAE) — status

**A1-bg.1 VAE training**: in progress, epoch ~43/400 at the time of this
checkpoint, ~50 s/epoch on MPS, val_loss train_loss=1.43e+3 still climbing
slowly. Estimated remaining wall-clock: **4–5 hours** (early stopping with
patience 45 may bring this in around epoch 200, but no plateau yet).

**A1-bg.2, A1-bg.3, A2-bg, A3-bg**: not started — depend on A1-bg.1.

The fresh NB VAE will produce a comparable dynamics field once trained. Its
purpose is to **disambiguate VAE config drift from latent-dim effects**: if
Track N also shows greedy_dyn_2 saturation at primary, then 64D itself is the
issue (and V3.1 latent-dim hypothesis is conclusively rejected). If Track N
shows greedy_dyn_2 < 0.95 at primary, then the legacy VAE config (no Hydra
snapshot, possibly older scVI scaffold) was the variable that made the field
saturable — and a fresh-NB Track L equivalent would be the V3.1 reference.

Given Track L's clear NEGATIVE signal on the V3 hypothesis, completing Track
N becomes a **confirmation test** rather than a survival test for V3.

---

## 3. Stop-condition check (per user instruction)

| Stop condition | Track L | Track N | Triggered? |
|---|---|---|---|
| Both tracks OOD Pearson < 0.40 | **0.515 PASS** | pending | NO |
| Both tracks K=3/bin 8-10/OOD beam reach < 50 % | **100 % PASS** | pending | NO |
| One track fails, other passes (continue with passing) | passes safety, fails hypothesis | pending | partial |

**No hard stop is triggered.** Track L PASSES both safety conditions
(OOD-generalizable dynamics, controllable field) but FAILS the V3 hypothesis
(greedy_dyn_2 still saturates ≥ 0.95 at primary). The V3A plan stop conditions
focus on the safety criteria, which are met.

---

## 4. Honest framing of the Track L result

### What V3A Track L is allowed to claim

* **Track L dynamics is V2-comparable in raw quality.** Val Pearson 0.620 ≈
  V2's 0.615; OOD Pearson 0.515 ≈ V2's 0.516; uncertainty calibration robust
  in both val and OOD.
* **The ridge baseline is STRONGER in 64D legacy than in V2 32D.** Val ridge
  Pearson 0.616 vs V2's 0.601 (+0.015). RoR captures less incremental signal
  in this latent.
* **The 64D legacy field is at-least-as-controllable as V2's 32D field.**
  Beam reach 8/8 at K=3/bin 8-10/OOD; 179/183 at K=2/bin 6-8/OOD.
* **The 64D legacy field is MORE saturated for greedy planning than V2.**
  greedy_dyn_2 at K=2/bin 8-10/OOD is 0.853 (V2: 0.300); at K=2/bin 6-8/OOD
  is 0.963 (V2: 0.790); at K=3 primary it stays at 1.000.

### What V3A Track L must NOT claim

* "V3.1 succeeds" — false. greedy_dyn_2 still saturates at primary; PPO
  cannot beat a saturated oracle by ≥ +0.05 pp.
* "64D is harder geometry" — false on Track L. 64D legacy is *easier*
  geometry for greedy planning.
* "Track L is conclusive about 64D" — premature. Track N is the control
  for VAE-config drift; conclusion deferred to A6 (cross-track compare).

### Implication for next phases

Per the V3A plan §6 anti-trap rules and §10.5: Track L alone is sufficient
evidence to anticipate a V3.1 PPO outcome that **mirrors V2** (PPO matches
greedy_dyn_2 within seed CI but does not exceed it ≥ +0.05 pp). Training PPO
on Track L is therefore high cost / low information value. The user has
correctly held PPO behind explicit approval. Recommended next moves (per the
V3A plan):

1. **Let Track N VAE complete** (4–5 h background). If Track N greedy_dyn_2
   < 0.95 at primary, the legacy VAE config was the bottleneck and V3.1.nb
   is the new candidate; otherwise V3.1 (any 64D) is dead.
2. **Pivot the V3 design** (per V3_RESEARCH_PLAN.md §5 fallback agenda):
   * V3.3 — 64D ZINB (different gene likelihood; tests whether NB vs ZINB
     changes the planning geometry).
   * V3.4 — SCANVI 32D (semi-supervised; tests whether perturbation labels
     produce a more separable, less saturable space).
   * V3.fallback.A — Ensemble dynamics (3 dynamics models with different
     seeds; disagreement as uncertainty; reward includes uncertainty).
   * V3.fallback.B — Contraction-regulariser dynamics (explicit penalty on
     contraction rate to break the "locally well-conditioned" property).
3. **Skip A4 PPO on Track L** until Track N confirms or refutes the
   VAE-config explanation. A K=3-saturated dynamics field is a known V2 trap.

---

## 5. Artifact inventory (V3A so far)

### Created under `artifacts_v3/`
* `audit_v3a.md` (A0)
* `vae_n64_legacy/` (copy of `artifacts_64/vae/`; full 1.0 GB)
* `vae_n64_legacy/epsilon_success.json` (recomputed p10/p25/p50/p75/p90)
* `pairs_n64_legacy/{train,val,ood,combo}_pairs.npz`, `metadata.json`,
  `pairing_noise.json`
* `dynamics_n64_legacy_ror_corr010/` (model.pt + best_nll/best_gate + gate
  json + val/ood metrics + ridge baseline + epoch metrics +
  checkpoint_comparison)
* `dynamics_n64_legacy_ror_corr010_training.log`
* `reachability_probe_v3a/legacy_k3_bin810_ood/` (8/8 reach)
* `reachability_probe_v3a/legacy_k2_bin68_ood/` (179/183 reach)
* `eval_v3a_hardness/legacy_baselines_seed42/` (greedy saturation table)
* `vae_n64_nb_training.log` (Track N in progress)
* `interpretation/v3a_checkpoint.md` (this file)

### Created elsewhere
* `config/paths.yaml` — additive V3 path keys (`v3_legacy_*`, `v3_nb_*`,
  `v3_reach_dir`, `v3_eval_dir`, `v3_figures_dir`, `v3_interpretation_dir`)
* `V3A_LATENT_AUDIT_AND_64D_PLAN.md` — V3A plan (copied from plan-mode file)
* `scripts/evaluate_baselines_only.py` — baselines-only evaluator helper
  (no PPO required; reuses `_make_env` etc. from `evaluate_rl_hard.py`)

### Frozen tiers verified untouched
* `artifacts/` (V1 baseline, 1.1 GB)
* `artifacts_64/` (legacy 64D, 1.1 GB)
* `artifacts_v2/` (V2 primary, 157 MB)
* `artifacts/rl_sweeps/` (V1 PPO sweep)

`find {artifacts,artifacts_64,artifacts_v2,artifacts/rl_sweeps} -newer
artifacts_v3/audit_v3a.md` returns empty for each frozen tier.

---

## 6. Which track is currently stronger?

**Track L is the only track with data.** It is *stronger on safety* (OOD
Pearson PASS, beam reach 100 %) and *weaker on V3 hypothesis* (greedy_dyn_2
saturates, more saturated than V2). Track N is still mid-training and will
take several more hours.

If a single-track verdict were required now: **Track L makes the V3.1
latent-dim hypothesis less likely, not more.** Track N must show
greedy_dyn_2 < 0.95 at primary to rescue the hypothesis.

---

## 7. Halt point per plan + user instruction

**Halting before A4 PPO.** Per `V3A_LATENT_AUDIT_AND_64D_PLAN.md` §10
("Do not train PPO until user approval after this checkpoint") and per the
user's explicit instruction in this session.

Awaiting user direction on:
* whether to let Track N VAE complete in background (~4–5 h) or kill it,
* whether to pivot to V3.3 (ZINB) / V3.4 (SCANVI) / V3.fallback now,
* whether to commit current `artifacts_v3/` to git or leave untracked.
