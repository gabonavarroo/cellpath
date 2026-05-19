# V3B Phase 2b — Leakage & Split-Strictness Audit

**Date:** 2026-05-18
**Author:** V3 research lead (CC agent, audit pass)
**Audit scope:** classify Phase 2 metrics by reward-fit vs reward-independent vs held-out, audit start-pool split strictness, reinterpret the V3B Phase 2 ACCEPT verdict accordingly.
**Sacred-rule conformance:** writes only under `artifacts_v3/eval_v3b_phase2b/` and `artifacts_v3/interpretation/`. `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/` untouched. **Phase 2 artifacts not modified**.

---

## 1. Final verdict

**`PHASE2_VALID_REWARD_FIT_NEEDS_SEED_ESCALATION`**

Split-strictness sub-verdict: **`DEV_BENCHMARK_ONLY_OOD_START_LEAKAGE`**
Held-out biological validation sub-verdict: **`pending_no_local_source`**

**One-paragraph plain-language summary.** Phase 2's reward-fit claims (zero common-essential picks, near-zero `mean_tox_path`, real-Chronos beats permuted-Chronos on safety-adjusted SR) are all correct by construction — they are *what the policy was trained to do*, not biological discovery. The reward-independent claim that survives audit is **+4.0 pp PPO_C raw success at K=2 / bin 8-10 / OOD** (vs PPO_A and vs greedy_dyn_2 both). This is single-seed; V2's historical seed-std at this cell was 0.045, so the +0.040 signal sits squarely inside the seed-noise window. The "OOD" label of the hard benchmark refers to the **dynamics model's gene-action extrapolation**, not to start-state distribution novelty — PPO training saw 14 549 OOD-gene-perturbed cells (14.6 % of its training start pool), which is the same V2 protocol but should be acknowledged as a *development* hard benchmark, not a *blind* OOD result. **No local biological source is available for independent (Bucket C) validation.** Recommendation: run a **4-seed escalation of PPO_C** before any new axis (Phase 3 path-length, Phase 4 uncertainty). Until 4-seed CI excludes zero at K=2 / bin 8-10 / OOD, the V3B Phase 2 ACCEPT verdict is provisional.

---

## 2. Source-usage classification (full table: `source_usage_table.md`)

### 2.1 Sources used inside the reward (Bucket A inputs)

| Source | In reward? | In training? | In eval? | Allowed Phase 2 claim |
|---|---|---|---|---|
| DepMap K562 Chronos (`data/processed/depmap_k562_chronos.parquet`) | ✅ via `tox_raw = max(0, −Chronos − 0.5)` and the `is_essential = Chronos < −0.5` flag | ✅ | ✅ (identical table loaded by `evaluate_rl_v3b.py`; `greedy_dyn_2_C` also uses it) | Reward-prior optimization, not biological discovery |
| Common-essential labels (Chronos < −0.5) | ✅ as `λ_ce · common_essential_count` | ✅ | ✅ | Same as Chronos |

### 2.2 Sources not used in reward, available locally

| Source | Status | Why not load-bearing in V3B |
|---|---|---|
| Norman 2019 K562 CRISPRa Perturb-seq | The training dataset itself; cannot be independent | Circular |
| Horlbeck 2018 K562 SL pairs | Structurally absent (0/1 523 overlap with Norman 105 genes; see `v3b_biology/coverage.json`) | The reward and eval are both blind to it |

### 2.3 Held-out biological sources (Bucket C) — NOT available locally

| Source | Why it would help | Local availability |
|---|---|---|
| Replogle 2022 K562 essential CRISPRi Perturb-seq | K562-specific, complementary direction (CRISPRi vs Chronos's CRISPR-Cas9) — strongest candidate for cross-source validation | ❌ (~600 MB download from figshare 20029387) |
| OGEE v3 essentiality | Cross-cell-line, broader than DepMap | ❌ (download from v3.ogee.info) |
| COSMIC Cancer Gene Census tier-1/2 | Oncogene / TSG annotations | ❌ (paywalled) |
| Open Targets tractability | Druggability tier per gene | ❌ + pharma-modality mismatch (low value for CRISPRa) |

**No Bucket C validation is possible in this audit.** A follow-up session can pull Replogle for a clean independent test.

---

## 3. Metric classification (Buckets A / B / C)

(See `source_usage_table.md` §2 for the full grid; this section calls out the load-bearing ones.)

### Bucket A — reward-fit metrics (cannot be cited as biological discovery)

* `mean_tox_path`, `mean_common_essential_per_ep`, `weighted_mean_chronos`, `weighted_mean_tox`, `fraction_actions_common_essential`, `fraction_zero_common_essential` — all derived from DepMap Chronos, which is in the reward.
* `safety_adjusted_success_rate` — built from `common_essential_count`, which is reward-fit.
* `realism` (composite) — also reward-fit because its `chr` and `ess` terms come from Chronos.

### Bucket B — reward-independent metrics (genuine claims, modulo seed variance and dynamics-OOD caveats)

* `success_rate` (raw)
* `mean_steps`
* `mean_final_distance`
* Deltas: PPO_C − PPO_A raw, PPO_C − greedy_dyn_2_C raw, PPO_C − greedy_dyn_2_A raw, real-Chronos PPO_C raw − permuted PPO_C raw
* Beam reachability oracle (V2 protocol)

### Bucket C — held-out biological validation

* Empty. Pending Replogle / OGEE / CGC / Open Targets local availability.

---

## 4. Reward-independent (Bucket B) deltas

From `leakage_safe_deltas.csv`:

| Cell | PPO_C − PPO_A raw | PPO_C − greedy_dyn_2_C raw | PPO_C − greedy_dyn_2_A raw | real − permuted raw |
|---|---:|---:|---:|---:|
| K=2 / bin 6-8 / OOD | **−0.037** | **−0.053** | **−0.053** | −0.037 |
| **K=2 / bin 8-10 / OOD ⭐** | **+0.040** | **+0.040** | **+0.040** | **+0.073** |
| K=3 / bin 6-8 / OOD | 0.000 | −0.003 | −0.003 | −0.003 |
| K=3 / bin 8-10 / OOD (V2 primary, saturated) | −0.060 | −0.060 | −0.060 | −0.060 |

**Observations:**

1. The +4.0 pp headline at K=2 / bin 8-10 / OOD is **the only positive Bucket-B cell**.
2. At K=2 / bin 6-8 / OOD, PPO_C is **strictly worse than PPO_A** on raw success by 3.7 pp — a Bucket-B counterexample to the headline. This was masked by the safety-adjusted reframing because Phase 2's accepted Rule 1 looked at safety-adjusted, not raw, deltas.
3. At K=3 / bin 8-10 / OOD (the V2 primary, saturated cell), PPO_C loses 6 pp vs both PPO_A and greedy_dyn_2.
4. The real-vs-permuted delta at K=2 / bin 8-10 / OOD (+7.3 pp) is the **largest** and **cleanest** reward-independent signal — *the label structure is functionally load-bearing*. But the comparison is still single-seed and still does not validate Chronos biologically.

The reward-independent picture is therefore mixed: 1 positive cell, 1 negative cell, 1 saturated, 1 with small regression at the saturated primary. The single-seed Phase 2 verdict was over-stated when collapsed to "ACCEPT".

### 4.1 Seed-variance reality check

V2 (`artifacts_v2/V2_FINAL_REPORT.md` §3) reports for the C2 configuration on the V2 primary RoR_corr010 dynamics at K=2 / bin 8-10 / OOD: **success = 0.283 ± 0.045** (4-seed std). At K=2 / bin 6-8 / OOD: **0.748 ± 0.053**. So:

* The Phase 2 single-seed PPO_C value at K=2 / bin 8-10 / OOD = **0.340** is within ~1.3 σ of PPO_A's 4-seed mean (0.300), with single-seed PPO_A 0.300 also matching the V2 4-seed mean exactly. The +0.04 single-seed delta is well within the seed-noise window.
* The Phase 2 single-seed PPO_C value at K=2 / bin 6-8 / OOD = **0.737** is within ~0.3 σ of V2 C2 mean 0.748 — the −3.7 pp regression vs PPO_A (0.773) is also within seed noise.

Both Bucket-B signals are inside V2's measured noise band. **4-seed escalation is the load-bearing next step** before any V3B claim is published.

---

## 5. Start-pool split-strictness audit

### 5.1 What the code does

* `src/rl/environment.py::_build_start_pool` (line 501): `pool = Z[pert_idx != 0]` — keeps all perturbed cells across all 105 genes. **No held-out-gene filter.**
* `src/rl/curriculum.py::DistanceCurriculumCallback._on_training_start` (line 107): `self._raw_pool = Z[pert_idx != 0]` — same. **No held-out-gene filter.**
* `evaluate_rl_v3b.py::_load_start_pool`: filters `pert_idx != 0 AND perturbation IN held_out_genes`. **Strict OOD-gene filter at eval only.**
* PPO_C trained for 1 M timesteps via `train_rl_v3b.py` → delegates to `train_rl.py` → `make_env_factory` → `_build_start_pool` → all-perturbed pool.

### 5.2 Quantification of the leakage

```
Total perturbed cells in artifacts/vae/latents.h5ad:   99 590
Cells perturbed by the 21 OOD-held-out genes:          14 549 (14.6 %)
Cells perturbed by train genes:                        85 041 (85.4 %)
```

The 21 OOD genes are: **AHR, ARRDC3, BAK1, CELF2, COL2A1, DLX2, DUSP9, FOSB, FOXL2, FOXO4, IER5L, ISL2, KLF1, KMT2A, MAP2K6, RHOXF2, RUNX1T1, SAMD1, SGK1, UBASH3B, ZNF318.**

**Three of these — RUNX1T1, ZNF318, FOXO4 — appear in V2 primary's top-10 action_freq** (see `posthoc_summary.md` for PPO_A action_freq at the primary cell). PPO has *learned to pick OOD-gene actions* on start states resulting from those same OOD-gene perturbations.

### 5.3 What the "OOD" label of the hard benchmark actually tests

The hard-benchmark OOD-ness is at the **dynamics model's gene-action extrapolation** level, NOT at the **start-state distribution** level:

* Dynamics model `RoR_corr010` was trained on `artifacts_v2/dynamics_v1ot_ror_corr010/` using `train_pairs.npz` (84 train genes only). For any of the 21 OOD genes, the dynamics' `μ(z, g_ood)` prediction is extrapolated — never directly fit.
* PPO_C training uses the dynamics' extrapolated predictions on a mixed start pool that includes OOD-gene-perturbed cells. The OOD signal that PPO encounters is in the dynamics' predictions, not in the start states themselves.

This is the V2 hard-bench protocol; V3B inherits it. The PPO_C vs PPO_A relative comparison is internally consistent because both saw the same start-state distribution.

### 5.4 Verdict

**`DEV_BENCHMARK_ONLY_OOD_START_LEAKAGE`**.

The hard-bench should be described as a **development hard benchmark**, not a **blind OOD result**. The dynamics-action OOD generalization is the real test in play. The PPO_C vs PPO_A relative claims stand.

A strict-start-pool retrain (excluding the 14 549 OOD-gene cells from training) would test **state-distribution generalization** — a stronger and orthogonal claim. It is not the load-bearing fix for Phase 2; it is a future enhancement.

---

## 6. Reinterpretation: what Phase 2 can honestly claim, after audit

### 6.1 Held claims (Bucket A — reward-fit, expected by construction)

* "PPO_C optimizes the V3B safety-aware reward better than greedy_dyn_2_C" (a planning-advantage claim on the *given* objective).
* "PPO_C eliminates common-essential picks (mean CE/ep = 0) across all cells" — reward-prior optimization.
* "Real-Chronos PPO_C beats permuted-Chronos PPO_C on safety-adjusted success rate" — confirms label structure is load-bearing, but reward-fit.

### 6.2 Held claims (Bucket B — reward-independent, single-seed)

* "**At K=2 / bin 8-10 / OOD (development hard benchmark), PPO_C achieves +4.0 pp raw success vs PPO_A AND vs greedy_dyn_2_C — single seed, within V2 4-seed CI of zero. 4-seed escalation pending.**"
* "Real-Chronos PPO_C raw success exceeds permuted-Chronos PPO_C raw success by +7.3 pp at K=2 / bin 8-10 / OOD — cleanest reward-independent signal, still single-seed."
* "PPO_C raw success is **lower** than PPO_A by 3.7 pp at K=2 / bin 6-8 / OOD — a Bucket-B counterexample to the headline."
* "At the saturated K=3 / bin 8-10 / OOD primary cell, PPO_C raw 0.940 vs PPO_A raw 1.000 — 6 pp regression, expected trade-off at a saturated cell."

### 6.3 Withdrawn / downgraded claims

* The Phase 2 interpretation's framing of the "first V3-era result where PPO strictly exceeds the depth-2 model-based oracle" is **technically true on Bucket B (raw success)** but **single-seed and not statistically established**. The 4-seed escalation is required before publication.
* The "biological breakthrough" exemplar (zero CE picks, low tox, etc.) was reward-prior optimization; reframe accordingly in any future V3B writeup.
* No "PPO_C is biologically validated" claim is allowed until a Bucket-C source (Replogle / OGEE) is in play.

---

## 7. Recommended next action

**Primary:** run **4-seed escalation of PPO_C** (seeds {42, 0, 1, 7} × real-Chronos + permuted-Chronos = 8 PPO retrains, each ~3.4 min at 1 M timesteps; plus 4 evals × ~1.7 min) before any new reward variant. Expected wall-clock: **~35 min**.

Acceptance for the 4-seed escalation:
1. 4-seed 95 % normal CI on `PPO_C − PPO_A` raw success at **K=2 / bin 8-10 / OOD** strictly excludes zero.
2. 4-seed CI on `real − permuted` raw success at the same cell strictly excludes zero.
3. The K=2 / bin 6-8 / OOD regression on raw success is bounded (4-seed mean ≥ PPO_A − 0.05).

If all three pass → Phase 2 headline is statistically established; **Phase 3 path-length B is unlocked**.

If the seed-CI straddles zero → either revise λ_tox / λ_ce or **proceed to Phase 3 first** and revisit C in conjunction (the C+B Phase 5 may stabilize what single-axis C alone could not).

**Secondary (parallel, optional):** download Replogle 2022 K562 essential CRISPRi Perturb-seq processed table (~600 MB from figshare 20029387) and build a Bucket-C scorer. This unlocks the first independent biological validation and would let us move from `PHASE2_VALID_REWARD_FIT_NEEDS_SEED_ESCALATION` to `PHASE2_VALID_WITH_HELDOUT_SUPPORT_NEEDS_SEED_ESCALATION` (or higher) on the next pass.

**Tertiary (deferred):** the strict-start-pool retrain (PPO_C with train-gene-only start pool, 85 041 cells) is a future enhancement, NOT a Phase 2b blocker.

### 7.1 Why not start Phase 3 immediately

Phase 3 path-length B builds on Phase 2's V2-primary dynamics-and-curriculum scaffold. If Phase 2's single-seed headline is noise, then a Phase 5 C+B conjunction inherits an unstable axis-C signal. Locking the 4-seed result first prevents this cascade and makes Phase 5's conjunction claim defensible.

### 7.2 Ideal alternative (consolidated)

1. Run the 4-seed escalation (35 min).
2. Download Replogle K562 essentials and build a Bucket-C scorer (≈ 20–30 min).
3. Re-evaluate: at this point we have both seed-CI excluding zero AND a held-out biology source confirming the Chronos prior generalizes.
4. *Then* proceed to Phase 3.

Cost: ~1 h of wall-clock. Output: a publishable V3B Phase 2 result.

---

## 8. Sacred-rule conformance

```
git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/   # → clean
ls artifacts_v3/eval_v3b_phase2/                                              # → unmodified
ls artifacts_v3/rl_v3b_safety_aware_v2primary_seed42*                         # → unmodified
```

* No VAE, dynamics, or PPO retraining performed this session.
* No edits to Phase 2 code paths; this audit is read-only over Phase 2 artifacts.
* All new outputs under `artifacts_v3/eval_v3b_phase2b/` and `artifacts_v3/interpretation/v3b_phase2b_leakage_split_audit.md`.

---

## 9. Files produced this session

```
artifacts_v3/eval_v3b_phase2b/
├── source_usage_table.md          # Full source × usage classification
├── leakage_safe_summary.csv        # 28 rows × 14 cols (A/B-tagged metrics)
├── leakage_safe_deltas.csv         # 4 rows × 5 cols (per-cell Bucket-B deltas)
└── phase2b_verdict.json            # Final verdict + sub-verdicts + recommended next

artifacts_v3/interpretation/
└── v3b_phase2b_leakage_split_audit.md   # (this file)
```

No code changes. Phase 2 PPO checkpoints, eval summaries, and interpretation file remain as-is.
