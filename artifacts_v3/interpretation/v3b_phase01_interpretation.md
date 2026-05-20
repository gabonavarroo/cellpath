# V3B Phase 0 + Phase 1 — Interpretation

**Date:** 2026-05-17
**Author:** V3 research lead (CC agent)
**Plan reference:** `V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md` §§ 0, 3, 4, 8, 9
**Sacred-rule conformance:** writes only under `artifacts_v3/`; `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/` untouched (verified via `git status`).
**Track N status:** still training (background PID 46735, ~5 h elapsed). Not interrupted.

---

## 1. Headline

* **Phase 0 (biology layer build): SHIPPED.** `artifacts_v3/v3b_biology/{gene_safety.parquet, k562_sl_pairs.parquet, coverage.json, README.md}` present and validated.
* **Phase 1 (post-hoc scoring of V2 primary): SHIPPED.** Verdict: **PROCEED** to Phase 2 (safety-aware reward retrain on 32D V2 primary, per the user-confirmed C → B → D → C+B → E sequential ablation order).

---

## 2. Phase 0 result summary

### 2.1 `gene_safety.parquet`
* 105 rows, **99 with non-null Chronos** (94.3 % coverage).
* **5 common-essential genes (Chronos < −0.5):** `CBFA2T3`, `HK2`, `PLK4`, `PTPN1`, `STIL`.
* 6 missing-Chronos genes (mostly renamed HGNC symbols): `C19orf26`, `C3orf72`, `ELMSAN1`, `KIAA1804`, `MAP4K5`, `SAMD1`. Treated as `tox_raw = 0` by the scorer.
* Chronos distribution on 105: mean −0.112, median −0.068, min −1.186 (likely PLK4 or HK2), max +0.328.

### 2.2 `k562_sl_pairs.parquet` — **structurally empty**
* Horlbeck-2018 supplementary Table S5 successfully fetched + parsed: **1,523 K562 SL pairs at GI < −3** (matches the SLKB-aggregated count exactly, validating our extraction).
* **0 / 1,523 pairs survive intersection with the Norman 105-gene action universe.** Horlbeck screened 459 essential / housekeeping genes; Norman selected 105 cell-fate transcription factors — by design, these are complementary, non-overlapping gene sets.
* **Plan implication:** the SL-pair penalty term in V3B reward Variant E (Phase 5b) is structurally inert on the current action space. Variant E reduces to B + C + D. Documented in `artifacts_v3/v3b_biology/README.md` §6.
* **Alternative SL sources considered and deferred:** DepMap co-essentiality (requires ~500 MB CRISPR_gene_effect.csv download, not in repo); SLKB / SynLethDB (same essential-gene focus, low likelihood of Norman coverage). A V3B follow-up session can attempt DepMap co-essentiality if Phase 5b otherwise underwhelms.

### 2.3 Scorer module
* `src/analysis/path_feasibility.py` — `load_biology_layer`, `score_episode`, `aggregate_episode_scores`. Pure-function, no side effects.
* `tests/test_path_feasibility.py` — 22 tests, all passing, including a real-disk-loader smoke test against `artifacts_v3/v3b_biology/`. Full suite: 284 passed / 2 skipped.

---

## 3. Phase 1 result summary

### 3.1 Verdict: **PROCEED**

The decisive question was *"does greedy_dyn_2 already have strictly better biology than PPO at primary?"*. **No** — the comparison is non-monotonic, opening a clear Phase 2 niche for the safety-aware reward.

### 3.2 Primary cell (K=3 / bin 8-10 / OOD), 4-seed mean

| Policy | Success | Wmean Chronos | Wmean tox_raw | Frac common-essential | Top-10 mean Chronos |
|---|---:|---:|---:|---:|---:|
| **ppo_deterministic** | **0.941** | −0.0718 | 0.0045 | **0.0140** | −0.0621 |
| greedy_dyn_1 | 1.000 | −0.1482 | 0.0025 | 0.0279 | −0.1394 |
| greedy_dyn_2 | 1.000 | −0.0989 | 0.0000 | **0.0000** | −0.1493 |
| random_uniform_valid | 0.170 | −0.0879 | 0.0116 | 0.0333 | −0.0437 |

**Nuance:** greedy_dyn_2 is **essential-avoidant by accident** at primary — 0.0 frac essential, but its weighted-mean Chronos (−0.099) is *more negative* than PPO's (−0.072). PPO occasionally picks the 5 common-essential genes (1.4 % of its actions) but uses **less Chronos-negative** genes overall. The safety reward can plausibly do two things at once: eliminate the 1.4 % essential picks **and** preserve PPO's lower-average-Chronos profile.

### 3.3 K=2 / bin 6-8 / OOD (the V2 informative cell)

| Policy | Success | Wmean Chronos | Frac common-essential |
|---|---:|---:|---:|
| **ppo_deterministic** | **0.748** | **−0.070** | **0.010** |
| greedy_dyn_1 | 0.837 | −0.049 | 0.018 |
| greedy_dyn_2 | 0.790 | −0.095 | 0.020 |

**Here PPO is unambiguously better on both biology axes** — fewer essential actions AND less Chronos-negative average. This is the cell at which V3B's safety reward has the highest leverage, because greedy_dyn_2 *cannot* fix its biology without a planning-time biology signal. *Phase 2 should report this cell as a primary success target.*

### 3.4 K=2 / bin 8-10 / OOD

PPO already picks fewer essential genes (1.9 %) than greedy_dyn_2 (0.0 %) — but again with a less-negative weighted-mean Chronos (−0.036 vs −0.078). Same pattern as primary cell; the safety reward should help even more here because the cell is unsaturated for greedy.

### 3.5 Per-episode training-rollout metrics (V2 PPO seeds, 500 episodes each)

| Seed | success | mean_n_gene_steps | mean_tox_path | frac_zero_CE | frac_zero_SL |
|---|---:|---:|---:|---:|---:|
| 42 | 1.000 | 1.22 | 0.0007 | 0.992 | 1.000 |
| 0 | 1.000 | 1.24 | 0.0003 | 0.996 | 1.000 |
| 1 | 0.998 | 1.22 | 0.0010 | 0.988 | 1.000 |
| 7 | 1.000 | 1.27 | 0.0047 | 0.954 | 1.000 |

**Seed variance** (`mean_tox_path`: 0.0003 → 0.0047, **15.6× spread**; `frac_zero_CE`: 0.954 → 0.996) is the lever for the safety reward. Seed 7 picks essential genes 4.6 % of episodes; seed 0 picks them 0.4 % of episodes — *V2 PPO has no notion of safety, and the seed picks essential-vs-safe arbitrarily within the same converged success rate.* A safety-aware reward should push **all four seeds** toward the seed-0 profile.

---

## 4. Updates to the V3B plan based on Phase 0 + 1 findings

These updates supersede the plan but do not contradict its strategy:

1. **§4 Variant E (full multi-objective reward)** — drop `λ_sl·sl_violations(path)` term because of the Horlbeck zero-overlap. Variant E ⇒ `R_T = 1·is_success − g(t) − λ_tox·tox_path − λ_unc·peak_unc`.
2. **§9.4 biological-breakthrough exemplar** — replace the now-trivial "mean SL violations ≤ 0.05" bullet with **"mean common-essential count per episode ≤ 0.02 (vs V2 PPO seed-7's 0.046)"**.
3. **§8.2 Phase 5b** — same correction; Variant E ⇒ B + C + D.
4. **§9.2 headline criterion 2 (safety-equivalent Pareto win)** — re-anchor on `frac_zero_CE` and `wmean_chronos` distance from V2 PPO's distribution, not on SL violations.
5. **Phase 2 emphasis** — the **K=2 / bin 6-8 / OOD cell** is the strongest expected lever (greedy_dyn_2 strictly worse on biology axes there). Phase 2 PPO-C should target headline success at that cell first.

The plan remains valid; these are post-evidence refinements documented here per CLAUDE.md §8 protocol.

---

## 5. What's preserved (sacred-rule conformance)

* `git status` shows `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/` clean (verified before write).
* No VAE retraining; no dynamics retraining; no PPO retraining in this session.
* All new outputs under `artifacts_v3/`.
* `config/dynamics.yaml::gate.*` thresholds untouched.
* `src/analysis/metrics.py` untouched (new scoring in `src/analysis/path_feasibility.py`).
* Track N VAE training (PID 46735) **not interrupted**; verified running before each subprocess that could have competed for resources.

---

## 6. Artifacts produced this session

```
artifacts_v3/v3b_biology/
├── gene_safety.parquet           # 105 rows, 99 with Chronos, 5 essential
├── k562_sl_pairs.parquet         # 0 rows (zero overlap with Norman 105)
├── coverage.json                  # provenance + caveats
└── README.md                      # full provenance + reproduction recipe

artifacts_v3/eval_v3b_posthoc/
├── aggregate_per_cell_per_policy.parquet   # 80 rows (4 seeds × 4 cells × 5 policies)
├── aggregate_per_cell_per_policy.csv       # same, CSV-friendly (drops list cols)
├── per_episode_training_rollouts.parquet   # 4 rows (one per seed, 500 eps each)
├── per_episode_training_rollouts.csv
├── posthoc_summary.md                       # the human-readable Phase 1 verdict
└── verdict.json                              # PROCEED/HALT + headline numbers

artifacts_v3/interpretation/
└── v3b_phase01_interpretation.md  (this file)

src/analysis/path_feasibility.py   # NEW: load_biology_layer, score_episode, aggregator
scripts/build_v3b_biology_layer.py # NEW: Phase-0 driver
scripts/posthoc_score_paths.py     # NEW: Phase-1 driver
tests/test_path_feasibility.py     # NEW: 22 unit tests
V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md  # NEW: 638-line approved plan
```

---

## 7. Next step (Phase 2)

Per user directive (2026-05-17): sequential ablation **C → B → D → C+B → E** is mandatory.

**Phase 2** is **C (safety-aware reward retrain on V2 primary 32D)**.

Pre-conditions (verify before starting Phase 2):
* `artifacts_v3/v3b_biology/gene_safety.parquet` exists and `coverage.json::gene_safety.n_with_chronos ≥ 99`. ✅
* `pytest tests/test_path_feasibility.py` passes. ✅
* V2 primary dynamics + 4 PPO checkpoints frozen and unchanged.

Phase 2 implementation requires (deferred to next session):
* `src/rl/biology_rewards.py` — pure reward functions for Variant C.
* `src/rl/reward.py` — add `safety_aware` branch to `reward_mode`.
* `src/rl/environment.py` — plumb `tox_path_so_far` accumulator + safety table.
* `scripts/train_rl_v3b.py` + `scripts/evaluate_rl_v3b.py` wrappers.
* `scripts/train_rl_v3b.py` — single-seed 1M-step retrain on V2 primary dynamics under reward C; expect ~5–10 min PPO wall-clock on CPU.
* `scripts/evaluate_rl_v3b.py` — full hardness frontier × {PPO_C, PPO_C_permuted_chronos, greedy_dyn_2_under_C, random, noop}.

Decisive Phase 2 acceptance:
1. `PPO_C − greedy_dyn_2_under_C ≥ +0.03 pp` at ≥ 1 cell (most likely K=2 / bin 6-8 / OOD per §3.3 above).
2. Cliff's δ on top-10 gene Chronos: PPO_C vs PPO_A ≤ −0.3 (PPO_C uses less essential genes).
3. Permuted-Chronos PPO performs strictly worse than real-Chronos PPO (real signal, not noise).

The plan halts before Phase 2 retraining in this session; awaiting user approval for the next session.
