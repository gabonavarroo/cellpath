# V3A Final Checkpoint — Both Tracks Complete, Halt Before PPO

**Date:** 2026-05-18
**Phases complete:** A0, A1 (Track L), A1-bg (Track N), A2 (Track L), A2-bg (Track N), A3 (Track L), A3-bg (Track N)
**Halt:** before PPO (A4) per user instruction and V3A plan §10
**Total wall-clock:** ~5h 30m (driven by Track N VAE training; everything else ≤ 15 min)

---

## 1. Headline result

Both 64D tracks (legacy reuse + fresh NB) produce dynamics fields where
**`greedy_dyn_2` saturates at 1.000 at the V2-equivalent primary cell**
(K=3 / ε=p25 / bin 8-10 / OOD). At the informative K=2 hardness-frontier
cells, both tracks are **more saturated than V2 32D**, not less.

**The V3.1 latent-dim hypothesis (a higher-dim latent exposes a less-saturated
field where PPO − greedy_dyn_2 ≥ +0.05 pp becomes achievable) is REJECTED on
both tracks.** Track N (fresh NB) is slightly less saturated than Track L
(legacy) at K=2 cells (∆ ≈ 5 pp), confirming the legacy VAE config was a
secondary variable, not the primary cause. The 64D geometry itself — across
both VAE configs — is the cause.

Per the V3A plan §7 stop conditions: **no hard stop triggered** (both tracks
PASS OOD Pearson ≥ 0.40 and beam reach ≥ 50 %). The hypothesis-test failure
is informative, not fatal, and per the plan §9 it triggers pivot to V3.3 /
V3.4 / V3.fallback rather than PPO training on a saturated field.

---

## 2. Track L (legacy 64D VAE reuse)

### 2.1 ε quantiles (recomputed on legacy 64D latents, 11 855 controls)

| Quantile | Track L | V2 32D ref |
|---|---:|---:|
| p10 | 2.918 | — |
| **p25** | **3.187** | 3.166 |
| p50 | 3.548 | 3.531 |
| p75 | 3.969 | — |
| p90 | 4.435 | — |
| mean / std | 3.639 / 0.662 | — |

**Finding:** 64D ε quantiles within 1 % of V2 32D — the centroid-relative
distance distribution is essentially preserved. Effective dimensionality of
the 64D scVI latent is much lower than nominal 64.

### 2.2 Pairs (current `build_pairs.py`, seed 42)

* `train_pairs.npz`: 38 958 pairs across 84 genes (same as V2)
* `val_pairs.npz`: 4 324 pairs (same as V2)
* `ood_pairs.npz`: 14 549 pairs across 21 held-out genes (bit-identical to V1/V2 OOD: AHR, ARRDC3, BAK1, CELF2, COL2A1, …)
* `combo_pairs.npz`: 35 995 pairs (same as V2)
* `pair_seed=42`, `ot_epsilon=0.05`, `pairing_method='ot'`

### 2.3 Pairing-noise diagnostic

| Statistic | Track L | V2 32D ref |
|---|---:|---:|
| median | **0.886** | 0.89 |
| mean | 0.859 | — |
| min / max | 0.518 / 0.994 | — |
| std | 0.108 | — |

**Finding:** 64D does NOT reduce OT pairing noise. The 0.89 floor in 32D was
not a dimensionality artifact.

### 2.4 Dynamics (RoR + corr 0.10, seed 42, MPS)

Training: early-stop at epoch 89 / 300 (`patience=35`), best epoch 54
(~6 min wall-clock). `selection_metric=gate_margin` →
`model.pt = model_best_gate.pt`. Best-gate and best-NLL coincided at the
same epoch — checkpoint recommendation flagged as
`keep_best_nll` warning (cosmetic; metrics identical).

| Metric | Track L | V2 RoR 32D | Δ |
|---|---:|---:|---:|
| val Pearson | 0.620 | 0.615 | +0.005 |
| val R² | 0.3994 | 0.398 | tied |
| **val MLP−ridge margin** | **+0.0043** | +0.0136 | **−0.0093 (worse 3×)** |
| val ridge Pearson | 0.6156 | 0.6011 | +0.015 (ridge captures *more*) |
| **OOD Pearson** | **0.515** | 0.516 | tied |
| OOD R² | 0.3022 | — | — |
| OOD MLP−ridge margin | +0.0470 | +0.077 | −0.030 |
| OOD ridge Pearson | 0.4675 | — | — |
| Uncertainty Spearman (val / OOD) | 0.805 / 0.738 | 0.245 (reported) | + (likely metric defn change) |
| Gate val passed | FAIL (0.0043 < 0.030) | FAIL (0.0136 < 0.030) | same pattern |
| OOD report passed (all checks) | PASS | PASS | same |

**OOD Pearson 0.515 ≥ 0.40 threshold: PASS.** Hard safety gate met. RoR
retraining lifts legacy 64D OOD Pearson from `state_linear_skip`'s 0.369 to
0.515 — the legacy architecture was the bottleneck, not the latent. But the
MLP−ridge margin is WORSE than V2: in 64D the ridge baseline captures more
of the structure (val ridge Pearson 0.616 vs V2's 0.601), leaving less
residual for the MLP to extract.

### 2.5 Reachability

| Cell | Track L reach | best_dist | V2 RoR 32D ref |
|---|---|---:|---|
| K=3 / bin 8-10 / OOD | **8/8 = 100 %** | 1.53 | 17/17 = 100 % |
| K=2 / bin 6-8 / OOD | **179/183 = 97.8 %** | 1.61 | (V2 K=3 only) |

Bin 8-10 OOD pool: only 8 cells in Track L (vs V2's 17). Tighter ε
distribution in 64D legacy means fewer controls fall in bin 8-10.

### 2.6 Greedy saturation (n_episodes=300, seed=42, deterministic)

| Cell | random | grd_1 | **grd_2** | grd_3 | V2 grd_2 |
|---|---:|---:|---:|---:|---:|
| K=2 / bin 6-8 / OOD | 0.173 | 0.963 | **0.963** | 0.943 | 0.790 |
| K=2 / bin 8-10 / OOD | 0.090 | 0.853 | **0.853** | 0.853 | 0.300 |
| K=3 / bin 6-8 / OOD | 0.397 | 1.000 | **1.000** | 1.000 | 1.000 |
| **K=3 / bin 8-10 / OOD (primary)** | 0.337 | 1.000 | **1.000** | 1.000 | 1.000 |

Wilson 95 % CIs available in `eval_v3a_hardness/legacy_baselines_seed42/summary.md`.

---

## 3. Track N (fresh 64D NB VAE)

### 3.1 Fresh 64D NB VAE training

* Stopped at epoch 380/400 by early-stop (patience 45 on `elbo_validation`); best score 1450.463
* Wall-clock: ~5h 23m on MPS
* z_reference_centroid norm 0.671 (Track L 0.71; comparable)

### 3.2 ε quantiles

| Quantile | Track N | Track L | V2 32D |
|---|---:|---:|---:|
| p10 | 3.003 | 2.918 | — |
| **p25** | **3.281** | 3.187 | 3.166 |
| p50 | 3.637 | 3.548 | 3.531 |
| p75 | 4.057 | 3.969 | — |
| p90 | 4.522 | 4.435 | — |
| mean / std | 3.736 / 0.684 | 3.639 / 0.662 | — |

Track N is ~3 % larger across the distribution than Track L; otherwise
similar. Hard-bench evaluations on Track N use Track N's own
ε_p25 = 3.281; on Track L, use Track L's ε_p25 = 3.187.

### 3.3 Pairs (current `build_pairs.py`, seed 42)

* `train_pairs.npz`: 38 958 (same)
* `val_pairs.npz`: 4 324 (same)
* `ood_pairs.npz`: 14 549 (same)
* `combo_pairs.npz`: 35 995 (same)
* `pair_seed=42`, `ot_epsilon=0.05`, `pairing_method='ot'` (same)
* Held-out genes: bit-identical to V1/V2/Track L

### 3.4 Pairing-noise diagnostic

| Statistic | Track N | Track L | V2 32D |
|---|---:|---:|---:|
| median | **0.894** | 0.886 | 0.89 |
| mean | 0.858 | 0.859 | — |
| min / max | 0.526 / 0.994 | 0.518 / 0.994 | — |
| std | 0.108 | 0.108 | — |

**Finding:** Pairing noise is essentially identical across all three (Track
L, Track N, V2). 64D + VAE config drift does NOT change the OT pairing
floor.

### 3.5 Dynamics (RoR + corr 0.10, seed 42)

Training: early-stop at epoch ~104 / 300, best_gate epoch 62, best_nll
epoch 49 (divergent). `selection_metric=gate_margin` → `model.pt =
model_best_gate.pt`. `checkpoint_comparison.json::rationale`: "best_gate
improves val mlp_minus_ridge_pearson (+0.0067 → +0.0074),
uncertainty_spearman=0.8023 ≥ 0.20, and OOD is within tolerance."

| Metric | Track N | Track L | V2 RoR 32D |
|---|---:|---:|---:|
| val Pearson | 0.621 | 0.620 | 0.615 |
| val R² | 0.404 | 0.399 | 0.398 |
| **val MLP−ridge margin** | **+0.0074** | +0.0043 | +0.0136 |
| val ridge Pearson | 0.6135 | 0.6156 | 0.6011 |
| **OOD Pearson** | **0.504** | 0.515 | 0.516 |
| OOD R² | 0.297 | 0.302 | — |
| OOD MLP−ridge margin | +0.035 | +0.047 | +0.077 |
| OOD ridge Pearson | 0.4687 | 0.4675 | — |
| Uncertainty Spearman (val / OOD) | 0.802 / 0.761 | 0.805 / 0.738 | — |
| Gate val passed | FAIL (0.0074 < 0.030) | FAIL (0.0043 < 0.030) | FAIL |
| OOD report passed | PASS | PASS | PASS |

**OOD Pearson 0.504 ≥ 0.40: PASS.** Track N is slightly cleaner
architecturally than Track L (val margin +0.0074 vs +0.0043) but slightly
weaker OOD (0.504 vs 0.515). Both 64D tracks lose to V2 32D on the
architecture margin.

### 3.6 Reachability

| Cell | Track N reach | best_dist | Track L reach | V2 32D ref |
|---|---|---:|---|---|
| K=3 / bin 8-10 / OOD | **15/15 = 100 %** | 1.21 | 8/8 = 100 % | 17/17 = 100 % |
| K=2 / bin 6-8 / OOD | **253/268 = 94.4 %** | 1.61 | 179/183 = 97.8 % | — |

Track N bin 8-10 OOD pool: 15 cells (Track L: 8; V2: 17). Slightly wider ε
distribution captures more bin-8-10 cells than Track L.

### 3.7 Greedy saturation

| Cell | random | grd_1 | **grd_2** | grd_3 | Track L grd_2 | V2 grd_2 |
|---|---:|---:|---:|---:|---:|---:|
| K=2 / bin 6-8 / OOD | 0.167 | 0.913 | **0.907** | 0.880 | 0.963 | 0.790 |
| K=2 / bin 8-10 / OOD | 0.110 | 0.807 | **0.807** | 0.667 | 0.853 | 0.300 |
| K=3 / bin 6-8 / OOD | 0.383 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 |
| **K=3 / bin 8-10 / OOD (primary)** | 0.280 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 |

**Track N is slightly less saturated than Track L at K=2 cells** (∆ ≈ 5 pp):
0.907 vs 0.963 at K=2/bin 6-8, 0.807 vs 0.853 at K=2/bin 8-10. **K=3 cells
fully saturate on both tracks.** This narrows the VAE-config explanation:
the legacy VAE config makes the field marginally easier for greedy
planning, but the **fundamental K=3 saturation is inherent to 64D scVI on
Norman 2019 K562**, not to any VAE-config detail.

Notable: Track N `greedy_dyn_3` ≤ `greedy_dyn_2` at K=2 cells (0.667 < 0.807
at K=2/bin 8-10), with a larger drop than Track L (0.853 = 0.853). This is
the receding-horizon weakness when search depth exceeds the env step
budget; Track N's planning surface appears slightly "rougher" at the
2-step neighborhood, but still well within the saturation regime.

---

## 4. Which track is stronger?

**Neither track supports the V3 latent-dim hypothesis.** Comparison by
criterion:

| Criterion | Track L | Track N | Winner |
|---|---|---|---|
| OOD Pearson (hard gate ≥ 0.40) | 0.515 | 0.504 | Track L (+0.011) |
| Val MLP−ridge margin (target > V2's +0.0136) | +0.0043 | +0.0074 | Track N (+0.0031) |
| Beam reach K=3/bin 8-10/OOD (≥ 50 %) | 100 % | 100 % | tied |
| Beam reach K=2/bin 6-8/OOD (≥ 50 %) | 97.8 % | 94.4 % | Track L (+3.4 pp) |
| **grd_2 saturation at K=3 primary** (lower = better for V3 hypothesis) | 1.000 | 1.000 | **TIED — both fully saturated; both REJECT V3 hypothesis** |
| grd_2 K=2/bin 6-8 (lower = better for V3 hypothesis) | 0.963 | **0.907** | Track N (less saturated, by 5.6 pp) |
| grd_2 K=2/bin 8-10 (lower = better for V3 hypothesis) | 0.853 | **0.807** | Track N (less saturated, by 4.6 pp) |
| Reproducibility (Hydra snapshot, current scvi-tools) | weak (legacy, no snapshot) | strong (current) | Track N |

**Track N is the cleaner reference for any V3.1 follow-up.** Track L is
informative as a "legacy VAE config doesn't dramatically change the
result" control. Both tracks consistently REJECT the V3.1 latent-dim
hypothesis at the K=3 primary cell. Neither offers PPO any advantage room
to claim a +0.05 pp improvement over greedy_dyn_2.

---

## 5. Stop-condition check

| Stop condition | Track L | Track N | Both? |
|---|---|---|---|
| OOD Pearson < 0.40 | PASS (0.515) | PASS (0.504) | NO — passes |
| Beam reach K=3/bin 8-10/OOD < 50 % | PASS (100 %) | PASS (100 %) | NO — passes |

**No hard stop triggered.** Both tracks meet safety + controllability
criteria. The V3.1 hypothesis-test failure (greedy_dyn_2 saturation) is
informative, not fatal, and triggers the V3 stub §5 pivot agenda rather
than a halt.

---

## 6. Honest framing of the V3A result

### Allowed claims

* **Track N dynamics is V2-comparable in raw quality**: val Pearson 0.621
  ≈ V2's 0.615; OOD Pearson 0.504 ≈ V2's 0.516.
* **64D geometry is MORE saturated for greedy planning than V2 32D**, both
  with the legacy VAE config (Track L) and with the current NB config
  (Track N). At K=2 / bin 8-10 / OOD, greedy_dyn_2 jumps from V2's 0.300
  to Track L's 0.853 / Track N's 0.807.
* **The supervised gate's MLP−ridge margin in 64D is WORSE than V2 32D**
  (Track L +0.0043, Track N +0.0074, V2 +0.0136). In 64D scVI on Norman
  2019, the ridge baseline captures more of the residual structure,
  leaving less for the MLP.
* **OT pairing noise floor is invariant** to latent dim and VAE config
  (medians 0.89 / 0.886 / 0.894 for V2 / Track L / Track N).
* **The legacy VAE config and the fresh NB VAE produce nearly identical
  V3.1 outcomes** at K=3 primary. Track N modestly relaxes the K=2
  saturation (∆ ≈ 5 pp) but does not change the qualitative conclusion.

### Disallowed claims

* "V3.1 succeeds" — false on both tracks. greedy_dyn_2 saturates at K=3
  primary.
* "64D is harder geometry" — false on this dataset. 64D scVI on Norman 2019
  is *easier* geometry for greedy planning than 32D.
* "Track L was the bottleneck" — partially false. Track N modestly improves
  K=2 saturation but does not rescue the K=3 primary cell.
* "PPO would succeed if trained" — implausible. PPO has no information
  advantage over a 2-step beam oracle in either track.

### Implication for next phases

Per the V3A plan §9 and V3 stub §5, V3.1 (any 64D config) is
**provisionally rejected**. Recommended next moves:

1. **Do NOT train PPO on Track L or Track N.** Both fields are saturated;
   PPO would either match the oracle (V2 outcome repeated) or be cut off
   by environment time limits without learning a useful policy.
2. **Pivot to V3.3 — 64D ZINB**: different gene likelihood; tests whether
   NB → ZINB changes the planning geometry. Same compute envelope as
   Track N (~6 h end-to-end).
3. **Pivot to V3.4 — SCANVI 32D**: semi-supervised; tests whether
   perturbation-label supervision produces a more separable, less
   saturable space. ~7 h end-to-end.
4. **V3.fallback.B — Contraction regulariser**: explicit penalty on
   contraction rate in dynamics training; orthogonal to latent geometry.
   Could be tested on Track N (existing pairs) for ~30 min.

The V3 success criterion (PPO − grd_2 ≥ +0.05 pp at one reachable cell)
remains achievable but requires a dynamics field where grd_2 < 1.0 at the
primary cell. Neither Track L nor Track N provides that field.

---

## 7. Artifact inventory (V3A complete)

### Created under `artifacts_v3/` (all V3A writes)

* `audit_v3a.md` — A0 audit verdict
* `vae_n64_legacy/` — 1.0 GB copy of `artifacts_64/vae/`; `epsilon_success.json` recomputed (p10/p25/p50/p75/p90)
* `vae_n64_nb/` — fresh 64D NB VAE (full Hydra-driven; model + latents + centroid + ε)
* `pairs_n64_legacy/{train,val,ood,combo}_pairs.npz`, `metadata.json`, `pairing_noise.json`
* `pairs_n64_nb/{train,val,ood,combo}_pairs.npz`, `metadata.json`, `pairing_noise.json`
* `dynamics_n64_legacy_ror_corr010/` — full dynamics artifacts (model.pt + best_nll/best_gate + ridge_baseline.npz + gate.json + val/ood metrics + epoch_metrics + checkpoint_comparison)
* `dynamics_n64_nb_ror_corr010/` — full dynamics artifacts (same structure)
* `dynamics_n64_legacy_ror_corr010_training.log`, `dynamics_n64_nb_ror_corr010_training.log`
* `reachability_probe_v3a/{legacy,nb}_k3_bin810_ood/`, `reachability_probe_v3a/{legacy,nb}_k2_bin68_ood/`
* `eval_v3a_hardness/{legacy,nb}_baselines_seed42/` — greedy saturation tables + summary.md per track
* `eval_v3a_hardness/{legacy,nb}_baselines_seed42.log`
* `interpretation/v3a_checkpoint.md` (post-Track-L checkpoint), `interpretation/v3a_final.md` (this file)
* `pairs_n64_legacy_build.log`, `pairs_n64_nb_build.log`, `vae_n64_nb_training.log`

### Created elsewhere

* `config/paths.yaml` — additive V3 path keys (`v3_legacy_*`, `v3_nb_*`, `v3_reach_dir`, `v3_eval_dir`, `v3_figures_dir`, `v3_interpretation_dir`)
* `V3A_LATENT_AUDIT_AND_64D_PLAN.md` — V3A plan at repo root
* `scripts/evaluate_baselines_only.py` — baselines-only evaluator helper (no PPO required)

### Frozen tiers — verified untouched by V3A

* `artifacts/` (V1 baseline, 1.1 GB): no new files from my pipeline.
  Tensorboard logs in `artifacts/tensorboard/rl_*/` are from a parallel
  V3B PPO session by the user (not my work); directory is gitignored.
* `artifacts_64/` (legacy 64D, 1.1 GB): 0 new files.
* `artifacts_v2/` (V2 primary, 157 MB): 0 new files.
* `artifacts/rl_sweeps/` (V1 PPO frozen): 0 new files.

`find {artifacts_64,artifacts_v2,artifacts/rl_sweeps} -newer
artifacts_v3/audit_v3a.md` returns empty for each tier.

---

## 8. Halt point per plan + user instruction

**Halting before A4 PPO.** Per `V3A_LATENT_AUDIT_AND_64D_PLAN.md` §10 and
the user's explicit instruction in this session.

Decision points awaiting user direction:

1. **Pivot agenda:** V3.3 (ZINB) / V3.4 (SCANVI) / V3.fallback (contraction
   regulariser, ensemble dynamics). Recommended: V3.fallback first (cheapest,
   orthogonal); V3.4 SCANVI second (most theoretically motivated); V3.3 ZINB
   third (least likely to help given pairing-noise invariance).
2. **PPO on either track:** strongly discouraged given grd_2 saturation, but
   could be run as a "negative control" to confirm V2 outcome repeats.
3. **Commit V3A artifacts to git:** `config/paths.yaml`,
   `V3A_LATENT_AUDIT_AND_64D_PLAN.md`, `scripts/evaluate_baselines_only.py`,
   `artifacts_v3/interpretation/*.md`. The large model files and latents
   are gitignored by `.gitignore` patterns.
