# VAE Retraining — Diagnostic Report

**Date:** 2026-05-18
**Author:** investigation triggered by user question "is retraining the VAE worth it?"
**Scope:** evidence-driven audit of the three VAE-relevant artifact families on disk
(`artifacts/vae/` = 32D V1, `artifacts_64/vae/` = 64D legacy, `artifacts_v3/...` = V3A work)
plus the V2 experiment trail under `artifacts_v2_experiments/`.

**Headline:** **Do NOT retrain the VAE.** The repo's own prior audit
([artifacts_v3/interpretation/v3a_checkpoint.md](../artifacts_v3/interpretation/v3a_checkpoint.md))
has already answered this question with stronger evidence than any new smoke
sweep would produce. The "artifacts_64 failed" framing is incorrect: scVI
training was fine, the legacy dynamics architecture was the culprit, and
swapping in V2's RoR architecture brings 64D to **parity with — not above —**
the V1 32D pipeline. The actual unlock for V3-era results came from changing
the reward (V3B safety-aware retrain), not the latent.

---

## 1. What's actually on disk

There is no separate "v2 VAE." V2 reused the V1 32D scVI checkpoint — the
"v2" applies to dynamics + RL only. Confirmed in
[config/paths.yaml](../config/paths.yaml) (`vae_dir: ${paths.artifacts}/vae`,
under the V1 tier) and in all `artifacts_v2_experiments/*.log` lines such as:

```
Latents h5ad: /…/cellpath/artifacts/vae/latents.h5ad
Loaded latents …  shape (111445, 32)
PerturbationDynamicsModel built | n_latent=32 | n_genes=105
```

So the genuine artifact families are:

| Artifact | n_latent | scVI config (verified) | Status |
|---|---|---|---|
| [artifacts/vae/](../artifacts/vae/) | 32 | NB, `dispersion=gene`, `latent_distribution=normal`, `n_layers=2`, `n_hidden=128` | V1 baseline, FROZEN. ε p90 = 4.5196 on 11,855 control cells. Used by V1 + V2 pipelines. |
| [artifacts_64/vae/](../artifacts_64/vae/) | 64 | NB, `dispersion=gene`, `latent_distribution=normal`, `n_layers=2`, `n_hidden=128` (verified by V3A audit) | Legacy ablation, FROZEN. ε p90 = 4.4347. |
| [artifacts_v3/vae_n64_legacy](../artifacts_v3/) | 64 (copy of `artifacts_64`) | same as above | Intermediate workspace; cleaned up — only `interpretation/` survives. |
| `artifacts_v3/vae_n64_nb` (planned "Track N" — fresh 64D NB retrain) | 64 | NB, V2 defaults | **Never finished.** Started, last seen at epoch ~43/400 (~5h MPS), then abandoned when V3B started. |

Training logs in [outputs/](../outputs/) (`outputs/2026-05-12/22-52-23/train_vae.log`
through `outputs/2026-05-13/17-30-53/train_vae.log`) confirm the V1 32D VAE
trained successfully (`Training complete.` at 2026-05-13 07:28). No new VAE
training logs exist beyond that date — every subsequent invocation hits the
existing-checkpoint guard.

---

## 2. What "artifacts_64 didn't work" actually means

The narrative "we trained at n_latent=64 and it didn't work" is supported by
[artifacts_64/dynamics/gate.json](../artifacts_64/dynamics/gate.json) (gate
FAIL, val Pearson 0.596, OOD Pearson 0.369). But this conflates a
**dynamics-architecture failure** with a VAE failure. The evidence below shows
the VAE itself was fine; the legacy dynamics model was wrong.

### 2.1 scVI checkpoint itself is healthy

From the V3A Phase A0 audit (loaded the model and inspected the module):

* `n_latent=64`, `gene_likelihood='nb'`, `dispersion='gene'`,
  `latent_distribution='normal'`, `n_layers=2`, `n_hidden=128` — standard V2-style config.
* Gene vocabulary **bit-identical** to V1's 105-gene set.
* Centroid `z_reference_centroid.npy` has shape `(64,)`, plausible norm.
* ε quantiles on 64D control cells are within **1%** of V2 32D:
  * p25: 3.187 (64D) vs 3.166 (32D)
  * p50: 3.548 vs 3.531
  * The expected √(64/32) ≈ 1.4× distance scaling **did not materialize** —
    the effective dimensionality of the 64D scVI latent is much lower than its
    nominal dimensionality.
* OT-pairing noise (per-gene residual variance / total Δz variance):
  * median 0.886 (64D) vs ~0.89 (32D) — essentially identical.

In other words: ε, centroid, pairing noise, and gene vocab all confirm that
the 64D latent is well-behaved and downstream-compatible. The VAE is not the
bottleneck.

### 2.2 The dynamics model in `artifacts_64` is the wrong architecture

The dynamics checkpoint at
[artifacts_64/dynamics/config.json](../artifacts_64/dynamics/config.json) uses
`use_state_linear_skip=true, use_gene_delta_bias=false` — that is, the **legacy**
dynamics scaffolding. V2's primary architecture is RoR (`use_residual_over_ridge=true,
lambda_corr=0.10, use_state_linear_skip=false`).

When the V3A audit retrained dynamics on the same 64D latent using **V2's RoR
architecture** (writing to `artifacts_v3/dynamics_n64_legacy_ror_corr010`), the
result was:

| Metric | 64D RoR (audit) | 32D RoR (V2 primary reference) | Δ |
|---|---:|---:|---:|
| val Pearson | **0.620** | 0.615 | +0.005 (tied) |
| val R² | 0.399 | 0.398 | tied |
| **OOD Pearson** | **0.515** | 0.516 | **tied** |
| OOD R² | 0.302 | — | — |
| val MLP − ridge Pearson margin | +0.0043 | +0.0136 | **−0.0093 (3× worse)** |
| OOD MLP − ridge Pearson margin | +0.0470 | +0.077 | −0.030 (worse) |
| Gate passed? | FAIL | FAIL | same (both fail on the same ridge-margin threshold) |
| Uncertainty calibration (val Spearman) | 0.805 | (calibration metric definition differs) | OK in both |

**Two key reads:**

1. **64D ≈ 32D on raw quality.** The 64D RoR is not a downstream regression;
   val and OOD Pearson are within ~0.005 of the V2 32D primary.
2. **64D is architecturally *worse* for the dynamics gate.** The ridge baseline
   is *stronger* in 64D (val ridge Pearson 0.616 vs V2's 0.601, +0.015),
   leaving less residual signal for the MLP. The gate margin shrinks from
   +0.0136 to +0.0043 — the gate is *farther* from passing in 64D, not closer.

The original `artifacts_64/dynamics/gate.json` failure (margin = −0.019) was
worse than this only because the legacy `state_linear_skip` architecture is
itself a worse dynamics model. The 32D V2 RoR also fails the gate (margin =
+0.014 < 0.030 threshold). **Failing this gate is a property of the
ridge-vs-MLP comparison in this OT-pairing regime, not of the latent
dimensionality.**

### 2.3 The 64D field is *more* greedy-saturated than 32D, not less

This is the most counterintuitive finding. The V3 latent-geometry hypothesis
predicted that a higher-dim latent would expose a *less*-saturated dynamics
field — leaving room for PPO to find policies that beat the depth-2
model-based oracle (`greedy_dyn_2`). The audit measured the opposite:

| Cell | greedy_dyn_2 (V1 32D) | greedy_dyn_2 (64D Track L) |
|---|---:|---:|
| K=2 / bin 8-10 / OOD | 0.300 | **0.853** |
| K=2 / bin 6-8 / OOD | 0.790 | **0.963** |
| K=3 / bin 6-8 / OOD | 1.000 | 1.000 |
| K=3 / bin 8-10 / OOD (primary) | 1.000 | 1.000 |

A *more* saturated field is *worse* for the V3 hypothesis: it leaves less
headroom for PPO to differentiate from the model-based oracle. The 64D Track
L outcome (per [artifacts_v3/interpretation/v3a_checkpoint.md](../artifacts_v3/interpretation/v3a_checkpoint.md)
§1.4):

> The Track L 64D field is MORE saturated than V2's 32D field, not less. The
> V3 latent-dim hypothesis is rejected on Track L.

### 2.4 What's missing from the audit

The audit defines a fair-comparison "Track N" — a **fresh** 64D NB VAE
trained from scratch with the current scVI scaffold — to disambiguate "legacy
VAE config drift" from "64D itself is the problem." That track was started
([artifacts_v3/interpretation/v3b_phase01_interpretation.md](../artifacts_v3/interpretation/v3b_phase01_interpretation.md)
mentions "Track N status: still training (background PID 46735, ~5 h
elapsed)") and **never completed.** When V3B's safety-reward direction
proved decisive, Track N was abandoned.

**Honest gap:** if you want a single confirmation experiment, finishing Track
N is the only one with a clear unanswered question. But the directional
result from Track L is already strong evidence that 64D does not unlock new
behavior downstream.

---

## 3. What the architecture doc predicted

[ARCHITECTURE.md](../ARCHITECTURE.md) D2 (the design-decision log) explicitly
named 32D as the empirical sweet spot:

> 32-dim latent. Alternatives: 16 / 32 / 64 / 128. Why: empirical sweet spot
> per scVI K562 published analyses; small enough that Euclidean distance is
> informative for RL. Ablation matrix in EXPERIMENTS.md.

And Concept 2:

> Latent dim is the most-ablated hyperparameter in this project: EXPERIMENTS.md
> compares {16, 32, 64} on the same data with the same VAE training budget.
> **We expect 32 to dominate**; if 16 wins on RL success, that signals our
> reward signal is brittle and needs reshaping.

The 64D experiment was a planned ablation, the prediction was that 32D would
win, and the audit confirmed the prediction. The repo is operating exactly as
designed — there is no surprise to resolve.

---

## 4. Where the real bottleneck lives

The V3B Phase 2 result
([artifacts_v3/interpretation/v3b_phase2_interpretation.md](../artifacts_v3/interpretation/v3b_phase2_interpretation.md))
shows where the unlock actually was — **the reward, not the latent.** On the
same V1 32D VAE + V2 RoR dynamics, adding a Chronos-based safety penalty
(`λ_tox=0.10, λ_ce=0.05`) produced:

> The first V3-era result where PPO strictly exceeds the depth-2 model-based
> oracle under the same reward at the same cell.

* K=2 / bin 8-10 / OOD: PPO_C 0.340 vs PPO_A 0.300 vs greedy_dyn_2_C 0.300 → **+4.0 pp** over both.
* PPO_C: `mean_common_essential_per_episode = 0.000` at **every** hardness cell.
* Real-Chronos vs permuted-Chronos: PPO_C strictly wins on safety-adjusted
  success at every cell — the signal is biological, not noise.

All on the V1 32D VAE. No VAE change was needed or helpful.

---

## 5. Plausible-but-not-recommended alternative VAE configurations

For completeness, here are the configurations a fresh sweep *could* explore,
along with why I'm not recommending any of them given the evidence above:

| Config change | Hypothesis it tests | Predicted outcome based on audit | Recommended? |
|---|---|---|---|
| n_latent = 16 (under-parameterize) | Smaller latent → more nonlinearity for MLP to capture → larger ridge-margin? Architecture doc says "if 16 wins, reward is brittle." | ε distribution likely shrinks; reward gradient may flatten; downstream success rate may drop because cell-cycle / metabolic axes start to overlap. | **No** — not motivated by any current observation, and Concept 2 already warns about reward brittleness. |
| n_latent = 48 / 96 / 128 | Linear interpolation / extrapolation of the 32→64 trend | 32→64 *shrunk* the MLP-ridge gap. 96 / 128 will almost certainly shrink it further. The ridge gets stronger as latent capacity grows because the perturbation displacements become more linear in a higher-rank latent. | **No** — predicted direction is wrong. |
| `gene_likelihood=zinb` | ZINB on K562 v3 chemistry | Per ARCHITECTURE.md D1 (citing Svensson 2020): "no measurable gain on 10x v3 K562 data." | **No** — already considered and rejected with a citation. |
| `latent_distribution=ln` (logistic-normal) | Less saturating tails → may move the ε / pairing-noise floor | Plausible but speculative; ε and pairing-noise are already V2-comparable on 64D, so the lever is small. | **No** — high cost, speculative. |
| β-VAE / KL annealing | Posterior collapse mitigation | Audit found no posterior collapse: clusters separate, pairing noise 0.89 floor is an OT pairing artifact, not a latent artifact. | **No** — solving a problem that isn't present. |
| Encoder/decoder depth (n_layers 3, n_hidden 256) | More capacity → better reconstruction | scVI ELBO was not the failure metric. Reconstruction was never reported as the bottleneck. | **No** — wrong axis. |
| Lower lr / larger batch / different seed | Training-noise sensitivity | Single-seed training, but ε/centroid stability between artifacts and artifacts_64 (both NB, same architecture) suggests low seed sensitivity in this regime. | **No** — speculative. |
| **Finish Track N (fresh 64D NB, same scVI config as V1)** | Disambiguate "legacy VAE config drift" from "64D itself." The one open question from the audit. | Most likely outcome: matches Track L within noise, confirming 64D ≈ 32D downstream. If it diverges, *then* there is something to investigate. | **Optional**, only if you want to close the open audit loop. ~2h MPS. |

---

## 6. Risks of retraining anyway

If you decide to retrain regardless, the concrete downstream risks are:

1. **Sacred-rule violation.** [CLAUDE.md §3](../CLAUDE.md) sacred rule 1: "Never
   retrain the VAE without first checking for an existing checkpoint."
   `train_vae.py` has the guard, but new artifacts written to a different path
   would bypass it. Any retrain must go under `artifacts_v3/` (or a new tier),
   never overwrite the V1 VAE.
2. **ε / centroid drift breaks frozen pipelines.** Changing `epsilon_success.json`
   or `z_reference_centroid.npy` invalidates every reachability probe,
   greedy-baseline result, and PPO checkpoint in `artifacts_v2/` and
   `artifacts/` that was conditioned on the V1 numbers. A new VAE forces a
   new pairs build, new dynamics train, new RL train, new evaluation — i.e.
   the full V3-era pipeline rerun, for a hypothesis the audit already
   rejected.
3. **Wall-clock cost.** ~2h MPS per VAE × N configs, then ~30min pairs,
   ~30-60min dynamics, ~5-10min PPO **per config**. A six-config sweep is a
   ~1 day commitment, with the most likely outcome being "32D still
   dominates."

---

## 7. Recommendation

**Do not retrain the VAE.** Specifically:

1. **Keep the V1 32D VAE as canonical.** Architecture doc D2 predicted it,
   audit confirmed it, V3B Phase 2 used it to deliver the only V3-era
   PPO > oracle result. Continue using `artifacts/vae/` as the canonical
   checkpoint.
2. **Treat `artifacts_64/` as a closed ablation.** Its existence is
   informative (it documents what 64D does and doesn't change), but it
   should not be promoted to production. The V3A interpretation already
   says so.
3. **If you have *any* compute to spend on the latent question**, spend it
   on **finishing Track N** (fresh 64D NB VAE) — this is the one open
   question. The expected outcome is "Track N matches Track L," at which
   point the audit is fully closed. If Track N diverges, that is itself
   information that warrants new analysis.
4. **The real lever is downstream of the VAE.** V3B Phase 2 demonstrated
   that the reward shape unlocks PPO > oracle behavior on the unchanged
   V1 latent. Future work should iterate on V3B (more seeds, more cells,
   λ tuning, additional biology terms) — not on the encoder.

### If you ignore this recommendation and run a sweep anyway

The minimum-defensible sweep would be:

```bash
# Track N completion (only experiment with an open question)
PYTHONPATH=. python scripts/train_vae.py --config-name default \
    vae.n_latent=64 \
    vae.gene_likelihood=nb \
    paths.vae_dir=${paths.v3_nb_vae_dir} \
    seed=42 \
    +force=true +dry_run=false
```

Plus the audit's downstream chain (pairs → RoR dynamics → eval) per V3A
Phase A1-bg / A2-bg in
[V3A_LATENT_AUDIT_AND_64D_PLAN.md](../V3A_LATENT_AUDIT_AND_64D_PLAN.md). The
acceptance criterion is: does Track N greedy_dyn_2 at K=3/bin 8-10/OOD also
saturate at 1.000? If yes (most likely), 64D is conclusively rejected. If no,
Track N becomes the new V3.1 reference and the picture changes — but plan
for the most likely outcome.

---

## 8. What is *not* in this report (intentionally)

* **A new smoke-run table.** I did not launch new training. The information a
  smoke run would produce is already covered by Track L and the existing 32D
  vs 64D comparison; running 16D / 48D / 96D / 128D smokes would burn ~1-4h
  MPS each to validate a directional prediction (ridge margin shrinks with
  more latent capacity) that the existing data already shows.
* **A β-VAE / KL-annealing / lr sweep.** The audit shows no posterior
  collapse and stable ε across two independent NB scVI runs (V1 32D, legacy
  64D). The training-regime levers are not the bottleneck.
* **Reconstruction-quality plots.** scVI ELBO was monitored and converged in
  the [outputs/2026-05-13/00-06-14/train_vae.log](../outputs/2026-05-13/00-06-14/train_vae.log)
  run. Reconstruction was never named as the failure mode in any of the V1,
  V2, or V3 interpretation documents.

---

## 9. Pointers

* [artifacts_v3/interpretation/v3a_checkpoint.md](../artifacts_v3/interpretation/v3a_checkpoint.md) — full Track L audit
* [artifacts_v3/interpretation/v3b_phase2_interpretation.md](../artifacts_v3/interpretation/v3b_phase2_interpretation.md) — where the actual V3 unlock came from
* [V3A_LATENT_AUDIT_AND_64D_PLAN.md](../V3A_LATENT_AUDIT_AND_64D_PLAN.md) — the planned 64D investigation
* [ARCHITECTURE.md](../ARCHITECTURE.md) — D2 (32D rationale) and §5.1 (VAE failure modes)
* [artifacts_64/dynamics/gate.json](../artifacts_64/dynamics/gate.json) — the failure that triggered the V3A audit
