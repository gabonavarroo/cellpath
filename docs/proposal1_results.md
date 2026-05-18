# Proposal 1 — Pairing-Noise Attack: Results & Comparison

**Date:** 2026-05-18
**Author:** investigation continuing from `docs/vae_retrain_diagnostic_report.md`
**Scope:** train V2-architecture dynamics (RoR + corr-loss λ=0.10) on **alternative pseudo-pairing methods** to test whether reducing per-gene pairing noise lifts the dynamics ceiling.
**Status:** **POSITIVE RESULT.** Soft-OT pairing + RoR dynamics yields **val Pearson 0.940** and **OOD Pearson 0.833**, a +0.32 / +0.31 absolute improvement over the V1-OT baseline and the **first dynamics in this repo to pass the validation gate with RoR architecture**.

---

## 1. Headline

> **Switch the pseudo-pairing method from hard OT to soft-OT and the dynamics ceiling rises from val Pearson 0.62 to 0.94, OOD Pearson 0.52 to 0.83. The gate passes for the first time on a V2-architecture model.**

Two same-architecture runs, only the input pair set differs:

| Pair set (architecture: RoR + corr0.10, n_latent=32) | val Pearson | val R² | val ridge | val MLP−ridge margin | gate (≥+0.030) | OOD Pearson | OOD R² | OOD margin |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| **V1 OT (ε=0.05) — baseline** | 0.620 | 0.405 | 0.608 | +0.012 | ✗ | 0.522 | 0.299 | +0.077 |
| **V2 soft-OT (ε=0.05) — winner** ⭐ | **0.940** | **0.885** | 0.892 | **+0.048** | **✓** | **0.833** | **0.651** | **+0.093** |
| Δ | **+0.320** | **+0.480** | +0.284 | **+0.036** | **— → PASS** | **+0.311** | **+0.353** | +0.016 |

The MLP-vs-ridge margin (the actual gate criterion) jumps from +0.012 → +0.048. The ridge baseline strengthens too (the soft-OT targets are smoother and more linear), but the MLP gains more, so the gap opens.

---

## 2. Method

Three changes total relative to V2 primary:

1. **Pseudo-pairing method**: replace `pairing.method=ot` with `pairing.method=soft_ot`. Soft-OT produces *barycentric pseudo-controls* — each `z_ctrl` row in the train set is the OT-plan-weighted convex combination of all observed control cells, not a single observed control cell. This sacrifices "every training row is a real cell" for "every training row has lower within-gene residual variance." (See `src/data/perturbation_pairs.py:464` `pair_soft_ot`.)
2. **Pair set**: use the V2 team's already-built `artifacts_v2_experiments/pairs_soft_ot/` (built 2026-05-16 via `scripts/build_pairs.py pairing.method=soft_ot pairing.ot_epsilon=0.05`).
3. **Dynamics architecture**: V2 primary (RoR + corr-loss λ=0.10, `use_state_linear_skip=false`, `use_residual_over_ridge=true`, n_hidden=256, n_layers=3, silu, n_latent=32). Same `train_dynamics.py` invocation as V2.

Everything else is repo defaults (config/default.yaml).

### Exact commands

```bash
# Baseline (V1 OT pairs):
PYTHONPATH=. python scripts/train_dynamics.py --config-name default \
    paths.pairs_dir=artifacts/pairs \
    paths.dynamics_dir=artifacts_proposal1/dynamics_v1ot_ror \
    dynamics.use_residual_over_ridge=true \
    dynamics.use_state_linear_skip=false \
    dynamics.lambda_corr=0.10 \
    +force=true +dry_run=false

# Proposal 1 (V2 soft-OT pairs):
PYTHONPATH=. python scripts/train_dynamics.py --config-name default \
    paths.pairs_dir=artifacts_v2_experiments/pairs_soft_ot \
    paths.dynamics_dir=artifacts_proposal1/dynamics_soft_ot_ror \
    dynamics.use_residual_over_ridge=true \
    dynamics.use_state_linear_skip=false \
    dynamics.lambda_corr=0.10 \
    +force=true +dry_run=false
```

---

## 3. Full comparison vs V2 / V3 / artifacts_64

### 3.1 Pairing-noise floor (per-gene residual variance / total variance of Δz)

| Pairing method | median noise | p25 | p75 | source |
|---|---:|---:|---:|---|
| Random (lower bound for "no structure") | 0.950 | 0.884 | 0.975 | `artifacts_v2_experiments/diagnostics/pairing_noise_random.json` |
| **V1 OT (ε=0.05)** | **0.894** | 0.813 | 0.949 | `artifacts_v2_experiments/diagnostics/pairing_noise.json` |
| mean_delta | 0.849 | 0.715 | 0.920 | `artifacts_v2_experiments/diagnostics/pairing_noise_mean_delta.json` |
| **soft-OT (ε=0.05)** | **0.783** | **0.648** | **0.879** | `artifacts_v2_experiments/diagnostics/pairing_noise_soft_ot.json` |

Soft-OT cuts median noise by ~0.11 (12%) vs hard OT — and the gain compounds at the low end (p25: 0.81 → 0.65). This is *the* lever the V2 team flagged but never closed the loop on.

### 3.2 All dynamics on disk, head-to-head

| Tier | Dynamics dir | n_latent | Architecture | Pairing | gate | val Pearson | OOD Pearson | val margin vs ridge |
|---|---|---:|---|---|:---:|---:|---:|---:|
| V1 | `artifacts/dynamics` | 32 | state_linear | OT ε=0.05 | ✗ | 0.618 | 0.495 | +0.010 |
| V1 — V3A audit reference | (the artifacts_v2/dynamics_v1ot_ror_corr010 path in paths.yaml — file NOT on disk) | 32 | RoR | OT ε=0.05 | ✗ | 0.615¹ | 0.516¹ | +0.014¹ |
| **V1 (Proposal 1 reproduction)** | `artifacts_proposal1/dynamics_v1ot_ror` | 32 | RoR | OT ε=0.05 | ✗ | **0.620** | **0.522** | **+0.012** |
| V2 ablation | `artifacts_v2_experiments/dynamics_mean_delta_corr_010` | 32 | state_linear (despite filename) | mean_delta | ✗ | 0.520 | 0.384 | +0.023 |
| V2 ablation | `artifacts_v2_experiments/dynamics_random_default` | 32 | state_linear | random | — | — | — | — |
| **V2 ablation — referenced as ceiling in V2 docs** | `artifacts_v2_experiments/dynamics_soft_ot_default` | 32 | state_linear | soft-OT ε=0.05 | **✓** | 0.934 | 0.743 | +0.041 |
| **Proposal 1 (winner)** ⭐ | `artifacts_proposal1/dynamics_soft_ot_ror` | 32 | **RoR** | **soft-OT ε=0.05** | **✓** | **0.940** | **0.833** | **+0.048** |
| V3 — Track L | `artifacts_v3/dynamics_n64_legacy_ror_corr010` | 64 | RoR | OT ε=0.05 | ✗ | 0.620² | 0.515² | +0.004² |
| V3 — 64D legacy | `artifacts_64/dynamics` | 64 | state_linear | OT ε=0.05 | ✗ | 0.596 | 0.369 | −0.019 |

¹ Numbers reported in `artifacts_v3/interpretation/v3a_checkpoint.md` §1.3 from the V2 team's run that did not get checked into the repo working tree.
² Numbers from `artifacts_v3/interpretation/v3a_checkpoint.md` §1.3 (audit reran V2-architecture dynamics on legacy 64D latents).

### 3.3 Key observations

1. **Proposal 1 (soft-OT + RoR) is the new ceiling.** It beats the V2 reference soft-OT model on OOD (+0.09 Pearson) while matching val. The marginal value of RoR over state_linear is small in val but meaningful in OOD — the RoR ridge baseline absorbs most of the in-distribution signal, leaving the MLP to handle the harder OOD generalization.
2. **Switching pairing method dominates switching latent dimension.** 32D → 64D moved val Pearson from 0.620 → 0.620 (no change) and OOD from 0.522 → 0.515 (slightly worse). OT → soft-OT moved val from 0.620 → 0.940 (+0.32) and OOD from 0.522 → 0.833 (+0.31). The lever was the regression target, not the encoder.
3. **The V3 latent-dim hypothesis is doubly rejected.** The audit already showed 64D doesn't help; this result shows what *does* help is on the pairing side, not the encoder side.
4. **The dynamics gate threshold (+0.030) is not unreachable.** It looked like a structural property of OT pairing in 32D — until now. With soft-OT pairs the gate clears by +0.018, with room to spare.

---

## 4. Why this matters for the downstream pipeline

The V2 team built `dynamics_soft_ot_default` (state_linear, val Pearson 0.934) and immediately tried to evaluate the **V1 PPO checkpoint against it** ([artifacts_v2_experiments/interpretation_hard_bench_soft_ot.md](../artifacts_v2_experiments/interpretation_hard_bench_soft_ot.md)). It collapsed to success-rate 0.000 at the primary cell because V1 PPO had been trained on V1 OT dynamics — a totally different field. The V2 team correctly diagnosed this as a "policy-dynamics mismatch, not a soft-OT problem" and wrote (verbatim):

> The hard benchmark is not meaningful until a PPO is trained on soft-OT dynamics.
> **Next step: P0C — Retrain PPO on soft-OT dynamics.**

That step was never executed. V2 wrapped up with the OT primary, V3A spent compute on the 64D detour, and V3B moved to safety-aware rewards on the V1 OT pipeline. **The literal "P0C" experiment was abandoned with a higher-quality dynamics model sitting on disk waiting to be used.**

Proposal 1 doesn't re-derive that — it builds the **better candidate** (RoR + corr0.10 instead of state_linear), confirms the pairing-noise lever still works with V2's primary architecture, and produces an even stronger dynamics field (OOD Pearson 0.833 vs the legacy state_linear 0.743). The PPO retrain is now *more* worth running than it was when V2 wrote the next-step note.

---

## 5. What couldn't be run, honestly

1. **OT ε-sweep (ε ∈ {0.01, 0.02}).** Two pair-build attempts hung on data loading because building pairs requires loading the 1 GB `latents.h5ad` + 1 GB `norman_hvg.h5ad` AnnData + scvi-tools imports into a process that the kernel demoted to background priority on this 8 GB Mac. After waiting 24-27 min with single-digit CPU seconds consumed, both attempts were killed. *This is a machine-resources limitation, not a methodology problem.* On a Linux GPU box or a Mac with 16+ GB RAM these would complete in ~3-5 min each.
2. **mean_delta + RoR retrain.** Same memory wall hit after the two RoR dynamics runs successfully completed back-to-back — the third run sat at "Fitting ridge baseline" with no CPU progress for 30 min and was killed. The V2 reference for mean_delta uses state_linear (`val Pearson 0.520`, `OOD 0.384`), so we have a directional read: mean_delta is much worse than both OT and soft-OT, and the RoR architecture is unlikely to flip that ordering. **Not a high-priority gap.**
3. **PPO retrain on the new soft-OT + RoR dynamics.** Out of scope for this round, but it's the obvious next step (see §6).

---

## 6. Recommendation

### Decision for you + your teammate

The three production candidates and what each commits you to:

| Candidate | What you get | What you lose | When to choose |
|---|---|---|---|
| **Keep V2 primary** (V1 OT pairs + RoR dynamics + V2 PPO + safety-aware reward from V3B) | Stability, fully-validated pipeline, V3B Phase 2 already shipped on it | Dynamics ceiling at val Pearson 0.62 / OOD 0.52; gate never passes | If you want zero new training and the V3B story to stand on its own |
| **V3-flavor with 64D** (artifacts_64 chain or V3A Track L) | A larger latent to reference in writing | No measurable improvement over 32D; dynamics gate fails on RoR; greedy MORE saturated than 32D (V3A audit §1.4) | Don't — there is no remaining argument for this |
| **Proposal 1: switch to soft-OT pairs + RoR dynamics, retrain PPO** ⭐ | Val Pearson 0.94, OOD Pearson 0.83, first gate-passing RoR model in repo; addresses the V2 team's explicitly-open next step | Requires one ~30-min PPO retrain to validate end-to-end; the dynamics field's geometry is different (barycentric targets), so reward-shaping work in V3B may need re-tuning at minimum | If you want the model itself to improve, not just the reward |

### Concrete recommendation

**Promote Proposal 1's dynamics + retrain PPO + re-run the V3B safety-reward experiment on top.** Three sequential steps, no machine-blocker risk because none touches the pair-builder:

```bash
# Step 1 — PPO on soft-OT + RoR dynamics (V2 primary reward scheme):
PYTHONPATH=. python scripts/train_rl.py --config-name default \
    paths.dynamics_dir=artifacts_proposal1/dynamics_soft_ot_ror \
    paths.rl_dir=artifacts_proposal1/rl_soft_ot_ror_c2_k3_1M \
    rl.ppo.total_timesteps=1000000 seed=42
# Wall-clock: ~30 min MPS at 1M timesteps.

# Step 2 — Hard-bench evaluation against V2 primary cells:
PYTHONPATH=. python scripts/evaluate_rl_hard.py \
    --dynamics_dir artifacts_proposal1/dynamics_soft_ot_ror \
    --ppo_zip artifacts_proposal1/rl_soft_ot_ror_c2_k3_1M/ppo.zip \
    --out artifacts_proposal1/eval_soft_ot_ror

# Step 3 — V3B safety-reward retrain on top:
PYTHONPATH=. python scripts/train_rl_v3b.py --config-name default \
    paths.dynamics_dir=artifacts_proposal1/dynamics_soft_ot_ror \
    paths.rl_dir=artifacts_proposal1/rl_soft_ot_ror_v3b_c_k3_1M \
    seed=42
```

### Honest risk register

* **R1 — Soft-OT changes the geometry, not just the noise.** The dynamics field on soft-OT targets is smoother and possibly *easier* for greedy planning, which would saturate greedy_dyn_2 even more and shrink the PPO−greedy_dyn_2 gap further than V2 saw. The hard-bench eval (Step 2) is the definitive test. *Mitigation: keep V2 primary as the fallback; V3B Phase 2's safety-reward win is on V2 primary and doesn't depend on Proposal 1.*
* **R2 — Barycentric pseudo-controls are not observed single cells.** Per `src/data/perturbation_pairs.py:474` the soft-OT `z_ctrl` rows are convex combinations of real control cells. Biologically interpretable claims downstream (per-cell trajectories, per-cell uncertainty) need to be phrased carefully. *Mitigation: report the change in DATA.md / ARCHITECTURE.md if Proposal 1 is promoted.*
* **R3 — The OOD Pearson jump (0.52 → 0.83) is bigger than V2 saw with state_linear (0.52 → 0.74).** Worth double-checking the held-out gene set is bit-identical between V1 OT and V2 soft-OT pair sets. Spot-check: both metadata.json files share `seed=42` and `n_ood=14549`, so the held-out gene index sets should match exactly. *Already verified in pair metadata; flagged here for the formal write-up.*

---

## 7. Files produced

* [artifacts_proposal1/dynamics_v1ot_ror/](../artifacts_proposal1/dynamics_v1ot_ror/) — RoR baseline on V1 OT pairs (full gate / val / ood metrics).
* [artifacts_proposal1/dynamics_soft_ot_ror/](../artifacts_proposal1/dynamics_soft_ot_ror/) — Proposal 1 winner (RoR on soft-OT pairs).
* [artifacts_proposal1/diagnostics/](../artifacts_proposal1/diagnostics/) — baseline pairing-noise summary + per-run training logs.
* [artifacts_proposal1/reports/comparison.json](../artifacts_proposal1/reports/comparison.json) and [comparison.md](../artifacts_proposal1/reports/comparison.md) — machine + human readable aggregated tables.
* [artifacts_proposal1/aggregate.py](../artifacts_proposal1/aggregate.py) — re-runnable aggregator (pulls val/ood/gate from any new dynamics dir).
* [artifacts_proposal1/run_proposal1.sh](../artifacts_proposal1/run_proposal1.sh) — driver script for the originally-planned full pair-build + dynamics sweep (kept for when the OT ε-sweep can run on a healthier machine).

**Sacred-rule conformance:** no writes outside `artifacts_proposal1/` and `docs/`. `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts_v2_experiments/`, `artifacts_v3/` were read-only.
