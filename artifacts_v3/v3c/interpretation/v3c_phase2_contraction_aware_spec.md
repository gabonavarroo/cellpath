# V3C Phase 2 — Contraction-aware dynamics, candidate v1 spec

**Status:** `PHASE2_SPEC` — implementation guide for `contraction_aware_v1`. The first conservative Phase 2 candidate; not the only one we'll try.

**Scope:** A single config-gated additive regularizer on the dynamics training loss, designed to attack the **two structural pathologies** the Phase 0C audit surfaced on every OT-trained field:

1. `CONTRACTION_NEAR_UNIVERSAL` — `contraction_fraction ≈ 1.000` on the OOD start pool (every (z, g) pair contracts toward z_ref).
2. `UNIVERSAL_ATTRACTOR_GENE` — `gene_universality_max ≈ 0.92` (one or few genes dominate the contraction signal across all states).

The reformulation is **not a greedy-sabotage field**. The Phase 0C audit demonstrated that the universal-attractor structure makes greedy_dyn_K saturated at K≥3 (i.e. the planning problem is trivial for the existing dynamics), but PPO_BCD still cannot reliably beat that saturated greedy because at the **non-saturated cells** (K=2/bin8-10) the reachable mass is too small. The goal here is to **shrink the universally-aligned mass without destroying the predictive fit**, and let utility unlock at K∈{2, 3, 5} cells via the reduced saturation.

The Phase 2 verdict labels (`PHASE2_STRONG_K5_UTILITY` … `PHASE2_FAILED_FIELD`) explicitly permit a non-K=5-headline outcome; we're aiming for **measurable geometry improvement that preserves prediction sanity**, not a single-shot K=5 win.

---

## §1 — Loss formulation

The training loss for `contraction_aware_v1` becomes:

```
L_total = L_NLL(μ, log σ²; Δz)                          # primary: existing
        + λ_combo · L_combo(model; combo pairs)         # existing
        + λ_corr  · L_corr(μ, Δz)                       # existing (V2 primary uses 0.10)
        + λ_lv    · ‖log σ²‖²                           # existing log_var L2
        + λ_ea    · L_ea(μ, z, z_ref; τ_ea)             # NEW — excessive alignment
        + λ_ua    · L_ua(μ, z, z_ref, gene_idx; τ_ua)   # NEW — universal-attractor
        + λ_ad    · L_ad(μ; τ_ad)                       # NEW — action diversity (OPTIONAL)
```

All three new terms default to `λ_* = 0.0` so existing V2/V3 dynamics behavior is byte-identical when the keys are absent or zero.

### 1.1 Excessive-alignment penalty `L_ea`

For each (z, g) in the training batch:

```
target_dir = z_ref − z                                       # (B, D)
α(z, g)    = cos(μ(z, g), target_dir)                        # (B,)
L_ea       = mean( relu(α − τ_ea)² )                         # scalar
```

- `τ_ea ∈ {0.70, 0.80}` is the alignment cap. Setting `τ_ea = 0.80` is the conservative default: the audit's `alignment_cos_median ≈ 0.50–0.60` on V2 anchor — most pairs are below the cap, the penalty fires only on the long-tail of over-aligned predictions that compose the universal-attractor structure.
- `relu(·)²` is one-sided (no reward for low alignment) and bounded near 0 for normal pairs.
- The cap is on the PREDICTED `μ`, not on the TARGET `Δz`. We are NOT teaching the model to fit a different target; we are teaching it to **not over-extrapolate the contractive structure onto every (z, g) pair**.

### 1.2 Universal-attractor penalty `L_ua`

Compute the per-gene mean alignment across the batch, then penalize the **batch-max** of that mean:

```
For each unique gene g_i in the batch:
    states_with_g = { (z, g) in batch : g = g_i }
    ᾱ(g_i)        = mean over states_with_g of α(z, g)
L_ua = relu( max_i ᾱ(g_i) − τ_ua )²
```

- `τ_ua ∈ {0.70, 0.80}` mirrors `τ_ea`. The audit's `gene_universality_max ≈ 0.92` on every OT field is the exact statistic this term targets.
- `max` over genes (rather than mean or sum) makes the penalty **focal** on the single dominant attractor gene — the audit's `UNIVERSAL_ATTRACTOR_GENE` flag was triggered by a *single* gene's mean alignment exceeding 0.85.
- Because batches are small (`batch_size=256`), most genes appear with `< 5` samples per batch — `ᾱ(g_i)` is noisy. A smoothed-EMA tracker would be more stable, but for v1 we accept the noise; the gradient signal is what matters, not point-estimate fidelity.

### 1.3 Action-diversity encouragement `L_ad` (optional)

```
σ²_per_state = var across batch dims of μ(z, g)              # (B, D) per dim
L_ad         = mean( relu(τ_ad − σ²_per_state.mean(dim=-1))² )
```

This is a **floor**, not a ceiling: penalize whenever the across-batch variance of `μ` falls below `τ_ad`. Default `τ_ad = 0.0` makes the term inactive even when `λ_ad > 0`.

**For v1 we leave `λ_ad = 0` and `τ_ad = 0`** — turning on action diversity is a Phase 2.5 candidate after we see whether `L_ea + L_ua` alone moves the geometry.

---

## §2 — Implementation surface

### 2.1 Code locations

| Module | Add | Why |
|---|---|---|
| `src/models/dynamics.py` | `excessive_alignment_penalty(mu, z, z_ref, tau)`, `universal_attractor_penalty(mu, z, z_ref, gene_idx, tau)`, `action_diversity_penalty(mu, tau)` | Per-pattern with existing `heteroscedastic_nll` / `composition_loss`. |
| `scripts/train_dynamics.py` | Wire the three new losses behind config keys; load `z_ref` once at training start. | The trainer already integrates `λ_combo` / `λ_corr` similarly. |
| `config/dynamics.yaml` | Add the keys below, all defaulting `λ_*=0.0` / `τ_*=0.80` / `0.0`. | Keeps default V2/V3 dynamics behavior unchanged. |
| `tests/test_dynamics.py` (or a new `test_dynamics_contraction_aware.py`) | TDD-style: a synthetic `NoOpDynamics`, `UniversalAttractorDynamics`, `DivergentDynamics` set of fixtures with known geometry, verifying each penalty sign / magnitude. | CLAUDE.md sacred-rule §3.10 (gate via tests). |

### 2.2 New config keys (additive, default-disabled)

```yaml
dynamics:
  # ... existing keys ...
  # V3C Phase 2 — contraction-aware regularizers (default disabled).
  contraction_aware:
    enabled: false                # master switch (also off when all λ_*=0)
    lambda_excessive_alignment: 0.0   # λ_ea — weight on L_ea
    tau_excessive_alignment: 0.80     # τ_ea — alignment cap
    lambda_universal_attractor: 0.0   # λ_ua — weight on L_ua
    tau_universal_attractor: 0.80     # τ_ua — per-gene max alignment cap
    lambda_action_diversity: 0.0      # λ_ad — weight on L_ad (default off)
    tau_action_diversity: 0.0         # τ_ad — variance floor (default inert)
```

### 2.3 `z_ref` handling in the trainer

`z_ref` is loaded from `cfg.paths.vae_z_reference_centroid` (Track L → `artifacts_v3/vae_n64_legacy/z_reference_centroid.npy`). The loader runs once at training start, converts to `torch.float32`, moves to `device`, and is shared across batches. It is NOT a model buffer — it is a per-VAE artifact that the trainer reads, identical to how the gate-baselines train data is loaded.

---

## §3 — Default values for the v1 candidate

| Knob | Value | Rationale |
|---|---|---|
| `λ_ea` | **0.05** | Conservative — same order of magnitude as `λ_corr=0.10`. Small enough that NLL loss dominates. |
| `τ_ea` | **0.80** | Above the empirical median alignment (~0.5–0.6) on training pairs, so most pairs pay 0 penalty. |
| `λ_ua` | **0.05** | Same conservativeness rationale. |
| `τ_ua` | **0.80** | Allows mild per-gene drift; only fires when one gene's mean alignment exceeds the cap. |
| `λ_ad` | **0.0** | Disabled in v1 (see §1.3). |
| `τ_ad` | **0.0** | Inert. |
| All other keys | (V2 primary defaults from `config/dynamics.yaml`) | Same backbone, same LR, same combo / corr / log_var_reg / RoR config. |

**The v1 candidate trains on Track L's `artifacts_v3/pairs_n64_legacy`** (64D legacy scVI VAE). Rationale: Track L has the highest K=2/bin8-10/OOD reachability under the V2 architecture (Phase 0C §5: 0.560 vs anchor 0.120) and its Phase 1 PPO_BCD ran 0.705 (anchor lift 5.9×) — most likely to retain that lift while gaining geometric breathing room. Track N (NB likelihood) is a second-pass candidate after v1.

---

## §4 — Decision criteria (Phase 2 verdicts)

Apply the V3C utility audit (`scripts/audit_dynamics_utility_v3c.py all`) to `contraction_aware_v1` and compare with the five reference fields (V2 anchor, Track L, Track N, mean_delta_corr_010, Soft-OT).

| Verdict | Conditions |
|---|---|
| `PHASE2_STRONG_K5_UTILITY` | (a) K=5/bin8-10/OOD beam reach is nontrivial (e.g. > 0.2) AND not saturated (< 0.95) AND shows depth leverage (`reach@K=5 > reach@K=3 + 0.05`), AND (b) U-A primary sanity preserved (val Pearson ≥ V2 anchor − 0.05). |
| `PHASE2_MODERATE_UTILITY` | (a) U-A primary sanity preserved (val Pearson ≥ V2 anchor − 0.05), (b) U-D `contraction_fraction` reduced (e.g. < 0.97) OR `gene_universality_max` reduced (e.g. < 0.85), AND (c) at least one control axis improves (K=2 reach lift, action_diversity_per_state up, or first_action_entropy_fused up) versus V2 anchor or Track N/L. |
| `PHASE2_DIAGNOSTIC_ONLY` | Geometry changes (U-D shifted) but no control utility axis improves. |
| `PHASE2_FAILED_FIELD` | (a) Val Pearson collapses (< V2 anchor − 0.10) OR (b) U-B reachability drops to zero at every cell OR (c) `contraction_fraction` < 0.2 with `alignment_cos_median < 0` (Soft-OT-like anti-contraction). |

**K=5 is aspirational, not a hard gate.** Per the user's explicit guidance in the session brief, we aim for `MODERATE` and use the U-bucket evidence to decide whether a PPO smoke is justified.

---

## §5 — PPO smoke (post-audit, gated)

Run the smoke ONLY if v1 reaches `PHASE2_MODERATE_UTILITY` or `PHASE2_STRONG_K5_UTILITY`. Configuration:

- Single seed (42), 500k first.
- Continue to 1M only if 500k is non-collapsing and the K=2/bin8-10/OOD signal is non-negative versus same-field reward-aware `greedy_dyn_2_fused`.
- Locked B+C+D reward stack (identical to Phase 1), per-VAE p15 ε computed from the new candidate's VAE distance distribution (will be Track L's p15 = 3.0193 since we share the VAE).
- Output: `artifacts_v3/v3c/rl_smokes/contraction_aware_v1_seed42_{500k,1M}/`.

**No bounded reward tuning yet.** Reward-tuning only after a robust multi-seed positive or a strong Phase 2 candidate, per the user's session brief.

---

## §6 — Sacred-rule conformance

- ✅ Additive code only; no edits to frozen-tier code paths.
- ✅ New artifacts under `artifacts_v3/v3c/`.
- ✅ Default `λ_*=0` makes the change byte-identical for V2/V3 retraining.
- ✅ Tests added with TDD discipline (one fixture per pathology class, ≥ 3 tests per loss term).
- ✅ V3B reward-stack lock unchanged; no reward-coefficient changes proposed here.
- ✅ K=5 framed as aspirational; Pareto/MODERATE verdicts explicitly defined.
