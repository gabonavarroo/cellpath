# P0B′ — Pairing Correction (V2 reorder)

> **Implementation plan for: `P0B_PRIME_PAIRING_CORRECTION_PLAN.md`.**
> When executed, Task 1 commits a verbatim copy of this document into the
> repo at `/Users/gabo/Developer/ITAM/IA/cellpath/P0B_PRIME_PAIRING_CORRECTION_PLAN.md`
> so the plan is itself a tracked artifact.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`
> to execute task-by-task with review checkpoints. Use `superpowers:test-driven-development`
> for any new code (the new diagnostic comparison script).

**Goal:** Decide whether the V1 dynamics gate failure (`val mlp_minus_ridge_pearson = +0.0074` vs. `+0.030` required) is bounded above by OT pseudo-pair noise *before* spending P0B on a dynamics loss/architecture fix. Do this by rebuilding pairs with `pairing.method=mean_delta` (and a `random` negative control), retraining the V1 default dynamics architecture verbatim on each, and re-running the gate. Outcome: either pivot V2 forward on cleaner pairs, or fall back to the original P0B with strengthened priors.

**Architecture:** No new model code. Reuse the existing pairing methods in `src/data/perturbation_pairs.py` (`mean_delta`, `random`, `ot`), the unchanged `PerturbationDynamicsModel`, and the unchanged validation gate. All variation is via Hydra overrides (`pairing.method`, `paths.pairs_dir`, `paths.dynamics_dir`). All outputs go under `artifacts_v2/` — V1 artifacts at `artifacts/`, `artifacts_64/`, and `artifacts/rl_sweeps/` remain frozen.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, PyTorch (MPS/CUDA via `src/utils/device.py`), scvi-tools (frozen VAE), numpy/scipy/pandas, pytest. No new dependencies.

---

## Recommendation

**Proceed to P0B′ implementation.** Codex's P0A interpretation is verified and its Recommendation B is justified.

---

## Why P0A changed the original V2 order

The original `V2_RESEARCH_PLAN.md` ordered: P0A (forensics, read-only) → P0B (dynamics loss/arch: correlation loss `λ_corr ∈ {0.05, 0.10, 0.30}` + residual-over-ridge) → P0C (RL retrain). Pairing correction (`mean_delta`, soft-OT) was deferred to **P1.1**, on the grounds that the architecture fix had a *higher prior* of yielding gate margin under tight time budget. The pre-P0A H1 (architecture is the bottleneck) had higher prior than H2 (pairing is the bottleneck).

P0A inverted those priors:

1. **Pairing-noise ratio is decisive.** Median residual/total Δz variance ratio = **0.8935** (mean 0.866; p25/p75 = 0.813/0.949; max 0.981 — gene_idx 99). With ~89% of the Δz variance per gene being within-gene residual (i.e., pseudo-pair assignment noise rather than gene-conditioned mean signal), any deterministic dynamics model has at most ~11% of variance available to beat ridge on. The observed `+0.0074` MLP-minus-ridge margin sits comfortably inside that noise ceiling — making the architecture fix a confounded test rather than a clean one.
2. **Gate failure is concentrated, not architectural.** Dim 11 contributes `val −0.124` and `OOD −0.433` (worst dims), and the worst val genes are ridge-favored only marginally (KIF18B `−0.026`, LYL1 `−0.025`, HK2 `−0.021`). The MLP is essentially riding the per-gene mean direction. This is symptomatic of a target whose remaining variance after ridge is mostly noise.
3. **The hard benchmark is real but PPO ≈ greedy.** At the primary cell (`K=3, ε=p25, bin 8-10, OOD, n=500`) PPO det = `1.000`, random = `0.178`, greedy_dyn_1 = `1.000`. PPO matches the one-step dynamics oracle exactly; across all 384 hard cells PPO − greedy_dyn_1 is ≤ `+16.0pp` (one cell) and is typically `≤ 0pp`. PPO learned the contraction structure of the *learned* dynamics field, not a planning capability beyond it. Hence a dynamics-quality fix is the highest-leverage next move; an RL-side intervention (P0C) would be premature.
4. **Contraction is diffuse, not architectural.** Gini = 0.175, entropy fraction = 0.986. PPO top-10 overlap with the top-contractive genes is 6/10. H5 is mixed: not a uniform field, not a sharp one. Not pivotal for the next step either way.
5. **Biology rerank is null.** PPO det Spearman vs. K562 Chronos: ρ = `−0.024`, p = `0.815`, CI `[−0.218, +0.171]`. No GSEA panel at q ≤ 0.10. Treat as a caveat; do not change reward yet.

**Net:** the cheapest, most discriminative next experiment is to vary the *pair-construction step* with the architecture held fixed. If `mean_delta` pairs (which zero the within-gene Δz residual to a minimum-distance pairing) clear the gate, the V1 OT target was the binding constraint. If they don't, we revert to the original P0B with the pairing-noise ceiling now quantified rather than assumed.

---

## Hypotheses (preregistered before runs)

* **H_pair_primary (was H2 in V2 plan, now promoted):** The OT pseudo-pair target's within-gene residual variance is the binding constraint on `val mlp_minus_ridge_pearson`. Switching to `pairing.method=mean_delta` will reduce the pairing-noise ratio by at least `0.10` (median) and will increase the dynamics val gate margin to `≥ +0.030`, all else equal.
* **H_pair_neg_control:** `pairing.method=random` should not pass the gate; if it does (margin near `mean_delta`), the pair assignments carry near-zero information and V2 assumptions need revision.
* **H_arch_fallback (was H1, demoted):** If H_pair_primary fails to clear `+0.030` (i.e., `mean_delta` margin stays in `[+0.005, +0.015]`), then the bottleneck is architecture/loss and the original P0B (correlation loss + residual-over-ridge) is the next step on the *best-available* pair target.

---

## Files to inspect (read-only)

* `V2_RESEARCH_PLAN.md` — original phase order and gate thresholds (line ~114 for the gate, lines 152–159 / 209–211 for pairing material, lines 388–434 for P0B, line 428 for the existing rollback clause to P1 pairing).
* `CLAUDE.md` §3 (Sacred rules), §6/§7 (env + device).
* `ARCHITECTURE.md` — Concepts 3 (dynamics) and 7 (pairing).
* `AGENTS.md` — Contract 2 (Pairs → Dynamics handshake schema).
* `artifacts_v2/p0a_summary.md`, `artifacts_v2/p0a_decision.md`.
* `artifacts_v2/diagnostics/pairing_noise.{json,md}`, `per_dim_margin.csv`, `per_gene_margin.csv`, `per_gene_contraction_summary.json`.
* `artifacts_v2/eval_hard_v1policy/{results_table.md, metadata.json}`.
* `src/data/perturbation_pairs.py` — `build_pairs`, `pair_mean_delta`, `pair_random`, `pair_ot` (all already implemented).
* `scripts/build_pairs.py`, `scripts/train_dynamics.py`, `scripts/diagnose_pairing_noise.py`, `src/analysis/gate_breakdown.py`.
* `src/models/dynamics.py`, `src/analysis/metrics.py` (gate + ridge), `config/{default,paths,dynamics,vae}.yaml`, `config/experiments/dynamics_legacy_mlp.yaml`.

## Files to modify (this phase)

* **New:** `scripts/compare_pairings.py` — thin aggregator that, given a list of pairing-run directories, emits `artifacts_v2/diagnostics/pairing_comparison.{json,md}` with side-by-side rows: (pairing_method, n_pairs, pairing_noise_median, val_margin, ood_margin, dim11_val, dim11_ood, uncertainty_spearman, gate_passed). Reads, does not retrain.
* **New:** `tests/test_p0b_prime_pairing.py` — unit tests for the new comparator and Contract-2 schema checks on `artifacts_v2/pairs_*/`.
* **New file at repo root:** `P0B_PRIME_PAIRING_CORRECTION_PLAN.md` — verbatim copy of this document (committed as the durable planning artifact).
* **Updated:** `PROGRESS.md` — one new session entry at the top per CLAUDE.md §8 conventions (do not edit prior entries).

## Files that must NOT be modified

* Anything under `artifacts/`, `artifacts_64/`, or `artifacts/rl_sweeps/` (V1 frozen artifacts).
* `src/data/perturbation_pairs.py` (pairing implementations are correct; we use them as-is — soft-OT is *not* in scope for P0B′).
* `src/models/dynamics.py` — architecture stays bit-identical to V1 default (this is the whole point).
* `src/analysis/metrics.py` (gate logic is the single source of truth; any change here invalidates V1 comparisons).
* `config/dynamics.yaml` — *no threshold lowering*. Do not change `gate.margin_vs_linear_ridge_pearson: 0.03`.
* `config/default.yaml` — do not change `pairing.method` default. All variation is via per-run Hydra overrides.
* `artifacts/vae/*` — VAE stays frozen (sacred rule #1).

---

## Experiment matrix

All runs reuse V1 VAE (`artifacts/vae`) and the V1-default dynamics architecture (`use_state_linear_skip=true`, `use_gene_delta_bias=false`, `selection_metric=gate_margin`, `lambda_mse_delta=0.0`, `lr=1e-4`, `max_epochs=300`, `patience=35`, `seed=42`).

| # | Stage | Pairs dir | Dynamics dir | Pairing | Purpose |
|---|---|---|---|---|---|
| 1 | Rebuild pairs (mean-delta) | `artifacts_v2/pairs_mean_delta/` | — | `mean_delta` | Test H_pair_primary |
| 2 | Train dynamics (mean-delta) | (reads #1) | `artifacts_v2/dynamics_mean_delta_default/` | — | Same architecture, new target |
| 3 | Diagnostics (mean-delta) | (reads #1 + #2) | — | — | `pairing_noise` + `gate_breakdown` |
| 4 | Rebuild pairs (random) | `artifacts_v2/pairs_random/` | — | `random` | H_pair_neg_control |
| 5 | Train dynamics (random) | (reads #4) | `artifacts_v2/dynamics_random_default/` | — | Negative control |
| 6 | Diagnostics (random) | (reads #4 + #5) | — | — | `pairing_noise` + `gate_breakdown` |
| 7 | Comparator | (reads #1–#6 + V1 `artifacts/{pairs,dynamics}`) | — | — | `pairing_comparison.{json,md}` |
| 8 | (conditional) Hard-bench rerun on dynamics #2 | — | — | — | Only if #2 passes the gate. Uses **V1 PPO** as-is; no retrain. |

Notes:

* Hydra invocation pattern is `paths.pairs_dir=… paths.dynamics_dir=… pairing.method=… +force=true`. Subordinate paths (`pairs_train`, `dynamics_model`, etc.) resolve via OmegaConf interpolation — verified by reading `config/paths.yaml`.
* `+force=true` is required so `train_dynamics.py` does not see a V1 checkpoint at `artifacts/dynamics/model.pt`; with the dynamics_dir override, no checkpoint exists at the new path, so `+force` is technically redundant but pass it anyway for clarity.
* OT-ε sweep, soft-OT expectation, and correlation-loss / residual-over-ridge are **explicitly deferred** to the post-P0B′ decision branches (see §"Final decision rules").

---

## Commands to run

All commands assume CWD `/Users/gabo/Developer/ITAM/IA/cellpath`, activated `.venv`, and `PYTHONPATH=.`. Each task is one-shot and idempotent given `+force=true`.

```bash
# --- Task 1 (mean-delta pairs) -------------------------------------------
.venv/bin/python scripts/build_pairs.py --config-name default \
    pairing.method=mean_delta \
    paths.pairs_dir=artifacts_v2/pairs_mean_delta

# --- Task 2 (dynamics on mean-delta pairs) -------------------------------
.venv/bin/python scripts/train_dynamics.py --config-name default \
    paths.pairs_dir=artifacts_v2/pairs_mean_delta \
    paths.dynamics_dir=artifacts_v2/dynamics_mean_delta_default \
    +force=true

# --- Task 3 (diagnostics on mean-delta) ----------------------------------
.venv/bin/python scripts/diagnose_pairing_noise.py \
    --pairs_dir artifacts_v2/pairs_mean_delta \
    --out artifacts_v2/diagnostics/pairing_noise_mean_delta.json

PYTHONPATH=. .venv/bin/python -m src.analysis.gate_breakdown \
    --dynamics_dir artifacts_v2/dynamics_mean_delta_default \
    --pairs_dir   artifacts_v2/pairs_mean_delta \
    --out         artifacts_v2/diagnostics/gate_breakdown_mean_delta

# --- Task 4 (random pairs) -----------------------------------------------
.venv/bin/python scripts/build_pairs.py --config-name default \
    pairing.method=random \
    paths.pairs_dir=artifacts_v2/pairs_random

# --- Task 5 (dynamics on random pairs) -----------------------------------
.venv/bin/python scripts/train_dynamics.py --config-name default \
    paths.pairs_dir=artifacts_v2/pairs_random \
    paths.dynamics_dir=artifacts_v2/dynamics_random_default \
    +force=true

# --- Task 6 (diagnostics on random) --------------------------------------
.venv/bin/python scripts/diagnose_pairing_noise.py \
    --pairs_dir artifacts_v2/pairs_random \
    --out artifacts_v2/diagnostics/pairing_noise_random.json

PYTHONPATH=. .venv/bin/python -m src.analysis.gate_breakdown \
    --dynamics_dir artifacts_v2/dynamics_random_default \
    --pairs_dir   artifacts_v2/pairs_random \
    --out         artifacts_v2/diagnostics/gate_breakdown_random

# --- Task 7 (comparator) -------------------------------------------------
.venv/bin/python scripts/compare_pairings.py \
    --runs ot:artifacts/pairs:artifacts/dynamics \
           mean_delta:artifacts_v2/pairs_mean_delta:artifacts_v2/dynamics_mean_delta_default \
           random:artifacts_v2/pairs_random:artifacts_v2/dynamics_random_default \
    --out artifacts_v2/diagnostics/pairing_comparison

# --- Task 8 (conditional: hard-bench on the new dynamics with V1 PPO) ----
# RUN ONLY IF Task 2 gate.json["passed"] == true.
.venv/bin/python scripts/evaluate_rl_hard.py \
    --vae_dir       artifacts/vae \
    --dynamics_dir  artifacts_v2/dynamics_mean_delta_default \
    --ppo_zip       artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip \
    --out_dir       artifacts_v2/eval_hard_mean_delta_v1policy \
    --k_values 1 2 3 8 --epsilon_values p25 p50 \
    --distance_bins 4-6 6-8 8-10 10-12 \
    --held_out_genes_only true,false --n_episodes 500 \
    --baselines random,always_noop,greedy_dyn_1,ridge_greedy,mean_delta_greedy
```

---

## Expected runtime (Apple Silicon M-series MPS; halve on Linux CUDA)

| Task | Wall-clock estimate |
|---|---|
| 1 — build mean-delta pairs (~84 genes, kd-tree, no Sinkhorn) | 5–10 min |
| 2 — train dynamics on mean-delta pairs (≤300 epochs, patience 35) | 60–90 min |
| 3 — diagnostics (mean-delta) | <5 min |
| 4 — build random pairs (trivial) | 1–2 min |
| 5 — train dynamics on random pairs (early-stop expected) | 30–60 min |
| 6 — diagnostics (random) | <5 min |
| 7 — comparator | <1 min |
| 8 — hard-bench (conditional) | 30–60 min |
| **Total (mandatory 1–7)** | **~2.0–3.0 hr** |
| **Total (incl. Task 8 if gate passes)** | **~2.5–4.0 hr** |

---

## Acceptance criteria (P0B′ as a whole)

Order matters: P0B′ has succeeded as a *diagnostic* phase as long as it yields a clear next move, regardless of whether the gate passes. The pass/fail bands below classify the verdict.

**Hard requirements (always enforced):**

1. `artifacts_v2/pairs_mean_delta/` and `artifacts_v2/pairs_random/` exist with all four Contract-2 npz files + `metadata.json`. Schemas pass `tests/test_p0b_prime_pairing.py`.
2. `artifacts_v2/dynamics_mean_delta_default/{model.pt, config.json, gate.json, val_metrics.json, ood_metrics.json, gate_diagnostics.json, epoch_metrics.json, checkpoint_comparison.json}` all written.
3. Same for `artifacts_v2/dynamics_random_default/`.
4. `artifacts_v2/diagnostics/pairing_comparison.{json,md}` produced.
5. **No file under `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/` is touched** (verify with `git status` on those paths — must be clean).
6. **No threshold in `config/dynamics.yaml::dynamics.gate` is modified.**
7. **No change to `src/models/dynamics.py` or `src/analysis/metrics.py`.**

**Outcome bands on `dynamics_mean_delta_default` (val pairs, MLP-minus-ridge Pearson):**

| Band | val margin | Interpretation |
|---|---|---|
| **PASS** | `≥ +0.030` | H_pair_primary supported. OT noise was the binding constraint. |
| **PARTIAL** | `[+0.015, +0.030)` | Pairing helped but not enough. Next step is soft-OT or correlation-loss on mean-delta pairs. |
| **NEUTRAL** | `[+0.005, +0.015)` | Pairing was not the bottleneck. Fall back to original P0B on V1 OT pairs. |
| **HURT** | `< +0.005` | mean-delta degraded the target. Investigate (see Final decision rules). |

Secondary checks (for *any* band): `uncertainty_calibration_spearman ≥ 0.20`, `ood_pearson ≥ 0.40`, `dim11_val_margin` strictly improved vs. V1 (`> −0.124`).

---

## Rollback criteria

* **Do not advance to P0C** under any band above. P0C is unblocked only when the gate passes AND the hard-bench rerun (Task 8) shows PPO either retains its primary-cell behavior or — better — shows reduced PPO−greedy_dyn_1 collinearity (evidence that the new dynamics field is less trivially monotone).
* **If the random-pairs dynamics passes the gate** (`dynamics_random_default` val margin ≥ +0.030): STOP. This indicates the gate metric or the V1 evaluation is degenerate. Escalate to the user; do not pivot V2 on random-pair results.
* **If `mean_delta` shows the HURT band** AND `random` matches or exceeds it: pairing semantics are not the dominant signal — escalate to V2 assumption review (option E in the user's decision list). Likely root cause: VAE latent geometry or a metrics-pipeline bug.
* **If any training run NaNs or fails the heteroscedastic NLL safety filter** (`gate_checkpoint_status == "none"`): record artifact, do not retry with hyper-param changes here — that crosses into P0B scope.

---

## Artifacts to write (new, all under `artifacts_v2/`)

* `artifacts_v2/pairs_mean_delta/{train,val,ood,combo}_pairs.npz`, `metadata.json`.
* `artifacts_v2/pairs_random/{train,val,ood,combo}_pairs.npz`, `metadata.json`.
* `artifacts_v2/dynamics_mean_delta_default/` — full dynamics output tree (see acceptance #2).
* `artifacts_v2/dynamics_random_default/` — full dynamics output tree (see acceptance #3).
* `artifacts_v2/diagnostics/pairing_noise_mean_delta.{json,md}`.
* `artifacts_v2/diagnostics/pairing_noise_random.{json,md}`.
* `artifacts_v2/diagnostics/gate_breakdown_mean_delta/{per_dim_margin.csv,per_gene_margin.csv,gate_breakdown_metadata.json}`.
* `artifacts_v2/diagnostics/gate_breakdown_random/{per_dim_margin.csv,per_gene_margin.csv,gate_breakdown_metadata.json}`.
* `artifacts_v2/diagnostics/pairing_comparison.{json,md}`.
* `artifacts_v2/interpretation_p0b_prime.md` — the filled interpretation template (see below).
* (Conditional) `artifacts_v2/eval_hard_mean_delta_v1policy/{results_table.md, metadata.json, ridge_buffers.npz, **/summary.json}`.
* Repo root: `P0B_PRIME_PAIRING_CORRECTION_PLAN.md` (copy of this plan).
* `PROGRESS.md` — appended new session entry only.

---

## Required tests

* `tests/test_p0b_prime_pairing.py` — TDD-first; tests must fail before the comparator is implemented:
  1. `test_mean_delta_pairs_schema_contract2` — given a fixture `pairs_mean_delta/` dir, asserts every npz has the Contract-2 keys and `n_latent == 32`.
  2. `test_random_pairs_schema_contract2` — same for random.
  3. `test_pairing_comparison_emits_required_keys` — runs `compare_pairings.py` on a tiny mock fixture (3 fake pairs dirs + 3 fake dynamics dirs containing only `gate.json` + `pairing_noise.json`) and asserts the `.json` carries `[pairing_method, n_train, pairing_noise_median, val_mlp_minus_ridge_pearson, ood_mlp_minus_ridge_pearson, dim11_val_margin, gate_passed]` per row.
  4. `test_v1_pairs_unchanged` — checksum verifier: snapshots `artifacts/pairs/metadata.json` SHA at test start, asserts it's the same at test end. Cheap insurance against accidental writes.
* Re-run existing suites unchanged: `tests/test_p0a_pairing_noise.py`, `tests/test_p0a_gate_breakdown.py`, `tests/test_dynamics.py`, `tests/test_environment.py`. Targeted: `pytest -k "pairing or gate or dynamics" -q`. Full repo: `pytest -q`.
* **No test runs against `artifacts/dynamics`'s model.pt** — Task 7 (the comparator) only reads `gate.json` and the breakdown CSVs; it must not load V1 weights from disk.

---

## Interpretation template (`artifacts_v2/interpretation_p0b_prime.md`)

```markdown
# P0B′ Interpretation — <YYYY-MM-DD>

## Inputs
| Run | Pairs dir | Dynamics dir | n_train pairs | pairing_noise_median |
|---|---|---|---:|---:|
| V1 OT (baseline) | artifacts/pairs | artifacts/dynamics | … | 0.8935 |
| mean_delta | artifacts_v2/pairs_mean_delta | artifacts_v2/dynamics_mean_delta_default | … | … |
| random | artifacts_v2/pairs_random | artifacts_v2/dynamics_random_default | … | … |

## Gate (val pairs)
| Run | mlp_pearson | ridge_pearson | mlp_minus_ridge | unc_spearman | passed? |
|---|---:|---:|---:|---:|---|
| V1 OT | 0.564 | 0.557 | +0.0074 | 0.249 | NO |
| mean_delta | … | … | … | … | … |
| random | … | … | … | … | … |

## OOD (report only)
| Run | mlp_pearson | ridge_pearson | mlp_minus_ridge | dim11_ood_margin |
|---|---:|---:|---:|---:|
| V1 OT | 0.490 | 0.450 | +0.0401 | −0.4331 |
| mean_delta | … | … | … | … |
| random | … | … | … | … |

## Verdict
- Outcome band (PASS / PARTIAL / NEUTRAL / HURT): **<band>**.
- H_pair_primary: **<supported / partially / not supported>**.
- H_pair_neg_control: random margin = … vs. mean_delta = … → **<distinct / collinear>**.
- dim 11 specifically: V1 val −0.124 → new val …; V1 OOD −0.433 → new OOD ….
- Pairing-noise drop: V1 0.8935 → new ….

## Next step (mechanically by §"Final decision rules")
… (one line) …
```

---

## Final decision rules after P0B′

Apply the matching row; do not interpolate.

| Result | Next action |
|---|---|
| mean_delta **PASS** AND random **does not pass** AND dim 11 val margin improved | (a) Promote `artifacts_v2/dynamics_mean_delta_default` to gate-passing status. (b) Update `V2_RESEARCH_PLAN.md` decision log: pairing was the binding constraint; demote original P0B to an ablation; queue P0C (hard-bench Task 8 + optional reward-side P0C.1/.2) for the next session. |
| mean_delta **PARTIAL** | Plan a P0B″: implement **soft-OT expectation** in `pair_ot` (replace `argmax` with `T[:, j].T @ z_ctrl` for each pert cell) AND/OR add correlation loss on `dynamics_mean_delta_default` pairs. Keep architecture pinned. Run one config first (soft-OT alone, λ_corr=0). |
| mean_delta **NEUTRAL** | Pairing was *not* the bottleneck. Resume the original P0B (correlation loss `λ_corr ∈ {0.05, 0.10, 0.30}` + residual-over-ridge) on V1 OT pairs. The pairing-noise ceiling is now quantified — record it as a known cap. |
| mean_delta **HURT** (margin < +0.005) | Investigate before retraining: (i) confirm Contract-2 integrity of `pairs_mean_delta/`, (ii) check whether the per-gene Δz signal collapsed for top-signal genes (compare `mean_delta_signal` in pairing-noise JSON before vs. after — should *increase*, not decrease, if mean-delta is doing its job). If integrity passes, escalate: this implies the OT noise was *informative* and mean-delta destroys real per-cell signal. Pivot back to original P0B but log this as a P1 finding. |
| random **passes the gate** (any margin ≥ +0.030) | STOP. Treat as a gate-metric defect. Escalate to V2 assumption review. Do not advance to P0C. |
| Any training NaN / no preferred checkpoint / OOD collapse (ood_pearson < 0.30) | Record, stop, escalate. Do not retry with hyper-param changes inside P0B′. |

---

## Task breakdown (executor-facing)

> Each task is a single commit-worthy unit. The execution agent should mark each task complete only after the **expected output** is observed.

### Task 1: Commit the plan file to the repo and stage it

**Files:**
- Create: `/Users/gabo/Developer/ITAM/IA/cellpath/P0B_PRIME_PAIRING_CORRECTION_PLAN.md`

- [ ] **Step 1.1: Copy this plan verbatim to the repo root.**

    Read `/Users/gabo/.claude/plans/you-are-the-cellpath-concurrent-bubble.md` and write the identical content to `/Users/gabo/Developer/ITAM/IA/cellpath/P0B_PRIME_PAIRING_CORRECTION_PLAN.md`. Do not paraphrase.

- [ ] **Step 1.2: Stage and commit (do not push).**

```bash
git add P0B_PRIME_PAIRING_CORRECTION_PLAN.md
git commit -m "$(cat <<'EOF'
docs: add P0B' pairing-correction plan

Why: P0A pairing-noise median 0.8935 makes the dynamics target's
within-gene residual the binding constraint. Reorder V2 so pairing
correction (mean_delta + random control) precedes the original P0B
architecture/loss fix.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: `git status` shows clean tree.

---

### Task 2: Write the failing tests (TDD)

**Files:**
- Create: `tests/test_p0b_prime_pairing.py`

- [ ] **Step 2.1: Author the test file.** Mirror the style of `tests/test_p0a_pairing_noise.py`. Tests must reference yet-to-exist `scripts/compare_pairings.py` and `artifacts_v2/pairs_*/` paths.

```python
"""P0B' — pairing comparator + Contract-2 schema regression tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


CONTRACT2_KEYS_TRAIN = {"z_ctrl", "gene_idx", "z_pert"}
CONTRACT2_KEYS_COMBO = {"z_ctrl", "gene_idx_a", "gene_idx_b", "z_pert_ab"}


def _schema_ok(npz_path: Path, keys: set[str], n_latent: int = 32) -> None:
    data = np.load(npz_path)
    assert set(data.files) >= keys, f"{npz_path} missing keys: {keys - set(data.files)}"
    if "z_ctrl" in data.files:
        assert data["z_ctrl"].shape[1] == n_latent


@pytest.mark.parametrize("subdir", ["pairs_mean_delta", "pairs_random"])
def test_pairs_schema_contract2(subdir: str) -> None:
    root = Path("artifacts_v2") / subdir
    if not root.exists():
        pytest.skip(f"{root} not produced yet — runs before Task 4+ are expected to skip")
    for fname, keys in [
        ("train_pairs.npz", CONTRACT2_KEYS_TRAIN),
        ("val_pairs.npz", CONTRACT2_KEYS_TRAIN),
        ("ood_pairs.npz", CONTRACT2_KEYS_TRAIN),
        ("combo_pairs.npz", CONTRACT2_KEYS_COMBO),
    ]:
        _schema_ok(root / fname, keys)
    assert (root / "metadata.json").exists()


def test_v1_pairs_unchanged() -> None:
    """Sanity check: V1 OT pair metadata is byte-identical to its session-start SHA.

    This guards against accidental writes to artifacts/pairs/ during P0B' runs.
    The expected SHA is read from a sibling fixture file maintained by the executor
    on the first P0B' run; if absent, the test seeds it.
    """
    target = Path("artifacts/pairs/metadata.json")
    if not target.exists():
        pytest.skip("artifacts/pairs/metadata.json not present")
    fixture = Path("tests/fixtures/v1_pairs_metadata.sha256")
    h = hashlib.sha256(target.read_bytes()).hexdigest()
    if not fixture.exists():
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(h)
        pytest.skip("seeded fixture; rerun")
    assert fixture.read_text().strip() == h, "V1 OT pairs metadata changed — investigate"


def test_pairing_comparison_emits_required_keys(tmp_path: Path) -> None:
    """Comparator must read gate.json + pairing_noise.json and emit a tidy row schema."""
    pytest.importorskip("scripts.compare_pairings")
    from scripts import compare_pairings  # noqa: F401

    fake_runs = []
    for name in ("ot", "mean_delta", "random"):
        pdir = tmp_path / f"pairs_{name}"
        ddir = tmp_path / f"dynamics_{name}"
        pdir.mkdir(); ddir.mkdir()
        (pdir / "metadata.json").write_text(json.dumps({"n_train": 1000, "pairing_method": name}))
        (ddir / "gate.json").write_text(json.dumps({
            "passed": False,
            "primary": {"pearson_r": 0.5, "baselines": {"linear_ridge": {"pearson_r": 0.49}}},
            "ood":     {"pearson_r": 0.45, "baselines": {"linear_ridge": {"pearson_r": 0.43}}},
            "uncertainty_calibration": {"spearman": 0.22},
        }))
        (ddir / "pairing_noise.json").write_text(json.dumps({"summary": {"median_noise_ratio": 0.7}}))
        fake_runs.append((name, str(pdir), str(ddir)))

    out = tmp_path / "pairing_comparison"
    compare_pairings.main(runs=fake_runs, out=str(out))

    record = json.loads((out.with_suffix(".json")).read_text())
    required = {
        "pairing_method", "n_train", "pairing_noise_median",
        "val_mlp_minus_ridge_pearson", "ood_mlp_minus_ridge_pearson",
        "uncertainty_spearman", "gate_passed",
    }
    assert set(record["rows"][0].keys()) >= required
```

- [ ] **Step 2.2: Run the tests; confirm they fail/skip correctly before any P0B′ runs.**

```bash
.venv/bin/pytest tests/test_p0b_prime_pairing.py -v
```

Expected: `test_pairs_schema_contract2[pairs_mean_delta]` and `[pairs_random]` SKIP (dirs not built yet). `test_v1_pairs_unchanged` SKIP-then-pass (seeds fixture). `test_pairing_comparison_emits_required_keys` FAILS with `ModuleNotFoundError: scripts.compare_pairings` (script not written yet).

- [ ] **Step 2.3: Commit.**

```bash
git add tests/test_p0b_prime_pairing.py tests/fixtures/v1_pairs_metadata.sha256
git commit -m "$(cat <<'EOF'
test: add P0B' pairing-comparator + Contract-2 schema regressions

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Implement `scripts/compare_pairings.py`

**Files:**
- Create: `scripts/compare_pairings.py`

- [ ] **Step 3.1: Implement the minimum to make `test_pairing_comparison_emits_required_keys` pass.**

The script reads, per run-spec `<method>:<pairs_dir>:<dynamics_dir>`:
- `<pairs_dir>/metadata.json` → `n_train`, `pairing_method`.
- `<dynamics_dir>/gate.json` → `val_mlp_minus_ridge_pearson` (= `primary.pearson_r - primary.baselines.linear_ridge.pearson_r`), same for ood, `uncertainty_calibration.spearman`, `passed`.
- `<dynamics_dir>/pairing_noise.json` if present, else `<pairs_dir>/pairing_noise.json` if present, else `None`. (We will write `pairing_noise_*.json` under `artifacts_v2/diagnostics/`, so accept an explicit override `--noise_json <method>=<path>` repeated for each run.)

Output: `<out>.json` (tidy `{rows: [...]}`) and `<out>.md` (markdown table). No model loading; no torch import.

```python
"""P0B' — side-by-side comparator across pairing-method runs.

Reads {gate.json, metadata.json, pairing_noise.json} and emits a single
machine + human-readable comparison. No model is loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _row(method: str, pairs_dir: str, dynamics_dir: str, noise_json: Path | None) -> dict[str, Any]:
    pmeta = json.loads((Path(pairs_dir) / "metadata.json").read_text())
    gate = json.loads((Path(dynamics_dir) / "gate.json").read_text())
    prim = gate["primary"]
    ridge_p = float(prim["baselines"]["linear_ridge"]["pearson_r"])
    ood = gate.get("ood") or {}
    ood_ridge = float((ood.get("baselines") or {}).get("linear_ridge", {}).get("pearson_r", float("nan"))) if ood else float("nan")
    noise = float("nan")
    cand: Path | None = noise_json
    if cand is None:
        for c in (Path(dynamics_dir) / "pairing_noise.json", Path(pairs_dir) / "pairing_noise.json"):
            if c.exists():
                cand = c
                break
    if cand is not None and cand.exists():
        noise = float(json.loads(cand.read_text())["summary"]["median_noise_ratio"])
    return {
        "pairing_method": method,
        "n_train": int(pmeta.get("n_train", 0)),
        "pairing_noise_median": noise,
        "val_mlp_pearson": float(prim["pearson_r"]),
        "val_ridge_pearson": ridge_p,
        "val_mlp_minus_ridge_pearson": float(prim["pearson_r"]) - ridge_p,
        "ood_mlp_pearson": float(ood.get("pearson_r", float("nan"))) if ood else float("nan"),
        "ood_ridge_pearson": ood_ridge,
        "ood_mlp_minus_ridge_pearson": (float(ood.get("pearson_r", float("nan"))) - ood_ridge) if ood else float("nan"),
        "uncertainty_spearman": float(gate["uncertainty_calibration"]["spearman"]),
        "gate_passed": bool(gate["passed"]),
    }


def main(*, runs: Iterable[tuple[str, str, str]] | None = None, out: str | None = None) -> int:
    if runs is None or out is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--runs", nargs="+", required=True,
                        help="One or more <method>:<pairs_dir>:<dynamics_dir>")
        ap.add_argument("--noise_json", nargs="*", default=[],
                        help="Optional <method>=<path> overrides for pairing_noise.json")
        ap.add_argument("--out", required=True)
        args = ap.parse_args()
        runs_parsed = [tuple(r.split(":", 2)) for r in args.runs]
        noise_map = dict(s.split("=", 1) for s in args.noise_json)
    else:
        runs_parsed = list(runs)
        noise_map = {}
        args = argparse.Namespace(out=out)

    rows = [_row(m, p, d, Path(noise_map[m]) if m in noise_map else None) for (m, p, d) in runs_parsed]
    out_json = Path(args.out + ".json")
    out_md = Path(args.out + ".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, indent=2))

    cols = ["pairing_method", "n_train", "pairing_noise_median",
            "val_mlp_minus_ridge_pearson", "ood_mlp_minus_ridge_pearson",
            "uncertainty_spearman", "gate_passed"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols
        ) + " |"
        for r in rows
    )
    out_md.write_text(f"# P0B' pairing comparison\n\n{header}\n{sep}\n{body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3.2: Confirm tests pass.**

```bash
.venv/bin/pytest tests/test_p0b_prime_pairing.py::test_pairing_comparison_emits_required_keys -v
```

Expected: PASS.

- [ ] **Step 3.3: Commit.**

```bash
git add scripts/compare_pairings.py
git commit -m "$(cat <<'EOF'
feat(p0b_prime): comparator for OT vs mean_delta vs random dynamics

Reads gate.json + pairing_noise.json from each run dir and emits a
side-by-side .json/.md table. No model loading; pure aggregation.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Build mean-delta pairs

- [ ] **Step 4.1: Run.**

```bash
.venv/bin/python scripts/build_pairs.py --config-name default \
    pairing.method=mean_delta \
    paths.pairs_dir=artifacts_v2/pairs_mean_delta 2>&1 | tee artifacts_v2/pairs_mean_delta.log
```

Expected stdout tail: `Done. Run 'make dynamics' next.` and four `.npz` + `metadata.json` in `artifacts_v2/pairs_mean_delta/`.

- [ ] **Step 4.2: Diagnose pairing noise.**

```bash
.venv/bin/python scripts/diagnose_pairing_noise.py \
    --pairs_dir artifacts_v2/pairs_mean_delta \
    --out artifacts_v2/diagnostics/pairing_noise_mean_delta.json
```

Expected: median noise ratio markedly below V1's 0.8935 (target band 0.55–0.70; flag if outside).

- [ ] **Step 4.3: Run schema test.**

```bash
.venv/bin/pytest tests/test_p0b_prime_pairing.py::test_pairs_schema_contract2 -v
```

Expected: `[pairs_mean_delta]` PASS, `[pairs_random]` SKIP (still pending).

- [ ] **Step 4.4: Commit artifacts (not into git LFS — they go under `artifacts_v2/` which is gitignored).** Verify they are gitignored, then commit only the log + diagnostics JSON if those are tracked. (Likely nothing to commit here besides PROGRESS.md updates, which we batch at the end.)

---

### Task 5: Train dynamics on mean-delta pairs

- [ ] **Step 5.1: Train.**

```bash
.venv/bin/python scripts/train_dynamics.py --config-name default \
    paths.pairs_dir=artifacts_v2/pairs_mean_delta \
    paths.dynamics_dir=artifacts_v2/dynamics_mean_delta_default \
    +force=true 2>&1 | tee artifacts_v2/dynamics_mean_delta_default.log
```

Expected: `Dynamics validation gate <PASSED|FAILED>. ...` then exit 0 (pass) or exit 1 (fail). Both exits are acceptable for the *diagnostic* purpose; record either.

- [ ] **Step 5.2: Read `artifacts_v2/dynamics_mean_delta_default/gate.json`.**

Capture for the interpretation file:
- `primary.pearson_r`, `primary.baselines.linear_ridge.pearson_r`, their difference.
- `ood.pearson_r`, ditto for ridge.
- `uncertainty_calibration.spearman`.
- `passed`.

- [ ] **Step 5.3: Gate breakdown on the new run.**

```bash
PYTHONPATH=. .venv/bin/python -m src.analysis.gate_breakdown \
    --dynamics_dir artifacts_v2/dynamics_mean_delta_default \
    --pairs_dir   artifacts_v2/pairs_mean_delta \
    --out         artifacts_v2/diagnostics/gate_breakdown_mean_delta
```

Expected: `per_dim_margin.csv`, `per_gene_margin.csv`, `gate_breakdown_metadata.json` written. Inspect dim 11 row in `per_dim_margin.csv` for both val and ood.

---

### Task 6: Build random pairs, train dynamics, diagnose

- [ ] **Step 6.1: Build random pairs.**

```bash
.venv/bin/python scripts/build_pairs.py --config-name default \
    pairing.method=random \
    paths.pairs_dir=artifacts_v2/pairs_random 2>&1 | tee artifacts_v2/pairs_random.log

.venv/bin/python scripts/diagnose_pairing_noise.py \
    --pairs_dir artifacts_v2/pairs_random \
    --out artifacts_v2/diagnostics/pairing_noise_random.json
```

Expected: random noise ratio at or near 1.0 (true upper bound). Anything markedly < 1.0 indicates a config or seeding bug.

- [ ] **Step 6.2: Train dynamics.**

```bash
.venv/bin/python scripts/train_dynamics.py --config-name default \
    paths.pairs_dir=artifacts_v2/pairs_random \
    paths.dynamics_dir=artifacts_v2/dynamics_random_default \
    +force=true 2>&1 | tee artifacts_v2/dynamics_random_default.log
```

Expected: gate FAILS (this is the desired negative-control behavior). If gate PASSES, halt and escalate (rollback rule).

- [ ] **Step 6.3: Gate breakdown.**

```bash
PYTHONPATH=. .venv/bin/python -m src.analysis.gate_breakdown \
    --dynamics_dir artifacts_v2/dynamics_random_default \
    --pairs_dir   artifacts_v2/pairs_random \
    --out         artifacts_v2/diagnostics/gate_breakdown_random
```

---

### Task 7: Run the comparator

- [ ] **Step 7.1: Run.**

```bash
.venv/bin/python scripts/compare_pairings.py \
    --runs ot:artifacts/pairs:artifacts/dynamics \
           mean_delta:artifacts_v2/pairs_mean_delta:artifacts_v2/dynamics_mean_delta_default \
           random:artifacts_v2/pairs_random:artifacts_v2/dynamics_random_default \
    --noise_json ot=artifacts_v2/diagnostics/pairing_noise.json \
                 mean_delta=artifacts_v2/diagnostics/pairing_noise_mean_delta.json \
                 random=artifacts_v2/diagnostics/pairing_noise_random.json \
    --out artifacts_v2/diagnostics/pairing_comparison
```

Expected: `pairing_comparison.json` and `pairing_comparison.md` written, three rows.

---

### Task 8 (conditional): Hard benchmark on the new dynamics

- [ ] **Step 8.1: Gate the conditional.** Read `artifacts_v2/dynamics_mean_delta_default/gate.json`. Proceed only if `passed: true`. Otherwise, skip Task 8 and proceed to Task 9.

- [ ] **Step 8.2: Run the hard benchmark with the V1 PPO checkpoint and the new dynamics.**

```bash
.venv/bin/python scripts/evaluate_rl_hard.py \
    --vae_dir       artifacts/vae \
    --dynamics_dir  artifacts_v2/dynamics_mean_delta_default \
    --ppo_zip       artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip \
    --out_dir       artifacts_v2/eval_hard_mean_delta_v1policy \
    --k_values 1 2 3 8 --epsilon_values p25 p50 \
    --distance_bins 4-6 6-8 8-10 10-12 \
    --held_out_genes_only true,false --n_episodes 500 \
    --baselines random,always_noop,greedy_dyn_1,ridge_greedy,mean_delta_greedy
```

Expected: `results_table.md` produced. Primary cell row (`k3_epsp25_bin8-10_splitood`) reports new PPO success and (most importantly) new PPO−greedy_dyn_1 delta — this is the planning-vs-greedy diagnostic for the new dynamics field.

---

### Task 9: Write interpretation + update PROGRESS.md

**Files:**
- Create: `artifacts_v2/interpretation_p0b_prime.md`
- Modify: `PROGRESS.md`

- [ ] **Step 9.1: Fill the interpretation template.** Use the template in §"Interpretation template" verbatim; replace `…` with measured numbers. Apply §"Final decision rules" to select the "Next step" line.

- [ ] **Step 9.2: Append a PROGRESS.md session entry at the top** per CLAUDE.md §8. Format:

```markdown
## Session 2026-MM-DD-HHMM  (agent: research-lead)

**Phase:** P0B′ — pairing correction (V2 reorder)
**Status:** Built artifacts_v2/{pairs_mean_delta,pairs_random,dynamics_mean_delta_default,dynamics_random_default}. Comparator + breakdown + interpretation written.
**Metrics:**
| Component | Target | Current | Status |
| --- | --- | --- | --- |
| val_mlp_minus_ridge_pearson (mean_delta) | ≥ +0.030 | … | …(PASS/PARTIAL/NEUTRAL/HURT) |
| ood_mlp_minus_ridge_pearson (mean_delta) | ≥ +0.030 | … | … |
| pairing_noise_median (mean_delta) | (drop vs 0.8935) | … | … |
| val_mlp_minus_ridge_pearson (random control) | (should not pass) | … | … |
**Blockers:** none / P2: …
**Next:** (one-line action from Final decision rules) / …
```

- [ ] **Step 9.3: Commit.**

```bash
git add artifacts_v2/interpretation_p0b_prime.md PROGRESS.md
git commit -m "$(cat <<'EOF'
docs(p0b_prime): interpretation + PROGRESS update

Outcome band: <band>. Next: <one-line>.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Full test sweep

- [ ] **Step 10.1: Run.**

```bash
.venv/bin/pytest -q
```

Expected: 217 passed, 2 skipped (baseline) + 4 new tests in `test_p0b_prime_pairing.py` → 221 passed, 2 skipped, or with one skip if the fixture-seed test skips on first session — match the baseline pattern.

- [ ] **Step 10.2: Verify V1 frozen artifacts are untouched.**

```bash
git status -- artifacts/ artifacts_64/ artifacts/rl_sweeps/
```

Expected: empty / clean. If anything appears here, halt and escalate.

---

## Self-review

* **Spec coverage:** Every requested section in the user prompt (Objective; Why P0A changed the order; Hypotheses; Files to inspect/modify/not-modify; Experiment matrix; Commands; Runtime; Acceptance; Rollback; Artifacts; Tests; Interpretation template; Final decision rules) is present.
* **No placeholders:** No "TBD" / "implement later" / "similar to Task N" — every code snippet and command is concrete.
* **Type consistency:** `compare_pairings.main(runs=..., out=...)` is referenced identically from both the test and the script; the field names in `pairing_comparison.json` (e.g., `val_mlp_minus_ridge_pearson`, `pairing_noise_median`, `gate_passed`) are consistent between test, comparator, and interpretation template.
* **Sacred-rule conformance:** No VAE retrain; no `torch.device()` outside `src/utils/device.py` (we don't add any); no `random.seed`/`np.random.seed`/`torch.manual_seed` outside `src/utils/seeding.py` (we don't add any); no path hardcoding (all paths go through Hydra `paths.*`); no inline metric definitions (we reuse `dynamics_validation_gate` and `gate_diagnostics`); no PPO retrain (Task 8 reuses V1 PPO); no gate-threshold lowering. RL training is not started — Task 8 is *evaluation* of V1 PPO on the new dynamics, which is read-only with respect to `artifacts/rl_sweeps/`.
* **V1 protection:** Test 4 (`test_v1_pairs_unchanged`) plus Task 10.2 give two independent guards against accidental writes to V1 dirs.
