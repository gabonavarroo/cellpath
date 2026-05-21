# V3A — Latent Audit & 64D Pivot Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Plan file location**: Authored at `/Users/gabo/.claude/plans/you-are-the-cellpath-cheeky-parasol.md` under Claude Code plan mode. After approval, copy to repo as `/Users/gabo/Developer/ITAM/IA/cellpath/V3A_LATENT_AUDIT_AND_64D_PLAN.md`. The V3A interpretation step (A8) upgrades `V3_RESEARCH_PLAN.md` from stub to full plan.

**Goal:** Test the V3 latent-geometry hypothesis (a higher-dim latent exposes a less-saturated dynamics field where PPO − greedy_dyn_2 ≥ +0.05 pp becomes achievable) with the smallest decisive experiment, by running two parallel 64D tracks — a reuse track (legacy VAE) and a fresh track (new 64D NB VAE) — and comparing both against V2 primary.

**Architecture:** Reuse Hydra + existing entry points (`train_vae.py`, `build_pairs.py`, `train_dynamics.py`, `train_rl.py`, `evaluate_rl_hard.py`, `probe_reachability.py`) with `paths.*` overrides routing all writes to `artifacts_v3/`. Two parallel tracks: `legacy` (copy `artifacts_64/vae/` → `artifacts_v3/vae_n64_legacy/`, then fresh OT pairs + fresh RoR dynamics + fresh PPO) and `nb_fresh` (fresh 64D NB VAE → fresh pairs → fresh RoR dynamics → fresh PPO). Single seed (42) first, escalate to 4 seeds only if PPO − grd2 ≥ +0.03 pp at any cell. `artifacts_64/`, `artifacts_v2/`, `artifacts/` remain frozen and read-only.

**Tech Stack:** Python 3.11, scvi-tools (VAE), Hydra (config), PyTorch (dynamics MLP, RoR), stable-baselines3-contrib (MaskablePPO), POT (OT pairing), anndata/scanpy. macOS Apple Silicon MPS / Linux CUDA.

---

## 1. Context

V2 final result (`artifacts_v2/V2_FINAL_REPORT.md`): under the V2 hard benchmark (K=3, ε=p25=3.166, bin 8–10, OOD, 4 seeds × 300 episodes), the V2 primary configuration `RoR_corr010 × C2` achieves **success = 0.941 ± 0.048**, **+77 pp over random**, but **PPO − greedy_dyn_2 is never ≥ +0.05 pp anywhere** on the hardness matrix. The honest framing: PPO compressed a 2-step lookahead into a feedforward controller; it did not discover a strategy superior to depth-2 planning. The hypothesized cause is that the 32D latent geometry is **locally well-conditioned** — `greedy_dyn_2` saturates at 1.000 at the primary cell, leaving no room for PPO to differentiate.

The V3 stub (`V3_RESEARCH_PLAN.md`) sets the headline V3 hypothesis: a higher-dim or semi-supervised latent produces a less-saturated field. V3A is the first decisive experiment toward this hypothesis: test 64D, the simplest available perturbation of the latent geometry. Two prior 64D references exist:

* `artifacts_64/` (legacy V1-era 64D ablation, May 2026): complete VAE + pairs + state_linear_skip dynamics (gate FAILED, OOD Pearson 0.369), **no RL artifacts**. Architecture is NOT V2's RoR.
* `artifacts/contraction_auto/` vs `artifacts_64/contraction_auto/` (per `V3_RESEARCH_PLAN.md` §11): 64D shows larger per-step contraction (1.349) than 32D (1.008), but higher variance (worst_improvement = −1.499). This is the empirical motivation that a 64D field *might* be harder to plan in.

**V3A intent**: head-to-head V3 primary candidate vs V2 primary, with both reuse-legacy and fresh-VAE tracks running in parallel as a Bayesian hedge against VAE-config drift.

---

## 2. Audit findings (from `artifacts_64/` inventory)

### 2.1 What exists

| Path | Size | Status | Notes |
|---|---|---|---|
| `artifacts_64/vae/model/model.pt` | 6.2 M | ✓ complete | scVI checkpoint; no Hydra snapshot |
| `artifacts_64/vae/latents.h5ad` | 1.0 G | ✓ complete | `X_scVI` shape (n_obs, 64) |
| `artifacts_64/vae/gene_vocab.json` | 1.4 K | ✓ complete | **105 genes (84 train + 21 OOD), ctrl_idx=0, noop_idx=105** |
| `artifacts_64/vae/z_reference_centroid.npy` | 384 B | ✓ complete | 64-d, float32 |
| `artifacts_64/vae/epsilon_success.json` | 106 B | ⚠ partial | **only p90 = 4.435 (11 855 control cells); p25 NOT present** |
| `artifacts_64/pairs/train_pairs.npz` | — | ✓ complete | 38 958 pairs, OT seed=42, ε=0.05 |
| `artifacts_64/pairs/val_pairs.npz` | — | ✓ complete | 4 324 |
| `artifacts_64/pairs/ood_pairs.npz` | — | ✓ complete | 14 549 (held-out genes) |
| `artifacts_64/pairs/combo_pairs.npz` | — | ✓ complete | 35 995 |
| `artifacts_64/dynamics/model.pt` | — | ✗ wrong architecture | uses `use_state_linear_skip=true`, NOT RoR; gate **FAILED** (margin_vs_linear_ridge_pearson = −0.019); val Pearson 0.596; **OOD Pearson 0.369** |
| `artifacts_64/contraction_auto/` | — | ✓ informative | fraction_improved 98.4 %, mean_improvement 1.35× — based on gate-failing dynamics |
| `artifacts_64/dynamics_variants/` | — | ✓ informative | 5 architecture experiments; all show similar gate failure |
| `artifacts_64/rl*` | — | ✗ absent | no PPO checkpoints, no eval, no reachability probe |
| `artifacts_64/.hydra/` | — | ✗ absent | no Hydra config snapshot |

### 2.2 Red flags

1. **Legacy dynamics is wrong architecture for V3**: uses `state_linear_skip`, not V2's `use_residual_over_ridge`. V3A must retrain dynamics.
2. **Legacy OOD Pearson 0.369 is below V3 stub threshold (0.40)**. Either (a) 64D fundamentally generalizes worse OOD, or (b) the legacy VAE used a non-standard scVI config (no Hydra snapshot to verify). The parallel-track design (Option C reuse + Option B fresh) directly disambiguates (a) vs (b).
3. **ε_p25 not present** — only p90. Must be recomputed on 64D latents to match V2 hard-bench protocol.
4. **No Hydra snapshot** — legacy scVI hyperparameters (KL weight, dispersion, gene_likelihood) are not recorded. Defaults are likely (`nb`, `dispersion: gene`, `latent_distribution: normal`), but unverified.
5. **Gene split sanity**: legacy 64D has 105 genes (84 train + 21 OOD); V1 has same count. Need to verify the exact OOD index set matches V2's (Phase A0).

---

## 3. Strategy decision: parallel reuse + fresh tracks

### 3.1 Ranked options

1. **Track L (legacy reuse) + Track N (NB fresh) IN PARALLEL** (this plan) — Bayesian hedge: Track L probes hypothesis "64D field is less saturated"; Track N controls for "legacy VAE config drift caused OOD Pearson 0.369". Total wall-clock ~6–7 h. **Recommended; user-selected.**
2. **Audit then sequential reuse** (Option C from prompt) — ~4 h but no insurance against legacy VAE drift.
3. **Fresh-only** (Option B from prompt) — ~6 h sequential; misses fast reuse signal and forgoes legacy artifacts entirely.
4. **Audit only with no retrain** (Option A from prompt) — rejected; cannot test V3 hypothesis.
5. **Direct to RoR+PPO with no audit** (Option D from prompt) — rejected; gene-split contamination risk.

### 3.2 Reuse vs retrain decision per artifact (Track L)

| Artifact | Decision | Why |
|---|---|---|
| Legacy 64D VAE (`model/`, `latents.h5ad`, `gene_vocab.json`, `z_reference_centroid.npy`) | **Reuse** (copy to `artifacts_v3/vae_n64_legacy/`) | Complete; deterministic. Saves ~2 h. Frozen-source safety via copy (no symlink). |
| Legacy `epsilon_success.json` (p90 only) | **Augment** | Recompute p10/p25/p50/p75/p90 on legacy 64D latents using V2 method. Write `artifacts_v3/vae_n64_legacy/epsilon_success.json`. |
| Legacy pairs (`train/val/ood/combo.npz`) | **Rebuild** | `build_pairs.py` has evolved since V1; V2 P0B′ / P0B″ corrections may not be in legacy pairs. Rebuild on legacy 64D latents → `artifacts_v3/pairs_n64_legacy/`. Cost ~30 min. |
| Legacy dynamics (`state_linear_skip`) | **Retrain** | Wrong architecture for V3 (V2 primary is RoR). Train fresh RoR + corr0.10 on legacy 64D pairs → `artifacts_v3/dynamics_n64_legacy_ror_corr010/`. |
| Legacy contraction diagnostics | **Reference only** | Inform V3A interpretation; do not consume in pipeline. |

### 3.3 Track N (fresh NB) artifacts

| Artifact | Decision | Notes |
|---|---|---|
| Fresh 64D NB VAE | **Train** (~2 h) | `vae.n_latent=64`, `vae.gene_likelihood=nb`, all other V2 defaults. Output `artifacts_v3/vae_n64_nb/`. |
| Fresh ε quantiles | **Compute** | Standard `train_vae.py` writes `epsilon_success.json` with p10/p25/p50/p75/p90. |
| Fresh pairs | **Build** (~30 min) | Same `build_pairs.py` invocation; output `artifacts_v3/pairs_n64_nb/`. |
| Fresh dynamics | **Train** (~30–60 min) | RoR + corr0.10; output `artifacts_v3/dynamics_n64_nb_ror_corr010/`. |
| Fresh PPO | **Train** (~5–10 min) | C2-style; seed 42 only; output `artifacts_v3/rl_n64_nb_c2_k3_1M_seed42/`. |

---

## 4. V3A pipeline — 9 phases

Phases A1-bg, A2-bg, A4-bg, A5-bg run in background concurrent with their foreground counterparts. A0 → A1 → A2 → A3 → A4 → A5 → A6 is the foreground critical path on Track L.

### Phase A0 — Audit legacy 64D VAE (read-only, ~30 min, FOREGROUND)

**Goal:** Verify legacy 64D VAE is reusable; lock the V3A path (Track L proceeds, Track N kicks off in background).

**Files / commands:**

- [ ] **A0.1 — Inspect gene_vocab and gene split alignment**

```bash
python -c "
import json, numpy as np
v1 = json.load(open('artifacts/vae/gene_vocab.json'))
v64 = json.load(open('artifacts_64/vae/gene_vocab.json'))
v2_pairs_meta = json.load(open('artifacts/pairs/metadata.json'))
print('V1 n_genes:', len(v1.get('gene_to_idx', v1)))
print('V64 n_genes:', len(v64.get('gene_to_idx', v64)))
print('V64 vocab == V1 vocab?:', v1 == v64)
print('V2 OOD split (first 5):', v2_pairs_meta.get('ood_gene_indices', [])[:5])
"
```

Expected: vocabularies bit-identical (same Norman HVG selection) and V2 OOD split documented. If vocabularies differ, **abort to Track-N-only** (Phase A1-bg becomes critical path).

- [ ] **A0.2 — Inspect latents.h5ad schema**

```bash
python -c "
import anndata as ad
a = ad.read_h5ad('artifacts_64/vae/latents.h5ad')
print('n_obs:', a.n_obs, 'n_vars:', a.n_vars)
print('obsm keys:', list(a.obsm.keys()))
print('X_scVI shape:', a.obsm['X_scVI'].shape if 'X_scVI' in a.obsm else 'MISSING')
print('obs columns:', list(a.obs.columns))
print('cell_barcode sample:', a.obs.index[:3].tolist())
"
```

Expected: `n_obs` matches V1 (`artifacts/vae/latents.h5ad`), `X_scVI` is (n_obs, 64), `obs` contains `perturbation` / `is_control` / `gene` columns matching V2 pipeline. If schema mismatches, abort Track L.

- [ ] **A0.3 — Load scVI model and inspect hyperparameters**

```bash
python -c "
import scvi, anndata as ad
adata = ad.read_h5ad('artifacts_64/vae/latents.h5ad')
m = scvi.model.SCVI.load('artifacts_64/vae/model', adata=adata)
print('n_latent:', m.module.n_latent)
print('gene_likelihood:', m.module.gene_likelihood)
print('dispersion:', m.module.dispersion if hasattr(m.module, 'dispersion') else 'unknown')
print('n_layers:', m.module.n_layers if hasattr(m.module, 'n_layers') else 'unknown')
"
```

Expected: `n_latent=64`, `gene_likelihood='nb'`. If anything else (e.g. `zinb`, `poisson`), flag in audit report — Track N becomes the V3.1 reference.

- [ ] **A0.4 — Inspect centroid**

```bash
python -c "
import numpy as np
c = np.load('artifacts_64/vae/z_reference_centroid.npy')
print('shape:', c.shape, 'dtype:', c.dtype)
print('L2 norm:', float(np.linalg.norm(c)))
print('first 8 components:', c[:8])
"
```

Expected: shape `(64,)`, L2 norm in plausible scVI-latent range (≈ 3–20). If degenerate (norm ≈ 0 or >>50), abort Track L.

- [ ] **A0.5 — Write audit verdict**

Write `artifacts_v3/audit_v3a.md` (one of: REUSE OK, REUSE WITH CAVEAT, RETRAIN ONLY). Include all printed outputs above and a one-line verdict.

**Acceptance criteria:**
* Gene vocabulary alignment with V1: **identical** OR audit notes the exact diff.
* scVI `n_latent == 64` and `gene_likelihood == 'nb'`: REUSE OK.
* Latents.h5ad schema parseable and ≥ 100 000 cells: REUSE OK.
* Centroid `shape == (64,)` and norm plausible: REUSE OK.

**Rollback:** If any of the four fails, set V3A to TRACK-N-ONLY (drop Track L; Phase A1-bg is the only path forward; expect +2 h delay before Phase A2 starts).

---

### Phase A1 — Bootstrap Track L (copy + ε + pairs, ~1 h, FOREGROUND)

**Goal:** Stand up `artifacts_v3/` Track L: VAE artifacts copied, ε_p25 recomputed, OT pairs rebuilt from current `build_pairs.py`.

- [ ] **A1.1 — Create `artifacts_v3/` skeleton**

```bash
mkdir -p artifacts_v3/{vae_n64_legacy,pairs_n64_legacy,dynamics_n64_legacy_ror_corr010,rl_n64_legacy_c2_k3_1M_seed42,reachability_probe_v3a_legacy,eval_v3a_hardness_legacy,figures,interpretation}
```

- [ ] **A1.2 — Copy (NOT symlink) legacy VAE artifacts**

```bash
cp -r artifacts_64/vae/. artifacts_v3/vae_n64_legacy/
# Preserves: model/, latents.h5ad (~1 GB), gene_vocab.json, z_reference_centroid.npy, epsilon_success.json (legacy, p90-only)
ls -lh artifacts_v3/vae_n64_legacy/
```

Expected: full directory tree mirrored, ~1.05 GB total. **Never write back to `artifacts_64/`** (CLAUDE.md §3 sacred rule 4 / V3 rule 4).

- [ ] **A1.3 — Recompute ε quantiles on legacy 64D latents**

```bash
python - <<'PY'
import json, anndata as ad, numpy as np
adata = ad.read_h5ad('artifacts_v3/vae_n64_legacy/latents.h5ad')
ctrl_mask = adata.obs['perturbation'] == 'control'  # adapt column name if A0.2 reveals different schema
Z = adata.obsm['X_scVI'][ctrl_mask.values]
z_ref = np.load('artifacts_v3/vae_n64_legacy/z_reference_centroid.npy')
d = np.linalg.norm(Z - z_ref[None, :], axis=1)
out = {
    'n_control_cells': int(len(d)),
    'p10': float(np.percentile(d, 10)),
    'p25': float(np.percentile(d, 25)),
    'p50': float(np.percentile(d, 50)),
    'p75': float(np.percentile(d, 75)),
    'p90': float(np.percentile(d, 90)),
    'mean': float(d.mean()), 'std': float(d.std()),
}
json.dump(out, open('artifacts_v3/vae_n64_legacy/epsilon_success.json', 'w'), indent=2)
print(out)
PY
```

Expected: non-degenerate distribution with `p25 < p50 < p75 < p90`. V2 32D had `p25 = 3.166`; V3 64D will be larger (scVI distances scale roughly with √n_latent). Record the new `p25` value as `EPS_P25_LEGACY`.

**Acceptance:** `p25` exists, finite, `p10 > 0`, `p90 < 100`. Otherwise abort Track L.

- [ ] **A1.4 — Add V3 path keys to `config/paths.yaml` (additive only)**

Append to `config/paths.yaml` (do NOT replace existing keys):

```yaml
# --- V3A path keys (additive, do not break V1/V2) ---
artifacts_v3_root: ${paths.repo_root}/artifacts_v3

# Track L (legacy 64D reuse)
v3_legacy_vae_dir:        ${paths.artifacts_v3_root}/vae_n64_legacy
v3_legacy_vae_latents:    ${paths.v3_legacy_vae_dir}/latents.h5ad
v3_legacy_vae_centroid:   ${paths.v3_legacy_vae_dir}/z_reference_centroid.npy
v3_legacy_vae_eps:        ${paths.v3_legacy_vae_dir}/epsilon_success.json
v3_legacy_vae_gene_vocab: ${paths.v3_legacy_vae_dir}/gene_vocab.json
v3_legacy_pairs_dir:      ${paths.artifacts_v3_root}/pairs_n64_legacy
v3_legacy_dynamics_dir:   ${paths.artifacts_v3_root}/dynamics_n64_legacy_ror_corr010
v3_legacy_rl_dir:         ${paths.artifacts_v3_root}/rl_n64_legacy_c2_k3_1M_seed42

# Track N (fresh NB 64D)
v3_nb_vae_dir:            ${paths.artifacts_v3_root}/vae_n64_nb
v3_nb_pairs_dir:          ${paths.artifacts_v3_root}/pairs_n64_nb
v3_nb_dynamics_dir:       ${paths.artifacts_v3_root}/dynamics_n64_nb_ror_corr010
v3_nb_rl_dir:             ${paths.artifacts_v3_root}/rl_n64_nb_c2_k3_1M_seed42

# Eval / reachability under v3
v3_eval_dir:              ${paths.artifacts_v3_root}/eval_v3a_hardness
v3_reach_dir:             ${paths.artifacts_v3_root}/reachability_probe_v3a
```

Verify: `python scripts/train_vae.py --cfg job paths.vae_dir=${paths.v3_legacy_vae_dir}` (dry-resolve) prints the expected interpolated path.

- [ ] **A1.5 — Rebuild OT pairs (current `build_pairs.py`) on legacy 64D latents**

```bash
PYTHONPATH=. python scripts/build_pairs.py --config-name default \
    paths.vae_latents_h5ad=${paths.v3_legacy_vae_latents} \
    paths.pairs_dir=${paths.v3_legacy_pairs_dir} \
    pairing.method=ot \
    pairing.ot_epsilon=0.05 \
    seed=42 \
    +dry_run=false
```

Expected outputs: `artifacts_v3/pairs_n64_legacy/{train,val,ood,combo}_pairs.npz` and `metadata.json`. The OOD gene index set in `metadata.json` should match V1's (per A0.1).

**Acceptance:**
* `train_pairs.npz` has `n_pairs ≥ 35 000`.
* `metadata.json::pairing_method == 'ot'`, `seed == 42`.
* OOD gene set matches V1 (or audit explains the diff).

- [ ] **A1.6 — Pairing-noise diagnostic**

```bash
python - <<'PY'
import numpy as np
p = np.load('artifacts_v3/pairs_n64_legacy/train_pairs.npz')
z_ctrl, z_pert = p['z_ctrl'], p['z_pert']
# Per-gene mean delta, residual variance / signal variance
import json
genes = p['gene_idx']
out = []
for g in np.unique(genes):
    mask = genes == g
    if mask.sum() < 5: continue
    d = z_pert[mask] - z_ctrl[mask]
    mu = d.mean(axis=0)
    res = d - mu[None, :]
    noise = float(np.var(res) / max(np.var(d), 1e-9))
    out.append(noise)
print('pairing_noise median:', float(np.median(out)))
print('pairing_noise mean:', float(np.mean(out)))
PY
```

Expected: `pairing_noise_median ≤ 0.85` would be a clean improvement over V2 32D's 0.89; ≥ 0.85 still acceptable, but logged as a finding.

---

### Phase A1-bg — Bootstrap Track N (fresh NB VAE, ~2.5 h, BACKGROUND)

**Kick off in background immediately after A0 verdict (REUSE OK or REUSE WITH CAVEAT).**

- [ ] **A1-bg.1 — Train fresh 64D NB VAE**

```bash
PYTHONPATH=. python scripts/train_vae.py --config-name default \
    vae.n_latent=64 \
    vae.gene_likelihood=nb \
    paths.vae_dir=${paths.v3_nb_vae_dir} \
    seed=42 \
    +force=true \
    +dry_run=false
```

Run in background (via `&` or `Bash run_in_background=true`). Expected wall-clock ~2 h on macOS MPS (~30 min on CUDA). Output: `artifacts_v3/vae_n64_nb/{model/, latents.h5ad, gene_vocab.json, z_reference_centroid.npy, epsilon_success.json}` with full quantiles auto-computed.

- [ ] **A1-bg.2 — Build OT pairs on fresh NB latents**

```bash
PYTHONPATH=. python scripts/build_pairs.py --config-name default \
    paths.vae_latents_h5ad=${paths.v3_nb_vae_dir}/latents.h5ad \
    paths.pairs_dir=${paths.v3_nb_pairs_dir} \
    pairing.method=ot pairing.ot_epsilon=0.05 \
    seed=42 +dry_run=false
```

Expected ~30 min. Verify OOD split matches V1's (must — same `seed=42`, same gene vocab).

---

### Phase A2 — Train V3.1 RoR dynamics on Track L (~30–60 min, FOREGROUND)

**Goal:** Train the V3.1 dynamics candidate on legacy-track pairs with V2 primary architecture (RoR + corr0.10).

- [ ] **A2.1 — Train RoR + corr0.10 dynamics**

```bash
PYTHONPATH=. python scripts/train_dynamics.py --config-name default \
    paths.pairs_train=${paths.v3_legacy_pairs_dir}/train_pairs.npz \
    paths.pairs_val=${paths.v3_legacy_pairs_dir}/val_pairs.npz \
    paths.pairs_combo=${paths.v3_legacy_pairs_dir}/combo_pairs.npz \
    paths.vae_gene_vocab_json=${paths.v3_legacy_vae_gene_vocab} \
    paths.dynamics_dir=${paths.v3_legacy_dynamics_dir} \
    dynamics.use_residual_over_ridge=true \
    dynamics.lambda_corr=0.10 \
    dynamics.use_state_linear_skip=false \
    dynamics.use_gene_delta_bias=false \
    dynamics.selection_metric=gate_margin \
    seed=42 +force=true +dry_run=false
```

Expected outputs:
* `model.pt`, `model_best_nll.pt`, `model_best_gate.pt`
* `config.json` (with `n_latent=64`, `n_genes=105`, RoR fields)
* `gate.json` with passed/failed and full metrics
* `val_metrics.json`, `ood_metrics.json`, `ridge_baseline.npz`, `checkpoint_comparison.json`

- [ ] **A2.2 — Inspect dynamics gate result**

```bash
python -c "
import json
g = json.load(open('artifacts_v3/dynamics_n64_legacy_ror_corr010/gate.json'))
v = json.load(open('artifacts_v3/dynamics_n64_legacy_ror_corr010/val_metrics.json'))
o = json.load(open('artifacts_v3/dynamics_n64_legacy_ror_corr010/ood_metrics.json'))
print('gate passed:', g['passed'])
print('val Pearson:', v.get('pearson'), 'ridge margin:', v.get('margin_vs_linear_ridge_pearson'))
print('ood Pearson:', o.get('pearson'), 'ood margin:', o.get('margin_vs_linear_ridge_pearson'))
"
```

**Acceptance:**
* `gate.passed == true` is **not required** (V2 RoR also failed by OT pairing-noise ceiling).
* `val_metrics.pearson ≥ 0.55` (V2 RoR 32D: 0.615; expect close to or above).
* `val_metrics.margin_vs_linear_ridge_pearson > +0.0136` is a STRETCH GOAL (V2 RoR 32D).
* **HARD GATE**: `ood_metrics.pearson ≥ 0.40`. Legacy non-RoR 64D was 0.369; RoR is expected to lift this.

**Rollback if OOD Pearson < 0.40**:
* (a) Wait for Track N (A2-bg) — if Track N OOD Pearson ≥ 0.40, legacy VAE config was the bottleneck; proceed Track-N-only.
* (b) If Track N also fails OOD threshold, V3.1 hypothesis is in trouble — proceed to A8 interpretation with this signal, pivot to V3.3 (ZINB) or V3.4 (SCANVI) per V3 stub §5.

---

### Phase A2-bg — Train V3.1 RoR dynamics on Track N (~30–60 min, BACKGROUND after A1-bg done)

- [ ] **A2-bg.1 — Train RoR + corr0.10 on fresh NB pairs**

```bash
PYTHONPATH=. python scripts/train_dynamics.py --config-name default \
    paths.pairs_train=${paths.v3_nb_pairs_dir}/train_pairs.npz \
    paths.pairs_val=${paths.v3_nb_pairs_dir}/val_pairs.npz \
    paths.pairs_combo=${paths.v3_nb_pairs_dir}/combo_pairs.npz \
    paths.vae_gene_vocab_json=${paths.v3_nb_vae_dir}/gene_vocab.json \
    paths.dynamics_dir=${paths.v3_nb_dynamics_dir} \
    dynamics.use_residual_over_ridge=true dynamics.lambda_corr=0.10 \
    dynamics.use_state_linear_skip=false dynamics.use_gene_delta_bias=false \
    dynamics.selection_metric=gate_margin \
    seed=42 +force=true +dry_run=false
```

Same gate inspection (A2.2) on Track N output. Compare side-by-side with Track L.

---

### Phase A3 — Reachability + greedy oracle saturation (~30 min × 2 tracks, FOREGROUND for Track L)

**Goal:** Determine whether the V3.1 64D field is reachable (controllable) and whether `greedy_dyn_2` saturates at the primary cell. This is the **single most critical V3 signal** — if greedy_dyn_2 still hits 1.000 at K=3 primary, the V3 latent-dim hypothesis is rejected before PPO is even trained.

- [ ] **A3.1 — Beam reachability probe (Track L, K=3 primary + K=2 frontier)**

```bash
PYTHONPATH=. python scripts/probe_reachability.py \
    --dynamics-path artifacts_v3/dynamics_n64_legacy_ror_corr010/model.pt \
    --vae-dir artifacts_v3/vae_n64_legacy \
    --max-depth 3 --beam-width 50 \
    --distance-bin 8-10 \
    --held-out-genes-only \
    --repeat-mask \
    --out-dir artifacts_v3/reachability_probe_v3a/legacy_k3_bin810_ood
```

```bash
PYTHONPATH=. python scripts/probe_reachability.py \
    --dynamics-path artifacts_v3/dynamics_n64_legacy_ror_corr010/model.pt \
    --vae-dir artifacts_v3/vae_n64_legacy \
    --max-depth 2 --beam-width 50 \
    --distance-bin 6-8 \
    --held-out-genes-only \
    --repeat-mask \
    --out-dir artifacts_v3/reachability_probe_v3a/legacy_k2_bin68_ood
```

**Acceptance:** beam reach (success at depth ≤ k) ≥ 50 % at K=3/bin 8–10/OOD. V2 RoR 32D was 17/17 (100 %).

**Rollback:** If reach < 50 % at K=3, the V3 field is uncontrollable. Either:
* Track N has solved it (Track L was VAE-bottlenecked) — proceed Track-N-only.
* Both tracks fail — V3.1 dead. Skip to A8 with negative result, pivot to V3.3 or contraction-regulariser fallback (V3 stub §5).

- [ ] **A3.2 — Greedy oracle eval at K=3 primary + K=2 frontier (Track L)**

The hard-bench evaluator already runs `greedy_dyn_1`, `greedy_dyn_2`, `greedy_dyn_3` as built-in baselines. We invoke it *without* a PPO checkpoint (use `--baselines-only` if the script supports it, otherwise pass a placeholder and ignore PPO rows):

```bash
PYTHONPATH=. python scripts/evaluate_rl_hard.py \
    --dynamics-model-path artifacts_v3/dynamics_n64_legacy_ror_corr010/model.pt \
    --vae-dir artifacts_v3/vae_n64_legacy \
    --output-dir artifacts_v3/eval_v3a_hardness/legacy_greedy_only \
    --distance-bins 6-8 8-10 \
    --k-values 1 2 3 8 \
    --epsilon-override $EPS_P25_LEGACY \
    --n-episodes 300 \
    --held-out-genes-only \
    --baselines-only
```

(If `--baselines-only` is not currently supported, A3.2 becomes an A5 sub-task — defer the saturation check to A5 and rely on A3.1 reachability as the standalone go/no-go for V3 hypothesis support.)

**Critical V3 signal — primary cell saturation:**

| Outcome | Verdict |
|---|---|
| `greedy_dyn_2 < 0.95` at K=3/bin 8–10 | **V3 hypothesis is alive** (less-saturated field); proceed to A4 |
| `greedy_dyn_2 ≥ 0.95` at K=3/bin 8–10 AND `greedy_dyn_2 ≥ 0.95` at K=2/bin 6–8 | **V3.1 latent hypothesis rejected on Track L**; defer to Track N. If Track N also saturates, A8 pivots to V3.3 (ZINB) / V3.4 (SCANVI) per V3 stub §5. |

- [ ] **A3-bg — Repeat A3 on Track N (after A2-bg done)**

Same probes and greedy eval, with paths swapped to `vae_n64_nb`, `dynamics_n64_nb_ror_corr010`, and `EPS_P25_NB` (computed by `train_vae.py` automatically).

---

### Phase A4 — Train V3.1 C2-style PPO on Track L (seed 42 only, ~10 min, FOREGROUND)

**Goal:** Train the V3.1 primary PPO candidate matching V2's C2 protocol exactly.

- [ ] **A4.1 — Train PPO (C2-style: RoR dynamics + terminal_only_step_cost + curriculum + K=3 + 1M timesteps)**

```bash
PYTHONPATH=. python scripts/train_rl.py --config-name default \
    paths.dynamics_dir=${paths.v3_legacy_dynamics_dir} \
    paths.vae_dir=${paths.v3_legacy_vae_dir} \
    paths.rl_dir=${paths.v3_legacy_rl_dir} \
    rl.ppo.total_timesteps=1_000_000 \
    rl.env.max_steps=3 \
    rl.env.epsilon_override=$EPS_P25_LEGACY \
    rl.reward.mode=terminal_only_step_cost \
    rl.reward.beta_step_cost=0.05 \
    rl.train.curriculum.enabled=true \
    rl.train.curriculum.start_d=4.0 \
    rl.train.curriculum.end_d=10.0 \
    rl.train.curriculum.end_fraction=0.7 \
    rl.train.skip_gate=true \
    seed=42 +dry_run=false
```

Expected outputs: `ppo.zip`, `rollouts.parquet`, `success_curves.png`, `action_freq.json`, `metadata.json`.

**Logging requirement (CLAUDE.md §3 sacred rule 9 / V3 stub §3):**
Add to `PROGRESS.md`:
> **V3A Track L PPO trained with `rl.train.skip_gate=true`**: rationale — V3.1 dynamics likely fails the supervised gate by the OT pairing-noise ceiling (same mechanism as V2 primary), but beam reach at K=3/bin 8–10/OOD ≥ 50 % verifies controllability. Beam reach value: `<measured from A3.1>`.

**Acceptance:** PPO converges (success_curves.png shows monotone success climb in last 30 % of training). If success rate is < 0.10 after 1M steps, training collapsed — investigate before A5.

---

### Phase A4-bg — Train V3.1 C2-style PPO on Track N (seed 42, ~10 min, BACKGROUND after A2-bg)

Same command as A4.1 with paths swapped to Track N and `EPS_P25_NB` substituted for `EPS_P25_LEGACY`.

---

### Phase A5 — Hard-bench evaluation, full K-sweep (~40 min × 2 tracks)

**Goal:** Evaluate PPO + all baselines at the V2-equivalent hard-bench cells AND across the full K-sweep. This is the only headline reportable in V3A.

- [ ] **A5.1 — Hard-bench eval Track L, full K-sweep**

```bash
PYTHONPATH=. python scripts/evaluate_rl_hard.py \
    --ppo-model-path artifacts_v3/rl_n64_legacy_c2_k3_1M_seed42/ppo.zip \
    --dynamics-model-path artifacts_v3/dynamics_n64_legacy_ror_corr010/model.pt \
    --vae-dir artifacts_v3/vae_n64_legacy \
    --output-dir artifacts_v3/eval_v3a_hardness/legacy_k_sweep_seed42 \
    --distance-bins 6-8 8-10 \
    --k-values 1 2 3 8 \
    --epsilon-override $EPS_P25_LEGACY \
    --n-episodes 300 \
    --held-out-genes-only \
    --include-stochastic
```

Cells produced: `{K=1, K=2, K=3, K=8} × {bin 6-8, bin 8-10} × {OOD} = 8 cells`. Each cell reports: random, greedy_dyn_1, greedy_dyn_2, greedy_dyn_3, PPO_det, PPO_stoch.

Per-cell metrics required in the resulting summary table:
* PPO success rate (deterministic) with Wilson 95 % pooled CI.
* PPO − random, PPO − greedy_dyn_2 deltas.
* All baseline success rates with Wilson CIs.

- [ ] **A5.2 — Cross-dynamics transfer check (Track L PPO → V2 dynamics; V2 PPO → Track L dynamics)**

```bash
# Track L PPO evaluated on V2 32D RoR dynamics
PYTHONPATH=. python scripts/evaluate_rl_hard.py \
    --ppo-model-path artifacts_v3/rl_n64_legacy_c2_k3_1M_seed42/ppo.zip \
    --dynamics-model-path artifacts_v2/dynamics_v1ot_ror_corr010/model.pt \
    --vae-dir artifacts/vae \
    --output-dir artifacts_v3/eval_v3a_hardness/transfer_v3legacy_on_v2dyn \
    --hard-bench-only

# V2 primary PPO evaluated on V3 Track L dynamics
PYTHONPATH=. python scripts/evaluate_rl_hard.py \
    --ppo-model-path artifacts_v2/rl_v1ot_ror_corr010_terminal_curric_k3_1M/ppo.zip \
    --dynamics-model-path artifacts_v3/dynamics_n64_legacy_ror_corr010/model.pt \
    --vae-dir artifacts_v3/vae_n64_legacy \
    --output-dir artifacts_v3/eval_v3a_hardness/transfer_v2_on_v3legacy_dyn \
    --hard-bench-only
```

**Caveat:** cross-dimensionality transfer (32D ↔ 64D) likely uses different observation shapes; if the env / policy embed-layer is dim-rigid, both transfer calls will error. Expected outcome: both fail cleanly, confirming that V3 PPO is dimensionality-bound. If they succeed (unlikely without padding), report the transfer numbers honestly.

- [ ] **A5-bg — Repeat A5.1 on Track N (after A4-bg done)**

Same command with Track N paths.

---

### Phase A6 — Cross-track comparison + escalation decision (~30 min)

**Goal:** Decide whether to escalate to 4 seeds, pivot to V3.3, or declare V3.1 success/failure.

- [ ] **A6.1 — Build the V3A hardness frontier table**

For each track (L, N), produce a Markdown table with columns: K, bin, PPO_succ, PPO − random, PPO − grd2, grd1, grd2, grd3, random. Save to `artifacts_v3/eval_v3a_hardness/{legacy,nb}_summary.md`.

- [ ] **A6.2 — Cross-track comparison**

```python
# pseudocode for analysis script artifacts_v3/scripts/compare_tracks.py (to be added in A6)
for cell in cells:
    L = legacy[cell]; N = nb[cell]
    delta_track = N['PPO_succ'] - L['PPO_succ']
    if abs(delta_track) > 0.05:
        print(f'{cell}: Track diff = {delta_track:+.3f} — VAE config matters')
```

- [ ] **A6.3 — Escalation decision per track**

For each track (L, N) independently:

| Track A6 signal | Decision |
|---|---|
| `PPO − grd2 ≥ +0.05 pp` at ≥ 1 cell | **HEADLINE PASS** → escalate to 4 seeds (A7 on that track) |
| `PPO − grd2 ≥ +0.03 pp` at ≥ 1 cell (directional only) | escalate to 4 seeds (A7 on that track) to confirm with seed CIs |
| `PPO − grd2 < +0.03 pp` everywhere AND `greedy_dyn_2 ≥ 0.95` at primary | **HEADLINE FAIL for this track** — VAE didn't unsaturate the field. Track pivots to V3.3 (ZINB) per stub. |
| `PPO − grd2 < +0.03 pp` everywhere AND `greedy_dyn_2 < 0.95` at primary | **AMBIGUOUS** — field is less saturated but PPO didn't exploit it; A7 still useful to rule out seed noise; if 4-seed confirms negative, look at PPO training dynamics in A8. |

---

### Phase A7 — Conditional 4-seed escalation (triggered by A6.3, ~30 min)

**Run only if A6.3 triggers escalation on Track L and/or Track N.**

- [ ] **A7.1 — Train seeds {0, 1, 7} on the qualifying track(s)**

```bash
for SEED in 0 1 7; do
  PYTHONPATH=. python scripts/train_rl.py --config-name default \
      paths.dynamics_dir=${paths.v3_legacy_dynamics_dir} \
      paths.vae_dir=${paths.v3_legacy_vae_dir} \
      paths.rl_dir=artifacts_v3/rl_n64_legacy_c2_k3_1M_seed$SEED \
      rl.ppo.total_timesteps=1_000_000 \
      rl.env.max_steps=3 \
      rl.env.epsilon_override=$EPS_P25_LEGACY \
      rl.reward.mode=terminal_only_step_cost \
      rl.train.curriculum.enabled=true \
      rl.train.skip_gate=true \
      seed=$SEED +dry_run=false
done
```

- [ ] **A7.2 — Hard-bench eval per seed**

Repeat A5.1 for each new seed, output to `artifacts_v3/eval_v3a_hardness/legacy_k_sweep_seed{0,1,7}`.

- [ ] **A7.3 — Aggregate 4-seed CIs**

Reuse `scripts/aggregate_v2_seeds.py` (it is dataset/path agnostic via CLI args) on the four seed eval dirs. Produce `artifacts_v3/eval_v3a_hardness/legacy_seed_aggregate/seed_aggregate_success_rate.{json,md}` matching V2 format.

**Acceptance:**
* 95 % normal CI of `PPO − grd2` at the qualifying cell **excludes zero** (i.e., CI low > 0.00 or up < 0.00).
* PPO mean − grd2 mean ≥ +0.05 pp (HEADLINE) or ≥ +0.03 pp (DIRECTIONAL).

---

### Phase A8 — Interpret + write V3A final report (~1–2 h)

**Goal:** Write `artifacts_v3/interpretation_v3a.md`, upgrade `V3_RESEARCH_PLAN.md` from stub to full plan, log session to `PROGRESS.md`.

- [ ] **A8.1 — Write `artifacts_v3/interpretation_v3a.md`**

Use the template below.

```markdown
# V3A — Latent Audit & 64D Pivot — Interpretation

**Date:** YYYY-MM-DD
**Tracks completed:** {L, N}
**Seeds:** {42}{, 0, 1, 7 if A7 ran}
**V3A status:** {HEADLINE PASS / DIRECTIONAL PASS / FAIL}

## 1. Audit findings (A0)
- Gene vocab alignment: {OK / DIFF}
- scVI config match (legacy → expected NB / 64 / normal): {OK / DRIFT, specifically: ...}
- Centroid sanity: {OK / DEGENERATE}
- ε_p25_legacy = {value}; ε_p25_nb = {value}

## 2. Pairing-noise diagnostic (A1.6, A1-bg.2)
| Track | n_pairs (train) | pairing_noise median | pairing_noise mean |
|---|---|---|---|
| Track L | {n} | {v} | {v} |
| Track N | {n} | {v} | {v} |
| V2 32D (baseline) | 38958 | 0.89 | — |

## 3. Dynamics metrics (A2, A2-bg)
| Track | val Pearson | val ridge margin | OOD Pearson | gate passed | uncertainty Spearman |
|---|---|---|---|---|---|
| Track L | {v} | {v} | {v} | {bool} | {v} |
| Track N | {v} | {v} | {v} | {bool} | {v} |
| V2 RoR 32D (baseline) | 0.615 | +0.0136 | 0.516 | false | 0.245 |

## 4. Reachability + greedy saturation (A3, A3-bg)
| Track | K=3/bin 8-10 beam reach | greedy_dyn_2 @ primary | greedy_dyn_2 @ K=2/bin 6-8 |
|---|---|---|---|
| Track L | {x/N} | {v} | {v} |
| Track N | {x/N} | {v} | {v} |
| V2 RoR 32D (baseline) | 17/17 | 1.000 | 0.790 |

**V3 hypothesis support check**: greedy_dyn_2 at K=3/bin 8-10/OOD **{drops below 0.95 / does not drop below 0.95}** on **{neither, Track L only, Track N only, both tracks}**.

## 5. PPO hardness frontier (A5, A5-bg, A7 if run)
Headline cell (K=3, bin 8-10, OOD):
| Config | success | PPO − random | PPO − grd2 | 95 % CI on PPO − grd2 |
|---|---|---|---|---|
| V2 primary RoR_corr010 × C2 (32D) | 0.941 ± 0.048 | +0.771 | −0.059 | [−0.106, −0.012] |
| V3.1 Track L (64D legacy + RoR + C2) | {v} | {v} | {v} | {v} |
| V3.1 Track N (64D NB fresh + RoR + C2) | {v} | {v} | {v} | {v} |

Full K-sweep (K ∈ {1, 2, 3, 8} × bin ∈ {6-8, 8-10}): see `eval_v3a_hardness/{legacy,nb}_summary.md`.

## 6. Cross-track delta
Largest `Track N − Track L` PPO-success delta = {v} at cell {v}. If |delta| > 0.05 pp, **VAE config materially affects V3 outcome**.

## 7. Honest claims (V3A passes only IF a claim in this section is supportable)
- ✅ V3A HEADLINE PASS: "Under the V3A 64D latent on {track}, MaskablePPO with RoR_corr010 + terminal_only_step_cost + curriculum exceeds greedy_dyn_2 by {Δ} pp at the {K, bin, OOD} cell, with 95 % CI excluding zero across 4 seeds."
- ✅ V3A DIRECTIONAL PASS: same but +0.03 pp ≤ Δ < +0.05 pp; reported as suggestive, not headline.
- ✅ V3A clean negative: "The 64D latent ablation rejects the H_V3_latent hypothesis at the single-track-{L,N} level. greedy_dyn_2 still saturates / PPO does not exceed it."

## 8. Claims V3A must NOT make
- ❌ "PPO discovers superior strategies" (require PPO − grd2 ≥ +0.05 pp with seed CI excluding zero).
- ❌ "64D solves the gate" (gate likely still fails per OT pairing-noise ceiling — same as V2).
- ❌ "PPO generalizes across dynamics" (A5.2 transfer check is expected negative).
- ❌ Any biological claim without Chronos / GSEA test (same protocol as V2 §6).

## 9. Decision for V3B
- If V3A HEADLINE PASS: V3B = scale up (more seeds, write paper draft, optional B5 ablation on the winning track).
- If V3A DIRECTIONAL PASS: V3B = confirm with full architectural ablation (B5 vs C2, λ_corr sweep) on the winning track.
- If V3A FAIL: V3B = V3.3 ZINB (per stub §4) or V3.4 SCANVI; if both fail, V3.fallback (ensemble / contraction regulariser per stub §5).
```

- [ ] **A8.2 — Promote `V3_RESEARCH_PLAN.md` from stub to full plan**

Update `V3_RESEARCH_PLAN.md` to include:
* §1–§3 of stub unchanged.
* §4 matrix expanded with V3A measured numbers.
* §5 stop conditions / pivot agenda confirmed or modified based on V3A outcome.
* New §12 — V3A summary (link to `artifacts_v3/interpretation_v3a.md`).

- [ ] **A8.3 — Log to `PROGRESS.md`**

Append session entry:
```markdown
## Session YYYY-MM-DD-HHMM  (agent: V3 research lead)

**Phase:** V3A — Latent audit and 64D pivot
**Status:** {HEADLINE PASS / DIRECTIONAL PASS / FAIL}
**Metrics:** see artifacts_v3/interpretation_v3a.md §5 table
**Skip-gate log:** Track L and Track N PPOs trained with rl.train.skip_gate=true. Justification: beam reach {x/17} ≥ 50 %; V3 dynamics likely fail OT pairing-noise gate ceiling (same mechanism as V2 primary).
**Blockers:** {none / P1 / P0 — describe}
**Next:** (1) V3B scope decided per A8 §9; (2) seed escalation if directional; (3) Chronos / GSEA defer to post-headline.
```

- [ ] **A8.4 — Commit and produce figures**

```bash
git add config/paths.yaml V3_RESEARCH_PLAN.md PROGRESS.md V3A_LATENT_AUDIT_AND_64D_PLAN.md artifacts_v3/interpretation_v3a.md
git commit -m "V3A: 64D latent audit + parallel reuse/fresh tracks; {pass/fail headline}"
```

Figures (reuse `scripts/make_v2_figures.py` with V3 paths if compatible; otherwise add `scripts/make_v3a_figures.py`):
* `artifacts_v3/figures/v3a_hardness_frontier.png` (PPO − grd2 scatter per cell, both tracks)
* `artifacts_v3/figures/v3a_dynamics_taxonomy.png` (val/OOD Pearson per track vs V2 baseline)
* `artifacts_v3/figures/v3a_seed_variance.png` (only if A7 ran)

---

## 5. Anti-trap rules (mandatory; supersede default reward-shaping reflexes)

The following are guards against the specific traps that V2 fell into. They apply to every V3A step.

1. **Never optimize only the dynamics gate.** Every step that touches dynamics also reports reachability (A3) and greedy_dyn_2 saturation (A3 / A5). Gate margin alone is **insufficient** signal.
2. **Never declare V3A pass on the saturated primary cell alone.** Headline PASS requires either (a) primary cell with grd2 < 0.95 *and* PPO − grd2 ≥ +0.05 pp, or (b) K=2 frontier cell with PPO − grd2 ≥ +0.05 pp. Either path acceptable, but both criteria reported explicitly.
3. **Always include `random, grd1, grd2, grd3` baselines at every cell.** A "weak PPO with bad greedy" is more informative than "strong PPO with saturated greedy". Do not discard low-PPO results without examining the baselines.
4. **Report PPO − random and PPO − greedy as separate columns.** Conflating them is the V2 trap (PPO − random = +77 pp looked great until PPO − grd2 = −0.06 pp emerged).
5. **`rl.train.skip_gate=true` is logged with rationale every time.** Sacred rule (CLAUDE.md §3.9 / V3 stub §3). The rationale references the beam-reach number from A3.
6. **`artifacts_64/`, `artifacts_v2/`, `artifacts/`, `artifacts/rl_sweeps/` are read-only.** All V3A writes go to `artifacts_v3/`. Frozen-tier sacred rule (V3 stub §3).
7. **No biological claims without Chronos and GSEA tests.** V2's Chronos correlation is null (ρ = −0.024, p = 0.815); V3 must rerun the test on whichever track wins, and report it honestly. Defer to V3B; never include biological claims in the V3A headline.
8. **No path hardcoding.** Use `${paths.v3_legacy_*}` / `${paths.v3_nb_*}` keys; never inline literal `artifacts_v3/...` strings in code. Sacred rule (CLAUDE.md §3.3).
9. **No new metrics inline.** Any new metric (e.g. "track delta", "saturation index") is added to `src/analysis/metrics.py` with a docstring containing its mathematical definition. Sacred rule (CLAUDE.md §3.4).
10. **Do NOT proceed to A4 if A3 fails reachability < 50 % on both tracks.** PPO on an uncontrollable field will look like NOOP collapse (V2 mean_delta failure mode); we already have that signal; do not burn 1M timesteps to re-discover it.

---

## 6. Files modified or created

### Created (V3A)

| Path | Purpose |
|---|---|
| `/Users/gabo/Developer/ITAM/IA/cellpath/V3A_LATENT_AUDIT_AND_64D_PLAN.md` | This plan, copied from `/Users/gabo/.claude/plans/...` |
| `artifacts_v3/audit_v3a.md` | A0 audit verdict |
| `artifacts_v3/vae_n64_legacy/` | Copy of `artifacts_64/vae/`; recomputed `epsilon_success.json` |
| `artifacts_v3/vae_n64_nb/` | Fresh 64D NB VAE |
| `artifacts_v3/pairs_n64_legacy/` | Fresh OT pairs (current code) on legacy 64D latents |
| `artifacts_v3/pairs_n64_nb/` | Fresh OT pairs on fresh 64D latents |
| `artifacts_v3/dynamics_n64_legacy_ror_corr010/` | RoR + corr0.10 on legacy pairs |
| `artifacts_v3/dynamics_n64_nb_ror_corr010/` | RoR + corr0.10 on fresh pairs |
| `artifacts_v3/rl_n64_legacy_c2_k3_1M_seed42/` | V3.1 PPO Track L seed 42 |
| `artifacts_v3/rl_n64_nb_c2_k3_1M_seed42/` | V3.1 PPO Track N seed 42 |
| `artifacts_v3/rl_n64_{legacy,nb}_c2_k3_1M_seed{0,1,7}/` | conditional, A7 escalation |
| `artifacts_v3/reachability_probe_v3a/{legacy,nb}_*/` | beam-search reach reports |
| `artifacts_v3/eval_v3a_hardness/{legacy,nb}_k_sweep_seed*/` | hard-bench eval outputs |
| `artifacts_v3/eval_v3a_hardness/{legacy,nb}_seed_aggregate/` | 4-seed aggregations (A7) |
| `artifacts_v3/eval_v3a_hardness/transfer_*` | cross-dynamics transfer (A5.2) |
| `artifacts_v3/interpretation_v3a.md` | A8.1 final report |
| `artifacts_v3/figures/v3a_*.png` | A8.4 figures |

### Modified (V3A)

| Path | Change |
|---|---|
| `config/paths.yaml` | Append V3A path keys (additive; A1.4) |
| `V3_RESEARCH_PLAN.md` | Stub → full plan (A8.2) |
| `PROGRESS.md` | New session entry (A8.3) |

### Frozen — must NOT modify (sacred rule)

* `artifacts/`
* `artifacts_64/`
* `artifacts_v2/`
* `artifacts/rl_sweeps/`
* `config/dynamics.yaml::gate.*` thresholds (gate logic locked per V3 stub §3)
* `src/analysis/metrics.py` (additive only; never edit existing metric definitions)

---

## 7. Stop conditions

V3A halts immediately and proceeds to A8 interpretation with the current data if **any** of the following:

| Trigger | Action |
|---|---|
| A0 reveals gene-vocab mismatch | Track L dropped; Track N becomes critical path (delays Phase A2 by ~2 h) |
| A2 / A2-bg: OOD Pearson < 0.40 on both tracks | A8 negative result; pivot to V3.3 (ZINB) per V3 stub §5 |
| A3 / A3-bg: beam reach < 50 % at K=3/bin 8-10/OOD on both tracks | Skip A4 entirely; A8 with controllability-loss diagnosis; pivot to V3.4 (SCANVI) or contraction regulariser |
| A3 / A3-bg: greedy_dyn_2 ≥ 0.95 at primary on both tracks AND greedy_dyn_2 ≥ 0.95 at K=2 frontier on both | A8 with hypothesis-rejected verdict; pivot to V3.3 / V3.4 |
| PPO training NaN / crash at A4 / A4-bg | Investigate; do NOT re-run with `--no-verify` style bypasses; root-cause first (CLAUDE.md general guidance) |
| Compute budget exhausted (> 12 h wall-clock) | Save state, A8 with partial results, defer A7 escalation to next session |

V3A succeeds (HEADLINE PASS) if:

* PPO success − greedy_dyn_2 ≥ +0.05 pp at **at least one** cell in K ∈ {1, 2, 3, 8} × bin ∈ {6-8, 8-10} × OOD on **either** track,
* AND 4-seed CI on that delta excludes zero (after A7 escalation),
* AND beam reach ≥ 50 % at K=3/bin 8-10/OOD on the winning track (controllability confirmed).

V3A succeeds DIRECTIONALLY (worth V3B follow-up) if:

* Same as HEADLINE but with +0.03 pp ≤ Δ < +0.05 pp, OR
* HEADLINE Δ but only 1 seed (A7 not yet run; A7 is then the V3B first action).

V3A fails cleanly (publishable negative result) if:

* Stop-condition trigger (above) AND no PASS criterion met AND both tracks consistent.

---

## 8. Compute budget and wall-clock

| Phase | Track L (foreground) | Track N (background) | Cumulative wall-clock |
|---|---|---|---|
| A0 audit | 30 min | — | 0:30 |
| A1 / A1-bg | 1 h (L) | 2.5 h (N) | 1:30 / 3:00 |
| A2 / A2-bg | 30–60 min (L) | 30–60 min (N, after A1-bg) | 2:30 / 4:00 |
| A3 / A3-bg | 30 min (L) | 30 min (N) | 3:00 / 4:30 |
| A4 / A4-bg | 10 min (L) | 10 min (N) | 3:10 / 4:40 |
| A5 / A5-bg | 40 min (L) | 40 min (N) | 3:50 / 5:20 |
| A6 | 30 min | — | 4:20 |
| A7 (conditional) | 30 min × seeds × cells | — | + up to 2 h |
| A8 interpret | 1–2 h | — | 5:20–7:20 (no A7) / 7:20–9:20 (A7) |

CPU-only (no GPU) on macOS Apple Silicon MPS: VAE training is the long-pole (~2 h). PPO training is ~5–10 min per run at 1M timesteps (max_steps=3, n_envs=4). On CUDA: VAE ~30 min; PPO 2–5 min.

---

## 9. Verification (end-to-end check before A8 commit)

- [ ] **V.1 — All `artifacts_v3/` writes are within `artifacts_v3/`** (`find artifacts_v3 -newer artifacts_v3/audit_v3a.md` returns only V3A-written files).
- [ ] **V.2 — `artifacts_64/`, `artifacts_v2/`, `artifacts/`, `artifacts/rl_sweeps/` are unchanged** (`git status` shows none of those directories modified; if any change appears in the working tree, abort and investigate).
- [ ] **V.3 — `config/paths.yaml` additions resolve correctly**: `python scripts/train_vae.py --cfg job paths.vae_dir=${paths.v3_legacy_vae_dir}` prints the interpolated path without raising InterpolationError.
- [ ] **V.4 — PROGRESS.md and V3_RESEARCH_PLAN.md updated** with V3A session and full plan (not stub).
- [ ] **V.5 — interpretation_v3a.md present and contains every required table cell**.
- [ ] **V.6 — At least one figure under `artifacts_v3/figures/`**.
- [ ] **V.7 — Headline verdict is supported by the data**: search interpretation_v3a.md §5 for the claimed PPO − grd2 number; verify it matches `eval_v3a_hardness/{legacy,nb}_summary.md`.

---

## 10. Final answers to the planning brief

### 10.1 What exists in `artifacts_64/`?

**VAE**: complete (model + 1 GB latents + 105-gene vocab + 64-d centroid + p90-only epsilon). No Hydra snapshot; scVI hyperparameters implicit but likely default (`nb`, `dispersion: gene`, `normal` prior). **Reusable after audit.**

**Pairs**: complete (38 958 train + 4 324 val + 14 549 OOD + 35 995 combo; OT seed=42 ε=0.05; clean gene split). **Will be rebuilt under V3 paths** to ensure consistency with current `build_pairs.py` (V2 P0B′ / P0B″ corrections).

**Dynamics**: 127-epoch `state_linear_skip` model; gate **failed** (margin_vs_linear_ridge_pearson = −0.019); OOD Pearson 0.369 (below V3 threshold 0.40). **NOT reusable** — wrong architecture (V2 primary is RoR). 5 sibling variants in `dynamics_variants/` all show similar gate failure.

**Contraction diagnostics**: present; informative but based on the gate-failing dynamics. **Reference only.**

**RL / eval**: **absent**. Clean slate for V3A.RL.

### 10.2 Reuse / copy / retrain the 64D VAE?

**COPY** (not symlink) of `artifacts_64/vae/` → `artifacts_v3/vae_n64_legacy/` as **Track L**. The copy preserves the frozen-tier rule (no writes to `artifacts_64/`). Recompute ε_p25 onto the copy.

In **parallel**, train a fresh 64D NB VAE under `artifacts_v3/vae_n64_nb/` as **Track N** — Bayesian hedge against unverified legacy scVI config drift. The cross-track comparison in A6 directly tells us whether the legacy VAE was the bottleneck.

### 10.3 Smallest decisive V3A experiment

**Option C with parallel Track N** (Track L = reuse legacy VAE, rebuild pairs/dyn/RL under V3 paths; Track N = fresh NB VAE + same downstream). Single seed (42) first; escalate to 4 seeds (Phase A7) only on directional signal. Full K-sweep eval at K ∈ {1, 2, 3, 8} × bin ∈ {6-8, 8-10} × OOD.

### 10.4 Pipeline summary

A0 audit → A1 / A1-bg bootstrap → A2 / A2-bg RoR dynamics → A3 / A3-bg reachability + greedy → A4 / A4-bg PPO (seed 42) → A5 / A5-bg hard-bench → A6 cross-track compare → A7 (conditional) 4-seed escalate → A8 interpret + figures + PROGRESS.

### 10.5 Success / failure

Headline pass: PPO − grd2 ≥ +0.05 pp at any cell with 4-seed CI excluding zero AND beam reach ≥ 50 %.
Directional pass: same with Δ ∈ [+0.03, +0.05] pp.
Fail (publishable negative): greedy_dyn_2 still saturates AND PPO − grd2 < +0.03 pp everywhere on both tracks → pivot to V3.3 (ZINB) per stub.

### 10.6 Anti-trap

See §5. Most important: always include greedy_dyn_2 and report PPO − grd2 separately from PPO − random; never declare PASS on the saturated primary cell alone.

### 10.7 Files modified / created

See §6.

### 10.8 Execution plan

See §4 (Phase A0 through A8 with exact commands, expected outputs, acceptance criteria, and rollback paths).

---

## 11. Ranked recommendation, reuse decision, first prompt, stop condition

**1. Ranked recommendation for V3 start**:
1. **Parallel Track L (reuse legacy 64D VAE) + Track N (fresh 64D NB VAE)** — Bayesian hedge; this plan. Recommended.
2. Sequential Track L only (Option C from prompt) — faster but no insurance.
3. Fresh-only Track N (Option B from prompt) — clean but discards the legacy artifact.
4. Audit-only (Option A from prompt) — rejected; cannot test V3 hypothesis.
5. Skip-audit jump to RoR+PPO (Option D from prompt) — rejected; gene-split contamination risk.

**2. Reuse vs retrain the 64D VAE**: **COPY (not symlink) artifacts_64/vae/ into artifacts_v3/vae_n64_legacy/ as Track L; ALSO train fresh 64D NB VAE as Track N in parallel.** Recompute ε_p25 on both. Never write to `artifacts_64/`.

**3. First implementation prompt** (Phase A0 — paste this into a fresh CC session after exiting plan mode):

```
You are executing Phase A0 of V3A_LATENT_AUDIT_AND_64D_PLAN.md. Your goal is to audit the legacy 64D VAE artifacts at /Users/gabo/Developer/ITAM/IA/cellpath/artifacts_64/ for V3 reuse readiness. Read-only — do NOT modify artifacts_64/.

Read first: V3A_LATENT_AUDIT_AND_64D_PLAN.md §2 (audit findings) and §4 Phase A0 (commands).

Execute A0.1, A0.2, A0.3, A0.4 in sequence using the exact Python inspections in the plan. For each, capture stdout. After A0.4, write artifacts_v3/audit_v3a.md (create artifacts_v3/ first via `mkdir -p artifacts_v3`) with:
  - Verdict: REUSE OK / REUSE WITH CAVEAT / RETRAIN ONLY
  - All four captured outputs (gene-vocab diff, latents schema, scVI hparams, centroid)
  - Recommended next phase (A1 if REUSE OK; A1-bg only if RETRAIN ONLY)

Halt after writing audit_v3a.md. Report the verdict in your response. Do not proceed to A1 without explicit user approval.

Reproducibility: PYTHONPATH=. for all python -c commands; do not install or upgrade packages; the existing .venv is the canonical environment (CLAUDE.md §6).
```

**4. V3A stop condition**:

* **HARD STOP** if both tracks fail OOD Pearson ≥ 0.40 at A2/A2-bg → A8 negative result; pivot to V3.3 (ZINB) or V3.4 (SCANVI) per V3_RESEARCH_PLAN.md §5.
* **HARD STOP** if both tracks fail beam reach ≥ 50 % at A3/A3-bg → A8 controllability-loss diagnosis; pivot to V3.4 (SCANVI) or contraction regulariser per V3_RESEARCH_PLAN.md §5.
* **HEADLINE PASS** when PPO − greedy_dyn_2 ≥ +0.05 pp at any cell with 4-seed 95 % CI excluding zero AND beam reach ≥ 50 % on the winning track → V3A complete; V3B = scale up + Chronos / GSEA test.
* **HEADLINE FAIL (publishable negative)** when greedy_dyn_2 still saturates (≥ 0.95 at K=3 primary) AND PPO − grd2 < +0.03 pp everywhere on both tracks → V3A complete with negative verdict; V3B = V3.3 / V3.4 / fallback agenda.
* **TIMEOUT** at 12 h wall-clock → save state, partial A8, defer remaining seeds/figures to next session.
