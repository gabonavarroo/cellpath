# V3C Final Replogle Held-out Audit (Bucket-C)

> **Status:** FINAL — gene-set source re-downloaded, reproduced Phase 2c counts exactly,
> ran the action-overlap audit against the final V3C champion.
>
> **Scope:** post-hoc held-out **action-overlap** audit of the V3C champion against an
> independent K562 essentiality source (Replogle 2022 CRISPRi). No retraining, no PPO
> training, no external dynamics validation, no biological discovery claims.

---

## 1. Champion under audit

| Field | Value |
|---|---|
| Champion name | `contraction_aware_v2_aggressive` + PPO_BCD seed 42 500k |
| Champion type | `CHAMPION_TUNED_RESULT` (per `final_champion_manifest.json`) |
| Discriminating cell | K=3 / bin 8-10 / OOD (primary signal cell) |
| Headline | PPO_BCD = 0.840 vs greedy_dyn_3_fused = 0.765 (**+0.075** at seed 42) |
| 4-seed verdict | `V2AGG_VARIANCE_BOUNDED` — 3/4 seeds reproduce 0.840, seed 0 outlier 0.705 |
| Secondary reference | Track L 4-seed × 1M (`LOCKED_DEFAULT_RESULT`) |

## 2. Replogle source

| | |
|---|---|
| Source | Harmonizome 3.0 mirror of Replogle et al., *Cell* 2022 — "K562 Essential Perturb-seq Gene Perturbation Signatures" |
| Slug | `reploglek562essential` |
| URL  | `https://maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.,+Cell,+2022+K562+Essential+Perturb-seq+Gene+Perturbation+Signatures` |
| Primary citation | Replogle JM et al. *Cell* 185, 2559-2575 (2022). DOI: 10.1016/j.cell.2022.05.013 |
| Files retrieved | `gene_list_terms.txt.gz` (64 KB), `attribute_list_entries.txt.gz` (33 KB), `gene_attribute_edges.txt.gz` (4.7 MB) under `data/raw/replogle/` |
| Download script | `scripts/download_replogle_heldout.py` |
| Downloaded at | 2026-05-21 02:07 UTC |
| Modality | dCas9-KRAB **CRISPRi** (knockdown) — independent direction from DepMap-Chronos CRISPR-Cas9 KO |
| Stage 1 availability | gene-set/signature only (Phase 2c residue persisted only as intersection JSON; raw source files were lost from /tmp) |

## 3. Gene-set construction (reproduces Phase 2c exactly)

* **Replogle essential perturbations** parsed from `attribute_list_entries.txt.gz`,
  column 1 (`Gene Perturbation`), regex `^\d+_(.+?)_(P\d+(?:P\d+)?)$`. NON-TARGETING
  control rows and ENST-suffixed isoform rows are excluded.
* **Unique HGNC symbols extracted:** **1875** ✅ matches Phase 2c (1875).
* **Norman 105 single-gene CRISPRa action universe**: from
  `data/processed/norman_hvg.h5ad::uns["perturbation_encoder"]` (105 single-gene actions,
  `NO_OP` is index 105).
* **DepMap K562 essentials**: 1877 genes with `is_essential = True` in
  `data/processed/depmap_k562_chronos.parquet`.

### 3.1 Intersection counts (matches Phase 2c precisely)

| Set | Count | Members |
|---|---:|---|
| Replogle ∩ Norman 105 | **6** | FOXL2, KIF18B, NCL, PLK4, SET, STIL |
| DepMap ∩ Norman 105 | 5 | CBFA2T3, HK2, PLK4, PTPN1, STIL |
| Agreement (both ess.) | 2 | PLK4, STIL |
| DepMap-only ∩ Norman 105 | 3 | CBFA2T3, HK2, PTPN1 |
| **Replogle-only ∩ Norman 105 (Bucket-C clean)** | **4** | **FOXL2, KIF18B, NCL, SET** |
| Random expected frac | 0.0381 | 4 / 105 |

All counts and members reproduce Phase 2c exactly.

## 4. V3C champion overlap metrics

### 4.1 Champion seed 42 at the primary cell — per-policy on the same dynamics field

K=3 / bin 8-10 / OOD on the `contraction_aware_v2_aggressive` field, seed-42 evaluation
(`artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed42_500k/eval/k3_bin8-10_splitood/`):

| Policy | success | total gene-actions | actions on Replogle-only | frac Replogle-only | enrichment vs random (×) | actions on Replogle-essential | frac Replogle-essential | actions on DepMap-essential |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **PPO_BCD** (champion) | **0.840** | 600 | **32 (all FOXL2)** | **0.0533** | **×1.40** | 32 | 0.0533 | 0 |
| greedy_dyn_3_fused | 0.765 | 600 | 23 (all FOXL2) | 0.0383 | ×1.01 | 23 | 0.0383 | 0 |
| greedy_dyn_2_fused | 0.705 | 600 | 0 | 0.0000 | ×0.00 | 0 | 0.0000 | 0 |
| greedy_dyn_1_fused | 0.705 | 600 | 0 | 0.0000 | ×0.00 | 0 | 0.0000 | 0 |
| random_uniform_valid | 0.020 | 591 | 17 (FOXL2, SET, KIF18B, NCL) | 0.0288 | ×0.76 | 19 | 0.0322 | 21 |
| always_noop | 0.000 | 0 | — | — | — | — | — | — |

**Reading.** Both the champion PPO and the depth-3 greedy oracle pick FOXL2 specifically at
this discriminating cell; PPO picks it ~40 % more often than greedy (32 vs 23 of 600
actions). Shallower greedy (depth 1, 2) does not pick FOXL2 at all. The random baseline
picks all four Replogle-only essentials roughly at the expected 3.8 % rate. DepMap-essential
frac is 0 % across every PPO_BCD trajectory at every cell on this field — the
locked safety constraint is structurally satisfied.

### 4.2 Champion-field 4-seed PPO_BCD at primary cell

| Seed | success | total | n_FOXL2 selected | frac Replogle-only | enrichment × |
|---:|---:|---:|---:|---:|---:|
| **42 (champion)** | **0.840** | 600 | **32** | **0.0533** | **×1.40** |
| 0 | 0.705 | 600 | 0 | 0.0000 | ×0.00 |
| 1 | 0.840 | 600 | 0 | 0.0000 | ×0.00 |
| 7 | 0.840 | 600 | 0 | 0.0000 | ×0.00 |
| **4-seed mean** | 0.806 | — | — | **0.0133** | — |
| 4-seed 95 % CI | — | — | — | **[−0.013, +0.040]** | — |
| 4-seed std | — | — | — | 0.0267 | — |

**Reading.** The seed-42 +0.075 advantage is mediated by FOXL2 selection. The other three
seeds (0, 1, 7) reproduce the seed-42 success rate of 0.840 *without* picking any Replogle-
only essential — meaning the +0.075 advantage at seed 42 specifically is correlated with
heavy FOXL2 use, not a stable property of the policy class. Across 4 seeds the
Replogle-only fraction CI straddles zero (and straddles the random expectation 3.81 %).

### 4.3 Champion seed 42 across the full 9-cell matrix

| Cell | success | total | n FOXL2 | frac Replogle-only | DepMap-essential |
|---|---:|---:|---:|---:|---:|
| K=2 / bin 6-8 / OOD | 0.185 | 400 | 1 | 0.0025 | 0 |
| K=2 / bin 8-10 / OOD | 0.000 | 400 | 32 | 0.0800 | 0 |
| K=3 / bin 6-8 / OOD | 0.800 | 563 | 1 | 0.0018 | 0 |
| **K=3 / bin 8-10 / OOD** (primary) | **0.840** | 600 | **32** | **0.0533** | 0 |
| K=4 / bin 6-8 / OOD | 0.960 | 603 | 1 | 0.0017 | 0 |
| K=4 / bin 8-10 / OOD | 1.000 | 632 | 32 | 0.0506 | 0 |
| K=5 / bin 6-8 / OOD | 1.000 | 611 | 1 | 0.0016 | 0 |
| K=5 / bin 8-10 / OOD | 1.000 | 632 | 32 | 0.0506 | 0 |
| K=8 / bin 8-10 / OOD | 1.000 | 632 | 32 | 0.0506 | 0 |

**Pattern.** Seed-42's FOXL2 sub-policy is bin-8-10-specific (32 selections at every
bin-8-10 cell regardless of K-budget) and absent at bin-6-8. This is the same
sub-policy applied across step-budgets — consistent with PPO having learned a fixed
gene-mixture for the hardest starting bin.

### 4.4 Track L 4-seed (LOCKED_DEFAULT secondary) at its discriminating cell

K=2 / bin 8-10 / OOD on the `track_l_n64_legacy_ror_corr010` field (the V3B Phase 4
binding cell):

| Seed | PPO_BCD success | n Replogle-only | frac Replogle-only | greedy_dyn_2 frac | random frac |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.705 | 0 | 0.0000 | 0.0000 | 0.0354 |
| 0 | 0.705 | 0 | 0.0000 | — | — |
| 1 | 0.705 | 0 | 0.0000 | — | — |
| 7 | 0.705 | 0 | 0.0000 | — | — |

**Reading.** Track L's locked 4-seed PPO_BCD never selects any Replogle-only essential at
its primary cell. Greedy and random baselines also avoid them (greedy at 0 %; random at
the expected 3.5 %). By this audit metric Track L is **safer** than the
v2_aggressive seed-42 champion at the cell where each is most discriminating.

### 4.5 V2 anchor (cross-field reference)

At V2 anchor K=2 / bin 8-10 / OOD:

| Policy | success | frac Replogle-only | frac Replogle-essential |
|---|---:|---:|---:|
| PPO_BCD | 0.120 | 0.0000 | 0.0000 |
| PPO_A   | 0.055 | 0.0000 | 0.0000 |
| greedy_dyn_1_fused | 0.250 | 0.0000 | 0.0000 |
| greedy_dyn_2_fused | 0.120 | 0.0000 | 0.0000 |
| random_uniform_valid | 0.010 | 0.0354 | 0.0505 |

**Reading.** V2 anchor is saturated and does not exercise Replogle-only essentials in
any planning policy. Confirms the V2 32D `RoR_corr010` field is biologically uninformative
on this axis — the audit signal originates entirely with the 64D contraction-aware field.

## 5. Comparison summary — champion vs alternatives at each field's discriminating cell

| Result | discriminating cell | PPO_BCD success | Δ vs same-field greedy | frac Replogle-only essential | frac DepMap-essential | Bucket-C verdict on this audit |
|---|---|---:|---:|---:|---:|---|
| **Champion seed 42** | K=3/b8-10 (v2_aggressive) | **0.840** | **+0.075** vs g_3 | **0.0533** (×1.40 random) | 0 | held-out essentiality usage *higher* than greedy |
| Champion 4-seed mean | K=3/b8-10 (v2_aggressive) | 0.806 | +0.041 | 0.0133 [CI −0.013, +0.040] | 0 | seed-42 elevation washed out by the 3 reproducing seeds |
| Champion seeds 1, 7 | K=3/b8-10 (v2_aggressive) | 0.840 | +0.075 | 0.0000 | 0 | reproduces seed-42 success *without* held-out essentiality use |
| Track L 4-seed | K=2/b8-10 (track_l) | 0.705 | +0.010 | 0.0000 | 0 | clean — no held-out essentiality selected |
| V2 anchor PPO_BCD | K=2/b8-10 (anchor) | 0.120 | 0.000 | 0.0000 | 0 | clean — saturated reference |

**The single most important observation:** of the four seeds that achieve PPO_BCD = 0.84 at
the champion cell, **only seed 42 routes through FOXL2**. Seeds 1 and 7 reach the same
success rate via gene mixtures that avoid every Replogle-only essential. Seed 0 reaches
only 0.705 and also avoids Replogle-only essentials. The seed-42 +0.075 *advantage* over
greedy is therefore confounded with (and partially explained by) heavy FOXL2 selection.

## 6. Does the V3C champion clear the held-out essentiality audit?

* **DepMap-Chronos safety prior**: faithfully optimised — `frac_DepMap_essential = 0`
  across every PPO_BCD evaluation on this field. This is reward-prior optimisation,
  consistent with V3B Phase 2c.
* **Replogle-only essentiality (held out)**: the Phase 2c verdict
  `HELDOUT_INCONCLUSIVE_NO_GENERALIZATION_DETECTED` is **preserved and sharpened**:
  - The 4-seed CI on champion frac_Replogle-only straddles zero AND straddles the random
    baseline. **No held-out generalisation detected.**
  - The seed-42 outlier *positively selects* FOXL2 at ×1.40 random — *higher* than
    greedy on the same field. So the seed-42 result is not "consistent with safer action
    selection under an external essentiality source"; on this metric it is
    consistent with the opposite.
  - Seeds 1, 7 — which reproduce the +0.075 advantage — *do* avoid Replogle-only
    essentials. So **the policy class can hit 0.84 at this cell without selecting FOXL2**;
    seed-42 is one realisation that happens to use FOXL2.

**Better, worse, or indistinguishable on this held-out audit?**

* vs same-field greedy_dyn_3: **statistically indistinguishable** (4-seed CI includes
  greedy's 3.83 % and random's 3.81 %), but the *champion checkpoint* (seed 42) is
  point-estimate-worse (×1.40 vs ×1.01 enrichment).
* vs Track L 4-seed: Track L is **cleaner** at its primary cell (0 % vs 1.33 % mean).
* vs V2 anchor: anchor is **cleaner** but also non-functional (0.12 success vs 0.806).

## 7. Limitations

1. **Action-overlap is not biological discovery.** The audit measures only which Norman-CRISPRa actions a policy selects, intersected with a published K562-essentiality panel. It does *not* run Replogle cells through the model, does not validate dynamics generalisation, and does not produce any therapeutic claim. See `v3c_replogle_single_cell_feasibility.md` (Stage 5) for why a state-side probe is out of scope.
2. **Modality mismatch (the central interpretive caveat).** Norman CRISPRa *over-expresses* the target gene. Replogle CRISPRi *knocks down* the target gene. The Replogle-essentiality label means "knockdown kills K562"; it does *not* automatically imply "over-expression is harmful". FOXL2 over-expression in K562 may or may not be a problem; this audit cannot decide. It can only flag the fact that the champion selects a gene that is essential under the opposite perturbation direction.
3. **Small Norman ∩ Replogle (n = 6).** With only 4 Replogle-only essentials in the 105 action universe, the audit has low statistical power. A 3 % selection-rate shift between PPO and greedy is the noise floor.
4. **PPO_C (Phase 2c) vs PPO_BCD (V3C).** This audit's V3C champion uses the locked B+C+D reward (path-length freeband + DepMap-Chronos safety + uncertainty). It is a different policy from the V3B Phase 2c PPO_C the original Replogle held-out check was applied to. Direct numerical comparison to Phase 2c per-seed CSVs is *only* valid on the gene-set construction (which we reproduced exactly), not on the policy behaviour.
5. **Variance bound persists.** The seed-42 advantage is `V2AGG_VARIANCE_BOUNDED` per the manifest. This audit adds an additional layer to the same caveat: the *mechanism* of the seed-42 advantage involves a CRISPRi-essential gene; the *advantage* is single-seed; the 4-seed mean does not exclude zero on the held-out metric.
6. **No CRISPRi action axis.** The CellPath action space is CRISPRa-only. The Replogle CRISPRi source cannot be used as an action universe in v1; only as a passive gene-set panel. The eventual CRISPRi integration (action-space extension, `DATA.md` §7) would change the auditability of the safety prior substantially.

## 8. Verdict

`V3C_REPLOGLE_HELDOUT_AUDIT_INCONCLUSIVE_CHAMPION_USES_FOXL2`

The V3C champion does not show a held-out essentiality advantage against the Replogle
2022 K562 CRISPRi panel. On the strict 4-seed Bucket-C metric the result is
**inconclusive** (CI [−0.013, +0.040], includes zero and random baseline). On the
seed-42 *point* estimate the champion specifically uses a Replogle-essential gene
(FOXL2, 32 selections / 600 actions, ×1.40 random) — disclosure-worthy and pulling
the audit in the *unfavourable* direction. Seeds 1 and 7 reproduce the seed-42 success
rate *without* using FOXL2, so the +0.075 advantage is realisable without held-out
essentiality selection.

## 9. Presentation-safe conclusion

> The V3C champion is consistent with V3B's earlier finding: the DepMap safety prior
> is faithfully optimised within the reward, but it does not transfer to an independent
> K562 essentiality assay (Replogle 2022 CRISPRi). The seed-42 PPO checkpoint that
> achieves the +0.075 advantage at K=3/bin8-10/OOD does so by routing through FOXL2 —
> a Replogle-essential gene not present in the DepMap reward. Three of four seeds
> reach equivalent success without selecting any Replogle-only essential, so the
> mechanism is not load-bearing for the policy class, but it is load-bearing for the
> specific checkpoint we ship as the champion. We disclose this as a Bucket-C limitation
> rather than a positive held-out validation.

## 10. Files produced

```
data/raw/replogle/
├── gene_list_terms.txt.gz            (64 KB, 8055 background genes — universe)
├── attribute_list_entries.txt.gz     (33 KB, 2012 essential perturbations)
└── gene_attribute_edges.txt.gz       (4.7 MB, sparse gene-attribute edges)

data/processed/replogle/
├── replogle_essential_genes.json     (1875 unique HGNC symbols + provenance)
├── replogle_essential_genes.csv      (one symbol per line — for grep / diff)
├── replogle_norman_intersection.json (Phase 2c-schema-compatible intersection summary)
├── replogle_only_essential_genes.json (4-gene Bucket-C clean set)
└── source_metadata.json              (URL, slugs, file checksums, parsing assumptions, timestamps)

artifacts_v3/v3c/
├── replogle_heldout_action_overlap.csv          (277-row per-run/seed/cell/policy table)
└── interpretation/
    ├── v3c_replogle_availability_check.md       (Stage 1)
    ├── v3c_final_replogle_heldout_audit.md      (this file — Stage 6)
    ├── v3c_replogle_single_cell_feasibility.md  (Stage 5)
    └── v3c_final_replogle_heldout_audit_metrics.json   (Stage 4 raw metrics)

artifacts_v3/v3c/figures/
└── replogle_heldout_action_overlap.png          (Stage 6 figure)

scripts/
├── download_replogle_heldout.py  (new — Stage 2: fetch + parse + persist)
└── audit_v3c_replogle_heldout.py (new — Stage 4: action-overlap metrics)
```

## 11. Reproduction

```bash
# Stage 2 — re-fetch the Harmonizome gene-set artifacts and rebuild processed JSON/CSV
python scripts/download_replogle_heldout.py

# Stage 4 — run the action-overlap audit (reads existing eval summary.json files)
python scripts/audit_v3c_replogle_heldout.py
```

Frozen-tier check:

```bash
git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/   # expect clean
```

## 12. Links

- Champion manifest: `artifacts_v3/v3c/final_champion_manifest.json`
- V3C closeout: `artifacts_v3/v3c/interpretation/v3c_final_closeout.md` (Bucket-C status §10)
- V3B Phase 2c (original Replogle source citation): `artifacts_v3/interpretation/v3b_phase2c_seed_escalation.md`
- V3B Phase 2c summary: `artifacts_v3/eval_v3b_phase2c/replogle_heldout_summary.md`
- Sacred rules: `CLAUDE.md` §3 (no retraining, no inline metrics, frozen-tier untouched).
