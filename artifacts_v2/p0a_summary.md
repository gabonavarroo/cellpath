# CellPath V2 P0A Summary

Scope: P0A.1 through P0A.5 only. No VAE, dynamics, or PPO retraining was run. V1 artifacts under `artifacts/`, `artifacts_64/`, and `artifacts/rl_sweeps/` were read only; all new outputs are under `artifacts_v2/`.

## Verification

- Targeted P0A/relevant shared tests: `83 passed, 15 warnings`.
- Full repository tests: `217 passed, 2 skipped, 19 warnings`.
- Syntax/import check: `py_compile` passed for all new P0A modules/scripts.

## P0A.1 Gate Breakdown

Command:

```bash
PYTHONPATH=. .venv/bin/python -m src.analysis.gate_breakdown --dynamics_dir artifacts/dynamics --pairs_dir artifacts/pairs --out artifacts_v2/diagnostics
```

Implemented:

- `src/analysis/gate_breakdown.py`
- `tests/test_p0a_gate_breakdown.py`

Outputs:

- `artifacts_v2/diagnostics/per_dim_margin.csv`
- `artifacts_v2/diagnostics/per_gene_margin.csv`
- `artifacts_v2/diagnostics/gate_breakdown_metadata.json`

Result:

- Val mean MLP-minus-ridge Pearson margin: `+0.007367`, matching the known failed primary gate against the `+0.030` requirement.
- OOD mean MLP-minus-ridge Pearson margin: `+0.040099`, so the average OOD ridge margin passes.
- Worst val dimension: dim 11, MLP Pearson `0.491241`, ridge `0.615107`, margin `-0.123865`.
- Worst OOD dimension: dim 11, MLP Pearson `0.036368`, ridge `0.469441`, margin `-0.433073`.
- Worst val genes by R2 margin: KIF18B `-0.026026`, LYL1 `-0.025133`, TGFBR2 `-0.024856`, CITED1 `-0.021502`, HK2 `-0.021404`.
- Worst OOD genes by R2 margin: KMT2A `-0.404466`, ARRDC3 `-0.230064`, FOXO4 `-0.212206`, RUNX1T1 `-0.199550`, CELF2 `-0.168811`.

Interpretation:

The primary gate failure is not uniform across the latent space. Dim 11 is the dominant negative contributor in both val and OOD. The val mean margin remains too small because the MLP only slightly improves over ridge in most dimensions, while a few dimensions/genes are ridge-favored. This supports the original diagnosis that the MLP is close to a ridge-like solution, but it also points to target/noise concentration in specific dimensions and genes.

## P0A.2 Per-Gene Action Contraction

Command:

```bash
PYTHONPATH=. .venv/bin/python scripts/diagnose_action_contraction.py --dynamics_dir artifacts/dynamics --vae_dir artifacts/vae --n_starts 500 --out artifacts_v2/diagnostics
```

Implemented:

- `scripts/diagnose_action_contraction.py`
- `gini_coefficient` and `shannon_entropy` in `src/analysis/metrics.py`
- `tests/test_p0a_rl_hard.py` coverage for baseline policy primitives used downstream

Outputs:

- `artifacts_v2/diagnostics/per_gene_contraction.csv`
- `artifacts_v2/diagnostics/per_gene_contraction_summary.json`

Result:

- Gini of mean improvement: `0.174806`.
- Entropy of absolute mean improvements: `4.588055`; max entropy `4.653960`; entropy fraction `0.985839`.
- Top 10 most contractive genes: HK2, MAP4K3, BCL2L11, MAML2, TSC22D1, BCORL1, KIF2C, MAP4K5, NCL, KIAA1804.
- Bottom 10 least contractive genes: IRF1, DUSP9, KLF1, TMSB4X, RHOXF2, SET, FEV, IKZF3, CEBPA, SLC4A1.
- PPO top-10 overlap with top contractive genes: 6 genes: HK2, KIAA1804, KIF2C, MAP4K3, MAP4K5, TSC22D1.

Interpretation:

H5 is only weakly supported. There is action structure because PPO overlaps strongly with the most contractive genes, but the global distribution is diffuse: Gini is below the `>0.30` clear-heterogeneity guide and entropy is near maximum. The dynamics field is therefore not perfectly uniform, but most genes remain broadly contractive enough that action choice is not sharply concentrated.

## P0A.3 p25 Epsilon and V2 Hard Benchmark

p25 epsilon command:

```bash
mkdir -p artifacts_v2 && PYTHONPATH=. .venv/bin/python -m src.utils.epsilon_percentile --vae_dir artifacts/vae --p 25 > artifacts_v2/epsilon_p25.txt
```

Result:

- `epsilon_p25 = 3.1662898064`

Smoke command:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_rl_hard.py --smoke --vae_dir artifacts/vae --dynamics_dir artifacts/dynamics --ppo_zip artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip --out_dir artifacts_v2/eval_hard_v1policy_smoke --k_values 3 --epsilon_values p25 --distance_bins 8-10 --held_out_genes_only true --n_episodes 20 --baselines random,always_noop,greedy_dyn_1,ridge_greedy,mean_delta_greedy
```

Smoke primary cell:

| policy | success | mean steps | mean final distance |
| --- | ---: | ---: | ---: |
| PPO deterministic | 1.000 | 2.40 | 2.819 |
| random uniform valid | 0.200 | 2.95 | 3.863 |
| greedy dynamics 1 | 1.000 | 2.40 | 2.815 |
| ridge greedy | 0.500 | 2.95 | 3.150 |
| mean-delta greedy | 0.750 | 2.80 | 3.165 |
| always NO-OP | 0.000 | 1.00 | 8.368 |

Full command:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_rl_hard.py --vae_dir artifacts/vae --dynamics_dir artifacts/dynamics --ppo_zip artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip --out_dir artifacts_v2/eval_hard_v1policy --k_values 1 2 3 8 --epsilon_values p25 p50 --distance_bins 4-6 6-8 8-10 10-12 --held_out_genes_only true,false --n_episodes 500 --baselines random,always_noop,greedy_dyn_1,ridge_greedy,mean_delta_greedy
```

Implemented:

- `src/utils/epsilon_percentile.py`
- `src/rl/baselines.py`
- `scripts/evaluate_rl_hard.py`
- `tests/test_p0a_epsilon.py`
- `tests/test_p0a_rl_hard.py`

Outputs:

- `artifacts_v2/epsilon_p25.txt`
- `artifacts_v2/eval_hard_v1policy_smoke/**/summary.json`
- `artifacts_v2/eval_hard_v1policy_smoke/results_table.md`
- `artifacts_v2/eval_hard_v1policy/**/summary.json`
- `artifacts_v2/eval_hard_v1policy/results_table.md`
- `artifacts_v2/eval_hard_v1policy/metadata.json`
- `artifacts_v2/eval_hard_v1policy/ridge_buffers.npz`

Full matrix accounting:

- Total policy-cell summaries: `384`.
- Completed cells: `336`.
- Skipped empty start-pool summaries: `48`, all corresponding to OOD `10-12` bins with no matching start states.

Primary V2 hard cell: `K=3`, `epsilon=p25`, `distance bin=8-10`, `split=ood`, `n=500`.

| policy | success | mean steps | mean final distance | top actions |
| --- | ---: | ---: | ---: | --- |
| PPO deterministic | 1.000 | 2.244 | 2.832 | CKS1B, TSC22D1, CELF2 |
| random uniform valid | 0.178 | 2.946 | 3.947 | SPI1, CDKN1A, S1PR2 |
| greedy dynamics 1 | 1.000 | 2.288 | 2.785 | CKS1B, TSC22D1, CELF2 |
| ridge greedy | 0.716 | 2.874 | 3.001 | ZC3HAV1, CSRNP1, MAML2 |
| mean-delta greedy | 0.824 | 2.804 | 3.035 | CSRNP1, MAML2, ZC3HAV1 |
| always NO-OP | 0.000 | 1.000 | 8.456 | NO_OP |

Primary-cell deltas:

- PPO minus random: `+82.2 pp`.
- PPO minus greedy dynamics 1: `0.0 pp`.

Interpretation:

The hard benchmark fixes the V1 random-baseline issue in the primary cell: random drops to `0.178`, well below the `<=0.40` guide. Existing PPO remains strong at `1.000`, and it matches the one-step dynamics oracle exactly on success. This is good for control feasibility, but it is not evidence of planning beyond the learned dynamics oracle. PPO appears to have learned the same high-contraction action structure used by greedy dynamics.

## P0A.4 Pairing-Noise Scoping

Command:

```bash
PYTHONPATH=. .venv/bin/python scripts/diagnose_pairing_noise.py --pairs_dir artifacts/pairs --out artifacts_v2/diagnostics/pairing_noise.json
```

Implemented:

- `scripts/diagnose_pairing_noise.py`
- `tests/test_p0a_pairing_noise.py`

Outputs:

- `artifacts_v2/diagnostics/pairing_noise.json`
- `artifacts_v2/diagnostics/pairing_noise.md`

Result:

- Genes measured: `84` train genes.
- Median noise ratio: `0.893540`.
- Mean noise ratio: `0.866617`.
- p25/p75 noise ratio: `0.812679` / `0.948555`.
- Max noise ratio: `0.981486`.

Interpretation:

This is the strongest P0A result. The current OT pair target has a very high residual/total Delta-z variance ratio. That means much of the supervised dynamics target is within-gene pseudo-pair residual variation rather than predictable gene-conditioned signal. This can directly cap the achievable MLP-over-ridge margin and makes a pure loss/architecture fix less scientifically clean unless pairing noise is corrected or at least ablated first.

## P0A.5 Biology Rerank

Command:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_biology_v2.py --rl_dir artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval --action_freq_ppo_det artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/eval_deterministic/action_freq.json --action_freq_random artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k_clean_eval/random_baseline/action_freq.json --depmap_csv data/processed/depmap_k562_chronos.parquet --out artifacts_v2/eval/depmap_v2
```

Implemented:

- `action_freq_chronos_spearman` in `src/analysis/metrics.py`
- `preranked_gsea` in `src/analysis/depmap_validation.py`
- `scripts/evaluate_biology_v2.py`
- `tests/test_p0a_biology.py`

Outputs:

- `artifacts_v2/eval/depmap_v2.csv`
- `artifacts_v2/eval/depmap_v2_summary.json`
- `artifacts_v2/eval/depmap_v2_metadata.json`

Result:

- PPO deterministic Spearman action frequency vs Chronos: rho `-0.023801`, p `0.815105`, 95% bootstrap CI `[-0.217706, 0.170790]`, n overlap `99`.
- Random Spearman: rho `0.123138`, p `0.224644`, CI `[-0.070253, 0.319652]`.
- PPO stochastic Spearman: rho `-0.060083`, p `0.554678`, CI `[-0.254899, 0.135793]`.
- PPO deterministic preranked GSEA: no panel reached `q <= 0.10`.

Interpretation:

The V2 ranked biology rerank is negative. Existing PPO action frequency does not detectably correlate with DepMap K562 Chronos over the action universe, and the panel GSEA has no q<=0.10 finding. This should be reported as negative evidence: the current surrogate/policy is steerable in latent space, but P0A does not support a claim that it prioritizes biologically dependency-associated genes beyond expectation.

## Overall P0A Scientific Read

P0A changes the V2 priority order. The hard benchmark no longer lets random look artificially competitive in the primary cell, so the evaluation protocol is useful. However, PPO matching greedy dynamics means the current policy quality is mostly explained by the learned dynamics field. The gate failure remains ridge-margin-specific and concentrated, and the pairing-noise ratio is high enough to make OT target quality the main scientific blocker. Biology rerank is null and must stay as a caveat.
