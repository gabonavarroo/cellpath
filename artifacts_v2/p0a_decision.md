# P0A Decision

Recommendation: **B. reorder toward pairing correction**.

Do not begin P0B exactly as planned yet, and do not begin P0C. The next V2 work should prioritize pairing correction/ablation before treating the ridge-margin failure as primarily a dynamics-loss problem.

## Evidence

### Gate breakdown

The failed primary validation margin was reproduced:

- Val MLP-minus-ridge Pearson margin: `+0.007367`, below the required `+0.030`.
- OOD MLP-minus-ridge Pearson margin: `+0.040099`, so OOD average margin passes.
- The failure is concentrated, not uniform: val dim 11 has margin `-0.123865`; OOD dim 11 has margin `-0.433073`.
- Worst val gene R2 margins are small but consistently ridge-favored: KIF18B `-0.026026`, LYL1 `-0.025133`, TGFBR2 `-0.024856`, CITED1 `-0.021502`, HK2 `-0.021404`.

Read: the model is close to the ridge baseline on the primary split, with a few dimensions/genes causing the most damage. This does not by itself prove architecture failure; it is compatible with a noisy pseudo-pair target where ridge averaging is hard to beat.

### Contraction heterogeneity

- Gini mean improvement: `0.174806`.
- Entropy fraction of max: `0.985839`.
- PPO top-10 overlap with top contractive genes: `6/10`.

Read: action choice has some structure, but not a sharply concentrated one. H5 is mixed: PPO is selecting many high-contraction genes, yet the contraction field remains diffuse enough that many actions are useful.

### Hard benchmark primary cell

Primary cell: `K=3`, `epsilon=p25`, `distance bin=8-10`, `split=ood`, `n=500`.

- PPO deterministic: success `1.000`, mean steps `2.244`.
- Random uniform valid: success `0.178`, mean steps `2.946`.
- Greedy dynamics 1: success `1.000`, mean steps `2.288`.
- Ridge greedy: success `0.716`.
- Mean-delta greedy: success `0.824`.
- Always NO-OP: success `0.000`.
- PPO minus random: `+82.2 pp`.
- PPO minus greedy dynamics 1: `0.0 pp`.

Read: the new hard benchmark is effective because random is no longer artificially high. But PPO exactly matches the dynamics oracle on success, so current evidence supports "PPO learned the greedy contraction structure" rather than multi-step planning beyond the learned surrogate.

### Pairing-noise ratio

- Median noise ratio: `0.893540`.
- Mean noise ratio: `0.866617`.
- p25/p75: `0.812679` / `0.948555`.

Read: this is the decisive result. The OT target appears dominated by within-gene residual pseudo-pair variation. That can cap any deterministic dynamics model's ability to beat ridge, and it makes a direct P0B loss/architecture sweep scientifically ambiguous. If P0B passes after correlation loss, it may be fitting noisy pair assignments rather than fixing the biological target. If P0B fails, the high pairing-noise ceiling was already the likely reason.

### Biology rerank

- PPO deterministic Spearman rho: `-0.023801`, p `0.815105`, CI `[-0.217706, 0.170790]`.
- No PPO deterministic GSEA panel reached `q <= 0.10`.

Read: current PPO actions are not detectably enriched by the ranked DepMap V2 tests. This does not invalidate the latent steering benchmark, but it blocks any stronger biology-prioritization claim.

## Decision Logic

Option A, proceed to P0B as planned, is not the best next step because the pairing-noise ratio is too high. A correlation loss or residual-over-ridge model may improve the gate, but P0A now shows that the target itself is likely noisy enough to dominate the margin story.

Option C, reorder toward reward/RL benchmark redesign, is also not first. The P0A hard benchmark already makes random hard enough in the primary cell, and PPO strongly beats random. Reward redesign can wait until the dynamics target is made cleaner.

Option D, stop and redesign all V2 assumptions, is too strong. The hard benchmark is informative, PPO is controllable under p25/K3/OOD, and diagnostics are now working.

Therefore the recommended path is **B: reorder toward pairing correction**:

1. Run a pairing corrective/ablation step next: mean-delta pairs and soft-OT expectation should be evaluated before or alongside any dynamics loss change.
2. Recompute gate breakdown and hard benchmark on the corrected pairing-derived dynamics only after that correction is tested.
3. Keep P0B correlation loss/residual-over-ridge as secondary ablations, not the immediate primary path.
4. Do not start P0C reward/RL retraining until the dynamics target and gate are resolved.

Stop here for P0A.
