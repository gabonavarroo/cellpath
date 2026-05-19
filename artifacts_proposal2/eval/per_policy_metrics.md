# V3B Phase 2 — Minimal Evaluation (Proposal 2)

- Dynamics: `artifacts_proposal1/dynamics_v1ot_ror` (V1 OT pairs + RoR + corr0.10)
- PPO_C: `artifacts_proposal2/rl_v3b_safety_aware_seed42/ppo.zip`
- PPO_C_permuted: `artifacts_proposal2/rl_v3b_safety_aware_seed42_permuted_chronos/ppo.zip`
- Start pool: 2249 held-out val cells with ‖z − z_ref‖ > 4.0
- ε = 4.5196 (p90 of control cell distances; epsilon_success.json)
- max_steps = 3; n_episodes per policy = 300

Reward used in env at eval time: `terminal_only_step_cost` (V2 reward, neutral cross-policy).
Both PPOs were *trained* under `safety_aware` reward; differing terminal eval reward isolates
the *policy* differences rather than reward-formula differences.

| Policy | success_rate | frac_zero_CE | mean_CE/ep | mean_path_tox | wmean_chronos | mean_n_steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_C | 1.000 | 1.000 | 0.0000 | 0.0000 | -0.1058 | 1.02 |
| ppo_C_permuted | 1.000 | 0.840 | 0.1600 | 0.0197 | -0.1101 | 1.00 |
| random_uniform_valid | 0.993 | 0.960 | 0.0400 | 0.0112 | -0.0936 | 1.14 |
| always_noop | 0.423 | 0.000 | 0.0000 | 0.0000 | +0.0000 | 1.00 |

**Legend**
- *success_rate* — fraction of episodes that reached ‖z − z_ref‖ < ε within max_steps.
- *frac_zero_CE* — fraction of episodes that picked **zero** common-essential genes (CBFA2T3, HK2, PLK4, PTPN1, STIL).
- *mean_CE/ep* — mean count of common-essential genes picked per episode.
- *mean_path_tox* — mean cumulative `tox_norm` of gene-action steps per episode.
- *wmean_chronos* — mean of (per-episode mean Chronos of gene picks). More negative = more essential-leaning.
- *mean_n_steps* — mean number of steps before termination.
