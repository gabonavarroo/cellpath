# P0C0 — Reachability Diagnostic (Diagnostics-Only, Before PPO Retrain)

> **Implementation plan for: `P0C0_REACHABILITY_PLAN.md`.**
> When executed, Task 1 commits a verbatim copy of this document into the repo at
> `/Users/gabo/Developer/ITAM/IA/cellpath/P0C0_REACHABILITY_PLAN.md`.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to
> execute task-by-task with review checkpoints.

**Goal:** Determine whether `artifacts_v2/dynamics_soft_ot_default/` is controllable by any
multi-step planner before spending compute on PPO retraining. The V2 hard benchmark (soft-OT
dynamics + V1 PPO) collapsed completely: 0/64 cells succeeded and the greedy oracle picked noop
in 40/64 cells. The hypothesis to test: *does the soft-OT field predict zero improvement for
ALL gene actions at ALL start states, or only at the 1-step greedy level?* Compare against
mean-delta dynamics and V1 OT (ground-truth working baseline) using the same start-pool logic.

**Architecture:** Pure diagnostics — no model retraining, no PPO training, no reward changes, no
VAE changes, no correlation-loss training. All new code is analysis scripts. Conditional
training paths (P0B‴, P0B2) are **recommendations only** — they are not executed in this plan.
All outputs go under `artifacts_v2/`. Large artifact directories are NOT committed; only code,
tests, plan documents, and PROGRESS.md are committed.

**Tech Stack:** Python 3.11, NumPy, PyTorch (MPS via `src/utils/device.py`), existing
`src/models/dynamics.py`, `src/rl/baselines.py`, `scripts/diagnose_action_contraction.py`,
`scripts/evaluate_rl_hard.py`.

---

## Context and Evidence

### What happened and why this plan exists

P0B″ produced `artifacts_v2/dynamics_soft_ot_default/` with:
- Val Pearson 0.9338 (gate PASSED, margin +0.0413)
- OOD Pearson 0.7434 (healthy, but MLP-ridge OOD margin collapses to +0.0026)
- OOD dim-11 regression: MLP Pearson on dim-11 OOD drops to 0.0045 (ridge holds at 0.7437)

The V2 hard benchmark on this dynamics + V1 PPO produced:
- 0/64 cells succeed (primary cell: PPO 0.000, greedy_dyn_1 0.000)
- 40/64 cells: greedy_dyn_1 picks noop exclusively (action_freq = {NO_OP: 500})
- Primary cell: greedy_dyn_1 final distance 8.456 (same as noop — no improvement)
- epsilon_p25 = 3.166 (gap to close from bin 8-10: ~5.3 units in ≤ 3 steps)

**Key questions:**
1. Does any 3-step gene sequence reach epsilon_p25 under soft-OT dynamics? (beam search)
2. If forced to pick genes (noop excluded), what distance is achieved? (NoopFreeGreedy)
3. What fraction of (start, gene) pairs are distance-reducing under soft-OT? (contraction)
4. What ε would make each dynamics field "solvable" for ≥25% of start cells? (ε-feasibility)

### Numeric summary table

| Dynamics | Val Pearson | Val margin | OOD Pearson | Greedy sr (primary) | Noise ratio |
|---|---:|---:|---:|---:|---:|
| V1 OT | 0.564 | +0.0074 | 0.490 | **1.000** | 0.8935 |
| mean_delta | 0.519 | +0.0214 | 0.383 | *(not run)* | 0.8493 |
| soft_ot | **0.934** | **+0.0413** | **0.743** | **0.000** | 0.7829 |

### Sacred rules (unchanged)
- No VAE retrain, no PPO retrain, no reward changes.
- No `torch.device()` outside `src/utils/device.py`.
- No seeding outside `src/utils/seeding.py`.
- No inline metric definitions (all new metrics go in `src/analysis/metrics.py`).
- No changes to `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/`.
- No gate threshold lowering.
- No force-adding large artifact directories to git.

---

## Hypotheses (preregistered)

**H_reach_soft_ot:** The soft-OT dynamics field is fundamentally anti-contractive: for all
OOD start cells in bin 8-10, no gene sequence of length ≤ 3 reaches distance < epsilon_p25
(3.166) according to the dynamics model.

**H_reach_mean_delta:** The mean-delta dynamics field is contractive enough for multi-step
planning to succeed on the primary cell, despite the gate failure.

**H_negative:** Neither soft-OT nor mean-delta is reachable even with multi-step planning.

---

## Files to create / modify

### New files (this phase)
- `scripts/probe_reachability.py` — multi-step beam-search reachability probe (repeat_mask both modes)
- `src/rl/baselines.py` (modify) — add `NoopFreeGreedyPolicy`
- `scripts/evaluate_rl_hard.py` (modify) — wire `greedy_dyn_1_noop_free` into `_policy_names()` and construction
- `artifacts_v2/diagnostics/action_contraction_soft_ot.{csv,json}` — local, not committed
- `artifacts_v2/diagnostics/action_contraction_mean_delta.{csv,json}` — local, not committed
- `artifacts_v2/reachability_probe/probe_results.json` — local, not committed
- `artifacts_v2/reachability_probe/decision.md` — local, not committed
- `artifacts_v2/interpretation_p0c0_reachability.md` — local, not committed
- `PROGRESS.md` (modify)

### Files that must NOT be modified
- Anything under `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/`
- `src/models/dynamics.py`
- `config/dynamics.yaml` gate thresholds
- `config/default.yaml` `pairing.method` default
- `src/rl/environment.py`, `src/rl/reward.py`, `src/rl/train_ppo.py`
- `artifacts/vae/*`

---

## Experiment sequence

```
D1: Per-gene contraction on soft-OT + mean-delta (reuse diagnose_action_contraction.py)
D2: NoopFreeGreedyPolicy hard benchmark (soft-OT, mean-delta, V1 OT)
D3: Beam-search reachability probe (soft-OT, mean-delta, V1 OT) — both repeat_mask modes
D4: ε-feasibility analysis (derived from D3 output)
───────────── REVIEW CHECKPOINT ─────────────
Write decision.md, interpretation .md, update PROGRESS.md, commit code
───────────── FUTURE WORK (not in this plan) ─────────────
PATH A → soft-OT reachable → P0C (PPO retrain on soft-OT)
PATH B → soft-OT not reachable, mean-delta is → P0B2 (corr loss on mean-delta)
PATH B' → soft-OT partially reachable → P0B‴ (corr loss on soft-OT)
PATH C → neither reachable → escalate
```

---

## Task breakdown

---

### Task 1: Commit the plan to the repo

**Files:**
- Create: `/Users/gabo/Developer/ITAM/IA/cellpath/P0C0_REACHABILITY_PLAN.md`

- [ ] **Step 1.1:** Copy the plan from `~/.claude/plans/you-are-the-cellpath-concurrent-bubble.md`
  to `/Users/gabo/Developer/ITAM/IA/cellpath/P0C0_REACHABILITY_PLAN.md`.

- [ ] **Step 1.2:** Commit.

```bash
git add P0C0_REACHABILITY_PLAN.md
git commit -m "$(cat <<'EOF'
docs: add P0C0 reachability diagnostic plan

Diagnostics-only. Determines whether soft-OT / mean-delta dynamics are
navigable by any multi-step planner before committing to PPO retrain.
Covers per-gene contraction (D1), NoopFree hard bench (D2), beam-search
reachability probe (D3), and ε-feasibility analysis (D4).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: D1 — Per-gene action-contraction analysis for soft-OT and mean-delta

Reuse `scripts/diagnose_action_contraction.py` (written in P0A). Outputs go under
`artifacts_v2/diagnostics/` (local, not committed).

- [ ] **Step 2.1:** Run contraction diagnostic on soft-OT dynamics.

```bash
PYTHONPATH=. .venv/bin/python scripts/diagnose_action_contraction.py \
  --dynamics_dir artifacts_v2/dynamics_soft_ot_default \
  --vae_dir artifacts/vae \
  --n_starts 500 \
  --out artifacts_v2/diagnostics/action_contraction_soft_ot
```

Key diagnostic: what fraction of (start, gene) pairs have `improvement > 0`?
V1 OT baseline was 95.5%. If soft-OT is below 30%, the field is net anti-contractive.

- [ ] **Step 2.2:** Run contraction diagnostic on mean-delta dynamics.

```bash
PYTHONPATH=. .venv/bin/python scripts/diagnose_action_contraction.py \
  --dynamics_dir artifacts_v2/dynamics_mean_delta_default \
  --vae_dir artifacts/vae \
  --n_starts 500 \
  --out artifacts_v2/diagnostics/action_contraction_mean_delta
```

- [ ] **Step 2.3:** Read both `per_gene_contraction_summary.json` files. Record:
  - `gini_mean_improvement`
  - `entropy_fraction_of_max`
  - `mean(fraction_positive)` from CSV
  - Top-5 contracting genes per model
  - Compare with V1 OT (in `artifacts_v2/diagnostics/per_gene_contraction_summary.json`)

---

### Task 3: D2 — Implement `NoopFreeGreedyPolicy` and run focused hard benchmark

**File to modify:** `src/rl/baselines.py`

- [ ] **Step 3.1:** Add `NoopFreeGreedyPolicy` to `src/rl/baselines.py`, after
  `GreedyDynamicsPolicy`. Constructor must match `GreedyDynamicsPolicy`'s keyword-only style
  (`dynamics` positional, remaining keyword-only):

```python
class NoopFreeGreedyPolicy:
    """Greedy 1-step dynamics policy that never picks noop.

    Identical to GreedyDynamicsPolicy but excludes noop_idx from candidates.
    Probes whether the dynamics field is navigable when the agent is forced
    to apply gene perturbations at every step (no early termination).
    Falls back to noop only if ALL gene actions are masked.
    """
    name = "greedy_dyn_1_noop_free"

    def __init__(self, dynamics: Any, *, n_genes: int, z_ref: np.ndarray, noop_idx: int) -> None:
        self.dynamics = dynamics
        self.n_genes = int(n_genes)
        self.z_ref = np.asarray(z_ref, dtype=np.float32)
        self.noop_idx = int(noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        import torch

        z = np.asarray(z, dtype=np.float32)
        valid = _valid_actions(mask)
        gene_actions = np.array([a for a in valid if 0 <= a < self.n_genes], dtype=np.int64)
        if len(gene_actions) == 0:
            return self.noop_idx  # only safe fallback when all genes are masked
        candidates: dict[int, np.ndarray] = {}
        z_batch = np.repeat(z[None, :], len(gene_actions), axis=0)
        gene_idx = gene_actions + 1  # 0-indexed action → 1-indexed gene_idx
        with torch.no_grad():
            out = self.dynamics(
                torch.from_numpy(z_batch).float(),
                torch.from_numpy(gene_idx.astype(np.int64)).long(),
            )
        z_next = out[0] if isinstance(out, tuple) else out
        if isinstance(z_next, torch.Tensor):
            z_next_np = z_next.detach().cpu().numpy().astype(np.float32)
        else:
            z_next_np = np.asarray(z_next, dtype=np.float32)
        for action, zn in zip(gene_actions, z_next_np, strict=True):
            candidates[int(action)] = zn
        return _argmin_distance(candidates, self.z_ref)
```

- [ ] **Step 3.2:** Wire `greedy_dyn_1_noop_free` into `scripts/evaluate_rl_hard.py`.

  **3.2a — Update the import at the top of `evaluate_rl_hard.py`** to include
  `NoopFreeGreedyPolicy`:

  ```python
  from src.rl.baselines import (
      AlwaysNoopPolicy,
      GreedyDynamicsPolicy,
      MeanDeltaGreedyPolicy,
      NoopFreeGreedyPolicy,
      RandomUniformValidPolicy,
      RidgeGreedyPolicy,
  )
  ```

  **3.2b — Update `_policy_names()`** to include `greedy_dyn_1_noop_free`:

  ```python
  def _policy_names(
      baseline_names: set[str],
      *,
      include_stochastic: bool,
  ) -> list[str]:
      names = ["ppo_deterministic"]
      if include_stochastic:
          names.append("ppo_stochastic")
      if "random" in baseline_names or "random_uniform_valid" in baseline_names:
          names.append("random_uniform_valid")
      if "always_noop" in baseline_names:
          names.append("always_noop")
      if "greedy_dyn_1" in baseline_names:
          names.append("greedy_dyn_1")
      if "greedy_dyn_1_noop_free" in baseline_names:
          names.append("greedy_dyn_1_noop_free")
      if "ridge_greedy" in baseline_names:
          names.append("ridge_greedy")
      if "mean_delta_greedy" in baseline_names:
          names.append("mean_delta_greedy")
      return names
  ```

  **3.2c — Add construction block** in the policy instantiation section (after the
  `greedy_dyn_1` block):

  ```python
  if "greedy_dyn_1_noop_free" in baseline_names:
      policies["greedy_dyn_1_noop_free"] = NoopFreeGreedyPolicy(
          dynamics_model, n_genes=n_genes, z_ref=z_ref, noop_idx=noop_idx,
      )
  ```

- [ ] **Step 3.3:** Run focused hard benchmark on primary cell only (n=100) for all
  three dynamics models with `greedy_dyn_1_noop_free` + `greedy_dyn_1` + `always_noop`.
  Outputs local, not committed.

```bash
# soft-OT dynamics
PYTHONPATH=. .venv/bin/python scripts/evaluate_rl_hard.py \
  --vae_dir       artifacts/vae \
  --dynamics_dir  artifacts_v2/dynamics_soft_ot_default \
  --ppo_zip       artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip \
  --out_dir       artifacts_v2/reachability_probe/hard_noop_free_soft_ot \
  --k_values 3 --epsilon_values p25 \
  --distance_bins 8-10 \
  --held_out_genes_only true \
  --n_episodes 100 \
  --baselines always_noop,greedy_dyn_1,greedy_dyn_1_noop_free

# mean-delta dynamics
PYTHONPATH=. .venv/bin/python scripts/evaluate_rl_hard.py \
  --vae_dir       artifacts/vae \
  --dynamics_dir  artifacts_v2/dynamics_mean_delta_default \
  --ppo_zip       artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip \
  --out_dir       artifacts_v2/reachability_probe/hard_noop_free_mean_delta \
  --k_values 3 --epsilon_values p25 \
  --distance_bins 8-10 \
  --held_out_genes_only true \
  --n_episodes 100 \
  --baselines always_noop,greedy_dyn_1,greedy_dyn_1_noop_free

# V1 OT dynamics (sanity check — greedy_dyn_1 must reproduce ≈1.000)
PYTHONPATH=. .venv/bin/python scripts/evaluate_rl_hard.py \
  --vae_dir       artifacts/vae \
  --dynamics_dir  artifacts/dynamics \
  --ppo_zip       artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip \
  --out_dir       artifacts_v2/reachability_probe/hard_noop_free_v1_ot \
  --k_values 3 --epsilon_values p25 \
  --distance_bins 8-10 \
  --held_out_genes_only true \
  --n_episodes 100 \
  --baselines always_noop,greedy_dyn_1,greedy_dyn_1_noop_free
```

  **V1 OT sanity check:** `greedy_dyn_1` must be ≈1.000. If V1 OT greedy fails, STOP — there
  is a bug in `NoopFreeGreedyPolicy` or the evaluate_rl_hard.py wiring. Debug before continuing.

- [ ] **Step 3.4:** Commit code changes only (no artifact directories).

```bash
git add src/rl/baselines.py scripts/evaluate_rl_hard.py
git commit -m "$(cat <<'EOF'
feat(p0c0): add NoopFreeGreedyPolicy + wire into evaluate_rl_hard

Forces gene selection at every step (noop excluded) to test whether
dynamics fields are navigable when the agent cannot give up early.
Constructor matches GreedyDynamicsPolicy keyword-only style.
Wired into _policy_names() and policy construction block.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: D3 — Beam-search multi-step reachability probe

**File to create:** `scripts/probe_reachability.py`

The probe does NOT use the RL environment. It calls the dynamics model directly in batch mode.
It supports `repeat_mask=true` (matches RL env: a gene used already in a sequence is excluded
from subsequent steps) and `repeat_mask=false` (upper-bound: any gene reusable at every step).
The RL env uses `repeat_mask=true` — that is the comparable result; `repeat_mask=false` is
only an upper-bound diagnostic.

Start pool is loaded using the exact same `_load_start_pool` logic as `evaluate_rl_hard.py`.
Import it directly rather than reimplementing.

- [ ] **Step 4.1:** Implement `scripts/probe_reachability.py`.

```python
"""P0C0 — multi-step beam-search reachability probe.

Tests whether any gene sequence of depth ≤ max_depth can bring OOD start
cells to distance < epsilon under a given dynamics model, without using the
RL environment (bypasses noop termination).

repeat_mask=True  (comparable to RL env): genes used in the current path
                   are excluded from subsequent steps.
repeat_mask=False (upper bound): any gene may be reused at every step.

Start pool is loaded via the same _load_start_pool() used in evaluate_rl_hard.py
to ensure identical distance-bin and held-out-gene semantics.

Outputs:
  probe_results.json   — per-dynamics summary + per-start-cell best trajectory
  probe_summary.md     — human-readable table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse start-pool loading from evaluate_rl_hard to guarantee identical semantics
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_rl_hard import _load_start_pool, _parse_bins  # noqa: E402


def _load_dynamics(dynamics_dir: Path) -> Any:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.analysis.gate_breakdown import load_dynamics_model
    return load_dynamics_model(dynamics_dir)


def beam_search(
    z_starts: np.ndarray,
    z_ref: np.ndarray,
    dynamics: Any,
    *,
    n_genes: int,
    max_depth: int,
    beam_width: int,
    epsilon: float,
    repeat_mask: bool,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run beam search from each start cell.

    gene_indices are 1-indexed to match the dynamics model convention
    (environment action a corresponds to gene_idx = a+1).
    Noop is never added to the beam.

    Returns one entry per start cell:
        {
            "start_distance": float,
            "best_final_distance": float,
            "best_gene_sequence": list[int],   # 1-indexed gene_idx
            "success": bool,
        }
    """
    all_gene_indices = np.arange(1, n_genes + 1, dtype=np.int64)  # 1-indexed
    results: list[dict[str, Any]] = []
    dynamics.eval()

    for z0 in z_starts:
        d_start = float(np.linalg.norm(z0 - z_ref))
        # Beam entry: (current_z, gene_sequence_as_set_of_1idx, gene_sequence_list, distance)
        beam: list[tuple[np.ndarray, frozenset[int], list[int], float]] = [
            (z0, frozenset(), [], d_start)
        ]

        for _depth in range(max_depth):
            if not beam:
                break
            candidates: list[tuple[np.ndarray, frozenset[int], list[int], float]] = []

            for z_cur, used_set, seq, _ in beam:
                if repeat_mask:
                    avail = np.array(
                        [g for g in all_gene_indices if g not in used_set],
                        dtype=np.int64,
                    )
                else:
                    avail = all_gene_indices

                if len(avail) == 0:
                    continue

                z_batch = np.repeat(z_cur[None, :], len(avail), axis=0)
                with torch.no_grad():
                    z_t = torch.from_numpy(z_batch).float().to(device)
                    g_t = torch.from_numpy(avail).long().to(device)
                    out = dynamics(z_t, g_t)
                    z_next_all = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy()

                dists = np.linalg.norm(z_next_all - z_ref, axis=1)
                for gi, (g, z_next, d_next) in enumerate(
                    zip(avail, z_next_all, dists, strict=True)
                ):
                    new_used = used_set | {int(g)} if repeat_mask else used_set
                    candidates.append((z_next, new_used, seq + [int(g)], float(d_next)))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[3])
            beam = candidates[:beam_width]

        if beam:
            z_best, _, seq_best, d_best = beam[0]
        else:
            z_best, seq_best, d_best = z0, [], d_start

        results.append({
            "start_distance": float(d_start),
            "best_final_distance": float(d_best),
            "best_gene_sequence": seq_best,
            "success": bool(d_best < epsilon),
        })

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Beam-search reachability probe for dynamics models.")
    ap.add_argument("--dynamics_dirs", nargs="+", required=True,
                    help="One or more <label>:<path> pairs, e.g. soft_ot:artifacts_v2/dynamics_soft_ot_default")
    ap.add_argument("--vae_dir", required=True)
    ap.add_argument("--pairs_dir", default="artifacts/pairs",
                    help="Pairs dir for held-out gene metadata (default: V1 OT pairs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epsilon", type=float, default=3.166289806365967)
    ap.add_argument("--distance_bin", default="8-10",
                    help="Single bin label, e.g. '8-10'")
    ap.add_argument("--held_out_genes_only", action="store_true", default=True)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--beam_width", type=int, default=50)
    ap.add_argument("--n_genes", type=int, default=105)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae_dir = Path(args.vae_dir)
    pairs_dir = Path(args.pairs_dir)

    import json as _json
    with open(pairs_dir / "metadata.json") as f:
        pairs_meta = _json.load(f)
    held_out_genes = [str(g) for g in pairs_meta.get("held_out_genes", [])]

    # Parse bin using the same helper as evaluate_rl_hard
    bins = list(_parse_bins([args.distance_bin]))
    if len(bins) != 1:
        raise ValueError(f"Expected a single bin, got {bins}")
    distance_bin = bins[0]

    z_ref = np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)
    z_starts = _load_start_pool(
        vae_dir,
        distance_bin=distance_bin,
        held_out_genes_only=args.held_out_genes_only,
        held_out_genes=held_out_genes,
    )
    print(f"Start pool: {len(z_starts)} cells in bin {args.distance_bin} "
          f"(held_out={args.held_out_genes_only})")

    all_results: dict[str, Any] = {
        "epsilon": args.epsilon,
        "distance_bin": args.distance_bin,
        "held_out_genes_only": args.held_out_genes_only,
        "max_depth": args.max_depth,
        "beam_width": args.beam_width,
        "n_start_cells": int(len(z_starts)),
        "dynamics_runs": {},
    }

    for spec in args.dynamics_dirs:
        label, dyn_path = spec.split(":", 1)
        for repeat_mask in (True, False):
            run_key = f"{label}_repeat{'on' if repeat_mask else 'off'}"
            print(f"\nBeam search: {run_key} ({dyn_path}), repeat_mask={repeat_mask}")
            dynamics = _load_dynamics(Path(dyn_path))
            dynamics.to(args.device)
            per_cell = beam_search(
                z_starts, z_ref, dynamics,
                n_genes=args.n_genes,
                max_depth=args.max_depth,
                beam_width=args.beam_width,
                epsilon=args.epsilon,
                repeat_mask=repeat_mask,
                device=args.device,
            )
            dynamics.to("cpu")
            n_success = sum(r["success"] for r in per_cell)
            best_dist = min(r["best_final_distance"] for r in per_cell) if per_cell else float("nan")
            mean_best = float(np.mean([r["best_final_distance"] for r in per_cell])) if per_cell else float("nan")
            print(f"  n_success={n_success}/{len(per_cell)} "
                  f"best_dist={best_dist:.4f} mean_best={mean_best:.4f}")
            all_results["dynamics_runs"][run_key] = {
                "dynamics_dir": dyn_path,
                "repeat_mask": repeat_mask,
                "n_successes": int(n_success),
                "success_rate": float(n_success / len(per_cell)) if per_cell else 0.0,
                "best_final_distance": float(best_dist),
                "mean_best_final_distance": float(mean_best),
                "per_cell": per_cell,
            }

    (out_dir / "probe_results.json").write_text(json.dumps(all_results, indent=2))

    lines = ["# Reachability probe summary\n\n"]
    lines.append(
        f"epsilon_p25={args.epsilon:.4f}, depth={args.max_depth}, "
        f"beam={args.beam_width}, n_starts={len(z_starts)}\n\n"
    )
    lines.append("| run_key | repeat_mask | n_success | success_rate | best_final_distance | mean_best |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for run_key, r in all_results["dynamics_runs"].items():
        lines.append(
            f"| {run_key} | {r['repeat_mask']} | {r['n_successes']} | "
            f"{r['success_rate']:.3f} | {r['best_final_distance']:.4f} | "
            f"{r['mean_best_final_distance']:.4f} |\n"
        )
    (out_dir / "probe_summary.md").write_text("".join(lines))
    print(f"\nResults written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4.2:** Run the beam-search probe on all three dynamics models.

```bash
PYTHONPATH=. .venv/bin/python scripts/probe_reachability.py \
  --dynamics_dirs \
      v1_ot:artifacts/dynamics \
      mean_delta:artifacts_v2/dynamics_mean_delta_default \
      soft_ot:artifacts_v2/dynamics_soft_ot_default \
  --vae_dir artifacts/vae \
  --pairs_dir artifacts/pairs \
  --out artifacts_v2/reachability_probe \
  --epsilon 3.1662898064 \
  --distance_bin 8-10 \
  --held_out_genes_only \
  --max_depth 3 \
  --beam_width 50 \
  --n_genes 105 \
  --device cpu
```

Expected outputs:
- `artifacts_v2/reachability_probe/probe_results.json`
- `artifacts_v2/reachability_probe/probe_summary.md`

**V1 OT sanity check:** `v1_ot_repeaton` success_rate must be > 0. If V1 OT also fails beam
search, the probe has a bug — STOP and debug before interpreting other results.

- [ ] **Step 4.3:** Read `probe_summary.md`. Record key results per the decision rules in
  Task 6.

- [ ] **Step 4.4:** Commit the probe script only.

```bash
git add scripts/probe_reachability.py
git commit -m "$(cat <<'EOF'
feat(p0c0): beam-search reachability probe for dynamics fields

Tests whether any k-step gene sequence can reach epsilon_p25 from OOD
bin-8-10 start cells under each dynamics model. Supports repeat_mask
on/off; reuses _load_start_pool from evaluate_rl_hard for identical
start-pool semantics. Calls dynamics directly (no RL env).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: D4 — ε-feasibility analysis (derived from probe output)

No new script. Derived analytically from `probe_results.json`.

- [ ] **Step 5.1:** Compute ε-reachability CDF from probe output using the
  `repeat_mask=True` runs (the comparable-to-RL-env result).

```python
import json, numpy as np
results = json.load(open("artifacts_v2/reachability_probe/probe_results.json"))
for run_key, r in results["dynamics_runs"].items():
    if not r["repeat_mask"]:
        continue  # report only RL-comparable results here
    dists = sorted(c["best_final_distance"] for c in r["per_cell"])
    print(f"{run_key}: p10={np.percentile(dists,10):.3f}, "
          f"p25={np.percentile(dists,25):.3f}, "
          f"p50={np.percentile(dists,50):.3f}, "
          f"p75={np.percentile(dists,75):.3f}")
```

This tells us: for soft-OT, what ε value would give ≥25% success? If that ε > epsilon_p50
(3.531), the field cannot support the current success criterion under any ε tightening.

- [ ] **Step 5.2:** Append the ε-feasibility table to `artifacts_v2/reachability_probe/probe_summary.md`.

---

### Task 6: REVIEW CHECKPOINT — Apply decision rules and write decision.md

Read all D1–D4 outputs. Apply the following rules to select the recommended next path.

```
Decision rule matrix (repeat_mask=True results):

┌──────────────────────────────────────────────────────┬────────────────────────────────────┐
│ Condition                                             │ Recommended path                   │
├──────────────────────────────────────────────────────┼────────────────────────────────────┤
│ soft_ot probe success_rate ≥ 0.10                     │ PATH A: P0C — PPO retrain on       │
│                                                       │ soft-OT as-is                      │
├──────────────────────────────────────────────────────┼────────────────────────────────────┤
│ soft_ot 0 < success_rate < 0.10 OR                    │ PATH B': P0B‴ — corr loss on        │
│ best_final_distance < 5.0 (field has direction)       │ soft-OT, then re-probe             │
├──────────────────────────────────────────────────────┼────────────────────────────────────┤
│ soft_ot infeasible (best_dist ≥ 5.0) AND              │ PATH B: P0B2 — corr loss on         │
│ mean_delta success_rate ≥ 0.10                        │ mean-delta                         │
├──────────────────────────────────────────────────────┼────────────────────────────────────┤
│ Both infeasible                                       │ PATH C: Escalate                   │
├──────────────────────────────────────────────────────┼────────────────────────────────────┤
│ V1 OT probe fails sanity check                        │ STOP — probe has a bug             │
└──────────────────────────────────────────────────────┴────────────────────────────────────┘
```

- [ ] **Step 6.1:** Write `artifacts_v2/reachability_probe/decision.md` with:
  - Numerical results from D1–D4
  - Which path applies and why (single recommended path)
  - Supporting evidence from the rule matrix

---

### Task 7: Write interpretation, update PROGRESS.md, and commit

- [ ] **Step 7.1:** Write `artifacts_v2/interpretation_p0c0_reachability.md` using
  this template (fill in all `…` with measured values):

```markdown
# P0C0 — Reachability Diagnostic Interpretation (2026-05-16)

## Configuration

| Parameter | Value |
|---|---|
| epsilon_p25 | 3.1663 |
| start pool | OOD cells, bin 8–10, n=… |
| beam_width | 50, max_depth=3 |
| NoopFree greedy n_episodes | 100 |

## D1: Per-gene contraction

| Dynamics | Gini | Entropy fraction | Mean fraction_positive | Top-3 genes |
|---|---:|---:|---:|---|
| V1 OT | 0.1748 | 0.9858 | 0.955 | … |
| mean_delta | … | … | … | … |
| soft_ot | … | … | … | … |

Interpretation: …

## D2: NoopFree greedy benchmark (primary cell, n=100)

| Dynamics | greedy_dyn_1 sr | noop_free sr | noop_free mean_final_dist |
|---|---:|---:|---:|
| V1 OT | ~1.000 | … | … |
| mean_delta | … | … | … |
| soft_ot | 0.000 | … | … |

Interpretation: …

## D3: Beam-search reachability probe (repeat_mask=True)

| Dynamics | n_success | success_rate | best_final_distance | mean_best |
|---|---:|---:|---:|---:|
| V1 OT | … | … | … | … |
| mean_delta | … | … | … | … |
| soft_ot | … | … | … | … |

Interpretation: …

## D4: ε-feasibility (repeat_mask=True)

| Dynamics | ε for 10% success | ε for 25% success | ε for 50% success |
|---|---:|---:|---:|
| V1 OT | … | … | … |
| mean_delta | … | … | … |
| soft_ot | … | … | … |

Interpretation: …

## Verdict

- **H_reach_soft_ot:** supported / rejected
- **H_reach_mean_delta:** supported / rejected

## Recommended next step

**PATH [A/B/B'/C]:** [one-sentence rationale]

[PATH A] `artifacts_v2/dynamics_soft_ot_default/` is navigable. Proceed to P0C:
  retrain PPO on soft-OT dynamics (command requires explicit approval).
  ```bash
  .venv/bin/python scripts/train_rl.py --config-name default \
      paths.dynamics_dir=artifacts_v2/dynamics_soft_ot_default \
      paths.rl_dir=artifacts_v2/rl_soft_ot \
      rl.total_timesteps=500000 rl.reward.start_epsilon_label=p50 seed=42
  ```

[PATH B'] soft-OT shows partial directionality (best_dist < 5.0) but success_rate < 0.10.
  Retrain dynamics with correlation loss (λ_corr ∈ {0.05, 0.10}) on soft-OT pairs,
  re-probe, then gate-check before proceeding to P0C.

[PATH B] soft-OT infeasible; mean-delta shows reachable trajectories. Retrain dynamics with
  correlation loss (λ_corr ∈ {0.05, 0.10}) on mean-delta pairs to close the gate
  (val margin +0.0214 → target +0.030), then proceed to P0C on mean-delta dynamics.

[PATH C] Neither dynamics supports the current epsilon objective. Escalate: investigate
  reducing epsilon to the observed beam p50, or switch to terminal-only reward, or revisit
  the V2 dynamics objective design.
```

- [ ] **Step 7.2:** Update `PROGRESS.md` with new session entry (format per CLAUDE.md §8).

```markdown
## Session 2026-05-16-HHMM  (agent: research-lead)

**Phase:** P0C0 — Reachability diagnostic
**Status:** Ran D1 (per-gene contraction), D2 (NoopFree greedy), D3 (beam-search probe),
D4 (ε-feasibility) on soft-OT, mean-delta, V1 OT. Decision: PATH …
**Metrics:**
| Component | soft-OT | mean-delta | V1 OT |
| --- | --- | --- | --- |
| fraction_positive (contraction) | … | … | ~0.955 |
| noop_free sr (n=100) | … | … | … |
| Probe success_rate (repeat=on) | … | … | … |
| Best final distance | … | … | … |
**Blockers:** none
**Next:** (one-line from selected path)
```

- [ ] **Step 7.3:** Commit code and documentation only.

```bash
git add artifacts_v2/interpretation_p0c0_reachability.md \
  artifacts_v2/reachability_probe/decision.md \
  PROGRESS.md
git commit -m "$(cat <<'EOF'
docs(p0c0): reachability interpretation + PROGRESS update

Path: <A/B/B'/C>. Soft-OT probe (repeat=on): success_rate=…, best_dist=….
Mean-delta probe (repeat=on): success_rate=…, best_dist=….

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Full test sweep + V1 artifact verification

- [ ] **Step 8.1:**

```bash
.venv/bin/pytest -q
```

Expected: all previously passing tests remain green. Zero regressions.

- [ ] **Step 8.2:** Verify V1 artifacts are clean.

```bash
git status -- artifacts/ artifacts_64/ artifacts/rl_sweeps/
```

Expected: empty output (no changes).

---

## Final decision rules

| Probe result (repeat=on) | Action |
|---|---|
| soft_ot success_rate ≥ 0.10 | **PATH A: P0C on soft-OT as-is** |
| soft_ot 0 < best_dist < 5.0 OR 0 < success_rate < 0.10 | **PATH B': P0B‴ — corr loss on soft-OT** |
| soft_ot infeasible (best_dist ≥ 5.0), mean_delta ≥ 0.10 | **PATH B: P0B2 — corr loss on mean-delta** |
| Both infeasible | **PATH C: Escalate** |
| V1 OT probe fails | **STOP: probe bug** |

**Definition of "success":** at least one start cell reaches distance < epsilon_p25 (3.166)
via a ≤ 3-step gene sequence, beam_width=50, repeat_mask=True.

**The recommended path must be singular.** The interpretation document states exactly one
next action.

---

## Self-review

- **Diagnostics-only:** no model training, no reward changes, no PPO, no VAE.
- **Tasks 8/9 from prior plan (corr-loss training)** are future work, documented only as
  conditional recommendations in the interpretation template. Not executable in this plan.
- **repeat_mask:** probe runs both modes; the RL-comparable result (repeat=on) drives decisions;
  repeat=off is labeled as "upper bound" in the summary.
- **Start-pool semantics:** directly imports `_load_start_pool` and `_parse_bins` from
  `evaluate_rl_hard.py` — guaranteed identical distance-bin and held-out-gene logic.
- **NoopFreeGreedyPolicy constructor:** matches `GreedyDynamicsPolicy` exactly — `dynamics`
  positional, `n_genes/z_ref/noop_idx` keyword-only.
- **_policy_names() and construction block:** both updated so the baseline is discovered and
  instantiated consistently.
- **No large artifacts committed:** only `.py`, `.md`, `PROGRESS.md` in git.
- **Single path recommendation:** interpretation template has exactly one conditional block
  in "Recommended next step".
- **Sacred-rule conformance:** no path hardcoding, no inline metrics, no `torch.device()`,
  no seeding outside canonical modules.
