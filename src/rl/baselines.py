"""Read-only policy baselines for the V2 hard RL benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _valid_actions(mask: np.ndarray) -> np.ndarray:
    valid = np.where(np.asarray(mask, dtype=bool))[0]
    return valid.astype(np.int64)


class RandomUniformValidPolicy:
    name = "random_uniform_valid"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        valid = _valid_actions(mask)
        if len(valid) == 0:
            return int(len(mask) - 1)
        return int(self.rng.choice(valid))


class AlwaysNoopPolicy:
    name = "always_noop"

    def __init__(self, noop_idx: int) -> None:
        self.noop_idx = int(noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        return self.noop_idx


class GreedyDynamicsPolicy:
    name = "greedy_dyn_1"

    def __init__(self, dynamics: Any, *, n_genes: int, z_ref: np.ndarray, noop_idx: int) -> None:
        self.dynamics = dynamics
        self.n_genes = int(n_genes)
        self.z_ref = np.asarray(z_ref, dtype=np.float32)
        self.noop_idx = int(noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        import torch

        z = np.asarray(z, dtype=np.float32)
        valid = _valid_actions(mask)
        if len(valid) == 0:
            return self.noop_idx
        gene_actions = np.array([a for a in valid if 0 <= a < self.n_genes], dtype=np.int64)
        candidates: dict[int, np.ndarray] = {}
        if self.noop_idx in valid:
            candidates[self.noop_idx] = z
        if len(gene_actions):
            z_batch = np.repeat(z[None, :], len(gene_actions), axis=0)
            gene_idx = gene_actions + 1
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


class RidgeGreedyPolicy:
    name = "ridge_greedy"

    def __init__(self, W_z: np.ndarray, W_gene: np.ndarray, b: np.ndarray, *, z_ref: np.ndarray, noop_idx: int) -> None:
        self.W_z = np.asarray(W_z, dtype=np.float32)
        self.W_gene = np.asarray(W_gene, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        self.z_ref = np.asarray(z_ref, dtype=np.float32)
        self.noop_idx = int(noop_idx)

    @classmethod
    def from_npz(cls, path: str | Path, *, z_ref: np.ndarray, noop_idx: int) -> "RidgeGreedyPolicy":
        with np.load(path) as data:
            return cls(data["W_z"], data["W_gene"], data["b"], z_ref=z_ref, noop_idx=noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        z = np.asarray(z, dtype=np.float32)
        valid = _valid_actions(mask)
        candidates: dict[int, np.ndarray] = {}
        for action in valid:
            action = int(action)
            if action == self.noop_idx:
                candidates[action] = z
            elif 0 <= action < len(self.W_gene):
                delta = z @ self.W_z + self.W_gene[action] + self.b
                candidates[action] = z + delta
        if not candidates:
            return self.noop_idx
        return _argmin_distance(candidates, self.z_ref)


class MeanDeltaGreedyPolicy:
    name = "mean_delta_greedy"

    def __init__(self, mean_delta_table: np.ndarray, *, z_ref: np.ndarray, noop_idx: int) -> None:
        self.mean_delta_table = np.asarray(mean_delta_table, dtype=np.float32)
        self.z_ref = np.asarray(z_ref, dtype=np.float32)
        self.noop_idx = int(noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        z = np.asarray(z, dtype=np.float32)
        valid = _valid_actions(mask)
        candidates: dict[int, np.ndarray] = {}
        for action in valid:
            action = int(action)
            if action == self.noop_idx:
                candidates[action] = z
            elif 0 <= action < len(self.mean_delta_table):
                candidates[action] = z + self.mean_delta_table[action]
        if not candidates:
            return self.noop_idx
        return _argmin_distance(candidates, self.z_ref)


def _argmin_distance(candidates: dict[int, np.ndarray], z_ref: np.ndarray) -> int:
    best_action = min(
        candidates,
        key=lambda action: (
            float(np.linalg.norm(np.asarray(candidates[action], dtype=np.float32) - z_ref)),
            int(action),
        ),
    )
    return int(best_action)


class GreedyDynamicsBeamPolicy:
    """Receding-horizon multi-step greedy planner using the dynamics model.

    At every env step, this policy runs a depth-limited beam search using ``dynamics`` to find
    the gene sequence (length ≤ ``depth``) that minimises ``||z_final − z_ref||``, then executes
    only the **first** action of that plan (receding horizon — the search is re-run from the
    new state at the next call).

    ``noop`` is treated as a terminal "stay here" candidate at *every* depth: at depth=1 it
    mirrors :class:`GreedyDynamicsPolicy`. The beam search excludes any gene already used
    (per ``mask`` and per intra-plan repeats) from subsequent depths, matching the RL env's
    repeat-mask semantics.

    Parameters
    ----------
    dynamics
        Frozen ``PerturbationDynamicsModel`` (must return ``(z_next, mu, log_var)`` or a tensor).
    n_genes
        Action-space gene count (excludes the NO-OP).
    z_ref
        Reference centroid, shape ``(n_latent,)``.
    noop_idx
        Index of the NO-OP / terminate action.
    depth
        Maximum plan length. ``depth=1`` is equivalent to :class:`GreedyDynamicsPolicy` (up to
        floating-point determinism of the dynamics model).
    beam_width
        Top-K partial-plan retention per depth level. Default 20.
    """

    def __init__(
        self,
        dynamics: Any,
        *,
        n_genes: int,
        z_ref: np.ndarray,
        noop_idx: int,
        depth: int = 2,
        beam_width: int = 20,
        safety_tox_per_action: np.ndarray | None = None,
        safety_essential_per_action: np.ndarray | None = None,
        lambda_tox: float = 0.0,
        lambda_ce: float = 0.0,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be ≥ 1, got {depth}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be ≥ 1, got {beam_width}")
        self.dynamics    = dynamics
        self.n_genes     = int(n_genes)
        self.z_ref       = np.asarray(z_ref, dtype=np.float32)
        self.noop_idx    = int(noop_idx)
        self.depth       = int(depth)
        self.beam_width  = int(beam_width)
        # V3B Phase 2: safety-aware scoring. When λ_tox = λ_ce = 0, behavior is byte-identical
        # to V2's distance-only greedy. When non-zero, the plan score becomes
        # ``dist(z_next, z_ref) + λ_tox · cumulative_tox + λ_ce · cumulative_essential``.
        # Required so greedy_dyn_2_under_reward_C is a fair comparator for PPO_C
        # (V3B_BIOREALISTIC_CONTROL_OBJECTIVE_PLAN.md §5).
        self.safety_tox_per_action = (
            np.asarray(safety_tox_per_action, dtype=np.float32)
            if safety_tox_per_action is not None else None
        )
        self.safety_essential_per_action = (
            np.asarray(safety_essential_per_action, dtype=bool)
            if safety_essential_per_action is not None else None
        )
        if self.safety_tox_per_action is not None and self.safety_tox_per_action.shape != (self.n_genes,):
            raise ValueError(
                f"safety_tox_per_action shape mismatch: expected ({self.n_genes},), "
                f"got {self.safety_tox_per_action.shape}"
            )
        if self.safety_essential_per_action is not None and self.safety_essential_per_action.shape != (self.n_genes,):
            raise ValueError(
                f"safety_essential_per_action shape mismatch: expected ({self.n_genes},), "
                f"got {self.safety_essential_per_action.shape}"
            )
        self.lambda_tox = float(lambda_tox)
        self.lambda_ce = float(lambda_ce)
        self._safety_active = (
            self.safety_tox_per_action is not None
            and (self.lambda_tox > 0.0 or self.lambda_ce > 0.0)
        )
        suffix = "_safety" if self._safety_active else ""
        self.name = f"greedy_dyn_{self.depth}{suffix}"

    def _expand(self, z_batch: np.ndarray, gene_actions: np.ndarray) -> np.ndarray:
        """One batched dynamics call expanding (B, n_latent) by all genes in (B,)."""
        import torch
        with torch.no_grad():
            out = self.dynamics(
                torch.from_numpy(z_batch).float(),
                torch.from_numpy(gene_actions.astype(np.int64) + 1).long(),
            )
        z_next = out[0] if isinstance(out, tuple) else out
        if isinstance(z_next, torch.Tensor):
            return z_next.detach().cpu().numpy().astype(np.float32)
        return np.asarray(z_next, dtype=np.float32)

    def _action_safety_cost(self, gene_action_idx: int) -> tuple[float, int]:
        """Per-action safety contributions: ``(tox_raw(g), is_essential(g))``."""
        tox = 0.0
        ce = 0
        if self.safety_tox_per_action is not None and 0 <= gene_action_idx < self.n_genes:
            tox = float(self.safety_tox_per_action[gene_action_idx])
        if self.safety_essential_per_action is not None and 0 <= gene_action_idx < self.n_genes:
            ce = int(bool(self.safety_essential_per_action[gene_action_idx]))
        return tox, ce

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        z = np.asarray(z, dtype=np.float32)
        valid = _valid_actions(mask)
        if len(valid) == 0:
            return self.noop_idx

        # Genes still permitted by the env's mask (these are 0-indexed action ids in [0, n_genes))
        env_avail_genes = np.array(
            [a for a in valid if 0 <= a < self.n_genes], dtype=np.int64
        )
        noop_available = self.noop_idx in valid

        # Episode-cumulative safety so far (set by env when reward_mode=safety_aware).
        # When greedy is queried mid-episode (after some gene actions), info["tox_path"]
        # and info["common_essential_count"] reflect those prior actions. Greedy plans
        # ADDITIONS on top of that baseline. Default to 0 when env did not provide them.
        path_tox0 = float(info.get("tox_path", 0.0)) if self._safety_active else 0.0
        path_ce0 = int(info.get("common_essential_count", 0)) if self._safety_active else 0

        def _score(dist: float, plan_tox: float, plan_ce: int) -> float:
            return dist + self.lambda_tox * plan_tox + self.lambda_ce * plan_ce

        # Best plan tracking: (best_score, first_action). NOOP branch = "stay here".
        best_first_action: int = self.noop_idx
        best_score: float = float("inf")
        if noop_available:
            d_here = float(np.linalg.norm(z - self.z_ref))
            best_score = _score(d_here, path_tox0, path_ce0)
            best_first_action = self.noop_idx

        if len(env_avail_genes) == 0:
            return best_first_action

        # Beam entries: (z_current, first_action_taken, used_set, plan_tox_added, plan_ce_added).
        beam: list[tuple[np.ndarray, int | None, frozenset[int], float, int]] = [
            (z, None, frozenset(), 0.0, 0)
        ]

        for _step in range(self.depth):
            new_candidates: list[tuple[np.ndarray, int | None, frozenset[int], float, int]] = []
            batch_z: list[np.ndarray] = []
            batch_gene: list[int] = []
            batch_meta: list[tuple[int | None, frozenset[int], int, float, int]] = []
            for z_cur, first_action, used, plan_tox, plan_ce in beam:
                remaining = [int(g) for g in env_avail_genes if int(g) not in used]
                for g in remaining:
                    batch_z.append(z_cur)
                    batch_gene.append(g)
                    batch_meta.append((first_action, used, g, plan_tox, plan_ce))

            if not batch_z:
                break

            z_arr = np.stack(batch_z, axis=0)
            g_arr = np.asarray(batch_gene, dtype=np.int64)
            z_next_arr = self._expand(z_arr, g_arr)
            dists = np.linalg.norm(z_next_arr - self.z_ref[None, :], axis=1)

            for ((first_action, used, g, plan_tox, plan_ce), zn, d) in zip(
                batch_meta, z_next_arr, dists, strict=True
            ):
                fa = g if first_action is None else first_action
                tox_g, ce_g = self._action_safety_cost(g)
                new_plan_tox = plan_tox + tox_g
                new_plan_ce = plan_ce + ce_g
                new_candidates.append((zn, fa, used | {g}, new_plan_tox, new_plan_ce))

                score = _score(float(d), path_tox0 + new_plan_tox, path_ce0 + new_plan_ce)
                if score < best_score:
                    best_score = score
                    best_first_action = int(fa)  # type: ignore[arg-type]

            # Beam pruning is by the safety-augmented score so that the search front
            # is consistent with the global objective. When λ=0 the score reduces to
            # `dist` and behavior matches the V2 greedy.
            new_candidates.sort(key=lambda x: (
                _score(float(np.linalg.norm(x[0] - self.z_ref)), path_tox0 + x[3], path_ce0 + x[4]),
                x[1] if x[1] is not None else -1,
            ))
            beam = new_candidates[: self.beam_width]

        return int(best_first_action)
