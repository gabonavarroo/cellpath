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
        self.name = f"greedy_dyn_{self.depth}"

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

        # Best plan tracking: (final_distance, first_action). Includes the noop branch.
        best_first_action: int = self.noop_idx
        best_distance: float = float("inf")
        if noop_available:
            best_distance = float(np.linalg.norm(z - self.z_ref))
            best_first_action = self.noop_idx

        if len(env_avail_genes) == 0:
            return best_first_action

        # Beam entries: (z_current, first_action_taken, used_set, depth_so_far, distance).
        # Initial beam = singleton (z, first=None, used=∅, d=0, dist=∞)
        beam: list[tuple[np.ndarray, int | None, frozenset[int], float]] = [
            (z, None, frozenset(), float("inf"))
        ]

        for _step in range(self.depth):
            # Build the candidate batch from the current beam: for each beam entry, expand by
            # every still-available gene (env mask ∩ not-yet-used-in-plan).
            new_candidates: list[tuple[np.ndarray, int | None, frozenset[int], float]] = []
            batch_z: list[np.ndarray] = []
            batch_gene: list[int] = []
            batch_meta: list[tuple[int | None, frozenset[int], int]] = []
            for z_cur, first_action, used, _ in beam:
                # Remaining genes = env_avail_genes minus what this plan has used so far.
                # (The env's mask already excludes prior-step env-side uses; intra-plan repeats
                # are caught by `used`.)
                remaining = [int(g) for g in env_avail_genes if int(g) not in used]
                for g in remaining:
                    batch_z.append(z_cur)
                    batch_gene.append(g)
                    batch_meta.append((first_action, used, g))

            if not batch_z:
                break

            # One batched dynamics call for the whole layer.
            z_arr = np.stack(batch_z, axis=0)
            g_arr = np.asarray(batch_gene, dtype=np.int64)
            z_next_arr = self._expand(z_arr, g_arr)
            dists = np.linalg.norm(z_next_arr - self.z_ref[None, :], axis=1)

            for idx, ((first_action, used, g), zn, d) in enumerate(
                zip(batch_meta, z_next_arr, dists, strict=True)
            ):
                fa = g if first_action is None else first_action
                new_candidates.append((zn, fa, used | {g}, float(d)))

            # Update best across this layer (any depth ≥ 1 is a valid termination point).
            for _, fa, _, d in new_candidates:
                if d < best_distance:
                    best_distance = float(d)
                    best_first_action = int(fa)  # type: ignore[arg-type]

            # Keep the top-`beam_width` partial plans (sorted by distance).
            new_candidates.sort(key=lambda x: (x[3], x[1] if x[1] is not None else -1))
            beam = new_candidates[: self.beam_width]

        return int(best_first_action)
