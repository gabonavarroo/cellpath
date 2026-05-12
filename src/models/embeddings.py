"""Gene / direction embedding utilities for the dynamics model.

Owner: Agent B.

The base dynamics model uses a plain ``nn.Embedding(n_genes + 1, d_emb)``. This module
provides drop-in alternatives that may improve OOD generalization across genes:

- :class:`FactorizedGeneEmbedding` — factorizes the gene embedding via a small set of latent
  factors (e.g. function class) so that genes with similar covariates share representation.
  Used only if OOD R² is needed and external gene side-information is loaded.
- Helper for the NO-OP action's embedding lookup (always zero vector, never learned).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FactorizedGeneEmbedding(nn.Module):
    """Factorized embedding: ``gene → low-rank latent → d_emb``.

    Useful only when external gene-feature side information is available (not in v1). The
    base ``nn.Embedding`` is the default for Norman alone.

    Parameters
    ----------
    n_genes
        Number of perturbable genes.
    d_emb
        Output embedding dim.
    rank
        Low-rank factor dim (default 16). Lower = more sharing across genes.
    """

    def __init__(self, n_genes: int, d_emb: int = 64, rank: int = 16) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.d_emb = d_emb
        self.rank = rank

    def forward(self, gene_idx: torch.Tensor) -> torch.Tensor:
        """Embed a batch of gene indices.

        Parameters
        ----------
        gene_idx
            Shape ``(B,)`` int.

        Returns
        -------
        torch.Tensor
            Shape ``(B, d_emb)``.

        Raises
        ------
        NotImplementedError
            Agent B: implement the factor → embedding map. Stub for now; not used in v1.
        """
        raise NotImplementedError(
            "Agent B: factorized embedding is future-work. Not required for v1 — keep stub."
        )


def noop_embedding(d_emb: int, device: Any = "cpu") -> torch.Tensor:
    """Return the (frozen, zero) embedding for the NO-OP action.

    NO-OP is the final index in the action space; it terminates the episode without applying
    any perturbation. The embedding is a learned-but-frozen zero vector to avoid leaking
    signal into the trunk.

    Parameters
    ----------
    d_emb
        Embedding dim.
    device
        Tensor device.

    Returns
    -------
    torch.Tensor
        Shape ``(d_emb,)`` zeros.

    Raises
    ------
    NotImplementedError
        Agent B: trivially return ``torch.zeros(d_emb, device=device)``.
    """
    raise NotImplementedError("Agent B: torch.zeros(d_emb, device=device). Trivial helper.")
