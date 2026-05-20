"""Residual heteroscedastic dynamics MLP.

Owner: Agent B. See ARCHITECTURE.md Concept 3.

Architecture
------------
::

    z (B, 32)   ─┐
                 ├─►  cat  ──► input_proj ──► residual blocks ──► (mu, log_var)
    gene_idx (B,)┘                                                  each (B, 32)

    z_next = z + mu                                  (residual head)
    loss   = heteroscedastic Gaussian NLL on Δz       (training)
    σ²     = exp(log_var.clamp(min, max))             (clamped output)

The residual head matches the small ``||Δz|| ≪ ||z||`` magnitude observed in Norman; the
heteroscedastic head provides per-step uncertainty used both for the validation gate and as an
optional exploration penalty in the RL reward.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class _ResidualBlock(nn.Module):
    """Single residual block: LayerNorm → Linear → SiLU → Dropout → Linear + skip."""

    def __init__(self, n_hidden: int, dropout: float = 0.1, use_layernorm: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_hidden) if use_layernorm else nn.Identity()
        self.linear1 = nn.Linear(n_hidden, n_hidden)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


class PerturbationDynamicsModel(nn.Module):
    """f_θ(z, gene_idx) → (z_next, μ, log σ²) with residual head + heteroscedastic outputs.

    Parameters
    ----------
    n_latent
        Latent dim (matches ``cfg.vae.n_latent``, default 32).
    n_genes
        Number of perturbable genes (matches ``gene_vocab["n_genes"]``). The embedding table is
        sized ``n_genes + 1`` so index 0 is a reserved placeholder (ctrl) and indices 1..n_genes
        are the actual gene embeddings, matching the 1-indexed ``gene_idx`` in Contract 2 pairs.
    d_emb
        Gene-embedding dimension (default 64).
    n_hidden
        MLP hidden width (default 256).
    n_layers
        Number of residual blocks in the trunk (default 3).
    log_var_min
        Lower clamp on predicted log σ² (default -5.0).
    log_var_max
        Upper clamp (default 3.0).
    use_state_linear_skip
        If ``True``, adds an extra ``nn.Linear(n_latent, n_latent)`` whose output is added
        into ``mu``: ``mu += state_linear(z)``. Gene-independent; intended to give the model
        an explicit linear-in-z scaffold that can generalize across genes (helps OOD when
        the perturbation effect has a z-dependent linear component the nonlinear trunk is
        missing). Default ``False`` — preserves baseline behavior.
    use_gene_delta_bias
        If ``True``, adds an ``nn.Embedding(n_genes + 1, n_latent)`` whose row for the input
        gene is added into ``mu``: ``mu += gene_delta(gene_idx)``. This is a per-gene additive
        offset, functionally equivalent to the "per-gene mean Δ" baseline absorbed into the
        model. Row 0 is initialized to all-zeros at construction so the ctrl-placeholder
        index is a true no-op at init. Default ``False``.
    use_residual_over_ridge
        If ``True``, the dynamics MLP predicts a *residual* on top of a frozen ridge baseline
        ``μ = ridge(z, gene) + mlp_residual(z, gene_emb)``. Three buffers are registered:
        ``ridge_W_z`` ``(n_latent, n_latent)``, ``ridge_W_gene`` ``(n_genes + 1, n_latent)``
        (row 0 = zeros for the ctrl placeholder), and ``ridge_b`` ``(n_latent,)``. The trainer
        is responsible for fitting the ridge baseline on train pairs once and assigning the
        coefficients to these buffers before optimisation starts. The buffers are saved as
        part of ``state_dict`` so reload is automatic. The ``head_mu`` weight + bias are
        zero-initialised when this flag is enabled so that the MLP begins at the ridge
        prediction. Mutually exclusive with ``use_state_linear_skip`` — both raise
        ``ValueError`` if set together. Default ``False``.

    Examples
    --------
    >>> model = PerturbationDynamicsModel(n_latent=32, n_genes=100)
    >>> z = torch.randn(8, 32)
    >>> g = torch.randint(1, 101, (8,))  # 1-indexed gene indices
    >>> z_next, mu, log_var = model(z, g)
    >>> z_next.shape, mu.shape, log_var.shape
    (torch.Size([8, 32]), torch.Size([8, 32]), torch.Size([8, 32]))
    """

    def __init__(
        self,
        n_latent: int = 32,
        n_genes: int = 100,
        d_emb: int = 64,
        n_hidden: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "silu",
        log_var_min: float = -5.0,
        log_var_max: float = 3.0,
        log_var_init_bias: float = -2.0,
        use_layernorm: bool = True,
        use_state_linear_skip: bool = False,
        use_gene_delta_bias: bool = False,
        use_residual_over_ridge: bool = False,
    ) -> None:
        super().__init__()
        if bool(use_state_linear_skip) and bool(use_residual_over_ridge):
            raise ValueError(
                "use_state_linear_skip and use_residual_over_ridge are mutually exclusive: "
                "the ridge baseline already includes a linear-in-z term."
            )
        self.n_latent = n_latent
        self.n_genes = n_genes
        self.d_emb = d_emb
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.use_state_linear_skip = bool(use_state_linear_skip)
        self.use_gene_delta_bias = bool(use_gene_delta_bias)
        self.use_residual_over_ridge = bool(use_residual_over_ridge)

        _act_map = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in _act_map:
            raise ValueError(f"Unknown activation {activation!r}. Choose from {list(_act_map)}.")
        act_cls = _act_map[activation]

        # Gene embedding: index 0 = ctrl placeholder (unused in training);
        # indices 1..n_genes correspond to the 1-indexed gene_idx in Contract 2 pairs.
        self.gene_embedding = nn.Embedding(n_genes + 1, d_emb)

        # Input projection: (n_latent + d_emb) → n_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(n_latent + d_emb, n_hidden),
            act_cls(),
        )

        # Residual trunk: n_layers blocks at n_hidden
        self.trunk = nn.Sequential(*[
            _ResidualBlock(n_hidden, dropout=dropout, use_layernorm=use_layernorm)
            for _ in range(n_layers)
        ])

        # Output heads
        self.head_mu = nn.Linear(n_hidden, n_latent)
        self.head_log_var = nn.Linear(n_hidden, n_latent)

        # Initialise log_var bias to avoid early overconfidence
        nn.init.constant_(self.head_log_var.bias, log_var_init_bias)

        # Optional gene-independent linear skip on z (off by default).
        self.state_linear: nn.Linear | None = (
            nn.Linear(n_latent, n_latent) if self.use_state_linear_skip else None
        )

        # Optional per-gene additive bias on mu (off by default).
        # Index 0 is the ctrl placeholder; zero it out so it contributes nothing at init.
        self.gene_delta: nn.Embedding | None = (
            nn.Embedding(n_genes + 1, n_latent) if self.use_gene_delta_bias else None
        )
        if self.gene_delta is not None:
            with torch.no_grad():
                self.gene_delta.weight[0].zero_()

        # Optional residual-over-ridge: register three buffers and zero-init head_mu so
        # the model starts at the ridge prediction. The trainer overwrites the buffers
        # before optimisation; this construction sets them to zeros so the model is
        # well-defined even before fit-and-assign.
        if self.use_residual_over_ridge:
            self.register_buffer("ridge_W_z",    torch.zeros(n_latent, n_latent))
            self.register_buffer("ridge_W_gene", torch.zeros(n_genes + 1, n_latent))
            self.register_buffer("ridge_b",      torch.zeros(n_latent))
            # Zero-init the MLP mu head so the MLP starts as the identity-on-ridge residual.
            with torch.no_grad():
                self.head_mu.weight.zero_()
                self.head_mu.bias.zero_()

    def forward(
        self,
        z: torch.Tensor,
        gene_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict (z_next, μ, log σ²) for a batch of (latent, gene_idx) pairs.

        Parameters
        ----------
        z
            Shape ``(B, n_latent)``.
        gene_idx
            Shape ``(B,)`` int. Range ``[1, n_genes]`` during training (1-indexed); the
            environment converts 0-indexed RL actions to 1-indexed gene_idx before calling.

        Returns
        -------
        z_next : torch.Tensor
            Shape ``(B, n_latent)``. Equals ``z + μ`` (residual).
        mu : torch.Tensor
            Predicted Δz mean, shape ``(B, n_latent)``.
        log_var : torch.Tensor
            Predicted per-dim log σ² (clamped to ``[log_var_min, log_var_max]``),
            shape ``(B, n_latent)``.
        """
        emb = self.gene_embedding(gene_idx)            # (B, d_emb)
        h = torch.cat([z, emb], dim=-1)                # (B, n_latent + d_emb)
        h = self.input_proj(h)                         # (B, n_hidden)
        h = self.trunk(h)                              # (B, n_hidden)
        mu = self.head_mu(h)                           # (B, n_latent)
        if self.state_linear is not None:
            mu = mu + self.state_linear(z)             # gene-independent linear skip
        if self.gene_delta is not None:
            mu = mu + self.gene_delta(gene_idx)        # per-gene additive offset
        if self.use_residual_over_ridge:
            # ridge_pred = z @ W_z + W_gene[gene_idx] + b
            ridge_pred = z @ self.ridge_W_z + self.ridge_W_gene[gene_idx] + self.ridge_b
            mu = mu + ridge_pred                       # MLP learns the ridge residual
        log_var = self.head_log_var(h).clamp(
            self.log_var_min, self.log_var_max
        )                                              # (B, n_latent)
        z_next = z + mu                                # residual connection
        return z_next, mu, log_var


def fit_ridge_baseline_from_pairs(
    z_ctrl: "np.ndarray",
    gene_idx: "np.ndarray",
    delta: "np.ndarray",
    n_genes: int,
    *,
    alpha: float = 1.0,
    random_state: int = 42,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Fit the gate's ridge baseline and split it into RoR buffer-shaped numpy arrays.

    Delegates to :func:`src.analysis.metrics._fit_ridge_baseline` (single source of truth,
    CLAUDE.md rule 4) and unpacks ``ridge.coef_`` + ``ridge.intercept_`` into three arrays
    sized for the ``PerturbationDynamicsModel`` buffers when ``use_residual_over_ridge=True``.

    The fitted ridge predicts ``Δz = X @ coef_.T + intercept_`` where ``X = [z_ctrl, one_hot]``.
    We reshape into:
      * ``W_z``    ``(n_latent, n_latent)``     — so that ``z @ W_z = X[:, :n_latent] @ coef_[:, :n_latent].T``.
      * ``W_gene`` ``(n_genes + 1, n_latent)``  — row 0 = zeros (ctrl placeholder, matches
        Contract-2 gene_idx=0 convention); rows 1..n_genes carry the per-gene one-hot weight.
      * ``b``      ``(n_latent,)``              — the ridge intercept.

    Parameters
    ----------
    z_ctrl, delta
        Each ``(M, n_latent)``. ``delta = z_pert - z_ctrl``.
    gene_idx
        Shape ``(M,)`` int, 1-indexed per Contract 2.
    n_genes
        One-hot width. Must match the dynamics model's ``n_genes``.
    alpha, random_state
        Forwarded to the ridge fit (defaults match the gate baseline at alpha=1.0).

    Returns
    -------
    W_z : np.ndarray
        Shape ``(n_latent, n_latent)`` — ready for ``z @ W_z``.
    W_gene : np.ndarray
        Shape ``(n_genes + 1, n_latent)`` — embedding-style lookup; row 0 = zeros.
    b : np.ndarray
        Shape ``(n_latent,)`` — the ridge intercept.
    """
    import numpy as np  # local import keeps the module's numpy dependency private to this fn
    from src.analysis.metrics import _fit_ridge_baseline

    ridge = _fit_ridge_baseline(z_ctrl, gene_idx, delta, n_genes,
                                alpha=alpha, random_state=random_state)
    coef = np.asarray(ridge.coef_, dtype=np.float32)        # (n_latent, n_latent + n_genes)
    intercept = np.asarray(ridge.intercept_, dtype=np.float32)  # (n_latent,)
    n_latent = int(coef.shape[0])
    if coef.shape[1] != n_latent + n_genes:
        raise ValueError(
            f"Ridge coefficient width {coef.shape[1]} != n_latent + n_genes "
            f"({n_latent + n_genes}). Did n_genes change between fit and assignment?"
        )

    # z-block: ridge predicts z @ coef[:, :n_latent].T. We want z @ W_z, so W_z = coef[:, :n_latent].T.
    W_z = coef[:, :n_latent].T.astype(np.float32, copy=True)              # (n_latent, n_latent)
    # gene-block: one_hot @ coef[:, n_latent:].T → picks one column per row. We store as
    # (n_genes + 1, n_latent) with row 0 = zeros (ctrl placeholder), rows 1..n_genes from coef.
    W_gene = np.zeros((n_genes + 1, n_latent), dtype=np.float32)
    W_gene[1:] = coef[:, n_latent:].T.astype(np.float32, copy=True)       # (n_genes, n_latent)
    b = intercept.astype(np.float32, copy=True)                            # (n_latent,)
    return W_z, W_gene, b


def heteroscedastic_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target_delta: torch.Tensor,
    log_var_reg: float = 0.01,
) -> torch.Tensor:
    """Per-dim heteroscedastic Gaussian NLL on Δz (with optional log_var L2 penalty).

    NLL_i = 0.5 * exp(-log_var_i) * (target_delta_i - mu_i)^2 + 0.5 * log_var_i

    Parameters
    ----------
    mu
        Predicted mean Δz, shape ``(B, n_latent)``.
    log_var
        Predicted log σ², shape ``(B, n_latent)``.
    target_delta
        Empirical Δz = z_pert - z_ctrl, shape ``(B, n_latent)``.
    log_var_reg
        L2 penalty on ``log_var`` to discourage extremes. Default 0.01.

    Returns
    -------
    torch.Tensor
        Scalar mean NLL across the batch and latent dims.
    """
    precision = torch.exp(-log_var)
    per_element = 0.5 * precision * (target_delta - mu).pow(2) + 0.5 * log_var
    loss = per_element.mean()
    if log_var_reg > 0.0:
        loss = loss + log_var_reg * log_var.pow(2).mean()
    return loss


def _alignment_cosine(
    mu: torch.Tensor,
    z: torch.Tensor,
    z_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-row cos(μ, z_ref − z) with safe denominators.

    Returns ``(B,)`` cosine in [−1, 1]. When ``‖μ‖ ≈ 0`` the cosine is undefined; we
    return 0 in that row (treated as "no contractive pressure"), which keeps the
    excessive-alignment penalty inert on null-magnitude predictions.
    """
    target_dir = z_ref.unsqueeze(0) - z                       # (B, D)
    target_norm = target_dir.norm(dim=-1)                      # (B,)
    mu_norm = mu.norm(dim=-1)                                  # (B,)
    denom = target_norm * mu_norm + eps                        # (B,)
    dot = (mu * target_dir).sum(dim=-1)                        # (B,)
    cos = dot / denom
    # Zero out rows where ‖μ‖ is effectively zero — cos is undefined there.
    safe_mask = (mu_norm > eps).to(cos.dtype)
    return cos * safe_mask


def excessive_alignment_penalty(
    mu: torch.Tensor,
    z: torch.Tensor,
    z_ref: torch.Tensor,
    tau: float = 0.80,
) -> torch.Tensor:
    """V3C Phase 2 — penalise per-(z, g) cos(μ, z_ref − z) above ``tau``.

    L_ea = mean over rows of relu(α − τ)² where α = cos(μ, z_ref − z).

    Used as a soft cap on universal-attractor structure: pairs whose predicted Δz is
    "too" aligned with the direction toward z_ref accumulate quadratic penalty above τ.
    Pairs at or below τ pay zero penalty (one-sided ReLU). Zero-magnitude μ rows are
    treated as α = 0 (see :func:`_alignment_cosine`).
    """
    cos = _alignment_cosine(mu, z, z_ref)                     # (B,)
    excess = torch.relu(cos - float(tau))                      # (B,)
    return (excess * excess).mean()


def universal_attractor_penalty(
    mu: torch.Tensor,
    z: torch.Tensor,
    z_ref: torch.Tensor,
    gene_idx: torch.Tensor,
    tau: float = 0.80,
) -> torch.Tensor:
    """V3C Phase 2 — penalise the batch-max per-gene mean alignment above ``tau``.

    For each gene id ``g_i`` appearing in ``gene_idx``, compute the mean of
    ``α(z, g)`` over rows with that gene. Take the max across genes. If it exceeds
    τ, return ``relu(max_g ᾱ(g) − τ)²``.

    This targets the audit's ``UNIVERSAL_ATTRACTOR_GENE`` flag: a single gene whose
    mean alignment across all states approaches 1.0. The max (rather than mean) makes
    the gradient focal on the dominant attractor.

    Implementation note: we compute per-gene means via ``index_add_`` for differentiability
    on small batches; this is O(B · D) and runs on the training device.
    """
    cos = _alignment_cosine(mu, z, z_ref)                     # (B,)
    gene_idx_long = gene_idx.to(dtype=torch.long)
    if gene_idx_long.numel() == 0:
        return torch.zeros((), device=cos.device, dtype=cos.dtype)
    # Aggregate per-gene sum and count via scatter-style index_add_.
    n_genes_seen = int(gene_idx_long.max().item()) + 1
    sums = torch.zeros(n_genes_seen, device=cos.device, dtype=cos.dtype)
    counts = torch.zeros(n_genes_seen, device=cos.device, dtype=cos.dtype)
    sums = sums.index_add(0, gene_idx_long, cos)
    counts = counts.index_add(0, gene_idx_long, torch.ones_like(cos))
    per_gene_mean = sums / counts.clamp(min=1.0)              # (n_genes_seen,)
    # Mask out genes that did not appear in this batch (count = 0)
    present = (counts > 0)
    if not present.any():
        return torch.zeros((), device=cos.device, dtype=cos.dtype)
    max_mean = per_gene_mean[present].max()
    excess = torch.relu(max_mean - float(tau))
    return excess * excess


def action_diversity_penalty(
    mu: torch.Tensor,
    tau_min: float = 0.0,
) -> torch.Tensor:
    """V3C Phase 2 — encourage across-batch variance of μ to stay ≥ ``tau_min``.

    L_ad = relu(τ_min − mean(σ²(μ)))² where σ² is the per-dim variance of μ across the
    batch. Default ``tau_min = 0`` keeps the term inert. Used (with λ_ad > 0 and a
    nonzero tau_min) to discourage near-constant Δz predictions.
    """
    if mu.numel() == 0:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    if mu.shape[0] < 2:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    var = mu.var(dim=0, unbiased=False)                       # (D,)
    mean_var = var.mean()
    deficit = torch.relu(float(tau_min) - mean_var)
    return deficit * deficit


def composition_loss(
    model: PerturbationDynamicsModel,
    z_ctrl: torch.Tensor,
    gene_idx_a: torch.Tensor,
    gene_idx_b: torch.Tensor,
    z_pert_ab: torch.Tensor,
) -> torch.Tensor:
    """Sequential-composition loss for combinatorial perturbations.

    Predicts z after applying ``gene_idx_a`` then ``gene_idx_b`` to ``z_ctrl``, compares to
    the empirical ``z_pert_ab`` of the dual-guide cells. This is the main signal that the
    dynamics model has learned *composable* perturbation effects.

    Parameters
    ----------
    model
        The dynamics model.
    z_ctrl
        Shape ``(B, n_latent)``.
    gene_idx_a, gene_idx_b
        Each shape ``(B,)``.
    z_pert_ab
        Shape ``(B, n_latent)``.

    Returns
    -------
    torch.Tensor
        Scalar MSE between predicted and empirical post-double-perturbation latent.

    Notes
    -----
    ``gene_idx_a`` and ``gene_idx_b`` must be ``LongTensor`` (1-indexed per Contract 2).
    Gradients flow through both forward passes; do **not** detach ``z_next_a`` so that
    composability is the actual training signal.
    """
    z_next_a, _, _ = model(z_ctrl, gene_idx_a)
    z_next_ab, _, _ = model(z_next_a, gene_idx_b)
    return F.mse_loss(z_next_ab, z_pert_ab)
