"""scripts/train_dynamics.py — dynamics-model training entry point.

Owner: Agent B. See AGENTS.md §2 Phase 1 + Phase 2.

Usage
-----
::

    python scripts/train_dynamics.py --config-name default
    python scripts/train_dynamics.py dynamics.lambda_combo=0.0
    python scripts/train_dynamics.py +dry_run=true          # validate config + paths, exit 0
    python scripts/train_dynamics.py +force=true            # retrain even if model.pt exists
    python scripts/train_dynamics.py --multirun dynamics.n_layers=2,3,4

Workflow
--------
1.  Compose Hydra config.
2.  ``src.utils.seeding.set_seed(cfg.seed)``.
3.  Log device summary.
4.  Check for an existing checkpoint at ``cfg.paths.dynamics_model``; if present and
    ``force=false``, skip training (prints a warning if ``n_genes`` changed vs saved config).
5.  Load Contract 2 pairs from ``cfg.paths.pairs_{train,val,combo}``. If the train file is
    missing, fall back to :func:`src.data.perturbation_pairs.generate_mock_pairs`.
6.  Infer ``n_genes`` from ``gene_vocab.json`` (real Norman path) or ``metadata.json`` (mock).
    Cross-check ``n_latent`` against ``cfg.vae.n_latent``.
7.  Build :class:`src.models.dynamics.PerturbationDynamicsModel` from ``cfg.dynamics``.
8.  Build ``DataLoader``\\s; train with AdamW + optional cosine LR scheduler.
    Loss = heteroscedastic NLL + ``lambda_combo`` × composition MSE (when combo pairs exist).
    Early stopping on val NLL; best checkpoint saved atomically.
9.  Write ``artifacts/dynamics/model.pt`` (best state dict) + ``config.json``.

Phase 2 hook (not yet active)
------------------------------
After step 9 the validation gate (:func:`src.analysis.metrics.dynamics_validation_gate`)
will be called to write ``gate.json``, ``val_metrics.json``, and ``ood_metrics.json``.
RL training refuses to start without a passing ``gate.json`` (CLAUDE.md sacred rule #9).
The hook is stubbed out at the bottom of ``main()`` with a clear comment.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Generator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _iter_infinite(loader: DataLoader) -> Generator:
    """Yield batches from *loader* indefinitely, restarting after each epoch."""
    while True:
        yield from loader


def _npz_to_tensors(path: Path, keys: list[str]) -> tuple[torch.Tensor, ...]:
    """Load selected arrays from a ``.npz`` file as CPU tensors.

    Integer arrays (int8/16/32) are upcast to ``int64`` as required by
    ``nn.Embedding``. Float arrays are kept as-is (float32).
    """
    data = np.load(path)
    tensors: list[torch.Tensor] = []
    for k in keys:
        arr = data[k].copy()
        t = torch.from_numpy(arr)
        if t.dtype in (torch.int8, torch.int16, torch.int32):
            t = t.long()
        tensors.append(t)
    return tuple(tensors)


def _build_loaders(
    train_path: Path,
    val_path: Path,
    combo_path: Path | None,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Return (train_loader, val_loader, combo_loader | None)."""

    def _loader(path: Path, keys: list[str], shuffle: bool) -> DataLoader:
        tensors = _npz_to_tensors(path, keys)
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    train_loader = _loader(train_path, ["z_ctrl", "gene_idx", "z_pert"], shuffle=True)
    val_loader   = _loader(val_path,   ["z_ctrl", "gene_idx", "z_pert"], shuffle=False)
    combo_loader = (
        _loader(
            combo_path,
            ["z_ctrl", "gene_idx_a", "gene_idx_b", "z_pert_ab"],
            shuffle=True,
        )
        if (combo_path is not None and combo_path.exists())
        else None
    )
    return train_loader, val_loader, combo_loader


def _resolve_n_genes(cfg: DictConfig, train_path: Path) -> int:
    """Infer ``n_genes`` with a three-level fallback chain.

    1. ``gene_vocab.json`` — written by Agent A after VAE training (real Norman path).
    2. ``pairs/metadata.json`` — written by ``generate_mock_pairs`` (mock path).
    3. ``int(train_pairs['gene_idx'].max())`` — last-resort defensive inference.
    """
    vocab_path = Path(cfg.paths.vae_gene_vocab_json)
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
        n = int(vocab["n_genes"])
        log.info("n_genes=%d  [source: gene_vocab.json]", n)
        return n

    meta_path = Path(cfg.paths.pairs_metadata)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "n_genes" in meta:
            n = int(meta["n_genes"])
            log.info("n_genes=%d  [source: pairs/metadata.json]", n)
            return n

    data = np.load(train_path)
    n = int(data["gene_idx"].max())
    log.warning(
        "n_genes=%d inferred from gene_idx.max() — neither gene_vocab.json nor "
        "pairs/metadata.json found; verify this matches the intended action-space size.",
        n,
    )
    return n


def _save_state_dict(model: nn.Module, path: Path) -> None:
    """Atomically write ``model.state_dict()`` to *path* via a tmp-then-rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(str(tmp), str(path))
    log.debug("Checkpoint saved → %s", path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra-driven main. See module docstring for the 9-step workflow."""
    # Deferred src.* imports so Hydra config composition never touches the model.
    from src.data.perturbation_pairs import generate_mock_pairs
    from src.models.dynamics import (
        PerturbationDynamicsModel,
        composition_loss,
        heteroscedastic_nll,
    )
    from src.utils.device import device_summary, get_device
    from src.utils.seeding import set_seed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    set_seed(int(cfg.seed))
    device = get_device()
    log.info(device_summary())

    # ------------------------------------------------------------------
    # Dry-run: validate config resolution, then exit
    # ------------------------------------------------------------------
    if cfg.get("dry_run", False):
        log.info(
            "DRY RUN — dynamics config: d_emb=%d  n_hidden=%d  n_layers=%d  "
            "lr=%.1e  lambda_combo=%.3f",
            cfg.dynamics.d_emb,
            cfg.dynamics.n_hidden,
            cfg.dynamics.n_layers,
            cfg.dynamics.lr,
            cfg.dynamics.lambda_combo,
        )
        log.info("DRY RUN — output dir: %s", cfg.paths.dynamics_dir)
        return 0

    # ------------------------------------------------------------------
    # Step 4 (pair loading) — mock fallback if real pairs are absent
    # ------------------------------------------------------------------
    train_path  = Path(cfg.paths.pairs_train)
    val_path    = Path(cfg.paths.pairs_val)
    combo_path  = Path(cfg.paths.pairs_combo)
    pairs_dir   = Path(cfg.paths.pairs_dir)

    if not train_path.exists():
        log.info(
            "Pair files not found at %s — generating mock pairs "
            "(n=10_000, n_genes=100, n_latent=%d, seed=%d).",
            pairs_dir,
            int(cfg.vae.n_latent),
            int(cfg.seed),
        )
        generate_mock_pairs(
            n=10_000,
            n_genes=100,
            n_latent=int(cfg.vae.n_latent),
            n_combo=1_000,
            seed=int(cfg.seed),
            out_dir=pairs_dir,
        )
        log.info("Mock pairs written to %s", pairs_dir)

    # ------------------------------------------------------------------
    # Steps 5+6 — infer n_genes; cross-check n_latent vs config
    # ------------------------------------------------------------------
    n_genes = _resolve_n_genes(cfg, train_path)

    train_np       = np.load(train_path)
    n_latent_data  = int(train_np["z_ctrl"].shape[1])
    n_latent_cfg   = int(cfg.vae.n_latent)
    if n_latent_data != n_latent_cfg:
        raise ValueError(
            f"Contract violation: train_pairs z_ctrl has n_latent={n_latent_data} "
            f"but cfg.vae.n_latent={n_latent_cfg}. "
            "Delete artifacts/pairs/ or update the config to match."
        )
    n_latent = n_latent_data
    log.info("n_latent=%d (verified against cfg.vae.n_latent)", n_latent)

    # ------------------------------------------------------------------
    # Checkpoint skip — reuse existing model.pt unless force=true
    # ------------------------------------------------------------------
    dynamics_dir = Path(cfg.paths.dynamics_dir)
    dynamics_dir.mkdir(parents=True, exist_ok=True)
    model_path   = Path(cfg.paths.dynamics_model)

    if model_path.exists() and not cfg.get("force", False):
        saved_cfg_path = Path(cfg.paths.dynamics_config)
        mismatches: list[str] = []
        if saved_cfg_path.exists():
            saved = json.loads(saved_cfg_path.read_text())
            if saved.get("n_genes") != n_genes:
                mismatches.append(
                    f"n_genes: saved={saved.get('n_genes')}  current={n_genes}"
                )
            if saved.get("n_latent") != n_latent:
                mismatches.append(
                    f"n_latent: saved={saved.get('n_latent')}  current={n_latent}"
                )
            meta_path = Path(cfg.paths.pairs_metadata)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                current_source = "mock" if meta.get("pairing_method") == "mock" else "real"
                if saved.get("trained_on") != current_source:
                    mismatches.append(
                        f"trained_on: saved={saved.get('trained_on')}  current={current_source}"
                    )
        if mismatches:
            raise RuntimeError(
                f"Checkpoint at {model_path} is incompatible with current data:\n"
                + "\n".join(f"  • {m}" for m in mismatches)
                + "\nRerun with +force=true to retrain from scratch."
            )
        log.info(
            "Checkpoint found at %s — config matches, skipping training. "
            "Pass +force=true to retrain.",
            model_path,
        )
        return 0

    # ------------------------------------------------------------------
    # Step 7 — build model
    # ------------------------------------------------------------------
    model = PerturbationDynamicsModel(
        n_latent       = n_latent,
        n_genes        = n_genes,
        d_emb          = int(cfg.dynamics.d_emb),
        n_hidden       = int(cfg.dynamics.n_hidden),
        n_layers       = int(cfg.dynamics.n_layers),
        dropout        = float(cfg.dynamics.dropout),
        activation     = str(cfg.dynamics.activation),
        log_var_min    = float(cfg.dynamics.log_var_min),
        log_var_max    = float(cfg.dynamics.log_var_max),
        log_var_init_bias = float(cfg.dynamics.log_var_init_bias),
        use_layernorm  = bool(cfg.dynamics.use_layernorm),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "PerturbationDynamicsModel built | n_latent=%d | n_genes=%d | "
        "params=%d | device=%s",
        n_latent, n_genes, n_params, device,
    )

    # ------------------------------------------------------------------
    # Step 8a — DataLoaders
    # ------------------------------------------------------------------
    pin_memory = (device.type == "cuda")
    train_loader, val_loader, combo_loader = _build_loaders(
        train_path  = train_path,
        val_path    = val_path,
        combo_path  = combo_path,
        batch_size  = int(cfg.dynamics.batch_size),
        num_workers = int(cfg.dynamics.num_workers),
        pin_memory  = pin_memory,
    )
    log.info(
        "DataLoaders ready | train=%d batches | val=%d batches | combo=%s",
        len(train_loader),
        len(val_loader),
        f"{len(combo_loader)} batches" if combo_loader is not None else "none",
    )

    # ------------------------------------------------------------------
    # Step 8b — optimizer + LR scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = float(cfg.dynamics.lr),
        weight_decay = float(cfg.dynamics.weight_decay),
    )

    scheduler_name = str(cfg.dynamics.scheduler).lower()
    if scheduler_name == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max   = int(cfg.dynamics.max_epochs),
                eta_min = float(cfg.dynamics.lr) * 1e-2,
            )
        )
        log.info(
            "CosineAnnealingLR: %.2e → %.2e over %d epochs",
            float(cfg.dynamics.lr),
            float(cfg.dynamics.lr) * 1e-2,
            int(cfg.dynamics.max_epochs),
        )
    else:
        scheduler = None

    # ------------------------------------------------------------------
    # Step 8c — training loop
    # ------------------------------------------------------------------
    lambda_combo      = float(cfg.dynamics.lambda_combo)
    lambda_lv_reg     = float(cfg.dynamics.lambda_log_var_reg)
    max_epochs        = int(cfg.dynamics.max_epochs)
    patience          = int(cfg.dynamics.early_stop_patience)

    combo_iter        = _iter_infinite(combo_loader) if combo_loader is not None else None
    best_val_nll      = float("inf")
    best_epoch        = 0
    patience_counter  = 0
    epochs_run        = 0
    final_val_nll     = float("inf")

    log.info(
        "Training start | max_epochs=%d | patience=%d | "
        "lambda_combo=%.3f | lambda_lv_reg=%.4f",
        max_epochs, patience, lambda_combo, lambda_lv_reg,
    )

    for epoch in range(max_epochs):
        # ---- train ----
        model.train()
        epoch_nll   = 0.0
        epoch_combo = 0.0
        n_train     = 0

        for z_c, g, z_p in train_loader:
            z_c = z_c.to(device)
            g   = g.to(device)
            z_p = z_p.to(device)

            _, mu, lv        = model(z_c, g)
            target_delta     = z_p - z_c
            loss_nll         = heteroscedastic_nll(mu, lv, target_delta, log_var_reg=lambda_lv_reg)
            loss             = loss_nll
            batch_combo_loss = 0.0

            if combo_iter is not None and lambda_combo > 0.0:
                z_cc, g_a, g_b, z_ab = next(combo_iter)
                z_cc = z_cc.to(device)
                g_a  = g_a.to(device)
                g_b  = g_b.to(device)
                z_ab = z_ab.to(device)
                loss_c           = composition_loss(model, z_cc, g_a, g_b, z_ab)
                loss             = loss + lambda_combo * loss_c
                batch_combo_loss = loss_c.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_nll   += loss_nll.item()
            epoch_combo += batch_combo_loss
            n_train     += 1

        if scheduler is not None:
            scheduler.step()

        # ---- validate ----
        model.eval()
        val_nll_sum = 0.0
        n_val       = 0
        with torch.no_grad():
            for z_c, g, z_p in val_loader:
                z_c = z_c.to(device)
                g   = g.to(device)
                z_p = z_p.to(device)
                _, mu_v, lv_v = model(z_c, g)
                val_nll_sum  += heteroscedastic_nll(
                    mu_v, lv_v, z_p - z_c, log_var_reg=0.0
                ).item()
                n_val += 1

        avg_nll   = epoch_nll   / max(n_train, 1)
        avg_combo = epoch_combo / max(n_train, 1)
        avg_val   = val_nll_sum / max(n_val,   1)
        avg_total = avg_nll + lambda_combo * avg_combo
        epochs_run    = epoch + 1
        final_val_nll = avg_val

        log.info(
            "epoch %03d/%03d | loss=%.4f  nll=%.4f  combo=%.4f | "
            "val_nll=%.4f  best=%.4f | lr=%.2e",
            epochs_run, max_epochs,
            avg_total, avg_nll, avg_combo,
            avg_val, best_val_nll,
            optimizer.param_groups[0]["lr"],
        )

        if avg_val < best_val_nll:
            best_val_nll     = avg_val
            best_epoch       = epochs_run
            patience_counter = 0
            _save_state_dict(model, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(
                    "Early stopping at epoch %d (no val improvement for %d epochs). "
                    "Best epoch=%d  val_nll=%.4f.",
                    epochs_run, patience, best_epoch, best_val_nll,
                )
                break

    # Safety fallback: if val_nll never improved (e.g. NaN on first epoch)
    if not model_path.exists():
        log.warning("No best checkpoint was written; saving current model state.")
        _save_state_dict(model, model_path)

    log.info(
        "Training complete | epochs_run=%d | best_epoch=%d | best_val_nll=%.4f",
        epochs_run, best_epoch, best_val_nll,
    )

    # ------------------------------------------------------------------
    # Step 9 — write config.json
    # ------------------------------------------------------------------
    meta_path  = Path(cfg.paths.pairs_metadata)
    trained_on = "unknown"
    if meta_path.exists():
        meta       = json.loads(meta_path.read_text())
        trained_on = "mock" if meta.get("pairing_method") == "mock" else "real"

    config_record: dict = {
        "n_latent"         : n_latent,
        "n_genes"          : n_genes,
        "d_emb"            : int(cfg.dynamics.d_emb),
        "n_hidden"         : int(cfg.dynamics.n_hidden),
        "n_layers"         : int(cfg.dynamics.n_layers),
        "dropout"          : float(cfg.dynamics.dropout),
        "activation"       : str(cfg.dynamics.activation),
        "log_var_min"      : float(cfg.dynamics.log_var_min),
        "log_var_max"      : float(cfg.dynamics.log_var_max),
        "use_layernorm"    : bool(cfg.dynamics.use_layernorm),
        "trained_on"       : trained_on,
        "epochs_run"       : epochs_run,
        "best_epoch"       : best_epoch,
        "best_val_nll"     : best_val_nll,
        "final_val_nll"    : final_val_nll,
        "lambda_combo"     : lambda_combo,
        "lambda_log_var_reg": lambda_lv_reg,
        "seed"             : int(cfg.seed),
    }
    config_path = Path(cfg.paths.dynamics_config)
    config_path.write_text(json.dumps(config_record, indent=2))
    log.info("config.json written → %s", config_path)

    # ------------------------------------------------------------------
    # Phase 2 hook — validation gate (not yet implemented)
    # ------------------------------------------------------------------
    # Uncomment and implement when Phase 2 lands (see PHASES.md Phase 2 / AGENTS.md §2):
    #
    # from src.analysis.metrics import dynamics_validation_gate
    # ood_path = Path(cfg.paths.pairs_ood)
    # gate = dynamics_validation_gate(
    #     z_ctrl          = <val_z_ctrl array>,
    #     gene_idx        = <val_gene_idx array>,
    #     z_pert_true     = <val_z_pert array>,
    #     z_pert_pred_mlp = <model predictions array>,
    #     log_var_pred    = <model log_var array>,
    #     cfg_gate        = cfg.dynamics.gate,
    # )
    # gate_path = Path(cfg.paths.dynamics_gate)
    # gate_path.write_text(json.dumps(gate, indent=2))
    # log.info("gate.json written (passed=%s) → %s", gate["passed"], gate_path)
    # if not gate["passed"]:
    #     log.error("Dynamics validation gate FAILED. RL training is blocked until it passes.")
    #     return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
