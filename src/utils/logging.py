"""Structured logging setup — TensorBoard + rich console.

Owner: shared.

Both agents log to the same TensorBoard logdir and use the same rich console
configuration. Centralising avoids drift between training scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO", rich_traceback: bool = True) -> logging.Logger:
    """Configure the root logger with a rich handler.

    Parameters
    ----------
    level
        Standard logging level name (INFO, DEBUG, WARNING, ...).
    rich_traceback
        If True, install ``rich.traceback`` for prettier exception rendering.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    from rich.logging import RichHandler

    if rich_traceback:
        from rich.traceback import install
        install(show_locals=False, max_frames=8)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=rich_traceback, markup=True)],
    )
    return logging.getLogger()


def get_tensorboard_writer(logdir: str | Path, run_name: str | None = None) -> Any:
    """Return a configured ``torch.utils.tensorboard.SummaryWriter``.

    Parameters
    ----------
    logdir
        Root directory for TensorBoard event files.
    run_name
        Sub-directory under ``logdir`` for this run. If ``None``, writes
        directly into ``logdir``.

    Returns
    -------
    torch.utils.tensorboard.SummaryWriter
    """
    from torch.utils.tensorboard import SummaryWriter

    log_path = Path(logdir)
    if run_name is not None:
        log_path = log_path / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_path))


def log_hyperparameters(writer: Any, cfg: Any) -> None:
    """Dump a Hydra DictConfig to the TensorBoard HPARAMS panel.

    Parameters
    ----------
    writer
        SummaryWriter from :func:`get_tensorboard_writer`.
    cfg
        Hydra ``DictConfig`` for the current run.
    """
    from omegaconf import OmegaConf

    flat = _flatten_dict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    scalar_params = {
        k: v for k, v in flat.items()
        if isinstance(v, (int, float, str, bool)) and v is not None
    }
    writer.add_hparams(scalar_params, metric_dict={})


def _flatten_dict(d: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict to dot-separated keys."""
    result: dict[str, Any] = {}
    if not isinstance(d, dict):
        return {prefix: d} if prefix else {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            result.update(_flatten_dict(v, key))
        else:
            result[key] = v
    return result
