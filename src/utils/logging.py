"""Structured logging setup — TensorBoard + rich console.

Owner: shared.

Why centralized: both agents log to the same TensorBoard logdir and use the same rich console
configuration. Drift between training scripts would make experiments hard to compare.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO", rich_traceback: bool = True) -> logging.Logger:
    """Configure the root logger with a ``rich`` handler.

    Parameters
    ----------
    level
        Standard ``logging`` level name.
    rich_traceback
        If True, install ``rich.traceback`` for prettier exceptions.

    Returns
    -------
    logging.Logger
        Configured root logger.

    Raises
    ------
    NotImplementedError
        Shared: implement with ``rich.logging.RichHandler`` and optional traceback install.
    """
    raise NotImplementedError(
        "Shared: implement rich-handler-backed logging. Used by every entry point."
    )


def get_tensorboard_writer(logdir: str | Path, run_name: str | None = None) -> Any:
    """Return a configured ``torch.utils.tensorboard.SummaryWriter``.

    Parameters
    ----------
    logdir
        Root directory for TensorBoard event files. Typically ``${paths.artifacts}/tensorboard``.
    run_name
        Sub-directory under ``logdir`` for this run. If ``None``, uses the current Hydra job id.

    Returns
    -------
    torch.utils.tensorboard.SummaryWriter
        Ready-to-use writer.

    Raises
    ------
    NotImplementedError
        Shared: implement.
    """
    raise NotImplementedError(
        "Shared: implement SummaryWriter creation. Both agents log scalars + histograms here."
    )


def log_hyperparameters(writer: Any, cfg: Any) -> None:
    """Dump a Hydra config to the TensorBoard ``HPARAMS`` panel.

    Parameters
    ----------
    writer
        SummaryWriter instance from :func:`get_tensorboard_writer`.
    cfg
        Hydra ``DictConfig`` for the current run.

    Raises
    ------
    NotImplementedError
        Shared: implement using ``OmegaConf.to_container`` + ``writer.add_hparams``.
    """
    raise NotImplementedError(
        "Shared: serialize cfg via OmegaConf.to_container and log via add_hparams."
    )
