"""Shared utilities for every other module.

Sacred rules (CLAUDE.md §3):
  - Only :mod:`src.utils.device` calls :func:`torch.device`.
  - Only :mod:`src.utils.seeding` calls :func:`random.seed`,
    :func:`numpy.random.seed`, :func:`torch.manual_seed`.
  - Checkpoint save/load goes through :mod:`src.utils.checkpointing` so the
    scVI official API is respected.
  - Logger setup goes through :mod:`src.utils.logging`.
"""

from src.utils.device import get_device  # noqa: F401
from src.utils.seeding import set_seed  # noqa: F401
